"""
A small GPT, built on nanoGPT (github.com/karpathy/nanoGPT), with four kinds of attention: standard multi-head
attention (MHA), Multi-head Latent Attention (MLA, DeepSeek-V2 Section 2.1), either one with a sliding window
(Mistral 7B Section 2), and a hybrid of windowed and full layers. RoPE for positions, and a KV cache for generation.
"""
import math
import functools
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def rope(x, pos):
    """Rotate the pairs (i, i + d/2) of x (B, H, T, d) by pos * 10000^(-2i/d)."""
    d = x.size(-1)
    angle = pos[:, None].float() * 10000.0 ** (-torch.arange(0, d, 2, device=x.device).float() / d)
    cos, sin = angle.cos(), angle.sin()
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)

@torch.compiler.disable  # built once per (length, window), outside the compiled graph, then reused
@functools.lru_cache
def block_mask(T, window, device):
    return create_block_mask(lambda b, h, q, kv: (q >= kv) & (q - kv < (window or T)), None, None, T, T, device=device)

def attend(q, k, v, scale, window):
    """Causal attention; with a window, each query sees only the last `window` keys."""
    if q.is_cuda:  # FlexAttention computes only the blocks the mask allows
        return flex_attention(q, k, v, block_mask=block_mask(q.size(2), window, q.device), scale=scale)
    i = torch.arange(q.size(2), device=q.device)
    mask = (i[:, None] >= i) & (i[:, None] - i < (window or q.size(2)))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)

class Cache:
    """KV cache of one layer, for generation. In windowed layers it is a ring buffer: token t goes to slot t % size."""

    def __init__(self, size):
        self.size, self.pos, self.bufs = size, 0, None

    def update(self, *new):  # each tensor (B, T, ...); returns every cached token
        T = new[0].size(1)
        n = min(T, self.size)
        if self.bufs is None:
            self.bufs = [x.new_zeros(x.size(0), self.size, *x.shape[2:]) for x in new]
        slots = torch.arange(self.pos + T - n, self.pos + T, device=new[0].device) % self.size
        for buf, x in zip(self.bufs, new):
            buf[:, slots] = x[:, T - n:]
        self.pos += T
        return [buf[:, :min(self.pos, self.size)] for buf in self.bufs]

class MHA(nn.Module):
    """Standard multi-head attention: every head caches its own key and value."""

    def __init__(self, config, window):
        super().__init__()
        self.H, self.window = config.n_head, window
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, pos, cache=None):
        B, T, C = x.size()
        q, k, v = (t.view(B, T, self.H, C // self.H).transpose(1, 2) for t in self.c_attn(x).split(C, dim=2))
        q, k = rope(q, pos), rope(k, pos)
        if cache is not None:
            ks, vs = cache.update(k.transpose(1, 2), v.transpose(1, 2))
        if cache is None or T > 1:  # training, or a prompt read in one pass into an empty cache
            y = attend(q, k, v, None, self.window)
        else:  # one new token against the cache
            y = F.scaled_dot_product_attention(q, ks.transpose(1, 2), vs.transpose(1, 2))
        return self.c_proj(y.transpose(1, 2).reshape(B, T, C))

class MLA(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2, Section 2.1), without query compression. Caches c and k^R only."""

    def __init__(self, config, window):
        super().__init__()
        E, H = config.n_embd, config.n_head
        self.H, self.dh, self.window = H, E // H, window
        self.dc = config.kv_rank or 4 * self.dh  # DeepSeek-V2 ratios (Table 1)
        self.dr = config.rope_dim or self.dh // 2
        self.w_q = nn.Linear(E, H * self.dh, bias=False)     # q^C
        self.w_qr = nn.Linear(E, H * self.dr, bias=False)    # q^R, eq. 14
        self.w_dkv = nn.Linear(E, self.dc, bias=False)       # c^KV, eq. 9
        self.kv_norm = nn.RMSNorm(self.dc)                   # Section 3.1.2
        self.w_kr = nn.Linear(E, self.dr, bias=False)        # k^R, shared by all heads, eq. 15
        self.w_uk = nn.Linear(self.dc, H * self.dh, bias=False)  # eq. 10
        self.w_uv = nn.Linear(self.dc, H * self.dh, bias=False)  # eq. 11
        self.c_proj = nn.Linear(H * self.dh, E, bias=False)  # W^O
        self.scale = (self.dh + self.dr) ** -0.5             # eq. 18

    def forward(self, x, pos, cache=None):
        B, T, _ = x.size()
        H, dh = self.H, self.dh
        c = self.kv_norm(self.w_dkv(x))                                       # (B, T, dc)
        kr = rope(self.w_kr(x)[:, None], pos)                                # (B, 1, T, dr)
        qc = self.w_q(x).view(B, T, H, dh).transpose(1, 2)                   # (B, H, T, dh)
        qr = rope(self.w_qr(x).view(B, T, H, self.dr).transpose(1, 2), pos)  # (B, H, T, dr)
        if cache is not None:
            cs, krs = cache.update(c, kr[:, 0])                              # (B, S, dc), (B, S, dr)
        if cache is None or T > 1:  # training or prompt: build K and V once
            k = torch.cat([self.w_uk(c).view(B, T, H, dh).transpose(1, 2), kr.expand(-1, H, -1, -1)], dim=-1)
            v = self.w_uv(c).view(B, T, H, dh).transpose(1, 2)
            y = attend(torch.cat([qc, qr], dim=-1), k, v, self.scale, self.window)
        else:  # generation, absorbed: q.(W^UK c) = (W^UK^T q).c, so K and V are never built
            w_uk = self.w_uk.weight.view(H, dh, self.dc)
            w_uv = self.w_uv.weight.view(H, dh, self.dc)
            q_lat = torch.einsum('bhtd,hdc->bhtc', qc, w_uk)
            # einsum, not @: all heads read the one shared cache, which is never copied per head
            att = torch.einsum('bhtc,bsc->bhts', q_lat, cs) + torch.einsum('bhtr,bsr->bhts', qr, krs)
            att = F.softmax(att.float() * self.scale, dim=-1).type_as(x)
            y = torch.einsum('bhtc,hdc->bhtd', torch.einsum('bhts,bsc->bhtc', att, cs), w_uv)  # average the latents, then W^UV
        return self.c_proj(y.transpose(1, 2).reshape(B, T, H * dh))

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

class Block(nn.Module):

    def __init__(self, config, layer):
        super().__init__()
        is_global = config.global_every and (layer + 1) % config.global_every == 0
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=False)
        self.attn = (MLA if config.attn == 'mla' else MHA)(config, 0 if is_global else config.window)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x, pos, cache=None):
        x = x + self.attn(self.ln_1(x), pos, cache)
        return x + self.mlp(self.ln_2(x))

@dataclass
class GPTConfig:
    vocab_size: int = 50304  # tokens the model knows: GPT-2's 50257, padded to a multiple of 64
    n_layer: int = 6         # l, number of blocks
    n_head: int = 6          # n_h, attention heads per layer
    n_embd: int = 384        # d, the vector of each token; d_h = d / n_h = 64 per head
    attn: str = 'mha'        # 'mha' or 'mla'
    window: int = 0          # W, a windowed layer sees the last W tokens, itself included; 0 = full attention
    global_every: int = 0    # with a window, every n-th layer stays full (3: local, local, global)
    kv_rank: int = 0         # d_c, KV compression dimension (DeepSeek-V2 eq. 9); 0 = 4 d_h
    rope_dim: int = 0        # d_h^R, per-head dimension of the decoupled queries and key (eq. 14-15); 0 = d_h / 2

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(  # GPT-2's names, kept from nanoGPT
            wte=nn.Embedding(config.vocab_size, config.n_embd),                  # token id -> its vector of size d
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),  # the l blocks, applied in order
            ln_f=nn.LayerNorm(config.n_embd, bias=False),                        # final norm, before the output layer
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # vector -> a score per vocabulary token
        self.transformer.wte.weight = self.lm_head.weight  # tied: one matrix reads tokens in and scores them out
        self.apply(self._init_weights)  # Linear and Embedding weights ~ N(0, 0.02), as in GPT-2
        for name, p in self.named_parameters():  # GPT-2: smaller init for the projections into the residual stream
            if name.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print(f"number of parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def new_cache(self, max_len):  # max_len = prompt + generated tokens: beyond it, full layers would wrap like a window
        return [Cache(block.attn.window or max_len) for block in self.transformer.h]

    def forward(self, idx, targets=None, cache=None):  # idx: token ids (B, T); targets: the next token of each
        start = cache[0].pos if cache else 0  # tokens already in the cache = position of the first new one
        assert not cache or start == 0 or idx.size(1) == 1, 'the prompt goes in one call, into an empty cache'
        pos = torch.arange(start, start + idx.size(1), device=idx.device)  # position of each token, for RoPE
        x = self.transformer.wte(idx)  # (B, T, d)
        for i, block in enumerate(self.transformer.h):
            x = block(x, pos, cache[i] if cache else None)  # block i reads and writes its own cache
        x = self.transformer.ln_f(x)
        if targets is None:  # generation: only the last position is needed
            return self.lm_head(x[:, [-1]]), None
        logits = self.lm_head(x)  # (B, T, vocab_size): a score for every possible next token
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
