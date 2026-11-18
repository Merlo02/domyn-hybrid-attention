import pytest
import torch
from torch.nn.attention.flex_attention import flex_attention
from model import GPT, GPTConfig, attend, block_mask

VARIANTS = {
    'mha': dict(attn='mha'),
    'mha_swa': dict(attn='mha', window=4),
    'mla': dict(attn='mla'),
    'mla_swa': dict(attn='mla', window=4),
    'hybrid': dict(attn='mla', window=4, global_every=3),
}

def tiny(**kw):
    torch.manual_seed(0)
    cfg = dict(vocab_size=64, n_layer=3, n_head=2, n_embd=32) | kw
    return GPT(GPTConfig(**cfg)).eval()

def all_logits(model, idx):
    return model(idx, targets=idx)[0]

@pytest.mark.parametrize('name', VARIANTS)
def test_cache_matches_full_forward(name):
    """Prefill, then one token at a time (ring buffer, absorbed MLA): same logits as one full forward."""
    model = tiny(**VARIANTS[name])
    idx = torch.randint(0, 64, (2, 20))
    cache = model.new_cache(20)
    got = [model(idx[:, :5], cache=cache)[0]] + [model(idx[:, [t]], cache=cache)[0] for t in range(5, 20)]
    assert torch.allclose(torch.cat(got, dim=1), all_logits(model, idx)[:, 4:], atol=1e-4)

@pytest.mark.parametrize('attn', ['mha', 'mla'])
def test_wide_window_is_full_attention(attn):
    idx = torch.randint(0, 64, (2, 16))
    assert torch.allclose(all_logits(tiny(attn=attn, window=16), idx), all_logits(tiny(attn=attn), idx), atol=1e-5)

@pytest.mark.parametrize('attn', ['mha', 'mla'])
def test_window_limits_what_a_token_sees(attn):
    """One layer, window 4: the output at position 10 ignores token 6 and depends on token 7."""
    model = tiny(attn=attn, window=4, n_layer=1)
    idx = torch.randint(0, 64, (1, 12))
    far, near = idx.clone(), idx.clone()
    far[0, 6] = (far[0, 6] + 1) % 64
    near[0, 7] = (near[0, 7] + 1) % 64
    base = all_logits(model, idx)[0, 10]
    assert torch.allclose(all_logits(model, far)[0, 10], base)
    assert not torch.allclose(all_logits(model, near)[0, 10], base)

def test_global_layer_sees_beyond_the_window():
    """Same setup, but the layer is global (global_every=1): token 6 now matters."""
    model = tiny(attn='mla', window=4, global_every=1, n_layer=1)
    idx = torch.randint(0, 64, (1, 12))
    far = idx.clone()
    far[0, 6] = (far[0, 6] + 1) % 64
    assert not torch.allclose(all_logits(model, far)[0, 10], all_logits(model, idx)[0, 10])

@pytest.mark.parametrize('window', [0, 8])
def test_flex_matches_masked_attention(window):
    """The GPU path (FlexAttention and its block mask) equals the masked path, at MLA head sizes (24 for q, k; 16 for v)."""
    q, k, v = torch.randn(2, 2, 128, 24), torch.randn(2, 2, 128, 24), torch.randn(2, 2, 128, 16)
    flex = flex_attention(q, k, v, block_mask=block_mask(128, window, torch.device('cpu')), scale=24 ** -0.5)
    assert torch.allclose(flex, attend(q, k, v, 24 ** -0.5, window), atol=1e-5)

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='no MPS')
def test_mps_matches_cpu_when_v_is_narrower():
    q, k, v = torch.randn(2, 2, 16, 24), torch.randn(2, 2, 16, 24), torch.randn(2, 2, 16, 16)
    mps = attend(q.to('mps'), k.to('mps'), v.to('mps'), 24 ** -0.5, 4).cpu()
    assert torch.allclose(mps, attend(q, k, v, 24 ** -0.5, 4), atol=1e-4)
