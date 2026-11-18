"""
KV-cache memory and decoding speed of each variant at a given context length and batch size.
Random weights: neither number depends on training.

    python bench_decode.py --device=cuda --contexts=[1024,8192,32768] --batches=[1,16,64]
"""
import json
import time
import torch
from model import GPT, GPTConfig

VARIANTS = {
    'mha': dict(attn='mha'),
    'mha_swa': dict(attn='mha', window=128),
    'mla': dict(attn='mla'),
    'mla_swa': dict(attn='mla', window=128),
    'hybrid': dict(attn='mla', window=128, global_every=3),
}
device = 'mps'
contexts = [1024, 4096, 16384]
batches = [1, 16]
steps = 20
exec(open('configurator.py').read())

def sync():
    getattr(torch, device.split(':')[0]).synchronize()

@torch.no_grad()
def decode(model, B, L):
    cache = model.new_cache(L + steps)
    for c in cache:
        c.pos = L - 1  # as if L - 1 tokens were already cached: same memory and same work per step
    tok = torch.zeros(B, 1, dtype=torch.long, device=device)
    for _ in range(3):
        model(tok, cache=cache)  # warm-up: allocates the cache, first-call kernel setup
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        model(tok, cache=cache)
    sync()
    dt = (time.perf_counter() - t0) / steps
    return dict(cache_MiB=sum(b.nbytes for c in cache for b in c.bufs) / 2**20, ms_per_step=dt * 1e3, tokens_per_s=B / dt)

results = []
for name, cfg in VARIANTS.items():
    model = GPT(GPTConfig(**cfg)).to(device, torch.bfloat16).eval()
    decode(model, 1, contexts[0])  # thrown away: the first call of each model pays one-time setup
    for L in contexts:
        for B in batches:
            try:
                r = dict(variant=name, context=L, batch=B, **decode(model, B, L))
            except torch.OutOfMemoryError:
                r = dict(variant=name, context=L, batch=B, oom=True)
            results.append(r)
            print(r)
json.dump(results, open(f"results/bench_decode_{device.split(':')[0]}.json", 'w'), indent=1)
