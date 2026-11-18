"""Figures and tables for the README, from results/bench_decode_cuda.json and evaluate.json."""
import json
import re
import numpy as np
import matplotlib.pyplot as plt

E, LAYERS, WINDOW, BYTES = 384, 6, 128, 2  # the trained models, bf16
PER_TOKEN = {'mha': 2 * E, 'mla': 256 + 32}  # numbers cached per token and layer: 2 n_h d_h, or d_c + d_h^R
VARIANTS = {'mha': ('mha', 0, 0), 'mha_swa': ('mha', WINDOW, 0), 'mla': ('mla', 0, 0),
            'mla_swa': ('mla', WINDOW, 0), 'hybrid': ('mla', WINDOW, 3)}
LABEL = {'mha': 'MHA', 'mha_swa': 'MHA + window', 'mla': 'MLA', 'mla_swa': 'MLA + window', 'hybrid': 'hybrid'}
STYLE = {'mha': '-', 'mha_swa': '--', 'mla': '-', 'mla_swa': ':', 'hybrid': '-.'}

def cache_mib(name, L):  # one sequence, all layers
    attn, w, every = VARIANTS[name]
    tokens = sum(L if not w or (every and (i + 1) % every == 0) else min(L, w) for i in range(LAYERS))
    return PER_TOKEN[attn] * BYTES * tokens / 2**20

bench = [r for r in json.load(open('results/bench_decode_cuda.json')) if 'oom' not in r]
plt.figure()
L = 2 ** np.arange(8, 18)
for name in VARIANTS:  # lines: formula; dots: measured on the A100
    line, = plt.plot(L, [cache_mib(name, l) for l in L], label=LABEL[name])
    pts = [r for r in bench if r['variant'] == name and r['batch'] == 1]
    plt.plot([r['context'] for r in pts], [r['cache_MiB'] for r in pts], 'o', color=line.get_color())
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('context (tokens)')
plt.ylabel('KV cache per sequence (MiB)')
plt.legend()
plt.tight_layout()
plt.savefig('figures/cache.png', dpi=150)

contexts = sorted({r['context'] for r in bench})
fig, axes = plt.subplots(1, len(contexts), figsize=(4 * len(contexts), 3.6), sharey=True)
for ax, ctx in zip(axes, contexts):
    for name in VARIANTS:
        pts = [r for r in bench if r['variant'] == name and r['context'] == ctx]
        ax.plot([r['batch'] for r in pts], [r['tokens_per_s'] for r in pts], 'o' + STYLE[name], label=LABEL[name])
    ax.set(title=f'context {ctx:,} tokens', xscale='log', yscale='log', xlabel='sequences in the batch')
axes[0].set_ylabel('generated tokens / s')
axes[0].legend()
fig.tight_layout()
fig.savefig('figures/decode.png', dpi=150)

quality = json.load(open('results/evaluate.json'))
seeds = {v: [q for name, q in quality.items() if re.sub(r'_s\d+$', '', name) == v] for v in ['mha', 'mla', 'mla_swa', 'hybrid']}
plt.figure()
for v, runs in seeds.items():  # mean over seeds, then a moving average over 64 positions
    curve = np.convolve(np.mean([q['filings']['8192'] for q in runs], axis=0), np.ones(64) / 64, 'valid')
    plt.plot(np.arange(len(curve)) + 32, curve, STYLE[v], label=LABEL[v])
plt.axvline(1024, ls='--', c='gray', label='training length')
plt.xlabel('position in the 10-K filing (tokens)')
plt.ylabel('loss')
plt.legend()
plt.tight_layout()
plt.savefig('figures/loss_by_position.png', dpi=150)

print('| | FineWeb-Edu validation | 10-K, first 1,024 tokens | 10-K, 4,096 | 10-K, 8,192 |\n|---|---|---|---|---|')
for v, runs in seeds.items():
    val = [q['val_loss'] for q in runs]
    tenk = [np.mean([np.mean(q['filings'][L]) for q in runs]) for L in ['1024', '4096', '8192']]
    print(f"| {LABEL[v]} | {np.mean(val):.3f} ± {np.std(val, ddof=1):.3f} | " + ' | '.join(f'{t:.2f}' for t in tenk) + ' |')
