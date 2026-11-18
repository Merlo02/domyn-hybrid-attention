"""
Quality of the trained checkpoints: loss on FineWeb-Edu validation, and loss at each position of long 10-K
filings, at the training length (1024) and beyond it.

    python evaluate.py --device=mps
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from model import GPT, GPTConfig

device = 'mps'
variants = [v + s for s in ['', '_s2', '_s3'] for v in ['mha', 'mla', 'mla_swa', 'hybrid']]
lengths = [1024, 4096, 8192]
exec(open('configurator.py').read())

def load(name):
    ckpt = torch.load(f'out/{name}/ckpt.pt', map_location=device)
    model = GPT(GPTConfig(**ckpt['model_args']))
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()})
    return model.to(device).eval()

@torch.no_grad()
def loss_per_position(model, rows):  # rows (N, L + 1) of tokens -> mean loss at each of the L positions
    total, batch = 0, max(1, 8192 // rows.shape[1])
    for i in range(0, len(rows), batch):
        x = torch.from_numpy(rows[i:i + batch].astype(np.int64)).to(device)
        y = x[:, 1:].contiguous()
        logits = model(x[:, :-1], y)[0]
        total = total + F.cross_entropy(logits.transpose(1, 2).float(), y, reduction='none').sum(0)
    return (total / len(rows)).tolist()

val = np.fromfile('data/fineweb_edu/val.bin', dtype=np.uint16)
val = val[:len(val) // 1025 * 1025].reshape(-1, 1025)
filings = np.fromfile('data/edgar/filings.bin', dtype=np.uint16).reshape(-1, 8193)
results = {}
for name in variants:
    model = load(name)
    results[name] = dict(val_loss=float(np.mean(loss_per_position(model, val))),
                         filings={L: loss_per_position(model, filings[:, :L + 1]) for L in lengths})
    print(name, 'val %.4f' % results[name]['val_loss'],
          ' '.join(f"10-K@{L} {np.mean(v):.4f}" for L, v in results[name]['filings'].items()))
json.dump(results, open('results/evaluate.json', 'w'))
