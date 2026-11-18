"""
Train one model from scratch on FineWeb-Edu. The defaults are the experiments in the README; any of them can be
changed from the command line (see configurator.py), for example

    python train.py --attn=mla --window=128 --global_every=3 --seed=2 --out_dir=out/hybrid_s2
"""
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch
from model import GPT, GPTConfig

attn = 'mha'          # 'mha' or 'mla'
window = 0            # sliding window in tokens; 0 = full attention
global_every = 0      # with a window, every n-th layer stays full (3 = the hybrid)
seed = 1337
batch_size, block_size = 64, 1024
max_iters, warmup_iters = 7500, 200  # 7500 x 64 x 1024 = 491M tokens
learning_rate, min_lr = 1e-3, 1e-4
weight_decay, beta1, beta2, grad_clip = 0.1, 0.9, 0.95, 1.0
eval_interval, eval_iters, log_interval = 2500, 100, 50
dataset = 'data/fineweb_edu'
out_dir = 'out'
device = 'cuda'
compile = True
exec(open('configurator.py').read())

torch.manual_seed(seed)
batch_rng = torch.Generator().manual_seed(seed)  # own generator: every variant sees the same batches
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
ctx = torch.autocast('cuda', dtype=torch.bfloat16) if 'cuda' in device else nullcontext()
os.makedirs(out_dir, exist_ok=True)

def get_batch(split):
    data = np.memmap(os.path.join(dataset, f'{split}.bin'), dtype=np.uint16, mode='r')  # reopened every time: a kept memmap leaks memory
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=batch_rng)
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        with ctx:
            losses[split] = float(np.mean([model(*get_batch(split))[1].item() for _ in range(eval_iters)]))
    model.train()
    return losses

def get_lr(it):  # linear warm-up, then cosine decay to min_lr at max_iters
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    return min_lr + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (max_iters - warmup_iters))) * (learning_rate - min_lr)

config = GPTConfig(attn=attn, window=window, global_every=global_every)
raw_model = GPT(config).to(device)
optimizer = torch.optim.AdamW([  # weight decay on matrices only, not on norms
    {'params': [p for p in raw_model.parameters() if p.dim() >= 2], 'weight_decay': weight_decay},
    {'params': [p for p in raw_model.parameters() if p.dim() < 2], 'weight_decay': 0.0},
], lr=learning_rate, betas=(beta1, beta2), fused='cuda' in device)
model = torch.compile(raw_model) if compile else raw_model

x, y = get_batch('train')
t0 = time.time()
for it in range(max_iters + 1):
    for group in optimizer.param_groups:
        group['lr'] = get_lr(it)
    if it % eval_interval == 0 or it == max_iters:
        losses = estimate_loss()
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if it > 0:
            torch.save({'model': raw_model.state_dict(), 'model_args': asdict(config), 'iter_num': it, 'val_loss': losses['val']},
                       os.path.join(out_dir, 'ckpt.pt'))
    if it == max_iters:
        break
    with ctx:
        loss = model(x, y)[1]
    x, y = get_batch('train')  # read the next batch while the GPU works on this one
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    t1 = time.time()
    dt, t0 = t1 - t0, t1
    if it % log_interval == 0:
        print(f"iter {it}: loss {loss.item():.4f}, time {dt * 1000:.2f}ms")
