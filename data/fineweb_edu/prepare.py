"""
Tokenize a slice of FineWeb-Edu (sample-10BT) with the GPT-2 tokenizer.
Writes train.bin and val.bin as flat uint16 token streams, the format
nanoGPT's train.py reads. Documents are separated by the end-of-text token.
Validation documents come first in the stream and are never used for training.
"""
import os
import numpy as np
import tiktoken
from datasets import load_dataset

TRAIN_TOKENS = 1_000_000_000
VAL_TOKENS = 2_000_000

enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

out_dir = os.path.dirname(__file__)
val = np.empty(VAL_TOKENS, dtype=np.uint16)
train = np.memmap(os.path.join(out_dir, "train.bin"), dtype=np.uint16, mode="w+", shape=(TRAIN_TOKENS,))
n_val = n_train = 0
batch = []

def write(tokens):
    global n_val, n_train
    if n_val < VAL_TOKENS:  # fill validation first, cut at a document boundary
        k = min(len(tokens), VAL_TOKENS - n_val)
        val[n_val:n_val + k] = tokens[:k]
        n_val += k
        return
    k = min(len(tokens), TRAIN_TOKENS - n_train)
    train[n_train:n_train + k] = tokens[:k]
    n_train += k

for doc in ds:
    batch.append(doc["text"])
    if len(batch) < 1024:
        continue
    for ids in enc.encode_ordinary_batch(batch):
        write(np.array(ids + [eot], dtype=np.uint16))
    batch = []
    if n_train >= TRAIN_TOKENS:
        break
    if n_train and n_train % 20_000_000 < 600_000:
        print(f"{n_train / 1e6:.0f}M train tokens", flush=True)

train.flush()
val.tofile(os.path.join(out_dir, "val.bin"))
print(f"done: {n_train:,} train tokens, {n_val:,} val tokens")
