"""
Long documents, for evaluation only: 10-K annual reports from EDGAR-CORPUS (eloukas/edgar-corpus, Apache-2.0),
year 2020, test split, tokenized with GPT-2. Writes filings.bin: the first ROW tokens of the first N filings
that are at least that long, one row each.
"""
import json
import os
import numpy as np
import tiktoken
from huggingface_hub import hf_hub_download

ROW, N = 8193, 128
enc = tiktoken.get_encoding("gpt2")
rows = []
for line in open(hf_hub_download("eloukas/edgar-corpus", "2020/test.jsonl", repo_type="dataset")):
    filing = json.loads(line)
    ids = enc.encode_ordinary("\n\n".join(v for k, v in filing.items() if k.startswith("section_") and v))
    if len(ids) >= ROW:
        rows.append(ids[:ROW])
    if len(rows) == N:
        break
np.array(rows, dtype=np.uint16).tofile(os.path.join(os.path.dirname(__file__), "filings.bin"))
print(f"{len(rows)} filings of {ROW} tokens")
