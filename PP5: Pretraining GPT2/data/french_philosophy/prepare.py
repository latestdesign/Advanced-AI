"""
Prepare the French Philosophy 10K dataset for GPT-2 pretraining.
Loads from HuggingFace, tokenizes with tiktoken GPT-2 BPE, and saves as binary files.
"""
import os
import numpy as np
import tiktoken
from datasets import load_dataset

# load dataset from HuggingFace
ds = load_dataset("Dorian2B/french-philosophy-10K")

# concatenate all texts with a separator
data = "\n\n".join(ds["train"]["text"])

# split into train and val (90/10)
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
