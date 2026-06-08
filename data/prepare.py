import os
import numpy as np
import tiktoken
from datasets import load_dataset
from config import Config

def prepare_data():
    config = Config()
    data_path = config.data_path

    if os.path.exists(data_path):
        tokens = np.memmap(data_path, dtype=np.uint16, mode='r')
        print(f"train.bin already exists — {len(tokens):,} tokens")
        print("Skipping download and tokenization")
    else:
        print("Downloading TinyStories (~2GB, takes ~3 mins)...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        print(f"Stories: {len(dataset):,}")

        # GPT-2 tiktoken — 50257 tokens, byte-level BPE
        enc = tiktoken.get_encoding("gpt2")
        eot = enc.eot_token   # end-of-text token = 50256, separates stories

        print("Tokenizing...")
        all_tokens = []
        for i, story in enumerate(dataset):
            # encode story + append end-of-text token as story separator
            toks = enc.encode_ordinary(story['text']) + [eot]
            all_tokens.extend(toks)
            if i % 50000 == 0:
                print(f"  {i:,}/{len(dataset):,} stories — {len(all_tokens):,} tokens")

        # save as uint16 (max value 65535, GPT-2 vocab fits in uint16)
        # uint16 vs int32 = 2x smaller file = faster I/O during training
        arr = np.array(all_tokens, dtype=np.uint16)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        arr.tofile(data_path)
        print(f"\nSaved {len(arr):,} tokens to {data_path}")
        print(f"File size: {os.path.getsize(data_path)/1e6:.0f}MB")
        del dataset, all_tokens, arr

if __name__ == "__main__":
    prepare_data()
