"""Packing dataloader and dataset utilities (skeleton)."""

from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Dataset that yields packed sequences from tokenized data."""

    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx):
        x = self.tokens[idx: idx + self.block_size]
        y = self.tokens[idx + 1: idx + 1 + self.block_size]
        return x, y
