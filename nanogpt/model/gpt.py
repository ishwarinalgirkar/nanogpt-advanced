"""Full GPT-like model constructed from blocks (skeleton)."""

import torch
import torch.nn as nn
from .block import Block


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([Block(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb[:, : idx.size(1), :]
        x = self.drop(x)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
