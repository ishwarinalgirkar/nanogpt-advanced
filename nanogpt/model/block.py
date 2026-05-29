"""Transformer block combining attention, normalization, and MLP."""

import torch.nn as nn
from .attention import CausalSelfAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU


class Block(nn.Module):
    def __init__(self, n_embd, n_head, mlp_hidden_ratio=4.0):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = RMSNorm(n_embd)
        hidden = int(n_embd * mlp_hidden_ratio)
        self.mlp = SwiGLU(n_embd, hidden)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
