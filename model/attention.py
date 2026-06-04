"""Causal multi-head attention with RoPE (skeleton).

Provides a minimal causal attention module that can be wired into the
transformer block. This is a structural placeholder and not optimized.
"""

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.out = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # Simple scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(self.dropout(out))
