"""SwiGLU feed-forward network (skeleton)."""

import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, n_in, n_hidden):
        super().__init__()
        self.w1 = nn.Linear(n_in, n_hidden)
        self.w2 = nn.Linear(n_in, n_hidden)
        self.proj = nn.Linear(n_hidden, n_in)

    def forward(self, x):
        return self.proj(self.w1(x) * F.silu(self.w2(x)))
