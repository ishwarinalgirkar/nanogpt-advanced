"""RMSNorm implementation (skeleton)."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (skeleton)."""

    def __init__(self, dim, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm / (x.size(-1) ** 0.5)
        return x / (rms + self.eps) * self.scale
