import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Simpler than LayerNorm: normalise by RMS, no mean subtraction, no bias.
    Used in: LLaMA, Mistral, Gemma — ~10% faster than LayerNorm.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))   # learnable scale γ

    def forward(self, x):
        # x: [B, T, dim]
        # rms = sqrt(mean(x²) + eps) — shape [B, T, 1] for broadcasting
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight
