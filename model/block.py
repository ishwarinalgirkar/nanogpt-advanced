import torch.nn as nn
from .attention import CausalSelfAttention
from .normalization import RMSNorm
from .swiglu import SwiGLU

class Block(nn.Module):
    """
    Transformer block: RMSNorm BEFORE attention and FFN (Pre-norm).
    """
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU(d_model)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x
