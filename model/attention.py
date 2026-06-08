import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rope

class CausalSelfAttention(nn.Module):
    """
    Multi-head attention with RoPE on Q and K (not V).
    Uses FlashAttention (scaled_dot_product_attention) for efficiency.
    """
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        # Q, K, V projections — no bias (RMSNorm before us handles mean)
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout  = dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape

        # project and reshape: [B, T, C] → [B, n_heads, T, head_dim]
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        # apply RoPE to Q and K only
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # FlashAttention handles causal mask, dropout, scaling
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        # merge heads: [B, n_heads, T, head_dim] → [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)
