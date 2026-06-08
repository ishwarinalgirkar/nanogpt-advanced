import torch

def precompute_freqs(head_dim, max_seq, base=10000.0):
    """
    Precompute cos/sin frequencies once at init — not every forward pass.
    Returns cos, sin each of shape [max_seq, head_dim/2]
    """
    # θᵢ = 1 / (base ^ (2i / head_dim)) — geometric sequence of frequencies
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t     = torch.arange(max_seq)                    # position indices [0..max_seq-1]
    freqs = torch.outer(t, theta)                    # [max_seq, head_dim/2]
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    """
    Apply rotary embeddings to x (Q or K tensor).
    x: [B, n_heads, T, head_dim]
    """
    T   = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)          # [1, 1, T, head_dim/2]
    sin = sin[:T].unsqueeze(0).unsqueeze(0)

    x1  = x[..., ::2]                               # even dims: [B, H, T, head_dim/2]
    x2  = x[..., 1::2]                              # odd  dims: [B, H, T, head_dim/2]

    # rotate and interleave back to [B, H, T, head_dim]
    x_rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * cos.repeat_interleave(2, -1) + x_rot * sin.repeat_interleave(2, -1)
