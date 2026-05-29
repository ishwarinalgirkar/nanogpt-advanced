"""Rotary positional embeddings (RoPE) utilities (skeleton)."""

import torch


def make_rotary_embeddings(dim, seq_len, device=None):
    """Return rotary sin/cos cached tensors (placeholder)."""
    # Placeholder implementation; real implementation should create
    # cos/sin buffers shaped for einsum with queries/keys.
    return None


def apply_rotary(q, k, cos, sin):
    """Apply rotary embeddings to query and key tensors."""
    raise NotImplementedError()
