import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .block import Block
from .normalization import RMSNorm
from .rope import precompute_freqs

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq, dropout=0.0):
        super().__init__()
        self.max_seq = max_seq

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            Block(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm    = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE frequencies
        cos, sin = precompute_freqs(d_model // n_heads, max_seq)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.apply(self._init_weights)

        # Special scaled init for residual projections
        for name, p in self.named_parameters():
            if name.endswith('out_proj.weight') or name.endswith('down.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_seq, f"Input length {T} > max_seq {self.max_seq}"

        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)
        return idx
