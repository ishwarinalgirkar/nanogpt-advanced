"""Byte-Pair Encoding (BPE) tokenizer implemented from scratch (skeleton).

This module provides a minimal BPETokenizer interface for training and
encoding/decoding. It's a placeholder to be expanded with a full implemen-
tation as needed.
"""

from typing import List, Dict


class BPETokenizer:
    """Minimal BPE tokenizer skeleton."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}

    def train(self, texts: List[str], vocab_size: int = 30000):
        """Train the BPE merges from a list of texts. (Not implemented)

        Args:
            texts: list of training strings
            vocab_size: target vocabulary size
        """
        raise NotImplementedError()

    def encode(self, text: str) -> List[int]:
        """Encode a string to token ids."""
        raise NotImplementedError()

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to a string."""
        raise NotImplementedError()
