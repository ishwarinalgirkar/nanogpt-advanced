"""Small helper script to train the BPE tokenizer from text files."""

from pathlib import Path
from .bpe import BPETokenizer


def train_from_files(files, vocab_size=30000):
    tok = BPETokenizer()
    texts = []
    for p in files:
        texts.append(Path(p).read_text(encoding='utf-8'))
    tok.train(texts, vocab_size=vocab_size)
    return tok


if __name__ == '__main__':
    print('This is a placeholder for tokenizer training.')
