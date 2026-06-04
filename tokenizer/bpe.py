"""
Byte-Level BPE Tokenizer — Version 3 (GPT-4 / tiktoken style)
==============================================================
Paper: Sennrich et al. 2016 - https://arxiv.org/abs/1508.07956  (core BPE)
       GPT-2 paper 2019    - Section 2.2                         (byte-level extension)
       tiktoken            - https://github.com/openai/tiktoken  (regex pre-tokenisation)

Three additions over basic BPE:
  1. Base vocabulary = all 256 bytes  → no [UNK] ever
  2. Regex pre-tokenisation           → merges never cross word/punctuation boundaries
  3. Merge order preserved at encode  → deterministic, reproducible tokenisation

Build order:
  Phase 1 — base vocab
  Phase 2 — regex pre-tokenisation
  Phase 3 — get_stats  (count adjacent pairs)
  Phase 4 — merge      (replace a pair with a new token id)
  Phase 5 — train      (repeat phases 3+4 until vocab_size reached)
  Phase 6 — encode     (apply learned merges to new text)
  Phase 7 — decode     (token ids → bytes → string)
"""

import regex as re          # pip install regex  (supports \p{L}, \p{N} Unicode props)
from collections import defaultdict


# PHASE 1 — Base vocabulary: all 256 byte values


def get_base_vocab() -> dict[int, bytes]:
    """
    Build the starting vocabulary: 256 tokens, one per byte value.

    vocab[0]   = b'\\x00'
    vocab[65]  = b'A'
    vocab[104] = b'h'
    vocab[255] = b'\\xff'

    Every possible input text decomposes into these 256 base tokens,
    so the tokenizer is lossless on any input — no [UNK] ever.
    """
    # bytes([i]) creates a single-byte bytes object for value i
    # e.g. bytes([65]) == b'A',  bytes([104]) == b'h'
    return {i: bytes([i]) for i in range(256)}



# PHASE 2 — Regex pre-tokenisation


# GPT-4 / cl100k_base pattern (from tiktoken)
# This splits text into linguistically meaningful chunks BEFORE BPE runs.
# Key effect: merges are never allowed to cross chunk boundaries.
#
# Pattern breakdown:
#   (?i:[sdmt]|ll|ve|re)         — English contractions: 's, 'd, 'm, 't, 'll, 've, 're
#   [^\r\n\p{L}\p{N}]?\p{L}+    — words (optionally preceded by a non-letter/digit like space)
#   \p{N}{1,3}                   — numbers, up to 3 digits per chunk
#    ?[^\s\p{L}\p{N}]+[\r\n]*   — punctuation / symbols
#   \s*[\r\n]                    — newlines
#   \s+(?!\S)                    — trailing whitespace
#   \s+                          — remaining whitespace

GPT4_SPLIT_PATTERN = re.compile(
    r"""(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

# Simpler GPT-2 pattern — good for understanding, less robust than GPT-4
GPT2_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""
)


def pretokenise(text: str, pattern=GPT4_SPLIT_PATTERN) -> list[list[int]]:
    """
    Split text into chunks using regex, then encode each chunk as bytes.

    Returns a list of token sequences (each sequence = one chunk as byte ids).

    Example:
        pretokenise("Hello world!")
        → [[72,101,108,108,111], [32,119,111,114,108,100], [33]]
          (roughly — exact splits depend on pattern)

    Why this matters:
        Without pre-tokenisation, BPE might merge "dog" + "." into one token.
        The regex ensures punctuation, whitespace, and words are split first,
        so merges only happen within natural linguistic units.
    """
    chunks = pattern.findall(text)
    # encode each chunk to UTF-8 bytes, convert to list of ints
    return [list(chunk.encode("utf-8")) for chunk in chunks]



# PHASE 3 — Count adjacent pairs


def get_stats(ids: list[list[int]]) -> dict[tuple[int, int], int]:
    """
    Count every adjacent pair across all token sequences.

    Args:
        ids: list of token sequences (one per chunk from pretokenise)

    Returns:
        dict mapping (token_a, token_b) → count

    Example:
        get_stats([[104, 101, 108, 108, 111]])
        → {(104,101):1, (101,108):1, (108,108):1, (108,111):1}

    Why list[list[int]] and not flat list[int]?
        If we flattened, we'd count pairs that cross chunk boundaries.
        "hello" followed by " world" would create a fake pair (111, 32)
        — the 'o' of hello and the space before world. That merge would
        be wrong because those tokens belong to different linguistic units.
        Keeping sequences separate prevents any cross-boundary counting.
    """
    counts = defaultdict(int)
    for chunk_ids in ids:
        # zip(chunk, chunk[1:]) gives us all adjacent pairs in one chunk
        # e.g. [1,2,3] → [(1,2), (2,3)]
        for pair in zip(chunk_ids, chunk_ids[1:]):
            counts[pair] += 1
    return dict(counts)



# PHASE 4 — Apply a single merge


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """
    Replace every occurrence of `pair` in `ids` with `new_id`.

    Args:
        ids:    a single token sequence (one chunk)
        pair:   the (token_a, token_b) pair to merge
        new_id: the new token id that replaces the pair

    Returns:
        new sequence with all occurrences of pair replaced

    Example:
        merge([104, 101, 108, 108, 111], (108, 108), 256)
        → [104, 101, 256, 111]

    Why a manual loop instead of something like str.replace()?
        We need to handle overlapping pairs correctly and work on
        lists of integers, not strings. A two-pointer scan is cleanest.
    """
    result = []
    i = 0
    while i < len(ids):
        # check if we're at the start of the target pair
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2          # skip both tokens in the pair
        else:
            result.append(ids[i])
            i += 1
    return result



# PHASE 5 — Training loop


class BPETokenizer:
    """
    Byte-level BPE tokenizer — GPT-4 style.

    Attributes:
        vocab_size:  target vocabulary size (256 base + num_merges)
        merges:      ordered dict of (pair → new_id), learned during training
        vocab:       dict of (id → bytes), the full vocabulary after training
        pattern:     regex pattern used for pre-tokenisation
    """

    def __init__(self, vocab_size: int = 4096, pattern=GPT4_SPLIT_PATTERN):
        assert vocab_size >= 256, "vocab_size must be at least 256 (one per byte)"
        self.vocab_size = vocab_size
        self.pattern = pattern

        # populated during train()
        self.merges: dict[tuple[int, int], int] = {}   # (pair) → new_id
        self.vocab:  dict[int, bytes] = {}              # id → bytes

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self, text: str) -> None:
        """
        Learn BPE merges from training text.

        Algorithm:
            1. Pre-tokenise text into chunks
            2. Encode each chunk as byte ids
            3. Repeat (vocab_size - 256) times:
               a. Count all adjacent pairs across all chunks
               b. Find the most frequent pair
               c. Assign it a new token id
               d. Apply the merge to all chunks
               e. Record the merge and update the vocab

        After training:
            self.merges contains all learned merges in order
            self.vocab   contains all 256 + num_merges tokens
        """
        num_merges = self.vocab_size - 256

        # Phase 1: start from base vocab (all 256 bytes)
        self.vocab = get_base_vocab()

        # Phase 2: pre-tokenise and convert to byte id sequences
        # ids is a list of lists — one inner list per chunk
        ids = pretokenise(text, self.pattern)

        # Phase 3-5: iteratively find and apply merges
        self.merges = {}

        for merge_idx in range(num_merges):
            # count all adjacent pairs across all chunks
            stats = get_stats(ids)

            if not stats:
                # can happen if text is very short — no pairs left to merge
                print(f"No more pairs to merge at step {merge_idx}. Stopping early.")
                break

            # find the most frequent pair
            # if there's a tie, max() picks the one that comes first
            # lexicographically — deterministic but arbitrary
            best_pair = max(stats, key=stats.get)
            best_count = stats[best_pair]

            # assign a new token id
            # merges start at 256 (after all base byte tokens)
            new_id = 256 + merge_idx

            # apply the merge to ALL chunks
            ids = [merge(chunk, best_pair, new_id) for chunk in ids]

            # record the merge (order matters for encode later)
            self.merges[best_pair] = new_id

            # extend the vocab: new token's bytes = concat of its two parts
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if (merge_idx + 1) % 100 == 0 or merge_idx < 10:
                print(
                    f"merge {merge_idx+1:4d}/{num_merges} | "
                    f"({best_pair[0]}, {best_pair[1]}) → {new_id} | "
                    f"'{self.vocab[new_id].decode('utf-8', errors='replace')}' | "
                    f"count: {best_count}"
                )

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Encode text to a list of token ids using learned merges.

        Algorithm:
            1. Pre-tokenise text into chunks (same pattern as training)
            2. Convert each chunk to byte ids
            3. Apply learned merges IN ORDER (order matters!)
            4. Concatenate all chunk id sequences

        Why apply merges in order?
            If we learned (101, 108) → 256 before (108, 111) → 257,
            then when encoding we must apply the (101, 108) merge first.
            Applying in wrong order produces different (incorrect) tokens.
        """
        all_ids = []

        for chunk in self.pattern.findall(text):
            # convert chunk string to list of byte ids
            chunk_ids = list(chunk.encode("utf-8"))

            # apply each learned merge in the order it was learned
            # self.merges is a regular dict — in Python 3.7+ dicts
            # preserve insertion order, so iteration order = merge order
            for pair, new_id in self.merges.items():
                chunk_ids = merge(chunk_ids, pair, new_id)

            all_ids.extend(chunk_ids)

        return all_ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode a list of texts. Simple wrapper around encode()."""
        return [self.encode(text) for text in texts]

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token ids back to a string.

        Algorithm:
            1. Look up each id in self.vocab to get its bytes
            2. Concatenate all bytes
            3. Decode as UTF-8

        Note: errors='replace' handles the edge case where a valid
        UTF-8 sequence was split across chunk boundaries during encoding.
        In practice this is rare with good pre-tokenisation.
        """
        token_bytes = b"".join(self.vocab[i] for i in ids)
        return token_bytes.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save merges to a text file.
        Format: one merge per line as "id_a id_b" (the pair that was merged)
        The new_id is implicit — it's 256 + line_number.
        """
        with open(path, "w", encoding="utf-8") as f:
            for (a, b), new_id in self.merges.items():
                f.write(f"{a} {b}\n")
        print(f"Saved {len(self.merges)} merges to {path}")

    def load(self, path: str) -> None:
        """Load merges from a saved file and rebuild vocab."""
        self.vocab = get_base_vocab()
        self.merges = {}

        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                a, b = map(int, line.strip().split())
                new_id = 256 + idx
                self.merges[(a, b)] = new_id
                self.vocab[new_id] = self.vocab[a] + self.vocab[b]

        print(f"Loaded {len(self.merges)} merges from {path}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def vocab_size_actual(self) -> int:
        return len(self.vocab)

    def token_to_str(self, token_id: int) -> str:
        """Return human-readable string for a token id."""
        return self.vocab[token_id].decode("utf-8", errors="replace")

    def str_to_token(self, s: str) -> int | None:
        """Look up token id for a string (reverse vocab lookup)."""
        target = s.encode("utf-8")
        for tid, b in self.vocab.items():
            if b == target:
                return tid
        return None



# Tests — run with: python bpe.py


def run_tests():
    print("=" * 60)
    print("TEST 1 — base vocab")
    print("=" * 60)
    vocab = get_base_vocab()
    assert len(vocab) == 256,         f"Expected 256, got {len(vocab)}"
    assert vocab[65]  == b'A',        f"Expected b'A', got {vocab[65]}"
    assert vocab[104] == b'h',        f"Expected b'h', got {vocab[104]}"
    assert vocab[0]   == b'\x00',     f"Expected b'\\x00', got {vocab[0]}"
    assert vocab[255] == b'\xff',     f"Expected b'\\xff', got {vocab[255]}"
    print("PASS — 256 tokens, spot checks correct\n")

    print("=" * 60)
    print("TEST 2 — pretokenise")
    print("=" * 60)
    chunks = pretokenise("Hello world!")
    print(f"'Hello world!' → {len(chunks)} chunks")
    for c in chunks:
        print(f"  {c} → '{bytes(c).decode('utf-8', errors='replace')}'")
    assert len(chunks) >= 2, "Expected at least 2 chunks"
    print("PASS\n")

    print("=" * 60)
    print("TEST 3 — get_stats")
    print("=" * 60)
    # [104,101,108,108,111] = b'hello'
    stats = get_stats([[104, 101, 108, 108, 111]])
    assert stats[(104, 101)] == 1
    assert stats[(101, 108)] == 1
    assert stats[(108, 108)] == 1
    assert stats[(108, 111)] == 1
    assert len(stats) == 4
    print(f"Stats for b'hello': {stats}")
    print("PASS\n")

    print("=" * 60)
    print("TEST 4 — get_stats does NOT cross chunk boundaries")
    print("=" * 60)
    # two chunks: [1,2] and [2,3]  — pair (2,3) only from chunk 2
    # pair (2,2) should NOT exist (would cross boundary)
    stats2 = get_stats([[1, 2], [2, 3]])
    assert (2, 2) not in stats2, "Cross-boundary pair should not exist"
    assert stats2.get((1, 2)) == 1
    assert stats2.get((2, 3)) == 1
    print(f"Stats for [[1,2],[2,3]]: {stats2}")
    print("PASS\n")

    print("=" * 60)
    print("TEST 5 — merge")
    print("=" * 60)
    result = merge([104, 101, 108, 108, 111], (108, 108), 256)
    assert result == [104, 101, 256, 111], f"Got {result}"
    print(f"merge([104,101,108,108,111], (108,108), 256) → {result}")

    # edge case: pair at very start
    r2 = merge([1, 2, 3], (1, 2), 99)
    assert r2 == [99, 3], f"Got {r2}"

    # edge case: pair at very end
    r3 = merge([1, 2, 3], (2, 3), 99)
    assert r3 == [1, 99], f"Got {r3}"

    # edge case: pair not present
    r4 = merge([1, 2, 3], (4, 5), 99)
    assert r4 == [1, 2, 3], f"Got {r4}"

    # edge case: overlapping — (1,1) in [1,1,1] → should give [256, 1] not [256, 256]
    # because after merging the first (1,1) we advance i by 2, leaving one 1
    r5 = merge([1, 1, 1], (1, 1), 256)
    assert r5 == [256, 1], f"Got {r5} — overlapping pair should not double-merge"
    print("PASS — all merge edge cases correct\n")

    print("=" * 60)
    print("TEST 6 — full train + encode + decode roundtrip")
    print("=" * 60)
    text = (
        "the quick brown fox jumps over the lazy dog. "
        "the dog barked at the fox. the fox ran away quickly. "
    ) * 20   # repeat to give BPE enough frequency signal

    tok = BPETokenizer(vocab_size=300)  # 44 merges on top of 256
    tok.train(text)

    # encode → decode should be lossless
    test_str = "the quick fox"
    ids = tok.encode(test_str)
    decoded = tok.decode(ids)
    assert decoded == test_str, f"Roundtrip failed: '{decoded}' != '{test_str}'"
    print(f"\nOriginal : '{test_str}'")
    print(f"Token ids: {ids}")
    print(f"Decoded  : '{decoded}'")
    print(f"Compression: {len(test_str.encode())} bytes → {len(ids)} tokens")
    print("PASS\n")

    print("=" * 60)
    print("TEST 7 — handles emoji and unicode (no UNK)")
    print("=" * 60)
    tok2 = BPETokenizer(vocab_size=260)
    tok2.train("hello world " * 50)   # train on simple text
    emoji_str = "hello 🎉 world 中文"
    ids2 = tok2.encode(emoji_str)
    decoded2 = tok2.decode(ids2)
    assert decoded2 == emoji_str, f"Unicode roundtrip failed: '{decoded2}'"
    print(f"'{emoji_str}' → {len(ids2)} tokens → '{decoded2}'")
    print("PASS — no [UNK], lossless on emoji and CJK\n")

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

    print("\n" + "=" * 60)
    print("DEMO — train on a larger text")
    print("=" * 60)

    sample = """
    In the beginning was the Word, and the Word was with God, and the Word was God.
    The quick brown fox jumps over the lazy dog.
    To be or not to be, that is the question.
    All happy families are alike; each unhappy family is unhappy in its own way.
    It was the best of times, it was the worst of times.
    """ * 30

    tokenizer = BPETokenizer(vocab_size=512)
    print(f"\nTraining on {len(sample)} chars with vocab_size=512...")
    tokenizer.train(sample)

    test = "the quick brown fox"
    ids = tokenizer.encode(test)
    print(f"\nEncoded '{test}':")
    print(f"  ids    : {ids}")
    print(f"  tokens : {[tokenizer.token_to_str(i) for i in ids]}")
    print(f"  decoded: '{tokenizer.decode(ids)}'")
    print(f"\nVocab size: {tokenizer.vocab_size_actual()}")
    print(f"Merges learned: {len(tokenizer.merges)}")