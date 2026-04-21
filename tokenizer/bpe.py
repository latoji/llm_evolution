"""
Track 1 – BPE Tokenizer
Byte Pair Encoding tokenizer implemented from scratch.
No tiktoken, no sentencepiece.

Vocabulary layout
─────────────────
  ID 0  : <|endoftext|>
  ID 1  : <|pad|>
  ID 2-257 : base byte tokens (byte value b → ID b + 2)
  ID 258+  : learned merge tokens (in merge order)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterator

import regex
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------
ENDOFTEXT = "<|endoftext|>"
PAD = "<|pad|>"
ENDOFTEXT_ID = 0
PAD_ID = 1
_BASE_OFFSET = 2  # byte b → token ID b + _BASE_OFFSET

# GPT-2 pre-tokenization regex (handles contractions, words, numbers, punct)
_PRETOK = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"""
)


# ---------------------------------------------------------------------------
# Byte ↔ printable-unicode mapping (GPT-2 approach)
# ---------------------------------------------------------------------------

def _build_byte_encoder() -> dict[int, str]:
    """Map each byte 0-255 to a unique printable unicode character."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = list(printable)
    extra = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(printable, [chr(c) for c in chars]))


BYTE_ENC: dict[int, str] = _build_byte_encoder()
BYTE_DEC: dict[str, int] = {v: k for k, v in BYTE_ENC.items()}


# ---------------------------------------------------------------------------
# Pre-tokenization
# ---------------------------------------------------------------------------

def pre_tokenize(text: str) -> list[str]:
    """Split text into pre-tokens using GPT-2 style regex."""
    return _PRETOK.findall(text)


def _word_to_bytes(word: str) -> tuple[str, ...]:
    """Convert a word string to a tuple of printable byte characters."""
    return tuple(BYTE_ENC[b] for b in word.encode("utf-8"))


# ---------------------------------------------------------------------------
# BPE training helpers
# ---------------------------------------------------------------------------

def _count_pairs(word_splits: dict[tuple[str, ...], int]) -> Counter:
    """Count every adjacent pair across all words, weighted by frequency."""
    pairs: Counter = Counter()
    for word, freq in word_splits.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def _apply_merge(
    word_splits: dict[tuple[str, ...], int],
    pair: tuple[str, str],
) -> dict[tuple[str, ...], int]:
    """Apply a single merge to every word in the split table."""
    merged = pair[0] + pair[1]
    new_splits: dict[tuple[str, ...], int] = {}
    for word, freq in word_splits.items():
        new_word: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_splits[tuple(new_word)] = freq
    return new_splits


def _iter_words(corpus_path: Path, chunk_lines: int = 100_000) -> Iterator[str]:
    """Yield pre-tokens from corpus_path in chunks."""
    chunk: list[str] = []
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            chunk.append(line.rstrip("\n"))
            if len(chunk) >= chunk_lines:
                for word in pre_tokenize(" ".join(chunk)):
                    yield word
                chunk = []
    if chunk:
        for word in pre_tokenize(" ".join(chunk)):
            yield word


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_bpe(
    corpus_path: Path | str,
    vocab_size: int = 8000,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """
    Train BPE on corpus_path.

    Args:
        corpus_path: UTF-8 text file, one paragraph per line.
        vocab_size: Target vocabulary size (≥ 258).

    Returns:
        merges: Ordered list of (token_a, token_b) merge pairs.
        vocab:  Mapping from token string → integer ID.
    """
    corpus_path = Path(corpus_path)
    num_merges = vocab_size - (256 + 2)  # subtract base bytes + special tokens
    if num_merges <= 0:
        raise ValueError(f"vocab_size must be > 258, got {vocab_size}")

    # Build word frequency table
    print("Building word frequency table…")
    word_freq: Counter = Counter()
    for word in tqdm(_iter_words(corpus_path), desc="Pre-tokenizing"):
        word_freq[word] += 1

    # Convert words to byte-character tuples
    word_splits: dict[tuple[str, ...], int] = {
        _word_to_bytes(word): freq for word, freq in word_freq.items()
    }

    # Seed vocabulary
    vocab: dict[str, int] = {
        ENDOFTEXT: ENDOFTEXT_ID,
        PAD: PAD_ID,
        **{BYTE_ENC[b]: b + _BASE_OFFSET for b in range(256)},
    }
    merges: list[tuple[str, str]] = []

    print(f"Training BPE: {num_merges} merges → vocab_size={vocab_size}…")
    for _ in tqdm(range(num_merges), desc="BPE merges"):
        pairs = _count_pairs(word_splits)
        if not pairs:
            break

        best: tuple[str, str] = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab[best[0] + best[1]] = len(vocab)
        word_splits = _apply_merge(word_splits, best)

    return merges, vocab


def encode(text: str, merges: list[tuple[str, str]]) -> list[int]:
    """
    Encode text to token IDs using learned BPE merges.
    The vocabulary is reconstructed deterministically from merges.
    """
    # Rebuild vocab from merges (deterministic)
    vocab: dict[str, int] = {
        ENDOFTEXT: ENDOFTEXT_ID,
        PAD: PAD_ID,
        **{BYTE_ENC[b]: b + _BASE_OFFSET for b in range(256)},
    }
    for a, b in merges:
        vocab[a + b] = len(vocab)

    merge_rank: dict[tuple[str, str], int] = {pair: i for i, pair in enumerate(merges)}

    ids: list[int] = []
    for word in pre_tokenize(text):
        ids.extend(_encode_word(list(_word_to_bytes(word)), merge_rank, vocab))
    return ids


def _encode_word(
    chars: list[str],
    merge_rank: dict[tuple[str, str], int],
    vocab: dict[str, int],
) -> list[int]:
    """Apply BPE merges to a single pre-token."""
    if len(chars) == 1:
        return [vocab.get(chars[0], ENDOFTEXT_ID)]

    word = list(chars)
    while len(word) >= 2:
        best_rank = float("inf")
        best_idx = -1
        for i in range(len(word) - 1):
            rank = merge_rank.get((word[i], word[i + 1]), float("inf"))
            if rank < best_rank:
                best_rank = rank
                best_idx = i
        if best_idx == -1:
            break
        merged = word[best_idx] + word[best_idx + 1]
        word = word[:best_idx] + [merged] + word[best_idx + 2:]

    return [vocab.get(tok, ENDOFTEXT_ID) for tok in word]


def decode(token_ids: list[int], vocab: dict[str, int]) -> str:
    """Decode token IDs back to a unicode string."""
    id_to_token = {v: k for k, v in vocab.items()}
    byte_chars: list[str] = []
    for tid in token_ids:
        tok = id_to_token.get(tid, "")
        if tok in (ENDOFTEXT, PAD, ""):
            continue
        byte_chars.append(tok)

    raw_bytes: list[int] = []
    for ch in "".join(byte_chars):
        if ch in BYTE_DEC:
            raw_bytes.append(BYTE_DEC[ch])

    return bytes(raw_bytes).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_tokenizer(
    merges: list[tuple[str, str]],
    vocab: dict[str, int],
    merges_path: Path | str,
    vocab_path: Path | str,
) -> None:
    """Persist merges and vocab to JSON files."""
    Path(merges_path).write_text(
        json.dumps(merges, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(vocab_path).write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(merges)} merges → {merges_path}")
    print(f"Saved {len(vocab)} vocab entries → {vocab_path}")


def load_tokenizer(
    merges_path: Path | str,
    vocab_path: Path | str,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Load merges and vocab from JSON files."""
    raw = json.loads(Path(merges_path).read_text(encoding="utf-8"))
    merges: list[tuple[str, str]] = [tuple(p) for p in raw]  # type: ignore[misc]
    vocab: dict[str, int] = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    return merges, vocab
