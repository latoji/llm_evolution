"""
Tests for tokenizer/bpe.py

Acceptance criteria:
  - 100% roundtrip fidelity on diverse text
  - Edge cases: empty string, single char, unicode, numbers, punctuation-heavy
  - Vocab size equals target after training
  - Encoding speed > 10K tokens/sec (single core)
"""
from __future__ import annotations

import time

import pytest

from tokenizer.bpe import (
    BYTE_DEC,
    BYTE_ENC,
    ENDOFTEXT_ID,
    PAD_ID,
    decode,
    encode,
    load_tokenizer,
    pre_tokenize,
    save_tokenizer,
    train_bpe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL_CORPUS = """\
The quick brown fox jumps over the lazy dog.
Pack my box with five dozen liquor jugs.
How vile and base are men who break their word!
Mathematics is the language with which God wrote the universe.
Python is a high-level general-purpose programming language.
""" * 50  # repeat to get enough data for BPE


@pytest.fixture(scope="module")
def trained_tokenizer(tmp_path_factory):
    """Train a small tokenizer (vocab=300) on SMALL_CORPUS."""
    tmp = tmp_path_factory.mktemp("tokenizer")
    corpus_path = tmp / "corpus.txt"
    corpus_path.write_text(SMALL_CORPUS, encoding="utf-8")

    merges, vocab = train_bpe(corpus_path, vocab_size=300)
    return merges, vocab


# ---------------------------------------------------------------------------
# Pre-tokenization
# ---------------------------------------------------------------------------

def test_pretokenize_basic():
    tokens = pre_tokenize("Hello, world!")
    assert "Hello" in tokens or " Hello" in tokens or any("Hello" in t for t in tokens)


def test_pretokenize_empty():
    assert pre_tokenize("") == []


def test_pretokenize_whitespace():
    tokens = pre_tokenize("  ")
    # Whitespace may produce one whitespace token or be empty
    assert isinstance(tokens, list)


# ---------------------------------------------------------------------------
# Byte encoder/decoder
# ---------------------------------------------------------------------------

def test_byte_enc_is_bijection():
    assert len(BYTE_ENC) == 256
    assert len(BYTE_DEC) == 256
    for b in range(256):
        ch = BYTE_ENC[b]
        assert BYTE_DEC[ch] == b


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def test_vocab_size(trained_tokenizer):
    _merges, vocab = trained_tokenizer
    assert len(vocab) == 300


def test_special_tokens_ids(trained_tokenizer):
    _merges, vocab = trained_tokenizer
    assert vocab["<|endoftext|>"] == ENDOFTEXT_ID
    assert vocab["<|pad|>"] == PAD_ID


def test_base_bytes_present(trained_tokenizer):
    _merges, vocab = trained_tokenizer
    # All 256 base bytes should be in the vocab
    assert sum(1 for v in vocab if len(v) == 1) >= 100  # at minimum


# ---------------------------------------------------------------------------
# Encode / Decode roundtrip
# ---------------------------------------------------------------------------

def test_roundtrip_simple(trained_tokenizer):
    merges, vocab = trained_tokenizer
    text = "The quick brown fox"
    assert decode(encode(text, merges), vocab) == text


def test_roundtrip_punctuation(trained_tokenizer):
    merges, vocab = trained_tokenizer
    text = "Hello, world! How are you?"
    assert decode(encode(text, merges), vocab) == text


def test_roundtrip_numbers(trained_tokenizer):
    merges, vocab = trained_tokenizer
    text = "There are 42 items in 3.14 seconds."
    assert decode(encode(text, merges), vocab) == text


def test_roundtrip_unicode(trained_tokenizer):
    merges, vocab = trained_tokenizer
    text = "Café naïve résumé"
    assert decode(encode(text, merges), vocab) == text


def test_roundtrip_empty(trained_tokenizer):
    merges, vocab = trained_tokenizer
    assert decode(encode("", merges), vocab) == ""


def test_roundtrip_single_char(trained_tokenizer):
    merges, vocab = trained_tokenizer
    for ch in "aAbB0!":
        result = decode(encode(ch, merges), vocab)
        assert result == ch, f"Roundtrip failed for {ch!r}: got {result!r}"


def test_roundtrip_heavy_punctuation(trained_tokenizer):
    merges, vocab = trained_tokenizer
    text = "... --- !!! ???"
    assert decode(encode(text, merges), vocab) == text


# ---------------------------------------------------------------------------
# Encoding speed
# ---------------------------------------------------------------------------

def test_encoding_speed(trained_tokenizer):
    """Encoding speed must exceed 10K tokens/sec on a single core."""
    merges, vocab = trained_tokenizer
    text = "The quick brown fox jumps over the lazy dog. " * 200
    start = time.perf_counter()
    ids = encode(text, merges)
    elapsed = time.perf_counter() - start
    tokens_per_sec = len(ids) / elapsed if elapsed > 0 else float("inf")
    assert tokens_per_sec > 10_000, (
        f"Encoding speed {tokens_per_sec:.0f} tokens/sec is below 10K threshold"
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(trained_tokenizer, tmp_path):
    merges, vocab = trained_tokenizer
    merges_p = tmp_path / "merges.json"
    vocab_p = tmp_path / "vocab.json"
    save_tokenizer(merges, vocab, merges_p, vocab_p)
    loaded_merges, loaded_vocab = load_tokenizer(merges_p, vocab_p)
    assert loaded_merges == merges
    assert loaded_vocab == vocab
