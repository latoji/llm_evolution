"""
Tests for generate/generator.py

Acceptance criteria:
  - Fixed seed produces identical output across runs
  - Temperature ≈ 0 always picks highest-probability token
  - Output is valid decodable text (no broken unicode)
  - Generation stops at <|endoftext|>
"""
from __future__ import annotations


import pytest

from generate.generator import generate, generate_stream
from model.language_model import LanguageModel
from tokenizer.bpe import encode, train_bpe



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL_CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vile and base are men who break their word. "
) * 60


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gen")
    cp = tmp / "corpus.txt"
    cp.write_text(SMALL_CORPUS, encoding="utf-8")
    merges, vocab = train_bpe(cp, vocab_size=300)
    return merges, vocab


@pytest.fixture(scope="module")
def lm(tokenizer):
    from model.ngram_counter import count_ngrams  # noqa: PLC0415
    merges, vocab = tokenizer
    ids = encode(SMALL_CORPUS, merges)
    counts = count_ngrams(iter(ids), max_n=4)
    return LanguageModel.from_counts(counts, len(vocab))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_fixed_seed_deterministic(lm, tokenizer):
    merges, vocab = tokenizer
    out1 = generate(lm, merges, vocab, "The quick", seed=42, max_tokens=20)
    out2 = generate(lm, merges, vocab, "The quick", seed=42, max_tokens=20)
    assert out1 == out2, "Different seed=42 runs produced different output"


def test_different_seeds_may_differ(lm, tokenizer):
    merges, vocab = tokenizer
    out1 = generate(lm, merges, vocab, "The quick", seed=1, max_tokens=20)
    out2 = generate(lm, merges, vocab, "The quick", seed=999, max_tokens=20)
    # Not guaranteed to differ, but with a trained model and 20 tokens they usually do
    # We just ensure both are valid strings
    assert isinstance(out1, str)
    assert isinstance(out2, str)


# ---------------------------------------------------------------------------
# Temperature behaviour
# ---------------------------------------------------------------------------

def test_low_temperature_greedy(lm, tokenizer):
    """Very low temperature should consistently pick the top token."""
    merges, vocab = tokenizer
    results = [
        generate(lm, merges, vocab, "The", temperature=0.01, max_tokens=5, seed=i)
        for i in range(3)
    ]
    # With temp ≈ 0, all runs should produce the same output
    assert len(set(results)) == 1, (
        f"Low temperature should be deterministic: {results}"
    )


# ---------------------------------------------------------------------------
# Valid text output
# ---------------------------------------------------------------------------

def test_output_is_valid_unicode(lm, tokenizer):
    merges, vocab = tokenizer
    output = generate(lm, merges, vocab, "Pack", seed=0, max_tokens=30)
    # Should not raise; should be valid unicode
    output.encode("utf-8")
    assert isinstance(output, str)


def test_max_tokens_respected(lm, tokenizer):
    merges, vocab = tokenizer
    ids = encode(
        generate(lm, merges, vocab, "The", seed=0, max_tokens=10) or "",
        merges,
    )
    assert len(ids) <= 10


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def test_stream_matches_full_output(lm, tokenizer):
    merges, vocab = tokenizer
    streamed = "".join(
        generate_stream(lm, merges, vocab, "The quick", seed=42, max_tokens=20)
    )
    full = generate(lm, merges, vocab, "The quick", seed=42, max_tokens=20)
    assert streamed == full
