"""
Tests for model/smoothing.py and model/language_model.py

Acceptance criteria:
  - All distributions sum to 1.0 (± 0.001)
  - No token gets probability 0 (smoothing working)
  - Seen n-grams get higher probability than unseen ones
  - Backoff: model returns valid distribution for novel context
  - Perplexity is finite
"""
from __future__ import annotations

import math
from collections import Counter

import pytest

from model.language_model import LanguageModel
from model.smoothing import (
    compute_continuation_counts,
    compute_discounts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 20

# Dense training data for reliable counts
TOKEN_SEQ = [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 2, 4, 7, 8, 9, 1, 2, 3, 5, 6]


def _build_counts(seq: list[int], max_n: int = 4) -> dict[int, Counter]:
    from model.ngram_counter import count_ngrams  # noqa: PLC0415
    return count_ngrams(iter(seq), max_n=max_n)


@pytest.fixture(scope="module")
def lm() -> LanguageModel:
    counts = _build_counts(TOKEN_SEQ)
    return LanguageModel.from_counts(counts, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# Discounts
# ---------------------------------------------------------------------------

def test_discounts_in_valid_range():
    counts = _build_counts(TOKEN_SEQ)
    for n, counter in counts.items():
        d = compute_discounts(counter)
        assert 0.0 <= d.d1 <= 1.0, f"d1 out of range at order {n}: {d.d1}"
        assert 0.0 <= d.d2 <= 2.0, f"d2 out of range at order {n}: {d.d2}"
        assert 0.0 <= d.d3plus <= 3.0, f"d3+ out of range at order {n}: {d.d3plus}"


def test_continuation_counts():
    counts = _build_counts(TOKEN_SEQ)
    continuation, total = compute_continuation_counts(counts.get(2, Counter()))
    assert total == len(counts.get(2, Counter())), "total != unique bigrams"
    for key, v in continuation.items():
        assert isinstance(key, tuple) and len(key) == 1


# ---------------------------------------------------------------------------
# Probability properties
# ---------------------------------------------------------------------------

def test_no_zero_probability(lm):
    """Every token should have non-zero probability under any context."""
    for context in [(), (1,), (1, 2), (1, 2, 3)]:
        for token in range(VOCAB_SIZE):
            p = lm.prob(token, context)
            assert p > 0, (
                f"Zero probability for token={token}, context={context}"
            )


def test_distribution_sums_to_one(lm):
    """Full next-token distribution must sum to ~1.0."""
    for context in [(), (1,), (1, 2), (99, 98, 97)]:  # last is unseen
        dist = lm.next_token_distribution(context)
        total = sum(dist.values())
        assert abs(total - 1.0) < 0.001, (
            f"Distribution sum {total:.6f} ≠ 1.0 for context {context}"
        )


def test_seen_higher_than_unseen(lm):
    """
    Tokens seen after context (1,) should have higher probability than
    tokens completely absent from training.
    """
    # Token 2 follows token 1 many times in TOKEN_SEQ
    p_seen = lm.prob(2, (1,))
    # Token 19 never appears in TOKEN_SEQ at all
    p_unseen = lm.prob(19, (1,))
    assert p_seen > p_unseen, (
        f"Seen token prob {p_seen:.6f} not > unseen token prob {p_unseen:.6f}"
    )


def test_novel_context_gives_valid_distribution(lm):
    """Distribution for a context never seen in training must still be valid."""
    novel = (50, 51, 52)  # way outside training data range
    dist = lm.next_token_distribution(novel)
    assert len(dist) > 0
    total = sum(dist.values())
    assert abs(total - 1.0) < 0.001


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def test_perplexity_finite(lm):
    """Perplexity on the training sequence must be finite."""
    ppl = lm.perplexity(iter(TOKEN_SEQ))
    assert math.isfinite(ppl), f"Perplexity is not finite: {ppl}"
    assert ppl > 1.0, "Perplexity should be > 1"


def test_perplexity_empty_stream(lm):
    ppl = lm.perplexity(iter([]))
    assert ppl == float("inf")
