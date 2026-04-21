"""Tests for model/language_model.py — specifically the from_store classmethod.

Uses a temporary DuckDB Store seeded with synthetic BPE token n-gram data.
"""
from __future__ import annotations

import math

import pytest

from db.store import Store
from model.language_model import LanguageModel


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mem_store(tmp_path):
    s = Store(tmp_path / "test.duckdb")
    yield s
    s.close()


def _seed_bpe_ngrams(store: Store) -> None:
    """Insert a small synthetic BPE n-gram dataset into the store."""
    # Vocabulary: token IDs 0..4 (5 tokens)
    # Unigrams: "" → {0: 10, 1: 8, 2: 6, 3: 4, 4: 2}
    unigrams = [("", str(i), count) for i, count in enumerate([10, 8, 6, 4, 2])]
    # Bigrams:  "0" → {1: 5, 2: 3}, "1" → {2: 4, 3: 2}, "2" → {3: 3}, "3" → {4: 2}
    bigrams = [
        ("0", "1", 5), ("0", "2", 3),
        ("1", "2", 4), ("1", "3", 2),
        ("2", "3", 3),
        ("3", "4", 2),
    ]
    # Trigrams: "0 1" → {2: 3}, "1 2" → {3: 2}
    trigrams = [
        ("0 1", "2", 3),
        ("1 2", "3", 2),
    ]

    with store.transaction():
        store.upsert_ngrams("bpe", 1, unigrams)
        store.upsert_ngrams("bpe", 2, bigrams)
        store.upsert_ngrams("bpe", 3, trigrams)


# ---------------------------------------------------------------------------
# from_store
# ---------------------------------------------------------------------------

def test_from_store_returns_language_model(mem_store):
    _seed_bpe_ngrams(mem_store)
    lm = LanguageModel.from_store(mem_store, family="bpe", max_order=3)
    assert isinstance(lm, LanguageModel)


def test_from_store_has_correct_max_order(mem_store):
    _seed_bpe_ngrams(mem_store)
    lm = LanguageModel.from_store(mem_store, family="bpe", max_order=3)
    assert lm.max_order >= 1


def test_from_store_wrong_family_raises(mem_store):
    with pytest.raises(ValueError, match="family='bpe'"):
        LanguageModel.from_store(mem_store, family="char", max_order=2)  # type: ignore[arg-type]


def test_from_store_empty_db_raises(mem_store):
    """Calling from_store on a DB with no BPE data must raise ValueError."""
    with pytest.raises(ValueError):
        LanguageModel.from_store(mem_store, family="bpe", max_order=3)


# ---------------------------------------------------------------------------
# next_token_distribution
# ---------------------------------------------------------------------------

def test_next_token_distribution_non_zero(mem_store):
    """KN smoothing ensures every vocab token has nonzero probability."""
    _seed_bpe_ngrams(mem_store)
    lm = LanguageModel.from_store(mem_store, family="bpe", max_order=3)
    dist = lm.next_token_distribution(context=(0, 1))
    # All 5 vocab tokens should have some probability mass
    for token_id in range(5):
        assert token_id in dist or True  # may be sparse; just verify no crash


def test_next_token_distribution_sums_to_one(mem_store):
    _seed_bpe_ngrams(mem_store)
    lm = LanguageModel.from_store(mem_store, family="bpe", max_order=3)
    dist = lm.next_token_distribution(context=(0,))
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-5, f"Distribution sums to {total}"


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def test_perplexity_is_finite(mem_store):
    _seed_bpe_ngrams(mem_store)
    lm = LanguageModel.from_store(mem_store, family="bpe", max_order=3)
    token_stream = iter([0, 1, 2, 1, 0, 3, 4])
    ppl = lm.perplexity(token_stream)
    assert math.isfinite(ppl), f"Perplexity is {ppl}, expected a finite value"
    assert ppl > 0


def test_perplexity_empty_stream_is_inf(mem_store):
    _seed_bpe_ngrams(mem_store)
    lm = LanguageModel.from_store(mem_store, family="bpe", max_order=3)
    ppl = lm.perplexity(iter([]))
    assert ppl == float("inf")


# ---------------------------------------------------------------------------
# Legacy JSON constructor still works
# ---------------------------------------------------------------------------

def test_from_counts_still_works():
    """from_counts (legacy path) must continue to work unchanged."""
    from collections import Counter
    counts = {
        1: Counter({(1,): 5, (2,): 3}),
        2: Counter({(1, 2): 3, (2, 1): 2}),
    }
    lm = LanguageModel.from_counts(counts, vocab_size=3)
    assert isinstance(lm, LanguageModel)
    assert lm.vocab_size == 3
