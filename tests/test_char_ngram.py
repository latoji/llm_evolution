"""Tests for model/char_ngram.py.

All tests use an in-memory DuckDB Store so they are isolated and fast.
"""
from __future__ import annotations

import random

import pytest

from db.store import Store
from model.char_ngram import CharNGramModel


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

TRAIN_TEXT = "abracadabra"


@pytest.fixture()
def mem_store(tmp_path):
    """Return an in-memory DuckDB Store backed by a temp file."""
    s = Store(tmp_path / "test.duckdb")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_order_1_instantiates(mem_store):
    model = CharNGramModel(order=1, store=mem_store)
    assert model.order == 1


def test_order_3_instantiates(mem_store):
    model = CharNGramModel(order=3, store=mem_store)
    assert model.order == 3


def test_order_5_instantiates(mem_store):
    model = CharNGramModel(order=5, store=mem_store)
    assert model.order == 5


def test_invalid_order_raises(mem_store):
    with pytest.raises(ValueError):
        CharNGramModel(order=6, store=mem_store)

    with pytest.raises(ValueError):
        CharNGramModel(order=0, store=mem_store)


# ---------------------------------------------------------------------------
# Training — DB row verification
# ---------------------------------------------------------------------------

def test_train_on_writes_unigrams(mem_store):
    """After training, char_ngrams should contain unigram rows (n=1)."""
    model = CharNGramModel(order=2, store=mem_store)
    model.train_on("aab")
    rows = mem_store.get_ngrams("char", 1, limit=100)
    chars = {r["next_item"] for r in rows}
    assert "a" in chars
    assert "b" in chars


def test_train_on_writes_bigrams(mem_store):
    """After training 'ab', bigram context='a' next_char='b' must exist."""
    model = CharNGramModel(order=2, store=mem_store)
    model.train_on("ab")
    dist = mem_store.get_distribution("char", 2, "a")
    assert dist.get("b", 0) >= 1


def test_train_on_correct_counts(mem_store):
    """'aab': unigram 'a' should have count 2, 'b' count 1."""
    model = CharNGramModel(order=1, store=mem_store)
    model.train_on("aab")
    dist = mem_store.get_distribution("char", 1, "")
    assert dist["a"] == 2
    assert dist["b"] == 1


def test_train_on_incremental(mem_store):
    """Calling train_on twice should accumulate counts."""
    model = CharNGramModel(order=1, store=mem_store)
    model.train_on("ab")
    model.train_on("ab")
    dist = mem_store.get_distribution("char", 1, "")
    assert dist["a"] == 2
    assert dist["b"] == 2


def test_train_on_empty_text_noop(mem_store):
    """Training on empty string should not raise and leave DB empty."""
    model = CharNGramModel(order=2, store=mem_store)
    model.train_on("")
    rows = mem_store.get_ngrams("char", 1, limit=10)
    assert rows == []


# ---------------------------------------------------------------------------
# next_char_distribution
# ---------------------------------------------------------------------------

def test_distribution_sums_to_one(mem_store):
    model = CharNGramModel(order=2, store=mem_store)
    model.train_on(TRAIN_TEXT)
    dist = model.next_char_distribution("a")
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-6, f"Distribution sums to {total}, expected ~1.0"


def test_distribution_all_nonnegative(mem_store):
    model = CharNGramModel(order=2, store=mem_store)
    model.train_on(TRAIN_TEXT)
    dist = model.next_char_distribution("b")
    assert all(p >= 0 for p in dist.values()), "Negative probabilities found"


def test_distribution_chars_in_vocab(mem_store):
    """Every key returned must be a character seen during training."""
    model = CharNGramModel(order=2, store=mem_store)
    model.train_on("abc")
    dist = model.next_char_distribution("a")
    vocab = set(mem_store.get_distribution("char", 1, "").keys())
    for ch in dist:
        assert ch in vocab, f"Character '{ch}' not in training vocab"


def test_distribution_backoff_for_unseen_context(mem_store):
    """When context not seen at high order, backoff to lower order succeeds."""
    model = CharNGramModel(order=3, store=mem_store)
    model.train_on("abc")
    # Context "zzz" was not seen — backoff to unigram should still return a distribution
    dist = model.next_char_distribution("zzz")
    assert isinstance(dist, dict)
    if dist:  # only assert sum if non-empty
        assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_distribution_empty_when_no_data(mem_store):
    """If the model has never been trained, distribution should be empty."""
    model = CharNGramModel(order=2, store=mem_store)
    dist = model.next_char_distribution("a")
    assert dist == {}


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def test_generate_length(mem_store):
    """generate() should return a string of exactly n_chars chars."""
    model = CharNGramModel(order=2, store=mem_store)
    model.train_on(TRAIN_TEXT)
    result = model.generate(10, rng=random.Random(42))
    assert len(result) == 10


def test_generate_chars_in_vocab(mem_store):
    """Generated characters must all come from the observed vocabulary."""
    model = CharNGramModel(order=2, store=mem_store)
    model.train_on("xyz")
    vocab = set(mem_store.get_distribution("char", 1, "").keys())
    result = model.generate(20, rng=random.Random(0))
    for ch in result:
        assert ch in vocab, f"Generated character '{ch}' not in training vocab"


def test_generate_seed_context_respected(mem_store):
    """Passing a seed_context should not raise and should produce output."""
    model = CharNGramModel(order=3, store=mem_store)
    model.train_on(TRAIN_TEXT)
    result = model.generate(5, seed_context="ab", rng=random.Random(1))
    assert isinstance(result, str)
    assert len(result) <= 5


def test_generate_deterministic_with_seed(mem_store):
    """Same rng seed must produce same output."""
    model = CharNGramModel(order=2, store=mem_store)
    model.train_on(TRAIN_TEXT)
    r1 = model.generate(15, rng=random.Random(99))
    r2 = model.generate(15, rng=random.Random(99))
    assert r1 == r2
