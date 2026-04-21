"""Tests for model/word_ngram.py.

All tests use a per-test DuckDB Store backed by a temp file, ensuring isolation.
"""
from __future__ import annotations

import random

import pytest

from db.store import Store
from model.word_ngram import WordNGramModel
from wvm.validator import Validator


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

TRAIN_TEXT = "the cat sat on the mat the cat"


@pytest.fixture()
def validator():
    return Validator()


@pytest.fixture()
def mem_store(tmp_path):
    s = Store(tmp_path / "test.duckdb")
    yield s
    s.close()


@pytest.fixture()
def model(mem_store, validator):
    """Order-2 WordNGramModel trained on TRAIN_TEXT."""
    m = WordNGramModel(order=2, store=mem_store, validator=validator)
    m.train_on(TRAIN_TEXT)
    return m


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_order_1_instantiates(mem_store, validator):
    m = WordNGramModel(order=1, store=mem_store, validator=validator)
    assert m.order == 1


def test_order_2_instantiates(mem_store, validator):
    m = WordNGramModel(order=2, store=mem_store, validator=validator)
    assert m.order == 2


def test_order_3_instantiates(mem_store, validator):
    m = WordNGramModel(order=3, store=mem_store, validator=validator)
    assert m.order == 3


def test_invalid_order_raises(mem_store, validator):
    with pytest.raises(ValueError):
        WordNGramModel(order=4, store=mem_store, validator=validator)
    with pytest.raises(ValueError):
        WordNGramModel(order=0, store=mem_store, validator=validator)


# ---------------------------------------------------------------------------
# Training — DB row verification
# ---------------------------------------------------------------------------

def test_train_on_writes_unigrams(mem_store, validator):
    m = WordNGramModel(order=1, store=mem_store, validator=validator)
    m.train_on("the cat sat")
    rows = mem_store.get_ngrams("word", 1, limit=100)
    words = {r["next_item"] for r in rows}
    assert "the" in words
    assert "cat" in words
    assert "sat" in words


def test_train_on_writes_bigrams(mem_store, validator):
    m = WordNGramModel(order=2, store=mem_store, validator=validator)
    m.train_on("the cat sat")
    dist = mem_store.get_distribution("word", 2, "the")
    assert dist.get("cat", 0) >= 1


def test_train_on_correct_unigram_counts(mem_store, validator):
    """'the cat the': 'the' appears twice, 'cat' once."""
    m = WordNGramModel(order=1, store=mem_store, validator=validator)
    m.train_on("the cat the")
    dist = mem_store.get_distribution("word", 1, "")
    assert dist["the"] == 2
    assert dist["cat"] == 1


def test_train_on_strips_punctuation(mem_store, validator):
    """Boundary punctuation is stripped by Validator; 'hello' not 'hello,'."""
    m = WordNGramModel(order=1, store=mem_store, validator=validator)
    m.train_on("hello, world!")
    dist = mem_store.get_distribution("word", 1, "")
    assert dist.get("hello", 0) >= 1
    assert dist.get("world", 0) >= 1


def test_train_on_empty_text_noop(mem_store, validator):
    m = WordNGramModel(order=2, store=mem_store, validator=validator)
    m.train_on("")
    rows = mem_store.get_ngrams("word", 1, limit=10)
    assert rows == []


def test_train_on_incremental(mem_store, validator):
    """Two calls accumulate counts."""
    m = WordNGramModel(order=1, store=mem_store, validator=validator)
    m.train_on("cat")
    m.train_on("cat")
    dist = mem_store.get_distribution("word", 1, "")
    assert dist["cat"] == 2


# ---------------------------------------------------------------------------
# next_word_distribution
# ---------------------------------------------------------------------------

def test_distribution_sums_to_one(model):
    dist = model.next_word_distribution(("the",))
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-6, f"Distribution sums to {total}"


def test_distribution_all_nonnegative(model):
    dist = model.next_word_distribution(("cat",))
    assert all(p >= 0 for p in dist.values())


def test_distribution_words_in_vocab(model, mem_store):
    dist = model.next_word_distribution(("the",))
    vocab = set(mem_store.get_distribution("word", 1, "").keys())
    for word in dist:
        assert word in vocab, f"Word '{word}' not in training vocab"


def test_distribution_empty_context_uses_unigram(model):
    """Empty context should fall back to unigram distribution."""
    dist = model.next_word_distribution(())
    assert len(dist) > 0
    assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_distribution_empty_when_no_data(mem_store, validator):
    m = WordNGramModel(order=2, store=mem_store, validator=validator)
    dist = m.next_word_distribution(("the",))
    assert dist == {}


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def test_generate_produces_words(model):
    result = model.generate(5, rng=random.Random(42))
    words = result.split()
    assert len(words) == 5


def test_generate_words_in_vocab(mem_store, validator):
    m = WordNGramModel(order=2, store=mem_store, validator=validator)
    m.train_on("dog cat bird")
    vocab = set(mem_store.get_distribution("word", 1, "").keys())
    result = m.generate(10, rng=random.Random(7))
    for word in result.split():
        assert word in vocab, f"Generated word '{word}' not in training vocab"


def test_generate_with_seed_context(model):
    result = model.generate(4, seed_context=("the",), rng=random.Random(3))
    assert isinstance(result, str)


def test_generate_deterministic(model):
    r1 = model.generate(8, rng=random.Random(55))
    r2 = model.generate(8, rng=random.Random(55))
    assert r1 == r2
