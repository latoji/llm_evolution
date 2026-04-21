"""
Tests for model/ngram_counter.py

Acceptance criteria:
  - Exact counts on hand-crafted sequences
  - Pruning removes expected entries
  - Unigram counts sum to total token count
  - Deterministic (same input → same output)
"""
from __future__ import annotations



from model.ngram_counter import count_ngrams, prune_counts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Hand-crafted 20-token sequence for exact verification
TOKEN_SEQ = [1, 2, 3, 1, 2, 3, 1, 2, 4, 5, 6, 7, 1, 2, 3, 8, 9, 10, 1, 2]
# Total tokens: 20
# Notable patterns:
#   unigram (1,) → 5 times
#   bigram (1, 2) → 5 times
#   trigram (1, 2, 3) → 3 times


# ---------------------------------------------------------------------------
# Basic counting
# ---------------------------------------------------------------------------

def test_unigram_total():
    counts = count_ngrams(iter(TOKEN_SEQ), max_n=1)
    total = sum(counts[1].values())
    assert total == len(TOKEN_SEQ), (
        f"Unigram total {total} != token count {len(TOKEN_SEQ)}"
    )


def test_exact_unigram_counts():
    counts = count_ngrams(iter(TOKEN_SEQ), max_n=1)
    assert counts[1][(1,)] == 5
    assert counts[1][(2,)] == 5
    assert counts[1][(3,)] == 3


def test_exact_bigram_counts():
    counts = count_ngrams(iter(TOKEN_SEQ), max_n=2)
    assert counts[2][(1, 2)] == 5
    assert counts[2][(2, 3)] == 3


def test_exact_trigram_counts():
    counts = count_ngrams(iter(TOKEN_SEQ), max_n=3)
    assert counts[3][(1, 2, 3)] == 3


def test_fourgram_counts():
    counts = count_ngrams(iter(TOKEN_SEQ), max_n=4)
    # (1, 2, 3, 1) appears at positions 0-3 and 3-6 → count 2
    assert counts[4][(1, 2, 3, 1)] == 2


def test_all_orders_present():
    counts = count_ngrams(iter(TOKEN_SEQ), max_n=4)
    assert set(counts.keys()) == {1, 2, 3, 4}


def test_max_n_respected():
    counts = count_ngrams(iter(TOKEN_SEQ), max_n=2)
    assert 3 not in counts
    assert 4 not in counts


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def test_pruning_removes_low_counts():
    counts = count_ngrams(iter(TOKEN_SEQ), max_n=4)
    pruned = prune_counts(counts, min_count=3)

    # Bigrams with count < 3 should be removed
    for ngram, c in pruned[2].items():
        assert c >= 3, f"Bigram {ngram} with count {c} survived pruning"


def test_pruning_keeps_high_counts():
    counts = count_ngrams(iter(TOKEN_SEQ), max_n=2)
    pruned = prune_counts(counts, min_count=3)
    assert pruned[1][(1,)] == 5
    assert pruned[2][(1, 2)] == 5


def test_pruning_count_3():
    counts = count_ngrams(iter(TOKEN_SEQ), max_n=3)
    pruned = prune_counts(counts, min_count=3)
    # trigram (1, 2, 3) has count 3 → should survive
    assert pruned[3][(1, 2, 3)] == 3


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic():
    c1 = count_ngrams(iter(TOKEN_SEQ), max_n=3)
    c2 = count_ngrams(iter(TOKEN_SEQ), max_n=3)
    for n in c1:
        assert c1[n] == c2[n], f"Counts are non-deterministic at order {n}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_token():
    counts = count_ngrams(iter([42]), max_n=4)
    assert counts[1][(42,)] == 1
    # No bigrams or higher
    assert len(counts[2]) == 0
    assert len(counts[3]) == 0


def test_empty_stream():
    counts = count_ngrams(iter([]), max_n=4)
    for n in range(1, 5):
        assert len(counts[n]) == 0


# ---------------------------------------------------------------------------
# NGramCounter class — char mode
# ---------------------------------------------------------------------------

from model.ngram_counter import NGramCounter  # noqa: E402


def test_char_mode_bigram_counts():
    """For 'abab', order 2: bigrams ('a','b') x2, ('b','a') x1."""
    counter = NGramCounter(max_order=2, mode="char")
    counts = counter.count("abab")
    # bigrams
    assert counts[2][('a', 'b')] == 2
    assert counts[2][('b', 'a')] == 1


def test_char_mode_trigram_counts():
    """For 'abab', order 3: trigrams ('a','b','a') x1, ('b','a','b') x1."""
    counter = NGramCounter(max_order=3, mode="char")
    counts = counter.count("abab")
    assert counts[3][('a', 'b', 'a')] == 1
    assert counts[3][('b', 'a', 'b')] == 1


def test_char_mode_includes_whitespace():
    """Whitespace must be kept as a token in char mode."""
    counter = NGramCounter(max_order=2, mode="char")
    counts = counter.count("a b")
    # Bigrams: ('a',' '), (' ','b')
    assert counts[2][('a', ' ')] == 1
    assert counts[2][(' ', 'b')] == 1


def test_char_mode_order5_supported():
    """Order 5 must be accepted for char mode."""
    counter = NGramCounter(max_order=5, mode="char")
    counts = counter.count("abcdefgh")
    assert 5 in counts
    assert len(counts[5]) > 0


def test_char_mode_invalid_input_raises():
    """Passing a list to char mode must raise TypeError."""
    counter = NGramCounter(max_order=2, mode="char")
    try:
        counter.count([1, 2, 3])  # type: ignore[arg-type]
        assert False, "Expected TypeError"
    except TypeError:
        pass


# ---------------------------------------------------------------------------
# NGramCounter class — word mode
# ---------------------------------------------------------------------------

def test_word_mode_bigram_counts():
    """For 'the cat sat', order 2: bigrams ('the','cat') x1, ('cat','sat') x1."""
    counter = NGramCounter(max_order=2, mode="word")
    counts = counter.count("the cat sat")
    assert counts[2][('the', 'cat')] == 1
    assert counts[2][('cat', 'sat')] == 1


def test_word_mode_trigram_counts():
    """For 'the cat sat', order 3: trigram ('the','cat','sat') x1."""
    counter = NGramCounter(max_order=3, mode="word")
    counts = counter.count("the cat sat")
    assert counts[3][('the', 'cat', 'sat')] == 1


def test_word_mode_strips_punctuation():
    """Boundary punctuation should be stripped via the Validator."""
    counter = NGramCounter(max_order=1, mode="word")
    counts = counter.count("Hello, world!")
    # 'hello' and 'world' should appear as unigrams (lowercased, stripped)
    assert counts[1][('hello',)] >= 1
    assert counts[1][('world',)] >= 1


def test_word_mode_invalid_input_raises():
    """Passing a list to word mode must raise TypeError."""
    counter = NGramCounter(max_order=2, mode="word")
    try:
        counter.count([1, 2])  # type: ignore[arg-type]
        assert False, "Expected TypeError"
    except TypeError:
        pass


# ---------------------------------------------------------------------------
# NGramCounter class — bpe mode
# ---------------------------------------------------------------------------

def test_bpe_mode_existing_behavior():
    """BPE mode should reproduce the legacy count_ngrams output."""
    counter = NGramCounter(max_order=3, mode="bpe")
    counts = counter.count(TOKEN_SEQ)
    # bigram (1,2) appears 5 times
    assert counts[2][(1, 2)] == 5
    # trigram (1,2,3) appears 3 times
    assert counts[3][(1, 2, 3)] == 3


def test_bpe_mode_invalid_input_raises():
    """Passing a str to bpe mode must raise TypeError."""
    counter = NGramCounter(max_order=2, mode="bpe")
    try:
        counter.count("hello")  # type: ignore[arg-type]
        assert False, "Expected TypeError"
    except TypeError:
        pass


def test_invalid_mode_raises():
    """An unknown mode string must raise ValueError at construction."""
    try:
        NGramCounter(max_order=2, mode="xyz")  # type: ignore[arg-type]
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_invalid_max_order_raises():
    """max_order < 1 must raise ValueError."""
    try:
        NGramCounter(max_order=0, mode="char")
        assert False, "Expected ValueError"
    except ValueError:
        pass
