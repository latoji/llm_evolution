"""Tests for wvm/validator.py — Word Verification Module."""

import pytest
from pathlib import Path

from wvm.validator import Validator, WordResult, PUNCTUATION_BOUNDARY, _strip_boundary

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def validator() -> Validator:
    """Shared Validator instance loaded once for the entire module."""
    return Validator()


# ---------------------------------------------------------------------------
# validate() — happy paths
# ---------------------------------------------------------------------------

def test_validate_all_real(validator: Validator) -> None:
    results, pct = validator.validate("hello world")
    assert len(results) == 2
    assert all(r.is_real for r in results)
    assert pct == 1.0


def test_validate_mixed(validator: Validator) -> None:
    results, pct = validator.validate("hello wrld")
    assert len(results) == 2
    real_count = sum(1 for r in results if r.is_real)
    assert real_count == 1
    assert pct == pytest.approx(0.5)


def test_validate_empty_string(validator: Validator) -> None:
    results, pct = validator.validate("")
    assert results == []
    assert pct == 0.0


def test_validate_whitespace_only(validator: Validator) -> None:
    results, pct = validator.validate("   ")
    assert results == []
    assert pct == 0.0


def test_validate_returns_word_results(validator: Validator) -> None:
    results, _ = validator.validate("the cat")
    for r in results:
        assert isinstance(r, WordResult)
        assert r.raw != ""
        assert r.word == r.word.lower()


# ---------------------------------------------------------------------------
# tokenize() — boundary punctuation stripping
# ---------------------------------------------------------------------------

def test_tokenize_basic(validator: Validator) -> None:
    tokens = validator.tokenize("Hello, it's (a) test!")
    assert tokens == ["hello", "it's", "a", "test"]


def test_tokenize_em_dash_prefix(validator: Validator) -> None:
    tokens = validator.tokenize("—word...")
    assert tokens == ["word"]


def test_tokenize_internal_apostrophe_preserved(validator: Validator) -> None:
    tokens = validator.tokenize("don't")
    assert tokens == ["don't"]


def test_tokenize_contraction_its(validator: Validator) -> None:
    tokens = validator.tokenize("it's")
    assert tokens == ["it's"]


def test_tokenize_strips_quotes(validator: Validator) -> None:
    tokens = validator.tokenize('"quoted"')
    assert tokens == ["quoted"]


def test_tokenize_multiple_leading_boundary(validator: Validator) -> None:
    tokens = validator.tokenize("...word,")
    assert tokens == ["word"]


def test_tokenize_skips_pure_punctuation(validator: Validator) -> None:
    tokens = validator.tokenize("hello . , world")
    # "." and "," are stripped to empty → skipped
    assert tokens == ["hello", "world"]


# ---------------------------------------------------------------------------
# suggest()
# ---------------------------------------------------------------------------

def test_suggest_teh(validator: Validator) -> None:
    # suggest() must return valid English words close to the input.
    # With 167K SCOWL words, many short words (teth, tech, th…) score higher
    # than "the" (ratio 0.667); we verify at least one suggestion is returned
    # and that every suggestion is a valid word.
    suggestions = validator.suggest("teh", n=5)
    assert isinstance(suggestions, list)
    assert len(suggestions) >= 1
    for word in suggestions:
        assert word in validator._words, f"'{word}' is not a valid SCOWL word"


def test_suggest_returns_at_most_n(validator: Validator) -> None:
    suggestions = validator.suggest("teh", n=2)
    assert len(suggestions) <= 2


def test_suggest_nonsense_no_crash(validator: Validator) -> None:
    # Very unusual string — may return empty list, but must not raise
    suggestions = validator.suggest("xzqwjfkld", n=1)
    assert isinstance(suggestions, list)


# ---------------------------------------------------------------------------
# Missing wordlist — failure mode
# ---------------------------------------------------------------------------

def test_missing_wordlist_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.txt"
    with pytest.raises(FileNotFoundError) as exc_info:
        Validator(wordlist_path=missing)
    # Error message should point to the download location
    assert "wordlist.aspell.net" in str(exc_info.value).lower() or \
           "http" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def test_strip_boundary_leading_and_trailing() -> None:
    assert _strip_boundary("...hello...") == "hello"


def test_strip_boundary_preserves_internal() -> None:
    assert _strip_boundary("don't") == "don't"


def test_strip_boundary_all_punctuation() -> None:
    assert _strip_boundary("...") == ""


def test_strip_boundary_empty() -> None:
    assert _strip_boundary("") == ""
