"""Tests for db/migrate_from_json.py.

Creates temporary directories with synthetic JSON count files and verifies
that migrate() correctly loads rows into DuckDB and is idempotent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from db.store import Store
from db.migrate_from_json import migrate


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mem_store(tmp_path):
    s = Store(tmp_path / "test.duckdb")
    yield s
    s.close()


def _write_json_counts(counts_dir: Path, filename: str, data: dict[str, int]) -> None:
    """Write a synthetic JSON count file to *counts_dir*."""
    counts_dir.mkdir(parents=True, exist_ok=True)
    (counts_dir / filename).write_text(
        json.dumps(data, separators=(",", ":")), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_migrate_unigrams_to_db(tmp_path, mem_store):
    """Unigrams in JSON land in token_ngrams table as n=1 rows."""
    counts_dir = tmp_path / "counts"
    _write_json_counts(counts_dir, "unigrams.json", {"1": 5, "2": 3})

    rows_migrated = migrate(store=mem_store, counts_dir=counts_dir)
    assert rows_migrated == 2

    dist = mem_store.get_distribution("bpe", 1, "")
    assert dist.get("1", 0) == 5
    assert dist.get("2", 0) == 3


def test_migrate_bigrams_to_db(tmp_path, mem_store):
    """Bigrams are parsed as context='1', next_token='2'."""
    counts_dir = tmp_path / "counts"
    _write_json_counts(counts_dir, "bigrams.json", {"1 2": 4, "2 3": 2})

    migrate(store=mem_store, counts_dir=counts_dir)

    dist = mem_store.get_distribution("bpe", 2, "1")
    assert dist.get("2", 0) == 4

    dist2 = mem_store.get_distribution("bpe", 2, "2")
    assert dist2.get("3", 0) == 2


def test_migrate_trigrams_to_db(tmp_path, mem_store):
    """Trigrams use 'a b' as context and 'c' as next_token."""
    counts_dir = tmp_path / "counts"
    _write_json_counts(counts_dir, "trigrams.json", {"1 2 3": 7})

    migrate(store=mem_store, counts_dir=counts_dir)

    dist = mem_store.get_distribution("bpe", 3, "1 2")
    assert dist.get("3", 0) == 7


def test_migrate_returns_row_count(tmp_path, mem_store):
    counts_dir = tmp_path / "counts"
    _write_json_counts(counts_dir, "unigrams.json", {"1": 5, "2": 3, "3": 1})
    _write_json_counts(counts_dir, "bigrams.json", {"1 2": 4})

    result = migrate(store=mem_store, counts_dir=counts_dir)
    assert result == 4  # 3 unigrams + 1 bigram


def test_migrate_moves_files_to_archive(tmp_path, mem_store):
    counts_dir = tmp_path / "counts"
    _write_json_counts(counts_dir, "unigrams.json", {"1": 2})

    migrate(store=mem_store, counts_dir=counts_dir)

    archive = counts_dir / "archive"
    assert archive.exists()
    assert (archive / "unigrams.json").exists()
    assert not (counts_dir / "unigrams.json").exists()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_migrate_second_call_is_noop(tmp_path, mem_store):
    """Running migrate() a second time (archive exists) returns 0."""
    counts_dir = tmp_path / "counts"
    _write_json_counts(counts_dir, "unigrams.json", {"1": 5})

    first = migrate(store=mem_store, counts_dir=counts_dir)
    assert first == 1

    # archive now exists — second call should be a no-op
    second = migrate(store=mem_store, counts_dir=counts_dir)
    assert second == 0


def test_migrate_db_unchanged_on_second_call(tmp_path, mem_store):
    """DB counts must not double after a second migrate() call."""
    counts_dir = tmp_path / "counts"
    _write_json_counts(counts_dir, "unigrams.json", {"1": 5})

    migrate(store=mem_store, counts_dir=counts_dir)
    migrate(store=mem_store, counts_dir=counts_dir)  # no-op

    dist = mem_store.get_distribution("bpe", 1, "")
    assert dist.get("1", 0) == 5  # still 5, not 10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_migrate_empty_dir_returns_zero(tmp_path, mem_store):
    """An empty counts dir (no JSON files) returns 0 and creates no archive."""
    counts_dir = tmp_path / "counts"
    counts_dir.mkdir()

    result = migrate(store=mem_store, counts_dir=counts_dir)
    assert result == 0
    assert not (counts_dir / "archive").exists()


def test_migrate_partial_files(tmp_path, mem_store):
    """Only present JSON files are migrated; missing orders are skipped."""
    counts_dir = tmp_path / "counts"
    _write_json_counts(counts_dir, "bigrams.json", {"1 2": 3})
    # No unigrams.json

    result = migrate(store=mem_store, counts_dir=counts_dir)
    assert result == 1  # only bigrams

    dist = mem_store.get_distribution("bpe", 2, "1")
    assert dist.get("2", 0) == 3
