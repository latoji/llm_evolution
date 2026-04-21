"""Pure-function tests for :mod:`api.ingest_helpers` and :mod:`api.worker_types`.

These tests never spawn a process or open a neural checkpoint — they cover
chunking, pre-screening, n-gram delta round-trips, event constructors, and
queue safety in isolation. The worker-integration tests live in
``test_ingest_worker.py``.
"""
from __future__ import annotations

import queue as _queue_module
from pathlib import Path
from typing import Any

import pytest

from api.ingest_helpers import (
    apply_ngram_deltas,
    chunk_text,
    clean_file_text,
    pre_screen_file,
    revert_deltas,
)
from api.worker_types import (
    EVENT_CHUNK_DONE,
    EVENT_CHUNK_START,
    EVENT_FILE_REJECTED,
    EVENT_INGEST_COMPLETE,
    make_chunk_done,
    make_chunk_progress,
    make_chunk_start,
    make_file_rejected,
    make_ingest_complete,
    make_ingest_paused,
    safe_put,
)


# ---------------------------------------------------------------------------
# Inline queue test double (shared with test_ingest_worker)
# ---------------------------------------------------------------------------


class _InlineQueue:
    """Thin wrapper over ``queue.Queue`` mimicking the ``mp.Queue`` API."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._q: _queue_module.Queue = _queue_module.Queue(maxsize=maxsize)

    def put_nowait(self, item: Any) -> None:
        self._q.put_nowait(item)

    def drain(self) -> list[Any]:
        events: list[Any] = []
        try:
            while True:
                events.append(self._q.get_nowait())
        except _queue_module.Empty:
            return events


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_short_text_single_chunk(self) -> None:
        text = "a" * 500
        assert chunk_text(text, min_size=1000, max_size=2500) == [text]

    def test_empty_text_returns_empty_list(self) -> None:
        assert chunk_text("") == []

    def test_respects_max_size(self) -> None:
        text = "word " * 1000  # 5000 chars
        chunks = chunk_text(text, min_size=1000, max_size=2500)
        assert all(len(c) <= 2500 for c in chunks)
        # Reassembly must not create new content.
        assert sum(len(c) for c in chunks) <= len(text)

    def test_prefers_paragraph_break(self) -> None:
        para = ("word " * 300).strip()  # ~1499 chars
        text = para + "\n\n" + para + "\n\n" + para
        chunks = chunk_text(text, min_size=1000, max_size=2500)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert not chunk.startswith(" ")
            assert not chunk.endswith(" ")

    def test_never_splits_mid_word(self) -> None:
        text = ("the " * 2000).strip()
        for chunk in chunk_text(text):
            first = chunk.split(" ", 1)[0]
            last = chunk.rsplit(" ", 1)[-1]
            assert first == "the"
            assert last == "the"


# ---------------------------------------------------------------------------
# safe_put
# ---------------------------------------------------------------------------


class TestSafePut:
    def test_enqueue_until_full_then_drop(self) -> None:
        q = _InlineQueue(maxsize=2)
        assert safe_put(q, {"type": "a"}) is True
        assert safe_put(q, {"type": "b"}) is True
        # Third event exceeds cap and is silently dropped.
        assert safe_put(q, {"type": "c"}) is False


# ---------------------------------------------------------------------------
# Pre-screening
# ---------------------------------------------------------------------------


class TestPreScreenFile:
    def test_rejects_empty_text(self, tmp_path: Path) -> None:
        from wvm.validator import Validator

        q = _InlineQueue()
        path = tmp_path / "empty.txt"
        path.write_text("")
        assert pre_screen_file("", Validator(), path, q) is False
        events = q.drain()
        assert len(events) == 1 and events[0]["type"] == EVENT_FILE_REJECTED

    def test_rejects_nonsense(self, tmp_path: Path) -> None:
        from wvm.validator import Validator

        q = _InlineQueue()
        path = tmp_path / "noise.txt"
        nonsense = " ".join(["zqxvj"] * 200)
        path.write_text(nonsense)
        assert pre_screen_file(nonsense, Validator(), path, q) is False
        events = q.drain()
        assert events[0]["type"] == EVENT_FILE_REJECTED
        assert "low real-word" in events[0]["reason"]

    def test_accepts_real_english(self, tmp_path: Path) -> None:
        from wvm.validator import Validator

        q = _InlineQueue()
        path = tmp_path / "clean.txt"
        clean = "The cat sat on the mat. The dog sat on the mat."
        path.write_text(clean)
        assert pre_screen_file(clean, Validator(), path, q) is True
        assert q.drain() == []


# ---------------------------------------------------------------------------
# N-gram delta round-trip
# ---------------------------------------------------------------------------


class TestNGramDeltas:
    def _store(self, tmp_path: Path) -> Any:
        from db.store import Store

        return Store(db_path=tmp_path / "test.duckdb")

    def test_apply_then_revert_leaves_table_empty(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        try:
            text = "abcdefghij" * 50
            deltas = apply_ngram_deltas(store, "char", list(text), max_order=3)
            rows_after_apply = store._conn.execute(
                "SELECT COUNT(*) FROM char_ngrams"
            ).fetchone()[0]
            assert rows_after_apply > 0

            revert_deltas(store, "char", deltas)
            rows_after_revert = store._conn.execute(
                "SELECT COUNT(*) FROM char_ngrams"
            ).fetchone()[0]
            assert rows_after_revert == 0
        finally:
            store.close()

    def test_revert_keeps_pre_existing_rows(self, tmp_path: Path) -> None:
        """Rolling back one chunk must leave earlier chunks' rows intact."""
        store = self._store(tmp_path)
        try:
            apply_ngram_deltas(store, "word", ["the", "cat", "sat"], max_order=2)
            baseline = store._conn.execute(
                "SELECT COUNT(*) FROM word_ngrams"
            ).fetchone()[0]
            assert baseline > 0

            deltas = apply_ngram_deltas(
                store, "word", ["a", "b", "c", "d"], max_order=2
            )
            revert_deltas(store, "word", deltas)

            after_revert = store._conn.execute(
                "SELECT COUNT(*) FROM word_ngrams"
            ).fetchone()[0]
            assert after_revert == baseline
        finally:
            store.close()

    def test_empty_tokens_produces_no_deltas(self, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        try:
            assert apply_ngram_deltas(store, "char", [], max_order=3) == {}
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Event constructors
# ---------------------------------------------------------------------------


class TestEventConstructors:
    def test_chunk_start_payload(self) -> None:
        assert make_chunk_start(2, 5, "Doing stuff") == {
            "type": EVENT_CHUNK_START,
            "chunk_index": 2,
            "total_chunks": 5,
            "operation": "Doing stuff",
        }

    def test_chunk_progress_clamps_pct(self) -> None:
        assert make_chunk_progress("X", 150)["pct"] == 100
        assert make_chunk_progress("X", -5)["pct"] == 0

    def test_chunk_done_default_reason_none(self) -> None:
        evt = make_chunk_done(1, "accepted", {"m": 0.1})
        assert evt["type"] == EVENT_CHUNK_DONE
        assert evt["reason"] is None

    def test_ingest_complete_counts(self) -> None:
        evt = make_ingest_complete(3, 1)
        assert evt["type"] == EVENT_INGEST_COMPLETE
        assert evt["chunks_accepted"] == 3 and evt["chunks_rejected"] == 1

    def test_ingest_paused_counts(self) -> None:
        evt = make_ingest_paused(2, 5)
        assert evt["chunks_completed"] == 2 and evt["chunks_remaining"] == 5

    def test_file_rejected_carries_reason(self) -> None:
        evt = make_file_rejected("foo.txt", "bad")
        assert evt["filename"] == "foo.txt" and evt["reason"] == "bad"


# ---------------------------------------------------------------------------
# clean_file_text
# ---------------------------------------------------------------------------


class TestCleanFileText:
    def test_returns_cleaned_paragraphs(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.txt"
        path.write_text(
            "First paragraph with enough content to survive cleaning.\n\n"
            "Second paragraph also long enough to be kept.\n\n"
            "x"  # below MIN_LEN; filtered out
        )
        cleaned = clean_file_text(path)
        assert "First paragraph" in cleaned
        assert "Second paragraph" in cleaned
        assert "\n\n" in cleaned

    def test_missing_file_returns_empty_string(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.txt"
        assert clean_file_text(missing) == ""


# Prevent pytest from flagging `pytest` as unused on re-imports.
_unused = pytest
