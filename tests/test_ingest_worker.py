"""Integration tests for :mod:`api.ingest_worker`.

Fast tests drive :func:`run_worker_inline` in the current process with
monkey-patched factories so they never touch torch, the BPE tokenizer, or a
real MC process pool. Slow tests exercise :func:`start_worker` end-to-end via
real ``multiprocessing`` primitives (opt in with ``pytest -m slow``).

Pure helper coverage (chunking, pre-screening, delta round-trip, event
constructors) lives in ``test_ingest_helpers.py``.
"""
from __future__ import annotations

import queue as _queue_module
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from api import ingest_helpers, ingest_worker
from api.worker_types import (
    EVENT_CHUNK_DONE,
    EVENT_CHUNK_START,
    EVENT_FILE_REJECTED,
    EVENT_INGEST_COMPLETE,
    EVENT_INGEST_PAUSED,
)


# ---------------------------------------------------------------------------
# Fixtures / test doubles
# ---------------------------------------------------------------------------


class _InlineQueue:
    """A ``queue.Queue``-backed stand-in for ``multiprocessing.Queue``."""

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


class _FakeEvaluator:
    """Writes canned accuracy rows — replaces :class:`MonteCarloEvaluator` in tests."""

    def __init__(self, store: Any, results: dict[str, float]) -> None:
        self.store = store
        self.results = results

    def evaluate_all(
        self, chunk_id: int, on_progress: Any = None
    ) -> dict[str, float]:
        for name, acc in self.results.items():
            self.store.insert_accuracy(name, chunk_id, acc, None)
        if on_progress is not None:
            for name, acc in self.results.items():
                on_progress("mc_model_start", {"model": name})
                on_progress(
                    "mc_complete",
                    {"model": name, "accuracy": acc, "run": 0},
                )
        return dict(self.results)


def _make_fake_trainers() -> dict[str, Any]:
    """Build trainers whose ``model.state_dict`` is deepcopy-safe."""
    trainers: dict[str, Any] = {}
    for name in ("feedforward", "transformer"):
        trainer = MagicMock()
        trainer.model.state_dict = MagicMock(return_value={"w": 1.0})
        trainer.model.load_state_dict = MagicMock()
        trainer.train_step = MagicMock(return_value=0.5)
        trainers[name] = trainer
    return trainers


def _patch_factories(
    monkeypatch: pytest.MonkeyPatch,
    *,
    merges: list[tuple[str, str]] | None = None,
    trainers: dict[str, Any] | None = None,
    evaluator_results: dict[str, float] | None = None,
) -> None:
    """Install fakes for every heavyweight factory in :mod:`api.ingest_helpers`."""
    monkeypatch.setattr(ingest_helpers, "load_bpe_merges", lambda: merges)
    monkeypatch.setattr(ingest_helpers, "build_trainers", lambda store: trainers)

    def _fake_build_evaluator(store: Any, validator: Any, trainers_arg: Any) -> Any:
        return _FakeEvaluator(store, evaluator_results or {})

    monkeypatch.setattr(ingest_helpers, "build_evaluator", _fake_build_evaluator)


def _sample_corpus(n_chunks: int) -> str:
    """Return a clean English paragraph-blob long enough to split into *n_chunks*."""
    para = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump! "
        "Sphinx of black quartz judge my vow. "
        "Bright vixens jump dozy fowl quack. "
    ) * 10   # ~1600 chars per para
    return ("\n\n".join([para] * n_chunks)).strip()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_accepts_clean_corpus_and_emits_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_factories(monkeypatch)   # no BPE, no NN, empty evaluator

        corpus = tmp_path / "corpus.txt"
        corpus.write_text(_sample_corpus(n_chunks=2))
        db_path = tmp_path / "llm.duckdb"
        q = _InlineQueue()

        ingest_worker.run_worker_inline([corpus], db_path, q, threading.Event())

        events = q.drain()
        types = [e["type"] for e in events]

        assert EVENT_INGEST_COMPLETE in types
        assert types.count(EVENT_CHUNK_START) >= 1
        assert types.count(EVENT_CHUNK_DONE) == types.count(EVENT_CHUNK_START)

        done_events = [e for e in events if e["type"] == EVENT_CHUNK_DONE]
        assert all(e["status"] == "accepted" for e in done_events)

        complete = next(e for e in events if e["type"] == EVENT_INGEST_COMPLETE)
        assert complete["chunks_accepted"] == len(done_events)
        assert complete["chunks_rejected"] == 0

    def test_ngram_rows_populated_after_accept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_factories(monkeypatch)
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(_sample_corpus(n_chunks=1))
        db_path = tmp_path / "llm.duckdb"

        ingest_worker.run_worker_inline(
            [corpus], db_path, _InlineQueue(), threading.Event()
        )

        from db.store import Store
        store = Store(db_path=db_path)
        try:
            char_rows = store._conn.execute(
                "SELECT COUNT(*) FROM char_ngrams"
            ).fetchone()[0]
            word_rows = store._conn.execute(
                "SELECT COUNT(*) FROM word_ngrams"
            ).fetchone()[0]
            assert char_rows > 0
            assert word_rows > 0
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def _pre_seed_accuracy(db_path: Path, accuracy: dict[str, float]) -> None:
    """Seed one accepted chunk whose accuracy rows become ``accuracy_before``."""
    from db.store import Store

    store = Store(db_path=db_path)
    try:
        chunk_id = store.insert_chunk(
            filename="seed.txt",
            chunk_index=0,
            text_hash="seedhash",
            char_count=10,
            accuracy_before={},
        )
        for name, acc in accuracy.items():
            store.insert_accuracy(name, chunk_id, acc, None)
        store.mark_chunk_accepted(chunk_id, accuracy)
    finally:
        store.close()


class TestRollback:
    def test_undoes_ngrams_and_restores_nn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "llm.duckdb"
        baseline = {"feedforward": 0.8, "transformer": 0.8, "char_1gram": 0.8}
        _pre_seed_accuracy(db_path, baseline)

        from db.store import Store

        store0 = Store(db_path=db_path)
        try:
            pre_char = store0._conn.execute(
                "SELECT COUNT(*) FROM char_ngrams"
            ).fetchone()[0]
            pre_word = store0._conn.execute(
                "SELECT COUNT(*) FROM word_ngrams"
            ).fetchone()[0]
            pre_accuracy = store0._conn.execute(
                "SELECT COUNT(*) FROM model_accuracy"
            ).fetchone()[0]
        finally:
            store0.close()

        fake_trainers = _make_fake_trainers()
        _patch_factories(
            monkeypatch,
            merges=[],
            trainers=fake_trainers,
            evaluator_results={k: 0.1 for k in baseline},  # catastrophic drop
        )
        # Bypass warm-up so the rollback guard is active from chunk 0.
        monkeypatch.setattr(ingest_worker, "WARMUP_CHUNKS", 0)
        # With merges=[] the BPE encoder still runs; neutralise by returning
        # an empty token list so the worker skips NN training and BPE counts.
        import tokenizer.bpe
        monkeypatch.setattr(tokenizer.bpe, "encode", lambda text, merges: [])

        corpus = tmp_path / "corpus.txt"
        corpus.write_text(_sample_corpus(n_chunks=1))

        ingest_worker.run_worker_inline(
            [corpus], db_path, _InlineQueue(), threading.Event()
        )

        store1 = Store(db_path=db_path)
        try:
            post_char = store1._conn.execute(
                "SELECT COUNT(*) FROM char_ngrams"
            ).fetchone()[0]
            post_word = store1._conn.execute(
                "SELECT COUNT(*) FROM word_ngrams"
            ).fetchone()[0]
            post_accuracy = store1._conn.execute(
                "SELECT COUNT(*) FROM model_accuracy"
            ).fetchone()[0]
            rejected_rows = store1._conn.execute(
                "SELECT COUNT(*) FROM corpus_chunks WHERE status = 'rejected'"
            ).fetchone()[0]
        finally:
            store1.close()

        assert post_char == pre_char, "char_ngrams row count changed after rollback"
        assert post_word == pre_word, "word_ngrams row count changed after rollback"
        assert post_accuracy == pre_accuracy, "model_accuracy row count changed after rollback"
        assert rejected_rows >= 1, "chunk should be marked rejected"

        for trainer in fake_trainers.values():
            trainer.model.load_state_dict.assert_called()


# ---------------------------------------------------------------------------
# Pause
# ---------------------------------------------------------------------------


class TestPause:
    def test_pause_before_any_chunk_emits_paused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_factories(monkeypatch)
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(_sample_corpus(n_chunks=3))

        q = _InlineQueue()
        pause = threading.Event()
        pause.set()  # pause *before* processing starts

        ingest_worker.run_worker_inline(
            [corpus], tmp_path / "llm.duckdb", q, pause
        )

        types = [e["type"] for e in q.drain()]
        assert EVENT_INGEST_PAUSED in types
        assert EVENT_INGEST_COMPLETE not in types
        assert EVENT_CHUNK_DONE not in types


# ---------------------------------------------------------------------------
# Pre-screening
# ---------------------------------------------------------------------------


class TestFilePreScreen:
    def test_binary_noise_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_factories(monkeypatch)
        bad = tmp_path / "noise.txt"
        bad.write_text("xq zj kw pvb mnrf lkh " * 500)

        q = _InlineQueue()
        ingest_worker.run_worker_inline(
            [bad], tmp_path / "llm.duckdb", q, threading.Event()
        )

        types = [e["type"] for e in q.drain()]
        assert EVENT_FILE_REJECTED in types
        assert EVENT_CHUNK_DONE not in types
        assert EVENT_INGEST_COMPLETE in types


# ---------------------------------------------------------------------------
# Event count (queue does not drop events)
# ---------------------------------------------------------------------------


class TestEventCount:
    def test_three_chunks_emit_expected_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_factories(monkeypatch)
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(_sample_corpus(n_chunks=3))

        q = _InlineQueue()
        ingest_worker.run_worker_inline(
            [corpus], tmp_path / "llm.duckdb", q, threading.Event()
        )

        types = [e["type"] for e in q.drain()]
        starts = [t for t in types if t == EVENT_CHUNK_START]
        dones = [t for t in types if t == EVENT_CHUNK_DONE]
        completes = [t for t in types if t == EVENT_INGEST_COMPLETE]

        assert len(starts) >= 3
        assert len(dones) >= 3
        assert len(completes) == 1


# ---------------------------------------------------------------------------
# Slow — real multiprocessing isolation
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestProcessIsolation:
    def test_force_kill_worker_does_not_crash_parent(self, tmp_path: Path) -> None:
        from api.ingest_worker import is_alive, start_worker

        corpus = tmp_path / "corpus.txt"
        corpus.write_text(_sample_corpus(n_chunks=2))

        handle = start_worker([corpus], db_path=tmp_path / "llm.duckdb")
        try:
            time.sleep(1.0)     # let the child boot
            if is_alive(handle):
                handle.process.kill()
                handle.process.join(timeout=5)
            assert not is_alive(handle)
        finally:
            if is_alive(handle):
                handle.process.kill()
                handle.process.join(timeout=5)

        # Parent can still open the DB file — no corruption / zombie lock.
        from db.store import Store
        Store(db_path=tmp_path / "llm.duckdb").close()
