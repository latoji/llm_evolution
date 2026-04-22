"""Ingest worker — multiprocessing orchestration, rollback, and pause.

The worker runs in a dedicated ``multiprocessing.Process`` so the FastAPI
event loop never blocks on CPU/GPU work. It owns its own DuckDB connection
(DuckDB does not allow cross-process connection sharing) and emits
JSON-serialisable progress events to the parent process via a
``multiprocessing.Queue``.

Per-chunk lifecycle
-------------------
1. Snapshot ``accuracy_before`` and a ``deepcopy`` of each neural trainer's
   ``state_dict``.
2. Insert a ``corpus_chunks`` row with ``status='processing'``.
3. Apply character / word / (optionally) BPE n-gram deltas, each inside a
   short ``store.transaction()`` so child processes launched by the Monte
   Carlo evaluator can read the newly-committed rows.
4. Run one forward+backward pass for the feedforward and transformer models
   (skipped if no BPE tokenizer is available).
5. Run ``MonteCarloEvaluator.evaluate_all`` and collect the 13 new accuracy
   scores (persisted as ``model_accuracy`` rows).
6. If any model's accuracy dropped below its pre-chunk snapshot by more than
   ``ACCURACY_EPSILON``, roll back every side effect of this chunk.
7. Mark the chunk as ``accepted`` or ``rejected``.

Pause semantics
---------------
``pause_event.is_set()`` is checked only between chunks. A pause never
interrupts a chunk mid-flight — that would leave the training state
partially updated and defeat the atomicity guarantee.
"""
from __future__ import annotations

import copy
import hashlib
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

from api import ingest_helpers
from api.ingest_helpers import (
    apply_ngram_deltas,
    chunk_text,
    clean_file_text,
    pre_screen_file,
    revert_deltas,
)
from api.worker_types import (
    ACCURACY_EPSILON,
    PROGRESS_QUEUE_MAXSIZE,
    WARMUP_CHUNKS,
    WorkerHandle,
    make_chunk_done,
    make_chunk_progress,
    make_chunk_start,
    make_ingest_complete,
    make_ingest_paused,
    safe_put,
)

_CHECKPOINT_DIR: Path = Path("model/checkpoints")


# ---------------------------------------------------------------------------
# Public process-management API (parent process)
# ---------------------------------------------------------------------------


def start_worker(
    file_paths: list[Path],
    db_path: Path = Path("db/llm_evolution.duckdb"),
) -> WorkerHandle:
    """Spawn a new ingest worker process and return a handle.

    Uses the ``spawn`` start method so the child gets a fresh Python
    interpreter — required for PyTorch on WSL2 and for clean DuckDB state.
    Does not block.
    """
    ctx = mp.get_context("spawn")
    progress_queue: Any = ctx.Queue(maxsize=PROGRESS_QUEUE_MAXSIZE)
    pause_event: Any = ctx.Event()

    process = ctx.Process(
        target=_worker_main,
        args=(list(file_paths), Path(db_path), progress_queue, pause_event),
        daemon=False,   # MC evaluator spawns its own child processes
    )
    process.start()

    return WorkerHandle(
        process=process,
        progress_queue=progress_queue,
        pause_event=pause_event,
        started_at=time.time(),
        file_paths=list(file_paths),
    )


def pause_worker(handle: WorkerHandle) -> None:
    """Ask the worker to exit after its current chunk finishes."""
    handle.pause_event.set()


def is_alive(handle: WorkerHandle) -> bool:
    """Return True if the worker process is still running."""
    return bool(handle.process.is_alive())


# ---------------------------------------------------------------------------
# Worker entry point (child process)
# ---------------------------------------------------------------------------


def _worker_main(
    file_paths: list[Path],
    db_path: Path,
    progress_queue: Any,
    pause_event: Any,
) -> None:
    """Run the ingest pipeline to completion, pause, or hard error.

    Imports for torch/duckdb are deferred so importing this module in the
    parent process stays cheap.
    """
    from db.store import Store
    from wvm.validator import Validator

    store = Store(db_path=db_path)
    validator = Validator()

    # Looked up via the helpers module so tests can monkey-patch them.
    merges = ingest_helpers.load_bpe_merges()
    # Trainers are built whenever torch is available, independently of whether
    # BPE merges exist.  The Markov MC evaluation must run regardless, and the
    # neural trainers will simply generate untrained output (score 0.0) until
    # BPE data is available and training steps have been performed.
    trainers = ingest_helpers.build_trainers(store)
    evaluator = (
        ingest_helpers.build_evaluator(store, validator, trainers)
        if trainers is not None
        else None
    )

    chunks_accepted = 0
    chunks_rejected = 0
    total_chunks_seen = 0

    try:
        for file_path in file_paths:
            cleaned = clean_file_text(file_path)
            if not pre_screen_file(cleaned, validator, file_path, progress_queue):
                continue

            chunks = chunk_text(cleaned)
            if not chunks:
                continue

            for chunk_index, chunk in enumerate(chunks):
                if pause_event.is_set():
                    safe_put(
                        progress_queue,
                        make_ingest_paused(
                            chunks_completed=total_chunks_seen,
                            chunks_remaining=len(chunks) - chunk_index,
                        ),
                    )
                    return

                safe_put(
                    progress_queue,
                    make_chunk_start(
                        chunk_index=chunk_index,
                        total_chunks=len(chunks),
                        operation=f"Processing chunk {chunk_index + 1}/{len(chunks)}",
                    ),
                )

                status, reason, delta = _process_chunk(
                    chunk_index=chunk_index,
                    chunk=chunk,
                    filename=file_path.name,
                    store=store,
                    validator=validator,
                    merges=merges,
                    trainers=trainers,
                    evaluator=evaluator,
                    progress_queue=progress_queue,
                    chunks_accepted=chunks_accepted,
                )
                if status == "accepted":
                    chunks_accepted += 1
                else:
                    chunks_rejected += 1
                total_chunks_seen += 1

                safe_put(
                    progress_queue,
                    make_chunk_done(
                        chunk_index=chunk_index,
                        status=status,
                        accuracy_delta=delta,
                        reason=reason,
                    ),
                )

        safe_put(
            progress_queue,
            make_ingest_complete(
                chunks_accepted=chunks_accepted,
                chunks_rejected=chunks_rejected,
            ),
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------


def _process_chunk(
    *,
    chunk_index: int,
    chunk: str,
    filename: str,
    store: Any,
    validator: Any,
    merges: list[tuple[str, str]] | None,
    trainers: dict[str, Any] | None,
    evaluator: Any,
    progress_queue: Any,
    chunks_accepted: int = 0,
) -> tuple[str, str | None, dict[str, float]]:
    """Process one chunk. Return (status, reason, per-model accuracy delta).

    Rollback is skipped during the warm-up period (``chunks_accepted <
    WARMUP_CHUNKS``) because MC accuracy estimates are too noisy to trust
    until enough data has been seen.
    """
    accuracy_before: dict[str, float] = store.get_latest_accuracy()

    nn_snapshots: dict[str, dict] = {}
    if trainers is not None:
        for name, trainer in trainers.items():
            nn_snapshots[name] = copy.deepcopy(trainer.model.state_dict())

    text_hash = hashlib.sha256(chunk.encode("utf-8", errors="replace")).hexdigest()
    chunk_id = store.insert_chunk(
        filename=filename,
        chunk_index=chunk_index,
        text_hash=text_hash,
        char_count=len(chunk),
        accuracy_before=accuracy_before,
    )

    safe_put(progress_queue, make_chunk_progress("Counting char n-grams", 10))
    char_deltas = apply_ngram_deltas(store, "char", list(chunk), max_order=5)

    safe_put(progress_queue, make_chunk_progress("Counting word n-grams", 25))
    word_tokens = validator.tokenize(chunk)
    word_deltas = apply_ngram_deltas(store, "word", word_tokens, max_order=3)

    bpe_deltas: dict[int, dict[tuple[str, str], int]] = {}
    token_ids: list[int] = []
    if merges is not None:
        safe_put(progress_queue, make_chunk_progress("Counting BPE n-grams", 40))
        from tokenizer.bpe import encode as bpe_encode
        token_ids = bpe_encode(chunk, merges)
        if token_ids:
            bpe_tokens = [str(t) for t in token_ids]
            bpe_deltas = apply_ngram_deltas(store, "bpe", bpe_tokens, max_order=3)

    if trainers is not None and token_ids:
        safe_put(progress_queue, make_chunk_progress("Training feedforward", 55))
        try:
            trainers["feedforward"].train_step(token_ids, chunk_id)
        except Exception:
            pass  # training errors are non-fatal; rollback cleans up artefacts
        safe_put(progress_queue, make_chunk_progress("Training transformer", 70))
        try:
            trainers["transformer"].train_step(token_ids, chunk_id)
        except Exception:
            pass

    safe_put(progress_queue, make_chunk_progress("Running Monte Carlo", 85))
    if evaluator is not None:
        def _on_mc(event_type: str, payload: dict) -> None:
            safe_put(progress_queue, {"type": event_type, **payload})
        accuracy_after = evaluator.evaluate_all(chunk_id=chunk_id, on_progress=_on_mc)
    else:
        accuracy_after = {}

    drops = {
        name: accuracy_before[name] - accuracy_after[name]
        for name in accuracy_after
        if name in accuracy_before
        and accuracy_after[name] < accuracy_before[name] - ACCURACY_EPSILON
    }
    accuracy_delta = {
        name: accuracy_after.get(name, 0.0) - accuracy_before.get(name, 0.0)
        for name in accuracy_after
    }

    if drops and chunks_accepted >= WARMUP_CHUNKS:
        _rollback_chunk(
            store=store,
            chunk_id=chunk_id,
            char_deltas=char_deltas,
            word_deltas=word_deltas,
            bpe_deltas=bpe_deltas,
            trainers=trainers,
            nn_snapshots=nn_snapshots,
        )
        worst = ", ".join(f"{k} (-{v:.3f})" for k, v in sorted(drops.items()))
        reason = f"accuracy dropped: {worst}"
        store.mark_chunk_rejected(chunk_id, reason)
        return "rejected", reason, accuracy_delta

    store.mark_chunk_accepted(chunk_id, accuracy_after)
    return "accepted", None, accuracy_delta


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def _rollback_chunk(
    *,
    store: Any,
    chunk_id: int,
    char_deltas: dict[int, dict[tuple[str, str], int]],
    word_deltas: dict[int, dict[tuple[str, str], int]],
    bpe_deltas: dict[int, dict[tuple[str, str], int]],
    trainers: dict[str, Any] | None,
    nn_snapshots: dict[str, dict],
) -> None:
    """Undo every side effect produced during processing of this chunk."""
    revert_deltas(store, "char", char_deltas)
    revert_deltas(store, "word", word_deltas)
    revert_deltas(store, "bpe", bpe_deltas)

    with store.transaction() as conn:
        conn.execute("DELETE FROM model_accuracy WHERE chunk_id = ?", [chunk_id])
        conn.execute("DELETE FROM nn_checkpoints WHERE chunk_id = ?", [chunk_id])

    if trainers is not None:
        for name, trainer in trainers.items():
            snapshot = nn_snapshots.get(name)
            if snapshot is not None:
                trainer.model.load_state_dict(snapshot)

    for prefix in ("feedforward", "transformer"):
        pt_file = _CHECKPOINT_DIR / f"{prefix}_{chunk_id}.pt"
        if pt_file.exists():
            pt_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test hook
# ---------------------------------------------------------------------------


def run_worker_inline(
    file_paths: list[Path],
    db_path: Path,
    progress_queue: Any,
    pause_event: Any,
) -> None:
    """Invoke :func:`_worker_main` synchronously in the current process.

    Intended for unit tests that need deterministic, in-process behaviour
    (e.g. to monkey-patch :class:`MonteCarloEvaluator`).  In production the
    worker is always spawned via :func:`start_worker`.
    """
    _worker_main(list(file_paths), Path(db_path), progress_queue, pause_event)
