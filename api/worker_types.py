"""Data types and helpers shared between the ingest worker and the FastAPI parent process.

Kept intentionally small — only the pieces that must be importable from both
processes without dragging heavyweight imports (torch, duckdb, etc.) into the
parent's startup path.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as _queue_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Progress queue cap: prefer dropping progress events over blocking the worker.
PROGRESS_QUEUE_MAXSIZE: int = 1000

# Default chunk sizing (see SPEC.md §17).
CHUNK_MIN_CHARS: int = 1000
CHUNK_MAX_CHARS: int = 2500

# Minimum real-word percentage for a file to be considered valid English.
MIN_REAL_WORD_PCT: float = 0.70

# Tolerance used when comparing before/after accuracy snapshots.
# Monte Carlo evaluation is stochastic; a threshold of 1e-6 causes excessive
# rejections because sampling noise alone routinely produces tiny apparent drops.
# 0.02 = 2 percentage-point drop required before a chunk is rolled back.
ACCURACY_EPSILON: float = 0.02

# Number of accepted chunks to treat as a warm-up period.  During warm-up the
# models have too little data for MC accuracy estimates to be stable, so the
# rollback guard is skipped entirely.
WARMUP_CHUNKS: int = 5

# WebSocket event types that originate in the worker.
EVENT_CHUNK_START: str = "chunk_start"
EVENT_CHUNK_PROGRESS: str = "chunk_progress"
EVENT_CHUNK_DONE: str = "chunk_done"
EVENT_INGEST_COMPLETE: str = "ingest_complete"
EVENT_INGEST_PAUSED: str = "ingest_paused"
EVENT_FILE_REJECTED: str = "file_rejected"
EVENT_MC_MODEL_START: str = "mc_model_start"
EVENT_MC_COMPLETE: str = "mc_complete"


# ---------------------------------------------------------------------------
# Worker handle (parent-side)
# ---------------------------------------------------------------------------


@dataclass
class WorkerHandle:
    """Parent-process handle for a running IngestWorker.

    Attributes:
        process:        The multiprocessing.Process executing ``_worker_main``.
        progress_queue: Cross-process queue carrying WebSocket-ready events.
        pause_event:    Setting this asks the worker to stop after its current chunk.
        started_at:     Monotonic start time (``time.time()``) for diagnostics.
        file_paths:     The list of input files this worker is processing.
    """

    process: mp.Process
    progress_queue: Any  # mp.Queue (cannot be used as a generic type at runtime)
    pause_event: Any     # mp.Event / synchronize.Event
    started_at: float
    file_paths: list[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Safe queue emission
# ---------------------------------------------------------------------------


def safe_put(q: Any, event: dict[str, Any]) -> bool:
    """Put *event* on *q* without ever blocking the worker.

    Uses ``put_nowait``; if the queue is full, the event is silently dropped.
    Returns ``True`` if the event was enqueued, ``False`` if dropped.
    """
    try:
        q.put_nowait(event)
        return True
    except _queue_module.Full:
        return False


# ---------------------------------------------------------------------------
# Event constructors — keep payload keys consistent with ``api/contracts.py``
# ---------------------------------------------------------------------------


def make_chunk_start(chunk_index: int, total_chunks: int, operation: str) -> dict[str, Any]:
    """Build a ``chunk_start`` event payload."""
    return {
        "type": EVENT_CHUNK_START,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "operation": operation,
    }


def make_chunk_progress(operation: str, pct: int) -> dict[str, Any]:
    """Build a ``chunk_progress`` event payload (pct in [0, 100])."""
    return {
        "type": EVENT_CHUNK_PROGRESS,
        "operation": operation,
        "pct": max(0, min(100, int(pct))),
    }


def make_chunk_done(
    chunk_index: int,
    status: str,
    accuracy_delta: dict[str, float],
    reason: str | None = None,
) -> dict[str, Any]:
    """Build a ``chunk_done`` event payload (status ∈ {'accepted', 'rejected'})."""
    return {
        "type": EVENT_CHUNK_DONE,
        "chunk_index": chunk_index,
        "status": status,
        "accuracy_delta": accuracy_delta,
        "reason": reason,
    }


def make_ingest_complete(chunks_accepted: int, chunks_rejected: int) -> dict[str, Any]:
    """Build an ``ingest_complete`` event payload."""
    return {
        "type": EVENT_INGEST_COMPLETE,
        "chunks_accepted": chunks_accepted,
        "chunks_rejected": chunks_rejected,
    }


def make_ingest_paused(chunks_completed: int, chunks_remaining: int) -> dict[str, Any]:
    """Build an ``ingest_paused`` event payload."""
    return {
        "type": EVENT_INGEST_PAUSED,
        "chunks_completed": chunks_completed,
        "chunks_remaining": chunks_remaining,
    }


def make_file_rejected(filename: str, reason: str) -> dict[str, Any]:
    """Build a ``file_rejected`` event payload."""
    return {
        "type": EVENT_FILE_REJECTED,
        "filename": filename,
        "reason": reason,
    }
