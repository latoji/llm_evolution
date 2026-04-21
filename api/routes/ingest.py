"""Ingest endpoints — upload .txt files, pause the worker, query status."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api import state
from api.contracts import IngestPauseResponse, IngestStatusResponse, IngestUploadResponse
from api.ingest_worker import is_alive, pause_worker, start_worker

router = APIRouter()

_UPLOAD_DIR: Path = Path("data/uploads")
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=IngestUploadResponse)
async def upload(files: list[UploadFile] = File(...)) -> IngestUploadResponse:
    """Accept .txt uploads and start the ingest worker.

    Rejects non-.txt files (listed in rejected_files) without starting a worker.
    Returns 409 if a worker is already running.
    NOTE: ``total_chunks`` in the response is 0 — the actual count arrives via
    WebSocket ``chunk_start`` events during ingest.
    """
    if state.worker_handle and is_alive(state.worker_handle):
        raise HTTPException(status_code=409, detail="ingest already running")

    accepted: list[str] = []
    rejected: list[dict] = []
    saved: list[Path] = []

    for f in files:
        fname = f.filename or "unknown"
        if not fname.endswith(".txt"):
            rejected.append({"filename": fname, "reason": "only .txt files are allowed"})
            continue
        dest = _UPLOAD_DIR / fname
        dest.write_bytes(await f.read())
        saved.append(dest)
        accepted.append(fname)

    if saved:
        # Reset progress counters for the new ingest session.
        state.chunks_accepted = 0
        state.chunks_rejected = 0
        state.current_chunk = None
        state.total_chunks = None

        state.worker_handle = start_worker(saved, db_path=state.db_path)

        # Create the relay task only when it is not already running.
        if state.progress_relay_task is None or state.progress_relay_task.done():
            from api.routes.ws import relay_progress  # local import avoids circular dep

            state.progress_relay_task = asyncio.create_task(relay_progress())

    return IngestUploadResponse(
        accepted_files=accepted,
        rejected_files=rejected,
        total_chunks=0,  # unknown until the worker processes files; use WS events
    )


@router.post("/pause", response_model=IngestPauseResponse)
async def pause() -> IngestPauseResponse:
    """Ask the running worker to stop after its current chunk.

    Returns 400 if no worker is running.  The worker finishes its current
    chunk before honouring the pause — this is by design.
    """
    if not state.worker_handle or not is_alive(state.worker_handle):
        raise HTTPException(status_code=400, detail="no worker running")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, pause_worker, state.worker_handle)
    return IngestPauseResponse(
        paused=True,
        message="Worker will pause after the current chunk completes",
    )


@router.get("/status", response_model=IngestStatusResponse)
async def status() -> IngestStatusResponse:
    """Return the current ingest state and live progress counters."""
    h = state.worker_handle
    if h is None:
        ingest_state = "idle"
    elif is_alive(h):
        ingest_state = "paused" if h.pause_event.is_set() else "running"
    else:
        ingest_state = "complete"

    return IngestStatusResponse(
        state=ingest_state,
        current_chunk=state.current_chunk,
        total_chunks=state.total_chunks,
        chunks_accepted=state.chunks_accepted,
        chunks_rejected=state.chunks_rejected,
    )
