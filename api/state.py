"""Application-level singletons shared across all API routes.

All fields are module-level globals initialised to safe defaults.
The FastAPI lifespan (api/main.py) sets ``store`` and ``db_path`` on startup.

DuckDB allows only ONE writer at a time across processes.  When the ingest
worker subprocess is running it holds the exclusive write lock.  ``store`` is
set to ``None`` just before the worker starts and restored via
:func:`get_store` once the worker exits.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from db.store import Store
    from api.worker_types import WorkerHandle

# ---------------------------------------------------------------------------
# Core singletons — set by lifespan on startup
# ---------------------------------------------------------------------------

store: Optional["Store"] = None
db_path: Path = Path("db/llm_evolution.duckdb")
worker_handle: Optional["WorkerHandle"] = None

# ---------------------------------------------------------------------------
# WebSocket fan-out — guarded by an asyncio.Lock
# ---------------------------------------------------------------------------

ws_clients: set = set()  # set[WebSocket]; mutated only under ws_clients_lock
ws_clients_lock: asyncio.Lock = asyncio.Lock()
progress_relay_task: Optional["asyncio.Task[None]"] = None

# ---------------------------------------------------------------------------
# Ingest progress counters — updated by relay_progress as events arrive
# ---------------------------------------------------------------------------

chunks_accepted: int = 0
chunks_rejected: int = 0
current_chunk: Optional[int] = None
total_chunks: Optional[int] = None


# ---------------------------------------------------------------------------
# Store accessor — reopens lazily after the worker releases the write lock
# ---------------------------------------------------------------------------


def get_store() -> "Optional[Store]":
    """Return the shared Store, reopening it if the worker has exited.

    Returns None while the ingest worker subprocess is still running (DB
    is locked by the worker). Callers should return empty/sensible data in
    that case and avoid crashing.
    """
    global store
    if store is not None:
        return store
    # Worker finished (or never started) — safe to reopen.
    if worker_handle is None or not worker_handle.process.is_alive():
        from db.store import Store as _Store
        store = _Store(db_path)
        return store
    return None  # Worker still holds the write lock.
