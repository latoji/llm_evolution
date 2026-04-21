"""Application-level singletons shared across all API routes.

All fields are module-level globals initialised to safe defaults.
The FastAPI lifespan (api/main.py) sets ``store`` and ``db_path`` on startup.
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
