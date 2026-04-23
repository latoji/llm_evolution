"""LLM Evolution FastAPI application — startup, middleware, and router registration."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import state
from api.routes import db as db_routes
from api.routes import generate, ingest, stats, ws
from db.migrate_from_json import migrate
from db.store import Store

_DIST = Path("frontend/dist")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the DuckDB store on startup and close it on shutdown.

    Also terminates any still-running ingest worker to avoid zombie processes.
    """
    _db_path = Path("db/llm_evolution.duckdb")
    state.db_path = _db_path
    state.store = Store(_db_path)
    migrate(state.store)  # idempotent JSON → DuckDB import (no-op if archive exists)
    yield
    if state.worker_handle and state.worker_handle.process.is_alive():
        state.worker_handle.process.terminate()
        state.worker_handle.process.join(timeout=5)
    if state.progress_relay_task and not state.progress_relay_task.done():
        state.progress_relay_task.cancel()
    if state.store:
        state.store.close()


app = FastAPI(title="LLM Evolution", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router,     prefix="/ingest",   tags=["ingest"])
app.include_router(stats.router,      prefix="/stats",    tags=["stats"])
app.include_router(generate.router,   prefix="/generate", tags=["generate"])
app.include_router(db_routes.router,  prefix="/db",       tags=["db"])
app.include_router(ws.router)  # /ws/progress — no extra prefix

# Serve the production React build when frontend/dist/ exists.
# API routes above always take precedence; this only handles unmatched paths.
if _DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> FileResponse:  # noqa: ARG001
        """Serve index.html for all unmatched routes (React SPA client-side routing)."""
        return FileResponse(_DIST / "index.html")
