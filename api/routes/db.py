"""DB viewer endpoints — read-only table inspection and full database reset."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from api import state
from api.contracts import DBResetResponse
from api.ingest_worker import is_alive

router = APIRouter()

# Tables that may be queried via the DB Viewer page.
ALLOWED_TABLES: frozenset[str] = frozenset({
    "chunks",
    "char_ngrams",
    "word_ngrams",
    "token_ngrams",
    "model_accuracy",
    "nn_checkpoints",
    "last_generations",
})

# Maps the DB-Viewer table name to the actual DuckDB table name.
# ``None`` means the table is not yet in the schema — returns empty rows.
_TABLE_MAP: dict[str, str | None] = {
    "chunks": "corpus_chunks",
    "char_ngrams": "char_ngrams",
    "word_ngrams": "word_ngrams",
    "token_ngrams": "token_ngrams",
    "model_accuracy": "model_accuracy",
    "nn_checkpoints": "nn_checkpoints",
    "last_generations": None,   # not in current schema; always returns empty
}


@router.get("/tables")
async def list_tables() -> dict[str, list[str]]:
    """List the tables available for inspection via the DB Viewer."""
    return {"tables": sorted(ALLOWED_TABLES)}


@router.get("/table/{name}")
async def get_table(
    name: str,
    limit: int = Query(default=100, ge=1, le=10_000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Return paginated rows from a whitelisted table.

    Returns 404 for table names not in the allowlist.
    """
    if name not in ALLOWED_TABLES:
        raise HTTPException(status_code=404, detail=f"Table '{name}' not found")

    actual = _TABLE_MAP[name]
    if actual is None:
        return {"rows": [], "total": 0}

    conn = state.store._conn  # type: ignore[union-attr]
    total_row = conn.execute(f"SELECT COUNT(*) FROM {actual}").fetchone()
    total = int(total_row[0]) if total_row else 0

    cursor = conn.execute(
        f"SELECT * FROM {actual} LIMIT ? OFFSET ?", [limit, offset]
    )
    col_names = [d[0] for d in cursor.description] if cursor.description else []
    rows = cursor.fetchall()

    return {
        "rows": [dict(zip(col_names, row)) for row in rows],
        "total": total,
    }


@router.get("/row_counts")
async def row_counts() -> dict[str, int]:
    """Return the row count for every allowed table."""
    conn = state.store._conn  # type: ignore[union-attr]
    counts: dict[str, int] = {}
    for viewer_name, actual in _TABLE_MAP.items():
        if actual is None:
            counts[viewer_name] = 0
            continue
        row = conn.execute(f"SELECT COUNT(*) FROM {actual}").fetchone()
        counts[viewer_name] = int(row[0]) if row else 0
    return counts


@router.post("/reset", response_model=DBResetResponse)
async def reset_db() -> DBResetResponse:
    """Drop and recreate all tables, delete .pt checkpoints, kill any running worker.

    WARNING: destroys all ingested data and model checkpoints.

    The worker is terminated first so it cannot write to the recreated DB
    with a stale connection.
    """
    if state.worker_handle and is_alive(state.worker_handle):
        state.worker_handle.process.terminate()
        state.worker_handle.process.join(timeout=5)
        state.worker_handle = None

    state.store.reset_all()  # type: ignore[union-attr]

    # Reset all progress counters
    state.chunks_accepted = 0
    state.chunks_rejected = 0
    state.current_chunk = None
    state.total_chunks = None

    return DBResetResponse(success=True, message="Database and checkpoints reset successfully")
