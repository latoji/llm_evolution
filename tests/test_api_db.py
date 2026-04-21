"""Tests for api/routes/db.py — tables, table/{name}, row_counts, reset endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import pytest
from starlette.testclient import TestClient

from api import state
from api.main import app
from api.routes.db import ALLOWED_TABLES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:  # type: ignore[return]
    """TestClient with isolated temp DuckDB; lifespan replaced with a no-op."""
    from db.store import Store

    @asynccontextmanager
    async def _noop_lifespan(a: object) -> AsyncIterator[None]:
        yield

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan  # type: ignore[assignment]

    state.store = Store(tmp_path / "test.duckdb")
    state.db_path = tmp_path / "test.duckdb"
    state.worker_handle = None
    state.ws_clients = set()
    state.chunks_accepted = 0
    state.chunks_rejected = 0
    state.current_chunk = None
    state.total_chunks = None
    state.progress_relay_task = None

    with TestClient(app) as c:
        yield c  # type: ignore[misc]

    app.router.lifespan_context = original_lifespan  # type: ignore[assignment]
    if state.store:
        state.store.close()
    state.store = None
    state.worker_handle = None
    state.ws_clients = set()
    state.progress_relay_task = None


# ---------------------------------------------------------------------------
# GET /db/tables
# ---------------------------------------------------------------------------


def test_tables_returns_200(client: TestClient) -> None:
    """GET /db/tables returns 200."""
    resp = client.get("/db/tables")
    assert resp.status_code == 200


def test_tables_lists_7_entries(client: TestClient) -> None:
    """GET /db/tables lists exactly 7 table names."""
    resp = client.get("/db/tables")
    data = resp.json()
    assert "tables" in data
    assert len(data["tables"]) == 7


def test_tables_contains_expected_names(client: TestClient) -> None:
    """GET /db/tables includes all names from ALLOWED_TABLES."""
    resp = client.get("/db/tables")
    returned = set(resp.json()["tables"])
    assert returned == ALLOWED_TABLES


# ---------------------------------------------------------------------------
# GET /db/table/{name}
# ---------------------------------------------------------------------------


def test_table_chunks_returns_200(client: TestClient) -> None:
    """GET /db/table/chunks returns 200 (empty rows on a fresh DB)."""
    resp = client.get("/db/table/chunks?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "rows" in data
    assert isinstance(data["rows"], list)


def test_table_chunks_respects_limit(client: TestClient) -> None:
    """GET /db/table/chunks with limit=5 returns at most 5 rows."""
    # Insert 7 chunks
    for i in range(7):
        state.store.insert_chunk("f.txt", i, f"h{i}", 100, {})

    resp = client.get("/db/table/chunks?limit=5")
    assert resp.status_code == 200
    assert len(resp.json()["rows"]) <= 5


def test_table_unknown_name_returns_404(client: TestClient) -> None:
    """GET /db/table/evil_table returns 404."""
    resp = client.get("/db/table/evil_table")
    assert resp.status_code == 404


def test_table_last_generations_returns_empty(client: TestClient) -> None:
    """GET /db/table/last_generations returns empty rows (not in schema)."""
    resp = client.get("/db/table/last_generations")
    assert resp.status_code == 200
    data = resp.json()
    assert data["rows"] == []


# ---------------------------------------------------------------------------
# GET /db/row_counts
# ---------------------------------------------------------------------------


def test_row_counts_returns_all_tables(client: TestClient) -> None:
    """GET /db/row_counts returns a count for each allowed table."""
    resp = client.get("/db/row_counts")
    assert resp.status_code == 200
    data = resp.json()
    for name in ALLOWED_TABLES:
        assert name in data, f"Missing count for table '{name}'"
        assert isinstance(data[name], int)


def test_row_counts_zero_on_empty_db(client: TestClient) -> None:
    """All row counts are 0 on a freshly created database."""
    resp = client.get("/db/row_counts")
    data = resp.json()
    for name, count in data.items():
        assert count == 0, f"Expected 0 rows for '{name}', got {count}"


# ---------------------------------------------------------------------------
# POST /db/reset
# ---------------------------------------------------------------------------


def test_reset_returns_success(client: TestClient) -> None:
    """POST /db/reset returns success=True."""
    resp = client.post("/db/reset")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_reset_clears_chunks(client: TestClient) -> None:
    """After reset, the chunks table is empty."""
    state.store.insert_chunk("f.txt", 0, "h", 100, {})

    resp_before = client.get("/db/table/chunks")
    assert resp_before.json()["total"] == 1

    client.post("/db/reset")

    resp_after = client.get("/db/table/chunks")
    assert resp_after.json()["total"] == 0


def test_reset_row_counts_back_to_zero(client: TestClient) -> None:
    """After reset, all row counts return to 0."""
    chunk_id = state.store.insert_chunk("f.txt", 0, "h", 100, {})
    state.store.insert_accuracy("char_1gram", chunk_id, 0.5, None)

    client.post("/db/reset")

    resp = client.get("/db/row_counts")
    for name, count in resp.json().items():
        assert count == 0, f"After reset: '{name}' has {count} rows"
