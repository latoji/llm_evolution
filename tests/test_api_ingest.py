"""Tests for api/routes/ingest.py — upload, pause, status endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from api import state
from api.main import app


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
# POST /ingest/upload
# ---------------------------------------------------------------------------


def test_upload_txt_returns_200(client: TestClient, tmp_path: Path) -> None:
    """A valid .txt upload returns 200 and lists the file as accepted."""
    mock_handle = MagicMock()
    mock_handle.process.is_alive.return_value = True
    mock_handle.pause_event.is_set.return_value = False
    mock_handle.file_paths = []
    mock_handle.started_at = 0.0

    with (
        patch("api.routes.ingest.start_worker", return_value=mock_handle),
        patch("api.routes.ingest.asyncio.create_task"),
    ):
        resp = client.post(
            "/ingest/upload",
            files=[("files", ("corpus.txt", b"Hello world " * 200, "text/plain"))],
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "corpus.txt" in data["accepted_files"]
    assert data["rejected_files"] == []


def test_upload_non_txt_is_rejected(client: TestClient) -> None:
    """A non-.txt file is listed in rejected_files and does not start a worker."""
    with patch("api.routes.ingest.start_worker") as mock_start:
        resp = client.post(
            "/ingest/upload",
            files=[("files", ("data.csv", b"a,b,c", "text/csv"))],
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted_files"] == []
    assert len(data["rejected_files"]) == 1
    assert data["rejected_files"][0]["filename"] == "data.csv"
    mock_start.assert_not_called()


def test_upload_while_worker_running_returns_409(client: TestClient) -> None:
    """Uploading while a worker is already running returns 409 Conflict."""
    mock_handle = MagicMock()
    mock_handle.process.is_alive.return_value = True
    state.worker_handle = mock_handle

    with patch("api.routes.ingest.is_alive", return_value=True):
        resp = client.post(
            "/ingest/upload",
            files=[("files", ("new.txt", b"text", "text/plain"))],
        )
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# POST /ingest/pause
# ---------------------------------------------------------------------------


def test_pause_with_no_worker_returns_400(client: TestClient) -> None:
    """Pausing when no worker is running returns 400 Bad Request."""
    state.worker_handle = None
    resp = client.post("/ingest/pause")
    assert resp.status_code == 400


def test_pause_while_worker_running_returns_200(client: TestClient) -> None:
    """Pausing a running worker returns 200 with paused=True."""
    mock_handle = MagicMock()
    mock_handle.pause_event = MagicMock()
    state.worker_handle = mock_handle

    with (
        patch("api.routes.ingest.is_alive", return_value=True),
        patch("api.routes.ingest.pause_worker"),
    ):
        resp = client.post("/ingest/pause")

    assert resp.status_code == 200
    data = resp.json()
    assert data["paused"] is True


# ---------------------------------------------------------------------------
# GET /ingest/status
# ---------------------------------------------------------------------------


def test_status_before_any_upload_is_idle(client: TestClient) -> None:
    """Status returns 'idle' when no worker has ever been started."""
    state.worker_handle = None
    resp = client.get("/ingest/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "idle"
    assert data["chunks_accepted"] == 0
    assert data["chunks_rejected"] == 0


def test_status_with_running_worker(client: TestClient) -> None:
    """Status returns 'running' when a worker is alive and not paused."""
    mock_handle = MagicMock()
    mock_handle.pause_event.is_set.return_value = False
    state.worker_handle = mock_handle

    with patch("api.routes.ingest.is_alive", return_value=True):
        resp = client.get("/ingest/status")

    assert resp.status_code == 200
    assert resp.json()["state"] == "running"


def test_status_after_worker_finishes(client: TestClient) -> None:
    """Status returns 'complete' when a worker has exited."""
    mock_handle = MagicMock()
    state.worker_handle = mock_handle

    with patch("api.routes.ingest.is_alive", return_value=False):
        resp = client.get("/ingest/status")

    assert resp.status_code == 200
    assert resp.json()["state"] == "complete"
