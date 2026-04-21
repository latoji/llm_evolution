"""Tests for api/routes/stats.py — models, accuracy, last_generations endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import pytest
from starlette.testclient import TestClient

from api import state
from api.main import app
from eval.monte_carlo import MODELS


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
# GET /stats/models
# ---------------------------------------------------------------------------


def test_models_returns_13_entries(client: TestClient) -> None:
    """GET /stats/models returns exactly 13 model descriptors."""
    resp = client.get("/stats/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 13


def test_models_have_required_fields(client: TestClient) -> None:
    """Each model descriptor has name, family, order, and display_order fields."""
    resp = client.get("/stats/models")
    data = resp.json()
    for entry in data:
        assert "name" in entry
        assert "family" in entry
        assert "display_order" in entry


def test_models_names_match_registry(client: TestClient) -> None:
    """Model names in the response match the MODELS registry exactly."""
    resp = client.get("/stats/models")
    data = resp.json()
    registry_names = {m.name for m in MODELS}
    response_names = {e["name"] for e in data}
    assert response_names == registry_names


# ---------------------------------------------------------------------------
# GET /stats/accuracy
# ---------------------------------------------------------------------------


def test_accuracy_returns_200_with_empty_db(client: TestClient) -> None:
    """GET /stats/accuracy returns 200 with an empty models dict when no data."""
    resp = client.get("/stats/accuracy")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert isinstance(data["models"], dict)
    assert len(data["models"]) == 0  # no accuracy rows ingested yet


def test_accuracy_groups_by_model_name(client: TestClient) -> None:
    """After inserting accuracy rows, each model's history is grouped correctly."""
    # Insert a chunk and accuracy rows
    chunk_id = state.store.insert_chunk("f.txt", 0, "hash", 100, {})
    state.store.insert_accuracy("char_1gram", chunk_id, 0.5, None)
    state.store.insert_accuracy("char_2gram", chunk_id, 0.6, None)

    resp = client.get("/stats/accuracy")
    assert resp.status_code == 200
    data = resp.json()["models"]
    assert "char_1gram" in data
    assert "char_2gram" in data
    assert data["char_1gram"][0]["accuracy"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# GET /stats/last_generations
# ---------------------------------------------------------------------------


def test_last_generations_returns_200(client: TestClient) -> None:
    """GET /stats/last_generations returns 200 with an outputs list."""
    resp = client.get("/stats/last_generations")
    assert resp.status_code == 200
    data = resp.json()
    assert "outputs" in data
    assert isinstance(data["outputs"], list)
