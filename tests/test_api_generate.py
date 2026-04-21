"""Tests for api/routes/generate.py — POST /generate endpoint."""

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
# POST /generate
# ---------------------------------------------------------------------------


def _mock_generate_one(*args, **kwargs) -> str:  # type: ignore[no-untyped-def]
    """Stub that returns a short English text without touching real models."""
    return "the cat sat on the mat"


def test_generate_returns_13_outputs(client: TestClient) -> None:
    """POST /generate with n=20 returns exactly 13 model outputs."""
    with patch("api.routes.generate._generate_one", side_effect=_mock_generate_one):
        resp = client.post("/generate", json={"word_count": 20, "auto_correct": False})

    assert resp.status_code == 200
    data = resp.json()
    assert "outputs" in data
    assert len(data["outputs"]) == 13


def test_generate_outputs_have_required_fields(client: TestClient) -> None:
    """Each output entry has model_name, raw_text, corrected_text, word_results, real_word_pct."""
    with patch("api.routes.generate._generate_one", side_effect=_mock_generate_one):
        resp = client.post("/generate", json={"word_count": 20, "auto_correct": False})

    data = resp.json()
    for entry in data["outputs"]:
        assert "model_name" in entry
        assert "raw_text" in entry
        assert "corrected_text" in entry
        assert "word_results" in entry
        assert "real_word_pct" in entry


def test_generate_with_auto_correct_does_not_crash(client: TestClient) -> None:
    """POST /generate with auto_correct=True completes without error."""
    with patch("api.routes.generate._generate_one", side_effect=_mock_generate_one):
        resp = client.post("/generate", json={"word_count": 20, "auto_correct": True})

    assert resp.status_code == 200
    data = resp.json()
    # At least one model should have corrected_text (text is all real words here;
    # corrected_text is only set when auto_correct=True AND word_results is non-empty)
    assert len(data["outputs"]) == 13


def test_generate_word_count_below_minimum_returns_422(client: TestClient) -> None:
    """word_count below the minimum (20) should be rejected with 422."""
    resp = client.post("/generate", json={"word_count": 5, "auto_correct": False})
    assert resp.status_code == 422


def test_generate_word_count_above_maximum_returns_422(client: TestClient) -> None:
    """word_count above the maximum (500) should be rejected with 422."""
    resp = client.post("/generate", json={"word_count": 1000, "auto_correct": False})
    assert resp.status_code == 422


def test_generate_all_model_names_present(client: TestClient) -> None:
    """The 13 expected model names are all present in the response."""
    from eval.monte_carlo import MODELS

    with patch("api.routes.generate._generate_one", side_effect=_mock_generate_one):
        resp = client.post("/generate", json={"word_count": 20, "auto_correct": False})

    data = resp.json()
    returned_names = {e["model_name"] for e in data["outputs"]}
    expected_names = {m.name for m in MODELS}
    assert returned_names == expected_names
