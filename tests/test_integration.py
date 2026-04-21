"""End-to-end integration test — exercises the full backend pipeline.

Marked ``slow``: starts a real uvicorn process and drives it via HTTP.
Run explicitly with ``pytest -m slow tests/test_integration.py``.
Excluded from the default ``pytest -m 'not slow'`` suite.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests


_BACKEND_URL = "http://localhost:8765"
_STARTUP_TIMEOUT_S = 30
_INGEST_TIMEOUT_S = 300
_N_MODELS = 13


def _wait_for_backend(base_url: str, timeout: float) -> None:
    """Poll /docs until the server responds or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/docs", timeout=2).status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    pytest.fail(f"Backend at {base_url} did not start within {timeout}s")


def _seed_path() -> Path:
    """Return the seed corpus path, generating it if absent."""
    seed = Path("data/seed.txt")
    if not seed.exists():
        from scripts.seed_corpus import write_seed_file
        write_seed_file(seed)
    return seed


@pytest.mark.slow
def test_end_to_end() -> None:
    """Upload a seed corpus and verify that all 13 models produce accuracy rows."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8765"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_backend(_BACKEND_URL, _STARTUP_TIMEOUT_S)

        # 1. Reset DB to guarantee a clean state.
        r = requests.post(f"{_BACKEND_URL}/db/reset")
        assert r.status_code == 200, f"reset failed: {r.text}"

        # 2. Upload seed corpus.
        seed = _seed_path()
        with seed.open("rb") as fh:
            r = requests.post(
                f"{_BACKEND_URL}/ingest/upload",
                files={"files": ("seed.txt", fh, "text/plain")},
            )
        assert r.status_code == 200, f"upload failed: {r.text}"
        body = r.json()
        assert "seed.txt" in body["accepted_files"]

        # 3. Poll /ingest/status until ingest is no longer running.
        deadline = time.time() + _INGEST_TIMEOUT_S
        while time.time() < deadline:
            status = requests.get(f"{_BACKEND_URL}/ingest/status").json()
            if status["state"] not in ("running", "paused"):
                break
            time.sleep(1)
        else:
            pytest.fail("Ingest did not complete within the timeout")

        assert status["state"] == "complete", f"unexpected state: {status['state']}"

        # 4. Verify accuracy rows for all 13 models.
        acc_resp = requests.get(f"{_BACKEND_URL}/stats/accuracy").json()
        model_data = acc_resp.get("models", acc_resp)  # defensive: handle flat dict too
        assert len(model_data) == _N_MODELS, (
            f"expected {_N_MODELS} model accuracy series, got {len(model_data)}: "
            f"{list(model_data.keys())}"
        )
        for name, series in model_data.items():
            assert len(series) >= 1, f"no accuracy data for model '{name}'"

        # 5. Verify generate returns 13 outputs.
        gen_resp = requests.post(
            f"{_BACKEND_URL}/generate",
            json={"word_count": 20, "auto_correct": False},
        )
        assert gen_resp.status_code == 200, f"generate failed: {gen_resp.text}"
        outputs = gen_resp.json().get("outputs", gen_resp.json())
        assert len(outputs) == _N_MODELS, (
            f"expected {_N_MODELS} generate outputs, got {len(outputs)}"
        )

    finally:
        if sys.platform == "win32":
            proc.terminate()
        else:
            proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
