# Track Z — Integration & Smoke Test

**Model**: `claude-sonnet-4-6`
**Dependencies**: All tracks (0, A, B1, B2, C, D1, D2, E) complete

## Scope

Wire the pieces together, verify the end-to-end path works on a fresh machine, and ship a one-line start command. Write the single Playwright smoke test that proves every page functions against a real backend.

This track does **not** implement features. If you find yourself writing business logic, you've crossed into another track's territory — file a bug against that track instead.

## Upstream dependencies

Everything.

## Downstream consumers

Humans.

## Files owned

```
README.md                              (project root — user-facing quickstart)
scripts/dev.sh                         (starts backend + frontend concurrently)
scripts/seed_corpus.py                 (generates a small known-good .txt for smoke tests)
frontend/tests/e2e.spec.ts             (the Playwright smoke test)
tests/test_integration.py              (one end-to-end Python test)
.gitignore                             (consolidates per-track .gitignore needs)
pyproject.toml / requirements.txt      (freeze dependency versions)
frontend/playwright.config.ts
```

## Files you must NOT modify

Any file owned by another track. You may only **read** them to write the smoke test and README.

---

## Implementation

### `README.md` (project root)

Target audience: a developer who just cloned the repo and has Python 3.11+ and Node 20+ installed. Sections:

1. **What this is** — two paragraphs. Educational LLM-evolution app: 11 Markov + 2 neural models, accuracy graphed over ingested chunks.
2. **Requirements** — Python 3.11+, Node 20+, ~2 GB disk, optional CUDA GPU with 4+ GB VRAM.
3. **Install** — `pip install -r requirements.txt` and `cd frontend && npm install`.
4. **Run** — `bash scripts/dev.sh` — opens browser to `http://localhost:5173`.
5. **First steps** — upload a .txt, watch the Ingest page, then check Stats.
6. **Architecture** — one diagram (ASCII) showing FastAPI ↔ IngestWorker ↔ DuckDB ↔ Frontend.
7. **Reset** — click "Reset DB" on the DB page, or delete `db/llm_evolution.duckdb` and `model/checkpoints/`.
8. **Links** — pointer to `SPEC.md` for architecture, `specs/` for per-track implementation.

Keep it under 150 lines. This is a quickstart, not a manual.

### `scripts/dev.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Start backend in background
cd "$(dirname "$0")/.."
python -m uvicorn api.main:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend (blocks)
cd frontend
npm run dev &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true" EXIT

wait
```

Windows users get `scripts/dev.ps1` with equivalent behavior (use `Start-Process` and `Wait-Process`).

### `scripts/seed_corpus.py`

Generates `data/seed.txt` — ~5 KB of clean English prose known to pass the WVM pre-screen. Used by both `test_integration.py` and the Playwright smoke test.

Simplest implementation: bundle a public-domain paragraph (e.g. a Project Gutenberg excerpt) as a string constant and write it to disk.

### `tests/test_integration.py`

One Python test that exercises the full pipeline without a browser:

```python
import pytest, time, requests, subprocess, signal
from pathlib import Path

@pytest.mark.slow
def test_end_to_end(tmp_path):
    # 1. Start uvicorn in subprocess
    proc = subprocess.Popen(["python", "-m", "uvicorn", "api.main:app", "--port", "8765"])
    try:
        # wait for /docs to respond
        for _ in range(30):
            try:
                if requests.get("http://localhost:8765/docs").status_code == 200:
                    break
            except requests.ConnectionError:
                time.sleep(0.5)
        else:
            pytest.fail("backend did not start")

        # 2. Reset DB to guarantee clean state
        requests.post("http://localhost:8765/db/reset")

        # 3. Upload seed corpus
        seed = Path("data/seed.txt")
        with seed.open("rb") as f:
            r = requests.post("http://localhost:8765/ingest/upload",
                              files={"files": ("seed.txt", f, "text/plain")})
        assert r.status_code == 200

        # 4. Poll status until ingest complete (timeout 300s)
        deadline = time.time() + 300
        while time.time() < deadline:
            if not requests.get("http://localhost:8765/ingest/status").json()["running"]:
                break
            time.sleep(1)
        else:
            pytest.fail("ingest did not complete")

        # 5. Verify accuracy rows for all 13 models
        acc = requests.get("http://localhost:8765/stats/accuracy").json()
        assert len(acc) == 13
        for name, series in acc.items():
            assert len(series) >= 1, f"no accuracy data for {name}"

        # 6. Generate and verify 13 outputs
        r = requests.post("http://localhost:8765/generate",
                          json={"n_words": 20, "augment": False})
        assert r.status_code == 200
        assert len(r.json()) == 13
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
```

Mark as `@pytest.mark.slow` — this test takes minutes. Excluded from default CI, run manually before a release.

### `frontend/tests/e2e.spec.ts` (Playwright)

```typescript
import { test, expect } from "@playwright/test";

test("full user journey: ingest → stats → generate → db", async ({ page }) => {
  await page.goto("http://localhost:5173");

  // Reset DB first (start from known state)
  await page.click("text=DB");
  await page.click('[data-testid="db-reset-btn"]');
  await page.click("text=OK");   // confirm dialog

  // Ingest
  await page.click("text=Ingest");
  const filePath = require("path").resolve(__dirname, "../../data/seed.txt");
  await page.setInputFiles('[data-testid="ingest-dropzone"] input', filePath);

  // Wait for ingest_complete (reflected in UI)
  await expect(page.locator("text=Ingest complete")).toBeVisible({ timeout: 300_000 });

  // Stats — verify 13 model cards rendered
  await page.click("text=Stats");
  for (const name of ["char_1gram", "char_3gram", "word_2gram", "bpe_3gram", "feedforward", "transformer"]) {
    await expect(page.locator(`[data-testid="model-card-${name}"]`)).toBeVisible();
  }

  // Generate
  await page.click("text=Generate");
  await page.fill('input[type="number"]', "20");
  await page.click('[data-testid="generate-btn"]');
  await expect(page.locator("text=Real words:")).toHaveCount(13, { timeout: 60_000 });

  // DB
  await page.click("text=DB");
  await expect(page.locator("text=char_ngrams")).toBeVisible();
});
```

Run with `npx playwright test`. Requires both backend and frontend running (use `scripts/dev.sh`).

### `pyproject.toml` / `requirements.txt`

Pin versions. Minimum set (each track may add more within its scope — this file is the union):

```
fastapi>=0.110,<0.120
uvicorn[standard]>=0.27
duckdb>=0.10
torch>=2.2
numpy>=1.26
pytest>=8.0
pytest-asyncio>=0.23
httpx>=0.27
python-multipart>=0.0.9
pyenchant>=3.2       # or the chosen SCOWL wrapper from Track 0
```

Do not add frontend dependencies here — they live in `frontend/package.json`.

### `.gitignore`

Merge what each track needs:

```
# Python
__pycache__/
*.pyc
.venv/
.pytest_cache/

# DuckDB
db/*.duckdb
db/*.duckdb.wal

# Checkpoints
model/checkpoints/*.pt
model/counts/*.json
model/counts/archive/

# Frontend
frontend/node_modules/
frontend/dist/
frontend/.vite/

# User uploads
data/uploads/

# OS
.DS_Store
Thumbs.db

# Editor
.vscode/
.idea/
```

---

## Testing

There is no separate test file owned by this track beyond `tests/test_integration.py` and `frontend/tests/e2e.spec.ts`. The point of this track is that the tests in other tracks all still pass when the full system is wired together.

Before declaring done:

```bash
# Backend unit tests (fast)
pytest -m "not slow"

# Frontend typecheck
cd frontend && npm run typecheck

# Integration (slow)
pytest -m slow tests/test_integration.py

# E2E (slowest — requires dev server running)
bash scripts/dev.sh &
sleep 10
cd frontend && npx playwright test
```

---

## Acceptance criteria

- [ ] `bash scripts/dev.sh` launches backend and frontend; browser at :5173 shows the Ingest page
- [ ] `pytest -m "not slow"` all green across every track
- [ ] `pytest -m slow tests/test_integration.py` green
- [ ] `cd frontend && npx playwright test` green
- [ ] `README.md` quickstart works on a fresh clone (verified by walking through it yourself)
- [ ] Every `data-testid` referenced in the E2E test exists in the frontend
- [ ] No hardcoded absolute paths anywhere in the repo

---

## Pitfalls

- **Do not "fix" bugs in other tracks.** If `test_integration.py` fails because of a Track A bug, file it back to Track A. This track enforces contracts; it doesn't patch them.
- **Windows vs POSIX paths.** `scripts/dev.sh` is POSIX-only. Provide `scripts/dev.ps1` too. Never assume `/` separators in the backend — `pathlib.Path` handles both.
- **Port collisions.** 8000 (backend) and 5173 (frontend) are defaults. Document how to change them (`--port` flag and `VITE_PORT` env var).
- **Playwright browsers must be installed separately** — `npx playwright install chromium`. Add this to the README install section.
- **The seed corpus must be deterministic** — if it changes, accuracy graphs in screenshots change, and the E2E test's timeouts may need adjustment.
- **`subprocess.Popen` on Windows** handles signals differently — use `proc.terminate()` rather than `send_signal(SIGTERM)` if the test needs to run on Windows CI.
- **Do not commit `data/seed.txt`** if it's large — generate it via `scripts/seed_corpus.py` in a fixture instead.

---

## Model assignment

**`claude-opus-4-6`.** Integration work is deceptively hard. When the E2E test fails at 2 a.m., the agent has to read output from eight other agents' code, diagnose which contract is actually being violated, and resist the temptation to "fix" it locally. That triage is judgment work, and it's exactly where Sonnet tends to produce a plausible-looking patch that hides the real defect. Opus 4.6 has enough reasoning depth to correctly route issues back to their owning track instead.
