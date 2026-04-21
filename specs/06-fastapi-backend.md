# Track D2 — FastAPI Backend (HTTP + WebSocket)

**Model**: `claude-opus-4-6`
**Dependencies**: Tracks 0, A, B1, B2, C, D1 complete

## Scope

Expose the pipeline over HTTP and WebSocket. This layer is thin by design: it forwards requests to `Store`, spawns the `IngestWorker`, and relays the worker's progress queue onto connected WebSocket clients. No business logic lives here.

## Upstream dependencies

- `db/store.py` (Track 0)
- `api/contracts.py` (Track 0) — request/response/WS message schemas
- `api/ingest_worker.py` (Track D1) — `start_worker`, `pause_worker`, `is_alive`, `WorkerHandle`
- `eval/monte_carlo.py` (Track C) — for the `/generate` endpoint's on-demand generation

## Downstream consumers

- **Track E** (frontend) — every page hits these routes
- **Track Z** (integration) — smoke tests exercise these endpoints end-to-end

## Files owned

```
api/main.py
api/routes/__init__.py
api/routes/ingest.py
api/routes/stats.py
api/routes/generate.py
api/routes/db.py
api/routes/ws.py
api/state.py
tests/test_api_ingest.py
tests/test_api_stats.py
tests/test_api_generate.py
tests/test_api_db.py
tests/test_api_ws.py
```

## Files you must NOT modify

- Everything in Tracks 0, A, B1, B2, C, D1
- `api/contracts.py`, `api/ingest_worker.py`, `api/worker_types.py`

---

## Architecture

```
FastAPI app (api/main.py)
├── /ingest/*       → api/routes/ingest.py    (upload, pause, status)
├── /stats/*        → api/routes/stats.py     (accuracy history, model list)
├── /generate       → api/routes/generate.py  (one-shot generation from all 13 models)
├── /db/*           → api/routes/db.py        (read-only DB inspection, reset)
└── /ws/progress    → api/routes/ws.py        (WebSocket fan-out from worker queue)

api/state.py  (module-level singletons: current WorkerHandle, connected WS clients)
```

The backend is **single-process** FastAPI (uvicorn). The IngestWorker runs in its own `multiprocessing.Process` spawned from `api/routes/ingest.py`. No other processes.

---

## Implementation

### `api/main.py` (~80 lines)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
from db.store import Store
from db.migrate_from_json import migrate
from api.routes import ingest, stats, generate, db as db_routes, ws
from api import state

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    state.store = Store(Path("db/llm_evolution.duckdb"))
    state.store.migrate()            # idempotent schema creation
    migrate()                        # one-shot JSON → DuckDB import if applicable
    yield
    # Shutdown
    if state.worker_handle and state.worker_handle.process.is_alive():
        state.worker_handle.process.terminate()
        state.worker_handle.process.join(timeout=5)
    state.store.close()

app = FastAPI(title="LLM Evolution", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])

app.include_router(ingest.router,   prefix="/ingest",   tags=["ingest"])
app.include_router(stats.router,    prefix="/stats",    tags=["stats"])
app.include_router(generate.router, prefix="/generate", tags=["generate"])
app.include_router(db_routes.router,prefix="/db",       tags=["db"])
app.include_router(ws.router)  # /ws/progress has no prefix
```

### `api/state.py` (~30 lines)

```python
from typing import Optional
from db.store import Store
from api.worker_types import WorkerHandle
import asyncio

store: Optional[Store] = None
worker_handle: Optional[WorkerHandle] = None
ws_clients: set = set()             # set[WebSocket]
ws_clients_lock = asyncio.Lock()    # guards ws_clients mutations
progress_relay_task: Optional[asyncio.Task] = None
```

All routes import from `api.state` and mutate these module-level singletons. This is intentional — the backend is a single-process app and globals are the simplest correct model.

### `api/routes/ingest.py` (~150 lines)

Endpoints:

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/ingest/upload` | multipart with `files: list[UploadFile]` | `{"status": "started", "file_count": N}` |
| POST | `/ingest/pause` | — | `{"status": "pausing"}` |
| GET  | `/ingest/status` | — | `{"running": bool, "started_at": float \| None, "files": [...]}` |

```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import tempfile
from api import state
from api.ingest_worker import start_worker, pause_worker, is_alive

router = APIRouter()
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    if state.worker_handle and is_alive(state.worker_handle):
        raise HTTPException(409, "ingest already running")
    saved = []
    for f in files:
        if not f.filename.endswith(".txt"):
            raise HTTPException(400, f"only .txt allowed: {f.filename}")
        dest = UPLOAD_DIR / f.filename
        dest.write_bytes(await f.read())
        saved.append(dest)
    state.worker_handle = start_worker(saved)
    # kick off the async relay task if not already running
    if state.progress_relay_task is None or state.progress_relay_task.done():
        from api.routes.ws import relay_progress
        import asyncio
        state.progress_relay_task = asyncio.create_task(relay_progress())
    return {"status": "started", "file_count": len(saved)}

@router.post("/pause")
async def pause():
    if not state.worker_handle or not is_alive(state.worker_handle):
        raise HTTPException(400, "no worker running")
    # pause_event.set() is thread-safe but not async-safe in old pythons; use executor
    import asyncio
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, pause_worker, state.worker_handle)
    return {"status": "pausing"}

@router.get("/status")
async def status():
    h = state.worker_handle
    running = bool(h and is_alive(h))
    return {
        "running": running,
        "started_at": h.started_at if h else None,
        "files": [str(p) for p in h.file_paths] if h else [],
    }
```

### `api/routes/stats.py` (~80 lines)

| Method | Path | Returns |
|---|---|---|
| GET | `/stats/models` | list of 13 `ModelSpec` dicts from `eval.monte_carlo.MODELS` |
| GET | `/stats/accuracy` | `{model_name: [{chunk_id, accuracy, ts}, ...]}` for all 13 models |
| GET | `/stats/last_generations` | `{model_name: "last generated text"}` for all 13 models |

```python
from fastapi import APIRouter
from api import state
from eval.monte_carlo import MODELS
from dataclasses import asdict

router = APIRouter()

@router.get("/models")
async def models():
    return [asdict(m) for m in MODELS]

@router.get("/accuracy")
async def accuracy():
    return state.store.get_accuracy_history_all()  # returns dict keyed by model name

@router.get("/last_generations")
async def last_generations():
    return state.store.get_last_generations()
```

`Store.get_accuracy_history_all` and `get_last_generations` live in Track 0 — if they don't exist yet, this spec does **not** empower you to add them. File a note and use `get_accuracy_history(model_name)` in a loop as a fallback.

### `api/routes/generate.py` (~100 lines)

Single endpoint: `POST /generate` with body `{"n_words": int, "augment": bool}`.

Runs one generation for each of the 13 models and returns:

```json
{
  "char_3gram": {"text": "...", "real_pct": 0.42, "words": [{"w":"the","real":true}, ...]},
  ...
}
```

If `augment=true`, pipe each output through an autocorrect step (keep it simple: for any word flagged fake by the validator, replace with the closest SCOWL word via Levenshtein; if no candidate within distance 2, leave it). Document this clearly — it is a demo feature, not production spell-correction.

Reuse `MonteCarloEvaluator._generate_sample` and `_score_sample` — do not reimplement generation here.

### `api/routes/db.py` (~80 lines)

Read-only DB inspection for the DB page:

| Method | Path | Query | Returns |
|---|---|---|---|
| GET | `/db/tables` | — | list of table names |
| GET | `/db/table/{name}` | `?limit=100&offset=0` | rows (list of dicts) |
| GET | `/db/row_counts` | — | `{table_name: count}` |
| POST | `/db/reset` | — | `{"status": "reset"}` — drops and recreates all tables via `store.reset()`; also deletes `model/checkpoints/*.pt` |

Whitelist the tables that can be queried — do **not** accept arbitrary SQL. Allowed tables:

```python
ALLOWED_TABLES = {
    "chunks", "char_ngrams", "word_ngrams", "token_ngrams",
    "model_accuracy", "nn_checkpoints", "last_generations"
}
```

Reject any other name with 404.

### `api/routes/ws.py` (~100 lines)

The critical piece. Fan-out from worker's `progress_queue` (a `multiprocessing.Queue`) to every connected WebSocket client.

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio, queue
from api import state

router = APIRouter()

@router.websocket("/ws/progress")
async def ws_progress(websocket: WebSocket):
    await websocket.accept()
    async with state.ws_clients_lock:
        state.ws_clients.add(websocket)
    try:
        while True:
            # Keep connection alive; actual pushes happen in relay_progress
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        async with state.ws_clients_lock:
            state.ws_clients.discard(websocket)

async def relay_progress():
    """Single background task: drain the worker's mp.Queue and broadcast to all WS clients."""
    while True:
        h = state.worker_handle
        if not h:
            await asyncio.sleep(0.1)
            continue
        try:
            # non-blocking drain, up to 50 events per tick
            for _ in range(50):
                event = h.progress_queue.get_nowait()
                await _broadcast(event)
        except queue.Empty:
            pass
        await asyncio.sleep(0.03)   # 30ms — within the 20-50ms budget from Track D1

async def _broadcast(event: dict):
    async with state.ws_clients_lock:
        dead = []
        for ws in state.ws_clients:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            state.ws_clients.discard(ws)
```

**Why a single relay task instead of per-client pollers:** the `mp.Queue` has exactly one consumer — the backend process. If every WS connection tried to `get_nowait()` the same queue, events would be stolen. One drainer fans out to N clients.

---

## Testing

Each route file gets a matching test file using `fastapi.testclient.TestClient` and `httpx.AsyncClient` for WebSocket tests.

### `tests/test_api_ingest.py`

- POST `/ingest/upload` with a small `.txt` file — returns 200, spawns worker
- POST `/ingest/upload` while worker running — returns 409
- POST `/ingest/pause` with no worker — returns 400
- GET `/ingest/status` — returns `{running: false, ...}` before any upload

### `tests/test_api_stats.py`

- `/stats/models` returns exactly 13 entries
- After one accepted chunk, `/stats/accuracy` contains 13 keys each with one data point

### `tests/test_api_generate.py`

- POST `/generate` with `n_words=20` returns 13 model outputs
- `augment=true` path does not crash (content correctness is a manual check)

### `tests/test_api_db.py`

- `/db/tables` lists the expected 7 tables
- `/db/table/chunks?limit=10` returns up to 10 rows
- `/db/table/evil_table` returns 404
- `/db/reset` clears row counts back to 0

### `tests/test_api_ws.py`

- Open a WS connection, upload a small file, receive at least one `chunk_start` event within 30s
- Disconnect during ingest — server does not crash, worker continues
- Two simultaneous WS clients both receive the same events (fan-out works)

---

## Acceptance criteria

- [ ] `pytest tests/test_api_*.py` all green
- [ ] `uvicorn api.main:app --reload` starts without errors; `/docs` lists all routes
- [ ] Manual: upload → see events stream in `ws/progress` via `websocat` or browser devtools
- [ ] No global state leaks between requests when tests run in sequence (fixture resets `state.store`, `state.worker_handle`, `state.ws_clients`)
- [ ] CORS configured for `http://localhost:5173` (frontend dev server)

---

## Pitfalls

- **Never `await` on `mp.Queue.get()` directly** — it blocks. Use `get_nowait()` inside the asyncio relay loop.
- **The relay task must be created once**, not per-upload. Store the task in `state.progress_relay_task` and check `.done()` before recreating.
- **`UploadFile.read()` reads the entire file into memory.** Files are capped at a few MB by design; no streaming needed, but document the cap.
- **Do not put business logic in routes.** If a route is longer than ~40 lines, you're probably reimplementing something that belongs in `store`, `eval`, or the worker.
- **`/db/reset` must also kill any running worker**, or the worker will keep writing to the recreated DB with stale connections.
- **WebSocket send is not thread-safe across tasks.** The broadcast loop must be the only coroutine calling `send_json` per client — do not fan out to per-client tasks.
- **The 30ms relay tick is a budget, not a target.** If events back up (progress queue hits `maxsize=1000`), the worker drops them — that's by design from Track D1.

---

## Model assignment

**`claude-opus-4-6`.** Most routes are thin, but the WebSocket fan-out is the kind of pattern where plausible-looking code is wrong in production: the single-drainer invariant on the `mp.Queue`, the async lock around the client set, recreating the relay task only when `.done()`, and making sure `pause_worker` is invoked via `run_in_executor` rather than directly from the async handler. Sonnet can write a FastAPI app in its sleep but is more likely to miss one of these concurrency constraints. Opus 4.6 hits the sweet spot.
