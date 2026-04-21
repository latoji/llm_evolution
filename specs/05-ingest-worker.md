# Track D1 — Ingest Worker (Multiprocessing Core)

**Model**: `claude-opus-4-7` — this is the hardest module in the project.
**Dependencies**: Track C complete (A, B1, B2 transitively complete)

## Scope

Implement the orchestration layer that runs in a separate `multiprocessing.Process` to keep the FastAPI event loop free. Handles chunking, training, Monte Carlo evaluation, rollback, pause signalling, and progress event emission.

This is the single most complex module in the project. The correctness requirements are:

1. **Atomicity**: when a chunk is rejected, every side effect (DB rows, neural checkpoints, BPE vocabulary updates) must roll back
2. **Responsiveness**: the main FastAPI process must never block on this worker
3. **Clean pause**: pause button halts between chunks, not mid-chunk, and leaves the system in a consistent state
4. **Progress fidelity**: every meaningful operation emits a WebSocket-ready event

## Upstream dependencies

- All modules from Tracks 0, A, B1, B2, C

## Downstream consumers

- Track D2 (`api/routes/ingest.py`) spawns the worker, reads the progress queue, sends control signals

## Files owned

```
api/ingest_worker.py
api/worker_types.py             (small file with message types if not already in contracts)
tests/test_ingest_worker.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ Parent process (FastAPI + asyncio event loop)                        │
│                                                                      │
│   ┌──────────────┐       ┌─────────────────┐                         │
│   │ HTTP route   │──────>│ start_worker()  │                         │
│   │ /ingest/up.. │       └────────┬────────┘                         │
│   └──────────────┘                │ spawns                           │
│                                   ▼                                  │
│                          ┌──────────────────────────────────────┐    │
│                          │  IngestWorker  (multiprocessing.Proc)│    │
│                          │  - pause_event (mp.Event)             │    │
│                          │  - progress_queue (mp.Queue)          │    │
│                          │  - run() loop                         │    │
│                          └──────────┬───────────────────────────┘    │
│                                     │                                │
│   ┌──────────────┐                  │                                │
│   │ WS /ws/prog  │<─────────────────┘ (asyncio task reads queue)    │
│   └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Files you must NOT modify

- Everything in Tracks 0, A, B1, B2, C
- `api/routes/*` — Track D2's territory (you only own the worker itself)

---

## Implementation

### `api/ingest_worker.py` (~500 lines — substantial)

**Public API (called from the parent process):**

```python
from multiprocessing import Process, Queue, Event
from pathlib import Path
from dataclasses import dataclass

@dataclass
class WorkerHandle:
    process: Process
    progress_queue: Queue
    pause_event: Event
    started_at: float
    file_paths: list[Path]

def start_worker(file_paths: list[Path], db_path: Path = Path("db/llm_evolution.duckdb")) -> WorkerHandle:
    """Spawn a new IngestWorker process. Returns handle the parent uses to monitor and control.
    Does not block."""

def pause_worker(handle: WorkerHandle) -> None:
    """Set the pause_event. Worker will exit after its current chunk completes."""

def is_alive(handle: WorkerHandle) -> bool:
    """Check worker process status."""
```

**Worker entry point (runs in child process):**

```python
def _worker_main(file_paths: list[Path], db_path: Path,
                 progress_queue: Queue, pause_event: Event) -> None:
    """The function passed to multiprocessing.Process.

    Structure:
    1. Open Store, instantiate all 13 model classes
    2. For each file:
       a. Run data/clean.py pre-screening; WVM pre-check
       b. Chunk into 1000-2500 char blocks (whitespace-aligned)
       c. For each chunk:
          - Check pause_event; if set, emit pause_ack and exit
          - Snapshot accuracy_before (from store.get_latest_accuracy)
          - store.insert_chunk(status='processing')
          - Save pre-chunk neural checkpoint paths for rollback
          - Open DB transaction
          - Update all 11 Markov model counts
          - Train feedforward step
          - Train transformer step
          - Call MonteCarloEvaluator.evaluate_all (streams mc_token/mc_complete)
          - Compare new accuracy to snapshot per model
          - If any model dropped: ROLLBACK, delete new checkpoints, restore old
          - Else: COMMIT, mark chunk accepted
    3. Emit ingest_complete event
    """
```

**Progress queue protocol:**

Every event is a JSON-serializable dict matching the `WSMessage` union in `api/contracts.py`:

```python
progress_queue.put({"type": "chunk_start", "chunk_index": 4, "total_chunks": 17, "operation": "..."})
progress_queue.put({"type": "chunk_progress", "operation": "Training transformer", "pct": 62})
progress_queue.put({"type": "mc_token", ...})
progress_queue.put({"type": "chunk_done", "chunk_index": 4, "status": "accepted", ...})
```

The parent process has an asyncio task that polls `progress_queue.get_nowait()` on a tight interval (e.g. every 50ms) and forwards each event to all connected WebSocket clients.

**Chunking logic:**

```python
def chunk_text(text: str, min_size: int = 1000, max_size: int = 2500) -> list[str]:
    """Split text into chunks of 1000-2500 characters, breaking on whitespace.
    Never split mid-word. Paragraphs preferred as break points."""
```

**Rollback strategy:**

A rejection requires undoing THREE kinds of state:

1. **DuckDB rows**: wrapped in `store.transaction()`. Exiting via exception → `ROLLBACK`.
2. **Neural net weights in memory**: before training, save a deep copy of `state_dict()`. If rejected, `load_state_dict(saved)`.
3. **Neural net checkpoint files**: if `train_step` wrote a new `.pt` file, delete it on rejection. The previous `.pt` file stays; the DB `nn_checkpoints` row from this chunk was rolled back in step 1.
4. **BPE vocabulary**: the BPE tokenizer may learn new merges during ingest if called with retrain=True. For simplicity, do not retrain BPE mid-ingest; retrain only on DB reset.

**Pause semantics:**

`pause_event.is_set()` is checked at two points:
- Immediately before starting a new chunk (fine-grained)
- Never inside a chunk (to preserve atomicity)

On pause, emit:
```python
progress_queue.put({"type": "ingest_paused", "chunks_completed": N, "chunks_remaining": M})
```

The worker process exits cleanly after pausing. Resuming is not supported in v1 — user re-uploads remaining files if they want to continue.

**Pre-screening:**

Before chunking a file:
1. Run `data.clean.clean_text(raw)` for Unicode normalisation
2. Run `validator.validate(cleaned)` → if real-word % < 0.70, reject the entire file
3. Emit `{"type": "file_rejected", "filename": ..., "reason": "low real-word %"}`

---

## Testing

### `tests/test_ingest_worker.py`

This test file is critical. Cover:

1. **Happy path**: start worker on a small clean corpus, consume all progress events, verify:
   - All chunks reach status='accepted'
   - DB contains expected n-gram rows
   - `ingest_complete` event emitted exactly once

2. **Rollback on forced accuracy drop**: monkeypatch `MonteCarloEvaluator.evaluate_all` to return scores lower than baseline. Run a chunk. Verify:
   - Chunk status = 'rejected'
   - No new rows in `char_ngrams`, `word_ngrams`, `token_ngrams` compared to pre-chunk counts
   - Neural models restored to pre-chunk weights
   - Any `.pt` file written during the aborted chunk is deleted

3. **Pause**: start worker on multi-chunk corpus, send pause after 1 second, verify:
   - Worker exits within 60 seconds (after current chunk finishes)
   - `ingest_paused` event emitted
   - Chunks completed before pause are accepted; remaining chunks untouched

4. **File pre-screening**: feed a file of random binary noise; verify `file_rejected` event and no chunks processed

5. **Process isolation**: force-kill the worker process mid-run; verify parent process does not crash; verify the DB is still openable from the parent (no corruption — in-progress chunk was inside a transaction that got rolled back implicitly)

6. **Progress queue does not drop events**: instrument the queue to count events. Run a 3-chunk ingest; expect at least 3 × (1 chunk_start + 1 chunk_done) + 1 ingest_complete events.

Use `multiprocessing.Process` for real process isolation in tests, not mocks. Tests may be slower (30–60s each) — mark them with `@pytest.mark.slow` and exclude from the default CI run.

---

## Acceptance criteria

- [ ] `pytest tests/test_ingest_worker.py -m "not slow"` — fast tests pass
- [ ] `pytest tests/test_ingest_worker.py -m slow` — integration tests pass
- [ ] Worker spawns in under 2 seconds on CPU-only systems
- [ ] Pause signal honored within 60 seconds (time to finish current chunk)
- [ ] Rollback leaves DB indistinguishable from pre-chunk state (verify via row-count asserts)
- [ ] No zombie processes after test suite completes

---

## Pitfalls (HIGH ATTENTION)

- **DuckDB connections are NOT shareable across processes.** The worker must open its own connection using the `db_path`. Do not pass the `Store` instance from the parent.
- **PyTorch models cannot be pickled cleanly across processes when they have CUDA tensors.** The worker must instantiate its own trainers in `_worker_main`. Do not try to pass trainer objects to `Process(args=...)`.
- **`multiprocessing.Queue` silently blocks** if the reader is too slow. The parent's asyncio reader must poll frequently (every 20–50ms) to avoid backpressure.
- **`multiprocessing.Event` is NOT thread-safe to set from an async context** in some Python versions. Use `loop.run_in_executor` to set it if called from an async route handler.
- **Emitting too many `mc_token` events floods the queue.** Rate-limit: emit one token event per 5 tokens generated, or batch into `mc_tokens` (plural). The UI can interpolate.
- **Rollback of neural net in-memory weights requires `deepcopy(state_dict())`**, not a shallow copy. Tensors are reference types.
- **If the worker crashes unexpectedly**, the DB transaction must be opened with `BEGIN` that auto-rolls back when the connection closes. Verify by forcefully killing the worker mid-chunk in a test.
- **The worker holds GPU memory for its entire lifetime.** If the user pauses and starts a new worker, the old GPU memory must be released. Use `torch.cuda.empty_cache()` at process exit; better yet, the process dying releases it automatically.
- **Do not use `threading`**, only `multiprocessing`. The Python GIL would negate the whole point.
- **Do not use `asyncio` inside the worker.** The worker is pure synchronous Python; async is only in the parent FastAPI process.
- **The progress queue has a size limit**. If unbounded, memory balloons. Use `Queue(maxsize=1000)`; if `put_nowait` raises `queue.Full`, drop the event (prefer losing a progress update to blocking the worker).

---

## Model assignment

**Opus 4.7.** Mandatory. The correctness requirements around multiprocessing, rollback atomicity, and pause semantics are exactly the class of problem where Sonnet tends to produce plausible-but-subtly-wrong code.
