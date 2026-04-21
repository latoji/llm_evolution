# Track C — Monte Carlo Accuracy Evaluator

**Model**: `claude-sonnet-4-6`
**Dependencies**: Tracks A, B1, B2 complete

## Scope

Build the evaluator that measures each of the 13 models' accuracy after every accepted chunk. 50 text generations per model, scored against SCOWL via WVM. Emits progress callbacks so the WebSocket layer can stream live updates.

## Upstream dependencies

- All 11 Markov models from Track A
- `FeedforwardTrainer` from Track B1
- `TransformerTrainer` from Track B2
- `wvm.validator.Validator` from Track 0
- `db.store.Store` from Track 0

## Downstream consumers

- **Track D1** (Ingest Worker) calls `evaluate_all(chunk_id, progress_cb)` after each chunk's training step

## Files owned

```
eval/monte_carlo.py
tests/test_monte_carlo.py
```

---

## Implementation

### `eval/monte_carlo.py` (~300 lines)

**Model registry:**

A single source of truth for the 13 models and how to instantiate each generator:

```python
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class ModelSpec:
    name: str                            # e.g. "char_3gram", "feedforward", "transformer"
    family: str                          # "char" | "word" | "bpe" | "neural"
    order: int | None                    # for Markov models only
    display_order: int                   # 1..13, for Stats page sort

MODELS: list[ModelSpec] = [
    ModelSpec("char_1gram", "char", 1, 1),
    ModelSpec("char_2gram", "char", 2, 2),
    ModelSpec("char_3gram", "char", 3, 3),
    ModelSpec("char_4gram", "char", 4, 4),
    ModelSpec("char_5gram", "char", 5, 5),
    ModelSpec("word_1gram", "word", 1, 6),
    ModelSpec("word_2gram", "word", 2, 7),
    ModelSpec("word_3gram", "word", 3, 8),
    ModelSpec("bpe_1gram",  "bpe",  1, 9),
    ModelSpec("bpe_2gram",  "bpe",  2, 10),
    ModelSpec("bpe_3gram",  "bpe",  3, 11),
    ModelSpec("feedforward","neural", None, 12),
    ModelSpec("transformer","neural", None, 13),
]
```

**Public API:**

```python
from typing import Protocol

class ProgressCallback(Protocol):
    def __call__(self, event_type: str, payload: dict) -> None: ...
    # event_type in: "mc_token", "mc_complete", "mc_model_start"

RUNS_PER_MODEL = 50
WORDS_PER_RUN = 100

class MonteCarloEvaluator:
    def __init__(self, store: Store, validator: Validator,
                 feedforward: FeedforwardTrainer, transformer: TransformerTrainer) -> None:
        """Caller passes in already-trained model instances."""

    def evaluate_all(self, chunk_id: int,
                     on_progress: ProgressCallback | None = None) -> dict[str, float]:
        """Run 50 generations × 13 models. Insert accuracy rows into DuckDB.
        Return {model_name: mean_accuracy}.

        Emits progress events via on_progress:
          on_progress("mc_model_start", {"model": "char_3gram"})
          on_progress("mc_token", {"model": ..., "token": "th", "run": 12})
          on_progress("mc_complete", {"model": ..., "accuracy": 0.47, "run": 12})
        """

    def _generate_sample(self, spec: ModelSpec, rng: random.Random) -> str:
        """Generate WORDS_PER_RUN words from this model. Returns generated text."""

    def _score_sample(self, text: str) -> float:
        """Return real-word % via validator.validate."""
```

**Parallelism:**

- The 11 Markov models run on CPU; the 2 neural models run on the trainer's device (CPU or CUDA).
- Use `concurrent.futures.ProcessPoolExecutor` for Markov models with `max_workers = os.cpu_count() - 2` (leave cores for FastAPI and frontend).
- Neural models run sequentially on the main process (they share the trainer + GPU, parallelizing them causes contention).
- Each ProcessPool worker needs its own `Store` connection (DuckDB is not multi-process safe across the same connection).

Recommended execution pattern:

```python
def evaluate_all(self, chunk_id, on_progress=None):
    results = {}

    # Markov models in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as pool:
        futures = {pool.submit(_run_markov_worker, spec, seed): spec
                   for spec in MODELS if spec.family != "neural"
                   for seed in range(RUNS_PER_MODEL)}
        # collect and score each generated sample

    # Neural models sequentially
    for spec in (s for s in MODELS if s.family == "neural"):
        scores = []
        for run in range(RUNS_PER_MODEL):
            text = self._generate_sample(spec, random.Random(run))
            if on_progress:
                on_progress("mc_complete", {...})
            scores.append(self._score_sample(text))
        results[spec.name] = sum(scores) / len(scores)

    # Persist to DB
    for name, acc in results.items():
        self.store.insert_accuracy(name, chunk_id, acc, perplexity=None)
    return results
```

**Token streaming (for live UI):**

During neural model generation, emit `mc_token` events token-by-token by hooking the trainer's `generate()` method. If `generate()` does not yield tokens, wrap it or extend it to yield. This is optional (the UI will work with just `mc_complete` events) but makes the Ingest page dramatically more engaging.

**RNG seeding:**

Every run gets `random.Random(run_index)` so evaluations are deterministic for testing but varied across runs. Seeding with a fixed seed is essential — without it, accuracy numbers fluctuate and the Stats graph looks noisy.

---

## Testing

### `tests/test_monte_carlo.py`

- `MODELS` list has exactly 13 entries with unique names and display_orders 1–13
- `evaluate_all` called on a trained `Store` returns a dict with 13 keys
- Mock trainers can be substituted; the evaluator does not directly call `torch.*` methods (they're encapsulated behind the trainer interfaces)
- `_score_sample("the cat sat")` returns ~1.0 (all real words)
- `_score_sample("qxzp wxyz")` returns 0.0 (no real words)
- `evaluate_all` with a recording callback captures the expected event sequence (13 × [model_start, 50 × complete])
- Accuracy rows are inserted into DuckDB: `store.get_accuracy_history()` returns 13 rows after one call

### Performance test

Not required to pass but should be measured:
- On a 10 KB corpus with trained models: `evaluate_all` completes in under 90 seconds on CPU
- On GPU: under 45 seconds

---

## Acceptance criteria

- [ ] `pytest tests/test_monte_carlo.py` all green
- [ ] Exactly 13 models in `MODELS` registry
- [ ] Progress callback fires with correct event types
- [ ] After execution, `model_accuracy` table contains 13 new rows with the given `chunk_id`

---

## Pitfalls

- **Do not parallelize neural models across processes.** Both would try to use the GPU, causing OOM or serialization failures. Run them sequentially on the main process.
- **Each ProcessPool worker needs its own Store instance.** Do not share a DuckDB connection across processes. Pass the db_path, not the connection.
- **Seed every run explicitly.** Without seeding, accuracy is noisy.
- **Markov models can be instantiated in the worker** — they are cheap. Neural models must be pre-loaded and passed in (loading a .pt file is slow).
- **If `generate()` for a Markov model returns an empty string** (e.g., unseen context with no backoff), count it as 0% accuracy, not a crash.
- **Perplexity is `None` for non-neural models.** Do not try to compute it for Markov chains here.

---

## Model assignment

**`claude-opus-4-6`.** The code shape is orchestration rather than novel algorithms, but the correctness bar is subtle: ProcessPool workers must each open their own `Store`, RNG seeding must be deterministic per run, neural models must stay sequential on the main process to avoid GPU contention, and the progress-callback protocol has to interleave cleanly with the pool's completion order. Sonnet tends to drop one of these constraints. Opus 4.6 is the right tier — stronger reasoning than Sonnet without reserving the Opus 4.7 budget for tracks that need it more (B2, D1).
