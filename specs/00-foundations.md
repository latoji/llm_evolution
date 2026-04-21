# Track 0 — Foundations

**Model**: `claude-sonnet-4-6`
**Status in plan**: `AVAILABLE` on start
**Dependencies**: none

## Scope

Build the three foundation modules that every later track depends on: the Word Verification Module (WVM), the DuckDB persistence layer, and the Pydantic API contracts. These must be fully tested and stable before any other track begins — bugs here propagate into every model and page.

## Downstream consumers

- **Track A** imports `db.store` for n-gram reads/writes and uses WVM for word-mode tokenization
- **Track B1, B2** import `db.store` for checkpoint metadata
- **Track C** imports WVM for real-word scoring and `db.store` for accuracy writes
- **Track D1/D2** import all three modules
- **Track E** reads `api/contracts.py` as the source of truth for TypeScript types

## Files owned

Create these files only. Do not modify anything outside this list.

```
wvm/__init__.py
wvm/validator.py
wvm/scowl_70.txt                  (downloaded, not written)
db/__init__.py
db/schema.py
db/store.py
api/__init__.py
api/contracts.py
tests/test_wvm.py
tests/test_db_store.py
tests/test_contracts.py
```

---

## Implementation

### `wvm/validator.py` (~120 lines)

**Purpose**: classify whether a token is a real English word; return structured results.

**Public API:**

```python
from pathlib import Path
from typing import NamedTuple

PUNCTUATION_BOUNDARY = set('.,!?;:"\'()[]{}—…"`')

class WordResult(NamedTuple):
    raw: str          # original token as it appeared
    word: str         # stripped + lowercased version used for lookup
    is_real: bool     # True if word is in SCOWL set

class Validator:
    def __init__(self, wordlist_path: Path = Path("wvm/scowl_70.txt")) -> None:
        """Load SCOWL wordlist into memory as a set."""

    def tokenize(self, text: str) -> list[str]:
        """Split on whitespace; strip boundary punctuation; preserve internal apostrophes."""

    def validate(self, text: str) -> tuple[list[WordResult], float]:
        """Return per-word results and overall real-word percentage (0.0 - 1.0).
        Empty input returns ([], 0.0)."""

    def suggest(self, word: str, n: int = 1) -> list[str]:
        """Return up to n nearest valid words via difflib.get_close_matches. Used by Generation page auto-correct."""
```

**Tokenization rule (must match exactly):**
1. Split on any whitespace
2. For each token, strip leading chars in `PUNCTUATION_BOUNDARY` until a non-punctuation char or end
3. Strip trailing chars in `PUNCTUATION_BOUNDARY` until a non-punctuation char or end
4. Lowercase
5. If resulting string is empty, skip (do not return in list)
6. Internal apostrophes (e.g. `it's`, `don't`) survive step 2 and 3 because they are mid-token

**Wordlist file:**
Download SCOWL size-70 English word list. One word per line, UTF-8. Source: http://wordlist.aspell.net/. The agent must not fabricate this file; if it does not exist, fail with a clear error message pointing to the download location.

### `db/schema.py` (~150 lines)

**Purpose**: define all DuckDB table DDL; provide a single `create_all(conn)` entry point.

**Public API:**

```python
import duckdb
from pathlib import Path

DB_PATH = Path("db/llm_evolution.duckdb")

SCHEMA_VERSION = 1

TABLES = {
    "corpus_chunks": """CREATE TABLE IF NOT EXISTS corpus_chunks (...)""",
    "char_ngrams":   """CREATE TABLE IF NOT EXISTS char_ngrams  (...)""",
    "word_ngrams":   """CREATE TABLE IF NOT EXISTS word_ngrams  (...)""",
    "token_ngrams":  """CREATE TABLE IF NOT EXISTS token_ngrams (...)""",
    "vocabulary":    """CREATE TABLE IF NOT EXISTS vocabulary (...)""",
    "model_accuracy":"""CREATE TABLE IF NOT EXISTS model_accuracy (...)""",
    "nn_checkpoints":"""CREATE TABLE IF NOT EXISTS nn_checkpoints (...)""",
    "schema_meta":   """CREATE TABLE IF NOT EXISTS schema_meta (version INTEGER PRIMARY KEY)""",
}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_char_ngrams_n_context ON char_ngrams(n, context)",
    "CREATE INDEX IF NOT EXISTS idx_word_ngrams_n_context ON word_ngrams(n, context)",
    "CREATE INDEX IF NOT EXISTS idx_token_ngrams_n_context ON token_ngrams(n, context)",
    "CREATE INDEX IF NOT EXISTS idx_model_accuracy_model ON model_accuracy(model_name, chunk_id)",
]

def create_all(conn: duckdb.DuckDBPyConnection) -> None:
    """Apply schema. Idempotent."""

def reset_all(conn: duckdb.DuckDBPyConnection) -> None:
    """Drop and recreate all tables. Called by POST /db/reset."""
```

**Exact table definitions** — use the schema in `SPEC.md` section 7 verbatim. Do not add or remove columns.

### `db/store.py` (~250 lines)

**Purpose**: typed helpers for inserts, queries, transactions. All DB access in the project goes through this module.

**Public API:**

```python
import duckdb
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator

class Store:
    def __init__(self, db_path: Path = Path("db/llm_evolution.duckdb")) -> None:
        """Open connection, create schema if absent."""

    @contextmanager
    def transaction(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """BEGIN; yield conn; COMMIT on success, ROLLBACK on exception."""

    # --- Chunk tracking ---
    def insert_chunk(self, filename: str, chunk_index: int, text_hash: str,
                     char_count: int, accuracy_before: dict) -> int: ...
    def mark_chunk_accepted(self, chunk_id: int, accuracy_after: dict) -> None: ...
    def mark_chunk_rejected(self, chunk_id: int, reason: str) -> None: ...

    # --- N-gram writes ---
    def upsert_ngrams(self, family: str, n: int, rows: list[tuple[str, str, int]]) -> None:
        """family: 'char' | 'word' | 'bpe'. rows: list of (context, next_item, count).
        Upsert semantics: existing (family, n, context, next_item) rows get count += delta."""

    # --- N-gram reads ---
    def get_ngrams(self, family: str, n: int, context: str | None = None,
                   limit: int = 100, offset: int = 0) -> list[dict]:
        """Paginated fetch for DB viewer page."""

    def get_distribution(self, family: str, n: int, context: str) -> dict[str, int]:
        """{next_item: count} for a given context. Used by Markov generators."""

    # --- Accuracy tracking ---
    def insert_accuracy(self, model_name: str, chunk_id: int,
                        accuracy: float, perplexity: float | None) -> None: ...
    def get_accuracy_history(self, model_name: str | None = None) -> list[dict]: ...
    def get_latest_accuracy(self) -> dict[str, float]:
        """{model_name: accuracy} for the most recent chunk."""

    # --- Neural checkpoints ---
    def insert_checkpoint(self, model_name: str, chunk_id: int,
                          filepath: Path, val_loss: float) -> None: ...
    def get_latest_checkpoint(self, model_name: str) -> dict | None: ...

    # --- Vocabulary ---
    def upsert_vocabulary(self, entries: list[tuple[int, str, str]]) -> None: ...
    def get_vocabulary(self, source: str, limit: int = 100, offset: int = 0) -> list[dict]: ...

    # --- Reset ---
    def reset_all(self) -> None:
        """Drop all tables, delete all checkpoint files, wipe DB file."""
```

**Transaction semantics:** `transaction()` context manager must `ROLLBACK` on any exception from the yielded block, including `KeyboardInterrupt` and `SystemExit`. This is the primary mechanism for chunk rejection in the Ingest Worker.

**Batching:** `upsert_ngrams` will be called with rows in the tens of thousands. Use DuckDB's bulk insert via `executemany` or `register` + `INSERT FROM`. Do not call `INSERT` in a Python loop per row.

### `api/contracts.py` (~200 lines)

**Purpose**: Pydantic v2 models for every HTTP request/response and every WebSocket message. This file is also the source of truth for frontend TypeScript types (Track E transcribes it).

**HTTP models:**

```python
from pydantic import BaseModel, Field
from typing import Literal

class IngestUploadResponse(BaseModel):
    accepted_files: list[str]
    rejected_files: list[dict]  # [{filename, reason}]
    total_chunks: int

class IngestStatusResponse(BaseModel):
    state: Literal["idle", "running", "paused", "complete"]
    current_chunk: int | None
    total_chunks: int | None
    chunks_accepted: int
    chunks_rejected: int

class GenerateRequest(BaseModel):
    word_count: int = Field(ge=20, le=500)
    auto_correct: bool = False

class ModelOutput(BaseModel):
    model_name: str
    raw_text: str
    corrected_text: str | None
    word_results: list[tuple[str, bool]]   # (word, is_real)
    real_word_pct: float

class GenerateResponse(BaseModel):
    outputs: list[ModelOutput]            # 13 entries

class AccuracyPoint(BaseModel):
    chunk_id: int
    accuracy: float
    perplexity: float | None
    timestamp: str

class AccuracyHistoryResponse(BaseModel):
    models: dict[str, list[AccuracyPoint]]  # {model_name: points[]}

class NGramRow(BaseModel):
    context: str
    next_item: str
    count: int
    probability: float

class NGramPageResponse(BaseModel):
    rows: list[NGramRow]
    total: int
    page: int
    page_size: int

class VocabularyRow(BaseModel):
    token_id: int
    token: str
    source: str
    frequency: int

# ... (full list: all endpoints in SPEC.md section 8 get a request/response pair)
```

**WebSocket models:**

```python
class WSChunkStart(BaseModel):
    type: Literal["chunk_start"] = "chunk_start"
    chunk_index: int
    total_chunks: int
    operation: str

class WSChunkProgress(BaseModel):
    type: Literal["chunk_progress"] = "chunk_progress"
    operation: str
    pct: int  # 0-100

class WSMCToken(BaseModel):
    type: Literal["mc_token"] = "mc_token"
    model: str
    token: str
    run: int  # which of the 50 runs

class WSMCComplete(BaseModel):
    type: Literal["mc_complete"] = "mc_complete"
    model: str
    accuracy: float
    run: int

class WSChunkDone(BaseModel):
    type: Literal["chunk_done"] = "chunk_done"
    chunk_index: int
    status: Literal["accepted", "rejected"]
    accuracy_delta: dict[str, float]
    reason: str | None = None

class WSIngestComplete(BaseModel):
    type: Literal["ingest_complete"] = "ingest_complete"
    chunks_accepted: int
    chunks_rejected: int

# Discriminated union
WSMessage = WSChunkStart | WSChunkProgress | WSMCToken | WSMCComplete | WSChunkDone | WSIngestComplete
```

Every message serializes to JSON via `model.model_dump_json()`.

---

## Testing

### `tests/test_wvm.py`

Must cover:
- `validate("hello world")` → 2 words, both real, 100%
- `validate("hello wrld")` → 2 words, 1 real, 50%
- `validate("")` → empty list, 0.0
- `tokenize("Hello, it's (a) test!")` → `["hello", "it's", "a", "test"]`
- `tokenize("—word...")` → `["word"]`
- Internal apostrophes preserved: `"don't"` → `"don't"` not `"dont"`
- `suggest("teh")` returns `["the", ...]`
- Missing wordlist file raises `FileNotFoundError` with actionable message

### `tests/test_db_store.py`

Must cover:
- `create_all` is idempotent (call twice, no errors)
- `insert_chunk` returns auto-increment id
- `upsert_ngrams` with duplicate rows adds counts correctly
- `transaction()` rolls back on exception (insert inside transaction, raise, verify row absent)
- `get_distribution("char", 2, "th")` returns correct `{next: count}` mapping
- `reset_all` deletes all rows and recreates schema
- Concurrent access: opening a second connection in a subprocess and writing works without corruption

### `tests/test_contracts.py`

Must cover:
- Every Pydantic model round-trips via `model_dump_json()` and `model_validate_json()`
- `GenerateRequest(word_count=10)` raises `ValidationError` (below min)
- `GenerateRequest(word_count=501)` raises `ValidationError` (above max)
- WSMessage discriminated union correctly parses each message type

---

## Acceptance criteria

- [ ] `pytest tests/test_wvm.py tests/test_db_store.py tests/test_contracts.py` — all green
- [ ] `wvm/scowl_70.txt` exists and contains ≥ 150,000 words
- [ ] Running `python -c "from db.store import Store; Store().reset_all()"` creates a fresh `.duckdb` file with all 8 tables
- [ ] All three modules importable from project root without side effects
- [ ] Zero `mypy` errors on files owned by this track (if mypy is used)

---

## Pitfalls

- **Do not use `pyenchant`.** It's been considered and rejected (see SPEC section 6). Load SCOWL as a plain Python `set`.
- **DuckDB is file-based but not concurrency-safe across processes by default.** The ingest worker runs in a separate process. Use a single `Store` instance per process; pass file paths (not connections) across the multiprocessing boundary. The worker opens its own connection.
- **Pydantic v2, not v1.** Use `model_dump_json()`, `model_validate_json()`, `Field(ge=..., le=...)`.
- **Do not add a new table or column** without updating `SPEC.md` section 6 first and surfacing to the user.
- **SCOWL wordlist** must be downloaded from http://wordlist.aspell.net/. Do not fabricate. If it's missing at runtime, fail fast with a message telling the user where to get it.

---

## Model assignment

**Sonnet 4.6.** Well-defined schema, mechanical implementation, no subtle state management. Opus is not needed here.
