# LLM Evolution App — Project Specification

*Last updated: 2026-04-20*

---

## 1. Purpose

An educational app that shows how language models grow smarter as training data increases and model architecture advances. The user watches data flow in, watches 13 models update in real time, and can compare their outputs side by side — from near-random character noise at the bottom to coherent attention-based text at the top.

The app demonstrates, in a single session:
- Why more context (higher n-gram order) improves output
- Why vocabulary choice (characters vs. words vs. BPE subwords) matters
- Why statistical Markov chains hit a ceiling that neural networks break through
- Why attention unlocks coherence that feedforward networks cannot achieve
- Why scale and data volume are what turn a correct architecture into a capable one

---

## 2. How Our Model 13 Relates to GPT-2

Model 13 (Neural Net + Attention) belongs to the same architectural family as GPT-2: a decoder-only causal Transformer with BPE tokenisation, next-token prediction, multi-head self-attention, position-wise feed-forward sublayers, residual connections, and layer normalisation. Every core concept from *Attention Is All You Need* (Vaswani et al., 2017) is present.

The difference is scale — and scale is everything:

| Property | Our Model 13 | GPT-2 Small |
|---|---|---|
| Transformer blocks | 2 | 12 |
| Model dimension (d_model) | 128 | 768 |
| Attention heads | 4 | 12 |
| Context window | 64 tokens | 1,024 tokens |
| Vocabulary size | 8,000 | 50,257 |
| Parameters | ~6.8M | ~117M |
| Training data | User corpus (MBs) | WebText (~40 GB) |
| Positional encoding | Sinusoidal (fixed) | Learned embeddings |
| Layer norm position | Post-norm (default) | Pre-norm |

GPT-2's 1,024-token context window is what allows it to maintain subject-verb agreement across a paragraph, sustain a narrative thread, and produce text that feels purposeful. Our 64-token window means the model learns short-range patterns well but loses the thread of a sentence beyond a few words. That gap — locally plausible vs. globally coherent — is the exact lesson the app should surface.

**How this should appear in the UI:**

- **Stats page**: each model card includes a one-line "architecture note" below the model name. For Model 13: *"Same architecture family as GPT-2. 17× smaller in every dimension — demonstrates why scale matters."*
- **Generation page**: a static info panel above the 13 output cards explains the ladder: Markov chains → feedforward neural net → attention → and then what GPT-2 and beyond represent. The panel ends with: *"The gap between Model 13 and GPT-2 is not a flaw in this app — it is the point."*
- **Stats page sort order**: Models 12 and 13 should display their accuracy alongside a note that real-word % undersells the attention model's advantage; coherence (sentence structure, grammar) is the true gain, which the side-by-side Generation page outputs make visible.

There is no user prompt input, no manual tuning, no accounts, no cloud. The system is self-contained, autonomous, and local.

---

## 3. Environment & Hard Constraints

| Property | Value |
|---|---|
| Machine | Lenovo ThinkBook 15p IMH |
| OS | Windows 11 + WSL2 Ubuntu |
| CPU | Intel Core i7-10750H (6 cores / 12 threads) |
| GPU | NVIDIA GTX 1650 Ti — **4 GB VRAM** |
| PyTorch device | `cuda` via CUDA on WSL2; auto-fallback to `cpu` |
| All files | Local only. No cloud, no GitHub. |

**CUDA setup prerequisite** (must be done before building neural modules):
1. Windows: NVIDIA driver ≥ 525
2. WSL2: install CUDA toolkit (`nvidia-cuda-toolkit` or NVIDIA's CUDA repo)
3. Verify: `nvidia-smi` shows GPU inside WSL2
4. Install torch: pytorch.org selector → Linux / pip / CUDA 12.x
5. Validate: `python -c "import torch; print(torch.cuda.is_available())"`

**CPU fallback**: all PyTorch code uses `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`. On CPU, small models (≤7M params, batch 64) train in under a minute per chunk on the i7.

**4 GB VRAM guard**: at startup, if `torch.cuda.get_device_properties(0).total_memory < 3 GB`, automatically halve batch size (64→32) and sequence length (64→32).

---

## 4. Existing Codebase — What to Keep

All modules below are production-quality. Do not rewrite them. Actions are minimal extensions only.

### `tokenizer/bpe.py` — **Keep as-is**
Full from-scratch GPT-2-style BPE tokenizer. GPT-2 pre-tokenization regex; byte→Unicode bijection; iterative merge training; roundtrip-verified encode/decode. Vocab: 256 base bytes + 2 special tokens + learned merges = **8,000 total**. Speed: >10,000 tokens/sec.
Powers models 9–13.

### `tokenizer/train_tokenizer.py` — **Keep, integrate**
Trains on `data/clean/train.txt`. Call from the ingest worker whenever a new corpus is introduced.

### `model/ngram_counter.py` — **Extend**
Streaming counter for n-gram orders 1–4, with document-boundary sentinels and min-count pruning.
**Change**: add `mode` parameter (`char` | `word` | `bpe`); add order 5 for `char` mode.

### `model/smoothing.py` — **Keep as-is**
Modified Kneser-Ney smoothing (Chen & Goodman 1998). Three-level discounts, continuation probabilities, full backoff chain. Powers all 11 Markov models.

### `model/language_model.py` — **Extend**
High-level wrapper: `prob()`, `next_token_distribution()`, `perplexity()`.
**Change**: replace JSON file reads/writes with DuckDB queries via `db/store.py`.

### `generate/sampling.py` — **Keep as-is**
Temperature, top-k, top-p, nucleus sampling. Works for all Markov models unchanged.

### `generate/generator.py` — **Keep as-is**
Markov chain text generator with sliding context window and streaming support.

### `demo/char_ngrams.py` — **Refactor into `model/char_ngram.py`**
Character n-gram model (orders 1–4) with `monte_carlo_walks()`. Move to `model/`, extend to order 5, replace walk display logic with WVM-based real-word % scoring.

### `eval/perplexity.py` — **Keep, expose via API**
Computes perplexity on validation split. Exposed as an API endpoint; shown on Stats page alongside accuracy for neural models.

### `data/clean.py` — **Keep, call from ingest**
Unicode normalisation and short-paragraph filtering. Run on every uploaded `.txt` before chunking.

### `tests/` — **Keep all, extend**
~730 lines of pytest (tokenizer, n-gram counter, smoothing, generator, sampling). All carry forward. New modules each get a new test file.

### `demo/` — **Keep intact, do not modify**
Standalone Flask demo. Kept as a reference and fallback. Does not conflict with the new `api/` entry point.

---

## 5. The 13 Models

Shown on Stats and Generation pages in this order. The visual story: accuracy climbs left to right on every graph as training data grows; higher-numbered models sit higher on the Y-axis.

| # | Name | Input | Predicts | Family |
|---|---|---|---|---|
| 1 | Char 1-gram | — | Next character from frequency table | Character Markov |
| 2 | Char 2-gram | 1 prior char | Next character | Character Markov |
| 3 | Char 3-gram | 2 prior chars | Next character | Character Markov |
| 4 | Char 4-gram | 3 prior chars | Next character | Character Markov |
| 5 | Char 5-gram | 4 prior chars | Next character | Character Markov |
| 6 | Word 1-gram | — | Next word from frequency table | Word Markov |
| 7 | Word 2-gram | 1 prior word | Next word | Word Markov |
| 8 | Word 3-gram | 2 prior words | Next word | Word Markov |
| 9 | BPE Token 1-gram | — | Next token from frequency table | Token Markov |
| 10 | BPE Token 2-gram | 1 prior token | Next token | Token Markov |
| 11 | BPE Token 3-gram | 2 prior tokens | Next token | Token Markov |
| 12 | Neural Net | 64-token context window | Next token (feedforward) | Neural |
| 13 | Neural Net + Attention | 64-token context window | Next token (transformer) | Neural |

All 11 Markov models use KN smoothing from `model/smoothing.py`. Models 6–8 tokenize by whitespace + boundary-punctuation strip (same logic as WVM). Models 9–13 use BPE tokens from `tokenizer/bpe.py`.

### Neural Net Architecture — Model 12 (Feedforward)
File: `model/feedforward.py`
```
Embedding:  8000 × 128  (trainable, ~4 MB)
Linear 1:   128 × 512   → ReLU
Linear 2:   512 × 512   → ReLU
Linear 3:   512 × 128   → ReLU
Output:     128 × 8000  → log-softmax
Total:      ~5.2M params | peak VRAM at batch=64: ~300 MB
```
Training: cross-entropy loss, Adam optimiser, **one forward+backward pass per chunk** (incremental, no full epochs). Checkpoint: `model/checkpoints/feedforward_{chunk_id}.pt`.

### Neural Net Architecture — Model 13 (Transformer)
File: `model/transformer.py`
```
Embedding:  8000 × 128  (trainable) + sinusoidal positional encoding
Block 1:    nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, batch_first=True)
Block 2:    nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, batch_first=True)
Output:     128 × 8000  → log-softmax
Total:      ~6.8M params | peak VRAM at batch=64: ~450 MB
```
Use `torch.nn.TransformerEncoderLayer` with causal mask. Same training cadence as feedforward. Checkpoint: `model/checkpoints/transformer_{chunk_id}.pt`. **Use Opus 4.7 for this file** — VRAM management, positional encoding, and causal masking require careful reasoning.

On app restart: scan `model/checkpoints/` and auto-load the highest-numbered checkpoint for each neural model.

---

## 6. Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Backend | **FastAPI** | Async, native WebSockets, background tasks. Replaces Flask (same decorator style, ~2h migration). Required for UI responsiveness during CPU-heavy ingest. |
| Frontend | **React + Vite + TypeScript** | Multi-page routing without backend re-runs, live WebSocket graph updates, TypeScript catches AI-agent type errors early. |
| Database | **DuckDB** | Embedded (no server), columnar storage for fast n-gram queries, SQL transactions for atomic rollback, single `.duckdb` file. |
| ML | **PyTorch** (`device="cuda"`, fallback `cpu`) | GTX 1650 Ti in WSL2; auto-fallback for CUDA setup failures. |
| Word validation | **SCOWL size-70 as Python `set`** | Zero native dependencies, O(1) lookup, ~200K words. `pyenchant` dropped — `libenchant` causes WSL venv conflicts. |
| Spell-correct | **`difflib.get_close_matches()`** | stdlib, no deps, sufficient for nearest-valid-word lookup on Generation page. |
| Graphs | **Recharts** | Lightweight React charting; easy to wire to WebSocket state; `syncId` aligns 13 charts on shared X-axis. |
| Process isolation | **Python `multiprocessing`** | Ingest worker runs in a separate `Process`, bypassing GIL. FastAPI event loop never blocks. |
| Existing tokenizer | **`tokenizer/bpe.py`** | Already implemented and tested. No replacement needed. |
| Existing smoothing | **`model/smoothing.py`** | Production-quality KN smoothing. No replacement needed. |

**Flask is not replaced — it is sidelined.** The `demo/` folder keeps its Flask server intact. The new `api/` folder is the live backend. No migration work required for `demo/app.py`.

---

## 7. Data Architecture

### DuckDB Schema
File: `db/schema.py`. Database file: `db/llm_evolution.duckdb`.

```sql
corpus_chunks (
  id            INTEGER PRIMARY KEY,
  filename      TEXT,
  chunk_index   INTEGER,
  text_hash     TEXT,
  char_count    INTEGER,
  status        TEXT,        -- 'accepted' | 'rejected' | 'processing'
  accuracy_before JSON,      -- {model_name: score} snapshot before ingest
  accuracy_after  JSON,      -- {model_name: score} snapshot after ingest
  ingested_at   TIMESTAMP
)

char_ngrams  (n INT, context TEXT, next_char  TEXT, count BIGINT)
word_ngrams  (n INT, context TEXT, next_word  TEXT, count BIGINT)
token_ngrams (n INT, context TEXT, next_token TEXT, count BIGINT)

vocabulary (
  token_id  INTEGER PRIMARY KEY,
  token     TEXT,
  source    TEXT   -- 'char' | 'word' | 'bpe'
)

model_accuracy (
  id          INTEGER PRIMARY KEY,
  model_name  TEXT,
  chunk_id    INTEGER REFERENCES corpus_chunks(id),
  accuracy    FLOAT,   -- mean real-word % over 50 MC runs
  perplexity  FLOAT,   -- NULL for Markov models
  timestamp   TIMESTAMP
)

nn_checkpoints (
  id          INTEGER PRIMARY KEY,
  model_name  TEXT,    -- 'feedforward' | 'transformer'
  chunk_id    INTEGER,
  filepath    TEXT,    -- path to .pt file
  val_loss    FLOAT,
  created_at  TIMESTAMP
)
```

### Rollback Pattern
Every chunk ingest is wrapped in a DuckDB transaction:
```python
conn.execute("BEGIN")
# insert n-gram rows, record new accuracy
if any model accuracy dropped vs accuracy_before:
    conn.execute("ROLLBACK")
    restore_nn_checkpoint(pre_chunk_filepath)
    mark chunk status='rejected'
else:
    conn.execute("COMMIT")
    mark chunk status='accepted'
```

### DB Reset
Triggered by the DB Viewer page reset button. Atomically:
1. Drop and recreate all tables
2. Delete all `.pt` files in `model/checkpoints/`
3. Delete `tokenizer/merges.json` and `tokenizer/vocab.json`
4. Redirect to Ingest page

### Migration from Existing JSON Storage
File: `db/migrate_from_json.py`. Runs once at startup if `model/counts/` directory exists with JSON files. Reads `unigrams.json` through `fourgrams.json`, inserts into DuckDB `token_ngrams`, then moves JSON files to `model/counts/archive/`. After migration, all writes go to DuckDB only.

---

## 8. Backend Architecture

### Process Model
```
FastAPI process (async event loop)
│
├── HTTP routes  (ingest, generate, stats, db_view)
├── WebSocket /ws/progress  ← reads from ProgressQueue
│
└── IngestWorker  (multiprocessing.Process)
        │
        ├── pause_event  (multiprocessing.Event)  ← Pause button signal
        ├── ProgressQueue  (multiprocessing.Queue) → WebSocket handler
        │
        └── per chunk:
              1. data/clean.py            — clean text
              2. ngram_counter (char)     — update char_ngrams
              3. ngram_counter (word)     — update word_ngrams
              4. ngram_counter (bpe)      — update token_ngrams
              5. feedforward.train_step() — one pass, save checkpoint
              6. transformer.train_step() — one pass, save checkpoint
              7. monte_carlo.evaluate()   — 50 runs × 13 models
              8. rollback or commit
```

The IngestWorker is a `multiprocessing.Process` (not a thread) to bypass the Python GIL. The FastAPI event loop reads from `ProgressQueue` in a background asyncio task and pushes messages to all connected WebSocket clients. The Pause button sends a signal via `multiprocessing.Event`; the worker checks it between chunks and exits cleanly.

**Use Opus 4.7 for `api/ingest_worker.py`** — process lifecycle, queue design, pause signalling, and rollback coordination must all be correct simultaneously.

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest/upload` | Accept `.txt` file(s), start IngestWorker |
| `POST` | `/ingest/pause` | Signal IngestWorker to stop after current chunk |
| `GET` | `/ingest/status` | Current worker state, chunks processed/rejected |
| `POST` | `/generate` | Generate text from all 13 models; returns array of outputs |
| `GET` | `/stats/accuracy` | Full accuracy history for all models from DuckDB |
| `GET` | `/stats/last-output` | Last MC-generated sample text per model |
| `GET` | `/db/ngrams` | Paginated n-gram table query (params: family, n, context, page) |
| `GET` | `/db/vocabulary` | Paginated vocabulary table |
| `GET` | `/db/accuracy-history` | All model_accuracy rows; supports CSV export |
| `POST` | `/db/reset` | Full DB + checkpoint + vocabulary reset |
| `WS` | `/ws/progress` | Real-time ingest progress and MC generation stream |

### WebSocket Message Schema
All messages are JSON with a `type` field:

```json
{ "type": "chunk_start",   "chunk_index": 4, "total_chunks": 17, "operation": "Counting char 3-grams" }
{ "type": "chunk_progress","operation": "Training feedforward", "pct": 62 }
{ "type": "mc_token",      "model": "char_3gram", "token": "th", "run": 12 }
{ "type": "mc_complete",   "model": "char_3gram", "accuracy": 0.47, "run": 12 }
{ "type": "chunk_done",    "chunk_index": 4, "status": "accepted", "accuracy_delta": { "char_3gram": +0.03, ... } }
{ "type": "chunk_rejected","chunk_index": 4, "reason": "transformer accuracy dropped 0.04" }
{ "type": "ingest_complete","chunks_accepted": 14, "chunks_rejected": 3 }
```

---

## 9. Frontend Architecture

### Structure
```
frontend/
├── src/
│   ├── App.tsx                         Router + persistent top nav bar
│   ├── pages/
│   │   ├── IngestPage.tsx
│   │   ├── StatsPage.tsx
│   │   ├── GenerationPage.tsx
│   │   └── DBPage.tsx
│   ├── components/
│   │   ├── ModelAccuracyChart.tsx      Recharts LineChart; appends points via WS
│   │   ├── ColoredText.tsx             Renders (word, is_real) pairs green/red
│   │   ├── MonteCarloStream.tsx        Live token stream during MC runs
│   │   ├── ProgressPanel.tsx           Op label, progress bar, Pause button
│   │   └── NGramTable.tsx              Paginated sortable table for DB viewer
│   └── hooks/
│       └── useProgressSocket.ts        WS connection, auto-reconnect, message dispatch
├── vite.config.ts                      Proxy /api and /ws to FastAPI on :8000
└── package.json
```

Navigation uses React Router v6. Switching pages never triggers a backend re-run. The `useProgressSocket` hook maintains a single persistent WebSocket connection for the app lifetime, dispatching typed messages to page-level state.

All 13 model accuracy charts use Recharts `LineChart` with `syncId="models"` so zooming one chart zooms all. New data points are appended to local state via WebSocket — no polling.

---

## 10. Pages

### Page 1 — Ingest

- Drag-and-drop zone + file picker for `.txt` files
- On drop: pre-screen with WVM — reject files where <70% of words are valid English (corrupted data guard); show rejection reason
- Display accepted file list with word counts
- **Start Ingestion** button → `POST /ingest/upload`
- **Processing panel** (live, during ingest):
  - Operation label: e.g. `"Counting BPE token 2-grams... chunk 4 of 17"`
  - Progress bar (chunks done / total)
  - Counters: bytes processed, chunks accepted, chunks rejected
  - **Pause** button → `POST /ingest/pause`; halts after current chunk
- **Monte Carlo live display** (during MC eval phase of each chunk):
  - For each of 13 models: stream generated tokens to screen as they arrive (via `mc_token` WS messages)
  - On `mc_complete`: show real-word % badge next to model name
  - Each of 50 runs overwrites the previous display
- After each chunk: accepted/rejected status with reason; before/after accuracy delta per model (green if improved, red if dropped)

---

### Page 2 — Stats

- 13 scrollable model cards, sorted by current accuracy (re-sortable via button)
- Each card:
  - **LineChart** (Recharts): X = chunk index, Y = mean real-word % over 50 MC runs
  - Neural models (12, 13) add a second line: validation perplexity on right Y-axis
  - Below chart: last MC-generated text sample with valid words in **green**, invalid in **red**
  - Current accuracy badge (large, prominent)
- All charts update live via WebSocket `chunk_done` messages — new point appended, no page refresh

---

### Page 3 — Generation

- **Word count slider**: 20–500 words
- **Generate** button → `POST /generate` → waits for all 13 model responses
- 13 output cards (same order as Stats), each showing:
  - Model name + family badge
  - Raw output: valid words **green**, invalid words **red**
  - Real-word % badge
- **Auto spell-correct toggle**:
  - When on: each invalid word is replaced by the nearest valid word from SCOWL via `difflib.get_close_matches()`
  - Display is **side by side**: left column = raw output (green/red), right column = corrected output (all green)
  - Makes the model's gaps explicit while showing what it was reaching for

---

### Page 4 — DB Viewer

- **Tab bar**: Char N-grams | Word N-grams | BPE Token N-grams | Vocabulary | Accuracy History
- N-gram tabs:
  - Order selector (1–5 for char; 1–3 for word/token)
  - Context filter input (e.g. `"th"` shows all rows where context = `"th"`)
  - Paginated table: context | next item | count | probability (count / sum of counts for this context)
  - Default sort: count descending
- Vocabulary tab: token_id, token string, source (char/word/bpe), frequency
- Accuracy History tab: full `model_accuracy` table + **Export CSV** button
- **DB Reset** button (bottom, red):
  - Confirmation modal: "This permanently deletes all training data, n-gram tables, neural network weights, and vocabulary. Start over?"
  - On confirm: calls `POST /db/reset`, redirects to Ingest page

---

## 11. Word Verification Module (WVM)

File: `wvm/validator.py`. Word list: `wvm/scowl_70.txt` (~200K words, SCOWL size 70).

**Tokenization logic** (shared by WVM and word n-gram model):
1. Split on whitespace
2. For each token: strip leading/trailing chars in `.,!?;:"'()[]{}—` but preserve internal apostrophes (`it's`, `don't`)
3. Lowercase
4. Look up in SCOWL `set`

Returns: `List[Tuple[str, bool]]` (word, is_real) and `float` (percentage real).

Loaded once at backend startup. O(1) per lookup. Zero native library dependencies — `pyenchant` is not used.

---

## 12. Monte Carlo Accuracy Evaluator

File: `eval/monte_carlo.py`.

For each checkpoint (after each accepted chunk):
1. For each of 13 models: generate 50 text samples of exactly 100 words
2. Run each sample through WVM → real-word %
3. Mean over 50 runs = accuracy score for this model at this checkpoint
4. Insert into `model_accuracy` table
5. Emit `mc_token` and `mc_complete` WebSocket messages during generation

**Performance**: 50 × 13 = 650 generations per chunk. Character Markov: <1ms each. Neural net: ~50ms each (CUDA). Use `multiprocessing.Pool` to run the 13 models in parallel across CPU cores. Neural models stay on CUDA; Markov models run on CPU. Total MC eval time per chunk: ~30–60 seconds.

---

## 13. File Structure (Target State)

```
LLM_prototype/
├── SPEC.md
├── plan.md                            (generated next)
├── requirements.txt                   (updated — see section 13)
│
├── wvm/                               NEW
│   ├── validator.py
│   └── scowl_70.txt
│
├── db/                                NEW
│   ├── schema.py
│   ├── store.py
│   ├── migrate_from_json.py
│   └── llm_evolution.duckdb          (runtime artifact)
│
├── data/                              EXISTING — unchanged
│   ├── clean.py
│   ├── split.py
│   ├── download_gutenberg.py
│   └── download_wikipedia.py
│
├── tokenizer/                         EXISTING — unchanged
│   ├── bpe.py
│   └── train_tokenizer.py
│
├── model/                             EXISTING + extensions
│   ├── char_ngram.py                  REFACTORED from demo/char_ngrams.py (orders 1–5, WVM scoring)
│   ├── ngram_counter.py               EXTENDED (mode param, order 5)
│   ├── smoothing.py                   UNCHANGED
│   ├── language_model.py              EXTENDED (DuckDB persistence)
│   ├── word_ngram.py                  NEW (models 6–8)
│   ├── feedforward.py                 NEW (model 12, PyTorch)
│   ├── transformer.py                 NEW (model 13, PyTorch — use Opus 4.7)
│   ├── build_model.py                 EXTENDED (new model types)
│   ├── count_ngrams.py                EXTENDED (word mode)
│   └── checkpoints/                   NEW (runtime .pt files)
│
├── generate/                          EXISTING — unchanged
│   ├── sampling.py
│   ├── generator.py
│   └── cli.py
│
├── eval/                              EXISTING + new
│   ├── perplexity.py                  UNCHANGED
│   ├── monte_carlo.py                 NEW
│   └── ablation.py                    UNCHANGED
│
├── api/                               NEW
│   ├── main.py
│   ├── ingest_worker.py               (use Opus 4.7)
│   ├── routes/
│   │   ├── ingest.py
│   │   ├── generate.py
│   │   ├── stats.py
│   │   └── db_view.py
│   └── ws/
│       └── progress.py
│
├── frontend/                          NEW (React + Vite + TypeScript)
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/
│   │   │   ├── IngestPage.tsx
│   │   │   ├── StatsPage.tsx
│   │   │   ├── GenerationPage.tsx
│   │   │   └── DBPage.tsx
│   │   ├── components/
│   │   │   ├── ModelAccuracyChart.tsx
│   │   │   ├── ColoredText.tsx
│   │   │   ├── MonteCarloStream.tsx
│   │   │   ├── ProgressPanel.tsx
│   │   │   └── NGramTable.tsx
│   │   └── hooks/
│   │       └── useProgressSocket.ts
│   ├── index.html
│   ├── vite.config.ts
│   └── package.json
│
├── demo/                              EXISTING — do not modify
│
└── tests/                             EXISTING + new
    ├── test_tokenizer.py              KEEP
    ├── test_ngram_counter.py          EXTEND
    ├── test_smoothing.py              KEEP
    ├── test_generator.py              KEEP
    ├── test_sampling.py               KEEP
    ├── test_wvm.py                    NEW
    ├── test_db_store.py               NEW
    ├── test_monte_carlo.py            NEW
    ├── test_feedforward.py            NEW
    └── test_transformer.py            NEW
```

---

## 14. Updated `requirements.txt`

```
# Existing
requests>=2.31.0
regex>=2023.12.25
pytest>=7.4.0
ruff>=0.1.0
tqdm>=4.66.0
flask>=3.0.0          # demo/ only

# Backend
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
websockets>=12.0
python-multipart>=0.0.9

# Database
duckdb>=0.10.0

# ML (install torch separately via pytorch.org with CUDA 12.x)
torch>=2.2.0
numpy>=1.26.0

# No additional packages for WVM — SCOWL loaded as flat file
# No additional packages for spell-correct — difflib is stdlib
```

---

## 15. Build Order

Dependencies determine order. Each step unblocks the next.

1. **`wvm/`** — No dependencies. Unblocks everything that scores words.
2. **`db/schema.py` + `db/store.py`** — No dependencies. Unblocks all persistence.
3. **`db/migrate_from_json.py`** — Depends on db/store. Ports existing data.
4. **Extend `model/ngram_counter.py`** — Add `mode` + order 5. Update `test_ngram_counter.py`.
5. **`model/char_ngram.py`** — Refactor from `demo/char_ngrams.py`. Depends on extended counter + WVM.
6. **`model/word_ngram.py`** — Depends on extended counter + db/store.
7. **Extend `model/language_model.py`** — Replace JSON I/O with db/store calls.
8. **`eval/monte_carlo.py`** — Depends on WVM + all 11 Markov models.
9. **`model/feedforward.py`** — PyTorch feedforward. Depends on db/store (checkpoints).
10. **`model/transformer.py`** — PyTorch transformer. Depends on db/store. **Use Opus 4.7.**
11. **`api/ingest_worker.py`** — Orchestrates all of the above. **Use Opus 4.7.**
12. **`api/routes/` + `api/ws/progress.py`** — Depends on ingest_worker.
13. **`api/main.py`** — Mounts all routers.
14. **`frontend/`** — Depends on all API endpoints being defined.
15. **Integration test** — End-to-end: upload `.txt` → ingest → MC eval → Stats page live update → DB viewer.

---

## 16. Agent Model Assignment

| File / Task | Model | Reason |
|---|---|---|
| `api/ingest_worker.py` | **Opus 4.7** | Process lifecycle, queue design, pause signalling, and DuckDB rollback must all coordinate correctly |
| `model/transformer.py` | **Opus 4.7** | VRAM management, causal attention mask, positional encoding — subtle correctness requirements |
| All other files | **Sonnet 4.6** | Straightforward implementations: well-defined inputs/outputs, existing patterns to follow |

---

## 17. All Decisions (Resolved)

| Decision | Resolution |
|---|---|
| BPE vocabulary on DB reset | Wiped atomically with DB reset. Deletes `merges.json`, `vocab.json`, all `.pt` checkpoints, and the `.duckdb` file. Next ingest retrains vocabulary from zero. |
| Spell-correct display (Generation page) | **Side by side**: left = raw output (green/red), right = corrected output (all green). Makes the gap between what the model generated and valid English explicit. |
| Neural net training cadence | **One forward+backward pass per chunk.** Keeps all 13 models on the same "training progress = chunks ingested" timeline for fair comparison on Stats page. |
| Neural net checkpoint reload | **Auto-reload on startup.** Backend scans `model/checkpoints/` and loads highest-numbered checkpoint per model. Accuracy history restored from DuckDB. |
| Word validation library | **SCOWL size-70 as Python `set`**. `pyenchant` dropped — WSL `libenchant` causes venv conflicts. |
| N-gram orders | Char: 1–5. Word: 1–3. BPE token: 1–3. Going beyond 5/3 adds marginal educational value at disproportionate compute cost. |
| Monte Carlo runs per checkpoint | 50 runs per model. Balances statistical reliability (~±2% variance) against compute time (~30–60s per chunk). |
| Chunk size | 1,000–2,500 characters. Small enough for granular accuracy progression on the Stats graph; large enough for meaningful n-gram statistics. |
| Train/validation split per chunk | 80/20. Training split feeds n-gram counts and neural net training step. Validation split feeds `eval/perplexity.py`. |
| Generation page prompt | No user prompt input. System picks its own starting context autonomously. |
