# LLM Evolution

An educational app that lets you watch 13 language models grow in real time as you feed them text.

Upload a `.txt` file. The pipeline splits it into chunks, trains 11 Markov models (character, word, and BPE n-grams in orders 1–5) and 2 neural models (feedforward and Transformer), then scores each model with 50 Monte Carlo generations. Watch accuracy evolve on the Stats page.

## Requirements

- Python 3.11+
- Node.js 20+ and npm
- ~2 GB free disk space
- (Optional) CUDA GPU with 4+ GB VRAM — CPU-only mode works but is ~5–10× slower for neural training

## Install

```bash
# Python backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# React frontend
cd frontend && npm install && cd ..
```

Install Playwright browsers (needed for E2E tests only):

```bash
cd frontend && npx playwright install chromium && cd ..
```

## Run

```bash
bash scripts/dev.sh
```

Opens the app at **http://localhost:5173**. Press `Ctrl+C` to stop.

Windows users:

```powershell
pwsh scripts/dev.ps1
```

To change ports, set `VITE_API_BASE=http://localhost:<port>` in the environment and pass `--port <port>` to uvicorn.

## First Steps

1. Click **Ingest** → drag a `.txt` file (plain English prose works best).
2. Watch the progress panel fill with chunk events.
3. When ingest completes, open **Stats** to see accuracy curves for all 13 models.
4. Open **Generate** → choose a word count → click Generate to sample from every model at once.
5. Open **DB** to browse raw DuckDB tables.

## Architecture

```
Browser (React / Vite :5173)
  │
  │  HTTP REST + WebSocket
  ▼
FastAPI (uvicorn :8000)
  ├── /ingest/*    ──► IngestWorker (multiprocessing.Process)
  │                        │
  │                        ├── char / word / BPE n-gram counters → DuckDB
  │                        ├── FeedforwardTrainer (PyTorch)
  │                        ├── TransformerTrainer (PyTorch)
  │                        └── MonteCarloEvaluator (ProcessPoolExecutor)
  │                                  │
  │                                  └── model_accuracy rows → DuckDB
  ├── /stats/*     ──► DuckDB reads
  ├── /generate    ──► CharNGramModel / WordNGramModel / neural trainers
  ├── /db/*        ──► DuckDB read-only inspection
  └── /ws/progress ──► relay_progress task (mp.Queue → WebSocket fan-out)
```

## Reset

- Click **Reset DB** on the DB page, or:

```bash
rm -f db/llm_evolution.duckdb db/llm_evolution.duckdb.wal
rm -f model/checkpoints/*.pt
```

## Testing

```bash
# Fast unit tests (all tracks)
pytest -m "not slow"

# Integration smoke test (takes minutes — requires no other server on :8765)
pytest -m slow tests/test_integration.py

# E2E Playwright test (requires dev server running)
bash scripts/dev.sh &
sleep 10
cd frontend && npx playwright test
```

## Further reading

- `SPEC.md` — full architecture and data-model spec
- `specs/` — per-track implementation notes for each of the 8 build tracks
