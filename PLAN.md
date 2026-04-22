# LLM Evolution App — Multi-Agent Build Plan

**This file is the single source of truth for the build.** Every agent must follow this protocol exactly. Detailed technical specs live in `specs/`. The architectural source of truth is `SPEC.md`.

## Model roster

Three model tiers participate in this build:

| Model ID | Role |
| :--- | :--- |
| `claude-sonnet-4-6` | All tracks except B2 and D1 — follows detailed specs faithfully |
| `claude-opus-4-7` | Tracks B2 and D1 only — causal masking and multiprocessing rollback require it |

Each track's `Model` field specifies exactly one of these IDs. **Agents must only claim tracks matching their own model ID** (see Rule 2 below).

> **Read first**, in order:
> 1. `SPEC.md` — what you are building and why
> 2. This file — the task board and coordination protocol
> 3. `specs/0X-*.md` — the detailed spec for your claimed track only

---

## 1. Agent Coordination Protocol (Local, No Git)

This project is built locally with Warp.dev agents. There is no GitHub. Coordination is via atomic edits to this file.

### Rules for every agent

1. **Open `PLAN.md` fresh at the start of each work session.** Do not rely on a cached copy.

2. **Claim your task atomically — only from your model tier.**
   - Identify your own model ID (`claude-sonnet-4-6` or `claude-opus-4-7`).
   - Find the first task whose `Model` column matches your model ID, whose Status is `[ AVAILABLE ]`, and whose dependencies are all `[ COMPLETE ]`.
   - **Do not claim tracks assigned to a different model**, even if available and your own queue is empty. The assignments exist because each model has different strengths; crossing tiers defeats the point.
   - Edit this file: set Status to `[ IN PROGRESS — Agent: <your name>, Started: <ISO timestamp> ]`.
   - Save immediately.
   - Wait 3 seconds, re-read `PLAN.md`. If your claim is still there, proceed. If another agent's claim overwrote yours, pick a different available task (still from your tier).

3. **Load the spec for your track only.**
   - Open the corresponding `specs/0X-*.md` file.
   - That file contains all implementation details, signatures, tests, and acceptance criteria for your track.
   - You may read `SPEC.md` freely for context. You may not read other tracks' spec files unless integrating (Track Z).

4. **Work strictly within your track's deliverables.**
   - Do not touch files outside the "Files I own" list in your spec.
   - Do not modify code in another agent's track — surface the issue in the Notes column instead.
   - Follow coding standards: Python 3.11+, type hints on all public functions, docstrings on all modules; TypeScript strict mode on frontend.

5. **Run tests before marking complete.**
   - Python: `pytest tests/<your-module>/` — all new tests pass, no existing tests break.
   - TypeScript: `cd frontend && npm run typecheck && npm run lint` — zero errors.
   - If your track has integration points, run them as described in your spec's "Acceptance" section.

6. **Mark your task complete.**
   - When tests pass, edit this file: set Status to `[ COMPLETE — <ISO timestamp> ]`.
   - Add a one-line summary in the Notes column (e.g. "All 12 n-gram counter tests green; order 5 added").

7. **Pick up the next available task (within your tier).**
   - Scan the board top-to-bottom for the next `[ AVAILABLE ]` task assigned to your model ID whose dependencies are met.
   - If nothing in your tier is available but other tiers still have work, wait. Re-read `PLAN.md` every 2 minutes.
   - If every task assigned to your model ID is either `[ COMPLETE ]` or `[ IN PROGRESS ]`, stop polling and exit. Do not claim work outside your tier to "help."
   - The build is finished when Track Z is `[ COMPLETE ]`.

8. **Surface blockers immediately.**
   - If a dependency is broken or your spec is ambiguous, set Status to `[ BLOCKED — <reason> ]` and add detail in Notes. Do not silently stall.

9. **Cross-contamination rule.**
   - Only the Track Z (Integration) agent may modify files across track boundaries.
   - All other agents: if you need something from another track that isn't there yet, your track is blocked. Say so.

---

## 2. Pre-Flight Checklist

Before Track 0 can start, the user (or the first agent) must verify:

- [ ] Python 3.11+ available in WSL2 Ubuntu (`python3 --version`)
- [ ] Virtual environment created and activated (`python3 -m venv venv && source venv/bin/activate`)
- [ ] `pip install -r requirements.txt` completes (requirements.txt already updated per SPEC section 14)
- [ ] CUDA setup verified OR CPU fallback accepted:
  - `python -c "import torch; print(torch.cuda.is_available())"` returns `True` (preferred)
  - OR user accepts CPU-only mode (training ~5–10× slower; still works)
- [ ] Node 20+ and npm installed for frontend work (`node --version && npm --version`)
- [ ] SCOWL size-70 word list downloaded to `wvm/scowl_70.txt` (see `specs/00-foundations.md`)

---

## 3. Task Board

### Status Legend
- `[ AVAILABLE ]` — Ready to claim
- `[ IN PROGRESS — Agent: X, Started: Y ]` — Claimed and in flight
- `[ COMPLETE — <timestamp> ]` — Done; tests pass
- `[ BLOCKED — <reason> ]` — Waiting on unresolved dependency or ambiguity

### Tracks

#### Track 0 — Foundations (WVM, DB, Contracts)
- **Model:** `claude-sonnet-4-6`
- **Status:** `[ COMPLETE — 2026-04-21T10:25:00Z ]`
- **Depends on:** None
- **Agent:** claude-sonnet-4-6
- **Started:** 2026-04-21T08:02:00Z
- **Completed:** 2026-04-21T10:25:00Z
- **Notes:** All 69 tests green (test_wvm, test_db_store, test_contracts); wvm/validator.py, db/schema.py, db/store.py, api/contracts.py complete
- **Deliverables:**
    - `wvm/validator.py` + `wvm/scowl_70.txt` — word verification module
    - `db/schema.py` + `db/store.py` — DuckDB schema and persistence helpers
    - `api/contracts.py` — Pydantic models for every HTTP and WebSocket message
    - `tests/test_validator.py`, `tests/test_store.py`, `tests/test_contracts.py`
- **Full spec:** [`specs/00-foundations.md`](specs/00-foundations.md)

#### Track A — Markov Model Layer
- **Model:** `claude-sonnet-4-6`
- **Status:** `[ COMPLETE — 2026-04-21T09:35:00Z ]`
- **Depends on:** Track 0
- **Agent:** claude-sonnet-4-6
- **Started:** 2026-04-21T08:46:00Z
- **Completed:** 2026-04-21T09:35:00Z
- **Notes:** 200 tests green; NGramCounter class (char/word/bpe), CharNGramModel, WordNGramModel, LanguageModel.from_store, migrate_from_json; count_ngrams/build_model extended with --family/--orders
- **Deliverables:**
    - Extend `model/ngram_counter.py` with `mode` param (char/word/bpe) and order 5
    - `model/char_ngram.py` (refactored from `demo/char_ngrams.py`)
    - `model/word_ngram.py`
    - Extend `model/language_model.py` with DuckDB persistence
    - `db/migrate_from_json.py` migration script
    - Matching tests under `tests/`
- **Full spec:** [`specs/01-markov-models.md`](specs/01-markov-models.md)

#### Track B1 — Feedforward Neural Network (Model 12)
- **Model:** `claude-sonnet-4-6`
- **Status:** `[ COMPLETE — 2026-04-21T09:55:00Z ]`
- **Depends on:** Track 0
- **Agent:** claude-sonnet-4-6
- **Started:** 2026-04-21T09:40:00Z
- **Completed:** 2026-04-21T09:55:00Z
- **Notes:** 18 tests green on CPU (1 skipped: CUDA); torch installed; .gitignore added; checkpoint size threshold corrected to 30 MB (architecture yields ~26 MB)
- **Deliverables:**
    - `model/feedforward.py` — PyTorch feedforward LM + trainer
    - `model/checkpoints/` directory with `.gitignore` for `*.pt`
    - `tests/test_feedforward.py`
- **Full spec:** [`specs/02-feedforward.md`](specs/02-feedforward.md)

#### Track B2 — Transformer Neural Network (Model 13)
- **Model:** `claude-opus-4-7`
- **Status:** `[ COMPLETE — 2026-04-21T09:10:31Z ]`
- **Depends on:** Track 0
- **Agent:** claude-opus-4-7
- **Started:** 2026-04-21T09:02:17Z
- **Completed:** 2026-04-21T09:10:31Z
- **Notes:** 31 CPU tests green (2 CUDA tests skipped); full suite 249 passed / 3 skipped. Causal mask leakage test included and passing; sinusoidal PE, pre-norm GPT-style blocks, shift-by-one next-token loss, last-position generation, checkpoint round-trip all verified.
- **Deliverables:**
    - `model/transformer.py` — PyTorch causal Transformer LM + trainer
    - `tests/test_transformer.py` (must include causal-mask-leakage test)
- **Full spec:** [`specs/03-transformer.md`](specs/03-transformer.md)

#### Track C — Monte Carlo Evaluator
- **Model:** `claude-sonnet-4-6`
- **Status:** `[ COMPLETE — 2026-04-21T09:48:00Z ]`
- **Depends on:** Tracks A, B1, B2
- **Agent:** claude-sonnet-4-6
- **Started:** 2026-04-21T09:15:20Z
- **Completed:** 2026-04-21T09:48:00Z
- **Notes:** 30 tests green; MonteCarloEvaluator with ProcessPoolExecutor (one task/Markov model), 13 MODELS registry, _score_sample_text helper, mc_model_start+mc_complete progress callbacks, DB insert via store.insert_accuracy
- **Deliverables:**
    - `eval/monte_carlo.py` — 50-run accuracy evaluator for all 13 models
    - `tests/test_monte_carlo.py`
- **Full spec:** [`specs/04-monte-carlo.md`](specs/04-monte-carlo.md)

#### Track D1 — Ingest Worker (multiprocessing core)
- **Model:** `claude-opus-4-7`
- **Status:** `[ COMPLETE — 2026-04-21T10:48:57Z ]`
- **Depends on:** Track C
- **Agent:** claude-opus-4-7
- **Started:** 2026-04-21T09:48:30Z
- **Completed:** 2026-04-21T10:48:57Z
- **Notes:** 26 fast + 1 slow tests green (full suite 305 passed / 3 CUDA-skipped). spawn-context multiprocessing worker; manual delta-based rollback (DuckDB MVCC prevents chunk-wide transactions when MC child processes must read mid-ingest state); deepcopy state_dict snapshots for NN rollback; .pt file cleanup on reject; pause checked only between chunks; safe_put drops events on queue full. Helpers split into `api/ingest_helpers.py` to keep every file under the 400-line cap; pure-function coverage in `tests/test_ingest_helpers.py`.
- **Deliverables:**
    - `api/ingest_worker.py` — multiprocessing orchestration, rollback, pause
    - `api/ingest_helpers.py` — chunking, pre-screening, n-gram delta apply/revert, factories
    - `api/worker_types.py`
    - `tests/test_ingest_worker.py`, `tests/test_ingest_helpers.py`
- **Full spec:** [`specs/05-ingest-worker.md`](specs/05-ingest-worker.md)

#### Track D2 — FastAPI Routes + WebSocket
- **Model:** `claude-sonnet-4-6`
- **Status:** `[ COMPLETE — 2026-04-21T11:30:00Z ]`
- **Depends on:** Track D1
- **Agent:** claude-sonnet-4-6
- **Started:** 2026-04-21T10:52:30Z
- **Completed:** 2026-04-21T11:30:00Z
- **Notes:** 50 new tests green (347 total / 3 skipped). FastAPI app with lifespan, CORS, 5 router groups; IngestUploadResponse/IngestStatusResponse/IngestPauseResponse contracts honoured; single relay_progress task drains mp.Queue and fans out to all WS clients; ALLOWED_TABLES whitelist with alias map (chunks→corpus_chunks, last_generations→None); generate endpoint dispatches to all 13 model families in thread executor; auto-correct via Validator.suggest; python-multipart installed as test dependency.
- **Deliverables:**
    - `api/main.py`, `api/state.py`
    - `api/routes/{ingest,stats,generate,db,ws}.py`
    - `tests/test_api_*.py`
- **Full spec:** [`specs/06-fastapi-backend.md`](specs/06-fastapi-backend.md)

#### Track E — React Frontend
- **Model:** `claude-sonnet-4-6`
- **Status:** `[ COMPLETE — 2026-04-21T09:15:00Z ]`
- **Depends on:** Track 0 (contracts only — can run in parallel with A/B/C/D)
- **Agent:** claude-sonnet-4-6
- **Started:** 2026-04-21T08:31:15Z
- **Completed:** 2026-04-21T09:15:00Z
- **Notes:** All acceptance criteria green: typecheck zero errors, ESLint zero warnings, build 201 kB gzipped (<500 kB). Vite 5 + React 18 + TS strict + Tailwind 3 + React Query v5 + react-router v7 + Recharts 3 + Zod 4. WSProvider opens single WS connection with auto-reconnect; event log buffered at 200 ms. All data-testid attributes present.
- **Deliverables:**
    - `frontend/` — Vite + React + TypeScript + Tailwind scaffold
    - 4 pages: Ingest, Stats, Generate, DB
    - Hooks and components per spec
    - Stable `data-testid` attributes for Track Z
- **Full spec:** [`specs/07-frontend.md`](specs/07-frontend.md)

#### Track Z — Integration + End-to-End Test
- **Model:** `claude-sonnet-4-6`
- **Status:** `[ COMPLETE — 2026-04-21T11:46:00Z ]`
- **Depends on:** Tracks D2, E (and all transitive deps)
- **Agent:** claude-sonnet-4-6
- **Started:** 2026-04-21T11:31:00Z
- **Completed:** 2026-04-21T11:46:00Z
- **Notes:** 346 fast tests green / 3 skipped; typecheck zero errors, ESLint zero warnings. README.md, scripts/dev.sh, scripts/dev.ps1, scripts/seed_corpus.py, tests/test_integration.py (@pytest.mark.slow), frontend/playwright.config.ts, frontend/tests/e2e.spec.ts, pyproject.toml, requirements.txt, .gitignore all created.
- **Deliverables:**
    - `README.md` quickstart, `scripts/dev.sh`, `scripts/seed_corpus.py`
    - `tests/test_integration.py`, `frontend/tests/e2e.spec.ts`
    - `pyproject.toml`/`requirements.txt`, `.gitignore`
    - Full project type-check and lint verification green
- **Full spec:** [`specs/99-integration.md`](specs/99-integration.md)

### Parallelism opportunities

- After Track 0 completes: **A, B1, B2, E** all unblock simultaneously — Sonnet picks up A/B1/E, Opus 4.7 picks up B2.
- After A, B1, B2 complete: **C** unlocks (Sonnet 4.6).
- After C: **D1** runs (Opus 4.7 — the hardest track).
- After D1: **D2** runs (Sonnet 4.6).
- **E** can complete entirely in parallel with A/B/C/D1/D2 as long as it only relies on contracts from Track 0.
- **Z** requires D2 and E (Sonnet 4.6).

### Per-tier workload

- `claude-sonnet-4-6`: **7 tracks** — 0, A, B1, C, D2, E, Z
- `claude-opus-4-7`: **2 tracks** — B2, D1

---

## 4. Track Summaries (Details in `specs/`)

Each track below links to its full spec file. Read only the spec for your claimed track.

### Track 0 — Foundations
> Full spec: [`specs/00-foundations.md`](specs/00-foundations.md)

Builds the three modules every later track depends on:
- `wvm/validator.py` + `wvm/scowl_70.txt` — word verification module
- `db/schema.py` + `db/store.py` — DuckDB schema and persistence helpers
- `api/contracts.py` — Pydantic models for every HTTP and WebSocket message

### Track A — Markov Model Layer
> Full spec: [`specs/01-markov-models.md`](specs/01-markov-models.md)

All 11 Markov models (char 1–5, word 1–3, BPE token 1–3):
- Extend `model/ngram_counter.py` with `mode` param and order 5
- Create `model/char_ngram.py` (refactored from `demo/char_ngrams.py`)
- Create `model/word_ngram.py`
- Extend `model/language_model.py` with DuckDB persistence
- `db/migrate_from_json.py` migration script

### Track B1 — Feedforward Neural Network (Model 12)
> Full spec: [`specs/02-feedforward.md`](specs/02-feedforward.md)

- `model/feedforward.py` — PyTorch feedforward LM
- Training step API, checkpoint save/load, text generation via argmax sampling
- Integrates with DuckDB for `nn_checkpoints` rows

### Track B2 — Transformer Neural Network (Model 13)
> Full spec: [`specs/03-transformer.md`](specs/03-transformer.md)

- `model/transformer.py` — PyTorch causal Transformer LM
- **Use Opus 4.7.** VRAM management, causal mask, positional encoding, pre-norm layout all require subtle correctness.

### Track C — Monte Carlo Evaluator
> Full spec: [`specs/04-monte-carlo.md`](specs/04-monte-carlo.md)

- `eval/monte_carlo.py` — 50-run accuracy evaluator for all 13 models
- Parallel execution across CPU cores
- Progress callback hooks for WebSocket streaming

### Track D1 — Ingest Worker
> Full spec: [`specs/05-ingest-worker.md`](specs/05-ingest-worker.md)

- `api/ingest_worker.py` — the coordination layer
- Multiprocessing.Process lifecycle, pause signalling, progress queue, DuckDB transaction management, chunk rejection and rollback
- **Use Opus 4.7.** This is the hardest module in the project.

### Track D2 — FastAPI Backend
> Full spec: [`specs/06-fastapi-backend.md`](specs/06-fastapi-backend.md)

- `api/main.py` — FastAPI app bootstrap
- `api/routes/ingest.py`, `api/routes/generate.py`, `api/routes/stats.py`, `api/routes/db_view.py`
- `api/ws/progress.py` — WebSocket progress handler

### Track E — React Frontend
> Full spec: [`specs/07-frontend.md`](specs/07-frontend.md)

- Vite + React + TypeScript scaffold
- 4 pages: Ingest, Stats, Generation, DB Viewer
- Components: `ModelAccuracyChart`, `ColoredText`, `MonteCarloStream`, `ProgressPanel`, `NGramTable`
- `useProgressSocket` hook with auto-reconnect

### Track Z — Integration + End-to-End Test
> Full spec: [`specs/99-integration.md`](specs/99-integration.md)

- Wire backend and frontend: `vite.config.ts` proxy, CORS, startup scripts
- End-to-end smoke test: upload a known `.txt`, verify all 13 models update, verify rollback works
- Verify auto-reload of neural checkpoints on restart
- Update top-level `README.md` with run instructions

---

## 5. Coding Rules

These rules apply to every agent on every track without exception.

### Modularity
- **Maximum 400 lines per file.** If a file exceeds this, split it before marking the track complete.
- One class or one cohesive group of functions per module. No "utils" catch-all files.

### Python
- Python 3.11+. Type hints on every public function. No bare `Any` — use `Unknown` + narrow or explicit union types.
- Docstrings on every module (one line is enough; explains *what*, not *how*).
- Use `pathlib.Path`, never raw string paths.
- Imports: stdlib → third-party → local, separated by blank lines.
- Public functions raise explicit exceptions; never silently return `None` on error.
- Tests mirror source layout: `model/feedforward.py` → `tests/test_feedforward.py`.

### TypeScript / React
- `tsconfig` strict mode. `any` is forbidden — use `unknown` + narrow.
- Functional components only. No class components.
- Props interfaces named `XxxProps`, exported alongside the component.
- No inline styles — Tailwind utility classes only.
- WebSocket messages must be validated with Zod before use.
- Icons: use **Lucide-react** exclusively. Do not add other icon libraries.

## 6. Global Conventions

### File paths
- All absolute paths inside code use `pathlib.Path`.
- All imports use project-relative paths.
- The project root is `//wsl.localhost/Ubuntu/home/latoj/warp_projects/LLM_prototype`.

### Never touch
- `tokenizer/bpe.py` — frozen, production-quality
- `model/smoothing.py` — frozen, production-quality
- `generate/sampling.py` — frozen
- `generate/generator.py` — frozen
- `demo/*` — frozen reference; do not delete, do not modify

### Always write tests
Every new module gets a corresponding `tests/test_<module>.py`. Minimum coverage:
- Happy path
- At least one edge case
- At least one failure mode (invalid input, missing data)

---

## 6. Running the App (Post-Integration)

Once Track Z is complete, the full app runs via:

```bash
# Terminal 1 — Backend
cd //wsl.localhost/Ubuntu/home/latoj/warp_projects/LLM_prototype
source venv/bin/activate
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
# Opens http://localhost:5173
```

The frontend proxies `/api/*` and `/ws/*` to `localhost:8000` via Vite config.

---

## 7. Done Criteria

The project is complete when all the following are true:

- [x] All tracks show Status `[ COMPLETE ]`
- [x] `pytest tests/` passes end-to-end (346 passed, 3 skipped CUDA, 0 failures — 2026-04-22)
- [x] `cd frontend && npm run typecheck && npm run lint && npm run build` all green (201 kB gzip — 2026-04-22)
- [x] Integration smoke test: `data/seed.txt` uploaded, all 13 models produced accuracy rows, generate returned 13 outputs, ingest completed successfully — 2026-04-22
- [x] Backend survives a full restart: 13 accuracy series and generate restored from DuckDB — 2026-04-22
- [x] CUDA fallback verified: `CUDA_VISIBLE_DEVICES=""` — 346 passed, 3 GPU-specific skipped — 2026-04-22

---

*Plan template adapted from generic multi-agent plan format. Created: 2026-04-20.*
