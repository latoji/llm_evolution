# Glossary — LLM Evolution App

This document explains every important concept, library, and design decision in the project in plain language. It is written for someone who understands software but is new to machine learning and natural language processing.

---

## Part 1 — The Big Picture

### Language Model
A program that has "learned" patterns in text and can predict what word or character is likely to come next. That's it. Everything in this project — from the simplest 1-gram to the Transformer — is a language model. They differ only in *how much context* they use and *how they represent* that context.

### LLM (Large Language Model)
A language model trained on enormous amounts of text using a Transformer architecture (see below). GPT-4, Claude, and Gemini are LLMs. Our Transformer (Model 13) is architecturally the same family — just very small. The project exists to show the *evolutionary steps* from a completely naive model to the modern approach.

### Accuracy (in this project)
We measure a model's quality by generating text from it and checking what percentage of the generated words are real English words (verified against SCOWL). A 1-gram character model might score 10–15%. A well-trained Transformer might score 60–80%. The Stats page graphs this over time so you can watch every model improve as training data is added.

### Training Data vs Validation Data
When a chunk of text is ingested:
- **Training data (80%)** — the model learns from this. It adjusts its internal counts or weights.
- **Validation data (20%)** — held back and used to *measure* how well the model learned. If performance on validation data drops, the model is getting worse, not better (a sign of bad input data).

### Chunk
The ingestion pipeline splits uploaded `.txt` files into blocks of 1,000–2,500 characters. Each block is called a **chunk**. The system trains all 13 models on one chunk, runs a full Monte Carlo evaluation, and only keeps the chunk if accuracy improved. This keeps the database clean — every piece of training data must earn its place.

### Rollback
If a chunk causes accuracy to drop, every change it made is undone:
- DuckDB rows inserted for that chunk are deleted (database transaction rollback)
- Neural network weights are reset to what they were before the chunk (saved copy restored)
- Checkpoint files written during the chunk are deleted

After rollback, the system is in exactly the state it was in before that chunk arrived.

---

## Part 2 — Text and Tokenization

### Token
A unit of text that a model operates on. Different models use different token types:
- **Characters** — `h`, `e`, `l`, `l`, `o` (5 tokens for "hello")
- **Words** — `hello` (1 token)
- **BPE subwords** — `hel`, `lo` (2 tokens, a middle ground)

Choosing the token type is a fundamental design decision. Smaller tokens (characters) give the model more flexibility but require it to learn more relationships. Larger tokens (words) are more meaningful but the vocabulary explodes — English has hundreds of thousands of words.

### Character N-Gram (Models 1–5)
The simplest possible approach. The model looks at the last *N-1* characters to predict the next character.
- A **1-gram** (unigram) ignores context entirely — it just knows that `e` appears 13% of the time in English text.
- A **2-gram** (bigram) looks at one character back — knowing the previous character is `t`, it predicts `h` is likely next (because `th` is common).
- A **5-gram** looks at 4 characters back — `" the"` strongly predicts `" "` (a space, ending the word).

More context = better predictions, but requires much more training data to fill in reliably.

### Word N-Gram (Models 6–8)
Same idea as character n-grams, but the token is a whole word.
- A **word 1-gram** just knows word frequencies — `the` is most common, `of` is second, etc.
- A **word 2-gram** knows that after `the` comes a noun most of the time.
- A **word 3-gram** knows that `the cat sat` is a plausible sequence.

Word n-grams produce more recognisable text than character n-grams, but they are terrible with words they have never seen.

### BPE — Byte Pair Encoding (Models 9–11)
A tokenization scheme invented for neural machine translation and now used by GPT-2, GPT-4, and most modern LLMs. It sits between characters and words.

**How it works:** Start with individual characters. Find the most frequently co-occurring pair (e.g., `t` + `h` → `th`). Merge them into one token. Repeat 8,000 times. You end up with a vocabulary of 8,000 tokens that includes common whole words (`the`, `is`) and common word-pieces (`ing`, `tion`, `un`). Unknown words are still representable as a sequence of subword pieces.

**Why it matters:** BPE gives the neural models a good vocabulary size (not too small, not too large) and handles new words gracefully. Our BPE tokenizer is pre-trained and frozen — it does not change as new text is ingested.

### Vocabulary
The complete set of tokens a model knows about. Our BPE vocabulary has exactly **8,000 tokens** (IDs 0–7999). Token 0 is `<|endoftext|>` (signals end of generation). Token 1 is `<|pad|>` (padding when context is shorter than expected).

### Context Window
The number of previous tokens a model can "see" when making its next prediction. Our neural models use a context window of **64 tokens**. Anything further back is invisible to the model — it has no memory beyond its window.

### Embedding
A way to turn a token ID (an integer like `42`) into a vector of numbers (like `[0.3, -0.1, 0.8, ...]`) that a neural network can do maths on. Our embedding table has 8,000 rows × 128 columns — each of the 8,000 tokens gets its own 128-number representation. The network learns these representations during training.

---

## Part 3 — The 13 Models

The project runs 13 models simultaneously. Every time a chunk is ingested, all 13 are trained and all 13 are evaluated. This lets you watch models of increasing sophistication compete on the same data.

| # | Name | Type | Token unit | Context |
|---|---|---|---|---|
| 1 | `char_1gram` | Markov | character | 0 (none) |
| 2 | `char_2gram` | Markov | character | 1 char |
| 3 | `char_3gram` | Markov | character | 2 chars |
| 4 | `char_4gram` | Markov | character | 3 chars |
| 5 | `char_5gram` | Markov | character | 4 chars |
| 6 | `word_1gram` | Markov | word | 0 (none) |
| 7 | `word_2gram` | Markov | word | 1 word |
| 8 | `word_3gram` | Markov | word | 2 words |
| 9 | `bpe_1gram` | Markov | BPE subword | 0 (none) |
| 10 | `bpe_2gram` | Markov | BPE subword | 1 subword |
| 11 | `bpe_3gram` | Markov | BPE subword | 2 subwords |
| 12 | `feedforward` | Neural net | BPE subword | 64 tokens |
| 13 | `transformer` | Neural net | BPE subword | 64 tokens |

### Markov Chain
A probabilistic model where the next state depends only on the current state (or the last *N* states). In language modelling, this means: given the last few characters/words, what comes next? The model learns a big table of probabilities from training text (e.g., "after `qu`, the letter `e` appears 92% of the time"). There are no weights, no gradient descent, no GPU — just a frequency table.

### Kneser-Ney Smoothing
A mathematical technique applied to Markov models to handle unseen sequences. If the model has never seen the 4-gram `xqzp`, it falls back to the 3-gram, then the 2-gram, then the 1-gram. Without smoothing, an unseen sequence would have zero probability, which breaks generation. KN smoothing is why `model/smoothing.py` is frozen — it's a well-understood algorithm that the project inherits.

### Feedforward Neural Network (Model 12)
The first neural model. Architecture:
1. Take the last 64 BPE token IDs as input
2. Look up each token's 128-dimensional embedding → a matrix of shape [64 × 128]
3. Flatten to a single vector of 8,192 numbers
4. Pass through three linear layers with ReLU activations, shrinking: 8192 → 512 → 512 → 128
5. Final layer: 128 → 8,000 (one score per vocabulary token)
6. Pick the token with the highest score as the prediction

The key word is **feedforward** — information only flows in one direction (input → output). Each position in the context is treated equally; there is no mechanism for the model to pay more attention to certain positions than others.

### Transformer (Model 13)
The modern architecture behind GPT-2, GPT-4, and most LLMs. It adds **attention** to the feedforward idea.

Key differences from the feedforward model:
- Instead of flattening the context into one vector, the Transformer keeps all 64 positions separate.
- **Self-attention** lets each position look at all other positions and decide which ones are most relevant to its prediction. For the word "bank", the model can learn to pay attention to whether "river" or "money" appeared earlier in the sentence.
- **Causal masking** ensures position *i* can only attend to positions *≤ i*. Without this, the model could "cheat" by looking at future tokens during training.
- **Positional encoding** adds information about *where* in the sequence a token sits (the feedforward model has no sense of order after flattening).

Our Transformer has 2 attention blocks with 4 attention heads each — tiny by modern standards (~6.8M parameters) but architecturally identical to GPT-2.

### Attention / Self-Attention
The mechanism that makes Transformers powerful. For each token in the sequence, attention asks: "which other tokens should I pay most attention to?" and computes a weighted average of their representations. The weights are learned during training.

**Why it matters:** A 5-gram Markov model can only look 4 characters back. Attention has no fixed window — a token can attend to anything in the 64-token context. This lets the model capture long-range dependencies that Markov chains completely miss.

### Parameters / Weights
The numbers inside a neural network that are adjusted during training. The feedforward model has ~5.2M parameters; the Transformer has ~6.8M. These are stored in `.pt` checkpoint files and loaded on restart so training does not start from scratch.

### Loss Function / Cross-Entropy Loss
A number that measures how wrong the model's predictions are. If the model predicted `e` with 90% probability and the actual next character was `e`, loss is low. If it predicted `z` with 90% probability, loss is high. Training adjusts weights to minimise this number.

### Gradient Descent / Adam Optimiser
The algorithm that adjusts weights to reduce loss. For every training example, it calculates the direction in which each weight should change to reduce the loss (the gradient), then takes a small step in that direction. **Adam** is a popular variant that adapts the step size per parameter. Our neural models use Adam with learning rates of 1e-3 (feedforward) and 3e-4 (transformer).

### Checkpoint
A saved snapshot of a neural network's weights at a point in time. Stored as a `.pt` file in `model/checkpoints/`. On restart, the latest checkpoint is loaded so training continues where it left off. If a chunk is rejected, the new checkpoint is deleted and the previous one is kept.

---

## Part 4 — Evaluation

### Monte Carlo Simulation (in this project)
A method of estimating something by running many random trials. Here: run 50 random text generations from each model, score each one, and average the scores. This gives a stable estimate of a model's current quality. The word "Monte Carlo" refers to the famous casino — the technique uses randomness to approximate a result that would be hard to compute directly.

### WVM — Word Verification Module
A module (`wvm/validator.py`) that checks whether a word is real English. It looks the word up in the **SCOWL** word list. The accuracy metric is simply: "of all words generated, what fraction were in SCOWL?" This is a simple but effective proxy for coherence — a model producing mostly real words is doing something right.

### SCOWL (Spell Checker Oriented Word Lists)
A freely available English word list used by many spell-checkers. We use size-70, which covers common and moderately common words (~70,000 entries). Words not in this list are counted as "fake" for scoring purposes. SCOWL is deliberately imperfect — it will sometimes mark rare but valid words as fake — but it is consistent, which is what matters for comparing models against each other.

### Perplexity
A measure of how "surprised" a language model is by text it has not seen before. Lower perplexity = the model finds the text unsurprising = it has learned patterns that match the text. Only computed for neural models (not Markov chains) because it requires probability distributions over the full vocabulary. Displayed on the Stats page alongside accuracy.

---

## Part 5 — Data and Storage

### DuckDB
An embedded analytical database — like SQLite but designed for fast column-scanned queries rather than row lookups. All n-gram counts, model accuracy history, neural network checkpoint metadata, and generated text samples are stored in a single file: `db/llm_evolution.duckdb`. No separate database server is needed.

**Why DuckDB over SQLite?** N-gram tables grow to millions of rows. Queries like "give me all trigrams starting with `th`" are analytical (column-scan) queries that DuckDB handles 10–100× faster than SQLite.

### DuckDB Transaction
A group of database operations that either all succeed or all fail together. When a chunk is processed, all inserts are wrapped in a single transaction. If anything goes wrong — including the worker process being force-killed — the transaction is automatically rolled back, leaving the database in the pre-chunk state. This is the mechanism that makes rollback reliable.

### Store (`db/store.py`)
A Python class that wraps all DuckDB operations. Every part of the application that reads or writes the database goes through `Store`. This keeps SQL in one place and makes the rest of the code independent of the database engine.

### Schema (`db/schema.py`)
Defines the structure of every table in the database:
- `chunks` — one row per ingested text chunk, with status and timestamps
- `char_ngrams` — character n-gram counts
- `word_ngrams` — word n-gram counts
- `token_ngrams` — BPE token n-gram counts
- `model_accuracy` — one row per (model, chunk) evaluation
- `nn_checkpoints` — metadata about saved `.pt` files
- `last_generations` — the most recent generated text from each model

---

## Part 6 — The Application Stack

### FastAPI
A Python web framework for building HTTP APIs. It handles incoming requests (file uploads, pause commands, stats queries) and routes them to the right functions. It also manages WebSocket connections for live progress streaming. Runs on **uvicorn** (an async Python web server).

### REST API
The communication style our frontend uses to talk to the backend. The frontend sends HTTP requests (`GET /stats/accuracy`, `POST /ingest/upload`) and gets JSON responses back. REST is stateless — each request is self-contained.

### WebSocket
A persistent, two-way connection between the browser and the server. Unlike REST (where the browser asks and the server answers), a WebSocket lets the server push data to the browser at any time. Used here to stream live progress events (chunk started, token generated, accuracy updated) to the Ingest and Stats pages without the browser needing to poll.

### Multiprocessing
Running code in a completely separate operating system process rather than a thread. The ingest worker (`api/ingest_worker.py`) runs as a separate process so that the heavy CPU/GPU work of training models does not freeze the FastAPI web server. Python's **GIL (Global Interpreter Lock)** means threads cannot truly run in parallel in Python — separate processes bypass this limitation entirely.

### Queue (`multiprocessing.Queue`)
A thread- and process-safe pipe for passing data between processes. The ingest worker puts progress events into the queue; the FastAPI process reads them out and sends them to connected WebSocket clients. The queue has a maximum size of 1,000 events — if it fills up, new events are dropped (better to lose a progress update than to block the worker).

### Pydantic
A Python library for data validation using type hints. `api/contracts.py` uses Pydantic to define the exact shape of every HTTP request/response and every WebSocket message. If incoming data does not match the schema, Pydantic raises a clear error immediately rather than letting bad data propagate.

### Vite
A frontend build tool and development server for JavaScript/TypeScript projects. Starts almost instantly (unlike webpack), serves files with hot module replacement (changes appear in the browser without a full reload), and produces optimised bundles for production.

### React
A JavaScript library for building user interfaces. The frontend is built as a React **single-page application** — the browser loads once and React updates the page as data changes, without full page reloads. Each page (Ingest, Stats, Generate, DB) is a React component.

### TypeScript
JavaScript with type annotations. Catches type mismatches at development time rather than at runtime. The frontend is written in strict TypeScript — the shape of every API response is defined as an interface and enforced at compile time.

### React Query (`@tanstack/react-query`)
A library that manages server state in React — fetching, caching, and refreshing data from the API. Instead of writing `fetch()` calls and `useState` hooks manually, React Query handles loading states, error states, background refetching, and cache invalidation automatically.

### Tailwind CSS
A CSS framework that provides utility classes (`text-green-600`, `flex`, `p-4`) instead of pre-built components. You style elements by combining small classes directly in HTML/JSX rather than writing separate CSS files.

### Recharts
A React charting library. Used on the Stats page to draw the accuracy-over-time line graphs for each of the 13 models. Declarative — you describe what the chart should look like and Recharts renders it.

### Lucide React
An icon library. Provides clean, consistent SVG icons as React components. Used throughout the frontend for UI icons (upload, pause, reset, etc.).

### Playwright
A browser automation framework used for end-to-end testing. Track Z uses Playwright to open a real browser, upload a file, wait for the ingest to complete, and verify that all 13 model cards appear on the Stats page — testing the entire system together rather than in isolation.

---

## Part 7 — The Tracks Explained

Each **track** is a self-contained unit of work assigned to one AI agent. Tracks have dependencies — later tracks build on earlier ones. Here is what each track builds and why it exists.

### Track 0 — Foundations
**Builds:** The three modules everything else depends on.
- **WVM** — the word checker. Without this, we cannot measure accuracy.
- **DB/Store** — the database layer. Without this, nothing can persist between sessions.
- **Contracts** — the agreed data shapes. Without this, the frontend and backend would use incompatible formats.

This track must be done first. Every other track imports from one of these three modules.

### Track A — Markov Models
**Builds:** All 11 Markov models (character n-grams 1–5, word n-grams 1–3, BPE n-grams 1–3).

These are the "primitive" models. They demonstrate that even simple frequency counting produces better-than-random text — the key educational point is that *pattern recognition does not require neural networks*.

### Track B1 — Feedforward Neural Network
**Builds:** Model 12 — the first neural model.

Introduces PyTorch, training loops, embeddings, and checkpoints. Demonstrates the step up from "count tables" (Markov) to "learned weights" (neural nets). The model sees a 64-token window and produces text that is noticeably more coherent than n-grams after sufficient training.

### Track B2 — Transformer
**Builds:** Model 13 — the attention-based model.

The architectural centrepiece of the project. Demonstrates why attention matters: unlike the feedforward model that flattens context into one vector, the Transformer can selectively focus on relevant earlier tokens. This is the mechanism behind modern LLMs. Assigned to **Opus 4.7** because causal masking bugs are silent — a broken mask produces grammatically plausible but conceptually wrong output.

### Track C — Monte Carlo Evaluator
**Builds:** The accuracy measurement system.

Runs 50 text generations from each of the 13 models, scores them via WVM, and stores the results in DuckDB. This is what populates the Stats page graphs and what the ingest pipeline uses to decide whether to keep or reject a chunk.

### Track D1 — Ingest Worker
**Builds:** The multiprocessing orchestration layer.

The most complex module. It runs as a separate process, coordinates all 13 models through a full training cycle per chunk, implements the rollback mechanism, handles pause signals, and streams progress events to the parent process via a queue. Assigned to **Opus 4.7** because getting multiprocessing, transactions, and rollback all correct simultaneously is where subtle bugs hide.

### Track D2 — FastAPI Backend
**Builds:** The HTTP and WebSocket API.

Exposes the work done in D1 (and all upstream tracks) over a network interface that the frontend can call. Handles file uploads, forwards pause commands to the worker, streams progress events to WebSocket clients, and provides read-only DB inspection endpoints.

### Track E — React Frontend
**Builds:** The entire user interface — all four pages.

- **Ingest page** — drag-and-drop file upload, live progress stream, pause button
- **Stats page** — 13 accuracy graphs, last generated text per model, live updates
- **Generate page** — on-demand generation from all 13 models with optional spell correction
- **DB page** — browsable tables, row counts, reset button

### Track Z — Integration
**Builds:** The connective tissue — startup scripts, README, end-to-end tests.

Verifies that all the pieces actually work together as a system. The Playwright test uploads a real file, watches all 13 models update, and confirms the browser UI reflects the changes. Nothing in this track is novel — it is the quality gate.

---

## Part 8 — Infrastructure and Dev Tools

### WSL2 (Windows Subsystem for Linux 2)
A compatibility layer that lets Windows run a real Linux kernel in a lightweight virtual machine. The entire Python backend runs inside WSL2 (Ubuntu), giving access to Linux-native tools like `tmux`, `bash`, and the Linux versions of PyTorch and DuckDB.

### Virtual Environment (`venv`)
An isolated Python installation for this project. Prevents the project's dependencies (torch, duckdb, fastapi…) from conflicting with packages installed elsewhere on the system. Always activate with `source venv/bin/activate` before running anything.

### PyTorch
The deep learning framework used for Models 12 and 13. Provides tensor operations, automatic differentiation (for computing gradients), neural network building blocks (`nn.Module`, `nn.Linear`, `nn.Embedding`), optimisers (Adam), and GPU acceleration via CUDA.

### CUDA
NVIDIA's parallel computing platform. When a compatible NVIDIA GPU is available, PyTorch can offload tensor operations to it — training that takes 60 seconds on CPU might take 6 seconds on GPU. The project detects CUDA automatically and falls back to CPU if unavailable.

### pytest
The Python testing framework. Every module has a matching test file (`model/feedforward.py` → `tests/test_feedforward.py`). Tests run with `pytest tests/` and must all pass before a track is marked complete.

### uvicorn
An async Python web server that runs the FastAPI application. Supports hot-reload during development (`--reload` flag) so changes to Python files are reflected immediately without restarting the server.

---

## Quick Reference — Abbreviations

| Abbreviation | Stands for | One-line explanation |
|---|---|---|
| LLM | Large Language Model | Neural network trained to predict text at scale |
| NLP | Natural Language Processing | The field of making computers understand text |
| BPE | Byte Pair Encoding | Subword tokenization algorithm used by GPT models |
| WVM | Word Verification Module | Our custom spell-checker wrapper around SCOWL |
| SCOWL | Spell Checker Oriented Word Lists | The English word list we use for accuracy scoring |
| MC | Monte Carlo | The 50-run random-sample evaluation method |
| KN | Kneser-Ney | The smoothing algorithm for Markov models |
| DB | Database | Here: DuckDB, stored in `db/llm_evolution.duckdb` |
| API | Application Programming Interface | The HTTP endpoints the frontend calls |
| WS | WebSocket | The persistent connection for live progress streaming |
| GPU | Graphics Processing Unit | Hardware that accelerates neural network training |
| CUDA | Compute Unified Device Architecture | NVIDIA's GPU programming platform |
| pt | PyTorch | File extension for saved neural network weights |
| venv | Virtual Environment | Isolated Python package installation |
