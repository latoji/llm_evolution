# Track Specs

This folder contains per-track implementation specs for the build plan in `../PLAN.md`.

**Rules for agents:**
- Read only the spec for the track you have claimed. Ignore the others — they are someone else's job.
- If you need information about another track's interface, it will be in your own spec under "Upstream dependencies" or "Downstream consumers".
- `../SPEC.md` is the architectural source of truth and is safe to read fully.

## Index

| Track | File | Owns | Model |
|---|---|---|---|
| 0 | [00-foundations.md](00-foundations.md) | WVM, DuckDB layer, API contracts | `claude-sonnet-4-6` |
| A | [01-markov-models.md](01-markov-models.md) | All 11 Markov models (char/word/BPE) | `claude-sonnet-4-6` |
| B1 | [02-feedforward.md](02-feedforward.md) | Model 12 (feedforward neural net) | `claude-sonnet-4-6` |
| B2 | [03-transformer.md](03-transformer.md) | Model 13 (Transformer) | `claude-opus-4-7` |
| C | [04-monte-carlo.md](04-monte-carlo.md) | 50-run accuracy evaluator | `claude-opus-4-6` |
| D1 | [05-ingest-worker.md](05-ingest-worker.md) | Multiprocessing orchestration | `claude-opus-4-7` |
| D2 | [06-fastapi-backend.md](06-fastapi-backend.md) | HTTP routes + WebSocket | `claude-opus-4-6` |
| E | [07-frontend.md](07-frontend.md) | React + Vite app, all 4 pages | `claude-sonnet-4-6` |
| Z | [99-integration.md](99-integration.md) | End-to-end wiring and smoke test | `claude-opus-4-6` |

## Spec File Structure

Every spec file follows the same structure:

1. **Scope** — what you own in one paragraph
2. **Upstream dependencies** — what must exist before you start
3. **Downstream consumers** — who will use your code
4. **Files owned** — exact paths you may create or modify
5. **Implementation** — file-by-file detail, with signatures and key logic
6. **Testing** — what tests to write and what they verify
7. **Acceptance criteria** — checklist to know you are done
8. **Pitfalls** — things that commonly go wrong
9. **Model assignment** — Sonnet or Opus, with rationale
