# Track A — Markov Model Layer

**Model**: `claude-sonnet-4-6`
**Dependencies**: Track 0 complete

## Scope

Build the 11 Markov models (character n-grams 1–5, word n-grams 1–3, BPE token n-grams 1–3) on top of the existing `ngram_counter` and `smoothing` modules. Migrate storage from JSON files to DuckDB. Port the character n-gram code out of `demo/` into `model/`.

## Upstream dependencies

- `db/store.py` (Track 0) — all reads/writes go through this
- `wvm/validator.py` (Track 0) — word-mode tokenization
- `tokenizer/bpe.py` — existing, frozen
- `model/smoothing.py` — existing, frozen
- `model/ngram_counter.py` — existing, you will extend it

## Downstream consumers

- **Track C** (Monte Carlo) generates from all 11 of your models
- **Track D1** (Ingest Worker) calls your counters during each chunk

## Files owned

```
model/ngram_counter.py            EXTEND (already exists)
model/char_ngram.py               NEW (refactor from demo/char_ngrams.py)
model/word_ngram.py               NEW
model/language_model.py           EXTEND (already exists)
model/count_ngrams.py             EXTEND (already exists)
model/build_model.py              EXTEND (already exists)
db/migrate_from_json.py           NEW
tests/test_ngram_counter.py       EXTEND (already exists)
tests/test_char_ngram.py          NEW
tests/test_word_ngram.py          NEW
tests/test_language_model.py      NEW
tests/test_migrate.py             NEW
```

## Files you must NOT modify

- `tokenizer/bpe.py`
- `model/smoothing.py`
- `generate/*.py`
- `demo/*`

---

## Implementation

### `model/ngram_counter.py` — Extend

Add a `mode` parameter to the existing class:

```python
from typing import Literal
Mode = Literal["char", "word", "bpe"]

class NGramCounter:
    def __init__(self, max_order: int, mode: Mode, min_count: int = 3) -> None:
        """
        mode='char': tokenize input string into characters (keep whitespace)
        mode='word': tokenize via wvm.validator.Validator.tokenize
        mode='bpe':  caller provides already-tokenized BPE token IDs
        """
```

- Character mode: input is a string; tokens are single characters including whitespace. Max order 5.
- Word mode: input is a string; tokens come from `Validator.tokenize`. Max order 3.
- BPE mode: input is a list of integer token IDs. Max order 3. (Existing behavior.)

Preserve all existing tests. Add new tests for each mode in `test_ngram_counter.py`.

### `model/char_ngram.py` — New (refactor from demo)

Refactor `demo/char_ngrams.py` into a clean DB-backed module:

```python
class CharNGramModel:
    def __init__(self, order: int, store: Store) -> None:
        """order: 1..5. Loads counts from DuckDB char_ngrams table."""

    def train_on(self, text: str) -> None:
        """Update char_ngrams table for this order given a chunk of text."""

    def next_char_distribution(self, context: str) -> dict[str, float]:
        """Return {char: probability} for the given context (last order-1 chars).
        Uses KN smoothing from model/smoothing.py."""

    def generate(self, n_chars: int, seed_context: str | None = None,
                 rng: random.Random | None = None) -> str:
        """Generate n_chars characters, sampling via temperature 1.0.
        Returns generated text."""
```

- Do NOT keep `monte_carlo_walks()` — that role is taken over by `eval/monte_carlo.py` in Track C.
- Backoff to lower orders handled by `smoothing.py`; your job is to populate counts correctly.

### `model/word_ngram.py` — New

Structurally identical to `char_ngram.py` but operates on words:

```python
class WordNGramModel:
    def __init__(self, order: int, store: Store, validator: Validator) -> None:
        """order: 1..3. Loads from word_ngrams table."""

    def train_on(self, text: str) -> None:
        """Tokenize via validator, update word_ngrams."""

    def next_word_distribution(self, context: tuple[str, ...]) -> dict[str, float]: ...

    def generate(self, n_words: int, seed_context: tuple[str, ...] = (),
                 rng: random.Random | None = None) -> str:
        """Return generated text as space-joined words."""
```

### `model/language_model.py` — Extend for DuckDB persistence

The existing `LanguageModel` reads from JSON pickle. Add a new constructor path:

```python
class LanguageModel:
    @classmethod
    def from_store(cls, store: Store, family: Literal["bpe"], max_order: int) -> "LanguageModel":
        """Load n-gram counts from DuckDB instead of JSON. For BPE models 9-11."""
```

Keep the existing JSON constructor for backward compat and for the `demo/` folder.

### `db/migrate_from_json.py` — New

One-time script. Run on startup if `model/counts/*.json` exists:

```python
from pathlib import Path
from db.store import Store

def migrate() -> int:
    """
    If model/counts/ contains JSON files from the existing pipeline,
    read them and insert rows into token_ngrams table.
    Move consumed files to model/counts/archive/.
    Return count of rows migrated.
    Idempotent: if migration already ran (archive exists), do nothing.
    """
```

### `model/count_ngrams.py` and `model/build_model.py` — Extend

Update these orchestration scripts to support the new multi-family build:
- `count_ngrams.py --family {char,word,bpe} --orders 1,2,3`
- `build_model.py --family {char,word,bpe}`

These are CLI entry points; the Ingest Worker in Track D1 does not use them (it uses the classes directly).

---

## Testing

### `tests/test_ngram_counter.py` — Extend

New tests per mode:
- `mode="char"`: feed `"abab"`, order 2, verify `{"ab": {"a": 1, "b": 0}, "ba": {"b": 1}}`
- `mode="word"`: feed `"the cat sat"`, order 2, verify `{("the",): {"cat": 1}, ("cat",): {"sat": 1}}`
- `mode="bpe"`: unchanged behavior — existing tests must still pass

### `tests/test_char_ngram.py`

- Train on known text, verify counts land in `char_ngrams` DuckDB table
- Verify `next_char_distribution` sums to 1.0 (within 1e-6)
- Verify generation produces characters from vocabulary only
- Verify seed_context is respected for first character
- Orders 1, 3, and 5 all instantiate without error

### `tests/test_word_ngram.py`

Mirror of `test_char_ngram.py` but for word models. Test boundary-punctuation stripping via shared Validator.

### `tests/test_language_model.py`

- `LanguageModel.from_store` produces a functional LM
- `next_token_distribution` returns non-zero probability for every vocab token (KN smoothing property)
- Perplexity on known validation text is finite

### `tests/test_migrate.py`

- Create a temp directory with fake JSON count files; run migrate; verify rows in DuckDB
- Run migrate twice: second run is a no-op

---

## Acceptance criteria

- [ ] `pytest tests/test_ngram_counter.py tests/test_char_ngram.py tests/test_word_ngram.py tests/test_language_model.py tests/test_migrate.py` all green
- [ ] Existing tests (tokenizer, smoothing, generator, sampling) still pass unchanged
- [ ] All 11 model classes can be instantiated, trained on a small corpus, and generate sensible-looking text
- [ ] DuckDB contains expected row counts after training on a 10 KB test corpus

---

## Pitfalls

- **Do not change `model/smoothing.py`.** If you think you need to, you're wrong — the KN smoother handles backoff for you regardless of source.
- **Character mode includes whitespace as a token.** This is correct. Do not strip spaces.
- **Word mode must use the shared Validator** from Track 0, not a local tokenizer. This guarantees the tokenization used for training matches the tokenization used for validation.
- **BPE mode still uses integer IDs**, not strings. The existing counter interface expects lists of ints for BPE.
- **Do not load all n-gram counts into memory** when initializing a model. Query the distribution on demand via `store.get_distribution`. These tables will be large.
- **The demo/ folder is frozen.** Do not edit `demo/char_ngrams.py` — copy logic into the new `model/char_ngram.py` and leave the demo file alone.

---

## Model assignment

**Sonnet 4.6.** Pattern-following work with established module templates.
