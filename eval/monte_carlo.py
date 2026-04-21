"""Monte Carlo accuracy evaluator — 50-run real-word accuracy for all 13 LLM models.

Measures each model's text quality by generating 50 samples of ~100 words each
and scoring them against the SCOWL English word list via WVM.  Results are
persisted to the model_accuracy DuckDB table after every chunk ingestion.

Markov models run in a ProcessPoolExecutor (one task per model, each task runs
all 50 generations).  Neural models run sequentially on the calling process so
they share the trainer's GPU state without contention.
"""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from db.schema import DB_PATH
from db.store import Store
from model.feedforward import FeedforwardTrainer
from model.transformer import TransformerTrainer
from wvm.validator import Validator

# Resolved at import time so worker processes can find the wordlist regardless
# of working directory.
_WVM_WORDLIST: Path = (Path(__file__).parent.parent / "wvm" / "scowl_70.txt").resolve()

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

RUNS_PER_MODEL: int = 50
WORDS_PER_RUN: int = 100

_CHARS_PER_RUN: int = 600    # char models: ~100 words × 6 chars/word
_TOKENS_PER_RUN: int = 200   # bpe / neural: subword tokens → ~100 decoded words


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Descriptor for one of the 13 LLM models in the evaluation pipeline."""

    name: str            # "char_1gram", "feedforward", …
    family: str          # "char" | "word" | "bpe" | "neural"
    order: int | None    # n-gram order for Markov models; None for neural
    display_order: int   # 1–13 — sort key for the Stats page


MODELS: list[ModelSpec] = [
    ModelSpec("char_1gram",  "char",   1,    1),
    ModelSpec("char_2gram",  "char",   2,    2),
    ModelSpec("char_3gram",  "char",   3,    3),
    ModelSpec("char_4gram",  "char",   4,    4),
    ModelSpec("char_5gram",  "char",   5,    5),
    ModelSpec("word_1gram",  "word",   1,    6),
    ModelSpec("word_2gram",  "word",   2,    7),
    ModelSpec("word_3gram",  "word",   3,    8),
    ModelSpec("bpe_1gram",   "bpe",    1,    9),
    ModelSpec("bpe_2gram",   "bpe",    2,    10),
    ModelSpec("bpe_3gram",   "bpe",    3,    11),
    ModelSpec("feedforward", "neural", None, 12),
    ModelSpec("transformer", "neural", None, 13),
]


# ---------------------------------------------------------------------------
# Progress callback protocol
# ---------------------------------------------------------------------------


class ProgressCallback(Protocol):
    """Callable protocol for streaming Monte Carlo progress to the WebSocket layer."""

    def __call__(self, event_type: str, payload: dict) -> None:
        ...
    # event_type ∈ {"mc_model_start", "mc_token", "mc_complete"}


# ---------------------------------------------------------------------------
# BPE vocabulary helper (used by main process and worker processes)
# ---------------------------------------------------------------------------


def _load_bpe_vocab_from_store(store: Store) -> dict[str, int]:
    """Return {token_string: token_id} for BPE decode.

    Reads from DuckDB vocabulary table first; falls back to tokenizer/vocab.json.
    Returns an empty dict when neither source has BPE vocabulary data.
    """
    rows = store.get_vocabulary("bpe", limit=200_000)
    if rows:
        return {row["token"]: row["token_id"] for row in rows}
    vocab_path = Path("tokenizer/vocab.json")
    if vocab_path.exists():
        data: dict = json.loads(vocab_path.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in data.items()}
    return {}


# ---------------------------------------------------------------------------
# Module-level scoring helper (used by tests and workers)
# ---------------------------------------------------------------------------


def _score_sample_text(text: str) -> float:
    """Score *text* against the default SCOWL wordlist; returns 0.0–1.0.

    Creates a Validator on demand.  Intended for testing and lightweight
    one-off scoring outside of a MonteCarloEvaluator instance.
    """
    if not text or not text.strip():
        return 0.0
    validator = Validator(_WVM_WORDLIST)
    _, pct = validator.validate(text)
    return pct


# ---------------------------------------------------------------------------
# Process-pool worker — top-level so ProcessPoolExecutor can pickle it
# ---------------------------------------------------------------------------


def _markov_worker_task(
    spec_name: str,
    family: str,
    order: int | None,
    db_path_str: str,
    wvm_path_str: str,
    n_runs: int,
    words_per_run: int,
) -> tuple[str, list[float]]:
    """Run *n_runs* generations for one Markov model; return (name, [accuracy]).

    Opens its own Store and Validator — DuckDB connections must not be shared
    across processes.  The model is instantiated once before the run loop to
    avoid redundant DB calls; models are cheap to create.
    """
    from model.char_ngram import CharNGramModel
    from model.word_ngram import WordNGramModel
    from model.language_model import LanguageModel
    from generate.sampling import combined_sample
    from tokenizer.bpe import decode as bpe_decode, ENDOFTEXT_ID

    store = Store(Path(db_path_str))
    validator = Validator(Path(wvm_path_str))

    # Pre-load BPE language model once per worker invocation (not per run).
    bpe_lm: LanguageModel | None = None
    bpe_vocab: dict[str, int] = {}
    if family == "bpe":
        try:
            bpe_lm = LanguageModel.from_store(store, "bpe", order)
            bpe_vocab = _load_bpe_vocab_from_store(store)
        except Exception:
            store.close()
            return spec_name, [0.0] * n_runs

    # Instantiate Markov models once before the run loop.
    char_model: CharNGramModel | None = None
    word_model: WordNGramModel | None = None
    if family == "char":
        char_model = CharNGramModel(order, store)
    elif family == "word":
        word_model = WordNGramModel(order, store, validator)

    scores: list[float] = []
    ctx_window: int = max(0, (order or 1) - 1)

    for run in range(n_runs):
        rng = random.Random(run)
        try:
            text: str = ""
            if family == "char" and char_model is not None:
                text = char_model.generate(_CHARS_PER_RUN, rng=rng)
            elif family == "word" and word_model is not None:
                text = word_model.generate(words_per_run, rng=rng)
            else:  # bpe
                assert bpe_lm is not None
                context: tuple[int, ...] = ()
                token_ids: list[int] = []
                for _ in range(_TOKENS_PER_RUN):
                    dist = bpe_lm.next_token_distribution(context)
                    if not dist:
                        break
                    token = combined_sample(dist, rng=rng)
                    if token == ENDOFTEXT_ID:
                        break
                    token_ids.append(token)
                    if ctx_window > 0:
                        context = (context + (token,))[-ctx_window:]
                text = bpe_decode(token_ids, bpe_vocab) if bpe_vocab else ""

            if text and text.strip():
                _, pct = validator.validate(text)
                scores.append(pct)
            else:
                scores.append(0.0)

        except Exception:
            scores.append(0.0)

    store.close()
    return spec_name, scores


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------


class MonteCarloEvaluator:
    """Accuracy evaluator: 50 text generations × 13 models, scored against SCOWL.

    Markov models run in parallel via ProcessPoolExecutor (one task per model,
    CPU-bound, each opens its own DuckDB connection).  Neural models run
    sequentially on the calling process to avoid GPU contention.
    """

    def __init__(
        self,
        store: Store,
        validator: Validator,
        feedforward: FeedforwardTrainer,
        transformer: TransformerTrainer,
        db_path: Path | None = None,
    ) -> None:
        """Caller passes in already-trained model instances.

        Args:
            store:       Open DuckDB Store for the main process.
            validator:   SCOWL-backed word validator.
            feedforward: Trained FeedforwardTrainer (model 12).
            transformer: Trained TransformerTrainer (model 13).
            db_path:     Path to the DuckDB file (for spawning worker processes).
                         Defaults to db.schema.DB_PATH if not given.
        """
        self._store = store
        self._validator = validator
        self._feedforward = feedforward
        self._transformer = transformer
        # Resolve to an absolute path so worker processes can find the file
        # regardless of their working directory.
        resolved_db = (db_path or DB_PATH).resolve()
        self._db_path: str = str(resolved_db)
        self._wvm_path: str = str(_WVM_WORDLIST)

    def evaluate_all(
        self,
        chunk_id: int,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, float]:
        """Run 50 generations × 13 models. Persist accuracy rows to DuckDB.

        Emits progress events via on_progress (optional):
          on_progress("mc_model_start", {"model": "char_3gram"})
          on_progress("mc_complete",    {"model": ..., "accuracy": 0.47, "run": 12})

        Args:
            chunk_id:    ID of the corpus_chunk just ingested (FK for accuracy rows).
            on_progress: Optional callback for streaming live updates.

        Returns:
            {model_name: mean_accuracy} for all 13 models.
        """
        results: dict[str, float] = {}
        markov_specs = [s for s in MODELS if s.family != "neural"]
        neural_specs = [s for s in MODELS if s.family == "neural"]

        # ---- Markov models: parallel across CPU cores ----
        max_workers = max(1, (os.cpu_count() or 4) - 2)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(
                    _markov_worker_task,
                    spec.name,
                    spec.family,
                    spec.order,
                    self._db_path,
                    self._wvm_path,
                    RUNS_PER_MODEL,
                    WORDS_PER_RUN,
                ): spec
                for spec in markov_specs
            }
            for future in as_completed(future_map):
                spec = future_map[future]
                try:
                    model_name, run_scores = future.result()
                except Exception:
                    model_name = spec.name
                    run_scores = [0.0] * RUNS_PER_MODEL

                mean_acc = sum(run_scores) / len(run_scores) if run_scores else 0.0
                results[model_name] = mean_acc

                if on_progress:
                    on_progress("mc_model_start", {"model": model_name})
                    for run_idx, score in enumerate(run_scores):
                        on_progress(
                            "mc_complete",
                            {"model": model_name, "accuracy": score, "run": run_idx},
                        )

        # ---- Neural models: sequential on main process ----
        for spec in neural_specs:
            if on_progress:
                on_progress("mc_model_start", {"model": spec.name})
            run_scores_n: list[float] = []
            for run_idx in range(RUNS_PER_MODEL):
                text = self._generate_sample(spec, random.Random(run_idx))
                score = self._score_sample(text)
                if on_progress:
                    on_progress(
                        "mc_complete",
                        {"model": spec.name, "accuracy": score, "run": run_idx},
                    )
                run_scores_n.append(score)
            results[spec.name] = (
                sum(run_scores_n) / len(run_scores_n) if run_scores_n else 0.0
            )

        # ---- Persist all 13 accuracy rows to DuckDB ----
        for name, acc in results.items():
            self._store.insert_accuracy(name, chunk_id, acc, perplexity=None)

        return results

    def _generate_sample(self, spec: ModelSpec, rng: random.Random) -> str:
        """Generate approximately WORDS_PER_RUN words from a neural model.

        Delegates token generation entirely to the trainer (no torch.* calls
        in this method) and decodes the resulting BPE token IDs via DuckDB
        vocabulary or the legacy tokenizer/vocab.json file.

        Args:
            spec: Neural model descriptor (feedforward or transformer).
            rng:  Seeded Random for interface compatibility (seed used indirectly).

        Returns:
            Decoded text string; empty string if vocab is unavailable.
        """
        if spec.name == "feedforward":
            token_ids: list[int] = self._feedforward.generate(n_tokens=_TOKENS_PER_RUN)
        elif spec.name == "transformer":
            token_ids = self._transformer.generate(n_tokens=_TOKENS_PER_RUN)
        else:
            return ""

        if not token_ids:
            return ""

        vocab = _load_bpe_vocab_from_store(self._store)
        if not vocab:
            return ""

        from tokenizer.bpe import decode as bpe_decode
        return bpe_decode(token_ids, vocab)

    def _score_sample(self, text: str) -> float:
        """Return real-word percentage for *text* via SCOWL validator (0.0–1.0).

        An empty or whitespace-only string scores 0.0.
        """
        if not text or not text.strip():
            return 0.0
        _, pct = self._validator.validate(text)
        return pct
