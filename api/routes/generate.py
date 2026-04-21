"""Generation endpoint — one text sample per model, with optional auto-correct.

Runs synchronous model generation in a thread executor so the async event
loop is never blocked.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from fastapi import APIRouter

from api import state
from api.contracts import GenerateRequest, GenerateResponse, ModelOutput
from eval.monte_carlo import MODELS, ModelSpec, _load_bpe_vocab_from_store
from wvm.validator import Validator

router = APIRouter()

# Scaling factors to translate word_count → model-specific token / char budget.
# Calibrated for ~100 words → 600 chars (char models) or 200 tokens (others).
_CHARS_PER_100_WORDS: int = 600
_TOKENS_PER_100_WORDS: int = 200


@router.post("", response_model=GenerateResponse)
async def generate_text(req: GenerateRequest) -> GenerateResponse:
    """Generate one text sample from each of the 13 LLM models.

    Heavy computation runs in a thread executor.  ``auto_correct=True``
    replaces non-SCOWL words with the nearest valid suggestion (demo feature).
    """
    loop = asyncio.get_running_loop()
    outputs = await loop.run_in_executor(
        None, _run_all_models, req.word_count, req.auto_correct
    )
    return GenerateResponse(outputs=outputs)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run_all_models(word_count: int, auto_correct: bool) -> list[ModelOutput]:
    """Generate and score one sample per model. Synchronous — run in executor."""
    validator = Validator()
    outputs: list[ModelOutput] = []

    for spec in MODELS:
        raw_text = _generate_one(spec, state.store, validator, word_count)

        if raw_text and raw_text.strip():
            word_results_raw, pct = validator.validate(raw_text)
            word_results: list[tuple[str, bool]] = [
                (r.word, r.is_real) for r in word_results_raw
            ]
        else:
            word_results, pct = [], 0.0

        corrected: str | None = None
        if auto_correct and word_results:
            corrected = _autocorrect(word_results, validator)

        outputs.append(
            ModelOutput(
                model_name=spec.name,
                raw_text=raw_text,
                corrected_text=corrected,
                word_results=word_results,
                real_word_pct=pct,
            )
        )

    return outputs


# ---------------------------------------------------------------------------
# Per-family generation helpers
# ---------------------------------------------------------------------------


def _generate_one(
    spec: ModelSpec,
    store: Any,
    validator: Validator,
    word_count: int,
) -> str:
    """Dispatch to the correct model family. Returns empty string on any error."""
    try:
        if spec.family == "char":
            return _generate_char(spec, store, word_count)
        if spec.family == "word":
            return _generate_word(spec, store, validator, word_count)
        if spec.family == "bpe":
            return _generate_bpe(spec, store, word_count)
        return _generate_neural(spec, store, word_count)
    except Exception:
        return ""


def _generate_char(spec: ModelSpec, store: Any, word_count: int) -> str:
    """Generate from a character n-gram model."""
    from model.char_ngram import CharNGramModel

    model = CharNGramModel(spec.order, store)  # type: ignore[arg-type]
    n_chars = max(1, int(word_count * _CHARS_PER_100_WORDS / 100))
    return model.generate(n_chars, rng=random.Random())


def _generate_word(
    spec: ModelSpec,
    store: Any,
    validator: Validator,
    word_count: int,
) -> str:
    """Generate from a word n-gram model."""
    from model.word_ngram import WordNGramModel

    model = WordNGramModel(spec.order, store, validator)  # type: ignore[arg-type]
    return model.generate(word_count, rng=random.Random())


def _generate_bpe(spec: ModelSpec, store: Any, word_count: int) -> str:
    """Generate from a BPE n-gram model.

    Returns empty string when no BPE data is available in DuckDB.
    """
    from generate.sampling import combined_sample
    from model.language_model import LanguageModel
    from tokenizer.bpe import ENDOFTEXT_ID, decode as bpe_decode

    try:
        bpe_lm = LanguageModel.from_store(store, "bpe", spec.order)  # type: ignore[arg-type]
    except (ValueError, Exception):
        return ""

    vocab = _load_bpe_vocab_from_store(store)
    if not vocab:
        return ""

    n_tokens = max(1, int(word_count * _TOKENS_PER_100_WORDS / 100))
    ctx_window = max(0, (spec.order or 1) - 1)
    context: tuple[int, ...] = ()
    token_ids: list[int] = []
    rng = random.Random()

    for _ in range(n_tokens):
        dist = bpe_lm.next_token_distribution(context)
        if not dist:
            break
        token = combined_sample(dist, rng=rng)
        if token == ENDOFTEXT_ID:
            break
        token_ids.append(token)
        if ctx_window > 0:
            context = (context + (token,))[-ctx_window:]

    return bpe_decode(token_ids, vocab) if token_ids else ""


def _generate_neural(spec: ModelSpec, store: Any, word_count: int) -> str:
    """Generate from a neural model (feedforward or transformer).

    Returns empty string when torch is unavailable or no checkpoint exists.
    This is expected early in training.
    """
    try:
        import torch
        from tokenizer.bpe import decode as bpe_decode

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_tokens = max(1, int(word_count * _TOKENS_PER_100_WORDS / 100))

        if spec.name == "feedforward":
            from model.feedforward import FeedforwardTrainer

            trainer = FeedforwardTrainer(store=store, device=device)
        elif spec.name == "transformer":
            from model.transformer import TransformerTrainer

            trainer = TransformerTrainer(store=store, device=device)
        else:
            return ""

        token_ids = trainer.generate(n_tokens=n_tokens)
        vocab = _load_bpe_vocab_from_store(store)
        return bpe_decode(token_ids, vocab) if (vocab and token_ids) else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Auto-correct helper
# ---------------------------------------------------------------------------


def _autocorrect(word_results: list[tuple[str, bool]], validator: Validator) -> str:
    """Replace non-SCOWL words with the nearest valid suggestion.

    Uses ``Validator.suggest`` (difflib, cutoff=0.6) as an approximation of
    Levenshtein distance ≤ 2.  This is a demo feature — not production
    spell-correction.  Words with no close match are left unchanged.
    """
    corrected: list[str] = []
    for word, is_real in word_results:
        if is_real:
            corrected.append(word)
        else:
            suggestions = validator.suggest(word, n=1)
            corrected.append(suggestions[0] if suggestions else word)
    return " ".join(corrected)
