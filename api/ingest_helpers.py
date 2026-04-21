"""Pure helpers used by :mod:`api.ingest_worker`.

Everything here is synchronous and safe to call from either the parent or the
child process. Keeping these functions separate from ``ingest_worker.py``
lets that module focus on multiprocessing orchestration while each helper
stays individually testable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from api.worker_types import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    MIN_REAL_WORD_PCT,
    make_file_rejected,
    safe_put,
)

_FAMILY_TABLE: dict[str, str] = {
    "char": "char_ngrams",
    "word": "word_ngrams",
    "bpe": "token_ngrams",
}


# ---------------------------------------------------------------------------
# File cleaning & pre-screening
# ---------------------------------------------------------------------------


def clean_file_text(path: Path) -> str:
    """Read *path* and apply ``data.clean`` paragraph-level normalisation.

    Returns the concatenation of all non-empty cleaned paragraphs (joined by
    two newlines so the chunker can still recognise paragraph boundaries).
    """
    from data.clean import _PARA_SPLIT, clean_paragraph

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    paragraphs: list[str] = []
    for para in _PARA_SPLIT.split(raw):
        cleaned = clean_paragraph(para)
        if cleaned:
            paragraphs.append(cleaned)
    return "\n\n".join(paragraphs)


def pre_screen_file(
    cleaned: str,
    validator: Any,
    path: Path,
    progress_queue: Any,
    min_pct: float = MIN_REAL_WORD_PCT,
) -> bool:
    """Return True iff *cleaned* passes the SCOWL real-word threshold.

    Emits a ``file_rejected`` event on *progress_queue* when the file fails.
    """
    if not cleaned.strip():
        safe_put(progress_queue, make_file_rejected(str(path), "empty after cleaning"))
        return False
    _, pct = validator.validate(cleaned)
    if pct < min_pct:
        safe_put(
            progress_queue,
            make_file_rejected(
                str(path),
                f"low real-word % ({pct:.1%} < {min_pct:.0%})",
            ),
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    min_size: int = CHUNK_MIN_CHARS,
    max_size: int = CHUNK_MAX_CHARS,
) -> list[str]:
    """Split *text* into whitespace-aligned chunks.

    Each chunk contains roughly ``min_size``–``max_size`` characters.  Paragraph
    boundaries (``\\n\\n``) are preferred over bare whitespace; words are
    never split.  Short inputs (``len(text) <= max_size``) return a single
    chunk.
    """
    if not text:
        return []
    if len(text) <= max_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        if len(text) - start <= max_size:
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break

        end = start + max_size
        # Prefer paragraph break within the target window.
        cut = text.rfind("\n\n", start + min_size, end)
        if cut == -1:
            cut = text.rfind(" ", start + min_size, end)
        if cut == -1:
            cut = end  # hard cut if no whitespace is available

        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = cut + 1 if cut < len(text) else cut

    return chunks


# ---------------------------------------------------------------------------
# N-gram delta management
# ---------------------------------------------------------------------------


def apply_ngram_deltas(
    store: Any,
    family: str,
    tokens: list[Any],
    max_order: int,
) -> dict[int, dict[tuple[str, str], int]]:
    """Count *tokens* up to ``max_order`` and upsert the counts into DuckDB.

    Returns the per-order deltas so a later :func:`revert_deltas` call can
    undo them precisely.
    """
    from model.ngram_counter import _count_ngrams_generic

    if not tokens:
        return {}

    raw = _count_ngrams_generic(tokens, max_order)
    deltas: dict[int, dict[tuple[str, str], int]] = {}
    separator = "" if family == "char" else " "

    with store.transaction():
        for n, counter in raw.items():
            rows: list[tuple[str, str, int]] = []
            delta_n: dict[tuple[str, str], int] = {}
            for ngram, count in counter.items():
                context = separator.join(str(t) for t in ngram[:-1])
                next_item = str(ngram[-1])
                rows.append((context, next_item, count))
                delta_n[(context, next_item)] = count
            if rows:
                store.upsert_ngrams(family, n, rows)
                deltas[n] = delta_n
    return deltas


def revert_deltas(
    store: Any,
    family: str,
    deltas: dict[int, dict[tuple[str, str], int]],
) -> None:
    """Subtract *deltas* and delete any rows that end up with count ≤ 0."""
    if not deltas:
        return

    table = _FAMILY_TABLE[family]

    with store.transaction() as conn:
        for n, delta_n in deltas.items():
            rows = [(ctx, nxt, -count) for (ctx, nxt), count in delta_n.items()]
            if rows:
                store.upsert_ngrams(family, n, rows)
        for n in deltas:
            conn.execute(f"DELETE FROM {table} WHERE n = ? AND count <= 0", [n])


# ---------------------------------------------------------------------------
# Factories for heavyweight components (torch, tokenizer)
# ---------------------------------------------------------------------------


def load_bpe_merges() -> list[tuple[str, str]] | None:
    """Load BPE merges from ``tokenizer/merges.json`` or return ``None`` if absent."""
    merges_path = Path("tokenizer/merges.json")
    vocab_path = Path("tokenizer/vocab.json")
    if not merges_path.exists() or not vocab_path.exists():
        return None
    from tokenizer.bpe import load_tokenizer
    merges, _vocab = load_tokenizer(merges_path, vocab_path)
    return merges


def build_trainers(store: Any) -> dict[str, Any] | None:
    """Instantiate the feedforward and transformer trainers, or ``None`` if torch is missing."""
    try:
        import torch
        from model.feedforward import FeedforwardTrainer
        from model.transformer import TransformerTrainer
    except ImportError:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "feedforward": FeedforwardTrainer(store=store, device=device),
        "transformer": TransformerTrainer(store=store, device=device),
    }


def build_evaluator(store: Any, validator: Any, trainers: dict[str, Any]) -> Any:
    """Construct a :class:`MonteCarloEvaluator` bound to *store* and *trainers*."""
    from eval.monte_carlo import MonteCarloEvaluator
    return MonteCarloEvaluator(
        store=store,
        validator=validator,
        feedforward=trainers["feedforward"],
        transformer=trainers["transformer"],
    )
