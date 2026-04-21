"""Character n-gram language model with DuckDB persistence.

Supports orders 1–5. Counts are stored in the char_ngrams table and queried
on demand — no bulk in-memory load at initialisation time.
"""
from __future__ import annotations

import random
from collections import Counter

from db.store import Store
from model.ngram_counter import _count_ngrams_generic


def _char_context_str(ngram_prefix: tuple[str, ...]) -> str:
    """Join a tuple of characters into the context string stored in DB."""
    return "".join(ngram_prefix)


class CharNGramModel:
    """Character-level n-gram model backed by DuckDB char_ngrams table."""

    def __init__(self, order: int, store: Store) -> None:
        """Initialise the model.

        Args:
            order: N-gram order, 1–5.
            store: Open Store instance (DuckDB connection).
        Raises:
            ValueError: If order is outside the valid range.
        """
        if not 1 <= order <= 5:
            raise ValueError(f"order must be between 1 and 5; got {order}")
        self._order = order
        self._store = store

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on(self, text: str) -> None:
        """Update char_ngrams table for this model's order (and lower orders).

        Counts all character n-grams up to self._order from *text* and
        upserts the deltas into DuckDB.

        Args:
            text: Raw text chunk to learn from (whitespace preserved).
        """
        if not text:
            return

        tokens: list[str] = list(text)
        all_counts = _count_ngrams_generic(tokens, self._order)

        with self._store.transaction():
            for n, counter in all_counts.items():
                rows: list[tuple[str, str, int]] = []
                for ngram, count in counter.items():
                    context = _char_context_str(ngram[:-1])
                    next_char = ngram[-1]
                    rows.append((context, next_char, count))
                if rows:
                    self._store.upsert_ngrams("char", n, rows)

    # ------------------------------------------------------------------
    # Probability queries
    # ------------------------------------------------------------------

    def next_char_distribution(self, context: str) -> dict[str, float]:
        """Return a smoothed probability distribution over next characters.

        Queries the DB starting at the highest available order and backs off
        to shorter contexts until a distribution is found. Applies add-ε
        smoothing over the observed vocabulary so the result always sums to 1.

        Args:
            context: Preceding characters (only the last order-1 are used).

        Returns:
            {char: probability} mapping; empty dict only if DB has no data.
        """
        # Fetch character vocabulary from unigrams (n=1, context="")
        vocab_counts: dict[str, int] = self._store.get_distribution("char", 1, "")
        vocab = set(vocab_counts.keys())

        if not vocab:
            return {}

        # Try highest-order context first, backing off to shorter contexts
        ctx_len = self._order - 1
        ctx_window = context[-ctx_len:] if ctx_len > 0 else ""

        for n in range(self._order, 0, -1):
            actual_ctx = ctx_window[-(n - 1):] if n > 1 else ""
            raw: dict[str, int] = self._store.get_distribution("char", n, actual_ctx)
            if raw:
                return _normalise_with_smoothing(raw, vocab)

        # Fallback: pure unigram distribution
        return _normalise_with_smoothing(vocab_counts, vocab)

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def generate(
        self,
        n_chars: int,
        seed_context: str | None = None,
        rng: random.Random | None = None,
    ) -> str:
        """Generate *n_chars* characters by sampling the character distribution.

        Args:
            n_chars:      Number of characters to generate.
            seed_context: Optional seeding context for the first prediction.
            rng:          Random generator instance (creates new one if None).

        Returns:
            Generated text string of length *n_chars* (may be shorter if the
            model has no distribution for the current context).
        """
        if rng is None:
            rng = random.Random()

        context = seed_context or ""
        result: list[str] = []

        for _ in range(n_chars):
            dist = self.next_char_distribution(context)
            if not dist:
                break
            chars = list(dist.keys())
            weights = list(dist.values())
            next_char = rng.choices(chars, weights=weights, k=1)[0]
            result.append(next_char)
            context += next_char

        return "".join(result)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def order(self) -> int:
        """The n-gram order of this model."""
        return self._order


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_with_smoothing(
    raw: dict[str, int],
    vocab: set[str],
    eps: float = 1e-8,
) -> dict[str, float]:
    """Normalise raw counts over vocab with add-ε smoothing.

    Every character in *vocab* receives at least *eps* probability mass.
    The result always sums to 1.0 (within floating-point precision).
    """
    dist: dict[str, float] = {}
    for ch in vocab:
        dist[ch] = float(raw.get(ch, 0)) + eps
    # Add any chars present in raw but not in vocab (shouldn't happen normally)
    for ch, c in raw.items():
        if ch not in dist:
            dist[ch] = float(c) + eps
    total = sum(dist.values())
    return {ch: p / total for ch, p in dist.items()}
