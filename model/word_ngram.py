"""Word n-gram language model with DuckDB persistence.

Supports orders 1–3. Words are tokenised via wvm.validator.Validator so that
training and validation use the same tokenisation. Counts are stored in the
word_ngrams table and queried on demand.
"""
from __future__ import annotations

import random

from db.store import Store
from model.ngram_counter import _count_ngrams_generic
from wvm.validator import Validator


def _word_context_str(ngram_prefix: tuple[str, ...]) -> str:
    """Join a tuple of words into the context string stored in DB."""
    return " ".join(ngram_prefix)


class WordNGramModel:
    """Word-level n-gram model backed by DuckDB word_ngrams table."""

    def __init__(self, order: int, store: Store, validator: Validator) -> None:
        """Initialise the model.

        Args:
            order:     N-gram order, 1–3.
            store:     Open Store instance (DuckDB connection).
            validator: Shared Validator instance for tokenisation.
        Raises:
            ValueError: If order is outside the valid range.
        """
        if not 1 <= order <= 3:
            raise ValueError(f"order must be between 1 and 3; got {order}")
        self._order = order
        self._store = store
        self._validator = validator

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on(self, text: str) -> None:
        """Tokenise *text* and update word_ngrams table.

        Counts all word n-grams up to self._order from the tokenised text and
        upserts the deltas into DuckDB.

        Args:
            text: Raw text chunk to learn from.
        """
        if not text:
            return

        tokens: list[str] = self._validator.tokenize(text)
        if not tokens:
            return

        all_counts = _count_ngrams_generic(tokens, self._order)

        with self._store.transaction():
            for n, counter in all_counts.items():
                rows: list[tuple[str, str, int]] = []
                for ngram, count in counter.items():
                    context = _word_context_str(ngram[:-1])
                    next_word = ngram[-1]
                    rows.append((context, next_word, count))
                if rows:
                    self._store.upsert_ngrams("word", n, rows)

    # ------------------------------------------------------------------
    # Probability queries
    # ------------------------------------------------------------------

    def next_word_distribution(
        self, context: tuple[str, ...]
    ) -> dict[str, float]:
        """Return a smoothed probability distribution over next words.

        Queries the DB starting at the highest available order and backs off
        to shorter contexts. Applies add-ε smoothing over the observed
        vocabulary so the result always sums to 1.

        Args:
            context: Preceding words (only the last order-1 are used).

        Returns:
            {word: probability} mapping; empty dict only if DB has no data.
        """
        # Fetch word vocabulary from unigrams (n=1, context="")
        vocab_counts: dict[str, int] = self._store.get_distribution("word", 1, "")
        vocab = set(vocab_counts.keys())

        if not vocab:
            return {}

        # Trim context to at most order-1 words
        ctx_words = context[-(self._order - 1):] if self._order > 1 else ()

        for n in range(self._order, 0, -1):
            actual_ctx_words = ctx_words[-(n - 1):] if n > 1 else ()
            actual_ctx = _word_context_str(actual_ctx_words)
            raw: dict[str, int] = self._store.get_distribution("word", n, actual_ctx)
            if raw:
                return _normalise_with_smoothing(raw, vocab)

        # Fallback: pure unigram distribution
        return _normalise_with_smoothing(vocab_counts, vocab)

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def generate(
        self,
        n_words: int,
        seed_context: tuple[str, ...] = (),
        rng: random.Random | None = None,
    ) -> str:
        """Generate *n_words* words and return them space-joined.

        Args:
            n_words:      Number of words to generate.
            seed_context: Optional tuple of seed words.
            rng:          Random generator instance (creates new one if None).

        Returns:
            Space-joined generated words.
        """
        if rng is None:
            rng = random.Random()

        context: tuple[str, ...] = seed_context
        result: list[str] = []

        for _ in range(n_words):
            dist = self.next_word_distribution(context)
            if not dist:
                break
            words = list(dist.keys())
            weights = list(dist.values())
            next_word = rng.choices(words, weights=weights, k=1)[0]
            result.append(next_word)
            context = (context + (next_word,))[-(self._order - 1):]

        return " ".join(result)

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

    Every word in *vocab* receives at least *eps* probability mass.
    The result always sums to 1.0 (within floating-point precision).
    """
    dist: dict[str, float] = {}
    for word in vocab:
        dist[word] = float(raw.get(word, 0)) + eps
    for word, c in raw.items():
        if word not in dist:
            dist[word] = float(c) + eps
    total = sum(dist.values())
    return {word: p / total for word, p in dist.items()}
