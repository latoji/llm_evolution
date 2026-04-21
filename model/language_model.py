"""
Smoothed n-gram language model for BPE token sequences.

Wraps KNSmoothing and exposes high-level probability/distribution APIs used
by the generator and evaluation code. Supports loading from either a pickle
file (legacy JSON pipeline) or directly from DuckDB (new DB pipeline).
"""
from __future__ import annotations

import math
import pickle
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Literal

from model.smoothing import KNSmoothing

if TYPE_CHECKING:
    from db.store import Store

MODEL_PATH = Path(__file__).parent / "lm.pkl"


class LanguageModel:
    """
    Smoothed n-gram language model.

    Usage
    ─────
        lm = LanguageModel.load()
        p = lm.prob(token_id, context=(t1, t2, t3))
        dist = lm.next_token_distribution(context)
    """

    def __init__(self, smoothing: KNSmoothing) -> None:
        self._kn = smoothing

    # -----------------------------------------------------------------------
    # Probability queries
    # -----------------------------------------------------------------------

    def prob(self, token: int, context: tuple[int, ...]) -> float:
        """
        P(token | context) with MKN smoothing.
        context: tuple of preceding token IDs (up to 3).
        """
        return self._kn.prob(token, context)

    def next_token_distribution(
        self,
        context: tuple[int, ...],
    ) -> dict[int, float]:
        """
        Return the full probability distribution P(· | context) over all vocab IDs.
        The distribution is guaranteed to be non-negative and sum to ~1.0.
        """
        vocab_size = self._kn.vocab_size
        dist: dict[int, float] = {}

        for token_id in range(vocab_size):
            p = self._kn.prob(token_id, context)
            if p > 0:
                dist[token_id] = p

        # Renormalise to correct floating-point drift
        total = sum(dist.values())
        if total > 0:
            dist = {t: p / total for t, p in dist.items()}

        return dist

    # -----------------------------------------------------------------------
    # Perplexity
    # -----------------------------------------------------------------------

    def perplexity(
        self,
        token_stream: Iterator[int],
        max_order: int = 3,
    ) -> float:
        """
        Compute perplexity on a token stream.
        Lower is better. Typical range for this architecture: 200–800.
        """
        log_prob_sum = 0.0
        n_tokens = 0
        history: list[int] = []

        for token in token_stream:
            context = tuple(history[-max_order:])
            p = self.prob(token, context)
            log_prob_sum += math.log(max(p, 1e-300))
            n_tokens += 1
            history.append(token)
            if len(history) > max_order:
                history.pop(0)

        if n_tokens == 0:
            return float("inf")

        avg_log = log_prob_sum / n_tokens
        return math.exp(-avg_log)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._kn.vocab_size

    @property
    def max_order(self) -> int:
        return max(self._kn.counts.keys()) if self._kn.counts else 1

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def save(self, path: Path | str = MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: Path | str = MODEL_PATH) -> "LanguageModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run model/build_model.py first."
            )
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected LanguageModel, got {type(obj)}")
        return obj

    @classmethod
    def from_counts(
        cls,
        counts: dict[int, Counter],
        vocab_size: int,
    ) -> "LanguageModel":
        """Build a LanguageModel directly from raw counts."""
        kn = KNSmoothing.from_counts(counts, vocab_size)
        return cls(kn)

    @classmethod
    def from_store(
        cls,
        store: "Store",
        family: Literal["bpe"],
        max_order: int,
    ) -> "LanguageModel":
        """Load n-gram counts from DuckDB and return a trained LanguageModel.

        Reads all token_ngrams rows for orders 1..max_order from the given
        store and builds a KNSmoothing-backed LanguageModel. Intended for
        BPE models (models 9–11) where token IDs are integers.

        Args:
            store:     Open Store instance connected to the project DuckDB.
            family:    Must be 'bpe' (only BPE token IDs are integer-typed).
            max_order: Maximum n-gram order to load (1–4).

        Returns:
            A fully initialised LanguageModel ready for generation.

        Raises:
            ValueError: If no n-gram data is found in the DB for the given family.
        """
        if family != "bpe":
            raise ValueError(
                f"from_store only supports family='bpe'; got {family!r}"
            )

        counts: dict[int, Counter] = {}
        for n in range(1, max_order + 1):
            counter: Counter = Counter()
            # Paginate through all rows for this order
            page_size = 50_000
            offset = 0
            while True:
                rows = store.get_ngrams("bpe", n, limit=page_size, offset=offset)
                if not rows:
                    break
                for row in rows:
                    ctx_str: str = row["context"]
                    context_ids = (
                        tuple(int(x) for x in ctx_str.split())
                        if ctx_str
                        else ()
                    )
                    next_id = int(row["next_item"])
                    ngram = context_ids + (next_id,)
                    counter[ngram] = int(row["count"])
                offset += page_size
                if len(rows) < page_size:
                    break
            if counter:
                counts[n] = counter

        if not counts:
            raise ValueError(
                "No BPE n-gram data found in DuckDB. "
                "Run the ingest pipeline or migrate from JSON first."
            )

        # Infer vocab_size from vocabulary table; fall back to max token ID + 1
        vocab_rows = store.get_vocabulary("bpe", limit=1)
        if vocab_rows:
            # Fetch total vocabulary count by reading with a large limit
            all_vocab = store.get_vocabulary("bpe", limit=200_000)
            vocab_size = len(all_vocab)
        else:
            # Derive from the maximum token ID observed in unigrams
            unigrams = counts.get(1, Counter())
            if unigrams:
                vocab_size = max(ngram[-1] for ngram in unigrams) + 1
            else:
                vocab_size = max(
                    ngram[-1] for n_counts in counts.values() for ngram in n_counts
                ) + 1

        return cls.from_counts(counts, vocab_size)
