"""
Track 3 – Smoothing & Backoff
Modified Kneser-Ney (MKN) smoothing.

Reference: Chen & Goodman (1998) — "An Empirical Study of Smoothing Techniques
for Language Modeling".

Key ideas
──────────
1. Discount: subtract D from each observed count (three levels: D1, D2, D3+).
2. Backoff weight γ(h): redistributes the discounted mass to lower-order model.
3. Continuation probability at unigram level: how many unique left contexts
   precede each token (replaces raw counts for lower-order models).
4. Backoff chain: 4-gram → 3-gram → 2-gram → 1-gram → uniform(1/V)
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Discount values (per order)
# ---------------------------------------------------------------------------

class Discounts(NamedTuple):
    d1: float
    d2: float
    d3plus: float

    def get(self, count: int) -> float:
        if count == 0:
            return 0.0
        if count == 1:
            return self.d1
        if count == 2:
            return self.d2
        return self.d3plus


def compute_discounts(counter: Counter) -> Discounts:
    """
    Compute Modified KN discounts D1, D2, D3+ from a per-order Counter.
    Uses the Chen & Goodman (1998) absolute discounting formula.
    """
    n1 = sum(1 for c in counter.values() if c == 1)
    n2 = sum(1 for c in counter.values() if c == 2)
    n3 = sum(1 for c in counter.values() if c == 3)
    n4 = sum(1 for c in counter.values() if c == 4)

    if n1 == 0 or n2 == 0:
        return Discounts(0.5, 0.5, 0.5)

    Y = n1 / (n1 + 2.0 * n2)
    d1 = max(1.0 - 2.0 * Y * n2 / n1, 0.0)
    d2 = max(2.0 - 3.0 * Y * (n3 / n2 if n2 else 0.0), 0.0)
    d3 = max(3.0 - 4.0 * Y * (n4 / n3 if n3 else 0.0), 0.0)
    return Discounts(d1, d2, d3)


# ---------------------------------------------------------------------------
# Precomputed auxiliary structures for a single order
# ---------------------------------------------------------------------------

@dataclass
class OrderStats:
    """
    All data needed for one n-gram order in MKN smoothing.
    Precomputed from raw counts for efficient query time.
    """
    # Sum of all counts for each prefix context h
    prefix_sum: dict[tuple[int, ...], int] = field(default_factory=dict)
    # Number of distinct continuations with count == 1, 2, >=3 per prefix
    prefix_n1: dict[tuple[int, ...], int] = field(default_factory=dict)
    prefix_n2: dict[tuple[int, ...], int] = field(default_factory=dict)
    prefix_n3plus: dict[tuple[int, ...], int] = field(default_factory=dict)
    # Discount values derived from training distribution
    discounts: Discounts = field(default_factory=lambda: Discounts(0.5, 0.5, 0.5))


def build_order_stats(
    counts: Counter,
    discounts: Discounts,
) -> OrderStats:
    """Precompute prefix aggregates and continuation counts from raw counts."""
    prefix_sum: dict[tuple[int, ...], int] = defaultdict(int)
    prefix_n1: dict[tuple[int, ...], int] = defaultdict(int)
    prefix_n2: dict[tuple[int, ...], int] = defaultdict(int)
    prefix_n3p: dict[tuple[int, ...], int] = defaultdict(int)

    for ngram, c in counts.items():
        h = ngram[:-1]  # prefix (context)
        prefix_sum[h] += c
        if c == 1:
            prefix_n1[h] += 1
        elif c == 2:
            prefix_n2[h] += 1
        else:
            prefix_n3p[h] += 1

    return OrderStats(
        prefix_sum=dict(prefix_sum),
        prefix_n1=dict(prefix_n1),
        prefix_n2=dict(prefix_n2),
        prefix_n3plus=dict(prefix_n3p),
        discounts=discounts,
    )


# ---------------------------------------------------------------------------
# Continuation counts for unigram-level interpolation
# ---------------------------------------------------------------------------

def compute_continuation_counts(bigram_counts: Counter) -> tuple[Counter, int]:
    """
    For each token w, count how many unique left contexts u precede it in bigrams.
    This is N_{1+}(·, w) — the Kneser-Ney continuation count.

    Returns:
        continuation: Counter mapping (token,) → N_{1+}(·,w)
        total: sum of all continuation counts (= total unique bigrams)
    """
    continuation: Counter = Counter()
    for bigram in bigram_counts:
        continuation[(bigram[-1],)] += 1
    total = sum(continuation.values())
    return continuation, total


# ---------------------------------------------------------------------------
# Unified smoothing structure (exported to language_model.py)
# ---------------------------------------------------------------------------

@dataclass
class KNSmoothing:
    """
    All precomputed data for MKN-smoothed n-gram LM.
    Build via KNSmoothing.from_counts().
    """
    counts: dict[int, Counter]                  # raw counts per order
    order_stats: dict[int, OrderStats]          # precomputed prefix stats
    continuation: Counter                        # N_{1+}(·,w): (token,) → int
    total_continuation: int                      # sum of continuation counts
    vocab_size: int

    @classmethod
    def from_counts(
        cls,
        counts: dict[int, Counter],
        vocab_size: int,
    ) -> "KNSmoothing":
        discounts_per_order = {
            n: compute_discounts(counter) for n, counter in counts.items()
        }
        order_stats = {
            n: build_order_stats(counter, discounts_per_order[n])
            for n, counter in counts.items()
        }
        continuation, total_continuation = compute_continuation_counts(
            counts.get(2, Counter())
        )
        return cls(
            counts=counts,
            order_stats=order_stats,
            continuation=continuation,
            total_continuation=total_continuation,
            vocab_size=vocab_size,
        )

    # -----------------------------------------------------------------------
    # Probability queries
    # -----------------------------------------------------------------------

    def prob(self, token: int, context: tuple[int, ...]) -> float:
        """
        P_MKN(token | context).
        context is trimmed to at most (max_order - 1) tokens.
        """
        max_order = max(self.counts.keys()) if self.counts else 1
        ctx = context[-(max_order - 1):] if context else ()
        return self._prob_recursive(token, ctx)

    def _prob_recursive(self, token: int, context: tuple[int, ...]) -> float:
        """Recursive MKN probability with backoff."""
        if not context:
            return self._prob_unigram(token)

        order = len(context) + 1
        stats = self.order_stats.get(order)
        raw_counts = self.counts.get(order)
        if stats is None or raw_counts is None:
            return self._prob_recursive(token, context[1:])

        ngram = context + (token,)
        c_ngram = raw_counts.get(ngram, 0)
        c_prefix = stats.prefix_sum.get(context, 0)

        if c_prefix == 0:
            return self._prob_recursive(token, context[1:])

        d = stats.discounts.get(c_ngram)
        numerator = max(c_ngram - d, 0.0)

        # Backoff weight γ(context)
        n1 = stats.prefix_n1.get(context, 0)
        n2 = stats.prefix_n2.get(context, 0)
        n3p = stats.prefix_n3plus.get(context, 0)
        disc = stats.discounts
        gamma = (disc.d1 * n1 + disc.d2 * n2 + disc.d3plus * n3p) / c_prefix

        p_lower = self._prob_recursive(token, context[1:])
        return numerator / c_prefix + gamma * p_lower

    def _prob_unigram(self, token: int) -> float:
        """
        Kneser-Ney continuation probability at unigram level.
        Falls back to add-ε smoothed count when continuation data is unavailable.
        """
        if self.total_continuation > 0:
            cont = self.continuation.get((token,), 0)
            eps = 1e-6
            return (cont + eps) / (self.total_continuation + eps * self.vocab_size)
        # Fall back to uniform
        return 1.0 / max(self.vocab_size, 1)
