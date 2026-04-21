"""
Character-level n-gram model for the Markov chain visualizer.

Builds count tables for orders 1–4 from the demo corpus, normalises to
probabilities, and provides:
  - predict(prefix)               → distribution over next chars with backoff
  - monte_carlo_walks(...)        → multiple random walks from a prefix
  - compute_entropy(dist)         → Shannon entropy in bits
  - get_bigram_matrix(tables, ...) → transition probabilities for a char set
"""
from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path

MAX_ORDER = 4

# Type aliases
CountTable = dict[str, Counter[str]]   # context_str → Counter(next_char → count)
ProbTable = dict[str, dict[str, float]]  # context_str → {next_char: prob}


def build_char_tables(
    corpus_path: Path | str,
    max_order: int = MAX_ORDER,
) -> dict[int, ProbTable]:
    """
    Read the corpus and build normalised probability tables for orders 1..max_order.

    Order n means: use (n-1) chars of context to predict the n-th char.
    Order 1 = unigram (no context), order 2 = bigram (1 char context), etc.
    """
    text = Path(corpus_path).read_text(encoding="utf-8", errors="replace")
    # Lowercase for consistency; keeps punctuation + spaces
    text = text.lower()

    # Count n-grams for each order
    counts: dict[int, CountTable] = {n: {} for n in range(1, max_order + 1)}

    for i in range(len(text)):
        ch = text[i]
        for n in range(1, max_order + 1):
            ctx_len = n - 1  # context length
            if i < ctx_len:
                continue
            ctx = text[i - ctx_len: i]
            if ctx not in counts[n]:
                counts[n][ctx] = Counter()
            counts[n][ctx][ch] += 1

    # Normalise to probabilities
    tables: dict[int, ProbTable] = {}
    for n, ctable in counts.items():
        prob_table: ProbTable = {}
        for ctx, counter in ctable.items():
            total = sum(counter.values())
            prob_table[ctx] = {ch: c / total for ch, c in counter.most_common()}
        tables[n] = prob_table

    return tables


def predict(
    tables: dict[int, ProbTable],
    prefix: str,
    max_order: int = MAX_ORDER,
) -> tuple[dict[str, float], int]:
    """
    Predict the next character distribution given a prefix string.

    Uses backoff: try the highest order that has the context, then back off
    to shorter contexts until we find a match.

    Returns (distribution, order_used).
    """
    prefix = prefix.lower()

    for n in range(min(max_order, len(prefix) + 1), 0, -1):
        ctx_len = n - 1
        ctx = prefix[-ctx_len:] if ctx_len > 0 else ""
        table = tables.get(n, {})
        if ctx in table:
            return table[ctx], n

    # Absolute fallback: uniform over common ASCII
    chars = "abcdefghijklmnopqrstuvwxyz .,;:!?'-"
    uniform = {ch: 1.0 / len(chars) for ch in chars}
    return uniform, 0


# ---------------------------------------------------------------------------
# Entropy & transition matrix helpers
# ---------------------------------------------------------------------------

def compute_entropy(dist: dict[str, float]) -> float:
    """
    Shannon entropy of a probability distribution in bits.

    H = -Σ p_i * log2(p_i)

    Returns 0.0 for a degenerate distribution and log2(n) at maximum
    uncertainty (uniform over n outcomes).
    """
    return -sum(p * math.log2(p) for p in dist.values() if p > 0.0)


def get_bigram_matrix(
    tables: dict[int, ProbTable],
    top_chars: list[str],
) -> dict[str, dict[str, float]]:
    """
    Return the bigram (order-2) transition distributions for each char in
    top_chars.  The result maps each source char to its distribution over
    destination chars (restricted to the top_chars set so the matrix is
    square and self-contained).

    Any missing context falls back to an empty dict (no data for that row).
    """
    bigram_table: ProbTable = tables.get(2, {})
    dest_set = set(top_chars)
    matrix: dict[str, dict[str, float]] = {}

    for src in top_chars:
        row = bigram_table.get(src, {})
        if not row:
            matrix[src] = {}
            continue
        # Keep only destinations that are in top_chars, then renormalise
        filtered = {dst: p for dst, p in row.items() if dst in dest_set}
        total = sum(filtered.values())
        if total > 0:
            matrix[src] = {dst: p / total for dst, p in filtered.items()}
        else:
            # Fall back to the full row so we always show something
            matrix[src] = dict(sorted(row.items(), key=lambda x: x[1], reverse=True)[:len(top_chars)])

    return matrix


def _weighted_choice(
    dist: dict[str, float],
    rng: random.Random,
) -> str:
    """Sample one character from a probability distribution."""
    chars = list(dist.keys())
    weights = list(dist.values())
    return rng.choices(chars, weights=weights, k=1)[0]


def monte_carlo_walks(
    tables: dict[int, ProbTable],
    prefix: str,
    n_walkers: int = 20,
    walk_length: int = 30,
    seed: int | None = None,
    max_order: int = MAX_ORDER,
) -> list[list[str]]:
    """
    Run n_walkers independent random walks, each starting from prefix and
    extending walk_length characters by sampling from the char n-gram model.

    Returns a list of walks, each walk being a list of characters.
    """
    rng = random.Random(seed)
    walks: list[list[str]] = []

    for _ in range(n_walkers):
        current = prefix.lower()
        path: list[str] = []
        for _ in range(walk_length):
            dist, _ = predict(tables, current, max_order=max_order)
            ch = _weighted_choice(dist, rng)
            path.append(ch)
            current += ch
        walks.append(path)

    return walks


# ---------------------------------------------------------------------------
# Singleton loader (used by the Flask app)
# ---------------------------------------------------------------------------

_tables: dict[int, ProbTable] | None = None
CORPUS_PATH = Path(__file__).parent / "demo_corpus.txt"


def get_tables() -> dict[int, ProbTable]:
    """Load or return cached character n-gram tables."""
    global _tables
    if _tables is not None:
        return _tables

    if not CORPUS_PATH.exists():
        raise RuntimeError(
            "demo_corpus.txt not found. Run: python3 demo/setup_demo.py"
        )

    print("Building character n-gram tables…", flush=True)
    _tables = build_char_tables(CORPUS_PATH)
    for n, tbl in _tables.items():
        print(f"  order {n}: {len(tbl):,} contexts", flush=True)
    return _tables
