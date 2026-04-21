"""
N-Gram Frequency Extraction — CLI entry point.

Supports three n-gram families: BPE tokens (original pipeline), character
n-grams, and word n-grams. Counts are saved to JSON for BPE (legacy) or
upserted into DuckDB for char/word families.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from model.ngram_counter import (
    COUNTS_DIR,
    MAX_N,
    PRUNE_MIN_COUNT,
    NGramCounter,
    _token_stream_from_file,
    count_ngrams,
    prune_counts,
    save_counts,
)
from tokenizer.bpe import load_tokenizer

TOKENIZER_DIR = Path(__file__).parent.parent / "tokenizer"
CORPUS_PATH = Path(__file__).parent.parent / "data" / "clean" / "train.txt"

Family = Literal["bpe", "char", "word"]


def _default_max_n(family: Family) -> int:
    """Return the default max order for each family."""
    return {"bpe": 4, "char": 5, "word": 3}[family]


def _parse_orders(orders_str: str) -> list[int]:
    """Parse a comma-separated list of order integers, e.g. '1,2,3'."""
    return [int(o.strip()) for o in orders_str.split(",") if o.strip()]


def main(
    corpus: Path = CORPUS_PATH,
    counts_dir: Path = COUNTS_DIR,
    max_n: int = MAX_N,
    min_count: int = PRUNE_MIN_COUNT,
    family: Family = "bpe",
    orders: list[int] | None = None,
) -> None:
    """Count n-grams from a corpus.

    For family='bpe'  : tokenise via BPE, save JSON counts (legacy path).
    For family='char' : count char n-grams and upsert to DuckDB.
    For family='word' : count word n-grams and upsert to DuckDB.
    """
    actual_max_n = max(orders) if orders else max_n

    if family == "bpe":
        _run_bpe(corpus, counts_dir, actual_max_n, min_count)
    elif family == "char":
        _run_char_or_word(corpus, "char", actual_max_n)
    else:
        _run_char_or_word(corpus, "word", actual_max_n)


def _run_bpe(
    corpus: Path,
    counts_dir: Path,
    max_n: int,
    min_count: int,
) -> None:
    """BPE family: tokenise with BPE merges and save JSON counts."""
    merges_path = TOKENIZER_DIR / "merges.json"
    vocab_path = TOKENIZER_DIR / "vocab.json"

    if not corpus.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus}")
    if not merges_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {merges_path}. Run train_tokenizer.py first."
        )

    print(f"Loading tokenizer from {TOKENIZER_DIR}…")
    merges, _vocab = load_tokenizer(merges_path, vocab_path)

    print(f"Counting BPE {max_n}-grams from {corpus}…")
    stream = _token_stream_from_file(corpus, merges)
    counts = count_ngrams(stream, max_n=max_n)

    for n, counter in counts.items():
        print(f"  order {n}: {sum(counter.values()):,} tokens, {len(counter):,} unique")

    print(f"Pruning n-grams with count < {min_count}…")
    pruned = prune_counts(counts, min_count=min_count)

    for n, counter in pruned.items():
        orig = len(counts[n])
        kept = len(counter)
        print(f"  order {n}: {kept:,}/{orig:,} n-grams kept")

    print("Saving counts…")
    save_counts(pruned, counts_dir)
    print("Done.")


def _run_char_or_word(corpus: Path, family: Literal["char", "word"], max_n: int) -> None:
    """Char/word family: count n-grams and upsert to DuckDB."""
    from db.store import Store  # noqa: PLC0415

    if not corpus.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus}")

    print(f"Counting {family} {max_n}-grams from {corpus}…")
    text = corpus.read_text(encoding="utf-8", errors="replace")
    counter = NGramCounter(max_order=max_n, mode=family)
    counts = counter.count(text)

    for n, c in sorted(counts.items()):
        print(f"  order {n}: {sum(c.values()):,} tokens, {len(c):,} unique")

    print(f"Upserting {family} n-grams into DuckDB…")
    store = Store()
    try:
        with store.transaction():
            for n, ngram_counter in counts.items():
                rows: list[tuple[str, str, int]] = []
                for ngram, count in ngram_counter.items():
                    if family == "char":
                        context = "".join(ngram[:-1])
                    else:
                        context = " ".join(ngram[:-1])
                    next_item = ngram[-1]
                    rows.append((context, next_item, count))
                store.upsert_ngrams(family, n, rows)
    finally:
        store.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count n-grams from corpus")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--counts-dir", type=Path, default=COUNTS_DIR)
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--min-count", type=int, default=PRUNE_MIN_COUNT)
    parser.add_argument(
        "--family",
        choices=["bpe", "char", "word"],
        default="bpe",
        help="N-gram tokenization family (default: bpe)",
    )
    parser.add_argument(
        "--orders",
        type=str,
        default=None,
        help="Comma-separated list of orders to count, e.g. '1,2,3' (overrides --max-n)",
    )
    args = parser.parse_args()
    fam: Family = args.family
    parsed_orders: list[int] | None = _parse_orders(args.orders) if args.orders else None
    effective_max_n = args.max_n or _default_max_n(fam)
    main(args.corpus, args.counts_dir, effective_max_n, args.min_count, fam, parsed_orders)
