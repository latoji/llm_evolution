"""
Language model builder — CLI entry point.

For the BPE family, loads JSON n-gram counts and builds a KNSmoothing model
saved to model/lm.pkl. For char/word families, the models query DuckDB on
demand and do not require a separate build step.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from model.language_model import MODEL_PATH, LanguageModel
from model.ngram_counter import COUNTS_DIR, load_counts
from tokenizer.bpe import load_tokenizer

TOKENIZER_DIR = Path(__file__).parent.parent / "tokenizer"

Family = Literal["bpe", "char", "word"]


def main(
    counts_dir: Path = COUNTS_DIR,
    model_path: Path = MODEL_PATH,
    force: bool = False,
    family: Family = "bpe",
) -> LanguageModel | None:
    """Build and (optionally) persist the language model.

    For family='bpe': loads JSON counts, builds KNSmoothing model, saves pickle.
    For family='char'/'word': prints a status message (no build step needed;
        models query DuckDB on demand via CharNGramModel/WordNGramModel).

    Returns the built LanguageModel for family='bpe', or None otherwise.
    """
    if family in ("char", "word"):
        print(
            f"family='{family}' models use DuckDB on demand via "
            f"model/{family}_ngram.py — no separate build step required."
        )
        return None

    # --- BPE path (original behaviour) ---
    if model_path.exists() and not force:
        print(f"Model already exists at {model_path}. Use --force to rebuild.")
        return LanguageModel.load(model_path)

    vocab_path = TOKENIZER_DIR / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocab file not found at {vocab_path}. Run train_tokenizer.py first."
        )

    print("Loading tokenizer vocab…")
    _merges, vocab = load_tokenizer(TOKENIZER_DIR / "merges.json", vocab_path)
    vocab_size = len(vocab)
    print(f"  Vocab size: {vocab_size}")

    print(f"Loading n-gram counts from {counts_dir}…")
    counts = load_counts(counts_dir)
    for n, counter in sorted(counts.items()):
        print(f"  order {n}: {len(counter):,} n-grams")

    if not counts:
        raise RuntimeError(
            f"No count files found in {counts_dir}. Run count_ngrams.py first."
        )

    print("Building language model with Modified Kneser-Ney smoothing…")
    lm = LanguageModel.from_counts(counts, vocab_size)
    print(f"  Max order: {lm.max_order}")
    print(f"  Vocab size: {lm.vocab_size}")

    lm.save(model_path)
    return lm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and serialize the language model")
    parser.add_argument("--counts-dir", type=Path, default=COUNTS_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even if model file exists"
    )
    parser.add_argument(
        "--family",
        choices=["bpe", "char", "word"],
        default="bpe",
        help="N-gram family to build (default: bpe)",
    )
    args = parser.parse_args()
    main(args.counts_dir, args.model_path, args.force, args.family)
