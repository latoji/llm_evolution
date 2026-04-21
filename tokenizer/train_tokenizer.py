"""
Track 1 – BPE Tokenizer
Train the BPE tokenizer on data/clean/train.txt and save merges.json + vocab.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tokenizer.bpe import load_tokenizer, save_tokenizer, train_bpe

TOKENIZER_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data" / "clean"

DEFAULT_CORPUS = DATA_DIR / "train.txt"
DEFAULT_VOCAB_SIZE = 8000
MERGES_PATH = TOKENIZER_DIR / "merges.json"
VOCAB_PATH = TOKENIZER_DIR / "vocab.json"


def main(corpus: Path, vocab_size: int) -> None:
    if not corpus.exists():
        raise FileNotFoundError(
            f"Training corpus not found at {corpus}. "
            "Run data/clean.py and data/split.py first."
        )

    if MERGES_PATH.exists() and VOCAB_PATH.exists():
        print("Tokenizer files already exist. Loading to verify…")
        merges, vocab = load_tokenizer(MERGES_PATH, VOCAB_PATH)
        if len(vocab) == vocab_size:
            print(f"  Tokenizer OK: {len(vocab)} vocab entries, {len(merges)} merges.")
            return
        print("  Vocab size mismatch — retraining.")

    merges, vocab = train_bpe(corpus, vocab_size=vocab_size)
    save_tokenizer(merges, vocab, MERGES_PATH, VOCAB_PATH)

    # Sanity check
    assert len(vocab) == vocab_size, (
        f"Expected vocab_size={vocab_size}, got {len(vocab)}"
    )
    print(f"\nTokenizer trained successfully. Vocab size: {len(vocab)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Path to training text file",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help="Target vocabulary size (default: 8000)",
    )
    args = parser.parse_args()
    main(args.corpus, args.vocab_size)
