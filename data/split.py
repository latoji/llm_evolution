"""
Track 0 – Data Pipeline
Shuffle paragraphs and split 95/5 into data/clean/train.txt and val.txt.
"""
from __future__ import annotations

import random
from pathlib import Path

CLEAN_DIR = Path(__file__).parent / "clean"
TRAIN_RATIO = 0.95
SEED = 42


def split_corpus(
    input_path: Path,
    train_path: Path,
    val_path: Path,
    ratio: float = TRAIN_RATIO,
    seed: int = SEED,
) -> tuple[int, int]:
    """
    Load all paragraphs from input_path, shuffle, and write train/val splits.
    Returns (n_train, n_val).
    """
    print(f"Reading {input_path}…")
    lines = input_path.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip()]
    print(f"  {len(lines):,} paragraphs loaded.")

    rng = random.Random(seed)
    rng.shuffle(lines)

    split_idx = int(len(lines) * ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    print(f"  Train: {len(train_lines):,} paragraphs → {train_path}")
    print(f"  Val:   {len(val_lines):,} paragraphs  → {val_path}")
    return len(train_lines), len(val_lines)


def main() -> None:
    input_path = CLEAN_DIR / "all_raw.txt"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Combined corpus not found at {input_path}. Run clean.py first."
        )

    split_corpus(
        input_path,
        train_path=CLEAN_DIR / "train.txt",
        val_path=CLEAN_DIR / "val.txt",
    )
    print("Split complete.")


if __name__ == "__main__":
    main()
