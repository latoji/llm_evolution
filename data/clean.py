"""
Track 0 – Data Pipeline
Clean raw text corpus:
  - Normalize unicode (NFKC)
  - Collapse whitespace
  - Strip non-printable characters
  - Remove paragraphs shorter than MIN_LEN characters (noise)
  - Output: one paragraph per line, UTF-8
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import IO

from tqdm import tqdm

RAW_DIR = Path(__file__).parent / "raw"
CLEAN_DIR = Path(__file__).parent / "clean"
MIN_LEN = 20  # drop paragraphs shorter than this

_NON_PRINTABLE = re.compile(r"[^\x09\x0a\x0d\x20-\x7e\x80-\ufffd]")
_WHITESPACE = re.compile(r"[ \t\r]+")
_PARA_SPLIT = re.compile(r"\n{2,}")


# ---------------------------------------------------------------------------
# Line-level helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """NFKC normalize, strip non-printable chars."""
    text = unicodedata.normalize("NFKC", text)
    return _NON_PRINTABLE.sub("", text)


def clean_paragraph(para: str) -> str | None:
    """
    Flatten, normalize, and validate a paragraph.
    Returns None if the paragraph should be dropped.
    """
    # Flatten multi-line paragraph to single line
    flat = " ".join(para.splitlines())
    flat = _WHITESPACE.sub(" ", normalize(flat)).strip()
    return flat if len(flat) >= MIN_LEN else None


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------

def clean_file(src: Path, out: IO[str]) -> int:
    """
    Clean src and write valid paragraphs to out.
    Returns the number of paragraphs written.
    """
    try:
        text = src.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"  [skip] {src}: {exc}")
        return 0

    count = 0
    for para in _PARA_SPLIT.split(text):
        cleaned = clean_paragraph(para)
        if cleaned:
            out.write(cleaned + "\n")
            count += 1
    return count


def clean_directory(source_dir: Path, output_path: Path) -> int:
    """Clean every .txt in source_dir and write to output_path."""
    files = sorted(source_dir.rglob("*.txt"))
    if not files:
        print(f"  No .txt files found in {source_dir}")
        return 0

    total = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for path in tqdm(files, desc=f"Cleaning {source_dir.name}"):
            total += clean_file(path, out)

    print(f"  {total:,} paragraphs → {output_path}")
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    combined = CLEAN_DIR / "all_raw.txt"

    sources: list[tuple[Path, str]] = []

    gutenberg_dir = RAW_DIR / "gutenberg"
    if gutenberg_dir.exists():
        sources.append((gutenberg_dir, "gutenberg_clean.txt"))

    wiki_file = RAW_DIR / "wikipedia" / "wiki_extracted.txt"
    if wiki_file.exists():
        wiki_clean = CLEAN_DIR / "wikipedia_clean.txt"
        print("Cleaning Wikipedia text…")
        with open(wiki_clean, "w", encoding="utf-8") as out:
            n = clean_file(wiki_file, out)
        print(f"  {n:,} paragraphs → {wiki_clean}")

    for src_dir, out_name in sources:
        clean_directory(src_dir, CLEAN_DIR / out_name)

    # Merge all cleaned sources into one file
    print("Merging cleaned sources…")
    with open(combined, "w", encoding="utf-8") as out:
        for p in sorted(CLEAN_DIR.glob("*_clean.txt")):
            out.write(p.read_text(encoding="utf-8"))

    lines = sum(1 for _ in open(combined, encoding="utf-8"))
    print(f"Combined corpus: {lines:,} paragraphs → {combined}")


if __name__ == "__main__":
    main()
