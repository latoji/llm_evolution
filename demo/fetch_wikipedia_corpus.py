"""
Fetch Wikipedia articles and append them to demo/demo_corpus.txt.

This script updates the corpus used by the Markov chain visualizer WITHOUT
retraining the full language model (demo_lm.pkl).

  - The character n-gram tables for the Markov visualizer are built fresh
    from demo_corpus.txt every time the Flask app starts, so appending text
    here takes effect on the next server restart automatically.
  - The word-level BPE model (demo_lm.pkl) is NOT updated by this script.
    Run demo/setup_demo.py (delete demo_lm.pkl first) for a full retrain.

Usage:
    python3 demo/fetch_wikipedia_corpus.py

Options (environment variables):
    WIKI_APPEND_ONLY=1   Skip download if Wikipedia marker already present.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DEMO_DIR = Path(__file__).parent
CORPUS_PATH = DEMO_DIR / "demo_corpus.txt"
WIKI_MARKER = "\n\n# ── Wikipedia articles ─────────────────────────────────────────\n\n"


def _already_has_wikipedia(corpus_path: Path) -> bool:
    """Return True if the corpus already contains Wikipedia content."""
    if not corpus_path.exists():
        return False
    # Read just the end of the file for efficiency
    with open(corpus_path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        # Check first 4KB for the marker
        fh.seek(0)
        head = fh.read(min(4096, size)).decode("utf-8", errors="replace")
    return "# ── Wikipedia articles" in head


def main() -> None:
    if not CORPUS_PATH.exists():
        print(f"ERROR: corpus not found at {CORPUS_PATH}")
        print("Run  python3 demo/setup_demo.py  first.")
        sys.exit(1)

    if os.environ.get("WIKI_APPEND_ONLY") and _already_has_wikipedia(CORPUS_PATH):
        print("Wikipedia content already present in corpus. Nothing to do.")
        return

    before_size = CORPUS_PATH.stat().st_size
    print("=" * 60)
    print("  Downloading Wikipedia articles for demo corpus")
    print("=" * 60)
    print(f"  Current corpus: {before_size / 1_000_000:.1f} MB")
    print()

    from demo.corpus_wikipedia import WIKIPEDIA_ARTICLES, download_all

    wiki_text = download_all()

    if not wiki_text.strip():
        print("\nNo Wikipedia text downloaded. Corpus unchanged.")
        sys.exit(1)

    # Append Wikipedia text to corpus with a clear marker
    with open(CORPUS_PATH, "a", encoding="utf-8") as fh:
        fh.write(WIKI_MARKER)
        fh.write(wiki_text)
        fh.write("\n")

    after_size = CORPUS_PATH.stat().st_size
    added = after_size - before_size

    print()
    print("=" * 60)
    print(f"  ✓ Appended {added / 1_000_000:.1f} MB of Wikipedia text")
    print(f"  ✓ New corpus size: {after_size / 1_000_000:.1f} MB")
    print(f"  ✓ Corpus path: {CORPUS_PATH}")
    print()
    print("  The Markov visualizer will pick up the new data on next")
    print("  server start (restart demo/app.py).")
    print()
    print("  To retrain the word-level model with Wikipedia data:")
    print("    rm demo/demo_lm.pkl && python3 demo/setup_demo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
