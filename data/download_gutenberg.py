"""
Track 0 – Data Pipeline
Fetch top 1000 English books from Project Gutenberg mirrors via requests.
Strips headers, licensing boilerplate, and *** START/END *** markers.
Saves one .txt per book to data/raw/gutenberg/.
"""
from __future__ import annotations

import csv
import io
import re
import time
from pathlib import Path
from typing import Iterator

import requests
from tqdm import tqdm

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
TEXT_URL = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
OUTPUT_DIR = Path(__file__).parent / "raw" / "gutenberg"
TOP_N = 1000
DELAY = 1.0  # seconds between requests (be polite)
REQUEST_TIMEOUT = 30

_START_RE = re.compile(
    r"\*{3}\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.+?\*{3}",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"\*{3}\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.+?\*{3}",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def fetch_catalog(top_n: int = TOP_N) -> list[dict[str, str]]:
    """Download the Gutenberg CSV catalog and return top_n English text entries."""
    print("Fetching Gutenberg catalog…")
    resp = requests.get(CATALOG_URL, timeout=60)
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    books = [
        row for row in reader
        if row.get("Language", "").strip().lower() == "en"
        and row.get("Type", "").strip().lower() == "text"
    ]
    print(f"  {len(books)} English texts found in catalog. Using top {top_n}.")
    return books[:top_n]


# ---------------------------------------------------------------------------
# Boilerplate stripping
# ---------------------------------------------------------------------------

def strip_boilerplate(text: str) -> str:
    """
    Extract the actual book content by removing Project Gutenberg
    headers, footers, and licensing blocks.
    """
    start_m = _START_RE.search(text)
    if start_m:
        text = text[start_m.end():]

    end_m = _END_RE.search(text)
    if end_m:
        text = text[: end_m.start()]

    return text.strip()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_book(book_id: str) -> str | None:
    """Download a single book. Returns cleaned text or None on failure."""
    url = TEXT_URL.format(book_id=book_id)
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        text = resp.text
        return strip_boilerplate(text)
    except requests.RequestException as exc:
        print(f"  [skip] book {book_id}: {exc}")
        return None


def book_ids(books: list[dict[str, str]]) -> Iterator[str]:
    for book in books:
        yield book.get("Text#", "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    books = fetch_catalog(TOP_N)

    success = 0
    for book in tqdm(books, desc="Gutenberg books"):
        book_id = book.get("Text#", "").strip()
        if not book_id:
            continue

        out_path = OUTPUT_DIR / f"{book_id}.txt"
        if out_path.exists():
            success += 1
            continue

        text = download_book(book_id)
        if text and len(text) > 500:
            out_path.write_text(text, encoding="utf-8")
            success += 1

        time.sleep(DELAY)

    print(f"\nDownloaded {success}/{len(books)} books → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
