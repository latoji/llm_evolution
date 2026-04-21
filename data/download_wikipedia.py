"""
Track 0 – Data Pipeline
Download English Wikipedia dump and extract plain text using wikiextractor.
Keeps first 2 GB of extracted output in data/raw/wikipedia/wiki_extracted.txt.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import requests
from tqdm import tqdm

# Use the first articles partition — large enough for 2 GB of text.
WIKI_DUMP_URL = (
    "https://dumps.wikimedia.org/enwiki/latest/"
    "enwiki-latest-pages-articles1.xml.bz2"
)
RAW_DIR = Path(__file__).parent / "raw" / "wikipedia"
DUMP_FILE = RAW_DIR / "enwiki-latest.xml.bz2"
EXTRACT_DIR = RAW_DIR / "extracted"
OUTPUT_FILE = RAW_DIR / "wiki_extracted.txt"
SIZE_LIMIT = 2 * 1024 ** 3  # 2 GB

_XML_TAG_RE = re.compile(r"<[^>]+>")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dump() -> None:
    """Stream-download the Wikipedia XML dump with a progress bar."""
    if DUMP_FILE.exists():
        print(f"Dump already exists at {DUMP_FILE}. Skipping download.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Wikipedia dump from:\n  {WIKI_DUMP_URL}")
    resp = requests.get(WIKI_DUMP_URL, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0))

    with open(DUMP_FILE, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc="Wikipedia dump"
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            fh.write(chunk)
            bar.update(len(chunk))

    print(f"Saved dump → {DUMP_FILE}")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_text() -> None:
    """Run wikiextractor then concatenate articles up to SIZE_LIMIT bytes."""
    if OUTPUT_FILE.exists():
        print(f"Extracted file already exists at {OUTPUT_FILE}. Skipping.")
        return

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print("Running wikiextractor (this may take a while)…")
    subprocess.run(
        [
            "python", "-m", "wikiextractor",
            str(DUMP_FILE),
            "--output", str(EXTRACT_DIR),
            "--bytes", "100M",
            "--no-templates",
            "--quiet",
        ],
        check=True,
    )

    # Concatenate extracted wiki_* files up to the size limit
    print(f"Concatenating extracted files (limit {SIZE_LIMIT // 1024**3} GB)…")
    written = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for wiki_file in sorted(EXTRACT_DIR.rglob("wiki_*")):
            if written >= SIZE_LIMIT:
                break
            try:
                raw = wiki_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # Strip XML/HTML tags left by wikiextractor
            clean = _XML_TAG_RE.sub("", raw)
            out.write(clean)
            written += len(clean.encode("utf-8"))

    print(f"Wrote {written / 1024**3:.2f} GB → {OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    download_dump()
    extract_text()
    print("Wikipedia pipeline complete.")


if __name__ == "__main__":
    main()
