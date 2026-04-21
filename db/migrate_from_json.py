"""One-time migration from legacy JSON n-gram count files to DuckDB.

If model/counts/ contains JSON files produced by the original BPE pipeline,
this script reads them and inserts the rows into the token_ngrams DuckDB
table, then moves the source files to model/counts/archive/.

The migration is idempotent: if the archive/ directory already exists (meaning
migration has already run), the function exits immediately.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from db.store import Store

# JSON filenames (order → name) as written by model/ngram_counter.save_counts
_ORDER_NAMES: dict[str, int] = {
    "unigrams.json": 1,
    "bigrams.json": 2,
    "trigrams.json": 3,
    "fourgrams.json": 4,
}

COUNTS_DIR = Path("model/counts")
ARCHIVE_DIR = COUNTS_DIR / "archive"


def migrate(store: Store | None = None, counts_dir: Path = COUNTS_DIR) -> int:
    """Migrate JSON count files to DuckDB token_ngrams table.

    Checks whether *counts_dir/archive/* already exists (sentinel for a
    completed migration). If not, reads every recognised JSON file, upserts
    rows into DuckDB, and moves the files to *archive/*.

    Args:
        store:      Open Store instance. If None, opens the default DB.
        counts_dir: Directory containing the JSON count files.

    Returns:
        Number of n-gram rows migrated (0 if already migrated or no files
        found).
    """
    archive = counts_dir / "archive"

    # Idempotency guard: if archive already exists, migration is done.
    if archive.exists():
        return 0

    own_store = store is None
    if own_store:
        store = Store()

    try:
        total_rows = 0

        for filename, order in _ORDER_NAMES.items():
            json_path = counts_dir / filename
            if not json_path.exists():
                continue

            raw: dict[str, int] = json.loads(
                json_path.read_text(encoding="utf-8")
            )
            if not raw:
                continue

            # Convert JSON keys ("1 2 3") → (context, next_token, count) rows
            rows: list[tuple[str, str, int]] = []
            for key, count in raw.items():
                parts = key.split()
                # Context is all token IDs except the last; last is next_token
                context = " ".join(parts[:-1])  # "" for unigrams
                next_token = parts[-1]
                rows.append((context, next_token, count))

            with store.transaction():
                store.upsert_ngrams("bpe", order, rows)

            total_rows += len(rows)

        if total_rows > 0 or any(
            (counts_dir / fn).exists() for fn in _ORDER_NAMES
        ):
            # Create archive and move processed files
            archive.mkdir(parents=True, exist_ok=True)
            for filename in _ORDER_NAMES:
                src = counts_dir / filename
                if src.exists():
                    shutil.move(str(src), str(archive / filename))

        return total_rows

    finally:
        if own_store:
            store.close()


if __name__ == "__main__":
    migrated = migrate()
    if migrated == 0:
        print("Migration already complete or no JSON files found.")
    else:
        print(f"Migrated {migrated:,} n-gram rows to DuckDB.")
