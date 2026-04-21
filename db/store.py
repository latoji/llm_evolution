"""Typed data-access helpers — all database I/O in the project goes through this module."""

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

from db.schema import DB_PATH, create_all, reset_all

# Column name used for the "next item" differs per n-gram family.
_FAMILY_META: dict[str, tuple[str, str]] = {
    "char":  ("char_ngrams",  "next_char"),
    "word":  ("word_ngrams",  "next_word"),
    "bpe":   ("token_ngrams", "next_token"),
}

_CHECKPOINTS_DIR = Path("model/checkpoints")


class Store:
    """Single-connection typed interface to the DuckDB database.

    Open one Store per process. Do not share connections across processes.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        """Open connection and create schema if absent."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: duckdb.DuckDBPyConnection = duckdb.connect(str(db_path))
        create_all(self._conn)

    # ------------------------------------------------------------------
    # Transaction context manager
    # ------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """BEGIN … COMMIT on success, ROLLBACK on any exception (including signals)."""
        self._conn.execute("BEGIN")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Chunk tracking
    # ------------------------------------------------------------------

    def insert_chunk(
        self,
        filename: str,
        chunk_index: int,
        text_hash: str,
        char_count: int,
        accuracy_before: dict,
    ) -> int:
        """Insert a new corpus_chunk row and return its auto-assigned id."""
        row = self._conn.execute(
            """
            INSERT INTO corpus_chunks
                (filename, chunk_index, text_hash, char_count, status, accuracy_before)
            VALUES (?, ?, ?, ?, 'processing', ?)
            RETURNING id
            """,
            [filename, chunk_index, text_hash, char_count, json.dumps(accuracy_before)],
        ).fetchone()
        if row is None:
            raise RuntimeError("INSERT INTO corpus_chunks returned no id")
        return int(row[0])

    def mark_chunk_accepted(self, chunk_id: int, accuracy_after: dict) -> None:
        """Set chunk status to 'accepted' and store the post-ingest accuracy snapshot."""
        self._conn.execute(
            """
            UPDATE corpus_chunks
               SET status = 'accepted', accuracy_after = ?
             WHERE id = ?
            """,
            [json.dumps(accuracy_after), chunk_id],
        )

    def mark_chunk_rejected(self, chunk_id: int, reason: str) -> None:
        """Set chunk status to 'rejected' and record the rejection reason in accuracy_after."""
        self._conn.execute(
            """
            UPDATE corpus_chunks
               SET status = 'rejected', accuracy_after = ?
             WHERE id = ?
            """,
            [json.dumps({"reason": reason}), chunk_id],
        )

    # ------------------------------------------------------------------
    # N-gram writes
    # ------------------------------------------------------------------

    def upsert_ngrams(
        self,
        family: str,
        n: int,
        rows: list[tuple[str, str, int]],
    ) -> None:
        """Bulk-upsert (context, next_item, count) rows into the appropriate ngram table.

        family: 'char' | 'word' | 'bpe'
        rows:   list of (context, next_item, delta_count)
        Existing (n, context, next_item) rows have their count incremented by delta.
        """
        if family not in _FAMILY_META:
            raise ValueError(f"Unknown ngram family '{family}'. Expected one of: {list(_FAMILY_META)}")
        table, next_col = _FAMILY_META[family]
        if not rows:
            return

        # Build a VALUES clause via a temporary in-memory relation for bulk insertion.
        # DuckDB's executemany is the most efficient path for large batches.
        params = [(n, context, next_item, count) for context, next_item, count in rows]
        self._conn.executemany(
            f"""
            INSERT INTO {table} (n, context, {next_col}, count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (n, context, {next_col})
            DO UPDATE SET count = {table}.count + excluded.count
            """,
            params,
        )

    # ------------------------------------------------------------------
    # N-gram reads
    # ------------------------------------------------------------------

    def get_ngrams(
        self,
        family: str,
        n: int,
        context: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Paginated fetch for the DB Viewer page."""
        if family not in _FAMILY_META:
            raise ValueError(f"Unknown ngram family '{family}'")
        table, next_col = _FAMILY_META[family]

        where = "WHERE ng.n = ?"
        params: list = [n]
        if context is not None:
            where += " AND ng.context = ?"
            params.append(context)

        params += [limit, offset]
        rows = self._conn.execute(
            f"""
            SELECT ng.context, ng.{next_col} AS next_item, ng.count,
                   ng.count * 1.0 / SUM(ng.count) OVER (PARTITION BY ng.n, ng.context)
                       AS probability
              FROM {table} ng
             {where}
             ORDER BY ng.count DESC
             LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [
            {"context": r[0], "next_item": r[1], "count": r[2], "probability": r[3]}
            for r in rows
        ]

    def get_distribution(self, family: str, n: int, context: str) -> dict[str, int]:
        """Return {next_item: count} for a given context; used by Markov generators."""
        if family not in _FAMILY_META:
            raise ValueError(f"Unknown ngram family '{family}'")
        table, next_col = _FAMILY_META[family]
        rows = self._conn.execute(
            f"SELECT {next_col}, count FROM {table} WHERE n = ? AND context = ?",
            [n, context],
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------
    # Accuracy tracking
    # ------------------------------------------------------------------

    def insert_accuracy(
        self,
        model_name: str,
        chunk_id: int,
        accuracy: float,
        perplexity: float | None,
    ) -> None:
        """Insert one row into model_accuracy."""
        self._conn.execute(
            """
            INSERT INTO model_accuracy (model_name, chunk_id, accuracy, perplexity)
            VALUES (?, ?, ?, ?)
            """,
            [model_name, chunk_id, accuracy, perplexity],
        )

    def get_accuracy_history(self, model_name: str | None = None) -> list[dict]:
        """Return all model_accuracy rows, optionally filtered by model_name."""
        if model_name is not None:
            rows = self._conn.execute(
                """
                SELECT id, model_name, chunk_id, accuracy, perplexity,
                       CAST(timestamp AS TEXT) AS timestamp
                  FROM model_accuracy
                 WHERE model_name = ?
                 ORDER BY chunk_id
                """,
                [model_name],
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT id, model_name, chunk_id, accuracy, perplexity,
                       CAST(timestamp AS TEXT) AS timestamp
                  FROM model_accuracy
                 ORDER BY model_name, chunk_id
                """
            ).fetchall()
        return [
            {
                "id": r[0],
                "model_name": r[1],
                "chunk_id": r[2],
                "accuracy": r[3],
                "perplexity": r[4],
                "timestamp": r[5],
            }
            for r in rows
        ]

    def get_latest_accuracy(self) -> dict[str, float]:
        """{model_name: accuracy} for the most recent chunk per model."""
        rows = self._conn.execute(
            """
            SELECT model_name, accuracy
              FROM model_accuracy
             WHERE chunk_id = (
                   SELECT MAX(chunk_id) FROM model_accuracy ma2
                    WHERE ma2.model_name = model_accuracy.model_name
             )
            """
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------
    # Neural checkpoints
    # ------------------------------------------------------------------

    def insert_checkpoint(
        self,
        model_name: str,
        chunk_id: int,
        filepath: Path,
        val_loss: float,
    ) -> None:
        """Record a new .pt checkpoint file in nn_checkpoints."""
        self._conn.execute(
            """
            INSERT INTO nn_checkpoints (model_name, chunk_id, filepath, val_loss)
            VALUES (?, ?, ?, ?)
            """,
            [model_name, chunk_id, str(filepath), val_loss],
        )

    def get_latest_checkpoint(self, model_name: str) -> dict | None:
        """Return the most recent checkpoint row for model_name, or None."""
        row = self._conn.execute(
            """
            SELECT id, model_name, chunk_id, filepath, val_loss,
                   CAST(created_at AS TEXT) AS created_at
              FROM nn_checkpoints
             WHERE model_name = ?
             ORDER BY chunk_id DESC
             LIMIT 1
            """,
            [model_name],
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "model_name": row[1],
            "chunk_id": row[2],
            "filepath": row[3],
            "val_loss": row[4],
            "created_at": row[5],
        }

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def upsert_vocabulary(self, entries: list[tuple[int, str, str]]) -> None:
        """Bulk upsert (token_id, token, source) rows into vocabulary."""
        self._conn.executemany(
            """
            INSERT INTO vocabulary (token_id, token, source)
            VALUES (?, ?, ?)
            ON CONFLICT (token_id) DO UPDATE SET token = excluded.token, source = excluded.source
            """,
            entries,
        )

    def get_vocabulary(
        self,
        source: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Return paginated vocabulary rows filtered by source."""
        rows = self._conn.execute(
            """
            SELECT token_id, token, source
              FROM vocabulary
             WHERE source = ?
             ORDER BY token_id
             LIMIT ? OFFSET ?
            """,
            [source, limit, offset],
        ).fetchall()
        return [{"token_id": r[0], "token": r[1], "source": r[2]} for r in rows]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_all(self) -> None:
        """Drop all tables, delete checkpoint files and DB file, then recreate schema."""
        # Delete .pt checkpoint files
        if _CHECKPOINTS_DIR.exists():
            for pt_file in _CHECKPOINTS_DIR.glob("*.pt"):
                pt_file.unlink(missing_ok=True)

        reset_all(self._conn)

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._conn.close()
