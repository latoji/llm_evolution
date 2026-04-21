"""DuckDB table DDL and schema management — single source of truth for all table definitions."""

from pathlib import Path

import duckdb

DB_PATH = Path("db/llm_evolution.duckdb")

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Sequences — provide auto-increment IDs for tables with INTEGER PRIMARY KEY
# ---------------------------------------------------------------------------

SEQUENCES: list[str] = [
    "CREATE SEQUENCE IF NOT EXISTS seq_corpus_chunks START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_model_accuracy START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_nn_checkpoints START 1",
]

# ---------------------------------------------------------------------------
# Table DDL — matches SPEC.md section 7 exactly
# ---------------------------------------------------------------------------

TABLES: dict[str, str] = {
    "corpus_chunks": """
        CREATE TABLE IF NOT EXISTS corpus_chunks (
            id              INTEGER PRIMARY KEY DEFAULT nextval('seq_corpus_chunks'),
            filename        TEXT,
            chunk_index     INTEGER,
            text_hash       TEXT,
            char_count      INTEGER,
            status          TEXT,
            accuracy_before JSON,
            accuracy_after  JSON,
            ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "char_ngrams": """
        CREATE TABLE IF NOT EXISTS char_ngrams (
            n           INT,
            context     TEXT,
            next_char   TEXT,
            count       BIGINT,
            PRIMARY KEY (n, context, next_char)
        )
    """,
    "word_ngrams": """
        CREATE TABLE IF NOT EXISTS word_ngrams (
            n           INT,
            context     TEXT,
            next_word   TEXT,
            count       BIGINT,
            PRIMARY KEY (n, context, next_word)
        )
    """,
    "token_ngrams": """
        CREATE TABLE IF NOT EXISTS token_ngrams (
            n           INT,
            context     TEXT,
            next_token  TEXT,
            count       BIGINT,
            PRIMARY KEY (n, context, next_token)
        )
    """,
    "vocabulary": """
        CREATE TABLE IF NOT EXISTS vocabulary (
            token_id    INTEGER PRIMARY KEY,
            token       TEXT,
            source      TEXT,
            frequency   INTEGER DEFAULT 0
        )
    """,
    "model_accuracy": """
        CREATE TABLE IF NOT EXISTS model_accuracy (
            id          INTEGER PRIMARY KEY DEFAULT nextval('seq_model_accuracy'),
            model_name  TEXT,
            chunk_id    INTEGER REFERENCES corpus_chunks(id),
            accuracy    FLOAT,
            perplexity  FLOAT,
            timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "nn_checkpoints": """
        CREATE TABLE IF NOT EXISTS nn_checkpoints (
            id          INTEGER PRIMARY KEY DEFAULT nextval('seq_nn_checkpoints'),
            model_name  TEXT,
            chunk_id    INTEGER,
            filepath    TEXT,
            val_loss    FLOAT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "schema_meta": """
        CREATE TABLE IF NOT EXISTS schema_meta (
            version INTEGER PRIMARY KEY
        )
    """,
}

INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_char_ngrams_n_context ON char_ngrams(n, context)",
    "CREATE INDEX IF NOT EXISTS idx_word_ngrams_n_context ON word_ngrams(n, context)",
    "CREATE INDEX IF NOT EXISTS idx_token_ngrams_n_context ON token_ngrams(n, context)",
    "CREATE INDEX IF NOT EXISTS idx_model_accuracy_model ON model_accuracy(model_name, chunk_id)",
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_all(conn: duckdb.DuckDBPyConnection) -> None:
    """Apply all sequences, tables, and indexes. Idempotent — safe to call on an existing database."""
    for seq_ddl in SEQUENCES:
        conn.execute(seq_ddl)
    for ddl in TABLES.values():
        conn.execute(ddl)
    for idx in INDEXES:
        conn.execute(idx)
    # Record schema version (ignore if already present)
    conn.execute(
        "INSERT INTO schema_meta (version) VALUES (?) ON CONFLICT DO NOTHING",
        [SCHEMA_VERSION],
    )


def reset_all(conn: duckdb.DuckDBPyConnection) -> None:
    """Drop and recreate all tables. Called by POST /db/reset."""
    for table_name in reversed(list(TABLES)):
        conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
    for seq in ["seq_corpus_chunks", "seq_model_accuracy", "seq_nn_checkpoints"]:
        conn.execute(f"DROP SEQUENCE IF EXISTS {seq}")
    create_all(conn)
