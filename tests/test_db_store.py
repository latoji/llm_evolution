"""Tests for db/schema.py and db/store.py — DuckDB persistence layer."""

import multiprocessing
import pytest
from pathlib import Path

from db.schema import create_all, reset_all, TABLES
from db.store import Store

import duckdb


# ---------------------------------------------------------------------------
# Fixtures — each test gets a fresh in-memory or temp-file database
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path: Path) -> Store:  # type: ignore[return]
    """Fresh Store backed by a temp file; closed after the test."""
    db_path = tmp_path / "test.duckdb"
    s = Store(db_path=db_path)
    yield s  # type: ignore[misc]
    s.close()


@pytest.fixture()
def conn_only(tmp_path: Path) -> duckdb.DuckDBPyConnection:  # type: ignore[return]
    """Bare DuckDB connection (not wrapped in Store) for schema-level tests."""
    path = tmp_path / "schema_test.duckdb"
    conn = duckdb.connect(str(path))
    yield conn  # type: ignore[misc]
    conn.close()


# ---------------------------------------------------------------------------
# schema — create_all is idempotent
# ---------------------------------------------------------------------------

def test_create_all_idempotent(conn_only: duckdb.DuckDBPyConnection) -> None:
    create_all(conn_only)
    create_all(conn_only)   # second call must not raise
    tables = {
        row[0]
        for row in conn_only.execute("SHOW TABLES").fetchall()
    }
    for name in TABLES:
        assert name in tables, f"Expected table '{name}' to exist after create_all"


# ---------------------------------------------------------------------------
# insert_chunk — returns auto-increment id
# ---------------------------------------------------------------------------

def test_insert_chunk_returns_id(store: Store) -> None:
    chunk_id = store.insert_chunk(
        filename="test.txt",
        chunk_index=0,
        text_hash="abc123",
        char_count=1000,
        accuracy_before={},
    )
    assert isinstance(chunk_id, int)
    assert chunk_id >= 1


def test_insert_chunk_ids_increment(store: Store) -> None:
    id1 = store.insert_chunk("a.txt", 0, "h1", 100, {})
    id2 = store.insert_chunk("a.txt", 1, "h2", 100, {})
    assert id2 > id1


# ---------------------------------------------------------------------------
# mark_chunk_accepted / mark_chunk_rejected
# ---------------------------------------------------------------------------

def test_mark_chunk_accepted(store: Store) -> None:
    chunk_id = store.insert_chunk("f.txt", 0, "h", 500, {"m1": 0.4})
    store.mark_chunk_accepted(chunk_id, {"m1": 0.5})
    row = store._conn.execute(
        "SELECT status FROM corpus_chunks WHERE id = ?", [chunk_id]
    ).fetchone()
    assert row is not None
    assert row[0] == "accepted"


def test_mark_chunk_rejected(store: Store) -> None:
    chunk_id = store.insert_chunk("f.txt", 1, "h2", 500, {})
    store.mark_chunk_rejected(chunk_id, "accuracy dropped")
    row = store._conn.execute(
        "SELECT status FROM corpus_chunks WHERE id = ?", [chunk_id]
    ).fetchone()
    assert row is not None
    assert row[0] == "rejected"


# ---------------------------------------------------------------------------
# upsert_ngrams — duplicate rows add counts
# ---------------------------------------------------------------------------

def test_upsert_ngrams_accumulates_counts(store: Store) -> None:
    store.upsert_ngrams("char", 2, [("th", "e", 5)])
    store.upsert_ngrams("char", 2, [("th", "e", 3)])
    dist = store.get_distribution("char", 2, "th")
    assert dist["e"] == 8


def test_upsert_ngrams_multiple_rows(store: Store) -> None:
    rows = [("ab", "c", 10), ("ab", "d", 20), ("xy", "z", 5)]
    store.upsert_ngrams("word", 2, rows)
    dist = store.get_distribution("word", 2, "ab")
    assert dist["c"] == 10
    assert dist["d"] == 20


def test_upsert_ngrams_unknown_family(store: Store) -> None:
    with pytest.raises(ValueError, match="Unknown ngram family"):
        store.upsert_ngrams("bad_family", 2, [("a", "b", 1)])


def test_upsert_ngrams_empty_rows_no_error(store: Store) -> None:
    store.upsert_ngrams("bpe", 1, [])   # must not raise


# ---------------------------------------------------------------------------
# get_distribution
# ---------------------------------------------------------------------------

def test_get_distribution_char(store: Store) -> None:
    store.upsert_ngrams("char", 2, [("th", "e", 10), ("th", "a", 3)])
    dist = store.get_distribution("char", 2, "th")
    assert dist == {"e": 10, "a": 3}


def test_get_distribution_empty_context(store: Store) -> None:
    dist = store.get_distribution("char", 2, "zz")
    assert dist == {}


# ---------------------------------------------------------------------------
# transaction() — rollback on exception
# ---------------------------------------------------------------------------

def test_transaction_rollback_on_exception(store: Store) -> None:
    """Insert inside transaction, raise → row must be absent afterwards."""
    with pytest.raises(RuntimeError):
        with store.transaction():
            store._conn.execute(
                "INSERT INTO vocabulary (token_id, token, source) VALUES (999, 'test', 'char')"
            )
            raise RuntimeError("deliberate failure")

    row = store._conn.execute(
        "SELECT * FROM vocabulary WHERE token_id = 999"
    ).fetchone()
    assert row is None, "Row must not exist after rolled-back transaction"


def test_transaction_commits_on_success(store: Store) -> None:
    with store.transaction():
        store._conn.execute(
            "INSERT INTO vocabulary (token_id, token, source) VALUES (1, 'hello', 'word')"
        )
    row = store._conn.execute(
        "SELECT token FROM vocabulary WHERE token_id = 1"
    ).fetchone()
    assert row is not None
    assert row[0] == "hello"


# ---------------------------------------------------------------------------
# insert_accuracy / get_accuracy_history / get_latest_accuracy
# ---------------------------------------------------------------------------

def test_insert_and_get_accuracy(store: Store) -> None:
    chunk_id = store.insert_chunk("f.txt", 0, "h", 100, {})
    store.insert_accuracy("char_2gram", chunk_id, 0.55, None)
    history = store.get_accuracy_history("char_2gram")
    assert len(history) == 1
    assert history[0]["accuracy"] == pytest.approx(0.55)
    assert history[0]["perplexity"] is None


def test_get_latest_accuracy(store: Store) -> None:
    c1 = store.insert_chunk("f.txt", 0, "h1", 100, {})
    c2 = store.insert_chunk("f.txt", 1, "h2", 100, {})
    store.insert_accuracy("char_2gram", c1, 0.4, None)
    store.insert_accuracy("char_2gram", c2, 0.6, None)
    latest = store.get_latest_accuracy()
    assert latest["char_2gram"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# insert_checkpoint / get_latest_checkpoint
# ---------------------------------------------------------------------------

def test_checkpoint_round_trip(store: Store, tmp_path: Path) -> None:
    pt = tmp_path / "model.pt"
    chunk_id = store.insert_chunk("f.txt", 0, "h", 100, {})
    store.insert_checkpoint("feedforward", chunk_id, pt, val_loss=1.23)
    ckpt = store.get_latest_checkpoint("feedforward")
    assert ckpt is not None
    assert ckpt["model_name"] == "feedforward"
    assert ckpt["val_loss"] == pytest.approx(1.23)


def test_get_latest_checkpoint_none(store: Store) -> None:
    result = store.get_latest_checkpoint("nonexistent_model")
    assert result is None


# ---------------------------------------------------------------------------
# upsert_vocabulary / get_vocabulary
# ---------------------------------------------------------------------------

def test_upsert_vocabulary_and_read(store: Store) -> None:
    entries = [(0, "hello", "word"), (1, "world", "word")]
    store.upsert_vocabulary(entries)
    rows = store.get_vocabulary("word", limit=10, offset=0)
    tokens = {r["token"] for r in rows}
    assert "hello" in tokens
    assert "world" in tokens


# ---------------------------------------------------------------------------
# reset_all
# ---------------------------------------------------------------------------

def test_reset_all_clears_and_recreates(store: Store) -> None:
    chunk_id = store.insert_chunk("f.txt", 0, "h", 100, {})
    store.upsert_ngrams("char", 2, [("ab", "c", 5)])
    store.reset_all()
    # All tables should still exist but be empty
    rows = store._conn.execute("SELECT * FROM corpus_chunks").fetchall()
    assert rows == []
    rows = store._conn.execute("SELECT * FROM char_ngrams").fetchall()
    assert rows == []


# ---------------------------------------------------------------------------
# Concurrent access — second process writes without corruption
# ---------------------------------------------------------------------------

def _writer_process(db_path: str) -> None:
    """Target function for subprocess write test."""
    s = Store(db_path=Path(db_path))
    s.insert_chunk("subprocess.txt", 0, "hx", 50, {})
    s.close()


def test_concurrent_subprocess_write(tmp_path: Path) -> None:
    db_path = tmp_path / "concurrent.duckdb"
    # Create schema in main process
    s = Store(db_path=db_path)
    s.close()

    proc = multiprocessing.Process(target=_writer_process, args=(str(db_path),))
    proc.start()
    proc.join(timeout=15)
    assert proc.exitcode == 0, "Subprocess write failed"
