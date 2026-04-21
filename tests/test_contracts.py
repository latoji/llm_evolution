"""Tests for api/contracts.py — Pydantic v2 model serialisation and validation."""

import json
import pytest
from pydantic import ValidationError, TypeAdapter

from api.contracts import (
    # Ingest
    IngestUploadResponse,
    IngestStatusResponse,
    IngestPauseResponse,
    # Generate
    GenerateRequest,
    ModelOutput,
    GenerateResponse,
    # Stats
    AccuracyPoint,
    AccuracyHistoryResponse,
    LastOutputEntry,
    LastOutputResponse,
    # DB
    NGramRow,
    NGramPageResponse,
    VocabularyRow,
    VocabularyPageResponse,
    DBResetResponse,
    # WebSocket
    WSChunkStart,
    WSChunkProgress,
    WSMCToken,
    WSMCComplete,
    WSChunkDone,
    WSIngestComplete,
    WSMessageAnnotated,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def round_trip(model_instance):
    """Serialise to JSON and back; returns the reconstructed model."""
    json_str = model_instance.model_dump_json()
    return type(model_instance).model_validate_json(json_str)


# ---------------------------------------------------------------------------
# HTTP models — round-trip
# ---------------------------------------------------------------------------

def test_ingest_upload_response_round_trip() -> None:
    obj = IngestUploadResponse(
        accepted_files=["a.txt"],
        rejected_files=[{"filename": "b.txt", "reason": "bad encoding"}],
        total_chunks=5,
    )
    assert round_trip(obj) == obj


def test_ingest_status_response_round_trip() -> None:
    obj = IngestStatusResponse(
        state="running",
        current_chunk=3,
        total_chunks=10,
        chunks_accepted=2,
        chunks_rejected=1,
    )
    assert round_trip(obj) == obj


def test_ingest_status_response_null_fields() -> None:
    obj = IngestStatusResponse(
        state="idle",
        current_chunk=None,
        total_chunks=None,
        chunks_accepted=0,
        chunks_rejected=0,
    )
    restored = round_trip(obj)
    assert restored.current_chunk is None
    assert restored.total_chunks is None


def test_generate_request_round_trip() -> None:
    obj = GenerateRequest(word_count=100, auto_correct=True)
    assert round_trip(obj) == obj


def test_generate_request_defaults() -> None:
    obj = GenerateRequest(word_count=50)
    assert obj.auto_correct is False


def test_model_output_round_trip() -> None:
    obj = ModelOutput(
        model_name="char_2gram",
        raw_text="hello wrld",
        corrected_text="hello world",
        word_results=[("hello", True), ("wrld", False)],
        real_word_pct=0.5,
    )
    restored = round_trip(obj)
    assert restored.word_results == [("hello", True), ("wrld", False)]


def test_generate_response_round_trip() -> None:
    outputs = [
        ModelOutput(
            model_name=f"model_{i}",
            raw_text="test",
            corrected_text=None,
            word_results=[("test", True)],
            real_word_pct=1.0,
        )
        for i in range(13)
    ]
    obj = GenerateResponse(outputs=outputs)
    restored = round_trip(obj)
    assert len(restored.outputs) == 13


def test_accuracy_point_null_perplexity() -> None:
    obj = AccuracyPoint(chunk_id=1, accuracy=0.55, perplexity=None, timestamp="2026-04-21T00:00:00")
    restored = round_trip(obj)
    assert restored.perplexity is None


def test_accuracy_history_response_round_trip() -> None:
    obj = AccuracyHistoryResponse(
        models={"char_2gram": [AccuracyPoint(chunk_id=1, accuracy=0.5, perplexity=None, timestamp="t")]}
    )
    assert round_trip(obj) == obj


def test_ngram_page_response_round_trip() -> None:
    obj = NGramPageResponse(
        rows=[NGramRow(context="th", next_item="e", count=10, probability=0.7)],
        total=100,
        page=1,
        page_size=50,
    )
    assert round_trip(obj) == obj


def test_db_reset_response_round_trip() -> None:
    obj = DBResetResponse(success=True, message="Reset complete")
    assert round_trip(obj) == obj


# ---------------------------------------------------------------------------
# GenerateRequest validation — field constraints
# ---------------------------------------------------------------------------

def test_generate_request_below_min_raises() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(word_count=10)   # below ge=20


def test_generate_request_above_max_raises() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(word_count=501)  # above le=500


def test_generate_request_at_min_valid() -> None:
    obj = GenerateRequest(word_count=20)
    assert obj.word_count == 20


def test_generate_request_at_max_valid() -> None:
    obj = GenerateRequest(word_count=500)
    assert obj.word_count == 500


# ---------------------------------------------------------------------------
# WebSocket models — round-trip
# ---------------------------------------------------------------------------

def test_ws_chunk_start_round_trip() -> None:
    obj = WSChunkStart(chunk_index=2, total_chunks=10, operation="Counting char 2-grams")
    assert round_trip(obj) == obj
    assert obj.type == "chunk_start"


def test_ws_chunk_progress_round_trip() -> None:
    obj = WSChunkProgress(operation="Training feedforward", pct=62)
    assert round_trip(obj) == obj


def test_ws_mc_token_round_trip() -> None:
    obj = WSMCToken(model="char_3gram", token="th", run=12)
    assert round_trip(obj) == obj


def test_ws_mc_complete_round_trip() -> None:
    obj = WSMCComplete(model="char_3gram", accuracy=0.47, run=12)
    assert round_trip(obj) == obj


def test_ws_chunk_done_round_trip() -> None:
    obj = WSChunkDone(
        chunk_index=4,
        status="accepted",
        accuracy_delta={"char_3gram": 0.03},
        reason=None,
    )
    restored = round_trip(obj)
    assert restored.reason is None
    assert restored.accuracy_delta == {"char_3gram": 0.03}


def test_ws_ingest_complete_round_trip() -> None:
    obj = WSIngestComplete(chunks_accepted=14, chunks_rejected=3)
    assert round_trip(obj) == obj


# ---------------------------------------------------------------------------
# Discriminated union — WSMessageAnnotated
# ---------------------------------------------------------------------------

_WS_ADAPTER: TypeAdapter = TypeAdapter(WSMessageAnnotated)


def _parse_ws(data: dict):
    return _WS_ADAPTER.validate_python(data)


def test_ws_discriminated_union_chunk_start() -> None:
    msg = _parse_ws({"type": "chunk_start", "chunk_index": 0, "total_chunks": 5, "operation": "x"})
    assert isinstance(msg, WSChunkStart)


def test_ws_discriminated_union_chunk_progress() -> None:
    msg = _parse_ws({"type": "chunk_progress", "operation": "y", "pct": 50})
    assert isinstance(msg, WSChunkProgress)


def test_ws_discriminated_union_mc_token() -> None:
    msg = _parse_ws({"type": "mc_token", "model": "m", "token": "t", "run": 1})
    assert isinstance(msg, WSMCToken)


def test_ws_discriminated_union_mc_complete() -> None:
    msg = _parse_ws({"type": "mc_complete", "model": "m", "accuracy": 0.5, "run": 1})
    assert isinstance(msg, WSMCComplete)


def test_ws_discriminated_union_chunk_done() -> None:
    msg = _parse_ws({
        "type": "chunk_done",
        "chunk_index": 2,
        "status": "rejected",
        "accuracy_delta": {},
        "reason": "dropped",
    })
    assert isinstance(msg, WSChunkDone)


def test_ws_discriminated_union_ingest_complete() -> None:
    msg = _parse_ws({"type": "ingest_complete", "chunks_accepted": 5, "chunks_rejected": 1})
    assert isinstance(msg, WSIngestComplete)


def test_ws_discriminated_union_invalid_type_raises() -> None:
    with pytest.raises((ValidationError, Exception)):
        _parse_ws({"type": "unknown_type", "foo": "bar"})


def test_ws_model_dump_json_produces_valid_json() -> None:
    """Every WS model's JSON must be parseable by stdlib json."""
    models = [
        WSChunkStart(chunk_index=0, total_chunks=1, operation="x"),
        WSChunkProgress(operation="y", pct=10),
        WSMCToken(model="m", token="t", run=1),
        WSMCComplete(model="m", accuracy=0.5, run=1),
        WSChunkDone(chunk_index=0, status="accepted", accuracy_delta={}, reason=None),
        WSIngestComplete(chunks_accepted=1, chunks_rejected=0),
    ]
    for m in models:
        raw = m.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["type"] == m.type
