"""Pydantic v2 contracts — source of truth for every HTTP and WebSocket message shape.

Track E (React frontend) transcribes these models to TypeScript interfaces.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ingest endpoints
# ---------------------------------------------------------------------------


class IngestUploadResponse(BaseModel):
    """Response from POST /ingest/upload."""

    accepted_files: list[str]
    rejected_files: list[dict]   # [{filename: str, reason: str}]
    total_chunks: int


class IngestStatusResponse(BaseModel):
    """Response from GET /ingest/status."""

    state: Literal["idle", "running", "paused", "complete"]
    current_chunk: int | None
    total_chunks: int | None
    chunks_accepted: int
    chunks_rejected: int


class IngestPauseResponse(BaseModel):
    """Response from POST /ingest/pause."""

    paused: bool
    message: str


# ---------------------------------------------------------------------------
# Generation endpoints
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Body for POST /generate."""

    word_count: Annotated[int, Field(ge=20, le=500)]
    auto_correct: bool = False


class ModelOutput(BaseModel):
    """Single model's output entry in GenerateResponse.outputs."""

    model_name: str
    raw_text: str
    corrected_text: str | None
    word_results: list[tuple[str, bool]]   # (word, is_real)
    real_word_pct: float


class GenerateResponse(BaseModel):
    """Response from POST /generate — 13 model outputs."""

    outputs: list[ModelOutput]


# ---------------------------------------------------------------------------
# Stats endpoints
# ---------------------------------------------------------------------------


class AccuracyPoint(BaseModel):
    """One data point in a model's accuracy history."""

    chunk_id: int
    accuracy: float
    perplexity: float | None
    timestamp: str


class AccuracyHistoryResponse(BaseModel):
    """Response from GET /stats/accuracy."""

    models: dict[str, list[AccuracyPoint]]   # {model_name: [AccuracyPoint, ...]}


class LastOutputEntry(BaseModel):
    """One model's most-recent MC-generated text sample."""

    model_name: str
    text: str
    real_word_pct: float


class LastOutputResponse(BaseModel):
    """Response from GET /stats/last-output."""

    outputs: list[LastOutputEntry]


# ---------------------------------------------------------------------------
# DB viewer endpoints
# ---------------------------------------------------------------------------


class NGramRow(BaseModel):
    """One row in the paginated n-gram table."""

    context: str
    next_item: str
    count: int
    probability: float


class NGramPageResponse(BaseModel):
    """Response from GET /db/ngrams."""

    rows: list[NGramRow]
    total: int
    page: int
    page_size: int


class VocabularyRow(BaseModel):
    """One row in the vocabulary table.

    ``frequency`` defaults to 0 because the current DB schema does not store
    per-token frequency; callers may populate it when the column is added.
    """

    token_id: int
    token: str
    source: str
    frequency: int = 0


class VocabularyPageResponse(BaseModel):
    """Response from GET /db/vocabulary."""

    rows: list[VocabularyRow]
    total: int
    page: int
    page_size: int


class AccuracyHistoryRow(BaseModel):
    """One row from the full model_accuracy table (DB viewer)."""

    id: int
    model_name: str
    chunk_id: int
    accuracy: float
    perplexity: float | None
    timestamp: str


class AccuracyHistoryTableResponse(BaseModel):
    """Response from GET /db/accuracy-history."""

    rows: list[AccuracyHistoryRow]
    total: int


class DBResetResponse(BaseModel):
    """Response from POST /db/reset."""

    success: bool
    message: str


# ---------------------------------------------------------------------------
# WebSocket messages — discriminated union on the `type` field
# ---------------------------------------------------------------------------


class WSChunkStart(BaseModel):
    """Emitted when the worker begins processing a new chunk."""

    type: Literal["chunk_start"] = "chunk_start"
    chunk_index: int
    total_chunks: int
    operation: str


class WSChunkProgress(BaseModel):
    """Emitted periodically while a chunk operation is running."""

    type: Literal["chunk_progress"] = "chunk_progress"
    operation: str
    pct: int   # 0–100


class WSMCToken(BaseModel):
    """Emitted for each token generated during a Monte Carlo run."""

    type: Literal["mc_token"] = "mc_token"
    model: str
    token: str
    run: int   # which of the 50 runs (1-indexed)


class WSMCComplete(BaseModel):
    """Emitted when a model's Monte Carlo evaluation finishes for one run."""

    type: Literal["mc_complete"] = "mc_complete"
    model: str
    accuracy: float
    run: int


class WSChunkDone(BaseModel):
    """Emitted when a chunk is fully processed (accepted or rejected)."""

    type: Literal["chunk_done"] = "chunk_done"
    chunk_index: int
    status: Literal["accepted", "rejected"]
    accuracy_delta: dict[str, float]
    reason: str | None = None


class WSIngestComplete(BaseModel):
    """Emitted when the ingest worker finishes all chunks."""

    type: Literal["ingest_complete"] = "ingest_complete"
    chunks_accepted: int
    chunks_rejected: int


# Discriminated union — validated with model_validate({'type': ..., ...})
WSMessage = Union[
    WSChunkStart,
    WSChunkProgress,
    WSMCToken,
    WSMCComplete,
    WSChunkDone,
    WSIngestComplete,
]

# Annotated form for Pydantic discriminated-union parsing
WSMessageAnnotated = Annotated[WSMessage, Field(discriminator="type")]
