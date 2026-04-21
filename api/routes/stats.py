"""Stats endpoints — model registry, accuracy history, last generation outputs."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter

from api import state
from api.contracts import AccuracyHistoryResponse, AccuracyPoint, LastOutputResponse
from eval.monte_carlo import MODELS

router = APIRouter()


@router.get("/models")
async def models() -> list[dict[str, Any]]:
    """Return the 13 model descriptors (name, family, order, display_order)."""
    return [asdict(m) for m in MODELS]


@router.get("/accuracy", response_model=AccuracyHistoryResponse)
async def accuracy() -> AccuracyHistoryResponse:
    """Return accuracy history for all 13 models, grouped by model name.

    Uses ``store.get_accuracy_history()`` (no filter) and groups rows by
    model_name — the ``get_accuracy_history_all`` convenience method is not
    yet in Track 0's Store; this is the documented fallback.
    """
    all_rows = state.store.get_accuracy_history()  # type: ignore[union-attr]
    grouped: dict[str, list[AccuracyPoint]] = {}
    for row in all_rows:
        point = AccuracyPoint(
            chunk_id=row["chunk_id"],
            accuracy=row["accuracy"],
            perplexity=row["perplexity"],
            timestamp=row["timestamp"],
        )
        grouped.setdefault(row["model_name"], []).append(point)
    return AccuracyHistoryResponse(models=grouped)


@router.get("/last_generations", response_model=LastOutputResponse)
async def last_generations() -> LastOutputResponse:
    """Return the most recent generated text for each model.

    NOTE: The ``last_generations`` table is not part of Track 0's schema.
    This endpoint returns an empty list until that table is added.
    """
    return LastOutputResponse(outputs=[])
