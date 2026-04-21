"""Tests for eval/monte_carlo.py — MonteCarloEvaluator, ModelSpec registry, and scoring."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eval.monte_carlo import (
    MODELS,
    RUNS_PER_MODEL,
    ModelSpec,
    MonteCarloEvaluator,
    _score_sample_text,
)


# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path):
    """Return a fresh Store backed by a temporary DuckDB file."""
    from db.store import Store

    return Store(db_path=tmp_path / "test.duckdb")


def _make_validator():
    """Return a Validator backed by the real SCOWL wordlist."""
    from wvm.validator import Validator

    return Validator()


def _make_mock_trainer() -> MagicMock:
    """Return a mock satisfying FeedforwardTrainer / TransformerTrainer generate API.

    Returns an empty token list so _generate_sample → '' → score 0.0.
    This avoids any torch.* dependency in evaluator tests.
    """
    mock = MagicMock()
    mock.generate.return_value = []
    return mock


def _make_evaluator(tmp_path: Path) -> tuple[MonteCarloEvaluator, int]:
    """Create an evaluator with mock trainers and return (evaluator, chunk_id).

    A dummy corpus_chunk is inserted so chunk_id satisfies FK constraints.
    """
    store = _make_store(tmp_path)
    chunk_id = store.insert_chunk("test.txt", 0, "dummyhash", 100, {})
    evaluator = MonteCarloEvaluator(
        store=store,
        validator=_make_validator(),
        feedforward=_make_mock_trainer(),
        transformer=_make_mock_trainer(),
        db_path=tmp_path / "test.duckdb",
    )
    return evaluator, chunk_id


# ---------------------------------------------------------------------------
# ModelSpec registry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_exactly_13_models(self):
        """MODELS list must have exactly 13 entries."""
        assert len(MODELS) == 13

    def test_unique_names(self):
        """Every model name is unique."""
        names = [s.name for s in MODELS]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"

    def test_display_orders_are_1_to_13(self):
        """display_order values form the exact set {1, 2, …, 13}."""
        orders = sorted(s.display_order for s in MODELS)
        assert orders == list(range(1, 14))

    def test_feedforward_display_order_12(self):
        ff = next(s for s in MODELS if s.name == "feedforward")
        assert ff.display_order == 12

    def test_transformer_display_order_13(self):
        tr = next(s for s in MODELS if s.name == "transformer")
        assert tr.display_order == 13

    def test_neural_models_have_none_order(self):
        """Neural model specs carry order=None."""
        for spec in MODELS:
            if spec.family == "neural":
                assert spec.order is None, f"{spec.name}.order should be None"

    def test_char_model_orders_1_to_5(self):
        char_orders = sorted(s.order for s in MODELS if s.family == "char")
        assert char_orders == [1, 2, 3, 4, 5]

    def test_word_model_orders_1_to_3(self):
        word_orders = sorted(s.order for s in MODELS if s.family == "word")
        assert word_orders == [1, 2, 3]

    def test_bpe_model_orders_1_to_3(self):
        bpe_orders = sorted(s.order for s in MODELS if s.family == "bpe")
        assert bpe_orders == [1, 2, 3]

    def test_all_families_present(self):
        families = {s.family for s in MODELS}
        assert families == {"char", "word", "bpe", "neural"}

    def test_modelspec_is_frozen(self):
        """ModelSpec instances must be immutable (frozen dataclass)."""
        spec = MODELS[0]
        with pytest.raises((AttributeError, TypeError)):
            spec.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _score_sample helper (module-level thin wrapper for test isolation)
# ---------------------------------------------------------------------------


class TestScoreSampleHelper:
    """Tests for the module-level _score_sample_text() helper."""

    def test_real_english_words_return_high_accuracy(self):
        """Common English words should score ≥ 0.9."""
        score = _score_sample_text("the cat sat on the mat")
        assert score >= 0.9, f"Expected ≥0.9 for real words, got {score}"

    def test_nonsense_tokens_return_zero(self):
        """Random consonant clusters score 0.0."""
        score = _score_sample_text("qxzp wxyz bzzft")
        assert score == 0.0

    def test_empty_string_returns_zero(self):
        assert _score_sample_text("") == 0.0

    def test_whitespace_only_returns_zero(self):
        assert _score_sample_text("   \t\n  ") == 0.0

    def test_mixed_real_and_nonsense(self):
        """Half real words → score ≈ 0.5."""
        score = _score_sample_text("the qxzp cat wxyz")
        assert 0.4 <= score <= 0.6, f"Expected ~0.5, got {score}"


# ---------------------------------------------------------------------------
# MonteCarloEvaluator._score_sample (instance method)
# ---------------------------------------------------------------------------


class TestInstanceScoreSample:
    def test_real_words_high_accuracy(self, tmp_path: Path):
        evaluator, _ = _make_evaluator(tmp_path)
        assert evaluator._score_sample("the cat sat") >= 0.9

    def test_nonsense_returns_zero(self, tmp_path: Path):
        evaluator, _ = _make_evaluator(tmp_path)
        assert evaluator._score_sample("qxzp wxyz") == 0.0

    def test_empty_returns_zero(self, tmp_path: Path):
        evaluator, _ = _make_evaluator(tmp_path)
        assert evaluator._score_sample("") == 0.0


# ---------------------------------------------------------------------------
# MonteCarloEvaluator.evaluate_all — structure & DB persistence
# ---------------------------------------------------------------------------


class TestEvaluateAll:
    def test_returns_dict_with_13_keys(self, tmp_path: Path):
        """evaluate_all always returns exactly 13 model names."""
        evaluator, chunk_id = _make_evaluator(tmp_path)
        results = evaluator.evaluate_all(chunk_id=chunk_id)
        assert len(results) == 13
        for spec in MODELS:
            assert spec.name in results, f"{spec.name} missing from results"

    def test_all_accuracies_are_floats_in_range(self, tmp_path: Path):
        """Every accuracy value is a float in [0.0, 1.0]."""
        evaluator, chunk_id = _make_evaluator(tmp_path)
        results = evaluator.evaluate_all(chunk_id=chunk_id)
        for name, acc in results.items():
            assert isinstance(acc, float), f"{name}: accuracy is not float"
            assert 0.0 <= acc <= 1.0, f"{name}: accuracy {acc} out of range"

    def test_accuracy_rows_inserted_in_db(self, tmp_path: Path):
        """After evaluate_all, DuckDB model_accuracy contains 13 rows."""
        from db.store import Store

        store = Store(db_path=tmp_path / "test.duckdb")
        chunk_id = store.insert_chunk("corpus.txt", 0, "hash42", 500, {})
        evaluator = MonteCarloEvaluator(
            store=store,
            validator=_make_validator(),
            feedforward=_make_mock_trainer(),
            transformer=_make_mock_trainer(),
            db_path=tmp_path / "test.duckdb",
        )
        evaluator.evaluate_all(chunk_id=chunk_id)

        rows = store.get_accuracy_history()
        assert len(rows) == 13, f"Expected 13 accuracy rows, got {len(rows)}"

    def test_accuracy_rows_have_correct_chunk_id(self, tmp_path: Path):
        """Inserted rows reference the chunk_id passed to evaluate_all."""
        from db.store import Store

        store = Store(db_path=tmp_path / "test.duckdb")
        chunk_id = store.insert_chunk("corpus.txt", 0, "hash99", 200, {})
        evaluator = MonteCarloEvaluator(
            store=store,
            validator=_make_validator(),
            feedforward=_make_mock_trainer(),
            transformer=_make_mock_trainer(),
            db_path=tmp_path / "test.duckdb",
        )
        evaluator.evaluate_all(chunk_id=chunk_id)

        rows = store.get_accuracy_history()
        assert all(r["chunk_id"] == chunk_id for r in rows), (
            "Some rows have wrong chunk_id"
        )

    def test_mock_trainers_substituted_without_torch(self, tmp_path: Path):
        """evaluate_all works with mock trainers — no direct torch.* in evaluator."""
        evaluator, chunk_id = _make_evaluator(tmp_path)
        mock_ff = evaluator._feedforward
        mock_tr = evaluator._transformer

        evaluator.evaluate_all(chunk_id=chunk_id)

        # Mocks must have been called (neural models delegated to trainers)
        mock_ff.generate.assert_called()
        mock_tr.generate.assert_called()

    def test_no_progress_callback_does_not_raise(self, tmp_path: Path):
        """evaluate_all runs successfully with on_progress=None."""
        evaluator, chunk_id = _make_evaluator(tmp_path)
        # Should not raise
        results = evaluator.evaluate_all(chunk_id=chunk_id, on_progress=None)
        assert len(results) == 13


# ---------------------------------------------------------------------------
# Progress callback event sequence
# ---------------------------------------------------------------------------


class TestProgressCallback:
    def test_fires_13_model_start_events(self, tmp_path: Path):
        """Exactly 13 mc_model_start events are emitted."""
        evaluator, chunk_id = _make_evaluator(tmp_path)
        events: list[tuple[str, dict]] = []

        def record(event_type: str, payload: dict) -> None:
            events.append((event_type, payload))

        evaluator.evaluate_all(chunk_id=chunk_id, on_progress=record)
        starts = [e for e in events if e[0] == "mc_model_start"]
        assert len(starts) == 13

    def test_fires_650_complete_events(self, tmp_path: Path):
        """Exactly 50 × 13 = 650 mc_complete events are emitted."""
        evaluator, chunk_id = _make_evaluator(tmp_path)
        events: list[tuple[str, dict]] = []

        def record(event_type: str, payload: dict) -> None:
            events.append((event_type, payload))

        evaluator.evaluate_all(chunk_id=chunk_id, on_progress=record)
        completions = [e for e in events if e[0] == "mc_complete"]
        assert len(completions) == 13 * RUNS_PER_MODEL

    def test_each_model_gets_50_complete_events(self, tmp_path: Path):
        """Each of the 13 models fires exactly RUNS_PER_MODEL mc_complete events."""
        evaluator, chunk_id = _make_evaluator(tmp_path)
        events: list[tuple[str, dict]] = []

        def record(event_type: str, payload: dict) -> None:
            events.append((event_type, payload))

        evaluator.evaluate_all(chunk_id=chunk_id, on_progress=record)
        completions = [e for e in events if e[0] == "mc_complete"]
        counts = Counter(e[1]["model"] for e in completions)
        for spec in MODELS:
            assert counts[spec.name] == RUNS_PER_MODEL, (
                f"{spec.name}: expected {RUNS_PER_MODEL} completions, got {counts[spec.name]}"
            )

    def test_model_start_before_completions_for_neural(self, tmp_path: Path):
        """For neural models, mc_model_start fires before any mc_complete for that model."""
        evaluator, chunk_id = _make_evaluator(tmp_path)
        events: list[tuple[str, dict]] = []

        def record(event_type: str, payload: dict) -> None:
            events.append((event_type, payload))

        evaluator.evaluate_all(chunk_id=chunk_id, on_progress=record)

        for neural_name in ("feedforward", "transformer"):
            start_idx = next(
                (i for i, e in enumerate(events)
                 if e[0] == "mc_model_start" and e[1]["model"] == neural_name),
                None,
            )
            first_complete_idx = next(
                (i for i, e in enumerate(events)
                 if e[0] == "mc_complete" and e[1]["model"] == neural_name),
                None,
            )
            assert start_idx is not None, f"No mc_model_start for {neural_name}"
            assert first_complete_idx is not None, f"No mc_complete for {neural_name}"
            assert start_idx < first_complete_idx, (
                f"{neural_name}: mc_model_start at {start_idx} "
                f"must precede first mc_complete at {first_complete_idx}"
            )

    def test_complete_payloads_have_required_keys(self, tmp_path: Path):
        """Every mc_complete payload has 'model', 'accuracy', and 'run' keys."""
        evaluator, chunk_id = _make_evaluator(tmp_path)
        events: list[tuple[str, dict]] = []

        def record(event_type: str, payload: dict) -> None:
            events.append((event_type, payload))

        evaluator.evaluate_all(chunk_id=chunk_id, on_progress=record)
        for etype, payload in events:
            if etype == "mc_complete":
                assert "model" in payload
                assert "accuracy" in payload
                assert "run" in payload
