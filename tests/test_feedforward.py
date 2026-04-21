"""Tests for model/feedforward.py — FeedforwardLM and FeedforwardTrainer."""

from pathlib import Path

import pytest
import torch

from model.feedforward import (
    ENDOFTEXT_ID,
    PAD_ID,
    VOCAB_SIZE,
    FeedforwardLM,
    FeedforwardTrainer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CPU = torch.device("cpu")


def _make_store(tmp_path: Path):
    """Return a fresh Store backed by a temporary DuckDB file."""
    from db.store import Store

    return Store(db_path=tmp_path / "test.duckdb")


def _make_corpus(n_tokens: int = 2000, vocab_size: int = VOCAB_SIZE) -> list[int]:
    """Generate a deterministic token sequence for training tests."""
    torch.manual_seed(42)
    return (
        torch.randint(2, vocab_size, (n_tokens,)).tolist()
    )  # avoid special IDs 0 and 1


# ---------------------------------------------------------------------------
# FeedforwardLM — architecture tests
# ---------------------------------------------------------------------------


class TestFeedforwardLM:
    def test_forward_output_shape(self):
        """Forward pass produces [batch, vocab_size] logits."""
        model = FeedforwardLM()
        x = torch.randint(0, VOCAB_SIZE, (4, 64))  # [batch=4, seq_len=64]
        logits = model(x)
        assert logits.shape == (4, VOCAB_SIZE)

    def test_forward_different_batch_sizes(self):
        """Model accepts batch sizes 1 and 64 without error."""
        model = FeedforwardLM()
        for batch in (1, 64):
            x = torch.randint(0, VOCAB_SIZE, (batch, 64))
            out = model(x)
            assert out.shape == (batch, VOCAB_SIZE)

    def test_forward_on_cpu(self):
        """Forward pass works on CPU device."""
        model = FeedforwardLM().to(CPU)
        x = torch.randint(0, VOCAB_SIZE, (2, 64))
        out = model(x)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# FeedforwardTrainer — training
# ---------------------------------------------------------------------------


class TestFeedforwardTrainerTraining:
    def test_train_step_returns_float(self, tmp_path: Path):
        """train_step returns a finite float loss."""
        store = _make_store(tmp_path)
        trainer = FeedforwardTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        corpus = _make_corpus(500)
        loss = trainer.train_step(corpus, chunk_id=1)
        assert isinstance(loss, float)
        assert loss < float("inf")

    def test_loss_decreases_over_five_steps(self, tmp_path: Path):
        """Training loss should trend down over 5 consecutive passes on the same corpus."""
        torch.manual_seed(42)
        store = _make_store(tmp_path)
        trainer = FeedforwardTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        corpus = _make_corpus(2000)

        losses: list[float] = []
        for chunk_id in range(5):
            loss = trainer.train_step(corpus, chunk_id=chunk_id)
            losses.append(loss)

        # Loss should be strictly lower at the end than at the start
        assert losses[-1] < losses[0], (
            f"Expected loss to decrease; got {losses}"
        )

    def test_validate_returns_float(self, tmp_path: Path):
        """validate returns a finite float on a short token list."""
        store = _make_store(tmp_path)
        trainer = FeedforwardTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        corpus = _make_corpus(200)
        val_loss = trainer.validate(corpus)
        assert isinstance(val_loss, float)
        assert val_loss < float("inf")

    def test_validate_short_corpus_returns_inf(self, tmp_path: Path):
        """validate on a corpus shorter than seq_len returns inf (no batches)."""
        store = _make_store(tmp_path)
        trainer = FeedforwardTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        short = list(range(10))  # 10 tokens < seq_len=64
        result = trainer.validate(short)
        assert result == float("inf")


# ---------------------------------------------------------------------------
# FeedforwardTrainer — generation
# ---------------------------------------------------------------------------


class TestFeedforwardTrainerGenerate:
    def test_generate_returns_correct_length(self, tmp_path: Path):
        """generate(n_tokens=20) returns at most 20 tokens."""
        store = _make_store(tmp_path)
        trainer = FeedforwardTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        result = trainer.generate(n_tokens=20)
        assert 1 <= len(result) <= 20

    def test_generate_all_tokens_in_vocab_range(self, tmp_path: Path):
        """All generated token IDs are valid vocab indices."""
        torch.manual_seed(42)
        store = _make_store(tmp_path)
        trainer = FeedforwardTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        rng = torch.Generator()
        rng.manual_seed(42)
        result = trainer.generate(n_tokens=20, rng=rng)
        assert all(0 <= t < VOCAB_SIZE for t in result), (
            f"Out-of-range token IDs: {[t for t in result if not (0 <= t < VOCAB_SIZE)]}"
        )

    def test_generate_with_seed_context(self, tmp_path: Path):
        """generate with a seed_context shorter than seq_len pads correctly."""
        store = _make_store(tmp_path)
        trainer = FeedforwardTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        seed = [42, 100, 200]
        result = trainer.generate(n_tokens=5, seed_context=seed)
        assert 1 <= len(result) <= 5
        assert all(0 <= t < VOCAB_SIZE for t in result)

    def test_generate_stops_on_endoftext(self, tmp_path: Path):
        """generate stops immediately when ENDOFTEXT_ID is sampled."""
        store = _make_store(tmp_path)
        trainer = FeedforwardTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        # Force the model to always produce ENDOFTEXT by zeroing all logits
        # then setting logit for ID 0 to a large positive value.
        with torch.no_grad():
            last_linear = trainer.model.net[-1]
            last_linear.weight.zero_()
            last_linear.bias.zero_()
            last_linear.bias[ENDOFTEXT_ID] = 100.0

        result = trainer.generate(n_tokens=20)
        assert result == [ENDOFTEXT_ID], (
            f"Expected early stop on ENDOFTEXT; got {result}"
        )

    def test_generate_returns_list_of_ints(self, tmp_path: Path):
        """generate return type is list[int]."""
        store = _make_store(tmp_path)
        trainer = FeedforwardTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        result = trainer.generate(n_tokens=10)
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)


# ---------------------------------------------------------------------------
# FeedforwardTrainer — checkpoints
# ---------------------------------------------------------------------------


class TestFeedforwardTrainerCheckpoints:
    def test_checkpoint_roundtrip_preserves_weights(self, tmp_path: Path):
        """save_checkpoint + load_checkpoint produces identical model outputs."""
        torch.manual_seed(42)
        store = _make_store(tmp_path)
        ckpt_dir = tmp_path / "ckpts"
        trainer = FeedforwardTrainer(store, device=CPU, checkpoint_dir=ckpt_dir)

        # Record outputs before saving
        x = torch.randint(0, VOCAB_SIZE, (2, 64))
        with torch.no_grad():
            logits_before = trainer.model(x).clone()

        saved_path = trainer.save_checkpoint(chunk_id=99, val_loss=3.14)
        assert saved_path.exists()
        assert saved_path.name == "feedforward_99.pt"

        # Scramble weights
        for param in trainer.model.parameters():
            param.data.uniform_(-1.0, 1.0)

        # Reload and verify
        trainer.load_checkpoint(saved_path)
        with torch.no_grad():
            logits_after = trainer.model(x)

        assert torch.allclose(logits_before, logits_after), (
            "Weights not restored correctly after checkpoint round-trip"
        )

    def test_checkpoint_file_under_30mb(self, tmp_path: Path):
        """Checkpoint file is smaller than 30 MB.

        Note: the architecture has ~6.6M float32 parameters (~26 MB raw),
        so the 30 MB ceiling gives reasonable headroom for pickle overhead.
        """
        store = _make_store(tmp_path)
        trainer = FeedforwardTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        path = trainer.save_checkpoint(chunk_id=0, val_loss=0.0)
        size_mb = path.stat().st_size / (1024**2)
        assert size_mb < 30, f"Checkpoint too large: {size_mb:.1f} MB"

    def test_checkpoint_registered_in_db(self, tmp_path: Path):
        """save_checkpoint writes a row to the nn_checkpoints table."""
        store = _make_store(tmp_path)
        trainer = FeedforwardTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        trainer.save_checkpoint(chunk_id=7, val_loss=2.5)
        row = store.get_latest_checkpoint("feedforward")
        assert row is not None
        assert row["chunk_id"] == 7
        assert abs(row["val_loss"] - 2.5) < 1e-5

    def test_maybe_load_latest_checkpoint_loads_existing(self, tmp_path: Path):
        """_maybe_load_latest_checkpoint loads weights when a checkpoint exists."""
        torch.manual_seed(0)
        ckpt_dir = tmp_path / "ckpts"
        store = _make_store(tmp_path)

        # First trainer — save a checkpoint
        trainer_a = FeedforwardTrainer(store, device=CPU, checkpoint_dir=ckpt_dir)
        # Perturb weights to a known state
        with torch.no_grad():
            for p in trainer_a.model.parameters():
                p.fill_(0.1)
        trainer_a.save_checkpoint(chunk_id=1, val_loss=1.0)

        x = torch.randint(0, VOCAB_SIZE, (1, 64))
        with torch.no_grad():
            expected = trainer_a.model(x).clone()

        # Second trainer — should auto-load the saved checkpoint
        trainer_b = FeedforwardTrainer(store, device=CPU, checkpoint_dir=ckpt_dir)
        with torch.no_grad():
            actual = trainer_b.model(x)

        assert torch.allclose(expected, actual), (
            "_maybe_load_latest_checkpoint did not restore weights"
        )

    def test_maybe_load_latest_checkpoint_no_op_when_empty(self, tmp_path: Path):
        """_maybe_load_latest_checkpoint is a no-op when no checkpoint exists."""
        store = _make_store(tmp_path)
        # Should not raise
        trainer = FeedforwardTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        assert trainer.model is not None  # successfully initialised

    def test_maybe_load_latest_checkpoint_missing_file(self, tmp_path: Path):
        """_maybe_load_latest_checkpoint skips loading if the .pt file is gone."""
        ckpt_dir = tmp_path / "ckpts"
        store = _make_store(tmp_path)

        # Register a checkpoint row but delete the actual file
        trainer_a = FeedforwardTrainer(store, device=CPU, checkpoint_dir=ckpt_dir)
        path = trainer_a.save_checkpoint(chunk_id=5, val_loss=0.9)
        path.unlink()  # delete the file

        # Should not raise — just skip loading
        trainer_b = FeedforwardTrainer(store, device=CPU, checkpoint_dir=ckpt_dir)
        assert trainer_b.model is not None


# ---------------------------------------------------------------------------
# Optional GPU test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
def test_cuda_forward_pass(tmp_path: Path):
    """Verify forward pass works on CUDA device."""
    cuda = torch.device("cuda")
    store = _make_store(tmp_path)
    trainer = FeedforwardTrainer(store, device=cuda, checkpoint_dir=tmp_path / "ckpts")
    x = torch.randint(0, VOCAB_SIZE, (4, trainer._seq_len), device=cuda)
    with torch.no_grad():
        logits = trainer.model(x)
    assert logits.shape == (4, VOCAB_SIZE)
    assert logits.device.type == "cuda"
