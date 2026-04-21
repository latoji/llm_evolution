"""Tests for model/transformer.py — TransformerLM and TransformerTrainer.

The most important test here is ``test_causal_mask_prevents_future_leakage``:
if the causal mask is wrong, the model is silently broken.
"""

import math
from pathlib import Path

import pytest
import torch

from model.transformer import (
    D_MODEL,
    ENDOFTEXT_ID,
    PAD_ID,
    SEQ_LEN,
    VOCAB_SIZE,
    TransformerLM,
    TransformerTrainer,
    causal_mask,
    sinusoidal_positional_encoding,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CPU = torch.device("cpu")


def _make_store(tmp_path: Path):
    """Return a fresh Store backed by a temporary DuckDB file."""
    from db.store import Store

    return Store(db_path=tmp_path / "test.duckdb")


def _make_corpus(n_tokens: int = 4000, vocab_size: int = VOCAB_SIZE) -> list[int]:
    """Generate a deterministic token sequence for training tests."""
    torch.manual_seed(42)
    # Avoid special IDs 0 (ENDOFTEXT) and 1 (PAD).
    return torch.randint(2, vocab_size, (n_tokens,)).tolist()


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class TestSinusoidalPositionalEncoding:
    def test_shape(self):
        """PE has shape [seq_len, d_model]."""
        pe = sinusoidal_positional_encoding(64, 128)
        assert pe.shape == (64, 128)

    def test_pe_at_position_zero(self):
        """PE(0, 0) = sin(0) = 0 and PE(0, 1) = cos(0) = 1."""
        pe = sinusoidal_positional_encoding(64, 128)
        assert pe[0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert pe[0, 1].item() == pytest.approx(1.0, abs=1e-6)

    def test_pe_formula_second_even_index(self):
        """PE(1, 0) = sin(1 / 10000^0) = sin(1)."""
        pe = sinusoidal_positional_encoding(4, 8)
        assert pe[1, 0].item() == pytest.approx(math.sin(1.0), abs=1e-5)
        assert pe[1, 1].item() == pytest.approx(math.cos(1.0), abs=1e-5)

    def test_rejects_odd_d_model(self):
        """Odd d_model raises ValueError (sin/cos interleave needs even dim)."""
        with pytest.raises(ValueError):
            sinusoidal_positional_encoding(8, 7)

    def test_rejects_non_positive_seq_len(self):
        """Non-positive seq_len raises ValueError."""
        with pytest.raises(ValueError):
            sinusoidal_positional_encoding(0, 8)


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------


class TestCausalMask:
    def test_shape(self):
        """Mask has shape [seq_len, seq_len]."""
        mask = causal_mask(5, CPU)
        assert mask.shape == (5, 5)

    def test_diagonal_is_zero(self):
        """Diagonal (i, i) is 0 — positions may attend to themselves."""
        mask = causal_mask(5, CPU)
        for i in range(5):
            assert mask[i, i].item() == 0.0

    def test_below_diagonal_is_zero(self):
        """Lower triangle (i > j) is 0 — positions may attend to the past."""
        mask = causal_mask(5, CPU)
        for i in range(5):
            for j in range(i):
                assert mask[i, j].item() == 0.0

    def test_above_diagonal_is_neg_inf(self):
        """Upper triangle (j > i) is -inf — positions may NOT attend to the future."""
        mask = causal_mask(5, CPU)
        for i in range(5):
            for j in range(i + 1, 5):
                assert math.isinf(mask[i, j].item())
                assert mask[i, j].item() < 0


# ---------------------------------------------------------------------------
# TransformerLM — architecture
# ---------------------------------------------------------------------------


class TestTransformerLMArchitecture:
    def test_instantiation_cpu(self):
        """Model constructs with default config on CPU."""
        model = TransformerLM()
        assert isinstance(model, torch.nn.Module)

    def test_forward_output_shape(self):
        """Forward pass: [2, 64] -> [2, 64, 8000]."""
        model = TransformerLM()
        x = torch.randint(0, VOCAB_SIZE, (2, 64))
        logits = model(x)
        assert logits.shape == (2, 64, VOCAB_SIZE)

    def test_forward_batch_size_one(self):
        """Model accepts batch size 1."""
        model = TransformerLM()
        x = torch.randint(0, VOCAB_SIZE, (1, 64))
        logits = model(x)
        assert logits.shape == (1, 64, VOCAB_SIZE)

    def test_forward_short_sequence(self):
        """Model accepts shorter sequences than max_seq_len."""
        model = TransformerLM()
        x = torch.randint(0, VOCAB_SIZE, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, VOCAB_SIZE)

    def test_rejects_oversized_sequence(self):
        """Input longer than max_seq_len raises ValueError."""
        model = TransformerLM(max_seq_len=32)
        x = torch.randint(0, VOCAB_SIZE, (1, 64))
        with pytest.raises(ValueError):
            model(x)

    def test_output_dtype(self):
        """Output logits are float32 on CPU."""
        model = TransformerLM().to(CPU)
        x = torch.randint(0, VOCAB_SIZE, (2, 64))
        out = model(x)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Causal mask leakage — the critical correctness test
# ---------------------------------------------------------------------------


def test_causal_mask_prevents_future_leakage():
    """Changing a future token must not affect the logits at earlier positions.

    This is the defining property of causal attention. If this fails, the mask
    is wrong and the model is silently broken during training.
    """
    torch.manual_seed(0)
    model = TransformerLM().eval()
    x1 = torch.randint(2, VOCAB_SIZE, (1, 64))
    x2 = x1.clone()
    # Change the token at position 32 (a "future" position relative to 0..31).
    x2[0, 32] = (int(x1[0, 32].item()) + 1) % VOCAB_SIZE
    # Sanity: the tokens actually differ.
    assert not torch.equal(x1, x2)
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
    # Positions 0..31 must be identical between the two forward passes.
    assert torch.allclose(out1[:, :32, :], out2[:, :32, :], atol=1e-5), (
        "Causal mask leaks future information: logits at positions < 32 "
        "changed when token at position 32 was modified."
    )
    # Position 32 onwards MUST differ (at least somewhere) — otherwise the model
    # is ignoring its input entirely.
    assert not torch.allclose(out1[:, 32:, :], out2[:, 32:, :], atol=1e-5), (
        "Output at position >= 32 did not change when input at position 32 "
        "was modified — model is ignoring its input."
    )


# ---------------------------------------------------------------------------
# TransformerTrainer — training
# ---------------------------------------------------------------------------


class TestTransformerTrainerTraining:
    def test_train_step_returns_float(self, tmp_path: Path):
        """train_step returns a finite float loss."""
        store = _make_store(tmp_path)
        trainer = TransformerTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        corpus = _make_corpus(500)
        loss = trainer.train_step(corpus, chunk_id=1)
        assert isinstance(loss, float)
        assert loss < float("inf")

    def test_loss_decreases_over_five_steps(self, tmp_path: Path):
        """Training loss trends down over 5 consecutive passes on the same corpus."""
        torch.manual_seed(42)
        store = _make_store(tmp_path)
        trainer = TransformerTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        corpus = _make_corpus(4000)

        losses: list[float] = []
        for chunk_id in range(5):
            loss = trainer.train_step(corpus, chunk_id=chunk_id)
            losses.append(loss)

        assert losses[-1] < losses[0], (
            f"Expected loss to decrease over 5 passes; got {losses}"
        )

    def test_validate_returns_float(self, tmp_path: Path):
        """validate returns a finite float on a reasonable token list."""
        store = _make_store(tmp_path)
        trainer = TransformerTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        corpus = _make_corpus(200)
        val_loss = trainer.validate(corpus)
        assert isinstance(val_loss, float)
        assert val_loss < float("inf")

    def test_validate_short_corpus_returns_inf(self, tmp_path: Path):
        """validate on a corpus shorter than seq_len returns inf."""
        store = _make_store(tmp_path)
        trainer = TransformerTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        short = list(range(10))  # 10 tokens < seq_len=64
        result = trainer.validate(short)
        assert result == float("inf")


# ---------------------------------------------------------------------------
# TransformerTrainer — generation
# ---------------------------------------------------------------------------


class TestTransformerTrainerGenerate:
    def test_generate_returns_correct_length(self, tmp_path: Path):
        """generate(n_tokens=20) returns at most 20 tokens."""
        store = _make_store(tmp_path)
        trainer = TransformerTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        result = trainer.generate(n_tokens=20)
        assert 1 <= len(result) <= 20

    def test_generate_all_tokens_in_vocab_range(self, tmp_path: Path):
        """All generated token IDs are valid vocab indices."""
        torch.manual_seed(42)
        store = _make_store(tmp_path)
        trainer = TransformerTrainer(
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
        trainer = TransformerTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        seed = [42, 100, 200]
        result = trainer.generate(n_tokens=5, seed_context=seed)
        assert 1 <= len(result) <= 5
        assert all(0 <= t < VOCAB_SIZE for t in result)

    def test_generate_stops_on_endoftext(self, tmp_path: Path):
        """generate stops immediately when ENDOFTEXT_ID is sampled."""
        store = _make_store(tmp_path)
        trainer = TransformerTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        # Replace the head with a biased Linear that always prefers ENDOFTEXT.
        # We need bias here because the default head has bias=False and the
        # preceding LayerNorm produces zero-mean outputs — scaling one row of
        # weights alone would collapse to ~0 logit.
        biased_head = torch.nn.Linear(trainer.model.d_model, VOCAB_SIZE, bias=True)
        with torch.no_grad():
            biased_head.weight.zero_()
            biased_head.bias.fill_(-100.0)
            biased_head.bias[ENDOFTEXT_ID] = 100.0
        trainer.model.head = biased_head.to(CPU)

        result = trainer.generate(n_tokens=20)
        assert result == [ENDOFTEXT_ID], (
            f"Expected early stop on ENDOFTEXT; got {result}"
        )

    def test_generate_returns_list_of_ints(self, tmp_path: Path):
        """generate return type is list[int]."""
        store = _make_store(tmp_path)
        trainer = TransformerTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        result = trainer.generate(n_tokens=10)
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)


# ---------------------------------------------------------------------------
# TransformerTrainer — checkpoints
# ---------------------------------------------------------------------------


class TestTransformerTrainerCheckpoints:
    def test_checkpoint_roundtrip_preserves_weights(self, tmp_path: Path):
        """save_checkpoint + load_checkpoint produces identical model outputs."""
        torch.manual_seed(42)
        store = _make_store(tmp_path)
        ckpt_dir = tmp_path / "ckpts"
        trainer = TransformerTrainer(store, device=CPU, checkpoint_dir=ckpt_dir)

        x = torch.randint(0, VOCAB_SIZE, (2, 64))
        trainer.model.eval()
        with torch.no_grad():
            logits_before = trainer.model(x).clone()

        saved_path = trainer.save_checkpoint(chunk_id=99, val_loss=3.14)
        assert saved_path.exists()
        assert saved_path.name == "transformer_99.pt"

        # Scramble weights.
        for param in trainer.model.parameters():
            param.data.uniform_(-1.0, 1.0)

        trainer.load_checkpoint(saved_path)
        trainer.model.eval()
        with torch.no_grad():
            logits_after = trainer.model(x)

        assert torch.allclose(logits_before, logits_after, atol=1e-5), (
            "Weights not restored correctly after checkpoint round-trip"
        )

    def test_checkpoint_file_under_30mb(self, tmp_path: Path):
        """Checkpoint file is smaller than 30 MB."""
        store = _make_store(tmp_path)
        trainer = TransformerTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        path = trainer.save_checkpoint(chunk_id=0, val_loss=0.0)
        size_mb = path.stat().st_size / (1024**2)
        assert size_mb < 30, f"Checkpoint too large: {size_mb:.1f} MB"

    def test_checkpoint_registered_in_db(self, tmp_path: Path):
        """save_checkpoint writes a row to the nn_checkpoints table."""
        store = _make_store(tmp_path)
        trainer = TransformerTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        trainer.save_checkpoint(chunk_id=7, val_loss=2.5)
        row = store.get_latest_checkpoint("transformer")
        assert row is not None
        assert row["chunk_id"] == 7
        assert abs(row["val_loss"] - 2.5) < 1e-5
        assert row["model_name"] == "transformer"

    def test_maybe_load_latest_checkpoint_loads_existing(self, tmp_path: Path):
        """_maybe_load_latest_checkpoint loads weights when a checkpoint exists."""
        torch.manual_seed(0)
        ckpt_dir = tmp_path / "ckpts"
        store = _make_store(tmp_path)

        trainer_a = TransformerTrainer(store, device=CPU, checkpoint_dir=ckpt_dir)
        # Perturb weights to a known, distinctive state.
        with torch.no_grad():
            for p in trainer_a.model.parameters():
                p.fill_(0.05)
        trainer_a.save_checkpoint(chunk_id=1, val_loss=1.0)

        x = torch.randint(0, VOCAB_SIZE, (1, 64))
        trainer_a.model.eval()
        with torch.no_grad():
            expected = trainer_a.model(x).clone()

        trainer_b = TransformerTrainer(store, device=CPU, checkpoint_dir=ckpt_dir)
        trainer_b.model.eval()
        with torch.no_grad():
            actual = trainer_b.model(x)

        assert torch.allclose(expected, actual, atol=1e-5), (
            "_maybe_load_latest_checkpoint did not restore weights"
        )

    def test_maybe_load_latest_checkpoint_no_op_when_empty(self, tmp_path: Path):
        """_maybe_load_latest_checkpoint is a no-op when no checkpoint exists."""
        store = _make_store(tmp_path)
        trainer = TransformerTrainer(
            store, device=CPU, checkpoint_dir=tmp_path / "ckpts"
        )
        assert trainer.model is not None

    def test_maybe_load_latest_checkpoint_missing_file(self, tmp_path: Path):
        """_maybe_load_latest_checkpoint skips loading if the .pt file is gone."""
        ckpt_dir = tmp_path / "ckpts"
        store = _make_store(tmp_path)

        trainer_a = TransformerTrainer(store, device=CPU, checkpoint_dir=ckpt_dir)
        path = trainer_a.save_checkpoint(chunk_id=5, val_loss=0.9)
        path.unlink()

        # Should not raise — just skip loading.
        trainer_b = TransformerTrainer(store, device=CPU, checkpoint_dir=ckpt_dir)
        assert trainer_b.model is not None


# ---------------------------------------------------------------------------
# Optional GPU test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_vram_under_one_gb(tmp_path: Path):
    """Peak VRAM usage stays under 1 GB on GPU with batch=64, seq=64."""
    cuda = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    store = _make_store(tmp_path)
    trainer = TransformerTrainer(
        store, device=cuda, checkpoint_dir=tmp_path / "ckpts"
    )
    x = torch.randint(0, VOCAB_SIZE, (64, trainer._seq_len), device=cuda)
    trainer.model.train()
    logits = trainer.model(x)
    # Fake loss for backward pass — still exercises VRAM.
    loss = logits.sum()
    loss.backward()
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    assert peak_mb < 1024, f"Peak VRAM too high: {peak_mb:.1f} MB"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_forward_pass(tmp_path: Path):
    """Verify forward pass works on CUDA device."""
    cuda = torch.device("cuda")
    store = _make_store(tmp_path)
    trainer = TransformerTrainer(
        store, device=cuda, checkpoint_dir=tmp_path / "ckpts"
    )
    x = torch.randint(0, VOCAB_SIZE, (4, trainer._seq_len), device=cuda)
    with torch.no_grad():
        logits = trainer.model(x)
    assert logits.shape == (4, trainer._seq_len, VOCAB_SIZE)
    assert logits.device.type == "cuda"
