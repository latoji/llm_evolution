"""Causal Transformer language model (Model 13) — GPT-2 family, ~6.8M params."""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from db.store import Store

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

VOCAB_SIZE: int = 8000
D_MODEL: int = 128
N_HEADS: int = 4
DIM_FF: int = 512
N_BLOCKS: int = 2
SEQ_LEN: int = 64

# Default device used as a parameter default only — tests override with cpu.
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Special token IDs — must match tokenizer/bpe.py and feedforward.py
ENDOFTEXT_ID: int = 0
PAD_ID: int = 1

_DEFAULT_CHECKPOINT_DIR: Path = Path("model/checkpoints")


# ---------------------------------------------------------------------------
# Positional encoding & causal mask
# ---------------------------------------------------------------------------


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Return ``[seq_len, d_model]`` fixed (non-trainable) sinusoidal encoding.

    ``PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))``
    ``PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))``
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if d_model <= 0 or d_model % 2 != 0:
        raise ValueError(f"d_model must be a positive even integer, got {d_model}")

    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)  # [seq_len, 1]
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )  # [d_model/2]
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Return ``[seq_len, seq_len]`` mask of ``-inf`` above the diagonal, 0 on/below.

    This prevents position ``i`` from attending to positions ``> i``.
    Strictly upper-triangular (``diagonal=1``) so positions CAN attend to themselves.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TransformerLM(nn.Module):
    """Decoder-only causal Transformer LM (GPT-2 style).

    Uses ``nn.TransformerEncoderLayer`` with a causal mask (mechanically
    equivalent to a decoder-only block — we have no encoder to cross-attend to).
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        dim_ff: int = DIM_FF,
        n_blocks: int = N_BLOCKS,
        max_seq_len: int = SEQ_LEN,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        # Fixed (non-trainable) sinusoidal positional encoding as a buffer.
        self.register_buffer(
            "pos_enc",
            sinusoidal_positional_encoding(max_seq_len, d_model),
            persistent=False,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        ``x``: ``[batch, seq_len]`` int64 token IDs.
        Returns ``[batch, seq_len, vocab_size]`` logits — one distribution per
        position (unlike the feedforward model, which produces one logit vector
        per sequence).
        """
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}"
            )
        # token_emb: [batch, seq_len, d_model] + pos_enc: [seq_len, d_model]
        emb = self.token_emb(x) + self.pos_enc[:seq_len]
        mask = causal_mask(seq_len, x.device)
        # Pass both the explicit mask AND is_causal=True:
        # the flag is an optimization hint; the mask is the correctness guarantee.
        h = self.blocks(emb, mask=mask, is_causal=True)
        h = self.ln_f(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class TransformerTrainer:
    """Manages training, checkpointing, validation, and generation for TransformerLM.

    Mirrors ``FeedforwardTrainer`` API so downstream tracks (C, D1) can treat
    both models uniformly.

    Differences vs. feedforward:
    - Checkpoint files: ``model/checkpoints/transformer_{chunk_id}.pt``
    - Generation uses last-position logits only (``logits[:, -1, :]``)
    - Shift-by-one next-token loss on every position (not single-target)
    - VRAM guard halves ``seq_len`` and batch size if total VRAM < 3 GB
    - Adam lr = 3e-4 (transformers prefer slightly lower than the FF 1e-3)
    """

    def __init__(
        self,
        store: Store,
        device: torch.device = DEVICE,
        checkpoint_dir: Path = _DEFAULT_CHECKPOINT_DIR,
    ) -> None:
        self.store = store
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # VRAM guard — resolved at instance creation (not import time).
        if device.type == "cuda":
            total_vram = torch.cuda.get_device_properties(0).total_memory
            if total_vram < 3 * 1024**3:
                self._batch_size: int = 32
                self._seq_len: int = 32
            else:
                self._batch_size = 64
                self._seq_len = SEQ_LEN
        else:
            self._batch_size = 16   # smaller batches are faster on CPU
            self._seq_len = 32      # shorter context window speeds up CPU training

        self.model = TransformerLM(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            dim_ff=DIM_FF,
            n_blocks=N_BLOCKS,
            max_seq_len=self._seq_len,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self._maybe_load_latest_checkpoint()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, token_ids: list[int], chunk_id: int) -> float:
        """One forward+backward pass over the chunk.

        Returns mean training loss. Saves checkpoint after training.
        """
        split = max(1, int(len(token_ids) * 0.8))
        train_tokens = token_ids[:split]
        val_tokens = token_ids[split:]

        mean_train_loss = self._run_epoch(train_tokens, training=True)
        val_loss = self.validate(val_tokens)
        self.save_checkpoint(chunk_id, val_loss)
        return mean_train_loss

    def validate(self, token_ids: list[int]) -> float:
        """Compute mean cross-entropy loss on held-out tokens. No gradient."""
        return self._run_epoch(token_ids, training=False)

    def _run_epoch(self, token_ids: list[int], *, training: bool) -> float:
        """Slide a window over ``token_ids``, run one epoch, return mean CE loss.

        For each window of length ``seq_len + 1`` we train the model to predict
        the next token at EVERY position (shift-by-one). Input is the first
        ``seq_len`` tokens; target is the last ``seq_len`` tokens.
        """
        seq_len = self._seq_len
        # Need at least seq_len + 1 tokens to form one (input, target) pair.
        if len(token_ids) <= seq_len:
            return float("inf")

        xs: list[list[int]] = []
        ys: list[list[int]] = []
        # Stride == seq_len so windows don't overlap — one pass over the corpus.
        for i in range(0, len(token_ids) - seq_len, seq_len):
            xs.append(token_ids[i : i + seq_len])
            ys.append(token_ids[i + 1 : i + 1 + seq_len])

        if not xs:
            return float("inf")

        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        n_batches = 0
        batch_size = self._batch_size

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:  # type: ignore[attr-defined]
            for start in range(0, len(xs), batch_size):
                x_batch = xs[start : start + batch_size]
                y_batch = ys[start : start + batch_size]

                x_t = torch.tensor(x_batch, dtype=torch.long, device=self.device)
                y_t = torch.tensor(y_batch, dtype=torch.long, device=self.device)
                logits = self.model(x_t)  # [batch, seq_len, vocab_size]

                # Cross-entropy over flattened (batch * seq_len) positions.
                # ignore_index=PAD_ID so padding tokens don't contribute to loss.
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y_t.reshape(-1),
                    ignore_index=PAD_ID,
                )

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches if n_batches > 0 else float("inf")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        n_tokens: int,
        seed_context: list[int] | None = None,
        temperature: float = 1.0,
        rng: torch.Generator | None = None,
    ) -> list[int]:
        """Autoregressive generation. Returns list of generated token IDs.

        Pads ``seed_context`` with ``PAD_ID`` on the left if shorter than
        ``seq_len``. Uses last-position logits only. Stops early if
        ``ENDOFTEXT_ID`` is sampled.
        """
        seq_len = self._seq_len

        # Build initial context window of exactly seq_len tokens.
        if seed_context is None or len(seed_context) == 0:
            context: list[int] = [PAD_ID] * seq_len
        elif len(seed_context) >= seq_len:
            context = list(seed_context[-seq_len:])
        else:
            pad_len = seq_len - len(seed_context)
            context = [PAD_ID] * pad_len + list(seed_context)

        generated: list[int] = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(n_tokens):
                x = torch.tensor(
                    [context[-seq_len:]], dtype=torch.long, device=self.device
                )
                logits = self.model(x)  # [1, seq_len, vocab_size]
                # Last-position logits only — the causal mask ensures position
                # seq_len-1 has seen the full context.
                last_logits = logits[0, -1, :]  # [vocab_size]
                last_logits = last_logits / max(temperature, 1e-8)
                probs = torch.softmax(last_logits, dim=-1)
                next_id = int(
                    torch.multinomial(probs, num_samples=1, generator=rng).item()
                )
                generated.append(next_id)
                if next_id == ENDOFTEXT_ID:
                    break
                context.append(next_id)

        return generated

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, chunk_id: int, val_loss: float) -> Path:
        """Save state_dict to ``model/checkpoints/transformer_{chunk_id}.pt`` and record in DB."""
        path = self.checkpoint_dir / f"transformer_{chunk_id}.pt"
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "chunk_id": chunk_id,
                "val_loss": val_loss,
                "vocab_size": VOCAB_SIZE,
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_blocks": N_BLOCKS,
                "seq_len": self._seq_len,
            },
            path,
        )
        self.store.insert_checkpoint(
            model_name="transformer",
            chunk_id=chunk_id,
            filepath=path,
            val_loss=val_loss,
        )
        return path

    def load_checkpoint(self, path: Path) -> None:
        """Load model and optimizer state from a .pt checkpoint file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])

    def _maybe_load_latest_checkpoint(self) -> None:
        """On ``__init__``, query store for the latest transformer checkpoint and load if found."""
        row = self.store.get_latest_checkpoint("transformer")
        if row is not None:
            ckpt_path = Path(row["filepath"])
            if ckpt_path.exists():
                self.load_checkpoint(ckpt_path)
