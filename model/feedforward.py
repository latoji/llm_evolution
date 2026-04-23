"""Feedforward neural network language model (Model 12) — BPE token-based, ~5.2M params."""

from pathlib import Path

import torch
import torch.nn as nn

from db.store import Store

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

VOCAB_SIZE: int = 8000
D_MODEL: int = 128
SEQ_LEN: int = 64

# Default device used as a parameter default only — tests override with cpu.
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Special token IDs — must match tokenizer/bpe.py
ENDOFTEXT_ID: int = 0
PAD_ID: int = 1

_DEFAULT_CHECKPOINT_DIR: Path = Path("model/checkpoints")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class FeedforwardLM(nn.Module):
    """Feedforward LM: Embedding → Flatten → 3× (Linear+ReLU) → logits."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        seq_len: int = SEQ_LEN,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len] int64 token IDs. Returns [batch, vocab_size] logits."""
        emb = self.embedding(x)  # [batch, seq_len, d_model]
        return self.net(emb)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class FeedforwardTrainer:
    """Manages training, checkpointing, validation, and generation for FeedforwardLM."""

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

        # VRAM guard — resolved at instance creation; device is pluggable for tests.
        if device.type == "cuda":
            total_vram = torch.cuda.get_device_properties(0).total_memory
            if total_vram < 3 * 1024**3:
                self._batch_size: int = 32
                self._seq_len: int = 32
            else:
                self._batch_size = 64
                self._seq_len = 64
        else:
            self._batch_size = 16   # smaller batches are faster on CPU
            self._seq_len = 32      # shorter context window speeds up CPU training

        self.model = FeedforwardLM(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, seq_len=self._seq_len
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
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
        """Slide a context window over token_ids, run one epoch, return mean CE loss."""
        seq_len = self._seq_len
        if len(token_ids) <= seq_len:
            return float("inf")

        # Build sliding-window (x, y) pairs: x = context, y = next token
        xs: list[list[int]] = []
        ys: list[int] = []
        for i in range(len(token_ids) - seq_len):
            xs.append(token_ids[i : i + seq_len])
            ys.append(token_ids[i + seq_len])

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
                logits = self.model(x_t)  # [batch, vocab_size]
                loss = self.criterion(logits, y_t)

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

        Pads seed_context with PAD_ID on the left if shorter than seq_len.
        Stops early if ENDOFTEXT_ID is sampled.
        """
        seq_len = self._seq_len

        # Build initial context window of exactly seq_len tokens
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
                x = torch.tensor([context[-seq_len:]], dtype=torch.long, device=self.device)
                logits = self.model(x)[0]  # [vocab_size]
                logits = logits / max(temperature, 1e-8)
                probs = torch.softmax(logits, dim=-1)
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
        """Save state_dict to model/checkpoints/feedforward_{chunk_id}.pt and record in DB."""
        path = self.checkpoint_dir / f"feedforward_{chunk_id}.pt"
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "chunk_id": chunk_id,
                "val_loss": val_loss,
                "vocab_size": VOCAB_SIZE,
            },
            path,
        )
        self.store.insert_checkpoint(
            model_name="feedforward",
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
        """On __init__, query store for the latest feedforward checkpoint and load if found."""
        row = self.store.get_latest_checkpoint("feedforward")
        if row is not None:
            ckpt_path = Path(row["filepath"])
            if ckpt_path.exists():
                self.load_checkpoint(ckpt_path)
