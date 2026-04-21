# Track B1 — Feedforward Neural Network (Model 12)

**Model**: `claude-sonnet-4-6`
**Dependencies**: Track 0 complete

## Scope

Implement Model 12: a small feedforward neural network language model in PyTorch. Operates on BPE tokens. One training pass per ingested chunk. Checkpoints saved to `.pt` files and registered in the DuckDB `nn_checkpoints` table.

## Upstream dependencies

- `db/store.py` (Track 0) — checkpoint metadata
- `tokenizer/bpe.py` — tokenizer already exists, frozen

## Downstream consumers

- **Track C** (Monte Carlo) calls `generate()` 50× per evaluation cycle
- **Track D1** (Ingest Worker) calls `train_step()` once per chunk

## Files owned

```
model/feedforward.py
tests/test_feedforward.py
model/checkpoints/                  (runtime artifact directory — create, .gitignore)
```

## Files you must NOT modify

- `model/transformer.py` (Track B2's territory)
- Anything else outside `model/feedforward.py` and its test file

---

## Implementation

### `model/feedforward.py` (~200 lines)

**Architecture (spec verbatim):**

```
Input:    context window of 64 BPE token IDs     [batch, 64]
Embedding: 8000 × 128 trainable                   [batch, 64, 128]
Flatten:  [batch, 64 * 128] = [batch, 8192]
Linear 1: 8192 -> 512, ReLU
Linear 2: 512 -> 512, ReLU
Linear 3: 512 -> 128, ReLU
Output:   128 -> 8000, log-softmax

Total params: ~5.2M | VRAM at batch=64: ~300 MB
```

**Device selection (required pattern — used identically by Track B2):**

```python
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VRAM guard
if DEVICE.type == "cuda":
    total_vram = torch.cuda.get_device_properties(0).total_memory
    if total_vram < 3 * 1024**3:
        BATCH_SIZE, SEQ_LEN = 32, 32
    else:
        BATCH_SIZE, SEQ_LEN = 64, 64
else:
    BATCH_SIZE, SEQ_LEN = 64, 64  # CPU can handle; just slower
```

**Public API:**

```python
import torch
import torch.nn as nn
from pathlib import Path
from db.store import Store

VOCAB_SIZE = 8000
D_MODEL = 128
SEQ_LEN = 64

class FeedforwardLM(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = D_MODEL, seq_len: int = SEQ_LEN):
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
        emb = self.embedding(x)
        return self.net(emb)


class FeedforwardTrainer:
    def __init__(self, store: Store, device: torch.device = DEVICE) -> None:
        self.store = store
        self.device = device
        self.model = FeedforwardLM().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self._maybe_load_latest_checkpoint()

    def train_step(self, token_ids: list[int], chunk_id: int) -> float:
        """One forward+backward pass over the chunk.
        Returns mean training loss. Saves checkpoint."""

    def validate(self, token_ids: list[int]) -> float:
        """Compute mean cross-entropy loss on held-out tokens. No gradient."""

    def generate(self, n_tokens: int, seed_context: list[int] | None = None,
                 temperature: float = 1.0, rng: torch.Generator | None = None) -> list[int]:
        """Autoregressive generation. Returns list of token IDs."""

    def save_checkpoint(self, chunk_id: int, val_loss: float) -> Path:
        """Save state_dict to model/checkpoints/feedforward_{chunk_id}.pt and record in DB."""

    def load_checkpoint(self, path: Path) -> None: ...

    def _maybe_load_latest_checkpoint(self) -> None:
        """On __init__, query store.get_latest_checkpoint('feedforward') and load if found."""
```

**Training step details:**

1. Split `token_ids` into 80/20 train/val (see SPEC assumption)
2. Build sliding-window batches of `SEQ_LEN` tokens each, targeting the next token
3. One pass (one epoch) over training split, batch size `BATCH_SIZE`
4. Compute validation loss on held-out split
5. Save checkpoint with `chunk_id` and `val_loss`

**Generation details:**

- If `seed_context` is None or shorter than `SEQ_LEN`, left-pad with `<|pad|>` (token ID 1)
- Sample via temperature + softmax + multinomial; no top-k or top-p (keep simple for this model)
- Stop if `<|endoftext|>` (token ID 0) is generated

**Checkpoint file format:**

```python
torch.save({
    "model_state": self.model.state_dict(),
    "optimizer_state": self.optimizer.state_dict(),
    "chunk_id": chunk_id,
    "val_loss": val_loss,
    "vocab_size": VOCAB_SIZE,
}, path)
```

---

## Testing

### `tests/test_feedforward.py`

- Model instantiates and forward pass produces `[batch, 8000]` logits
- Training step reduces loss over 5 consecutive passes on a fixed small corpus
- `generate(n_tokens=20)` returns 20 ints, all in range [0, VOCAB_SIZE)
- Checkpoint save/load: round-trip preserves model weights (verify by comparing output tensors before/after)
- `_maybe_load_latest_checkpoint` loads when one exists, no-ops when it doesn't
- Tests run on CPU (use `device="cpu"` explicitly to keep CI portable)
- One optional GPU-gated test via `pytest.mark.skipif(not torch.cuda.is_available(), ...)` that verifies CUDA path works

---

## Acceptance criteria

- [ ] `pytest tests/test_feedforward.py` — all green on CPU
- [ ] Training on a 50 KB corpus completes in under 60 seconds on CPU, under 10 seconds on GPU
- [ ] Checkpoint file < 25 MB
- [ ] `model/checkpoints/` directory is created and has a `.gitignore` with `*.pt`
- [ ] No `torch.cuda` calls without the `DEVICE.type == "cuda"` guard

---

## Pitfalls

- **Do not train for multiple epochs per chunk.** SPEC decision: one pass per chunk, same as n-gram counting cadence.
- **Do not use dropout** — model is tiny, dataset is tiny, dropout hurts here.
- **`<|endoftext|>` is token ID 0, `<|pad|>` is token ID 1.** Do not hardcode different IDs. Read from the BPE tokenizer if unsure.
- **The VRAM guard must be in the class, not a module-level constant** — because the device must be pluggable for tests.
- **Do not use `torch.compile`** — it adds 30+ seconds of first-call latency, not worth it for this model size.
- **Set `torch.manual_seed(42)` in tests** for deterministic assertions on loss reduction.

---

## Model assignment

**Sonnet 4.6.** Standard PyTorch; well-specified architecture; no subtle correctness concerns.
