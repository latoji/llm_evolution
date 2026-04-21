# Track B2 — Transformer Neural Network (Model 13)

**Model**: `claude-opus-4-7` — this track requires careful reasoning about causal masking, positional encoding, and VRAM.
**Dependencies**: Track 0 complete

## Scope

Implement Model 13: a small decoder-only causal Transformer language model, architecturally in the GPT-2 family. Same training/checkpoint lifecycle as the feedforward model. Must fit inside 4 GB VRAM.

## Upstream dependencies

- `db/store.py` (Track 0)
- `tokenizer/bpe.py` — frozen

## Downstream consumers

- **Track C** — generation called 50× per evaluation cycle
- **Track D1** — `train_step` called once per chunk

## Files owned

```
model/transformer.py
tests/test_transformer.py
```

---

## Implementation

### `model/transformer.py` (~300 lines)

**Architecture (spec verbatim):**

```
Input:    [batch, 64] int64 token IDs
Token embedding:   8000 × 128
Positional encoding: sinusoidal, fixed, [64, 128]
Add: token_emb + pos_enc
↓
Block 1: nn.TransformerEncoderLayer(
    d_model=128, nhead=4, dim_feedforward=512,
    activation='gelu', batch_first=True, norm_first=True
) with causal mask
↓
Block 2: same
↓
LayerNorm (final)
↓
Linear: 128 → 8000 (optionally weight-tied to token embedding)

Total: ~6.8M params | peak VRAM at batch=64, seq=64: ~450 MB
```

**Why `norm_first=True` (pre-norm):** matches GPT-2's architecture (section 2 of SPEC.md). Post-norm is unstable for small transformers.

**Why `nn.TransformerEncoderLayer` + causal mask rather than `TransformerDecoderLayer`:** We do not have a separate encoder sequence to cross-attend to. A decoder-only Transformer is mechanically equivalent to an encoder layer with a causal mask. This is the standard GPT-family pattern.

**Sinusoidal positional encoding:**

```python
def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Returns [seq_len, d_model] fixed (non-trainable) encoding.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
```

Register as a buffer (`self.register_buffer`), not a parameter.

**Causal mask:**

```python
def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Returns [seq_len, seq_len] upper-triangular mask of -inf above diagonal, 0 on/below.
    This prevents position i from attending to positions > i."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask
```

Pass as `src_mask` to `TransformerEncoderLayer`.

**Public API (same shape as FeedforwardTrainer for consistency):**

```python
import torch
import torch.nn as nn
from pathlib import Path
from db.store import Store

VOCAB_SIZE = 8000
D_MODEL = 128
N_HEADS = 4
DIM_FF = 512
N_BLOCKS = 2

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS, dim_ff: int = DIM_FF, n_blocks: int = N_BLOCKS,
                 max_seq_len: int = 64):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_enc", sinusoidal_positional_encoding(max_seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Optional weight tying:
        # self.head.weight = self.token_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len]. Returns [batch, seq_len, vocab_size] logits (one per position)."""
        seq_len = x.size(1)
        emb = self.token_emb(x) + self.pos_enc[:seq_len]
        mask = causal_mask(seq_len, x.device)
        h = self.blocks(emb, mask=mask, is_causal=True)
        h = self.ln_f(h)
        return self.head(h)


class TransformerTrainer:
    """Mirrors FeedforwardTrainer API exactly. See specs/02-feedforward.md for the full interface.

    Differences:
    - Checkpoint files: model/checkpoints/transformer_{chunk_id}.pt
    - Generation uses last-position logits only
    - VRAM guard halves max_seq_len if total_vram < 3 GB
    """
```

**Training step:**

1. Split token stream 80/20 train/val
2. Build sliding-window batches: input `x[:-1]`, target `x[1:]` (shift-by-one next-token prediction on every position — unlike feedforward which predicts one target)
3. Loss: `F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))`, ignore index = pad token if padding is used
4. One pass, batch size `BATCH_SIZE`, Adam lr=3e-4 (transformers prefer slightly lower)
5. Save checkpoint

**Generation:**

- Take last `SEQ_LEN - 1` tokens of context; left-pad to SEQ_LEN with pad token if shorter
- Forward pass → take logits at last position: `logits[:, -1, :]`
- Apply temperature (default 1.0), softmax, multinomial sample
- Append sampled token to context, slide window, repeat
- Stop at `<|endoftext|>` (ID 0) or `n_tokens` limit

**Checkpoint format:** identical structure to feedforward (see spec 02). Use separate filename prefix `transformer_*.pt`.

---

## Testing

### `tests/test_transformer.py`

Must cover:
- Instantiation on CPU with default config succeeds
- Forward pass shape: input `[2, 64]` → output `[2, 64, 8000]`
- Causal mask verification: modifying input token at position `i` must not change output logits at positions `< i` (this is the defining property of causal attention — if this fails, the mask is wrong)
- Training loss decreases over 5 passes on a fixed small corpus
- Generation produces tokens in vocabulary range
- Checkpoint save/load round-trips model weights exactly
- Sinusoidal pos_enc values: PE(0, 0) = 0, PE(0, 1) = 1 (verify the formula)
- GPU test (skip if CUDA unavailable): VRAM usage stays under 1 GB with batch=64

**The causal mask test is the single most important test in this file.** If it fails, the model is silently broken. Example:

```python
def test_causal_mask_prevents_future_leakage():
    model = TransformerLM().eval()
    x1 = torch.randint(0, 8000, (1, 64))
    x2 = x1.clone()
    x2[0, 32] = (x1[0, 32] + 1) % 8000  # change one future token
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
    # Positions 0..31 should be identical between the two forward passes
    assert torch.allclose(out1[:, :32, :], out2[:, :32, :], atol=1e-5)
```

---

## Acceptance criteria

- [ ] `pytest tests/test_transformer.py` all green on CPU
- [ ] Causal mask test passes (critical)
- [ ] Training on a 50 KB corpus completes in under 3 minutes on CPU, under 30 seconds on GPU
- [ ] Checkpoint file < 30 MB
- [ ] Peak VRAM on GPU run under 1 GB (verified via `torch.cuda.max_memory_allocated()`)

---

## Pitfalls (HIGH ATTENTION — this is why the track is Opus)

- **Causal mask must be strictly upper triangular (diagonal=1 in `torch.triu`)**. If you include the diagonal, positions cannot attend to themselves, which is wrong. If you exclude too much, positions leak future information.
- **Positional encoding added, not concatenated**. The shape math must match `d_model`, not `2 * d_model`.
- **`batch_first=True` everywhere**. Do not mix with the default `batch_first=False` — silent shape bugs result.
- **`norm_first=True`** for pre-norm (GPT-style). Without this, the model diverges on small data.
- **`is_causal=True` argument to `TransformerEncoder.forward`** is a PyTorch optimization hint but does NOT replace passing the `src_mask`. Pass both for safety.
- **Do not apply softmax before cross-entropy loss.** `F.cross_entropy` expects raw logits. Double softmax = silently wrong gradients.
- **Training loss target is shifted by one**, not same-position. If `x = [t0, t1, t2, t3]`, input is `[t0, t1, t2]` and target is `[t1, t2, t3]`. Getting this wrong trains an identity function.
- **The VRAM guard must run inside `__init__`**, not at module import time — device may not be available at import.
- **Do not use weight tying unless you test it.** It works, but the initialization order matters (tie after both are created).
- **Test output is `[batch, seq_len, vocab]`**, not `[batch, vocab]` like the feedforward model. Generation must index `[:, -1, :]`.

---

## Model assignment

**Opus 4.7.** The correctness constraints above are too easy to violate silently. Causal masking, shift-by-one targets, and pre-norm vs. post-norm are the three places where small transformers most often go wrong. Opus is chosen for this file specifically for that reason.
