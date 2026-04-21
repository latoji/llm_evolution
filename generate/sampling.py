"""
Track 5 – Sampling Controls
top-k and top-p (nucleus) sampling filters, and a combined sampler that
applies temperature → top-k → top-p → multinomial sample.
"""
from __future__ import annotations

import math
import random


# ---------------------------------------------------------------------------
# Filter functions (operate on probability dicts)
# ---------------------------------------------------------------------------

def top_k_filter(
    distribution: dict[int, float],
    k: int,
) -> dict[int, float]:
    """
    Keep only the top-k most probable tokens and renormalise.

    Args:
        distribution: token_id → probability (must sum to ~1.0).
        k: Number of tokens to keep. k <= 0 means no filtering.

    Returns:
        Renormalised distribution over at most k tokens.
    """
    if k <= 0 or k >= len(distribution):
        return distribution

    top = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:k]
    total = sum(p for _, p in top)
    if total <= 0:
        return dict(top)
    return {t: p / total for t, p in top}


def top_p_filter(
    distribution: dict[int, float],
    p: float,
) -> dict[int, float]:
    """
    Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability mass ≥ p, then renormalise.

    Args:
        distribution: token_id → probability.
        p: Cumulative probability threshold in (0, 1].

    Returns:
        Renormalised nucleus distribution.
    """
    if p >= 1.0:
        return distribution

    sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    cumulative = 0.0
    nucleus: list[tuple[int, float]] = []

    for token, prob in sorted_items:
        nucleus.append((token, prob))
        cumulative += prob
        if cumulative >= p:
            break

    total = sum(prob for _, prob in nucleus)
    if total <= 0:
        return dict(nucleus)
    return {t: prob / total for t, prob in nucleus}


def apply_temperature(
    distribution: dict[int, float],
    temperature: float,
) -> dict[int, float]:
    """
    Sharpen or flatten a distribution by temperature scaling.
    Works in log-space to avoid underflow.

    temperature < 1 → sharper (more deterministic)
    temperature > 1 → flatter (more random)
    temperature = 1 → unchanged
    """
    if abs(temperature - 1.0) < 1e-9:
        return distribution
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    # Apply temperature in log-space then re-exponentiate
    log_probs = {t: math.log(max(p, 1e-300)) / temperature for t, p in distribution.items()}
    max_log = max(log_probs.values())
    exp_probs = {t: math.exp(lp - max_log) for t, lp in log_probs.items()}
    total = sum(exp_probs.values())
    return {t: p / total for t, p in exp_probs.items()}


# ---------------------------------------------------------------------------
# Combined sampler
# ---------------------------------------------------------------------------

def combined_sample(
    distribution: dict[int, float],
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    rng: random.Random | None = None,
) -> int:
    """
    Apply temperature → top-k → top-p, then sample one token ID.

    Args:
        distribution: Full probability distribution over vocab.
        temperature:  Scaling factor (> 0). Default 1.0 (no change).
        top_k:        Keep top-k tokens. 0 = disabled.
        top_p:        Nucleus threshold. 1.0 = disabled.
        rng:          Optional seeded Random for reproducibility.

    Returns:
        A single sampled token ID.
    """
    if not distribution:
        return 0

    dist = apply_temperature(distribution, temperature)
    dist = top_k_filter(dist, top_k)
    dist = top_p_filter(dist, top_p)

    tokens = list(dist.keys())
    weights = list(dist.values())

    if rng is not None:
        return rng.choices(tokens, weights=weights, k=1)[0]
    return random.choices(tokens, weights=weights, k=1)[0]
