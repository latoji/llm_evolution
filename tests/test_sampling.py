"""
Tests for generate/sampling.py

Acceptance criteria:
  - top_k=1 always returns the highest-probability token
  - top_p=0.01 returns only the top token(s)
  - Filtered distributions still sum to 1.0
  - Combined sampling produces valid tokens
"""
from __future__ import annotations

import random

import pytest

from generate.sampling import (
    apply_temperature,
    combined_sample,
    top_k_filter,
    top_p_filter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIST = {0: 0.5, 1: 0.3, 2: 0.15, 3: 0.05}  # sums to 1.0
UNIFORM = {i: 0.1 for i in range(10)}       # uniform over 10 tokens


# ---------------------------------------------------------------------------
# top_k_filter
# ---------------------------------------------------------------------------

def test_top_k_1_keeps_only_top():
    result = top_k_filter(DIST, k=1)
    assert set(result.keys()) == {0}  # token 0 has highest prob


def test_top_k_2():
    result = top_k_filter(DIST, k=2)
    assert set(result.keys()) == {0, 1}


def test_top_k_renormalises():
    result = top_k_filter(DIST, k=2)
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_top_k_disabled():
    result = top_k_filter(DIST, k=0)
    assert result == DIST


def test_top_k_larger_than_vocab():
    result = top_k_filter(DIST, k=100)
    assert result == DIST


# ---------------------------------------------------------------------------
# top_p_filter
# ---------------------------------------------------------------------------

def test_top_p_very_small_keeps_top():
    result = top_p_filter(DIST, p=0.01)
    # Only the highest probability token should survive
    assert 0 in result


def test_top_p_1_keeps_all():
    result = top_p_filter(DIST, p=1.0)
    assert result == DIST


def test_top_p_renormalises():
    result = top_p_filter(DIST, p=0.5)
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_top_p_0_8():
    result = top_p_filter(DIST, p=0.8)
    # Tokens 0 (0.5) + 1 (0.3) = 0.8 — should include at least these two
    assert 0 in result
    assert 1 in result


# ---------------------------------------------------------------------------
# apply_temperature
# ---------------------------------------------------------------------------

def test_temperature_1_no_change():
    result = apply_temperature(DIST, temperature=1.0)
    for t, p in result.items():
        assert abs(p - DIST[t]) < 1e-9


def test_temperature_sharpens():
    """temperature < 1 should make the max-prob token even more likely."""
    sharpened = apply_temperature(DIST, temperature=0.1)
    assert sharpened[0] > DIST[0]


def test_temperature_flattens():
    """temperature > 1 should bring probabilities closer to uniform."""
    flattened = apply_temperature(DIST, temperature=10.0)
    # The gap between max and min probs should shrink
    orig_gap = max(DIST.values()) - min(DIST.values())
    flat_gap = max(flattened.values()) - min(flattened.values())
    assert flat_gap < orig_gap


def test_temperature_invalid():
    with pytest.raises(ValueError):
        apply_temperature(DIST, temperature=0.0)


# ---------------------------------------------------------------------------
# combined_sample
# ---------------------------------------------------------------------------

def test_combined_sample_returns_valid_token():
    rng = random.Random(0)
    token = combined_sample(DIST, temperature=1.0, top_k=0, top_p=1.0, rng=rng)
    assert token in DIST


def test_combined_sample_top_k_1_deterministic():
    """top_k=1 should always return the highest probability token."""
    for seed in range(10):
        rng = random.Random(seed)
        token = combined_sample(DIST, temperature=1.0, top_k=1, top_p=1.0, rng=rng)
        assert token == 0, f"top_k=1 returned {token} not 0"


def test_combined_sample_all_controls():
    rng = random.Random(42)
    token = combined_sample(DIST, temperature=0.7, top_k=3, top_p=0.9, rng=rng)
    assert token in DIST


def test_combined_sample_empty_dist():
    result = combined_sample({}, temperature=1.0)
    assert result == 0  # fallback
