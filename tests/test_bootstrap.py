"""Tests for the stationary (Politis-Romano) bootstrap utility."""

from __future__ import annotations

import numpy as np
import pytest

from market_prediction_study.bootstrap import (
    stationary_bootstrap_auc_ci,
    stationary_bootstrap_indices,
)


def test_stationary_bootstrap_indices_respect_block_structure():
    """Blocks must be contiguous runs (mod n) with mean length ~ l_bar.

    With expected block length l_bar the per-step restart probability is
    1/l_bar, so the fraction of "continue" steps must be ~ 1 - 1/l_bar and
    the mean run length ~ l_bar (deterministic for the fixed seed below).
    """
    n, l_bar = 5000, 10
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(n, l_bar, rng)

    assert idx.shape == (n,)
    assert idx.min() >= 0 and idx.max() < n

    cont = idx[1:] == (idx[:-1] + 1) % n  # continue = previous index + 1 (circular)
    frac_cont = float(cont.mean())
    assert abs(frac_cont - (1.0 - 1.0 / l_bar)) < 0.02

    starts = np.concatenate([[0], np.where(~cont)[0] + 1])
    runs = np.diff(np.concatenate([starts, [n]]))
    assert abs(float(runs.mean()) - l_bar) < 1.5


def test_stationary_bootstrap_indices_input_validation():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(0, 5, rng)
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(100, 0, rng)


def test_auc_bootstrap_reproducible_ordered_and_signal_aware():
    """CI ordering, seed reproducibility, and behaviour on signal vs noise."""
    rng = np.random.default_rng(3)
    n = 500
    # (a) informative scores -> the CI must sit clearly above chance
    latent = rng.normal(size=n)
    y_signal = (latent > 0).astype(int)
    s_signal = latent + rng.normal(scale=0.8, size=n)
    r1 = stationary_bootstrap_auc_ci(
        y_signal, s_signal, B=300, expected_block_len=5, seed=7
    )
    r2 = stationary_bootstrap_auc_ci(
        y_signal, s_signal, B=300, expected_block_len=5, seed=7
    )
    assert r1 == r2  # fully reproducible for a fixed seed
    assert 0.0 <= r1["ci_low"] <= r1["auc_point"] <= r1["ci_high"] <= 1.0
    assert r1["ci_low"] > 0.5  # real signal: interval excludes chance
    assert r1["B"] == 300 and r1["block_len"] == 5 and r1["seed"] == 7
    assert r1["n"] == n and r1["n_degenerate"] == 0

    # (b) pure noise scores -> the CI must include 0.5 (honest null)
    y_noise = (rng.random(n) < 0.5).astype(int)
    s_noise = rng.normal(size=n)
    r3 = stationary_bootstrap_auc_ci(y_noise, s_noise, B=300, expected_block_len=5, seed=11)
    assert r3["ci_low"] <= 0.5 <= r3["ci_high"]

    # (c) perfect separation -> every resample has AUC 1.0, degenerate CI
    y_sep = np.array([0] * 100 + [1] * 100)
    s_sep = np.linspace(0.0, 1.0, 200)
    r4 = stationary_bootstrap_auc_ci(y_sep, s_sep, B=100, expected_block_len=21, seed=1)
    assert r4["auc_point"] == pytest.approx(1.0)
    assert r4["ci_low"] == pytest.approx(1.0) and r4["ci_high"] == pytest.approx(1.0)


def test_auc_bootstrap_nan_pairs_dropped_and_validation():
    y = np.array([0, 1, 0, 1, np.nan, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.5, np.nan])
    res = stationary_bootstrap_auc_ci(y, s, B=50, expected_block_len=2, seed=0)
    assert res["n"] == 4  # two NaN pairs dropped
    with pytest.raises(ValueError):
        stationary_bootstrap_auc_ci([0, 0, 0], [0.1, 0.2, 0.3], B=10, seed=0)
    with pytest.raises(ValueError):
        stationary_bootstrap_auc_ci([0, 1], [0.1, 0.4], B=0, seed=0)
