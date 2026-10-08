"""Shared fixtures: deterministic synthetic OHLCV data (NO network in tests)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # allow running tests without installing the package
    sys.path.insert(0, str(SRC))


def make_ohlcv(n: int = 400, seed: int = 0, start: str = "2015-01-01") -> pd.DataFrame:
    """Synthetic daily OHLCV + fed-funds change series with realistic magnitudes."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0003, 0.011, n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    spread_up = rng.uniform(0.0005, 0.006, n)
    spread_dn = rng.uniform(0.0005, 0.006, n)
    high = close * (1.0 + spread_up)
    low = close * (1.0 - spread_dn)
    open_ = low + (high - low) * rng.uniform(0.0, 1.0, n)
    volume = rng.lognormal(mean=16.0, sigma=0.35, size=n)
    idx = pd.bdate_range(start, periods=n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df["dff"] = 0.05 + np.cumsum(rng.normal(0.0, 0.0008, n)).clip(-0.03, 0.03)
    df["dff_change"] = df["dff"].diff().fillna(0.0)
    return df


@pytest.fixture(scope="session")
def ohlcv() -> pd.DataFrame:
    """Default synthetic dataset for feature/model tests."""
    return make_ohlcv(n=420, seed=7)


@pytest.fixture
def small_config(tmp_path):
    """A StudyConfig pointed at a temp repo root with tiny walk-forward geometry."""
    from market_prediction_study.config import StudyConfig

    return StudyConfig(
        repo_root=tmp_path,
        start="2015-01-01",
        n_splits=2,
        test_days=12,
        min_train_days=80,
        gap=1,
        holdout_frac=0.2,
        min_rows=100,
    )
