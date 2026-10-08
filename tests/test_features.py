"""Feature-engineering tests: the leakage test is the critical one.

The leakage test builds a price path, computes features, then modifies ONLY
rows strictly after day ``t`` and verifies that (a) no feature at rows <= t
moves, and (b) the target at row t DOES move -- documenting the single
intended use of future information (the target's shift(-1)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from market_prediction_study.features import (
    FEATURE_COLUMNS,
    TARGET_RET,
    TARGET_UP,
    build_features,
    build_model_matrices,
    log_returns,
    macd,
    rolling_volatility,
    rsi,
    sma_ratio,
    volume_zscore,
)


def test_no_feature_leakage_from_future_rows():
    """THE leakage test: features at row t must not depend on rows > t."""
    df = make_ohlcv(n=260, seed=3)
    t = 150  # day under inspection

    base = build_features(df)
    perturbed_df = df.copy()
    # modify ONLY rows strictly after t (prices, volumes, macro change)
    perturbed_df.iloc[t + 1 :, df.columns.get_loc("close")] *= 1.41
    perturbed_df.iloc[t + 1 :, df.columns.get_loc("volume")] *= 0.37
    perturbed_df.iloc[t + 1 :, df.columns.get_loc("dff_change")] = 0.123
    perturbed = build_features(perturbed_df)

    # (a) every feature column at rows <= t is bit-identical
    pd.testing.assert_frame_equal(
        base[FEATURE_COLUMNS].iloc[: t + 1], perturbed[FEATURE_COLUMNS].iloc[: t + 1]
    )
    # ...and features strictly after t DID change (the perturbation is live)
    assert not np.allclose(
        base[FEATURE_COLUMNS].iloc[t + 2 : t + 12].to_numpy(),
        perturbed[FEATURE_COLUMNS].iloc[t + 2 : t + 12].to_numpy(),
    )

    # (b) the target at row t uses day t+1 -- it MUST move when day t+1 moves
    assert base[TARGET_RET].iloc[t] != pytest.approx(perturbed[TARGET_RET].iloc[t])
    # The direction label at row t is fully determined by the t -> t+1 return.
    # Force the perturbed day t+1 close to the opposite side of close[t] and
    # the label must flip (robust against direction coincidence).
    flip_df = df.copy()
    base_up = bool(base[TARGET_UP].iloc[t] == 1.0)
    flip_df.iloc[t + 1, flip_df.columns.get_loc("close")] = df["close"].iloc[t] * (
        0.95 if base_up else 1.05
    )
    flipped = build_features(flip_df)
    assert flipped[TARGET_UP].iloc[t] == (0.0 if base_up else 1.0)


def test_targets_are_one_day_ahead(ohlcv):
    """target_up at row t equals sign of the day t+1 log return."""
    feats = build_features(ohlcv)
    ret = log_returns(ohlcv["close"])
    manual_up = (ret.shift(-1) > 0).astype(float)
    manual_up[ret.shift(-1).isna()] = np.nan
    pd.testing.assert_series_equal(feats[TARGET_UP], manual_up, check_names=False)
    assert feats[TARGET_UP].iloc[-1] != feats[TARGET_UP].iloc[-1]  # last row: NaN


def test_rsi_matches_hand_computed_wilder():
    """RSI(period=3) hand-computed with Wilder's recursion.

    prices [100,102,101,103,104,102,105]; diffs [2,-1,2,1,-2,3].
    gains [2,0,2,1,0,3]; losses [0,1,0,0,2,0].
    seed (SMA of first 3): avg_gain=4/3, avg_loss=1/3 -> RS=4 -> RSI=80.
    idx4: avg_gain=(4/3*2+1)/3=11/9, avg_loss=(1/3*2+0)/3=2/9 -> RS=5.5 -> 84.615385.
    idx5: avg_gain=22/27, avg_loss=22/27 -> RS=1 -> 50.
    idx6: avg_gain=125/81, avg_loss=44/81 -> RS=125/44 -> 73.964497.
    """
    prices = pd.Series([100.0, 102, 101, 103, 104, 102, 105])
    out = rsi(prices, period=3)
    assert np.isnan(out.iloc[:3]).all()
    expected = [80.0, 84.6153846, 50.0, 73.9644970]
    np.testing.assert_allclose(out.iloc[3:].to_numpy(), expected, atol=1e-6)


def test_rsi_extremes():
    up = pd.Series([100.0, 101, 102, 103, 104, 105])  # never falls
    out = rsi(up, period=3)
    assert (out.dropna() == 100.0).all()
    flat = pd.Series([100.0] * 8)
    out_flat = rsi(flat, period=3)
    assert (out_flat.dropna() == 50.0).all()


def test_macd_matches_hand_computed():
    """MACD(3,6,3) on 1..5; EMA alpha=2/(span+1), seeded at first value.

    ema3: 1, 1.5, 2.25, 3.125, 4.0625.
    ema6: 1, 9/7, 87/49, 827/343, 7565/2401.
    macd = ema3 - ema6; signal = ema3(macd); hist = macd - signal.
    """
    prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = macd(prices, fast=3, slow=6, signal=3)
    np.testing.assert_allclose(
        out["macd"].to_numpy(),
        [0.0, 0.2142857, 0.4744898, 0.7139213, 0.9117295],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out["macd_signal"].to_numpy(),
        [0.0, 0.1071429, 0.2908163, 0.5023688, 0.7070491],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out["macd_hist"].to_numpy(),
        [0.0, 0.1071429, 0.1836735, 0.2115525, 0.2046803],
        atol=1e-6,
    )


def test_rolling_volatility_and_sma_ratio(ohlcv):
    ret = log_returns(ohlcv["close"])
    vol5 = rolling_volatility(ret, 5)
    # hand-check one row with numpy
    t = 60
    manual = np.std(ret.iloc[t - 4 : t + 1], ddof=1)
    assert vol5.iloc[t] == pytest.approx(manual)
    assert vol5.iloc[4] != vol5.iloc[4]  # NaN before window fills

    ratio = sma_ratio(ohlcv["close"])
    manual_ratio = (
        ohlcv["close"].iloc[t - 4 : t + 1].mean() / ohlcv["close"].iloc[t - 19 : t + 1].mean() - 1.0
    )
    assert ratio.iloc[t] == pytest.approx(manual_ratio)


def test_volume_zscore(ohlcv):
    z = volume_zscore(ohlcv["volume"], 21)
    t = 100
    v = ohlcv["volume"]
    manual = (v.iloc[t] - v.iloc[t - 20 : t + 1].mean()) / np.std(
        v.iloc[t - 20 : t + 1], ddof=1
    )
    assert z.iloc[t] == pytest.approx(manual)


def test_build_model_matrices_alignment(ohlcv):
    n = len(ohlcv)
    mats = build_model_matrices(ohlcv)
    assert list(mats.X.columns) == FEATURE_COLUMNS
    # warm-up: lag_21 + vol_21 are the last to fill (row 22 is the first usable)
    assert mats.X.index[0] == ohlcv.index[22]
    # last row's target is undefined (no next day) and must be dropped
    assert mats.X.index[-1] == ohlcv.index[n - 2]
    assert len(mats.X) == n - 23
    assert len(mats.y_up) == len(mats.X) == len(mats.y_next_ret)
    assert set(mats.y_up.unique()) <= {0, 1}
    assert mats.X.notna().all().all()


def test_features_deterministic(ohlcv):
    a = build_model_matrices(ohlcv)
    b = build_model_matrices(ohlcv)
    pd.testing.assert_frame_equal(a.X, b.X)
