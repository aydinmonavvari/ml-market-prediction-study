"""Leakage-free feature engineering -- the shift-discipline module.

Shift discipline
================
Row ``t`` of the design matrix must contain **only** information available at
the end of trading day ``t``; the label of row ``t`` is the market direction on
day ``t+1``.  In this module future information is touched in exactly one
place: the single ``shift(-1)`` that creates the target columns from the daily
log return.  ``tests/test_features.py`` enforces this functionally: it modifies
only rows strictly after day ``t`` and asserts that no feature value at rows
``<= t`` moves, while the target does move by construction.

Definitions (all computed on daily data)
========================================
log return:
    ``r_t = ln(P_t / P_{t-1})`` -- known at the end of day ``t``.
rolling volatility:
    ``vol_k(t) = std(r_{t-k+1..t})`` (sample std, ddof=1).
SMA ratio:
    ``sma_ratio(t) = SMA_5(P)_t / SMA_20(P)_t - 1``.
RSI (Wilder 1978):
    gains ``g_t = max(r_t, 0)``, losses ``l_t = max(-r_t, 0)``;
    smoothed averages use Wilder recursion with seed = SMA of the first
    ``period`` observations, then ``avg_t = (avg_{t-1}*(period-1) + x_t)/period``;
    ``RS = avg_gain / avg_loss`` and ``RSI = 100 - 100/(1+RS)``
    (RSI = 100 if ``avg_loss == 0`` and gains exist; 50 if both are 0).
MACD (Appel):
    ``MACD = EMA_fast(P) - EMA_slow(P)``; signal = ``EMA_signal(MACD)``;
    histogram = ``MACD - signal``; EMAs use ``alpha = 2/(span+1)``, seeded with
    the first value (pandas ``ewm(adjust=False)`` convention).
volume z-score:
    ``(V_t - mean(V_{t-20..t})) / std(V_{t-20..t})``.
day-of-week:
    four dummies (Tue..Fri; Monday is the reference class).
macro covariate:
    daily change of the FRED effective fed funds rate (level forward-filled
    onto the exchange calendar; carry days contribute 0).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

RET_LAGS = (1, 2, 3, 5, 10, 21)
VOL_WINDOWS = (5, 21)
SMA_FAST, SMA_SLOW = 5, 20
VOLUME_WINDOW = 21

FEATURE_COLUMNS = [
    *(f"log_ret_lag_{lag}" for lag in RET_LAGS),
    *(f"vol_{w}d" for w in VOL_WINDOWS),
    "sma_5_20_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "volume_z_21",
    *(f"dow_{d}" for d in (1, 2, 3, 4)),
    "dff_change",
]
TARGET_RET = "target_next_log_ret"
TARGET_UP = "target_up"


def log_returns(close: pd.Series) -> pd.Series:
    """Daily log returns ``ln(P_t / P_{t-1})`` (first value is NaN)."""
    return np.log(close / close.shift(1))


def _wilder_smooth(values: pd.Series, period: int) -> pd.Series:
    """Wilder (1978) smoothing.

    Seed: SMA of the first ``period`` consecutive valid observations.
    Recursion: ``avg_t = (avg_{t-1} * (period - 1) + x_t) / period``.
    Observations before the seed (or after a NaN hole) are NaN.
    """
    x = values.to_numpy(dtype=float)
    out = np.full_like(x, np.nan)
    valid = ~np.isnan(x)
    start = None
    run = 0
    for i, ok in enumerate(valid):
        run = run + 1 if ok else 0
        if run == period:
            start = i
            break
    if start is None:
        return pd.Series(out, index=values.index)
    out[start] = float(np.nanmean(x[start - period + 1 : start + 1]))
    for i in range(start + 1, len(x)):
        if np.isnan(x[i]):
            out[i] = np.nan
        else:
            out[i] = (out[i - 1] * (period - 1) + x[i]) / period
    return pd.Series(out, index=values.index)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder smoothing (see module docstring)."""
    diff = close.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - 100.0 / (1.0 + rs)
    warmup = avg_gain.isna() | avg_loss.isna()
    out = out.where(avg_loss.fillna(np.nan).notna() & (avg_loss != 0), 100.0)
    flat = (avg_gain == 0) & (avg_loss == 0)
    out = out.where(~flat, 50.0)  # no movement -> neutral
    out[warmup | diff.isna()] = np.nan  # smoothing warm-up stays undefined
    return out


def _ema(series: pd.Series, span: int) -> pd.Series:
    """EMA with ``alpha = 2/(span+1)`` seeded at the first observation."""
    return series.ewm(span=span, adjust=False).mean()


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line and histogram (columns ``macd, macd_signal, macd_hist``)."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    line = ema_fast - ema_slow
    sig = _ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def rolling_volatility(log_ret: pd.Series, window: int) -> pd.Series:
    """Rolling sample std (ddof=1) of log returns over ``window`` days."""
    return log_ret.rolling(window, min_periods=window).std(ddof=1)


def sma_ratio(close: pd.Series, fast: int = SMA_FAST, slow: int = SMA_SLOW) -> pd.Series:
    """``SMA_fast / SMA_slow - 1``; positive = short-term trend above long-term."""
    return close.rolling(fast, min_periods=fast).mean() / close.rolling(
        slow, min_periods=slow
    ).mean() - 1.0


def volume_zscore(volume: pd.Series, window: int = VOLUME_WINDOW) -> pd.Series:
    """Z-score of volume against its trailing ``window``-day mean/std."""
    mean = volume.rolling(window, min_periods=window).mean()
    std = volume.rolling(window, min_periods=window).std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (volume - mean) / std


def day_of_week_dummies(index: pd.DatetimeIndex) -> pd.DataFrame:
    """One-hot day-of-week dummies for Tue..Fri (Monday = reference class)."""
    dow = index.dayofweek
    return pd.DataFrame(
        {f"dow_{d}": (dow == d).astype(float) for d in (1, 2, 3, 4)},
        index=index,
    )


def build_features(df: pd.DataFrame, rsi_period: int = 14) -> pd.DataFrame:
    """Build the full feature frame **plus** the two target columns.

    Requires columns ``close``, ``volume``, ``dff_change`` indexed by trading
    day.  The ONLY forward-looking operation is the single ``shift(-1)`` below
    that turns the day-``t``->``t+1`` log return into the targets of row ``t``.

    Returns a DataFrame with ``FEATURE_COLUMNS`` and the targets
    ``target_next_log_ret`` (regression) and ``target_up``
    (1 if the next-day log return is > 0, else 0 -- ties count as down).
    The final row has a NaN target by construction (there is no next day yet).
    """
    close, volume = df["close"], df["volume"]
    ret = log_returns(close)

    out = pd.DataFrame(index=df.index)
    for lag in RET_LAGS:
        out[f"log_ret_lag_{lag}"] = ret.shift(lag)
    for window in VOL_WINDOWS:
        out[f"vol_{window}d"] = rolling_volatility(ret, window)
    out["sma_5_20_ratio"] = sma_ratio(close)
    out["rsi_14"] = rsi(close, period=rsi_period)
    out = out.join(macd(close))
    out["volume_z_21"] = volume_zscore(volume)
    out = out.join(day_of_week_dummies(df.index))
    out["dff_change"] = df["dff_change"].astype(float)

    # ---- THE single forward-looking step in the entire codebase ------------
    next_log_ret = ret.shift(-1)  # r_{t+1}: known only after day t+1 closes
    out[TARGET_RET] = next_log_ret
    out[TARGET_UP] = (next_log_ret > 0).astype(float)
    out.loc[next_log_ret.isna(), TARGET_UP] = np.nan
    # ------------------------------------------------------------------------
    return out


class ModelMatrices(NamedTuple):
    """Design matrix and targets aligned on rows where everything is defined."""

    X: pd.DataFrame
    y_up: pd.Series
    y_next_ret: pd.Series


def build_model_matrices(df: pd.DataFrame, rsi_period: int = 14) -> ModelMatrices:
    """``build_features`` + drop warm-up rows and the last (targetless) row."""
    feats = build_features(df, rsi_period=rsi_period)
    usable = feats.dropna(subset=FEATURE_COLUMNS + [TARGET_RET, TARGET_UP])
    return ModelMatrices(
        X=usable[FEATURE_COLUMNS].astype(float),
        y_up=usable[TARGET_UP].astype(int),
        y_next_ret=usable[TARGET_RET].astype(float),
    )
