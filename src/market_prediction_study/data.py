"""Data acquisition, caching and validation.

Sources
-------
1. **SPY daily bars** -- Yahoo Finance via ``yfinance`` (adjusted OHLCV,
   ``auto_adjust=True`` so that ``Close`` is the dividend/split-adjusted close).
   SPY is used as the equity-index proxy: it is the most liquid S&P 500
   instrument and its adjusted series is directly tradable in spirit, unlike
   the price-only ``^GSPC`` index.
2. **FRED DFF** -- the daily effective federal funds rate, fetched from the
   public FRED CSV endpoint (``fredgraph.csv?id=DFF``).  Only its *daily
   change* is used as a macro covariate.

Caching
-------
Every download is cached as CSV under ``data/raw/`` (git-ignored).  Repeated
runs are fully offline.  Downloads retry with exponential backoff.

Gap policy
----------
The NYSE calendar has scheduled closures (weekends, holidays); therefore the
business-day index is *not* gap-free by construction.  The validation policy
here is: (a) the index must be strictly increasing and unique; (b) no calendar
gap between consecutive trading days may exceed 14 days (any larger hole would
indicate a data error rather than a holiday); (c) prices must be strictly
positive with no NaNs.  The federal funds rate is published on federal
business days, so the level is forward-filled onto the exchange calendar
before differencing; carry days therefore contribute a change of exactly 0.
"""

from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .config import FRED_DFF_URL, StudyConfig

USER_AGENT = (
    "ml-market-prediction-study/1.0 (research; "
    "contact: 245710241+aydinmonavvari@users.noreply.github.com)"
)
MAX_CALENDAR_GAP_DAYS = 14


class DataValidationError(ValueError):
    """Raised when downloaded data fail the integrity checks."""


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
    return df


def _validate_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"{symbol}: missing columns {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError(f"{symbol}: index is not a DatetimeIndex")
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise DataValidationError(f"{symbol}: index must be unique and increasing")
    if df[["close", "volume"]].isna().any().any():
        raise DataValidationError(f"{symbol}: NaNs present in close/volume")
    if (df["close"] <= 0).any():
        raise DataValidationError(f"{symbol}: non-positive prices present")
    gaps = df.index.to_series().diff().dt.days.dropna()
    if len(gaps) and gaps.max() > MAX_CALENDAR_GAP_DAYS:
        raise DataValidationError(
            f"{symbol}: calendar gap of {int(gaps.max())} days exceeds "
            f"{MAX_CALENDAR_GAP_DAYS}-day policy"
        )
    return df


def fetch_spy(
    start: str,
    end: str | None = None,
    cache_path=None,
    force: bool = False,
    max_retries: int = 3,
    symbol: str = "SPY",
) -> pd.DataFrame:
    """Download (or load cached) daily adjusted OHLCV bars for ``symbol``.

    Returns a DataFrame indexed by trading day with columns
    ``open, high, low, close, volume`` where ``close`` is the adjusted close.
    """
    import yfinance as yf

    if cache_path is not None and Path(cache_path).exists() and not force:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return _validate_ohlcv(df, symbol)

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            raw = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is None or raw.empty:
                raise DataValidationError(f"{symbol}: empty response from Yahoo Finance")
            df = _flatten_yf_columns(raw)
            df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            df = _validate_ohlcv(df, symbol)
            if cache_path is not None:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_path)
            return df
        except Exception as err:  # noqa: BLE001 - retry any transport-level failure
            last_err = err
            time.sleep(2**attempt)
    raise RuntimeError(f"{symbol}: download failed after {max_retries} attempts") from last_err


def fetch_fred_dff(
    start: str | None = None,
    end: str | None = None,
    cache_path=None,
    force: bool = False,
    max_retries: int = 3,
) -> pd.Series:
    """Download (or load cached) the FRED daily effective fed funds rate (DFF).

    Returns a float Series indexed by observation date.  Missing FRED prints
    (encoded as ".") become NaN and are dropped.
    """
    if cache_path is not None and Path(cache_path).exists() and not force:
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = "dff"
        return s.sort_index()

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(FRED_DFF_URL, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed URL
                payload = resp.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(payload))
            date_col, val_col = df.columns[0], df.columns[1]
            s = (
                pd.Series(
                    pd.to_numeric(df[val_col].replace(".", pd.NA), errors="coerce").to_numpy(),
                    index=pd.to_datetime(df[date_col]),
                    name="dff",
                )
                .dropna()
                .sort_index()
            )
            if s.empty:
                raise DataValidationError("FRED DFF: no usable observations")
            if start is not None:
                s = s[s.index >= pd.Timestamp(start)]
            if end is not None:
                s = s[s.index <= pd.Timestamp(end)]
            if cache_path is not None:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                s.to_frame().to_csv(cache_path)
            return s
        except Exception as err:  # noqa: BLE001 - retry any transport-level failure
            last_err = err
            time.sleep(2**attempt)
    raise RuntimeError(f"FRED DFF: download failed after {max_retries} attempts") from last_err


def load_dataset(config: StudyConfig, force: bool = False) -> pd.DataFrame:
    """Load SPY + FRED DFF, aligned on the SPY trading calendar.

    Returns a DataFrame with columns ``open, high, low, close, volume, dff,
    dff_change`` where ``dff`` is the fed funds level forward-filled onto the
    exchange calendar and ``dff_change`` its day-over-day difference (0 on
    carry days).
    """
    config.ensure_dirs()
    spy = fetch_spy(
        config.start,
        config.end,
        cache_path=config.raw_dir / "spy_daily.csv",
        force=force,
        symbol=config.symbol,
    )
    dff = fetch_fred_dff(
        config.start,
        cache_path=config.raw_dir / "fred_dff.csv",
        force=force,
    )
    dff_aligned = dff.reindex(spy.index).ffill()
    out = spy.copy()
    out["dff"] = dff_aligned
    out["dff_change"] = dff_aligned.diff()
    first_valid = out["dff_change"].first_valid_index()
    if first_valid is not None:
        out.loc[:first_valid, "dff_change"] = out.loc[:first_valid, "dff_change"].fillna(0.0)
    if out[["close", "volume", "dff_change"]].isna().any().any():
        raise DataValidationError("load_dataset: NaNs remain after alignment")
    return out


__all__ = [
    "DataValidationError",
    "fetch_fred_dff",
    "fetch_spy",
    "load_dataset",
]
