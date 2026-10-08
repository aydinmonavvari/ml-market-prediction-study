#!/usr/bin/env python
"""Download and cache the study data (SPY via yfinance, FRED DFF via public CSV).

Usage:
    python scripts/download_data.py [--config configs/default.yaml] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_prediction_study.config import DEFAULT_CONFIG_PATH, StudyConfig
from market_prediction_study.data import fetch_fred_dff, fetch_spy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    config = StudyConfig.from_yaml(args.config)
    config.ensure_dirs()

    spy = fetch_spy(
        config.start,
        config.end,
        cache_path=config.raw_dir / "spy_daily.csv",
        force=args.force,
        symbol=config.symbol,
    )
    dff = fetch_fred_dff(
        config.start,
        cache_path=config.raw_dir / "fred_dff.csv",
        force=args.force,
    )

    print(f"SPY: {len(spy)} rows, {spy.index[0].date()} -> {spy.index[-1].date()}")
    print(f"     cached at {config.raw_dir / 'spy_daily.csv'}")
    print(f"FRED DFF: {len(dff)} rows, {dff.index[0].date()} -> {dff.index[-1].date()}")
    print(f"     cached at {config.raw_dir / 'fred_dff.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
