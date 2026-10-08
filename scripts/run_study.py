#!/usr/bin/env python
"""Run the full study pipeline end-to-end.

Usage:
    python scripts/run_study.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_prediction_study.config import DEFAULT_CONFIG_PATH, StudyConfig
from market_prediction_study.pipeline import run_study


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config = StudyConfig.from_yaml(args.config)
    results = run_study(config)

    print()
    print("=" * 72)
    print(f"Study complete: {results['n_rows']} usable rows "
          f"({results['sample_start']} -> {results['sample_end']}), "
          f"CV={results['cv_rows']} holdout={results['holdout_rows']}")
    print("=" * 72)
    print(results["summary_md"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
