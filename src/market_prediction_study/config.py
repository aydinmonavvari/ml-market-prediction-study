"""Central configuration for the study.

A single frozen dataclass holds every knob the experiment needs (symbols,
dates, validation geometry, transaction costs, random seed, paths) so that the
pipeline, the tests and the CLIs all share one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FRED_DFF_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"


@dataclass(frozen=True)
class StudyConfig:
    """Configuration for one run of the study.

    Attributes
    ----------
    symbol:
        Ticker used as the equity-index proxy.  SPY (SPDR S&P 500 ETF) is used
        because a long, clean, adjusted daily price history is available from
        Yahoo Finance; the S&P 500 index itself (^GSPC) is price-only.
    start, end:
        Sample window (inclusive).  ``end=None`` means "up to the most recent
        close available".
    holdout_frac:
        Fraction of the most recent observations reserved as the untouched
        final holdout set (evaluated exactly once).
    n_splits, test_days, min_train_days, gap:
        Expanding-window walk-forward geometry.  ``gap`` is the embargo, in
        trading days, inserted between the end of a training block and the
        start of the matching test block.
    cost_bps:
        Round-trip-agnostic transaction cost, in basis points, charged per
        unit of position change in the long/flat backtest (10 bps per switch).
    seed:
        Global random seed; every model pins ``random_state=seed``.
    """

    symbol: str = "SPY"
    start: str = "2010-01-01"
    end: str | None = None

    holdout_frac: float = 0.20

    n_splits: int = 8
    test_days: int = 63
    min_train_days: int = 756
    gap: int = 1

    cost_bps: float = 10.0
    seed: int = 42
    min_rows: int = 1000

    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # directory layout (created lazily by the pipeline)
    repo_root: Path = field(default_factory=lambda: REPO_ROOT)

    @property
    def data_dir(self) -> Path:
        return self.repo_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.repo_root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.repo_root / "figures"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.reports_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> StudyConfig:
        """Build a config from a YAML file (missing keys fall back to defaults)."""
        import yaml  # local import keeps module import cost low

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f for f in cls.__dataclass_fields__ if f != "repo_root"}
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(**filtered)


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
