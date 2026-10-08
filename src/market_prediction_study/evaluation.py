"""Evaluation: metrics, McNemar tests, Holm correction, permutation control,
and a heavily-caveated cost-adjusted long/flat backtest.

Economic backtest -- explicit assumptions block
-----------------------------------------------
1. Signals are binary long/flat: predicted "up" -> hold the asset for the
   *next* day; predicted "down" -> hold cash earning zero.
2. The decision uses only information available at the close of day ``t`` and
   is applied to day ``t+1``'s simple return (the shift is explicit in code).
3. Fills are assumed to occur exactly at the close used for the decision.
   This is optimistic: real fills would occur near the next open/close with
   slippage.  There is no intraday path, no borrow, no shorting.
4. Transaction cost = ``cost_bps`` (in basis points of traded notional),
   charged on the *absolute change* in position between consecutive days
   (a switch costs one lot of ``cost_bps``; the initial entry from flat also
   costs ``cost_bps``).
5. Single asset, single instrument, single historical period; the strategy is
   evaluated once on the holdout.  Results are therefore one draw from one
   non-stationary process -- no statistical significance is claimed for the
   economic figures, and the buy-and-hold comparator carries the same one-off
   entry cost for fairness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from statsmodels.stats.contingency_tables import mcnemar

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Classification metrics
# --------------------------------------------------------------------------- #
def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]
) -> dict[str, float]:
    """Accuracy, balanced accuracy, ROC-AUC and PR-AUC (positive class = 1).

    Rows whose prediction or score is NaN are excluded before computing the
    metrics.  This is required for out-of-fold series that are only defined
    on the walk-forward test blocks (the initial ``min_train_days`` region
    and embargo gaps are legitimately uncovered); reported ``n`` reflects
    the number of rows actually evaluated.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_score_arr = np.asarray(y_score, dtype=float)
    covered = ~(np.isnan(y_pred_arr) | np.isnan(y_score_arr))
    y_true_arr = y_true_arr[covered].astype(int)
    y_pred_arr = y_pred_arr[covered].astype(int)
    y_score_arr = y_score_arr[covered]
    if len(y_true_arr) == 0 or len(np.unique(y_true_arr)) < 2:
        roc = pr = float("nan")
        acc = bal = float("nan")
        base_rate = float("nan")
    else:
        roc = float(roc_auc_score(y_true_arr, y_score_arr))
        pr = float(average_precision_score(y_true_arr, y_score_arr))
        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        bal = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
        base_rate = float(y_true_arr.mean())
    return {
        "accuracy": acc,
        "balanced_accuracy": bal,
        "roc_auc": roc,
        "pr_auc": pr,
        "n": float(len(y_true_arr)),
        "base_rate": base_rate,
    }


# --------------------------------------------------------------------------- #
# McNemar's test vs. the majority baseline
# --------------------------------------------------------------------------- #
def mcnemar_vs_baseline(
    y_true: Sequence[int], model_pred: Sequence[int], baseline_pred: Sequence[int]
) -> dict[str, float]:
    """McNemar (1947) test of paired correctness on the same evaluation rows.

    Builds the 2x2 table of (model correct?, baseline correct?) and runs the
    exact (binomial) McNemar test via statsmodels.  ``b`` counts rows where
    the model is right and the baseline wrong; ``c`` the converse.  The exact
    two-sided p-value equals ``2 * BinomCDF(min(b, c); b + c, 0.5)``.
    """
    y_arr = np.asarray(y_true, dtype=int)
    m_arr = np.asarray(model_pred, dtype=int)
    b_arr = np.asarray(baseline_pred, dtype=int)
    if not (len(y_arr) == len(m_arr) == len(b_arr)):
        raise ValueError("length mismatch")
    model_right = m_arr == y_arr
    base_right = b_arr == y_arr
    both_right = int((model_right & base_right).sum())
    b = int((model_right & ~base_right).sum())
    c = int((~model_right & base_right).sum())
    both_wrong = int((~model_right & ~base_right).sum())
    table = [[both_right, b], [c, both_wrong]]
    result = mcnemar(np.asarray(table), exact=True)
    return {
        "both_right": both_right,
        "model_right_only": b,
        "baseline_right_only": c,
        "both_wrong": both_wrong,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def holm_bonferroni(pvals: Mapping[str, float]) -> dict[str, float]:
    """Holm (1979) step-down adjustment for a family of p-values.

    Sort p-values ascending; multiply the k-th smallest by (m - k + 1);
    enforce monotonicity by taking running maxima; clip at 1.
    """
    if not pvals:
        return {}
    names = sorted(pvals, key=lambda k: pvals[k])
    m = len(names)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, name in enumerate(names):
        adj = min(1.0, (m - rank) * pvals[name])
        running_max = max(running_max, adj)
        adjusted[name] = float(running_max)
    return adjusted


# --------------------------------------------------------------------------- #
# Label-permutation sanity control
# --------------------------------------------------------------------------- #
def label_permutation_control(
    estimator: BaseEstimator,
    X_train,
    y_train: Sequence[int],
    X_eval,
    y_eval: Sequence[int],
    seed: int = 42,
) -> float:
    """One-run sanity control: destroy the label-feature relationship.

    The training labels are randomly permuted, the estimator is refit on the
    permuted labels, and the ROC-AUC of its scores is computed against the
    (same-permutation) evaluation labels.  A correctly behaving pipeline must
    land near 0.5; a materially higher value indicates leakage or a bug, not
    alpha.  One permutation run, reported as a sanity check -- NOT a full
    permutation test.
    """
    rng = np.random.default_rng(seed)
    y_tr = np.asarray(y_train, dtype=int)
    y_ev = np.asarray(y_eval, dtype=int)
    perm_train = rng.permutation(len(y_tr))
    perm_eval = rng.permutation(len(y_ev))
    est = clone(estimator)
    est.fit(X_train, y_tr[perm_train])
    scores = est.predict_proba(X_eval)[:, 1]
    return float(roc_auc_score(y_ev[perm_eval], scores))


# --------------------------------------------------------------------------- #
# Cost-adjusted long/flat backtest
# --------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    """Result container for the long/flat strategy vs buy-and-hold."""

    strategy_cum_return: float
    bh_cum_return: float
    strategy_annualised: float
    bh_annualised: float
    n_switches: int
    total_cost: float  # sum of per-day costs (in return units)
    exposure: float  # fraction of days long
    strategy_curve: pd.Series = field(repr=False)
    bh_curve: pd.Series = field(repr=False)
    strategy_daily: pd.Series = field(repr=False)


def long_flat_backtest(
    held_positions: pd.Series,
    returns: pd.Series,
    cost_bps: float = 10.0,
    initial_position: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> BacktestResult:
    """Backtest a long/flat next-day strategy against buy-and-hold.

    Parameters
    ----------
    held_positions:
        Binary position (1 = long, 0 = flat) held during each day, indexed by
        that day.  The mapping "decision made at close of day ``t`` -> position
        during day ``t+1``" is an explicit ``shift(1)`` at the call site
        (see :func:`build_holdout_positions`) and is covered by unit tests.
    returns:
        Simple (arithmetic) daily returns of the asset, same index.
    cost_bps:
        Cost in basis points per unit of position change (see module
        assumptions block).  The initial entry from ``initial_position`` is
        charged on the first day as well.

    Returns a :class:`BacktestResult`; ``*_cum_return`` are simple cumulative
    returns (wealth - 1) with daily compounding of net arithmetic returns.
    """
    if len(held_positions) == 0 or len(returns) == 0:
        raise ValueError("empty inputs")
    ret = returns.astype(float).sort_index()
    held = held_positions.astype(float).reindex(ret.index)
    if held.isna().any():
        raise ValueError("held_positions must cover every return day")
    held = held.clip(0.0, 1.0)

    cost_rate = cost_bps / 1e4
    turnover = held.diff()
    turnover.iloc[0] = held.iloc[0] - initial_position
    turnover = turnover.abs()
    cost = cost_rate * turnover

    strat_daily = held * ret - cost
    bh_daily = ret - cost_rate * pd.Series(
        np.arange(len(ret)) == 0, index=ret.index, dtype=float
    )
    strat_curve = (1.0 + strat_daily).cumprod()
    bh_curve = (1.0 + bh_daily).cumprod()
    strat_cum = float(strat_curve.iloc[-1] - 1.0)
    bh_cum = float(bh_curve.iloc[-1] - 1.0)

    def annualised(cum: float, n: int) -> float:
        if n == 0 or cum <= -1.0:
            return -1.0
        return float((1.0 + cum) ** (trading_days_per_year / n) - 1.0)

    return BacktestResult(
        strategy_cum_return=strat_cum,
        bh_cum_return=bh_cum,
        strategy_annualised=annualised(strat_cum, len(ret)),
        bh_annualised=annualised(bh_cum, len(ret)),
        n_switches=int((turnover > 0).sum()),
        total_cost=float(cost.sum()),
        exposure=float(held.mean()),
        strategy_curve=strat_curve,
        bh_curve=bh_curve,
        strategy_daily=strat_daily,
    )


def build_holdout_positions(
    predictions: pd.Series,
    holdout_index: pd.Index,
    first_position: float,
) -> pd.Series:
    """Map per-row predictions onto the position held on each holdout day.

    ``predictions`` is indexed by decision day ``t``: the decision is made at
    the close of day ``t`` and applies to day ``t+1``.  The position held on
    holdout day ``d`` is therefore the prediction made at ``d - 1``, i.e. an
    explicit ``shift(1)`` re-indexed onto the holdout days.  The first holdout
    day is governed by ``first_position`` -- the prediction the same
    train-region-only model made at the last pre-holdout row.
    """
    positions = predictions.shift(1).reindex(holdout_index)
    positions.iloc[0] = float(first_position)
    if positions.isna().any():
        raise ValueError(
            "holdout positions contain NaNs: predictions must cover every "
            "holdout day after the first"
        )
    return positions


def metrics_table(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Convenience: list of metric dicts -> tidy DataFrame."""
    return pd.DataFrame(rows)
