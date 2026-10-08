"""Tests for metrics, McNemar, Holm adjustment and the cost-adjusted backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

from market_prediction_study.evaluation import (
    build_holdout_positions,
    classification_metrics,
    holm_bonferroni,
    label_permutation_control,
    long_flat_backtest,
    mcnemar_vs_baseline,
)


def test_mcnemar_known_contingency_table():
    """Realise the table [[both_right=30, model_only=5], [base_only=14, both_wrong=1]].

    Exact McNemar p = 2 * BinomCDF(5; 19, 0.5) = 2 * 16664 / 524288 = 0.0635620.
    """
    n = 50
    y = np.zeros(n, dtype=int)
    model = np.zeros(n, dtype=int)
    base = np.zeros(n, dtype=int)
    # rows 0-29: both right (y=1, model=1, base=1)
    y[:30] = 1
    model[:30] = 1
    base[:30] = 1
    # rows 30-34: model right only (y=1, model=1, base=0)
    y[30:35] = 1
    model[30:35] = 1
    # rows 35-48: baseline right only (y=0, model=1 wrong, base=0)
    model[35:49] = 1
    # row 49: both wrong (y=1, model=0, base=0)
    y[49] = 1

    res = mcnemar_vs_baseline(y, model, base)
    assert res["both_right"] == 30
    assert res["model_right_only"] == 5
    assert res["baseline_right_only"] == 14
    assert res["both_wrong"] == 1
    expected_p = binomtest(5, 19, 0.5).pvalue  # exact two-sided binomial
    assert res["p_value"] == pytest.approx(expected_p)
    assert res["p_value"] == pytest.approx(0.063568115234375, abs=1e-9)


def test_mcnemar_handles_degenerate_tables():
    y = np.array([1, 1, 0, 0])
    same = np.array([1, 1, 0, 0])
    res = mcnemar_vs_baseline(y, same, same)
    assert res["model_right_only"] == 0 and res["baseline_right_only"] == 0
    assert res["p_value"] == 1.0


def test_holm_bonferroni_step_down():
    adjusted = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
    assert holm_bonferroni({}) == {}
    assert holm_bonferroni({"x": 0.9})["x"] == pytest.approx(0.9)


def test_classification_metrics_known_case():
    y = [1, 1, 0, 0]
    pred = [1, 1, 0, 1]
    score = [0.9, 0.6, 0.4, 0.8]
    m = classification_metrics(y, pred, score)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["balanced_accuracy"] == pytest.approx(0.75)  # (1.0 + 0.5) / 2
    assert m["roc_auc"] == pytest.approx(0.75)  # 3 of 4 pairs ordered correctly
    assert m["pr_auc"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert m["base_rate"] == pytest.approx(0.5)


def test_long_flat_backtest_hand_computed():
    """Toy case, hand-computed.

    held positions [1, 0, 1]; returns [0.01, -0.02, 0.03]; cost 10 bps.
    turnover = [1, 1, 1] (entry from flat + 2 switches) -> cost = 0.001/day.
    strategy daily = [0.009, -0.001, 0.029]
      cum = 1.009 * 0.999 * 1.029 - 1 = 0.0372227...
    buy & hold daily = [0.009, -0.02, 0.03]
      cum = 1.009 * 0.98 * 1.03 - 1 = 0.0184846...
    """
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    held = pd.Series([1.0, 0.0, 1.0], index=idx)
    rets = pd.Series([0.01, -0.02, 0.03], index=idx)

    res = long_flat_backtest(held, rets, cost_bps=10.0)
    assert res.strategy_cum_return == pytest.approx(0.0372227, abs=1e-6)
    assert res.bh_cum_return == pytest.approx(0.0184846, abs=1e-6)
    assert res.n_switches == 3
    assert res.total_cost == pytest.approx(0.003)
    assert res.exposure == pytest.approx(2.0 / 3.0)
    assert res.strategy_curve.index.equals(idx)


def test_long_flat_backtest_zero_cost_matches_always_long():
    idx = pd.date_range("2024-01-01", periods=5)
    held = pd.Series(1.0, index=idx)
    rets = pd.Series([0.01, -0.02, 0.03, 0.005, -0.001], index=idx)
    res = long_flat_backtest(held, rets, cost_bps=0.0)
    assert res.strategy_cum_return == pytest.approx(res.bh_cum_return)


def test_long_flat_backtest_input_validation():
    idx = pd.date_range("2024-01-01", periods=3)
    held = pd.Series([1.0, 0.0, np.nan], index=idx)
    rets = pd.Series([0.01, -0.02, 0.03], index=idx)
    with pytest.raises(ValueError):
        long_flat_backtest(held, rets)
    with pytest.raises(ValueError):
        long_flat_backtest(pd.Series(dtype=float), pd.Series(dtype=float))


def test_build_holdout_positions_uses_previous_day_decision():
    all_days = pd.date_range("2024-01-01", periods=5)
    preds = pd.Series([1, 0, 1, 1, 0], index=all_days, dtype=float)
    holdout_idx = all_days[2:]
    # position on day d = decision made at d-1: [preds[1], preds[2], preds[3]]
    positions = build_holdout_positions(preds, holdout_idx, first_position=0.0)
    assert list(positions) == [0.0, 1.0, 1.0]
    # the first holdout day can be overridden by the pre-holdout decision
    positions2 = build_holdout_positions(preds, holdout_idx, first_position=1.0)
    assert list(positions2) == [1.0, 1.0, 1.0]


def test_label_permutation_control_is_chance_level():
    """With pure noise features, permuted-label AUC must be near 0.5."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = (rng.random(400) < 0.5).astype(int)
    est = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    auc = label_permutation_control(est, X[:300], y[:300], X[300:], y[300:], seed=1)
    assert 0.35 < auc < 0.65
