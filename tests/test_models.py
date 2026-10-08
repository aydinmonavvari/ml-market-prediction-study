"""Model zoo smoke tests + an offline end-to-end pipeline test on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from conftest import make_ohlcv
from market_prediction_study.models import (
    MODEL_ORDER,
    PARAM_GRIDS,
    clone_with_params,
    fit_predict_proba,
    make_estimators,
)
from market_prediction_study.pipeline import run_study


@pytest.fixture(scope="module")
def tiny_xy():
    rng = np.random.default_rng(42)
    n = 240
    X = rng.normal(size=(n, 6))
    # weak but real signal so classifiers can fit without warnings
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=1.5, size=n) > 0).astype(int)
    return X[:200], y[:200], X[200:], y[200:]


def test_all_models_train_and_predict(tiny_xy):
    X_tr, y_tr, X_te, _ = tiny_xy
    estimators = make_estimators(seed=42)
    for name in MODEL_ORDER:
        est = clone_with_params(estimators[name], PARAM_GRIDS[name][0])
        classes, scores = fit_predict_proba(est, X_tr, y_tr, X_te)
        assert 1 in classes
        assert scores.shape == (len(X_te),)
        assert np.all((scores >= 0.0) & (scores <= 1.0))
        assert np.isfinite(scores).all()


def test_seeded_models_are_deterministic(tiny_xy):
    """Pinned seeds make training reproducible up to floating-point noise.

    Histogram-based learners (XGBoost) sum floats in a thread-pool order that
    is not guaranteed bit-identical across runs, so identical seeds can differ
    by ~1e-16.  We therefore assert numerical equivalence at a tolerance far
    below any decision-relevant scale instead of bit equality.
    """
    X_tr, y_tr, X_te, _ = tiny_xy
    for name in ("random_forest", "xgboost", "gradient_boosting", "logistic_regression"):
        est = make_estimators(seed=42)[name]
        _, s1 = fit_predict_proba(est, X_tr, y_tr, X_te)
        _, s2 = fit_predict_proba(est, X_tr, y_tr, X_te)
        np.testing.assert_allclose(s1, s2, rtol=1e-10, atol=1e-10)


def test_scalers_live_inside_pipelines():
    """Scaling-sensitive models must carry their scaler inside a Pipeline so the
    scaler is fit on training folds only (no statistic leakage across folds)."""
    estimators = make_estimators(seed=42)
    for name in ("logistic_regression", "mlp"):
        assert isinstance(estimators[name], Pipeline)
        assert "scaler" in estimators[name].named_steps
    for name in ("random_forest", "gradient_boosting", "xgboost"):
        assert not isinstance(estimators[name], Pipeline)


def test_scaler_uses_train_statistics_only(tiny_xy):
    X_tr, y_tr, X_te, _ = tiny_xy
    pipe = clone_with_params(make_estimators(seed=42)["logistic_regression"], {"clf__C": 1.0})
    pipe.fit(X_tr, y_tr)
    scaler = pipe.named_steps["scaler"]
    # fitted statistics come from the training block only
    np.testing.assert_allclose(scaler.mean_, X_tr.mean(axis=0), atol=1e-10)
    np.testing.assert_allclose(scaler.scale_, np.sqrt(X_tr.var(axis=0)), atol=1e-10)
    # and transforming the test set applies those train statistics
    expected = (X_te - X_tr.mean(axis=0)) / np.sqrt(X_tr.var(axis=0))
    np.testing.assert_allclose(scaler.transform(X_te), expected, atol=1e-10)


def test_majority_baseline_predicts_training_majority(tiny_xy):
    X_tr, y_tr, X_te, _ = tiny_xy
    est = make_estimators(seed=42)["majority_baseline"]
    est.fit(X_tr, y_tr)
    preds = est.predict(X_te)
    assert np.all(preds == np.bincount(y_tr).argmax())


def test_end_to_end_pipeline_offline(tmp_path, small_config):
    """Full study run on synthetic cached data -- no network, small geometry."""
    df = make_ohlcv(n=320, seed=11, start="2014-01-02")
    small_config.ensure_dirs()
    df.to_csv(small_config.raw_dir / "spy_daily.csv")  # cache hit -> no download
    df[["dff"]].to_csv(small_config.raw_dir / "fred_dff.csv")

    results = run_study(small_config)

    assert results["n_rows"] > 100
    metrics = results["metrics"]
    assert set(metrics["split"]) == {"cv_oof", "holdout"}
    assert set(metrics["model"]) == set(MODEL_ORDER)

    mcn = results["mcnemar"]
    assert len(mcn) == len(MODEL_ORDER) - 1
    assert mcn["p_value_holm"].between(0, 1).all()

    bt = results["backtest"]
    assert len(bt) == len(MODEL_ORDER)
    assert bt["bh_cum_return"].nunique() == 1  # identical buy&hold for all rows

    # reports and figures were produced
    for fname in (
        "metrics.csv",
        "mcnemar.csv",
        "backtest.csv",
        "hp_search.csv",
        "holdout_predictions.csv",
        "summary.md",
    ):
        assert (small_config.reports_dir / fname).exists()
    for fig in (
        "walkforward_folds.png",
        "equity_curves.png",
        "roc_pr_curves.png",
        "feature_importance.png",
        "cv_fold_scores.png",
    ):
        assert (small_config.figures_dir / fig).exists()
    assert "Verdict" in results["summary_md"] or "verdict" in results["summary_md"]
