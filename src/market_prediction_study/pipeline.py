"""End-to-end study pipeline.

Run: ``python -m market_prediction_study.pipeline`` or ``scripts/run_study.py``.

Stages
------
1. Load cached (or freshly downloaded) SPY + FRED DFF data.
2. Build the leakage-free design matrix (single explicit shift in
   :mod:`market_prediction_study.features`).
3. Reserve the chronologically-last 20% as an untouched holdout.
4. Inside the CV region only: expanding walk-forward hyperparameter search per
   model (small grids, scored by mean per-fold balanced accuracy).
5. Refit each model with its best hyperparameters on the whole CV region;
   predict the holdout exactly once.
6. Metrics + exact McNemar tests vs the majority baseline + Holm adjustment;
   one-run label-permutation sanity control; stationary-bootstrap CIs for the
   holdout ROC-AUC of every model (serially dependent daily data -- IID
   bootstrap would be invalid).
7. Cost-adjusted long/flat backtest vs buy-and-hold on the holdout only.
8. Figures and reports (CSV + Markdown) with the ACTUAL numbers.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from . import evaluation as ev
from . import plots
from .bootstrap import stationary_bootstrap_auc_ci
from .config import StudyConfig
from .data import load_dataset
from .features import FEATURE_COLUMNS, build_model_matrices
from .models import MODEL_ORDER, PARAM_GRIDS, clone_with_params, fit_predict_proba, make_estimators
from .validation import ExpandingWalkForward, chronological_holdout_split


def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def _params_str(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True) if params else "{}"


def hyperparameter_search(
    model_name: str,
    estimator,
    grid: list[dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    wf: ExpandingWalkForward,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Score every hyperparameter combination by mean per-fold balanced accuracy.

    All fitting happens strictly inside the CV region on the walk-forward
    training blocks; the holdout is never touched.
    """
    rows = []
    for params in grid:
        est = clone_with_params(estimator, params)
        fold_bal_acc: list[float] = []
        oof_pred = pd.Series(index=X.index, dtype=float)
        oof_score = pd.Series(index=X.index, dtype=float)
        for train_idx, test_idx in folds:
            classes, scores = fit_predict_proba(
                est, X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx]
            )
            pred = (scores >= 0.5).astype(int)
            fold_bal_acc.append(float(balanced_accuracy_score(y.iloc[test_idx], pred)))
            oof_pred.iloc[test_idx] = pred
            oof_score.iloc[test_idx] = scores
        valid = oof_pred.notna()
        rows.append(
            {
                "model": model_name,
                "params": _params_str(params),
                "mean_bal_acc": float(np.mean(fold_bal_acc)),
                "fold_scores": json.dumps([round(v, 6) for v in fold_bal_acc]),
                "oof_bal_acc": float(balanced_accuracy_score(y[valid], oof_pred[valid])),
            }
        )
    frame = pd.DataFrame(rows)
    # stable: best = highest mean fold balanced accuracy, ties -> earliest row
    return frame.sort_values("mean_bal_acc", ascending=False, kind="stable").reset_index(drop=True)


def run_study(config: StudyConfig | None = None, save_outputs: bool = True) -> dict[str, Any]:
    """Execute the full study; returns a results dict and writes reports/figures."""
    config = config or StudyConfig()
    seed = config.seed
    config.ensure_dirs()

    # -- 1-2. data + features ------------------------------------------------
    raw = load_dataset(config)
    mats = build_model_matrices(raw, rsi_period=config.rsi_period)
    X, y = mats.X, mats.y_up
    n = len(X)
    if n < config.min_rows:
        raise ValueError(f"usable sample too small for the configured design: {n} rows")

    # simple next-day returns of each row's *own* day, for the backtest
    simple_ret = raw["close"].pct_change().reindex(X.index)

    # -- 3. holdout -----------------------------------------------------------
    cv_idx, hold_idx = chronological_holdout_split(n, config.holdout_frac)
    X_cv, y_cv = X.iloc[cv_idx], y.iloc[cv_idx]
    X_hold, y_hold = X.iloc[hold_idx], y.iloc[hold_idx]

    # -- 4. walk-forward hyperparameter search inside the CV region -----------
    wf = ExpandingWalkForward(
        n_splits=config.n_splits,
        test_days=config.test_days,
        min_train_days=config.min_train_days,
        gap=config.gap,
    )
    wf.validate_length(len(cv_idx))
    folds = list(wf.split(len(cv_idx)))

    estimators = make_estimators(seed=seed)
    search_frames: dict[str, pd.DataFrame] = {}
    best_params: dict[str, dict[str, Any]] = {}
    oof_preds: dict[str, pd.Series] = {}
    oof_scores: dict[str, pd.Series] = {}
    for name in MODEL_ORDER:
        frame = hyperparameter_search(
            name, estimators[name], PARAM_GRIDS[name], X_cv, y_cv, wf, folds
        )
        search_frames[name] = frame
        best = frame.iloc[0]
        best_params[name] = json.loads(best["params"])
        # recompute OOF predictions for the winning combination
        est = clone_with_params(estimators[name], best_params[name])
        pred = pd.Series(index=X_cv.index, dtype=float)
        score = pd.Series(index=X_cv.index, dtype=float)
        for train_idx, test_idx in folds:
            _cls, scores = fit_predict_proba(
                est, X_cv.iloc[train_idx], y_cv.iloc[train_idx], X_cv.iloc[test_idx]
            )
            pred.iloc[test_idx] = (scores >= 0.5).astype(int)
            score.iloc[test_idx] = scores
        oof_preds[name] = pred
        oof_scores[name] = score

    # -- 5. final fit on the CV region, single holdout evaluation -------------
    holdout_scores: dict[str, pd.Series] = {}
    holdout_preds: dict[str, pd.Series] = {}
    first_positions: dict[str, float] = {}
    fitted: dict[str, Any] = {}
    prev_row = X.iloc[[hold_idx[0] - 1]]  # last pre-holdout row (in CV region)
    for name in MODEL_ORDER:
        est = clone_with_params(estimators[name], best_params[name])
        est.fit(X_cv, y_cv)
        fitted[name] = est
        scores = est.predict_proba(X_hold)[:, list(est.classes_).index(1)]
        holdout_scores[name] = pd.Series(scores, index=X_hold.index)
        holdout_preds[name] = pd.Series(est.predict(X_hold), index=X_hold.index)
        # decision governing the first holdout day, made by the same
        # train-region-only model at the last pre-holdout row
        first_positions[name] = float(est.predict(prev_row)[0])

    # -- 6. metrics + significance + sanity control ---------------------------
    metric_rows = []
    for name in MODEL_ORDER:
        for split, (yt, yp, ys) in {
            "cv_oof": (y_cv, oof_preds[name], oof_scores[name]),
            "holdout": (y_hold, holdout_preds[name], holdout_scores[name]),
        }.items():
            m = ev.classification_metrics(yt, yp, ys)
            metric_rows.append({"model": name, "split": split, **m})

    # Exact class balance per split, persisted so every quoted base rate is
    # artifact-backed (``base_rate`` above covers only the rows a model is
    # scored on; these rows cover the full sample and the whole CV region).
    nan = float("nan")
    base_rate_rows = [
        {
            "model": "base_rate_full",
            "split": "full",
            "accuracy": nan,
            "balanced_accuracy": nan,
            "roc_auc": nan,
            "pr_auc": nan,
            "n": float(n),
            "base_rate": float(y.mean()),
        },
        {
            "model": "base_rate_cv_region",
            "split": "cv_region",
            "accuracy": nan,
            "balanced_accuracy": nan,
            "roc_auc": nan,
            "pr_auc": nan,
            "n": float(len(y_cv)),
            "base_rate": float(y_cv.mean()),
        },
        {
            "model": "base_rate_holdout",
            "split": "holdout",
            "accuracy": nan,
            "balanced_accuracy": nan,
            "roc_auc": nan,
            "pr_auc": nan,
            "n": float(len(y_hold)),
            "base_rate": float(y_hold.mean()),
        },
    ]
    metrics = pd.DataFrame(metric_rows + base_rate_rows)

    base_pred = holdout_preds["majority_baseline"]
    mcnemar_rows = []
    for name in MODEL_ORDER:
        if name == "majority_baseline":
            continue
        res = ev.mcnemar_vs_baseline(y_hold, holdout_preds[name], base_pred)
        mcnemar_rows.append({"model": name, **res})
    mcnemar_tbl = pd.DataFrame(mcnemar_rows)
    holm = ev.holm_bonferroni(dict(zip(mcnemar_tbl["model"], mcnemar_tbl["p_value"])))
    mcnemar_tbl["p_value_holm"] = mcnemar_tbl["model"].map(holm)

    perm_rows = []
    for name in MODEL_ORDER:
        auc = ev.label_permutation_control(
            clone_with_params(estimators[name], best_params[name]),
            X_cv, y_cv, X_hold, y_hold, seed=seed + 1,
        )
        perm_rows.append({"model": name, "auc_label_shuffled": auc})
    perm_tbl = pd.DataFrame(perm_rows)

    # Stationary (Politis-Romano) bootstrap CIs for the holdout ROC-AUC.
    # Daily observations are serially dependent, so an IID bootstrap is
    # invalid; see bootstrap.py for rationale, assumptions and limitations.
    boot_rows = []
    for name in MODEL_ORDER:
        res = stationary_bootstrap_auc_ci(
            y_hold,
            holdout_scores[name],
            B=2000,
            expected_block_len=21,
            seed=seed,
        )
        boot_rows.append({"model": name, **res})
    boot_tbl = pd.DataFrame(boot_rows)

    # -- 7. cost-adjusted long/flat backtest (holdout only) -------------------
    backtest_rows = []
    curves: dict[str, pd.Series] = {}
    backtests: dict[str, Any] = {}
    for name in MODEL_ORDER:
        positions = ev.build_holdout_positions(
            holdout_preds[name], X_hold.index, first_position=first_positions[name]
        )
        result = ev.long_flat_backtest(
            positions, simple_ret.loc[X_hold.index], cost_bps=config.cost_bps
        )
        backtests[name] = result
        curves[name] = result.strategy_curve
        backtest_rows.append(
            {
                "model": name,
                "strategy_cum_return": result.strategy_cum_return,
                "bh_cum_return": result.bh_cum_return,
                "strategy_minus_bh_pp": 100.0 * (result.strategy_cum_return - result.bh_cum_return),
                "strategy_annualised": result.strategy_annualised,
                "bh_annualised": result.bh_annualised,
                "n_switches": result.n_switches,
                "total_cost_pct": 100.0 * result.total_cost,
                "exposure": result.exposure,
            }
        )
    backtest_tbl = pd.DataFrame(backtest_rows)

    # -- 8. figures ------------------------------------------------------------
    holdout_start = int(hold_idx[0])
    plots.plot_walkforward_folds(
        wf.plan(len(cv_idx)), n, holdout_start, config.figures_dir / "walkforward_folds.png"
    )
    bh_curve = backtests["majority_baseline"].bh_curve
    plots.plot_equity_curves(curves, bh_curve, config.figures_dir / "equity_curves.png")
    plots.plot_roc_pr(holdout_scores, y_hold, config.figures_dir / "roc_pr_curves.png")
    plots.plot_feature_importance(
        {k: fitted[k] for k in ("random_forest", "gradient_boosting", "xgboost")},
        FEATURE_COLUMNS,
        config.figures_dir / "feature_importance.png",
    )
    fold_scores = {
        name: [float(v) for v in json.loads(search_frames[name].iloc[0]["fold_scores"])]
        for name in MODEL_ORDER
        if name != "majority_baseline"
    }
    plots.plot_cv_fold_scores(fold_scores, config.figures_dir / "cv_fold_scores.png")

    # -- reports ---------------------------------------------------------------
    summary = build_summary(
        config,
        X.index,
        metrics,
        mcnemar_tbl,
        perm_tbl,
        backtest_tbl,
        search_frames,
        n,
        boot_tbl=boot_tbl,
        n_cv_rows=int(len(cv_idx)),
    )

    if save_outputs:
        metrics.to_csv(config.reports_dir / "metrics.csv", index=False)
        mcnemar_tbl.to_csv(config.reports_dir / "mcnemar.csv", index=False)
        perm_tbl.to_csv(config.reports_dir / "permutation_control.csv", index=False)
        backtest_tbl.to_csv(config.reports_dir / "backtest.csv", index=False)
        boot_tbl[
            ["model", "auc_point", "ci_low", "ci_high", "B", "block_len", "seed"]
        ].to_csv(config.reports_dir / "auc_bootstrap.csv", index=False)
        pd.concat(
            [f.assign(model=name) for name, f in search_frames.items()]
        ).to_csv(config.reports_dir / "hp_search.csv", index=False)
        preds_out = pd.DataFrame({"y_true": y_hold})
        for name in MODEL_ORDER:
            preds_out[f"score_{name}"] = holdout_scores[name]
            preds_out[f"pred_{name}"] = holdout_preds[name]
        preds_out.to_csv(config.reports_dir / "holdout_predictions.csv")
        mats.X.join(mats.y_up).to_csv(config.processed_dir / "dataset_features.csv")
        (config.reports_dir / "summary.md").write_text(summary, encoding="utf-8")

    return {
        "metrics": metrics,
        "mcnemar": mcnemar_tbl,
        "permutation": perm_tbl,
        "backtest": backtest_tbl,
        "auc_bootstrap": boot_tbl,
        "best_params": best_params,
        "summary_md": summary,
        "n_rows": n,
        "cv_rows": int(len(cv_idx)),
        "holdout_rows": int(len(hold_idx)),
        "sample_start": str(X.index[0].date()),
        "sample_end": str(X.index[-1].date()),
    }


def build_summary(
    config: StudyConfig,
    index: pd.DatetimeIndex,
    metrics: pd.DataFrame,
    mcnemar_tbl: pd.DataFrame,
    perm_tbl: pd.DataFrame,
    backtest_tbl: pd.DataFrame,
    search_frames: dict[str, pd.DataFrame],
    n_rows: int,
    boot_tbl: pd.DataFrame | None = None,
    n_cv_rows: int | None = None,
) -> str:
    """Render reports/summary.md from the ACTUAL results."""
    hold = metrics[(metrics["split"] == "holdout") & metrics["model"].isin(MODEL_ORDER)].set_index(
        "model"
    )
    base_rates = metrics[metrics["model"].str.startswith("base_rate_")].set_index("model")
    non_base = [m for m in MODEL_ORDER if m != "majority_baseline"]
    best_model = max(non_base, key=lambda m: hold.loc[m, "roc_auc"])
    best_auc = float(hold.loc[best_model, "roc_auc"])
    best_acc = float(hold.loc[best_model, "accuracy"])
    base_acc = float(hold.loc["majority_baseline", "accuracy"])
    sig_rows = mcnemar_tbl[mcnemar_tbl["p_value_holm"] < 0.05]
    bt = backtest_tbl.set_index("model")
    bh_cum = float(bt.loc[best_model, "bh_cum_return"])
    best_strat = float(bt.loc[best_model, "strategy_cum_return"])

    if len(sig_rows) and best_auc > 0.5:
        verdict = (
            "At least one model beat the majority baseline on the holdout at the "
            "Holm-adjusted 5% level, and the best holdout ROC-AUC is above 0.5 -- "
            "evidence of *some* incremental out-of-sample signal, to be treated with "
            "multiple-testing caution."
        )
    else:
        verdict = (
            "No model beats the majority baseline on the holdout at the Holm-adjusted "
            "5% level (or no model's ROC-AUC exceeds 0.5 meaningfully). Under this "
            "leakage-free walk-forward design, the study finds **little to no "
            "statistically significant incremental out-of-sample signal** beyond "
            "simple baselines -- the honest and expected outcome under near-form "
            "market efficiency."
        )
    # Direction-aware refinement: a two-sided McNemar p-value below 0.05 with
    # model_right_only < baseline_right_only means the model is significantly
    # *worse*, not better. Call that out explicitly so the summary cannot be
    # misread as a positive result.
    worse_names, better_names = [], []
    for _, row in mcnemar_tbl.iterrows():
        if row["p_value_holm"] < 0.05:
            if row["model_right_only"] < row["baseline_right_only"]:
                worse_names.append(str(row["model"]))
            else:
                better_names.append(str(row["model"]))
    if better_names:
        verdict = (
            f"On the holdout, {', '.join(better_names)} made significantly more "
            "correct direction calls than the majority baseline (Holm-adjusted "
            "McNemar p < 0.05) -- evidence of *some* incremental out-of-sample "
            "signal, to be treated with multiple-testing caution."
        )
    elif worse_names:
        worse_txt = (
            f"{' and '.join(worse_names)} made significantly *fewer* correct calls "
            "than the baseline (Holm-adjusted p < 0.05), i.e. it was significantly "
            "worse. "
            if len(worse_names) == 1
            else
            f"{', '.join(worse_names)} made significantly *fewer* correct calls "
            "than the baseline (Holm-adjusted p < 0.05), i.e. they were "
            "significantly worse. "
        )
        verdict = (
            "No model beats the majority baseline on the holdout: no model shows a "
            "significant Holm-adjusted McNemar *improvement* and the best holdout "
            f"ROC-AUC is {best_auc:.4f} -- statistically indistinguishable from "
            f"chance. {worse_txt}"
            "Under this leakage-free walk-forward design, the study finds **little "
            "to no statistically significant incremental out-of-sample signal** "
            "beyond simple baselines -- the honest and expected outcome under "
            "near-form market efficiency."
        )

    lines: list[str] = []
    lines.append("# ML Market Prediction Study -- Results Summary")
    lines.append("")
    lines.append(
        "*Generated by the study pipeline on real data; all numbers below are actual outputs.*"
    )
    lines.append("")
    sample_line = (
        f"- Sample: `{config.symbol}` {index[0].date()} -> {index[-1].date()} "
        f"({n_rows} usable rows)"
    )
    lines.append(sample_line)
    lines.append(
        f"- Design: {config.n_splits}-fold expanding walk-forward, test {config.test_days}d/fold, "
        f"embargo {config.gap}d; holdout = chronologically last {config.holdout_frac:.0%} of rows"
    )
    lines.append(f"- Transaction cost: {config.cost_bps:.0f} bps per position switch")
    lines.append("")
    lines.append("## Headline verdict")
    lines.append("")
    lines.append(verdict)
    lines.append("")
    lines.append(
        f"Best non-baseline model on holdout: **{best_model}** "
        f"(ROC-AUC {best_auc:.4f}, accuracy {best_acc:.4f}; "
        f"majority baseline accuracy {base_acc:.4f}). "
        f"Buy-and-hold over the holdout returned {100 * bh_cum:.2f}%; the {best_model} long/flat "
        f"strategy net of {config.cost_bps:.0f} bps returned {100 * best_strat:.2f}%."
    )
    lines.append("")
    lines.append("## Holdout metrics (single evaluation, once)")
    lines.append("")
    lines.append(
        hold[["accuracy", "balanced_accuracy", "roc_auc", "pr_auc", "n"]].round(4).to_markdown()
    )
    lines.append("")
    lines.append("## McNemar tests vs majority baseline (holdout)")
    lines.append("")
    lines.append(
        mcnemar_tbl[["model", "model_right_only", "baseline_right_only", "p_value", "p_value_holm"]]
        .assign(
            p_value=lambda d: d["p_value"].round(4),
            p_value_holm=lambda d: d["p_value_holm"].round(4),
        )
        .to_markdown(index=False)
    )
    lines.append("")
    lines.append(
        "*Holm (1979) step-down adjustment across the five non-baseline models; the family "
        "is small but the search over models x hyperparameters x metrics is wider -- interpret "
        "with Bonferroni-style caution.*"
    )
    lines.append("")
    lines.append("## Label-permutation sanity control (one run)")
    lines.append("")
    lines.append(perm_tbl.round(4).to_markdown(index=False))
    lines.append("")
    lines.append(
        "*Shuffled labels must yield AUC ~ 0.5; materially higher values would indicate "
        "target leakage. A single permutation run is a SANITY CHECK -- it can catch certain "
        "target-leakage bugs (features derived from the target) but is NOT a proof that no "
        "leakage exists; see also the leakage mutation test, which shows the walk-forward "
        "evaluation DOES inflate AUC when a known leak is injected.*"
    )
    lines.append("")
    if boot_tbl is not None and len(boot_tbl):
        lines.append("## Holdout ROC-AUC -- stationary bootstrap 95% CI")
        lines.append("")
        lines.append(
            boot_tbl[["model", "auc_point", "ci_low", "ci_high", "B", "block_len", "seed"]]
            .round(4)
            .to_markdown(index=False)
        )
        lines.append("")
        lines.append(
            f"*Politis-Romano stationary bootstrap, B={int(boot_tbl['B'].iloc[0])}, expected "
            f"block length {int(boot_tbl['block_len'].iloc[0])} trading days, "
            f"seed={int(boot_tbl['seed'].iloc[0])}. Daily observations are serially "
            "dependent, so an IID bootstrap would be invalid; assumes approximate "
            "stationarity of the holdout window and weak dependence beyond ~1 month.*"
        )
        lines.append("")
    lines.append("## Cost-adjusted long/flat vs buy & hold (holdout only)")
    lines.append("")
    lines.append(backtest_tbl.round(4).to_markdown(index=False))
    lines.append("")
    lines.append("## Best hyperparameters (searched inside walk-forward CV only)")
    lines.append("")
    for name in MODEL_ORDER:
        best = search_frames[name].iloc[0]
        lines.append(
            f"- `{name}`: `{best['params']}` "
            f"(mean fold balanced accuracy {best['mean_bal_acc']:.4f})"
        )
    lines.append("")
    lines.append("## Class balance by split (exact)")
    lines.append("")
    lines.append(
        base_rates.reset_index()[["model", "n", "base_rate"]].round(6).to_markdown(index=False)
    )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    oof_n = int(
        metrics[(metrics["split"] == "cv_oof") & metrics["model"].isin(MODEL_ORDER)]["n"].max()
    )
    cv_rows_txt = f"{n_cv_rows}-row" if n_cv_rows is not None else "CV-region"
    coverage = (
        f"- CV out-of-fold metrics cover only the {oof_n} rows of the walk-forward test "
        f"blocks ({config.n_splits} x {config.test_days}d), not the full {cv_rows_txt} CV "
        "region: the first rows of the CV region cannot be scored out-of-fold because the "
        f"first fold requires >= {config.min_train_days} training days, and the {config.gap}-day "
        "embargo gaps are legitimately uncovered.\n"
    )
    lines.append(
        "- No intraday fills; decisions at the close, applied to the next day's return; "
        "slippage ignored.\n"
        + coverage
        + "- Single asset (SPY as S&P 500 proxy), single period, one random seed.\n"
        "- Hyperparameters were selected on the same walk-forward predictions reported as CV "
        "metrics -> CV numbers are mildly optimistic (selection/overfitting-the-CV bias); the "
        "holdout is the honest estimate.\n"
        "- This is a study, not investment advice."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover
    result = run_study(StudyConfig.from_yaml("configs/default.yaml"))
    print(result["summary_md"])
