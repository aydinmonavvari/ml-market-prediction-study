"""Figure generation for the study (pure matplotlib, non-interactive backend)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_curve  # noqa: E402

from .validation import FoldPlan  # noqa: E402

DPI = 150
COLORS = plt.get_cmap("tab10").colors


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_walkforward_folds(
    plans: Sequence[FoldPlan],
    n_samples: int,
    holdout_start: int,
    path: str | Path,
) -> Path:
    """Fold diagram: expanding train blocks, embargo gaps, test blocks, holdout."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, plan in enumerate(plans):
        y = i
        ax.barh(
            y,
            plan.train_end - plan.train_start,
            left=plan.train_start,
            height=0.62,
            color="#4878CF",
            alpha=0.85,
            label="train (expanding)" if i == 0 else None,
        )
        if plan.skipped[1] > plan.skipped[0]:
            ax.barh(
                y,
                plan.skipped[1] - plan.skipped[0],
                left=plan.skipped[0],
                height=0.62,
                color="#EE854A",
                hatch="///",
                edgecolor="white",
                label="embargo gap" if i == 0 else None,
            )
        ax.barh(
            y,
            plan.test_end - plan.test_start,
            left=plan.test_start,
            height=0.62,
            color="#797979",
            alpha=0.9,
            label="test" if i == 0 else None,
        )
    ax.axvspan(
        holdout_start,
        n_samples,
        color="#D65F5F",
        alpha=0.18,
        label="holdout (untouched)",
    )
    ax.axvline(holdout_start, color="#D65F5F", linestyle="--", linewidth=1)
    ax.set_xlabel("Row index (chronological)")
    ax.set_ylabel("Walk-forward fold")
    ax.set_yticks(range(len(plans)))
    ax.invert_yaxis()
    ax.set_title("Expanding-window walk-forward with 1-day embargo + final holdout")
    ax.legend(loc="lower left", fontsize=8, ncol=4)
    return _save(fig, path)


def plot_equity_curves(
    curves: Mapping[str, pd.Series],
    bh_curve: pd.Series,
    path: str | Path,
    title: str = "Holdout equity curves (net of 10 bps per switch)",
) -> Path:
    """Strategy equity curves vs buy-and-hold on the holdout."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bh_curve.index, bh_curve.values, color="black", linewidth=1.8, label="buy & hold")
    for i, (name, curve) in enumerate(curves.items()):
        ax.plot(curve.index, curve.values, linewidth=1.1, color=COLORS[i % 10], label=name)
    ax.axhline(1.0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_title(title)
    ax.set_ylabel("Cumulative wealth (start = 1.0)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return _save(fig, path)


def plot_roc_pr(
    scores: Mapping[str, pd.Series],
    y_true: pd.Series,
    path: str | Path,
) -> Path:
    """ROC and precision-recall curves per model on the holdout."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.6))
    base_rate = float(y_true.mean())
    for i, (name, s) in enumerate(scores.items()):
        fpr, tpr, _ = roc_curve(y_true, s)
        auc = np.trapezoid(tpr, fpr)
        ap = average_precision_score(y_true, s)
        ax_roc.plot(fpr, tpr, color=COLORS[i % 10], label=f"{name} (AUC={auc:.3f})")
        precision, recall, _ = _precision_recall(y_true, s)
        ax_pr.plot(recall, precision, color=COLORS[i % 10], label=f"{name} (AP={ap:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="grey", linewidth=0.8, linestyle="--", label="chance")
    ax_pr.axhline(base_rate, color="grey", linewidth=0.8, linestyle="--", label="chance")
    ax_roc.set_title("Holdout ROC curves")
    ax_pr.set_title("Holdout precision-recall curves")
    for ax in (ax_roc, ax_pr):
        ax.set_xlabel("false positive rate" if ax is ax_roc else "recall")
        ax.set_ylabel("true positive rate" if ax is ax_roc else "precision")
        ax.legend(fontsize=7, loc="lower right" if ax is ax_roc else "lower left")
        ax.grid(alpha=0.25)
    return _save(fig, path)


def _precision_recall(y_true: pd.Series, scores: pd.Series):
    from sklearn.metrics import precision_recall_curve

    return precision_recall_curve(y_true, scores)


def plot_feature_importance(
    fitted_models: Mapping[str, object],
    feature_names: Sequence[str],
    path: str | Path,
    top_k: int = 12,
) -> Path:
    """Impurity-based importances for the tree ensembles that expose them."""
    usable = {
        name: model
        for name, model in fitted_models.items()
        if hasattr(model, "feature_importances_")
    }
    if not usable:
        raise ValueError("no fitted model exposes feature_importances_")
    fig, axes = plt.subplots(1, len(usable), figsize=(6.2 * len(usable), 4.8), squeeze=False)
    for ax, (i, (name, model)) in zip(axes[0], enumerate(usable.items())):
        imp = np.asarray(model.feature_importances_, dtype=float)
        order = np.argsort(imp)[::-1][:top_k]
        labels = [feature_names[j] for j in order]
        ax.barh(range(len(order)), imp[order][::-1], color=COLORS[i % 10], alpha=0.85)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.set_title(f"{name}: feature importance")
        ax.set_xlabel("importance")
        ax.grid(alpha=0.25, axis="x")
    return _save(fig, path)


def plot_cv_fold_scores(
    fold_scores: Mapping[str, Sequence[float]],
    path: str | Path,
    title: str = "Per-fold walk-forward balanced accuracy (best hyperparameters)",
) -> Path:
    """Balanced accuracy per walk-forward fold per model, with a 0.5 chance line."""
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (name, values) in enumerate(fold_scores.items()):
        ax.plot(
            range(len(values)),
            values,
            marker="o",
            markersize=3.5,
            linewidth=1.0,
            color=COLORS[i % 10],
            label=name,
        )
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.9, label="chance (0.5)")
    ax.set_xlabel("walk-forward fold")
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0.25, 0.85)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    return _save(fig, path)
