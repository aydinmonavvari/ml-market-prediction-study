"""Leakage MUTATION test -- what the permutation control alone cannot give.

The label-permutation control (evaluation.py) is a sanity check: with labels
destroyed, a clean pipeline must return chance-level AUC.  It can catch
certain target-leakage bugs (features derived from the target) but it is NOT
a proof that no leakage exists.  This module provides the complementary,
positive control: INJECT a known leak and verify that the walk-forward
evaluation actually catches it by inflating AUC.

Epistemic status of the shipped pipeline's leakage controls (exact roles):
  1. label-permutation control (one run) -- sanity check that permuted
     labels yield chance-level AUC;
  2. THIS mutation test -- demonstrates the walk-forward design detects
     injected leaks (a feature equal to the future label inflates OOF AUC
     to ~1.0 while the clean pipeline stays at chance);
  3. scaler-isolation tests (test_models.py) -- the scaler is fit on
  training folds only.
Together: the design detects leaks when present, and no evidence of leakage
exists in the shipped pipeline.  Neither is a formal proof of absence.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from conftest import make_ohlcv
from market_prediction_study.features import build_model_matrices
from market_prediction_study.models import fit_predict_proba
from market_prediction_study.validation import ExpandingWalkForward


def _walk_forward_oof_auc(X, y, folds) -> float:
    """Walk-forward OOF AUC exactly as the study computes it (test blocks only)."""
    y_true, y_score = [], []
    for train_idx, test_idx in folds:
        _cls, scores = fit_predict_proba(
            Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
            X.iloc[train_idx],
            y.iloc[train_idx],
            X.iloc[test_idx],
        )
        y_true.append(np.asarray(y.iloc[test_idx], dtype=int))
        y_score.append(scores)
    return float(roc_auc_score(np.concatenate(y_true), np.concatenate(y_score)))


def test_leakage_mutation_is_detected():
    """Inject a known leak (feature = the row's own future label) and assert
    the walk-forward AUC inflates dramatically, while the clean pipeline
    stays at chance level.

    The synthetic path is a random walk, so its features carry no genuine
    next-day directional signal: the clean OOF AUC concentrates near 0.5
    (0.4987 for this fixed fixture/seed).  Adding a feature that equals the
    label of the same row -- the label at row *t* is the direction of day
    *t+1*, i.e. pure future information -- must be picked up by the
    walk-forward folds and push the OOF AUC towards 1.0.  The test therefore
    demonstrates that the evaluation design DOES catch leaks when present.
    """
    df = make_ohlcv(n=1200, seed=7)
    mats = build_model_matrices(df)
    X, y = mats.X, mats.y_up

    # Small walk-forward geometry (same shape as the offline test fixture):
    # 4 folds x 40 test days, min 80 training days, 1-day embargo.
    wf = ExpandingWalkForward(n_splits=4, test_days=40, min_train_days=80, gap=1)
    folds = list(wf.split(len(X)))

    auc_clean = _walk_forward_oof_auc(X, y, folds)
    assert 0.35 < auc_clean < 0.65, (
        f"clean pipeline should be at chance level on the noise fixture, got {auc_clean:.4f}"
    )

    X_leak = X.copy()
    X_leak["leaked_next_day_label"] = y.astype(float)  # THE injected leak
    auc_leak = _walk_forward_oof_auc(X_leak, y, folds)

    assert auc_leak > 0.8, f"leaked pipeline must inflate dramatically, got {auc_leak:.4f}"
    assert auc_leak - auc_clean > 0.15  # inflation is large, not a fluke
