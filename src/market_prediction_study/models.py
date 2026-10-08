"""Model zoo: majority baseline, regularised logistic, forests, boosting, MLP.

Every estimator is created with ``random_state`` pinned to the study seed so
results are reproducible bit-for-bit.  Scaling-sensitive models (logistic
regression, MLP) are wrapped in a :class:`~sklearn.pipeline.Pipeline` whose
``StandardScaler`` is fit **inside** each training fold only -- this is what
prevents scaler statistics from leaking across the walk-forward boundary.
Tree-based models receive raw features (they are scale-invariant).

Hyperparameter grids are deliberately small; they are searched *only inside*
the walk-forward CV region (never on the holdout), and we treat the resulting
CV scores with Holm/Bonferroni-style caution because selecting over many
model x hyperparameter x metric combinations is itself a multiple-testing
exercise (see README section 10 and Holm 1979).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_ORDER = (
    "majority_baseline",
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "mlp",
)

PARAM_GRIDS: dict[str, list[dict[str, Any]]] = {
    # The baseline has no hyperparameters; included for a uniform loop.
    "majority_baseline": [{}],
    "logistic_regression": [{"clf__C": c} for c in (0.01, 0.1, 1.0)],
    "random_forest": [
        {"max_depth": 4},
        {"max_depth": 8},
    ],
    "gradient_boosting": [
        {"learning_rate": lr, "max_depth": d}
        for lr in (0.05, 0.1)
        for d in (2, 3)
    ],
    "xgboost": [
        {"learning_rate": lr, "max_depth": d}
        for lr in (0.05, 0.1)
        for d in (2, 3)
    ],
    "mlp": [{"clf__alpha": a} for a in (1e-4, 1e-2)],
}


def make_estimators(seed: int = 42) -> dict[str, BaseEstimator]:
    """Instantiate one unfitted estimator per model, all seeds pinned."""
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, random_state=seed),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            subsample=0.8,
            random_state=seed,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            tree_method="hist",
            eval_metric="logloss",
            subsample=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
        ),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(16,),
                        early_stopping=True,
                        max_iter=400,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def clone_with_params(estimator: BaseEstimator, params: dict[str, Any]) -> BaseEstimator:
    """Clone ``estimator`` and apply one hyperparameter combination."""
    return clone(estimator).set_params(**params)


def fit_predict_proba(
    estimator: BaseEstimator,
    X_train,
    y_train,
    X_eval,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on the training block and return ``(classes, P(class=1 | eval))``.

    ``classes`` is the estimator's ``classes_``; callers must map scores back
    to the positive class via this array, not by column position blindly.
    """
    est = clone(estimator)
    est.fit(X_train, y_train)
    proba = est.predict_proba(X_eval)
    cls = np.asarray(est.classes_)
    if 1 not in cls:
        return cls, np.zeros(len(proba))
    pos = int(np.where(cls == 1)[0][0])
    return cls, proba[:, pos]
