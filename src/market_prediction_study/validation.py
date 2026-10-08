"""Temporal validation: expanding walk-forward with embargo, plus holdout split.

Design
------
The study uses two disjoint temporal regions:

* **CV region** -- the earliest ``(1 - holdout_frac)`` of the sample.  Inside
  it, an *expanding-window walk-forward* scheme produces ``n_splits`` folds:
  the test blocks are the ``n_splits * test_days`` most recent rows of the CV
  region, laid back to back; every training block stretches from the very
  first row up to ``gap`` rows before the matching test block (so the training
  window grows monotonically across folds).
* **Holdout region** -- the chronologically last ``holdout_frac`` of the
  sample.  It is excluded from all model selection and touched exactly once.

Embargo semantics (``gap=1`` by default)
----------------------------------------
Let the first test index of a fold be ``s``.  With ``gap = g`` the training
block ends at index ``s - g - 1`` and indices ``s - g .. s - 1`` are skipped.
A training label at row ``t`` depends on the price move between ``t`` and
``t+1``; with ``g >= 1`` the last training label's horizon terminates at index
``s - g <= s - 1``, strictly *before* the first test day ``s``.  Hence no
training label horizon ever overlaps a test day -- the leakage mode that
embargo exists to prevent (cf. Lopez de Prado, *Advances in Financial Machine
Learning*, 2018, ch. 7 "Purged k-fold CV and embargo").  This is the same
contract as ``sklearn.model_selection.TimeSeriesSplit(gap=1)`` with an
expanding window and fixed test size, implemented manually so the guarantees
are asserted inside the code and unit-tested.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


def assert_temporal_guarantees(
    train: np.ndarray, test: np.ndarray, gap: int
) -> None:
    """Raise ``AssertionError`` unless train/test respect all temporal rules.

    Rules: non-empty; train entirely before test; no overlap; at least ``gap``
    skipped rows between the last training row and the first test row; contiguous,
    strictly increasing indices.
    """
    assert len(train) > 0 and len(test) > 0, "empty fold"
    assert train[-1] < test[0], "train/test overlap or non-chronological"
    assert test[0] - train[-1] > gap, (
        f"embargo violated: first test {test[0]} must be more than {gap} "
        f"rows after last train {train[-1]}"
    )
    assert np.all(np.diff(train) == 1) and np.all(np.diff(test) == 1), "non-contiguous indices"
    assert len(np.intersect1d(train, test)) == 0, "train/test intersection"


@dataclass(frozen=True)
class FoldPlan:
    """Bookkeeping for one walk-forward fold (0-based indices)."""

    fold: int
    train_start: int
    train_end: int  # exclusive
    test_start: int  # inclusive
    test_end: int  # exclusive
    skipped: tuple[int, int]  # embargo rows [start, end) between train and test


class ExpandingWalkForward:
    """Expanding-window walk-forward splitter with a fixed embargo gap."""

    def __init__(
        self,
        n_splits: int = 8,
        test_days: int = 63,
        min_train_days: int = 756,
        gap: int = 1,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if test_days < 1:
            raise ValueError("test_days must be >= 1")
        if min_train_days < 1:
            raise ValueError("min_train_days must be >= 1")
        if gap < 0:
            raise ValueError("gap must be >= 0")
        self.n_splits = n_splits
        self.test_days = test_days
        self.min_train_days = min_train_days
        self.gap = gap

    @property
    def n_test_rows(self) -> int:
        return self.n_splits * self.test_days

    def validate_length(self, n_samples: int) -> None:
        needed = self.min_train_days + self.gap + self.n_test_rows
        if n_samples < needed + 1:
            raise ValueError(
                f"need at least {needed + 1} CV rows for "
                f"min_train_days={self.min_train_days}, gap={self.gap}, "
                f"n_splits={self.n_splits} x test_days={self.test_days}; got {n_samples}"
            )

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` pairs with all guarantees asserted."""
        self.validate_length(n_samples)
        cv_test_start = n_samples - self.n_test_rows
        for fold in range(self.n_splits):
            test_start = cv_test_start + fold * self.test_days
            test_end = test_start + self.test_days
            train_end = test_start - self.gap  # exclusive
            train = np.arange(0, train_end)
            test = np.arange(test_start, test_end)
            assert_temporal_guarantees(train, test, self.gap)
            if len(train) < self.min_train_days:
                raise RuntimeError(
                    f"fold {fold}: train size {len(train)} < min_train_days "
                    f"{self.min_train_days}"
                )
            yield train, test

    def plan(self, n_samples: int) -> list[FoldPlan]:
        """Materialise the fold plan (used for plots and tests)."""
        self.validate_length(n_samples)
        cv_test_start = n_samples - self.n_test_rows
        plans = []
        for fold in range(self.n_splits):
            test_start = cv_test_start + fold * self.test_days
            plans.append(
                FoldPlan(
                    fold=fold,
                    train_start=0,
                    train_end=test_start - self.gap,
                    test_start=test_start,
                    test_end=test_start + self.test_days,
                    skipped=(test_start - self.gap, test_start),
                )
            )
        return plans


def chronological_holdout_split(
    n_samples: int, holdout_frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``[0, n)`` into CV indices and a chronologically-last holdout.

    Returns ``(cv_idx, holdout_idx)``; the two sets are disjoint, contiguous
    and chronological.  The holdout is the *most recent* ``holdout_frac`` of
    the rows.
    """
    if not 0.0 < holdout_frac < 0.5:
        raise ValueError("holdout_frac must be in (0, 0.5)")
    split = int(round(n_samples * (1.0 - holdout_frac)))
    cv = np.arange(0, split)
    holdout = np.arange(split, n_samples)
    assert len(cv) > 0 and len(holdout) > 0
    assert cv[-1] < holdout[0], "holdout must be strictly after the CV region"
    assert len(np.intersect1d(cv, holdout)) == 0, "holdout overlaps CV region"
    return cv, holdout
