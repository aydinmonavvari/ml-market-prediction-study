"""Tests for the walk-forward splitter and holdout protocol."""

from __future__ import annotations

import numpy as np
import pytest

from market_prediction_study.validation import (
    ExpandingWalkForward,
    assert_temporal_guarantees,
    chronological_holdout_split,
)


def test_folds_are_chronological_non_overlapping_and_embargoed():
    wf = ExpandingWalkForward(n_splits=5, test_days=10, min_train_days=50, gap=1)
    folds = list(wf.split(150))

    assert len(folds) == 5
    last_train_size = -1
    for train, test in folds:
        assert_temporal_guarantees(train, test, gap=1)
        assert train[0] == 0  # expanding window
        assert len(train) > last_train_size  # strictly expanding
        last_train_size = len(train)
        # exactly `gap` rows are skipped between the last train row and first test row
        assert test[0] - train[-1] == wf.gap + 1

    # test blocks are contiguous and cover exactly the last n_splits*test_days rows
    all_test = np.concatenate([test for _, test in folds])
    assert np.array_equal(all_test, np.arange(150 - 50, 150))
    # training windows expand monotonically
    sizes = [len(train) for train, _ in folds]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


def test_folds_match_planned_geometry():
    wf = ExpandingWalkForward(n_splits=3, test_days=7, min_train_days=20, gap=2)
    plans = wf.plan(60)
    folds = list(wf.split(60))
    for plan, (train, test) in zip(plans, folds):
        assert train[-1] == plan.train_end - 1
        assert test[0] == plan.test_start and test[-1] == plan.test_end - 1
        assert plan.skipped == (plan.test_start - wf.gap, plan.test_start)
        # embargo: the last training label's horizon ends within the skipped block
        assert plan.train_end <= plan.skipped[0] and plan.skipped[1] <= plan.test_start


def test_holdout_split_disjoint_chronological():
    cv, hold = chronological_holdout_split(200, 0.2)
    assert cv[-1] == 159 and hold[0] == 160 and hold[-1] == 199
    assert len(np.intersect1d(cv, hold)) == 0
    with pytest.raises(ValueError):
        chronological_holdout_split(200, 0.9)
    with pytest.raises(ValueError):
        chronological_holdout_split(200, 0.0)


def test_splitter_rejects_impossible_geometry():
    wf = ExpandingWalkForward(n_splits=5, test_days=30, min_train_days=100, gap=1)
    with pytest.raises(ValueError):
        list(wf.split(200))  # needs 100 + 1 + 150 + 1 rows

    for bad in ({"n_splits": 0}, {"test_days": 0}, {"min_train_days": 0}, {"gap": -1}):
        with pytest.raises(ValueError):
            ExpandingWalkForward(**bad)


def test_guarantee_assertions_fire():
    train = np.arange(0, 10)
    test_overlap = np.arange(8, 15)
    with pytest.raises(AssertionError):
        assert_temporal_guarantees(train, test_overlap, gap=1)
    test_embargo = np.arange(10, 15)  # no skipped row with gap=1
    with pytest.raises(AssertionError):
        assert_temporal_guarantees(train, test_embargo, gap=1)
    test_ok = np.arange(12, 15)
    assert_temporal_guarantees(train, test_ok, gap=1)  # must not raise
