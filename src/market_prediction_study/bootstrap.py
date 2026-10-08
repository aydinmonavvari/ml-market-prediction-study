"""Stationary (Politis-Romano) bootstrap for uncertainty quantification.

Why an IID bootstrap is invalid here
------------------------------------
Daily financial observations are serially dependent (autocorrelation in
levels/volumes, volatility clustering, day-of-week effects).  The IID
bootstrap resamples rows independently and therefore destroys exactly the
dependence that determines the sampling variability of statistics computed
on the holdout path; on dependent data its intervals are typically too
narrow.  The stationary bootstrap (Politis & Romano, 1994) instead resamples
*blocks* of consecutive observations whose lengths are geometrically
distributed with expected length ``l_bar``: the first index of every new
block is drawn uniformly, and each subsequent step either continues the
block (next day, wrapping circularly) with probability ``1 - 1/l_bar`` or
restarts a block with probability ``1/l_bar``.  Dependence at lags below
about ``l_bar`` is thereby preserved inside blocks.

Assumptions and limitations (stated up front)
---------------------------------------------
1. *Approximate stationarity* of the joint distribution over the resampled
   window.  The holdout is a single 839-day path; if the data-generating
   process shifts regime inside it, block resampling cannot represent the
   shift.
2. *Weak dependence beyond the expected block length* (here 21 trading
   days, ~ one calendar month).  If material dependence survives beyond
   ``l_bar``, the intervals are still too narrow; the choice of ``l_bar``
   is a judgement call and sensitivity to it should be checked before
   relying on the interval widths.
3. Percentile intervals from ``B`` resamples carry Monte-Carlo noise of
   order ``1/sqrt(B)`` and are first-order accurate at best.  They are an
   uncertainty *summary* for the holdout AUC, not a formal hypothesis test,
   and they quantify sampling uncertainty only -- they say nothing about
   other paths, periods, or instruments.

Reference: Politis, D. N. & Romano, J. P. (1994). "The Stationary
Bootstrap." *Journal of the American Statistical Association*, 89(428),
1303-1313.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def stationary_bootstrap_indices(
    n_samples: int, expected_block_len: int, rng: np.random.Generator
) -> np.ndarray:
    """One stationary-bootstrap resample of row indices (Politis-Romano 1994).

    Parameters
    ----------
    n_samples:
        Number of observations; the resample has the same length.
    expected_block_len:
        Expected (mean) block length ``l_bar``; the per-step restart
        probability is ``1 / l_bar``.
    rng:
        Numpy random ``Generator`` supplying the randomness.

    Returns an ``int64`` array of length ``n_samples`` where each element is
    either the previous element + 1 (mod ``n_samples``, i.e. the block
    continues to the next trading day) or a fresh uniform start (a new block).
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if expected_block_len < 1:
        raise ValueError("expected_block_len must be >= 1")
    p_restart = 1.0 / float(expected_block_len)
    idx = np.empty(n_samples, dtype=np.int64)
    idx[0] = rng.integers(n_samples)
    for t in range(1, n_samples):
        if rng.random() < p_restart:
            idx[t] = rng.integers(n_samples)
        else:
            idx[t] = (idx[t - 1] + 1) % n_samples
    return idx


def stationary_bootstrap_auc_ci(
    y_true,
    y_score,
    B: int = 2000,
    expected_block_len: int = 21,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """95% percentile CI for the holdout ROC-AUC via the stationary bootstrap.

    Parameters
    ----------
    y_true, y_score:
        True labels and model scores for the *holdout* rows (pairs with NaN
        are dropped).  Daily observations are serially dependent, which is
        why the block bootstrap is used (see module docstring).
    B:
        Number of bootstrap resamples (default 2000).
    expected_block_len:
        Expected block length in trading days (default 21 ≈ one month);
        resamples preserve dependence at shorter lags.
    seed:
        Seed for ``numpy.random.default_rng``; the function is fully
        reproducible for a fixed seed.
    alpha:
        Two-sided miscoverage; percentiles ``100*alpha/2`` and
        ``100*(1-alpha/2)`` of the bootstrap AUC distribution.

    Returns a dict with the point AUC, the percentile interval
    (``ci_low <= ci_high``), the settings (``B``, ``block_len``, ``seed``,
    ``n``) and ``n_degenerate`` -- the number of resamples skipped because a
    draw contained a single class (AUC undefined; with realistic class
    balance this is vanishingly rare).
    """
    if B < 1:
        raise ValueError("B must be >= 1")
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(s))
    y = y[ok].astype(int)
    s = s[ok]
    n = len(y)
    if n < 2 or len(np.unique(y)) < 2:
        raise ValueError("AUC bootstrap needs >= 2 rows and both classes")

    auc_point = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    aucs = np.full(B, np.nan)
    n_degenerate = 0
    for b in range(B):
        idx = stationary_bootstrap_indices(n, expected_block_len, rng)
        y_b = y[idx]
        if y_b.min() == y_b.max():  # single-class resample: AUC undefined
            n_degenerate += 1
            continue
        aucs[b] = roc_auc_score(y_b, s[idx])
    valid = aucs[~np.isnan(aucs)]
    ci_low, ci_high = np.percentile(valid, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return {
        "auc_point": auc_point,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "B": int(B),
        "block_len": int(expected_block_len),
        "seed": int(seed),
        "n": int(n),
        "n_degenerate": int(n_degenerate),
    }


__all__ = ["stationary_bootstrap_auc_ci", "stationary_bootstrap_indices"]
