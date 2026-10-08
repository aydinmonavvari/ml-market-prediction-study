# ml-market-prediction-study

![CI](https://github.com/aydinmonavvari/ml-market-prediction-study/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 1 · Short description

A leakage-free, walk-forward machine-learning study asking whether historical daily
equity data contains *statistically significant incremental out-of-sample information*
about next-day index direction — evaluated against honest baselines, formal significance
tests, a label-permutation control, a leakage mutation test, and a cost-adjusted backtest.

*Educational research study. Not investment advice. The headline finding is negative —
and that is the point.*

## 2 · Research question

> Does machine learning provide statistically significant incremental out-of-sample
> predictive information for next-day equity index direction beyond simple baselines,
> under a strict walk-forward, leakage-free framework?

## 3 · Motivation

"AI predicts the stock market" is one of the most persistent claims in applied ML — and
one of the least often tested properly. Designs that shuffle time-series rows, scale
features across the full sample, tune hyperparameters on the test set, or compare
against no baseline at all routinely "discover" signal that evaporates in real use. This
project is built as the counterexample: a study engineered so that every classical
failure mode is prevented *by construction and by unit test*, and where a negative
result is a publishable, expected outcome rather than a disappointment.

## 4 · Why this matters

- **For finance:** the honest answer to the predictive-signal question under near-form
  market efficiency is foundational knowledge for anyone moving from financial
  management into quantitative research — knowing *why* naive baselines are hard to beat
  is more valuable than a leaderboard score.
- **For machine learning:** temporal data break standard cross-validation. This repo
  demonstrates walk-forward validation with embargo, pipeline-internal scaling,
  McNemar paired testing, Holm correction, and permutation controls — the toolkit that
  transfers directly to the later deep-learning study in this portfolio.
- **Negative results are results.** Documenting *no significant signal* under a design
  engineered and unit-tested against the classical leakage failure modes is scientifically
  more informative than an unsubstantiated positive.

## 5 · Methodology

1. **Data** — SPY daily adjusted prices (2010 → 2026) from Yahoo Finance and the daily
   effective federal funds rate (DFF) from FRED; cached locally, validated.
2. **Features** — lagged log returns (1–21 days), rolling volatility (5d/21d),
   SMA 5/20 ratio, RSI(14), MACD(12,26,9), 21-day volume z-score, day-of-week, and the
   daily fed-funds change. All features at row *t* use information dated ≤ *t*.
3. **Target** — next-day direction of SPY (binary), aligned by a single explicit shift.
4. **Validation** — 8-fold expanding-window walk-forward with a 1-day embargo between
   train and test blocks; the chronologically last 20% of rows form a holdout set
   evaluated exactly once.
5. **Hyperparameters** — small fixed grids searched *inside* the walk-forward CV only;
   the winning combination per model is refit on the CV region and applied to the
   holdout.
6. **Significance** — McNemar tests of each model vs the majority baseline on the
   holdout, with Holm (1979) step-down correction; label-permutation control (shuffled
   labels must give AUC ≈ 0.5); leakage mutation test (an injected leak must inflate the
   walk-forward AUC); stationary-bootstrap confidence intervals for holdout ROC-AUC.
7. **Economic evaluation** — next-day long/flat strategy vs buy-and-hold on the holdout
   only, with 10 bps cost per position switch.

## 6 · Dataset

| Property | Value (actual run) |
| --- | --- |
| Instrument | SPY (SPDR S&P 500 ETF) as S&P 500 proxy |
| Price source | Yahoo Finance daily adjusted bars via `yfinance` |
| Macro covariate | FRED DFF (effective federal funds rate, daily) |
| Window | 2010-01-04 → 2026-10-07 download; 2010-02-04 → 2026-10-06 usable after warm-up |
| Usable rows | 4,193 (features + aligned target) |
| Class balance | Up days (exact, `reports/metrics.csv`): **55.23% full sample** (2,316/4,193); 54.95% CV region (1,843/3,354); **56.38% holdout** (473/839); 50.20% over the 504 CV-OOF rows (the majority baseline exploits the holdout's 56.38%) |

## 7 · Data sources

- Yahoo Finance (via `yfinance`) — delayed, adjusted daily bars; used for research only.
- FRED (Federal Reserve Bank of St. Louis) — DFF series, public CSV endpoint.
Both cached under `data/raw/` (git-ignored) so the study reruns offline.

## 8 · Architecture

```
yfinance + FRED ──► download/cache ──► validation
        │
        ▼
feature engineering (single shift-discipline module; unit-tested leakage test)
        │
        ▼
chronological holdout split (last 20%)  ──────────────► holdout (evaluated once)
        │
        ▼
8-fold expanding walk-forward CV (embargo = 1 day)
   ├─ per-model small grid search (balanced accuracy)
   └─ out-of-fold predictions
        │
        ▼
refit winners on CV region ──► single holdout evaluation
        │
        ▼
metrics · McNemar + Holm · permutation control · cost-adjusted backtest
        │
        ▼
reports/*.csv · summary.md · figures/
```

## 9 · Experimental design

**Leakage checklist** — each classical failure mode and its prevention:

| Failure mode | Prevention |
| --- | --- |
| Future information in features | single shift-discipline module; unit test mutates future rows and asserts features at earlier rows are bit-identical |
| Scaler/encoder fitted on full sample | scaling models carry their scaler inside a sklearn `Pipeline`, fit per training fold (unit-tested) |
| Shuffled splits | custom expanding walk-forward splitter; unit tests assert monotonic, non-overlapping, embargo-respecting folds |
| Hyperparameter tuning on test data | grids searched inside walk-forward CV only; holdout evaluated once, after model selection |
| Selecting on many correlated metrics | model selection uses one metric (balanced accuracy); Holm correction across McNemar comparisons; Bonferroni-style caution in interpretation |
| Undetected leakage inflating AUC | three complementary controls with distinct roles: (i) **label-permutation control** (one run — with labels destroyed by permutation the pipeline must return chance-level AUC ≈ 0.5, actual 0.474–0.499; a sanity check that can catch certain target-leakage bugs, i.e. features derived from the target, but **not** a proof that no leakage exists); (ii) **leakage mutation test** (`tests/test_mutation.py` — injects a known leak, a feature equal to the future label, and asserts the walk-forward AUC inflates to ≈ 1.0 while the clean pipeline stays ≈ 0.5, demonstrating the evaluation design *does* detect leaks when present); (iii) **scaler-isolation tests** (`tests/test_models.py` — scaling statistics are fit on training folds only). Combined: the design detects injected leaks, and the permutation control shows no evidence of leakage in the shipped pipeline; neither constitutes a formal proof of absence |

**Walk-forward geometry:** 8 folds, 63 test days per fold, minimum 756 training days
(≈ three trading years — the production setting in `config.py` / `configs/default.yaml`,
enforced by the splitter's length validation; the first fold therefore trains on 2,849 rows
≈ 2,850), 1-day embargo (gap) between train and test blocks.

## 10 · Models

| Model | Configuration |
| --- | --- |
| Majority baseline | `DummyClassifier(strategy="most_frequent")` |
| Logistic regression | `StandardScaler` + `LogisticRegression` (C ∈ {0.01, 0.1, 1.0}) inside a Pipeline |
| Random forest | 300 trees, `min_samples_leaf=5`, max depth ∈ {4, 8} |
| Gradient boosting | sklearn, learning rate ∈ {0.05, 0.1}, depth ∈ {2, 3} |
| XGBoost | learning rate ∈ {0.05, 0.1}, depth ∈ {2, 3} |
| MLP | 1 hidden layer, early stopping, alpha ∈ {1e-4, 1e-2} |

All seeds pinned (`random_state = 42`).

## 11 · Evaluation metrics

- **Accuracy / balanced accuracy / ROC-AUC / PR-AUC** (holdout = the honest estimate;
  CV out-of-fold = model-selection view, mildly optimistic).
- **McNemar's test** vs the majority baseline (exact binomial, two-sided) with **Holm
  step-down correction** across the five non-baseline models.
- **Label-permutation control** (one run): a *sanity check* — with labels destroyed by
  permutation, the refit pipeline must yield AUC ≈ 0.5; it can catch certain
  target-leakage bugs but is not a proof that no leakage exists (see the mutation test
  above for the complementary positive control).
- **Stationary (Politis–Romano) bootstrap CI** for the holdout ROC-AUC of every model:
  B = 2,000 resamples, expected block length 21 trading days, seed 42, 95% percentile
  intervals. Daily observations are serially dependent, so an IID bootstrap would be
  invalid; the block resampling preserves dependence at lags below ~1 month.
- **Cost-adjusted long/flat backtest** on the holdout: 10 bps per position switch,
  decisions at the close applied to next-day returns; reported against buy-and-hold.

## 12 · Results

All numbers below are actual outputs of the committed run
(`reports/metrics.csv`, `reports/mcnemar.csv`, `reports/backtest.csv`,
`reports/auc_bootstrap.csv`, [`reports/summary.md`](reports/summary.md)).

**Class balance by split (exact, persisted in `reports/metrics.csv`):** up-days are
**55.23% of the full sample** (2,316/4,193), **56.38% of the holdout** (473/839) and
50.20% of the 504 CV-OOF rows — every base rate quoted anywhere in this README refers to
one of these labeled splits.

**Holdout metrics (single evaluation, n = 839):**

| Model | Accuracy | Balanced acc. | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- |
| Majority baseline | **0.564** | 0.500 | 0.500 | 0.564 |
| Logistic regression | 0.558 | 0.497 | 0.486 | 0.579 |
| Random forest | 0.546 | 0.513 | **0.502** | 0.565 |
| Gradient boosting | 0.493 | 0.484 | 0.496 | 0.558 |
| XGBoost | 0.510 | 0.496 | 0.477 | 0.538 |
| MLP | 0.517 | 0.496 | 0.498 | 0.565 |

**Headline verdict (auto-generated, direction-aware):** no model beats the majority
baseline on the holdout — no significant Holm-adjusted McNemar *improvement*, best
ROC-AUC 0.5022 (indistinguishable from chance), and gradient boosting made significantly
*fewer* correct calls than the baseline (Holm-adjusted p = 0.011, i.e. significantly
worse). Under this leakage-free design the study finds **little to no statistically
significant incremental out-of-sample signal** beyond simple baselines — the honest,
expected outcome under near-form market efficiency.

**McNemar vs baseline (holdout, Holm-adjusted):**

| Model | Model right only | Baseline right only | p | Holm p |
| --- | --- | --- | --- | --- |
| Logistic regression | 6 | 11 | 0.332 | 0.652 |
| Random forest | 94 | 109 | 0.326 | 0.652 |
| Gradient boosting | 151 | 210 | 0.002 | **0.011 (worse)** |
| XGBoost | 141 | 186 | 0.015 | 0.059 |
| MLP | 119 | 158 | 0.022 | 0.067 |

**Label-permutation control:** shuffled-label AUCs 0.474–0.499 across models — the
pipeline shows no evidence of leakage (a leaking pipeline would show inflated AUC here).
This is a one-run sanity check, not a proof; the complementary leakage mutation test
(`tests/test_mutation.py`) confirms the same walk-forward evaluation *does* inflate to
AUC ≈ 1.0 when a known leak is injected.

**Uncertainty quantification (stationary bootstrap, holdout ROC-AUC):** daily
observations are serially dependent, so an IID bootstrap would be invalid; 95%
percentile CIs use the Politis–Romano stationary bootstrap (B = 2,000, expected block
length 21 trading days, seed 42). **The CIs include 0.5 for every model** — e.g. random
forest 0.502 [0.463, 0.545], logistic regression 0.486 [0.444, 0.527], XGBoost 0.476
[0.434, 0.521] — consistent with no directional signal. Full table:
[`reports/auc_bootstrap.csv`](reports/auc_bootstrap.csv).

**Cost-adjusted long/flat vs buy-and-hold (holdout):** buy-and-hold returned **+92.7%**;
every model's strategy underperformed after 10 bps costs — random forest +59.5%
(173 switches, 17.3% total cost), logistic regression +70.1%, gradient boosting and MLP
*negative* (−8.2% / −8.4%, 305/332 switches, ~30–33% eaten by costs).

**CV out-of-fold (model-selection view):** all models' balanced accuracy 0.497–0.510,
AUC 0.479–0.495 — no fold-level signal either. Note the coverage: CV-OOF metrics are
computed on the **504 rows of the 8×63-day walk-forward test blocks only**, not on all
3,354 CV rows — the earliest CV rows cannot be scored out-of-fold (the first fold needs
≥ 756 training days) and the 1-day embargo gaps are legitimately uncovered.

Figures (generated from the actual run): [`equity_curves.png`](figures/equity_curves.png)
(strategy vs buy-and-hold), [`roc_pr_curves.png`](figures/roc_pr_curves.png),
[`walkforward_folds.png`](figures/walkforward_folds.png) (fold geometry),
[`feature_importance.png`](figures/feature_importance.png) (RF/XGB importances —
dominated by recent-return lags and volatility, with no stable ranking across folds),
[`cv_fold_scores.png`](figures/cv_fold_scores.png).

## 13 · Interpretation

1. **No exploitable directional signal was found — and the design makes that finding
   trustworthy.** Every safeguard (walk-forward, embargo, pipeline-internal scaling,
   permutation control, leakage mutation test) is in place, and the models still cannot
   beat "always predict up" (which wins on the holdout simply because up-days are 56.4%
   of the holdout sample — 473/839; 55.23% of the full sample).
2. **The one "significant" result is a significant *failure*.** Gradient boosting's
   Holm-adjusted p = 0.011 reflects *worse*-than-baseline accuracy (0.493 vs 0.564) —
   the flexible model learned patterns that anti-correlated with the holdout period.
   This is a concrete demonstration that flexible models on weak signal can be
   confidently wrong out of sample.
3. **Transaction costs turn "almost flat" performance into clear underperformance.**
   RF traded 173 times (17.3% cost drag) for a strategy 33 pp below buy-and-hold;
   high-frequency switching on noise is precisely how costs destroy marginal signal.
4. **Consistency with the literature.** The result aligns with the efficient-market
   hypothesis tradition (Fama 1970) and with the modern ML-finance literature's emphasis
   (Lopez de Prado 2018) that most claimed edges dissolve under correct validation.
5. **What this study cannot rule out:** signal at other frequencies (intraday, weekly),
   in cross-sectional rather than single-asset settings, under regime-conditional
   features, or with microstructure data — all future work, all harder than this setup.

## 14 · Limitations

- **Single asset, single period, single seed.** SPY 2010–2026 is one bull-dominated
  path; results do not generalize to other instruments or regimes.
- **Daily closes only.** No intraday fills, no slippage modeling; the backtest is
  illustrative, not an executable strategy.
- **Small hyperparameter grids.** Deliberate (anti-overfitting-the-CV), but models are
  not exhaustively tuned — the study tests *whether simple, properly validated models
  find signal*, not whether any possible configuration could.
- **Linear technical features.** No order-book, options-implied, or alternative data;
  features are standard daily indicators available at the close.
- **One permutation run.** The label-permutation control is a sanity check, not a full
  permutation test distribution — and no single-run control can *prove* the absence of
  leakage. The complementary leakage mutation test demonstrates the evaluation design
  detects injected leaks; together they bound what the negative result means.
- **Bootstrap CI assumptions.** The stationary-bootstrap intervals assume approximate
  stationarity of the holdout window and weak dependence beyond the 21-day expected block
  length; they quantify sampling uncertainty on this one path only, and interval widths
  can be sensitive to the block-length choice.
- **Multiple metrics inspected.** Even with Holm correction on McNemar and one selection
  metric, reading many tables invites selection bias; interpretations are written with
  that in mind.

## 15 · Reproducibility

```bash
# 1) environment (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# 2) data (network, then cached)
python scripts/download_data.py

# 3) full study (~5 min on 4 cores; deterministic, seed = 42)
python scripts/run_study.py

# 4) verify: lint + 34 offline unit tests (incl. the leakage and mutation tests)
ruff check .
pytest -q
```

Every number and figure in this README regenerates exactly from steps 2–3.

## 16 · Installation

```bash
git clone https://github.com/aydinmonavvari/ml-market-prediction-study.git
cd ml-market-prediction-study
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

## 17 · Usage

```bash
python scripts/download_data.py   # fetch + cache SPY and FRED DFF
python scripts/run_study.py       # full walk-forward study -> reports/ + figures/
pytest -q                         # offline test suite
```

As a library:

```python
from market_prediction_study.features import build_model_matrices
from market_prediction_study.data import load_dataset
from market_prediction_study.config import StudyConfig

raw = load_dataset(StudyConfig())          # cached SPY + DFF
mats = build_model_matrices(raw)           # X, y with shift-discipline applied
print(mats.X.shape, round(float(mats.y_up.mean()), 4))  # (4193, 19) 0.5523
```

## 18 · Example

Actual holdout behavior of the best model (random forest), from
[`reports/holdout_predictions.csv`](reports/holdout_predictions.csv):

- ROC-AUC **0.502** — the model's score ordering carries essentially no directional
  information about the next day;
- accuracy **0.546** vs the majority baseline's **0.564** — you would have done better
  by always predicting "up";
- trading its scores long/flat with 10 bps costs earned **+59.5%** while buy-and-hold
  earned **+92.7%** on the same 839 days.

This is what an honest "no signal" result looks like in practice.

## 19 · Project structure

```
ml-market-prediction-study/
├── README.md
├── LICENSE · CITATION.cff · pyproject.toml · requirements.txt
├── .python-version · .gitignore
├── src/market_prediction_study/
│   ├── config.py        # dataclass config (geometry, costs, seed, paths)
│   ├── data.py          # yfinance + FRED download, caching, validation
│   ├── features.py      # shift-discipline features (RSI, MACD, rolling stats)
│   ├── validation.py    # expanding walk-forward + embargo, holdout split
│   ├── models.py        # seeds-pinned estimators, grids, fit/predict helpers
│   ├── evaluation.py    # metrics, exact McNemar, Holm, cost-adjusted backtest
│   ├── bootstrap.py     # stationary (Politis-Romano) bootstrap CI for holdout AUC
│   ├── plots.py         # equity curves, ROC/PR, fold diagram, importances
│   └── pipeline.py      # end-to-end study orchestration + summary writer
├── tests/               # 34 offline tests incl. the temporal-leakage + mutation tests
├── notebooks/           # feature/label EDA
├── scripts/             # download_data.py, run_study.py
├── configs/default.yaml
├── data/raw · data/processed   # git-ignored caches (.gitkeep tracked)
├── reports/             # metrics/backtest/mcnemar/auc_bootstrap CSVs, summary.md (generated)
├── figures/             # 5 generated figures (committed)
├── docs/research_report.md
└── .github/workflows/ci.yml
```

## 20 · Future work

- Purged k-fold cross-validation with overlapping-label purging (Lopez de Prado 2018).
- Triple-barrier labeling and meta-labeling for position sizing.
- Weekly/monthly horizons and cross-sectional equity universes (survivorship-safe
  point-in-time constituents).
- Regime features (VIX, yield-curve slope) and volatility-targeted evaluation.
- Full permutation-test distributions rather than a single control run.
- Deep-learning counterpart with identical validation (see the DL time-series project
  in this portfolio).

## 21 · Citation

If you use this work, please cite (see also [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{monavvari2026mlmarketprediction,
  author  = {Monavvari, Aydin},
  title   = {ml-market-prediction-study: a leakage-free walk-forward study of ML signal in daily index direction},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/aydinmonavvari/ml-market-prediction-study}
}
```

## 22 · License

MIT — see [`LICENSE`](LICENSE).

## 23 · Acknowledgments

Market data via Yahoo Finance (`yfinance`); macro data via FRED, Federal Reserve Bank of
St. Louis. Built with pandas, scikit-learn, XGBoost and matplotlib.
