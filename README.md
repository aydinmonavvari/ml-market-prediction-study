# ml-market-prediction-study

![CI](https://github.com/aydinmonavvari/ml-market-prediction-study/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 1 · Short description

A leakage-free, walk-forward machine-learning study asking whether historical daily
equity data contains *statistically significant incremental out-of-sample information*
about next-day index direction — evaluated against honest baselines, formal significance
tests, a label-permutation control, and a cost-adjusted backtest.

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
  that cannot leak is scientifically more informative than an unsubstantiated positive.

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
   labels must give AUC ≈ 0.5).
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
| Class balance | Up days ≈ 55.2% (the majority baseline exploits this) |

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
| Undetected leakage inflating AUC | label-permutation control: shuffled labels must produce AUC ≈ 0.5 (actual: 0.474–0.499) |

**Walk-forward geometry:** 8 folds, 63 test days per fold, minimum 80 training days,
1-day embargo (gap) between train and test blocks.

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
- **Label-permutation control** (one run): shuffled labels must yield AUC ≈ 0.5.
- **Cost-adjusted long/flat backtest** on the holdout: 10 bps per position switch,
  decisions at the close applied to next-day returns; reported against buy-and-hold.

## 12 · Results

All numbers below are actual outputs of the committed run
(`reports/metrics.csv`, `reports/mcnemar.csv`, `reports/backtest.csv`,
[`reports/summary.md`](reports/summary.md)).

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

**Cost-adjusted long/flat vs buy-and-hold (holdout):** buy-and-hold returned **+92.7%**;
every model's strategy underperformed after 10 bps costs — random forest +59.5%
(173 switches, 17.3% total cost), logistic regression +70.1%, gradient boosting and MLP
*negative* (−8.2% / −8.4%, 305/332 switches, ~30–33% eaten by costs).

**CV out-of-fold (model-selection view):** all models' balanced accuracy 0.497–0.510,
AUC 0.479–0.495 — no fold-level signal either.

Figures (generated from the actual run): [`equity_curves.png`](figures/equity_curves.png)
(strategy vs buy-and-hold), [`roc_pr_curves.png`](figures/roc_pr_curves.png),
[`walkforward_folds.png`](figures/walkforward_folds.png) (fold geometry),
[`feature_importance.png`](figures/feature_importance.png) (RF/XGB importances —
dominated by recent-return lags and volatility, with no stable ranking across folds),
[`cv_fold_scores.png`](figures/cv_fold_scores.png).

## 13 · Interpretation

1. **No exploitable directional signal was found — and the design makes that finding
   trustworthy.** Every safeguard (walk-forward, embargo, pipeline-internal scaling,
   permutation control) is in place, and the models still cannot beat "always predict
   up" (which wins simply because up-days are 54.6% of the sample).
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
- **One permutation run.** The control is a sanity check, not a full permutation test
  distribution.
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

# 4) verify: lint + 30 offline unit tests (incl. the leakage test)
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
│   ├── plots.py         # equity curves, ROC/PR, fold diagram, importances
│   └── pipeline.py      # end-to-end study orchestration + summary writer
├── tests/               # 30 offline tests incl. the temporal-leakage test
├── notebooks/           # feature/label EDA
├── scripts/             # download_data.py, run_study.py
├── configs/default.yaml
├── data/raw · data/processed   # git-ignored caches (.gitkeep tracked)
├── reports/             # metrics/backtest/mcnemar CSVs, summary.md (generated)
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
