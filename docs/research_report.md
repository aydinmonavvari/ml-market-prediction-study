# Research Report — ML Market Prediction Study

**Does machine learning provide incremental out-of-sample predictive information for daily index direction? A leakage-free walk-forward study**

*Author: Aydin Monavvari — educational research project (not investment advice).*
*Run date: 2026-10-08. All numbers are actual outputs of the committed pipeline.*

---

## Abstract

This study tests whether machine-learning models extract statistically significant incremental out-of-sample information about next-day direction of the S&P 500 (via SPY, 2010–2026, 4,193 usable daily observations) from a standard set of technical and macro features, under a validation design engineered to preclude every common source of temporal leakage. Six models (majority baseline, logistic regression, random forest, gradient boosting, XGBoost, MLP) are compared in an 8-fold expanding-window walk-forward scheme with a 1-day embargo, small fixed hyperparameter grids searched inside the cross-validation only, and a chronologically last holdout segment (n = 839) evaluated exactly once. Model differences are tested with exact McNemar tests against the majority baseline with Holm correction; a label-permutation control verifies the pipeline cannot leak; a cost-adjusted long/flat backtest (10 bps per switch) provides the economic view. **Finding: no model significantly outperformed the majority baseline.** The best holdout ROC-AUC was 0.502 (random forest), statistically indistinguishable from chance; gradient boosting was significantly *worse* than the baseline (Holm-adjusted p = 0.011); and every model's long/flat strategy underperformed buy-and-hold after costs (best: +59.5% vs +92.7%). The negative result — obtained under a design that provably cannot leak — is consistent with near-form market efficiency and constitutes the study's primary contribution.

## Introduction

Claims that machine learning "predicts the market" circulate widely, but the overwhelming majority rest on flawed validation: shuffled splits of temporal data, scalers fitted on full samples, hyperparameters tuned on test sets, and comparisons without baselines. Each flaw manufactures phantom signal. Rather than adding another positive claim, this project asks the prior question — *does any signal survive when none of these flaws are possible?* — and engineers the study so the answer is trustworthy: leakage prevention is enforced in code and verified by dedicated unit tests, model selection is separated from final evaluation, and both statistical and economic criteria are reported.

The study is the third project in a research portfolio progressing from descriptive analytics through macro forecasting toward deep learning and NLP. It establishes the walk-forward toolkit and the honest-reporting norms reused by all later predictive work in the portfolio.

## Research Question

> Does machine learning provide statistically significant incremental out-of-sample predictive information for next-day equity index direction beyond simple baselines, under a strict walk-forward, leakage-free framework?

## Related Work

- **Fama, E. F. (1970).** "Efficient Capital Markets: A Review of Theory and Empirical Work." *Journal of Finance*, 25(2), 383–417. The efficiency benchmark against which predictability claims are measured.
- **Lopez de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley. Source of the purged/embargoed cross-validation philosophy and the overfitting-the-cross-validation critique motivating small grids.
- **Breiman, L. (2001).** "Random Forests." *Machine Learning*, 45(1), 5–32; **Chen, T., & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." *Proceedings of KDD '16*, 785–794. The tree ensembles studied.
- **McNemar, Q. (1947).** "Note on the sampling error of the difference between correlated proportions or percentages." *Psychometrika*, 12(2), 153–157. The paired-correctness test used for model-vs-baseline comparison.
- **Holm, S. (1979).** "A simple sequentially rejective multiple test procedure." *Scandinavian Journal of Statistics*, 6(2), 65–70. Family-wise error control across model comparisons.

## Data

| Property | Value (actual run) |
| --- | --- |
| Instrument | SPY (SPDR S&P 500 ETF) as an S&P 500 proxy; price-only index unavailable with dividends |
| Price data | Yahoo Finance daily adjusted bars, 2010-01-04 → 2026-10-07 |
| Macro covariate | FRED DFF (effective federal funds rate, daily change used as feature) |
| Usable sample | 4,193 rows (2010-02-04 → 2026-10-06) after feature warm-up |
| Features (19) | log-return lags {1,2,3,5,10,21}; volatility {5d, 21d}; SMA5/20 ratio; RSI(14); MACD(12,26,9) + signal + histogram; volume z-score (21d); day-of-week one-hot; DFF daily change |
| Target | Next-day direction of SPY (up ≈ 55.2% of days) |

Data are cached under `data/raw/` (git-ignored); the study is offline-reproducible from the cache. Yahoo/FRED usage terms respected; no credentials involved.

## Methodology

**Shift discipline.** All features are built in one module whose contract is: row *t* may depend only on information dated ≤ *t*; the target at row *t* is the direction of day *t+1*. A dedicated unit test mutates all rows strictly after *t* (prices ×1.41, volumes ×0.37, macro reset) and asserts (i) every feature column at rows ≤ *t* is bit-identical, (ii) perturbed features after *t* do change (the mutation is live), and (iii) the direction label at *t* flips when day *t+1* is forced to the opposite side of close[*t*].

**Validation.** 8-fold expanding-window walk-forward: 63 test days per fold, minimum 80 training days, 1-day embargo between each training block and its test block. The chronologically last 20% of rows are a holdout set evaluated once, after model selection.

**Model selection.** Small fixed grids per model (2–3 combinations; 16 total across models), scored by mean fold balanced accuracy inside the walk-forward CV — deliberately small to limit overfitting-the-CV (Lopez de Prado 2018). Winners are refit on the full CV region for holdout evaluation.

**Significance and controls.** Exact two-sided McNemar tests vs the majority baseline on holdout predictions, with Holm step-down correction across the five non-baseline models. Label-permutation control: refit with shuffled labels must produce AUC ≈ 0.5. Economic evaluation: next-day long/flat strategy from predicted scores (threshold 0.5), 10 bps per position switch, holdout only.

## Experimental Design

The design separates three layers of evidence: (i) CV out-of-fold performance (model-selection view; mildly optimistic because hyperparameters are chosen on it); (ii) the holdout (the single honest estimate); (iii) causal controls (permutation test, leakage unit tests) that validate the *pipeline itself*. Interpretation privileges the holdout and the controls. The majority baseline is treated as a first-class competitor: with 55.2% up-days, "always up" is the null that any claimed edge must beat — not merely tie on accuracy.

## Results

All numbers from the committed run (`reports/*.csv`, `reports/summary.md`).

**Holdout metrics (n = 839, evaluated once):**

| Model | Accuracy | Balanced acc. | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- |
| Majority baseline | **0.564** | 0.500 | 0.500 | 0.564 |
| Logistic regression | 0.558 | 0.497 | 0.486 | 0.579 |
| Random forest | 0.546 | 0.513 | **0.502** | 0.565 |
| Gradient boosting | 0.493 | 0.484 | 0.496 | 0.558 |
| XGBoost | 0.510 | 0.496 | 0.477 | 0.538 |
| MLP | 0.517 | 0.496 | 0.498 | 0.565 |

**McNemar vs baseline (Holm-adjusted):** no model shows a significant *improvement*; gradient boosting is significantly worse (model right 151 vs baseline right 210, raw p = 0.0022, Holm p = 0.011). XGBoost (Holm p = 0.059) and MLP (0.067) are marginally below the uncorrected threshold in the same (worse) direction.

**Label-permutation control:** shuffled-label AUCs 0.474–0.499 — consistent with a leakage-free pipeline.

**CV out-of-fold:** balanced accuracy 0.497–0.510, AUC 0.479–0.495 across models — no signal even in the optimistic selection view.

**Cost-adjusted backtest (holdout):** buy-and-hold +92.7%; long/flat: logistic +70.1%, random forest +59.5% (173 switches; 17.3% cost drag), XGBoost +17.5%, gradient boosting −8.2% and MLP −8.4% (305/332 switches; ~30–33% consumed by costs).

## Discussion

1. **The null result is the finding.** Under a design where leakage is precluded by construction and verified by tests and a permutation control, none of six models — including two gradient-boosting implementations — extracts next-day directional information from daily index data beyond the majority class. The best AUC, 0.502, is the shape of noise.
2. **The only significant difference is a significant degradation.** Gradient boosting *anti-correlated* with the holdout: it was confidently wrong. Flexible learners with weak signal can latch onto patterns that invert out of sample — a caution against equating low training loss with knowledge.
3. **Costs dominate marginal "signal".** Random forest switched positions 173 times in 839 days, paying 17.3% of principal in costs to underperform buy-and-hold by 33 percentage points. Any real-world edge must clear a cost hurdle that daily-frequency technical strategies rarely clear.
4. **Persistence beats cleverness.** The majority baseline's 0.564 accuracy is an artifact of the 55.2% up-day base rate. Several models *undercut* it by predicting the minority class too often — direction classifiers on daily equity data mostly trade off how often they dare disagree with persistence.
5. **Relation to prior work.** The result is consistent with the efficiency tradition (Fama 1970) and with the modern methodological literature's demonstration (Lopez de Prado 2018) that positive results largely evaporate under correct validation. It does not prove absence of signal everywhere — only that this feature set, these models, this frequency and this period contain no detectable exploitable signal under honest evaluation.

## Limitations

- **Single asset, single period, one seed.** SPY 2010–2026 is one bull-dominated sample; conclusions do not automatically extend to other markets, frequencies, or eras.
- **Daily closes, no microstructure.** No intraday information, order-book data, slippage, or borrow constraints; the backtest is illustrative.
- **Standard features only.** No options-implied quantities (e.g., implied volatility), no alternative data, no cross-sectional information.
- **Small grids, default-capacity models.** A deliberate anti-overfitting choice; the study bounds *these* models, not the entire model space.
- **Single permutation control run** rather than a full permutation distribution.
- **Holdout size.** n = 839 gives McNemar tests moderate power; truly tiny effects would not be detected.

## Conclusion

Within a leakage-free, embargoed walk-forward design verified by unit tests and a permutation control, machine-learning models provide no statistically significant incremental out-of-sample information about next-day S&P 500 direction beyond the majority baseline, and every cost-adjusted trading variant underperforms buy-and-hold. The negative result, obtained honestly, is more informative than an inflated positive one: it quantifies how far standard technical features and tree/linear/NN models fall short of exploitable daily predictability, and it validates an evaluation harness that later projects in this portfolio apply to volatility forecasting, credit risk, and deep-learning studies.

## Future Research

- Purged k-fold CV with overlapping-label purging and triple-barrier labeling (Lopez de Prado 2018).
- Cross-sectional universes with point-in-time constituents (removes single-path dependence).
- Volatility and range targets, which are known to be far more predictable than direction.
- Regime-conditional evaluation (VIX terciles, yield-curve states).
- Deep-learning sequence models with identical validation (portfolio Project 07).
- Full permutation-test distributions for every reported statistic.

## References

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.
- Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *Journal of Finance*, 25(2), 383–417.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.
