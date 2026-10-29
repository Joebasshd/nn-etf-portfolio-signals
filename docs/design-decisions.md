# Capstone design decisions (agreed with Prof Blanco)

Running log of methodology choices confirmed with the supervisor. Update when a decision changes. Anything not yet agreed sits in the DRAFT section at the end and is marked `# DRAFT` in `config.yaml`.

## Data
- Tradable: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, QQQ, IWM. Benchmark only: SPY. XLRE and XLC excluded (short history).
- Daily, Jan 2000 – Jun 2026. Yahoo Finance (adjusted prices); FRED for DTB3, VIXCLS, T10Y3M (lagged 1 day).
- Reproducible runs use the cached `data/raw/prices.parquet` and `data/raw/macro.parquet` written by `src/data.py`.
- IWM enters May 2001, once 252 days of history exist for its longest indicators.
- 31 features in five families: trend, momentum, volatility, volume, market context.

## Target and validation
- Label = 1 if return from close t+1 (execution) to close t+6 is positive. Label aligned with execution (supervisor point).
- Expanding walk-forward, refit annually, test 2008–2026; last 12 months of each training window = validation.
- 11-day gap (1 execution lag + 5 horizon + 5 embargo) between train / validation / test.
- Scaling, feature selection and tuning redone inside every window.

## Models: identical inputs (supervisor feedback on the proposal, point 3, Oct 2026)
Two stages in every window:
1. **Selection only**, on all 31 features: LASSO (L1 logistic, C by validation log loss) keeps its non-zero coefficients; LightGBM (fixed settings, early stopping) keeps its top 15 by gain importance.
2. **Forecasting**, every model on the same union of the two selections (15–30 features per window):
   - Linear model: ridge (L2) logistic regression on the union, C re-tuned by validation log loss. It keeps every union feature. Reported as "Linear (L1 selection, ridge on union)". Refitting with L1 was rejected because it re-zeroed on average 9 of 20 union features and picked up LightGBM's features in only 2 of 19 windows; it is kept as a robustness check.
   - LightGBM: same fixed settings, early stopping on validation, no further selection.
   - LSTM: 1 layer, 32 units, dropout 0.2, 20-day sequences, early stopping on validation.
- Ridge C grid must be widened before the specification is fixed (supervisor, Oct 2026): C = 0.002 was the lower grid edge in 12 of 19 windows and 0.1 the upper edge in 3. Proposed grid in the DRAFT section.
- H4 framed as a null: LSTM does not significantly outperform LASSO/GBM net of costs (approved).

## Portfolio
- Long/flat only (approved as main specification).
- Main signal: s = min(1, max(0, (p − b) / (p90 − b))), b = training-window base rate, p90 = 90th percentile of **validation** forecasts (never the test window; supervisor feedback on the proposal, point 2).
- Floor: if p90 − b < 1 percentage point, stay flat for that window (supervisor request, Oct 2026).
- Sensitivity checks: base-rate centred (p − b)/(1 − b); original 2p − 1.
- Inverse-volatility base weights, 10% vol target (63-day), no leverage, 25% cap per ETF.
- Weekly rebalance at next day's close; 2-percentage-point no-trade band.

## Benchmarks, costs, tests
- SPY buy-and-hold; equal-weight (monthly); vol-targeted SPY; SMA 50/200 rule; Always-on (same vol-scaled weights, all signals = 1). Always-on shows whether trades come from the model.
- 5 bps per side base case (approved); sensitivity 0/2/5/10 bps; break-even cost.
- Stationary block bootstrap on Sharpe differences (10,000 resamples, ~20-day blocks).
- **Exposure-matched placebo (supervisor request, Oct 2026).** The LSTM is invested only about 23% of the time, so its Sharpe could come from being out of the market in bad periods rather than timing. For each of 1,000 draws, circularly shift each ETF's actual weekly signal series by a random number of weeks (uniform between 52 and W − 52). This keeps exposure, position sizes and holding lengths, but breaks the timing. Run each draw through the same backtest and costs. Report the actual Sharpe, the placebo mean and 5th/95th percentiles, and p = (1 + number of draws with Sharpe ≥ actual) / (N + 1). Run it for all three models.

## Indicator stability (supervisor request)
- Selection frequency per indicator; average Jaccard overlap between consecutive windows; average share of selected indicators by family.

## DRAFT: awaiting team or supervisor confirmation
| Topic | Draft |
|---|---|
| Ridge C grid | {0.00001, 0.00003, 0.0001, …, 1.0, 3.0, 10.0} (half-decade steps). Widened twice: first because 0.002 was the lower edge in 12 of 19 windows and 0.1 the upper edge in 3; then because the `ridge-widegrid` run still chose 0.0001 (the lower edge) in 11 windows and 1.0 in 2. If the strongest penalty still wins, report that the linear model finds no usable signal rather than widening again |
| LASSO selection grid | L1 logistic (liblinear), C in {0.002, 0.005, 0.01, 0.05, 0.1}, chosen by validation log loss; a C that zeroes every coefficient is skipped, and if all do, the weakest penalty ×10 is used |
| Final fit | Training block only; validation is for tuning, early stopping and p90 |
| `b` and `p90` | Taken from the forecasting fits: `p90` is the 90th percentile of each forecasting model's validation forecasts |
| LightGBM settings | Fixed, same for the selection and forecasting fits: num_leaves 15, learning rate 0.02, min_child_samples 200, subsample 0.8 (every round), colsample 0.8, lambda 1.0; up to 2,000 trees with early stopping after 100 rounds on validation log loss |
| LSTM training | Dropout 0.2 after the LSTM layer, no recurrent dropout; Adam 0.001, batch 512, up to 25 epochs, early stopping with patience 3, best weights restored; several seeds (median and range) — number of seeds to confirm with the supervisor |
| Calibration | Analysis only (Brier score, calibration table), not used for position sizing |
| No-trade band | Per ETF weight: an ETF trades only if its target differs from its current weight by at least 2 points |
| Benchmark costs | Same cost through the same backtest; buy-and-hold pays only the first purchase; the other benchmarks use the same no-trade band |
| Significance | 5%, one-sided; deflated-Sharpe trials = 12 (3 models × 3 signals + L1-refit and other checks) — asked the supervisor |
| Sub-periods | GFC 2008-09-01 to 2009-06-30; COVID 2020-02-01 to 2020-06-30; rate shock 2022-01-01 to 2022-10-31; break on 2018-09-24 (GICS change, XLC launch) — asked the supervisor |
| Extension | One pre-specified extension (21-day horizon or a fixed smaller indicator set) — asked the supervisor |
