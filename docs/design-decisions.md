# Capstone design decisions (agreed with Prof Blanco)

Running log of methodology choices confirmed with the supervisor. Update when a decision changes.

## Data
- Tradable: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, QQQ, IWM. Benchmark only: SPY. XLRE and XLC excluded (short history).
- Daily, Jan 2000 – Jun 2026. Yahoo Finance (adjusted prices); FRED for DTB3, VIXCLS, T10Y3M (lagged 1 day).
- IWM enters May 2001, once 252 days of history exist for its longest indicators.
- 31 features in five families: trend, momentum, volatility, volume, market context.

## Target and validation
- Label = 1 if return from close t+1 (execution) to close t+6 is positive. Label aligned with execution (supervisor point).
- Expanding walk-forward, refit annually, test 2008–2026; last 12 months of each training window = validation.
- 11-day gap (1 execution lag + 5 horizon + 5 embargo) between train / validation / test.
- Scaling, feature selection and tuning redone inside every window.

## Models
- LASSO (L1 logistic), LightGBM (top 15 features), LSTM (union of both selections, 20-day sequences).
- H4 framed as a null: LSTM does not significantly outperform LASSO/GBM net of costs (approved).

## Portfolio
- Long/flat only (approved as main specification).
- Main signal: s = min(1, max(0, (p − b) / (p90 − b))), b = training-window base rate, p90 = 90th percentile of validation forecasts.
- Floor: if p90 − b < 1 percentage point, stay flat for that window (supervisor request, Oct 2026).
- Sensitivity checks: base-rate centred (p − b)/(1 − b); original 2p − 1.
- Inverse-volatility base weights, 10% vol target (63-day), no leverage, 25% cap per ETF.
- Weekly rebalance at next day's close; 2-percentage-point no-trade band.

## Benchmarks, costs, tests
- SPY buy-and-hold; equal-weight (monthly); vol-targeted SPY; SMA 50/200 rule; Always-on (same vol-scaled weights, all signals = 1). Always-on shows whether trades come from the model.
- 5 bps per side base case (approved); sensitivity 0/2/5/10 bps; break-even cost.
- Stationary block bootstrap on Sharpe differences (10,000 resamples, ~20-day blocks).

## Indicator stability (supervisor request)
- Selection frequency per indicator; average Jaccard overlap between consecutive windows; average share of selected indicators by family.

## DRAFT: decisions before Step 3 (team to confirm; not yet agreed with Prof Blanco)
Drafted Oct 2026 from the defaults in `docs/bridge-plan.md` and the prototype (`notebooks/prototype/capstone_signal_to_portfolio.ipynb`). Move each line into its section above once agreed; mention the substantive ones to Prof Blanco.

### Models
- Final model is fitted on the training block only, not refitted on train + validation. Validation is used for tuning, early stopping and `p90`, so `p90` stays out-of-sample.
- LASSO: L1 logistic regression (liblinear), `C` in {0.002, 0.005, 0.01, 0.05, 0.1}, chosen by validation log loss. A `C` that zeroes every coefficient is skipped; if all do, the weakest penalty ×10 is used.
- LightGBM: fixed settings, no grid: learning rate 0.02, 15 leaves, min 200 rows per leaf, row and column subsampling 0.8, L2 penalty 1.0, up to 2,000 trees with early stopping after 100 rounds on validation log loss. Rank by gain importance, keep the top 15, refit on those.
- LSTM: 1 layer, 32 units, dropout 0.2 after the LSTM layer (no recurrent dropout), 20-day sequences, Adam (learning rate 0.001), batch 512, up to 25 epochs, early stopping after 3 epochs without improvement in validation loss, best weights restored.
- Calibration (Platt or isotonic) is analysis only: Brier score and calibration table. Raw probabilities drive the signal.

### Portfolio and benchmarks
- The 2-point no-trade band applies per ETF weight: an ETF is traded only if its target differs from its current weight by at least 2 points.
- Benchmarks pay the same cost per side through the same backtest. SPY buy-and-hold pays only the initial purchase; the other benchmarks use the same no-trade band.

### Statistics and sub-periods
- Significance level 5%, one-sided (H4: LSTM better than LASSO / LightGBM; each model better than always-on).
- Deflated Sharpe ratio: 9 trials (3 models × 3 signal versions).
- Crisis windows: GFC 2008-09-01 to 2009-06-30; COVID 2020-02-01 to 2020-06-30; 2022 rate shock 2022-01-01 to 2022-10-31.
- 2018 break: 2018-09-24, the GICS reclassification that created Communication Services (XLC launch) and moved holdings out of XLK and XLY.
