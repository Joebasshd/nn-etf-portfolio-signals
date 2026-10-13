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
