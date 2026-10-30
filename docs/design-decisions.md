# Capstone design decisions

Running log of the methodology. Decisions marked **(supervisor)** were agreed with Prof Blanco; decisions marked **(team)** were taken by the team, with the reason given, and are reported as such. Update this file whenever a decision changes, and keep `config.yaml`, `CLAUDE.md` and the report in line with it. Last updated 30 October 2026.

## Data
- Tradable: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, QQQ, IWM. Benchmark only: SPY. XLRE and XLC excluded (short history). **(supervisor)**
- Daily, Jan 2000 – Jun 2026. Yahoo Finance (adjusted prices); FRED for DTB3, VIXCLS, T10Y3M (lagged 1 day). **(supervisor)**
- Reproducible runs use the cached `data/raw/prices.parquet` and `data/raw/macro.parquet` written by `src/data.py`. **(team)**
- IWM enters May 2001, once 252 days of history exist for its longest indicators.
- 31 features in five families: trend, momentum, volatility, volume, market context.

## Target and validation
- Label = 1 if the return from close t+1 (execution) to close t+6 is positive, aligned with execution. **(supervisor)**
- Expanding walk-forward, refit annually, test 2008–2026; last 12 months of each training window = validation.
- 11-day gap (1 execution lag + 5 horizon + 5 embargo) between train / validation / test.
- Scaling, feature selection and tuning redone inside every window; nothing fitted sees the test year.
- Final fit on the training block only. Validation is used for tuning, early stopping and p90, which keeps p90 out of sample. **(team)**

## Models: identical inputs (supervisor feedback on the proposal, point 3)
Two stages in every window:
1. **Selection only**, on all 31 features: LASSO (L1 logistic, C by validation log loss) keeps its non-zero coefficients; LightGBM keeps its top 15 by gain importance.
2. **Forecasting**, every model on the same union of the two selections (15–30 features per window):
   - **Linear model:** ridge (L2) logistic regression on the union, C tuned by validation log loss, keeping every union feature. Reported as "Linear (L1 selection, ridge on union)". An L1 refit was rejected because it re-zeroed on average 9 of 20 union features and picked up LightGBM's features in only 2 of 19 windows, so the inputs would not really be identical; it is kept as a robustness check. **(team)**
   - **LightGBM:** fixed settings (num_leaves 15, learning rate 0.02, min_child_samples 200, subsample 0.8, colsample 0.8, lambda 1.0), early stopping after 100 rounds on validation, no further selection. **(team)**
   - **LSTM:** 1 layer, 32 units, dropout 0.2, 20-day sequences; Adam 0.001, batch 512, up to 25 epochs, early stopping with patience 3. **(supervisor for architecture; team for training settings)**
- **Ridge grid:** widened twice, to C = 0.00001–10, as planned (supervisor asked for the widening). No further widening (pre-committed). Ridge chooses maximum shrinkage, which amounts to predicting the base rate, in 7 of 19 windows, so the linear model is reported as **finding no usable signal**. Its Sharpe ratio moved from 0.45 to 0.18 between grids, so its portfolio results and comparisons are not used for conclusions.
- **LSTM seeds:** 10 seeds, 42–51 (five is the minimum; placebo p-values sit near 0.05). **(supervisor)** Each fit is seeded with `keras.utils.set_random_seed` and restores the global random state afterwards. Report the median and range across seeds, plus the seed-averaged forecast (mean probability across seeds for validation and test, then the usual b, p90 and floor) as a robustness check next to them, never instead of them. **(supervisor)**
- **Seeding fix (state in the methodology):** the prototype's `tf.random.set_seed` did not control Keras 3's starting weights, which came from Python's global random state, so earlier LSTM results depended on the notebook's run order. Every LSTM figure produced before 29 October 2026 (including the single-run Sharpe of 0.73–0.74 and placebo p = 0.004) is superseded.

## Hypothesis H4
- Approved wording: H4 (null) — the LSTM does not significantly outperform LASSO or LightGBM net of costs. **(supervisor)**
- Both comparisons are tested and reported, with a Holm correction across the two. **(team)**
- **LSTM vs LightGBM** is the substantive test, because LightGBM is a working forecasting model.
- **LSTM vs linear** is reported in full but not interpreted as evidence of LSTM skill, because the linear model finds no usable signal and the comparison depends on the ridge grid. **(team)**
- Conclusion wording: "H4 is not rejected against LightGBM. The LSTM's significant lead over the linear model reflects the linear model finding no usable signal, not LSTM skill, and is not robust to the ridge penalty grid."

## Portfolio
- Long/flat only (main specification). **(supervisor)**
- Main signal: s = min(1, max(0, (p − b) / (p90 − b))), b = training-window base rate, p90 = 90th percentile of **validation** forecasts, never the test window. **(supervisor)**
- Floor: if p90 − b < 1 percentage point, stay flat for that window. **(supervisor)**
- Sensitivity checks: base-rate centred (p − b)/(1 − b); original 2p − 1. Their Sharpe ratios are inflated by near-zero exposure and are not read as evidence of skill.
- Inverse-volatility base weights, 10% vol target (63-day), no leverage, 25% cap per ETF.
- Weekly rebalance at the next day's close; 2-percentage-point no-trade band applied to each ETF's weight. **(team)**
- Calibration (Brier score, reliability table) is analysis only and is not used for position sizing. **(team)**

## Benchmarks and costs
- SPY buy-and-hold; equal-weight (monthly); vol-targeted SPY; SMA 50/200 rule; always-on (same vol-scaled weights, all signals = 1). Always-on shows whether the trades come from the model. **(supervisor)**
- Benchmarks pay the same cost per side through the same backtest; buy-and-hold pays only the first purchase. **(team)**
- 5 bps per side base case; sensitivity 0/2/5/10 bps; break-even cost. **(supervisor)**

## Statistical tests
- **Significance:** 5%, one-sided. **(team)**
- **Stationary block bootstrap** on Sharpe differences: 10,000 resamples, mean block length 20 days. **(supervisor)**
- **Exposure-matched placebo (supervisor request).** For each of 1,000 draws, circularly shift the model's actual weekly signals by a random number of weeks (uniform between 52 and W − 52). This keeps exposure, position sizes and holding lengths, but breaks the timing. Each draw goes through the same backtest and costs. Two variants (second approved by the supervisor):
  - independent shifts per ETF;
  - common shift: one random shift per draw for all ETFs, keeping the cross-ETF pattern of positions.

  Report the actual Sharpe; the placebo median and 5th/95th percentiles; p = (1 + number of draws with Sharpe ≥ actual) / (N + 1); and the placebo median maximum drawdown next to the actual one, so lower drawdowns from low exposure are not mistaken for skill. Run for all three models, every LSTM seed and the seed-average.
- **Deflated Sharpe ratio:** trials = 25. **(team)** Rule: one trial for every model × signal-version combination whose test-period Sharpe ratio was computed: 9 main specifications (3 models × 3 signals); the linear model's L1 refit and its two earlier ridge grids (3 signal versions each = 9); the original input design, where LASSO and LightGBM forecast from their own selections (3 signal versions each = 6); and the seed-averaged LSTM (main signal = 1). The count was 18 until 30 October 2026, when the original input design and the seed average were added under the same rule. The benchmark SR0 is the expected maximum Sharpe ratio of 25 trials with no skill, from the variance of the 25 trials' Sharpe ratios (Bailey & López de Prado, 2014); a DSR above 0.95 is significant at 5%. Trials not computed in the current run enter with their recorded Sharpe ratios, listed in `config.yaml`. LSTM seeds are not counted as trials, because they are reported as a distribution, never chosen from. Placebo and sub-period runs are not counted, because they re-test the same specifications. Counting generously makes the correction stricter, not looser.
- Ledoit–Wolf test and Diebold–Mariano test on forecast loss, as complements to the bootstrap.

## Sub-periods (team)
- GFC: Sep 2008 – Jun 2009.
- COVID: Feb – Jun 2020.
- Rate shock: Jan – Oct 2022.
- Structural break on 24 Sep 2018, the GICS reclassification that created the Communication Services sector and moved large holdings out of XLK and XLY into the new XLC, changing those two ETFs' composition.
- Dates follow the conventional market-stress windows; they were fixed before the sub-period results were computed.

## Scope (team)
- No extension (longer horizon or a smaller fixed indicator set) is added to the main study. Adding one after the main results are known would risk specification search. A longer horizon is listed as future work.

## Framing of the findings (supervisor)
- Null framing, kept precise: no model adds reliable value net of costs over the always-on portfolio; the LSTM has a small but consistent ranking edge; the linear model finds no usable signal.

## Indicator stability (supervisor request)
- Selection frequency per indicator; average Jaccard overlap between consecutive windows; average share of selected indicators by family.
