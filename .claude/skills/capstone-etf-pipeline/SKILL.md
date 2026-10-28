---
name: "capstone-etf-pipeline"
description: "Use for any code work on the MScFE 690 capstone ETF signal-to-portfolio pipeline: data, indicators, walk-forward models (LASSO, LightGBM, LSTM), signals, backtests, tests."
---

# Capstone ETF signal-to-portfolio pipeline

Project: "Neural Network-Driven Signal Generation for Systematic Portfolio Construction" (WorldQuant University MScFE 690, Research Track). Team: Temitope Emmanuel Fagbuyi, Hart Bassey Ofigwe, Joseph Edet. Supervisor: Prof Iván Blanco. Repo: `Joebasshd/nn-etf-portfolio-signals`.

## Before writing any code

1. Read the source of truth for methodology: `docs/design-decisions.md` and `config.yaml` in the repo (copies in the Capstone project under `claude/`). Numbers and settings live there, not in this skill. If they disagree with the specification below, `docs/design-decisions.md` wins; point out the mismatch.
2. Read `docs/architecture.md` for the repo layout and quality gates, `docs/bridge-plan.md` for the step being worked on, and `CLAUDE.md`.
3. If the request changes methodology (anything in the specification below), do not change it silently. Explain the change in plain English, note whether it needs supervisor approval, and update code, `docs/design-decisions.md`, `config.yaml` and the report methodology together. Mark unapproved settings `# DRAFT` in `config.yaml`.

## Current specification (agreed with the supervisor, Oct 2026)

- **Universe:** tradable XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, QQQ, IWM; SPY benchmark only. XLRE and XLC excluded.
- **Data:** daily, Jan 2000 to Jun 2026; Yahoo Finance adjusted prices; FRED DTB3, VIXCLS, T10Y3M lagged one trading day. IWM usable from May 2001 (252-day indicators). For reproducible runs, use the cached `data/raw/*.parquet` written by `src/data.py`.
- **Features:** 31 in five families (trend, momentum, volatility, volume, market context), expressed as ratios, returns or bounded oscillators.
- **Label:** 1 if the return from close t+1 (execution) to close t+6 is positive.
- **Validation:** expanding walk-forward, annual refit, test 2008 to 2026; last 12 months of each training window = validation; 11-trading-day gap (1 execution lag + 5 horizon + 5 embargo) between train, validation and test. Final fits use the training block only; validation is for tuning, early stopping and p90.
- **Model inputs: two stages in every window (supervisor feedback point 3, identical inputs).**
  1. Selection only, on all 31 features: LASSO (L1 logistic, C by validation log loss) keeps its non-zero coefficients; LightGBM (fixed settings, early stopping) keeps its top 15 by gain.
  2. Forecasting, every model on the same union of the two selections: linear model = ridge (L2) logistic regression, C re-tuned by validation log loss, keeps every union feature; LightGBM with the same fixed settings and early stopping, no further selection; LSTM (1 layer, 32 units, dropout 0.2, 20-day sequences, early stopping). The L1-refit version is a robustness check only.
  - In tables and reports, call the linear model "Linear (L1 selection, ridge on union)", not plain LASSO. Code keys may stay `lasso`; say so wherever the label could mislead.
- **Ridge penalty grid:** C in {0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0} (DRAFT: widened at both ends because C = 0.002 was the lower grid edge in 12 of 19 windows and 0.1 the upper edge in 3; fix the final grid after the rerun).
- **Signal (long/flat):** s = min(1, max(0, (p - b) / (p90 - b))), b = training-window base rate, p90 = 90th percentile of VALIDATION forecasts (never the test window). If p90 - b < 0.01, stay flat for that window. Sensitivity: (p - b)/(1 - b) and 2p - 1; their Sharpe ratios are inflated by near-zero exposure and are not evidence of skill.
- **Portfolio:** inverse-volatility base weights (63-day), scaled to 10% annual volatility, no leverage, 25% cap per ETF, cash earns T-bill. Weekly signals, executed at next day's close, 2-percentage-point no-trade band per ETF.
- **Benchmarks:** SPY buy-and-hold; equal-weight monthly; vol-targeted SPY; SMA 50/200; Always-on (same vol-scaled weights, signals = 1). Benchmarks pay the same costs through the same backtest.
- **Costs:** 5 bps per side base case; 0/2/5/10 sensitivity; break-even cost.
- **Hypothesis H4 (approved wording):** a null, the LSTM does not significantly outperform LASSO or LightGBM net of costs. Do not relabel other findings as H4.
- **Tests:**
  - Stationary block bootstrap on Sharpe differences (10,000 resamples, mean block 20 days, one-sided, 5%).
  - Exposure-matched placebo (supervisor request, Oct 2026; see below).
  - Planned: deflated Sharpe ratio over all versions tried, Ledoit-Wolf, Diebold-Mariano.
- **Stability reporting:** selection frequency per indicator, average Jaccard overlap between consecutive windows, share of selections by family.

## Exposure-matched placebo test

Purpose: show whether a model's Sharpe ratio comes from timing skill or just from being out of the market in bad periods (the LSTM is invested only about 23% of the time).

- Take the model's actual weekly signal matrix (decision weeks x ETFs) from the run.
- Each draw: for every ETF independently, circularly shift its signal series by a random whole number of weeks k, uniform in [52, W - 52], where W is the number of decision weeks. This keeps each ETF's exposure, the distribution of position sizes and holding lengths (so turnover and costs stay comparable), and destroys the alignment with future returns.
- Push every draw through the SAME backtest: vol scaling, caps, no-trade band, 5 bps costs, T-bill cash. Never a simplified return calculation.
- Draws: 1,000 (100 in QUICK_MODE), with a fixed seed from `config.yaml`.
- Report per model: actual Sharpe; placebo mean, 5th and 95th percentiles; placebo p = (1 + number of draws with Sharpe >= actual) / (N + 1); and mean placebo exposure next to actual exposure as a check.
- Optional robustness: random permutation of weeks (keeps exposure, not holding lengths).
- Run it for all three models; the LSTM is the focus.

## Non-negotiable leakage rules

Check every change against these. A violation is a bug even if results improve.

- Anything fitted (scalers, feature selection, hyperparameters, base rate b, p90) uses only that window's training or validation data.
- No training label may use a price from the validation or test period; keep the 11-day gap.
- Features at date t use only information available at the close of t; external series are lagged.
- Trades execute after the signal date; never at the signal date's close.
- Suspiciously good accuracy or Sharpe is a reason to look for leakage before celebrating.

## Tuning hygiene

- After every run, count how often each tuned hyperparameter lands on the edge of its grid. If it does in more than a few windows, widen the grid and rerun before reporting.
- Do not choose a specification because it scored better on the test years. Justify choices on design grounds, and report rejected variants as robustness checks.
- Neural-network results from one seed are provisional. Report the LSTM across several seeds (median and range) once the supervisor confirms the number.

## Coding conventions

- Every design choice lives in one configuration block or `config.yaml`, never hard-coded inside functions.
- Python with pandas, NumPy, scikit-learn, LightGBM, TensorFlow/Keras, matplotlib. Keep `requirements.txt` current.
- Fixed random seed from `config.yaml`.
- Results go to `results/<date>_<git commit>_<label>/` at the repo root (label e.g. `prototype-identical`, `ridge-widegrid`), with small CSVs at the top level, figures in `figures/`, large files (predictions, equity curves) in `large/` (git-ignored), and a `run_info.json` recording commit, settings, data source and dates. Never write to a bare `results/` folder; commit after code changes and before a run so the folder name points at the code that produced it.
- The prototype notebook `notebooks/prototype/capstone_signal_to_portfolio.ipynb` is evidence: add switches rather than deleting the original design. `notebooks/01_full_pipeline.ipynb` stays thin and calls `src/`.
- Code must run in Google Colab on a CPU runtime; keep a QUICK_MODE for fast checks.
- Short comments that say why, not what. Docstrings on public functions.
- Follow the course source code guidelines: private repo, README, requirements.txt, CONTRIBUTORS file, clear commit messages, instructor read-only access.

## How to verify before handing back

1. Run the changed code. If market data is unreachable, run the whole pipeline on synthetic prices (random-walk ETFs with a small momentum term, IWM missing until late May 2000) in QUICK_MODE.
2. Confirm every cell or module runs, outputs have the expected shape, and no leftover references to removed settings remain.
3. When porting prototype logic into `src/`, reproduce the prototype's reference run on the cached data (same settings and seed) and report any differences in AUC or Sharpe.
4. Run `pytest tests`.
5. Report what was changed, what was tested, and what was not tested.

## How to explain results and changes to the team

The team prefers plain language. For any non-trivial change:
- Say what changed in one or two sentences, without jargon.
- Explain why it matters with a concrete example (for instance, an RSI value and what the model does with it).
- Flag anything the supervisor should be told.
- End with a one-sentence summary.
