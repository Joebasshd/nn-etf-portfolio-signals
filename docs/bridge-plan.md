# Bridge plan

Steps to bring the repository up to what the proposal describes: the full pipeline implemented in `src/`, tested end to end, and run on market data, reproducing the prototype's reference run.

Keep this file at `docs/bridge-plan.md`. Each step has a prompt to paste into Claude Code; the prompts refer back to this file. Last updated 30 October 2026.

**Working rules for every step**

- All `src/` work (Steps 3 to 7) happens on the branch `tofag_main`. Do not create other branches, commit, push, open pull requests, tag or release unless asked (`CLAUDE.md`, Git and GitHub).
- Ask Claude Code to plan first; read the plan before letting it edit.
- Finish each step with `pytest tests`.
- Methodology comes from `docs/design-decisions.md` and settings from `config.yaml`; nothing is hard-coded in `src/`.
- Port logic from the reference prototype, tag `prototype-reference`: `notebooks/prototype/capstone_signal_to_portfolio.ipynb` with `INPUT_DESIGN = "identical"`. Its reference results are in `results/2026-10-30_363ec0b_lstm-10seeds-dsr/`. The `src/` pipeline must reproduce them on the cached data (Step 8).
- Follow the code-style rules in `CLAUDE.md`: comments only for a non-obvious "why", concise docstrings on public functions, no TODO markers.

---

## Done

| Step | What | Where |
|---|---|---|
| 1 | Prototype committed as evidence; `claude/` renamed to `docs/` | PR #3 |
| 2 | Repository fixes: Colab clone URL, `pandas-datareader`, LSTM 1 layer and 32 units | PR #4 |
| 2b | Methodology settled with Prof Blanco and the team; no draft settings left | `docs/design-decisions.md` (30 Oct 2026), PR #5 |
| 2c | Prototype reference run: identical inputs, final ridge grid, 10 LSTM seeds and seed average, seeding fix, placebo in two variants, Holm-corrected H4, deflated Sharpe ratio (25 trials), sub-periods | PR #6, tag `prototype-reference` |

What the `src/` port must carry over from the prototype, beyond the original proposal:

- **Two stages per window.** LASSO (L1) and LightGBM (top 15 by gain) only select. All three forecasting models use the union: ridge (L2) logistic regression with `C` from `models.lasso.ridge_C_grid`, LightGBM with no further selection, and the LSTM.
- **LSTM seeds.** One LSTM per seed in `models.lstm.seeds` (42 to 51), plus the seed-averaged forecast (mean probability across seeds, then the usual `b`, `p90` and floor). Each fit calls `keras.utils.set_random_seed(seed)` and restores the global random state afterwards; `tf.random.set_seed` alone does not fix Keras 3's starting weights.
- **Tuning log.** The chosen LASSO and ridge `C` per window, with counts of grid-edge hits.
- **Placebo.** Exposure-matched circular shift in two variants (independent per ETF; common to all ETFs), 1,000 draws, through the same backtest. Report the median, the 5th and 95th percentiles, p, and the median maximum drawdown next to the actual one. Draw the independent variant for every model first, so its draws do not depend on the common variant.
- **H4.** LSTM vs linear and LSTM vs LightGBM, Holm-corrected across the two, for every seed and the seed average. LSTM vs LightGBM is the substantive test; LSTM vs linear is reported but not interpreted.
- **Deflated Sharpe ratio.** 25 trials: the 9 main specifications and the seed average are recomputed in every run; the 15 historical trials (L1 refit, the two earlier ridge grids, the original input design) are read from the frozen `results/2026-10-30_363ec0b_lstm-10seeds-dsr/dsr_trials.csv`. SR0 comes from the variance of the trials' Sharpe ratios.
- **Sub-periods.** GFC, COVID, the 2022 rate shock, and either side of the 2018 break, from `subperiods` in `config.yaml`.
- **Run folders.** `results/<date>_<commit>_<label>/` with small CSVs, `figures/`, `large/` (git-ignored) and `run_info.json`; run times from `time.monotonic()` plus CPU minutes.

---

## Step 3. Models (`src/models.py`)

```
Read CLAUDE.md (Phase 1), docs/design-decisions.md, config.yaml and
docs/bridge-plan.md (Done section). Implement src/models.py by porting the
model logic of the prototype at tag prototype-reference (fit_lasso, fit_gbm,
fit_ridge_forecast, fit_gbm_forecast, make_sequences, fit_predict_lstm and the
walk-forward loop with INPUT_DESIGN = "identical"), adapted to the repo:
data from features.build_dataset, splits from splits.walk_forward_splits,
settings from config.yaml.
run_walk_forward returns probabilities (one column per model: lasso, lightgbm,
one per LSTM seed, and the seed average), validation probabilities, selections
(LASSO, LightGBM and the union per window), window_stats (b and p90 per window
and model) and a tuning table (chosen LASSO and ridge C per window, with
grid-edge counts). Seed each LSTM fit with keras.utils.set_random_seed and
restore the global random state afterwards.
Add the Phase 1 tests from CLAUDE.md, including: changing test-year data leaves
the scaler, selected features, b and p90 unchanged; LightGBM selects exactly
15; all three models' forecasting inputs equal the union; ridge keeps every
union feature (no zero coefficients); seed 42's forecasts are identical with
or without other seeds in the run; the seed average equals the mean of the
seeds. Quick mode must finish in about a minute.
Plan first and wait for my approval.
```

Done when: the new tests pass and `run_walk_forward` returns its tables on synthetic data.

## Step 4. Signals (`src/signals.py`)

```
Read CLAUDE.md (Phase 2). Implement src/signals.py: scaled_signal with the
floor rule, base_rate_centred_signal, linear_signal and flat_windows, using
window_stats from models.run_walk_forward, for every model including each
LSTM seed and the seed average. Port the logic from the prototype at tag
prototype-reference (columns s_, sb_, sr_ and the edge table). The
sensitivity signals are needed for the three main models only. Add the
Phase 2 tests from CLAUDE.md. Plan first.
```

Done when: every signal is in [0, 1], `p = b` gives 0, `p >= p90` gives 1, and the floor sets a whole window to 0.

## Step 5. Portfolio and benchmarks (`src/portfolio.py`)

```
Read CLAUDE.md (Phase 3) and docs/design-decisions.md. Implement
src/portfolio.py: target_weights, backtest and benchmark_returns, porting
base_weights, backtest and the benchmark construction from the prototype at
tag prototype-reference. Weekly decisions, trade at close t+1, per-ETF
no-trade band, costs per side, cash at the T-bill rate. The backtest must be
the single function used by every strategy, the cost sensitivity and the
placebo. Add the Phase 3 tests from CLAUDE.md, including: always-on equals the
main strategy when every signal is 1, and SPY buy-and-hold reproduces SPY's
return. Plan first.
```

Done when: weights are non-negative, each at most 25%, summing to at most 100%, and the Phase 3 tests pass.

## Step 6. Evaluation and statistics (`src/evaluate.py`, `src/stats.py`)

```
Read CLAUDE.md (Phase 4), docs/design-decisions.md (Statistical tests,
Sub-periods) and config.yaml (stats, subperiods). Implement, porting from the
prototype at tag prototype-reference:
- src/evaluate.py: prediction metrics, calibration_table, portfolio metrics,
  cost sensitivity and breakeven_cost, selection_frequency, jaccard_overlap,
  family_share (features.feature_family), sub-period metrics, the LSTM seed
  summary (median, range, seeds with p < 0.05, seeds with DSR > 0.95, seed
  average on its own row), and results_dir writing
  results/<date>_<commit>_<label>/ with figures/, large/ and run_info.json
  (monotonic run time and CPU minutes).
- src/stats.py: bootstrap_sharpe_difference; Holm correction across the two
  H4 comparisons; the exposure-matched placebo in both variants through
  portfolio.backtest (independent variant drawn first for every model);
  deflated_sharpe with 25 trials; ledoit_wolf_test, the HAC version of
  Ledoit and Wolf (2008) for the Sharpe-ratio difference, reported next to
  the block bootstrap; diebold_mariano on forecast loss.
  deflated_sharpe recomputes the 10 current trials (9 main specifications and
  the seed average) and reads the 15 historical trial Sharpe ratios (L1
  refit, the two earlier ridge grids, the original input design) from the
  frozen results/2026-10-30_363ec0b_lstm-10seeds-dsr/dsr_trials.csv; its docstring says so.
Keep calibration_table (analysis only), ledoit_wolf_test and diebold_mariano.
Add the Phase 4 tests from CLAUDE.md, including: placebo draws keep each ETF's
exposure; a signal built from future returns beats every placebo draw; a
random signal gives a roughly uniform placebo p across repeated draws; Holm
never lowers a p-value; the deflated Sharpe ratio falls as the number of
trials rises. Plan first.
```

Done when: Jaccard is 1 for identical sets and 0 for disjoint ones, identical return series give a non-significant Sharpe difference, and the Phase 4 tests pass.

## Step 7. End-to-end test and notebook

```
Extend tests/test_synthetic_run.py so it runs the entire pipeline on synthetic
prices in quick mode: dataset, splits, models (two seeds and their average),
signals, portfolio, benchmarks, evaluation, bootstrap with Holm, both placebo
variants, the deflated Sharpe ratio and sub-periods, and checks the output
tables have the expected shape. Then fill in notebooks/01_full_pipeline.ipynb
so it calls src/ and shows: prediction metrics, the tuning table, the signal
edge table (b, p90, p90 - b, flat windows), stability, portfolio performance,
cost sensitivity, the LSTM seed table, the placebo table and chart, the H4
tests, the deflated Sharpe ratio and the sub-periods. Keep the notebook thin.
```

Done when: `pytest tests` runs the full pipeline end to end.

## Step 8. Reproduce the reference run on market data

1. Locally, set `run.quick_mode: false` and run `notebooks/01_full_pipeline.ipynb` on the cached data (about 25 minutes).
2. Compare its tables with `results/2026-10-30_363ec0b_lstm-10seeds-dsr/`.
3. In Colab, run the same from the merged code (CPU runtime).
4. Save the outputs to `results/<date>_<commit>_<label>/`, with the small CSVs only in git.

**Tolerances against the reference run.** Same cached data, settings and seeds, so:

| What | Tolerance |
|---|---|
| Linear and LightGBM forecasts (test and validation probabilities) | Identical: absolute difference below 1e-9 |
| LSTM, per seed | AUC within 0.001; Sharpe ratio within 0.01 |
| Tables built on those forecasts (signals, edges, portfolio metrics, placebo, bootstrap, DSR, sub-periods) | The same tolerance as the forecasts they come from |

Anything outside these tolerances is a bug until explained.

```
Compare results/<src run folder>/ with
results/2026-10-30_363ec0b_lstm-10seeds-dsr/ (the prototype reference run):
prediction metrics, portfolio performance, tuning edges, signal edges, LSTM
seed table, placebo, bootstrap with Holm, deflated Sharpe ratio and
sub-periods. Check them against the tolerances in Step 8 of
docs/bridge-plan.md: linear and LightGBM forecasts identical (abs diff < 1e-9);
LSTM per seed AUC within 0.001 and Sharpe within 0.01; tables built on those
forecasts to the same tolerance. Treat anything outside as a bug until
explained, and list each one with its cause. Then
add a markdown cell at the end of notebooks/01_full_pipeline.ipynb titled
"Findings on market data", worded as in the "Framing of the findings" and
"Hypothesis H4" sections of docs/design-decisions.md, without overstating.
```

Done when: every comparison is within tolerance, or each difference is explained, and the findings cell is written.

## Step 9. Close the loop

```
Update the README status table and the CLAUDE.md status table to show which
modules are implemented and tested. Check config.yaml, docs/design-decisions.md,
the notebooks and the report agree. List anything that still disagrees.
```

Then merge `tofag_main` into `main` and tag the merge commit `draft-results-v1`.

## Step 10. Carry it into the draft report

- Methodology: the two-stage identical-inputs design; ridge on the union and why the L1 refit was rejected; the ridge grid widened twice and then fixed; 10 LSTM seeds plus the seed average; the seeding fix, with every LSTM figure from before 29 October 2026 superseded; both placebo variants; Holm across the two H4 comparisons; the deflated Sharpe ratio with 25 trials and its counting rule; sub-periods fixed before results were computed.
- Results: LSTM median and range across seeds (not a single seed); the seed average next to them as a robustness check; placebo, H4 and DSR results; sub-periods, including that the lower drawdowns come from low exposure.
- Conclusion: the framing agreed with the supervisor. No model adds reliable value over always-on net of costs; the LSTM has a small but consistent ranking edge; the linear model finds no usable signal.
- Fixes carried from earlier feedback: `p90` is the 90th percentile of the **validation-period** forecasts; the 55% figure refers to **five-day** returns; the LSTM is a single layer with 32 units.
