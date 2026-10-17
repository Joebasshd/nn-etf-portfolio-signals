# Bridge plan

Steps to bring the repository up to what the proposal describes: the three models, signals, portfolio, benchmarks, costs and bootstrap implemented in `src/`, tested end to end, and run on market data.

Keep this file at `docs/bridge-plan.md`. Each step has a prompt to paste into Claude Code; the prompts refer back to this file.

**Working rules for every step**

- Start each step on a new branch from an up-to-date `main` (`git checkout main && git pull && git checkout -b <branch>`).
- Ask Claude Code to plan first; read the plan before letting it edit.
- Finish with `pytest tests`, then commit, push and open a pull request into `main`.
- Port logic from the prototype notebook rather than writing it from scratch: `notebooks/prototype/capstone_signal_to_portfolio.ipynb`.

---

## Decisions to agree before Step 3

Confirm these as a team, record them in `docs/design-decisions.md` and `config.yaml`, and mention the substantive ones to Prof Blanco.

| Question | Suggested default |
|---|---|
| Refit the final model on train + validation, or train only? | Train only. Validation is used for tuning, early stopping and `p90`, which keeps `p90` out-of-sample. |
| Hyperparameter grids | LASSO `C` in [0.002, 0.005, 0.01, 0.05, 0.1]; LightGBM fixed settings with early stopping (100 rounds); LSTM 1 layer, 32 units, dropout 0.2, 20-day lookback (as in the proposal). |
| Calibration: pipeline step or analysis only? | Analysis only (Brier score, calibration table). Not used for position sizing. |
| No-trade band: per ETF or on total turnover? | Per ETF weight. |
| Do benchmarks pay trading costs? | Yes, the same cost per side through the same backtest. Buy-and-hold pays only the initial purchase. |
| Significance level; deflated-Sharpe trials | 5% one-sided; trials = 9 (3 models x 3 signal versions). |
| Crisis windows; 2018 break | GFC 2008-09 to 2009-06; COVID 2020-02 to 2020-06; 2022 rate shock 2022-01 to 2022-10; break at 2018-09-24 (XLC launch). |

---

## Step 1. Commit the prototype as evidence

Branch: `evidence/prototype`

Put this file at `docs/bridge-plan.md` and the prototype at `notebooks/prototype/capstone_signal_to_portfolio.ipynb`, then:

```
The claude/ folder was renamed to docs/. Find every reference to
claude/design-decisions.md, claude/architecture.md or claude/ in CLAUDE.md,
README.md, config.yaml, src/ and tests/, and update it to docs/. Show me the
list of changes before editing.
```

```
Read docs/bridge-plan.md, Step 1. The notebook
notebooks/prototype/capstone_signal_to_portfolio.ipynb is the working prototype
of the full pipeline (models, signals, portfolio, benchmarks, costs, bootstrap,
stability). Do not change its logic. Add a markdown cell at the top saying it
is the proposal-stage prototype, tested end to end on synthetic data, and that
its logic is being ported into src/. Add a line to the README layout table for
notebooks/prototype/.
```

Done when: the prototype and this plan are on `main`, and no file refers to `claude/`.

## Step 2. Repository fixes

Branch: `fix/repo-hygiene`

```
Read docs/bridge-plan.md, Step 2. Then:
1. In notebooks/01_full_pipeline.ipynb, replace OWNER/REPO in the clone cell
   with Joebasshd/nn-etf-portfolio-signals.
2. Check the README Colab badge points to a notebook that exists on main.
3. Add pandas-datareader to requirements.txt (used by 00_initial_exploration).
4. In config.yaml set the LSTM to layers: 1 and add units: 32, to match the
   proposal.
5. Check whether CLAUDE.md is tracked by git. If the README links to it and it
   is not tracked, ask me whether to track it or remove the link.
Run pytest tests and show me the diff before committing.
```

Done when: the Colab badge opens a notebook whose first cell clones the right repository, and the instructor has read access to the repository.

## Step 3. Models (`src/models.py`)

Branch: `feature/models`. First record the decisions above in `docs/design-decisions.md` and `config.yaml`.

```
Read CLAUDE.md (Phase 1), docs/design-decisions.md and docs/bridge-plan.md.
Implement src/models.py by porting the model logic from
notebooks/prototype/capstone_signal_to_portfolio.ipynb (fit_lasso, fit_gbm,
make_sequences, fit_predict_lstm and the walk-forward loop), adapted to the
repo's interfaces: data from features.build_dataset, splits from
splits.walk_forward_splits, settings from config.yaml, and the return values
described in the models.py docstring (probabilities, selections, window_stats
with b and p90). Train on decision dates, as the existing synthetic test does.
Respect the leakage rules in CLAUDE.md.
Add the Phase 1 tests from CLAUDE.md, including: changing test-year data leaves
the scaler, selected features, b and p90 unchanged; LightGBM selects exactly
15; LSTM inputs equal the union. Quick mode must finish in about a minute.
Plan first and wait for my approval.
```

Done when: the new tests pass and `run_walk_forward` returns the three tables on synthetic data.

## Step 4. Signals (`src/signals.py`)

Branch: `feature/signals`

```
Read CLAUDE.md (Phase 2). Implement src/signals.py: scaled_signal with the
floor rule, base_rate_centred_signal, linear_signal and flat_windows, using
window_stats from models.run_walk_forward. Port the logic from the prototype
notebook (columns s_, sb_, sr_ and the edge table). Add the Phase 2 tests from
CLAUDE.md. Plan first.
```

Done when: every signal is in [0, 1], `p = b` gives 0, `p >= p90` gives 1, and the floor sets a whole window to 0.

## Step 5. Portfolio and benchmarks (`src/portfolio.py`)

Branch: `feature/portfolio`

```
Read CLAUDE.md (Phase 3) and docs/design-decisions.md. Implement
src/portfolio.py: target_weights, backtest and benchmark_returns, porting
base_weights, backtest and the benchmark construction from the prototype
notebook. Weekly decisions, trade at close t+1, per-ETF no-trade band, costs
per side, cash at the T-bill rate. Add the Phase 3 tests from CLAUDE.md,
including: always-on equals the main strategy when every signal is 1, and SPY
buy-and-hold reproduces SPY's return. Plan first.
```

Done when: weights are non-negative, each at most 25%, summing to at most 100%, and the Phase 3 tests pass.

## Step 6. Evaluation and statistics (`src/evaluate.py`, `src/stats.py`)

Branch: `feature/evaluation`

```
Read CLAUDE.md (Phase 4). Implement src/evaluate.py (prediction metrics,
calibration_table, portfolio metrics, breakeven_cost, selection_frequency,
jaccard_overlap, family_share using features.feature_family, results_dir) and
src/stats.py (bootstrap_sharpe_difference first; then ledoit_wolf_test,
deflated_sharpe, diebold_mariano). Port perf_stats, the stationary bootstrap
and the stability code from the prototype notebook. Add the Phase 4 tests.
Plan first.
```

Done when: Jaccard is 1 for identical sets and 0 for disjoint ones, identical return series give a non-significant Sharpe difference, and the Phase 4 tests pass.

## Step 7. End-to-end test and notebook

Branch: `feature/end-to-end`

```
Extend tests/test_synthetic_run.py so it runs the entire pipeline on synthetic
prices in quick mode: dataset, splits, models, signals, portfolio, benchmarks,
evaluation and the bootstrap, and checks the output tables have the expected
shape. Then fill in sections 4 to 7 of notebooks/01_full_pipeline.ipynb so it
calls src/ and shows the prediction metrics, the signal edge table (b, p90,
p90 - b, flat windows), stability tables, portfolio performance, cost
sensitivity and bootstrap results. Keep the notebook thin.
```

Done when: `pytest tests` runs the full pipeline end to end.

## Step 8. First results on market data

Branch: `results/first-run`

1. Locally, set `run.quick_mode: false` and run the notebook once with refits every three years to check it finishes.
2. In Colab, run the full annual-refit version from `main` (CPU runtime, about 20 to 40 minutes).
3. Save the outputs to `results/<date>_<commit>/` and commit the small CSV tables, not raw data.

```
Using the CSVs in results/<run folder>/, add a markdown cell at the end of
notebooks/01_full_pipeline.ipynb titled "Initial findings on market data".
Report, without overstating: the base rate per window, the mean and standard
deviation of each model's probabilities, how many windows hit the floor,
out-of-sample AUC against 0.5, and net-of-cost Sharpe ratios against the
always-on portfolio. State clearly whether the probabilities cluster near the
base rate on real data, as observed on synthetic data.
```

Done when: the findings cell and the results tables are on `main`.

## Step 9. Close the loop

Branch: `docs/status-update`

```
Update the README status table and the CLAUDE.md status table to show which
modules are implemented and tested. Check config.yaml, docs/design-decisions.md
and the notebook agree. List anything that still disagrees.
```

Merge to `main`, then tag the commit: `git tag draft-results-v1 && git push --tags`.

## Step 10. Carry it into the draft report

- Section 4 (implementation status) and the Conclusion: describe what is implemented, tested and run on market data, using the results from Step 8.
- Section 3.5: `p90` is the 90th percentile of the **validation-period** forecasts, and the 55% figure refers to **five-day** returns.
- Section 3.3: LSTM is a single layer with 32 units, matching `config.yaml`.
