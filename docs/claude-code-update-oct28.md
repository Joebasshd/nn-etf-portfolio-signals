
## Prompt for Claude Code

```
Read docs/claude-code-update-oct28.md, docs/design-decisions-oct2026.md and
.claude/skills/capstone-etf-pipeline/SKILL.md. Then, showing me the diff before
saving anything:

1. .gitignore: the line ".claude/" ignores the whole folder, including the shared
   skill. Replace it with ".claude/*" followed by "!.claude/skills/", keep
   .claude/settings.local.json ignored, and add "results/*/large/".
2. docs/design-decisions.md: merge in docs/design-decisions-oct2026.md. Replace
   the Models section with the two-stage identical-inputs design, add the placebo
   test under tests, and keep one DRAFT table at the end that combines your
   existing DRAFT rows with the new ones (the new file wins where they disagree:
   LightGBM's forecasting fit is on the union, not a refit on its own top 15).
   Then delete docs/design-decisions-oct2026.md.
3. config.yaml, each new or changed key marked "# DRAFT":
   models.input_design: identical; models.lasso.forecast_penalty: l2;
   models.lasso.ridge_C_grid: [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1];
   stats.placebo: {method: circular_shift, n_draws: 1000, quick_draws: 100,
   min_shift_weeks: 52}; stats.deflated_sharpe_trials: 12.
4. CLAUDE.md:
   - change every claude/ path to docs/;
   - in "Main strategy" step 2 and Phase 1, describe the two stages: selection on
     all 31 features, then all three models fitted on the union (ridge for the
     linear model, LightGBM without further selection, LSTM as before);
   - replace the Phase 1 test "LSTM inputs equal the union" with "all three
     models' forecasting inputs equal the union" and add "ridge keeps every union
     feature (no zero coefficients)";
   - add the placebo test to Phase 4 (method and tests from the skill: placebo
     exposure equals actual exposure per ETF; a signal built from future returns
     beats every placebo draw; a random signal does not);
   - results folders: results/<date>_<git commit>_<label>/ with large/ and
     run_info.json;
   - update the status table and mark the open points that are now settled.
5. docs/bridge-plan.md, Step 3 prompt: same change to the union test.
Do not change anything in src/ or the notebooks in this step.
```

## After this update

Implement the two supervisor requests on a new branch:

```
Read docs/design-decisions.md, config.yaml and the capstone-etf-pipeline skill.
1. In the prototype notebook, widen the ridge C grid to the config grid and log
   how often the chosen C sits on a grid edge.
2. Add the exposure-matched placebo test exactly as the skill specifies, for all
   three models, saving placebo_tests.csv.
Keep the results-folder code (results/<date>_<commit>_<label>/, run_info.json,
equity curves). Commit, then run on the cached data with label ridge-widegrid.
Plan first.
```
