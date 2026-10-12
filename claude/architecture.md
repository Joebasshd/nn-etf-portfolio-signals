# Capstone architecture framework

How the team, Claude and the code fit together for the MScFE 690 capstone (Oct 2026 – final submission). Source of truth for methodology: `claude/design-decisions.md`.

## The six layers

| Layer | What lives there | Tool | Owner |
|---|---|---|---|
| 1. Knowledge | Problem statement, design decisions, supervisor feedback log, literature review, course guidelines | Claude Project "Capstone" | Whole team |
| 2. Instructions | How Claude should code, write and handle feedback for this project | Skills (claude.ai account) + `CLAUDE.md` and `.claude/skills/` in the repo | Temitope |
| 3. Connections | Access to GitHub, Google Drive, Slack | MCP connectors | Whoever uses them |
| 4. Code | The research pipeline as tested Python modules | Private GitHub repo | Split by module (below) |
| 5. Compute | Running the full walk-forward and backtests | Google Colab (CPU; GPU optional for LSTM) | Whoever runs the job |
| 6. Delivery | Proposal → Draft → Final report, repo link, Turnitin PDF | Word/Docs + PDF export | Whole team |

## Layer 4: repository structure (target)

```
mscfe-capstone/                      (private; instructor = read-only)
├── README.md                        how to install and run; Colab badge
├── CLAUDE.md                        short project facts for Claude Code
├── CONTRIBUTORS.md                  required by the source code guidelines
├── requirements.txt
├── config.yaml                      every design choice in one place
├── .claude/skills/                  project skills shared with the team
├── src/
│   ├── data.py                      download Yahoo + FRED, cache to data/raw
│   ├── features.py                  31 indicators + execution-aligned 5-day label
│   ├── splits.py                    expanding walk-forward, purge + embargo
│   ├── models.py                    LASSO, LightGBM, LSTM; in-window selection
│   ├── signals.py                   (p − b)/(p90 − b) with floor; sensitivity versions
│   ├── portfolio.py                 vol-scaled weights, benchmarks, cost-aware backtest
│   ├── evaluate.py                  prediction + portfolio metrics, stability measures
│   └── stats.py                     block bootstrap, Ledoit–Wolf, deflated Sharpe
├── tests/                           leakage tests, label alignment, synthetic-data run
├── notebooks/
│   └── 01_full_pipeline.ipynb       thin: imports src/, shows tables and charts
├── data/                            raw and processed (cached; large files git-ignored)
├── results/                         CSVs and figures written by each run (dated)
└── reports/                         proposal, draft, final (PDF exports)
```

Why modules instead of one notebook: three people can work in parallel, each piece can be tested on its own, and an examiner can read one file at a time.

## Module ownership (suggested)

| Area | Modules | Owner |
|---|---|---|
| Data and features | `data.py`, `features.py`, `tests/test_features.py` | TBD |
| Models and validation | `splits.py`, `models.py`, `tests/test_leakage.py` | TBD |
| Portfolio and evaluation | `signals.py`, `portfolio.py`, `evaluate.py`, `stats.py` | TBD |
| Report and literature | `reports/`, references | Rotating |

## Quality gates (run before every merge)

1. **Leakage tests pass:** no training label uses a price from the validation or test period; features at date t use only data up to t; FRED series lagged one day.
2. **Synthetic-data run passes:** the whole pipeline runs end to end on generated prices in quick mode.
3. **Config, not code, for choices:** any new parameter goes into `config.yaml` and `design-decisions.md`.
4. **Consistency check:** a change to methodology updates the notebook, the report methodology section and `design-decisions.md` together.
5. **Reproducibility:** fixed seed; results folder named by date and git commit.

## Workflow

1. Supervisor feedback arrives (Slack or grading comments) → logged and translated into concrete changes (skill: `supervisor-feedback-response`).
2. Team replies on Slack with the proposed change; waits for approval on design changes.
3. Approved change → branch → code change + tests (skill: `capstone-etf-pipeline`) → pull request reviewed by one teammate → merge.
4. Full run in Colab → results saved → report updated (skill: `capstone-report-writing`).

## Milestones

| Stage | Deliverable | Key code state |
|---|---|---|
| Proposal (done) | Proposal, literature review, code outline | Notebook runs end to end on synthetic data |
| Implementation | First full run on market data | Modules split out; leakage tests; calibration analysis |
| Draft project | Results + discussion | Robustness checks (2018 break, crises, vol targets); Ledoit–Wolf, deflated Sharpe, Diebold–Mariano |
| Final project | Final report + final repo | Frozen results, README run instructions verified on a clean Colab |
