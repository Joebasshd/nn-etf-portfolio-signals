# nn-etf-portfolio-signals

Neural Network-Driven Signal Generation for Systematic Portfolio Construction: A Risk-Adjusted Extension of Optimized Technical Indicator Models

MScFE 690 Capstone (Research Track). Team: see [CONTRIBUTORS.md](CONTRIBUTORS.md).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Joebasshd/nn-etf-portfolio-signals/blob/main/notebooks/01_full_pipeline.ipynb)

## What this does

Three classifiers (LASSO, LightGBM, LSTM) predict whether each of 11 US equity ETFs will rise over the next five trading days. Each probability `p` becomes a signal `s = min(1, max(0, (p − b) / (p90 − b)))`, where `b` is the training-window base rate and `p90` is the 90th percentile of validation forecasts. Signals are scaled by volatility into a long/flat portfolio with a 10% volatility target and compared with five benchmarks after trading costs. All models are refitted once a year on an expanding walk-forward, with test years 2008 to 2026.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
pytest tests                 # leakage, label alignment and synthetic-data checks
jupyter lab notebooks/01_full_pipeline.ipynb
```

The first run downloads prices from Yahoo and macro series from FRED and caches them in `data/raw`. Set `run.quick_mode: true` in `config.yaml` to use generated prices instead, with no network needed.

On Colab, open the notebook with the badge above; its first cell clones the repository and installs the requirements.

## Layout

| Path | Contents | Status |
| --- | --- | --- |
| `config.yaml` | Every design choice in one place | In use |
| `src/data.py` | Yahoo and FRED download, cache, synthetic prices | Done |
| `src/features.py` | 31 indicators and the execution-aligned 5-day label | Done |
| `src/splits.py` | Expanding walk-forward with an 11-day gap between blocks | Done |
| `src/models.py` | LASSO, LightGBM (top 15 features), LSTM (20-day sequences); in-window selection | Interface only |
| `src/signals.py` | Main signal with floor rule; two sensitivity versions | Interface only |
| `src/portfolio.py` | Vol-scaled weights, benchmarks, cost-aware backtest | Interface only |
| `src/evaluate.py` | Prediction and portfolio metrics, indicator stability | Interface only |
| `src/stats.py` | Block bootstrap, Ledoit–Wolf, deflated Sharpe | Interface only |
| `tests/` | Leakage tests, label alignment, synthetic-data run | Covers the done modules |
| `notebooks/` | `00_initial_exploration` (proposal stage), `01_full_pipeline` (thin, imports `src/`) | |
| `data/` | Raw and processed data (git-ignored) | |
| `results/` | Tables and figures from each run, in a folder named by date and commit | |
| `reports/` | Proposal, draft and final report (PDF) | |
| `docs/` | `design-decisions.md` (methodology, agreed with the supervisor) and `architecture.md` | |

## Before every merge

1. `pytest tests` passes.
2. Any new parameter is in `config.yaml` and `docs/design-decisions.md`, not hard-coded.
3. A change to methodology updates the notebook, the report and `design-decisions.md` together.
4. Runs use the fixed seed in `config.yaml`.

Methodology: [docs/design-decisions.md](docs/design-decisions.md). Team process: [docs/architecture.md](docs/architecture.md).
