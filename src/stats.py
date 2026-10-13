"""Significance tests for strategy comparisons.

Not implemented yet. Settings come from the `stats` section of config.yaml.

Agreed in design-decisions.md:
  - stationary block bootstrap of Sharpe ratio differences, net of costs
    (10,000 resamples, about 20-day blocks)

H4 is framed as a null: the LSTM does not significantly outperform LASSO or
LightGBM net of costs. The comparisons that test it are LSTM minus LASSO and
LSTM minus LightGBM.

Listed in architecture.md for the draft stage:
  - Ledoit-Wolf (2008) robust test for a difference in Sharpe ratios
  - deflated Sharpe ratio, adjusting for the number of strategies tried
  - Diebold-Mariano test on forecast losses
"""

import pandas as pd


def bootstrap_sharpe_difference(returns_a: pd.Series, returns_b: pd.Series, cfg: dict) -> dict:
    """Point estimate, confidence interval and p-value for Sharpe(a) - Sharpe(b)."""
    raise NotImplementedError


def ledoit_wolf_test(returns_a: pd.Series, returns_b: pd.Series) -> dict:
    raise NotImplementedError


def deflated_sharpe(returns: pd.Series, n_trials: int) -> float:
    raise NotImplementedError


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, horizon: int) -> dict:
    raise NotImplementedError
