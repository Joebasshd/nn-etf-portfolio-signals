"""Vol-scaled weights, benchmarks and the cost-aware backtest.

Not implemented yet. Settings come from the `portfolio` section of config.yaml.

Weights:    inverse-volatility base weights  w_i = s_i / sigma_i  (63-day),
            scaled to a 10% portfolio vol target, no leverage, 25% cap per
            ETF, long/flat only
Trading:    weekly; a decision at close t is traded at close t + 1; changes
            inside the 2-percentage-point no-trade band are skipped; costs are
            charged per side; unallocated cash earns the T-bill rate
Benchmarks: SPY buy-and-hold, equal weight (monthly), vol-targeted SPY,
            SMA 50/200 rule, always-on (same vol-scaled weights with every
            signal set to 1)
Costs:      5 bps per side base case; 0, 2, 5 and 10 bps sensitivity
"""

import pandas as pd


def target_weights(signals: pd.DataFrame, prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Weights per decision date, using prices up to that date only."""
    raise NotImplementedError


def backtest(weights: pd.DataFrame, prices: pd.DataFrame, macro: pd.DataFrame, cfg: dict, cost_bps=None) -> pd.DataFrame:
    """Daily gross and net returns, turnover and costs for one strategy."""
    raise NotImplementedError


def benchmark_returns(prices: pd.DataFrame, macro: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Daily net returns for each benchmark in cfg['portfolio']['benchmarks']."""
    raise NotImplementedError
