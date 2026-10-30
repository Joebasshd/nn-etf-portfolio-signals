"""Volatility-scaled target weights, benchmarks and the cost-aware weekly backtest."""

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
