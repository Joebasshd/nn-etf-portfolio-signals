"""Significance tests for strategy comparisons."""

import pandas as pd


def bootstrap_sharpe_difference(returns_a: pd.Series, returns_b: pd.Series, cfg: dict) -> dict:
    """Point estimate, confidence interval and p-value for Sharpe(a) - Sharpe(b)."""
    raise NotImplementedError


def ledoit_wolf_test(returns_a: pd.Series, returns_b: pd.Series) -> dict:
    raise NotImplementedError


def deflated_sharpe(returns: pd.Series, trial_sharpes: pd.Series) -> float:
    """Probability that the Sharpe ratio exceeds the best expected by luck across the trials."""
    raise NotImplementedError


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, horizon: int) -> dict:
    raise NotImplementedError
