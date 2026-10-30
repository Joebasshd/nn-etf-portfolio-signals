"""Prediction metrics, portfolio metrics, indicator stability and run folders."""

import pandas as pd


def prediction_metrics(probabilities: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    raise NotImplementedError


def calibration_table(probabilities: pd.Series, labels: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    raise NotImplementedError


def portfolio_metrics(returns: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def breakeven_cost(weights: pd.DataFrame, prices: pd.DataFrame, macro: pd.DataFrame, benchmark: pd.Series, cfg: dict) -> float:
    """Cost per side, in bps, at which the strategy's Sharpe equals the benchmark's."""
    raise NotImplementedError


def selection_frequency(selections: pd.DataFrame) -> pd.DataFrame:
    """Share of windows in which each indicator was selected, per model."""
    raise NotImplementedError


def jaccard_overlap(selections: pd.DataFrame) -> pd.DataFrame:
    """Average Jaccard overlap of selected sets between consecutive windows, per model."""
    raise NotImplementedError


def family_share(selections: pd.DataFrame) -> pd.DataFrame:
    """Average share of selected indicators in each of the five families, per model."""
    raise NotImplementedError


def results_dir(cfg: dict):
    """Create and return results/<date>_<git commit>_<label>/ for the current run."""
    raise NotImplementedError
