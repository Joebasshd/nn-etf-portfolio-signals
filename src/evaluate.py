"""Prediction metrics, portfolio metrics and indicator stability.

Not implemented yet.

Prediction:  accuracy against the base rate, AUC, log loss, Brier score,
             calibration table
Portfolio:   annual return, volatility, Sharpe, Sortino, max drawdown,
             turnover, average exposure, total cost, break-even cost
Stability:   selection frequency per indicator, average Jaccard overlap
             between consecutive windows, average share of selected
             indicators by family (features.feature_family)
Sub-periods: the same metrics by test year, around the 2018 break and in
             crisis windows

Each run writes its tables and figures to results/<date>_<git commit>/.
"""

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
    """Create and return results/<date>_<git commit>/ for the current run."""
    raise NotImplementedError
