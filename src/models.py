"""Walk-forward selection and forecasting: LASSO and LightGBM select features; ridge, LightGBM and LSTM forecast from their union."""

import pandas as pd

MODEL_NAMES = ["lasso", "lightgbm", "lstm"]


def select_features(train: pd.DataFrame, validation: pd.DataFrame, cfg: dict) -> dict:
    """Features kept by the LASSO and LightGBM selection fits, and their union."""
    raise NotImplementedError


def fit_predict(name: str, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> dict:
    """Fit one forecasting model for one window on the union of the selections.

    Returns a dict with: test probabilities (indexed like test), validation
    probabilities, the input features, b and p90.
    """
    raise NotImplementedError


def run_walk_forward(data: pd.DataFrame, splits: list, cfg: dict):
    """Run every model across all windows.

    Returns three DataFrames:
      probabilities  indexed by (date, ticker), one column per model
      selections     one row per (test_year, model, feature) selected
      window_stats   indexed by (test_year, model), columns b and p90
    """
    raise NotImplementedError
