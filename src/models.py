"""LASSO, LightGBM and LSTM classifiers with in-window feature selection.

Not implemented yet. The interface below is what the rest of the pipeline
expects; settings come from the `models` section of config.yaml.

  LASSO     L1 logistic regression; selected features = non-zero coefficients
  LightGBM  top 15 features by importance
  LSTM      union of the LASSO and LightGBM selections, 20-day sequences

Rules for whoever fills this in:
  - scaling, feature selection and tuning are redone inside every window,
    using its training and validation blocks only, never the test year
  - for each window and model, also return
        b    share of positive labels in the training block
        p90  90th percentile of the model's validation forecasts
    because signals.py needs them
  - the features chosen in each window are returned so that the stability
    measures in evaluate.py can be computed
"""

import pandas as pd

MODEL_NAMES = ["lasso", "lightgbm", "lstm"]


def select_features(train: pd.DataFrame, validation: pd.DataFrame, cfg: dict) -> dict:
    """Features kept by LASSO, by LightGBM (top 15) and their union for the LSTM."""
    raise NotImplementedError


def fit_predict(name: str, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> dict:
    """Fit one model for one window.

    Returns a dict with: test probabilities (indexed like test), validation
    probabilities, the selected features, b and p90.
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
