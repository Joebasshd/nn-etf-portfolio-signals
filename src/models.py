"""LASSO, LightGBM and LSTM classifiers with in-window feature selection.

Not implemented yet. The interface below is what the rest of the pipeline
expects; settings come from the `models` section of config.yaml.

Two stages in every window:
  selection    LASSO     L1 logistic regression; selected = non-zero coefficients
               LightGBM  top 15 features by importance
  forecasting  LASSO, LightGBM and LSTM are all fitted on the union of the
               two selections (the LSTM as 20-day sequences); the selection
               fits never forecast; LASSO's forecasting fit is L2 (ridge)
               logistic so it keeps every union feature

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
    """Features kept by the LASSO and LightGBM selection fits (top 15), and their union, the forecasting inputs."""
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
