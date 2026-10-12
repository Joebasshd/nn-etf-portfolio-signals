"""Turn model probabilities into position signals in [0, 1].

Not implemented yet. Settings come from the `signals` section of config.yaml.

Main:         s = min(1, max(0, (p - b) / (p90 - b)))
                b   = share of positive labels in the training window
                p90 = 90th percentile of the model's validation forecasts
Floor:        if p90 - b < 0.01, s = 0 for every date in that window
Sensitivity:  base-rate centred  s = (p - b) / (1 - b)
              linear             s = 2p - 1
              both limited to [0, 1], since the portfolio is long/flat

b and p90 are fixed per window and model (models.run_walk_forward returns
them as window_stats). They never use test-year data.
"""

import pandas as pd


def scaled_signal(probabilities: pd.DataFrame, window_stats: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Main signal, including the floor rule."""
    raise NotImplementedError


def base_rate_centred_signal(probabilities: pd.DataFrame, window_stats: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def linear_signal(probabilities: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def flat_windows(window_stats: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Windows and models where the floor rule applied, for reporting."""
    raise NotImplementedError
