"""Map model probabilities to long/flat position signals in [0, 1]."""

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
