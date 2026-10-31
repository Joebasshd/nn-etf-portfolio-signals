"""Map model probabilities to long/flat position signals in [0, 1].

Signals are computed on the daily test rows; the portfolio takes them on the
weekly decision dates. b and p90 come from models.run_walk_forward window_stats,
fixed per window from training and validation data.
"""

import pandas as pd


def main_models(cfg: dict) -> list:
    """The three main specifications; the first LSTM seed is the main LSTM."""
    return ["lasso", "lightgbm", f"lstm_{cfg['models']['lstm']['seeds'][0]}"]


def _window_values(probabilities: pd.DataFrame, window_stats: pd.DataFrame, column: str) -> pd.DataFrame:
    years = probabilities.index.get_level_values("date").year
    table = window_stats[column].unstack("model")
    missing = set(years) - set(table.index) or set(probabilities.columns) - set(table.columns)
    if missing:
        raise KeyError(f"window_stats has no {column} for {sorted(missing)}")
    return pd.DataFrame(table.loc[years, probabilities.columns].to_numpy(), index=probabilities.index,
                        columns=probabilities.columns)


def scaled_signal(probabilities: pd.DataFrame, window_stats: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Main signal min(1, max(0, (p - b) / (p90 - b))); 0 for the whole window if p90 - b < the floor."""
    b = _window_values(probabilities, window_stats, "b")
    p90 = _window_values(probabilities, window_stats, "p90")
    edge = p90 - b
    signal = ((probabilities - b) / edge).clip(0, 1)
    return signal.mask(edge < cfg["signals"]["floor"], 0.0)


def base_rate_centred_signal(probabilities: pd.DataFrame, window_stats: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity signal (p - b) / (1 - b), limited to [0, 1]; no floor."""
    b = _window_values(probabilities, window_stats, "b")
    return ((probabilities - b) / (1 - b)).clip(0, 1)


def linear_signal(probabilities: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity signal 2p - 1, limited to [0, 1]; no floor."""
    return (2 * probabilities - 1).clip(0, 1)


def signal_edges(window_stats: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """b, p90, edge = p90 - b and whether the floor keeps the model flat, per window and model."""
    edges = window_stats[["b", "p90"]].copy()
    edges["edge"] = edges["p90"] - edges["b"]
    edges["flat"] = edges["edge"] < cfg["signals"]["floor"]
    return edges


def flat_windows(window_stats: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Windows and models where the floor rule applied, for reporting."""
    edges = signal_edges(window_stats, cfg)
    return edges[edges["flat"]]
