"""Download Yahoo prices and FRED macro series, cache them to data/raw.

Prices are returned as one DataFrame with two column levels, (field, ticker),
where field is one of open, high, low, close, adj_close, volume.
Macro series are aligned to the trading calendar and lagged so that the value
on date t is the one already published before t.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["open", "high", "low", "close", "adj_close", "volume"]
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def load_config(path=None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def all_tickers(cfg: dict) -> list:
    return [cfg["universe"]["benchmark"]] + list(cfg["universe"]["tradable"])


def download_prices(cfg: dict) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers=all_tickers(cfg),
        start=cfg["data"]["start"],
        end=cfg["data"]["end"],
        interval="1d",
        auto_adjust=False,
        group_by="column",
        progress=False,
    )
    raw = raw.rename(columns=lambda c: c.lower().replace(" ", "_"), level=0)
    prices = raw[FIELDS].sort_index()
    prices.columns.names = ["field", "ticker"]
    prices.index.name = "date"
    return prices


def download_macro(cfg: dict) -> pd.DataFrame:
    """Raw FRED series on their own calendar, no alignment and no lag."""
    import requests

    out = {}
    for series, name in cfg["data"]["fred_series"].items():
        response = requests.get(FRED_CSV.format(series=series), timeout=60)
        response.raise_for_status()
        s = pd.read_csv(io.StringIO(response.text), index_col=0, parse_dates=True, na_values=".")
        out[name] = s.iloc[:, 0]
    macro = pd.DataFrame(out).loc[cfg["data"]["start"]:cfg["data"]["end"]]
    macro.index.name = "date"
    return macro


def align_macro(macro: pd.DataFrame, calendar: pd.DatetimeIndex, lag_days: int) -> pd.DataFrame:
    """Put macro series on the trading calendar, forward-filled then lagged.

    Only forward-fill is used: a back-fill would copy later values into
    earlier dates.
    """
    return macro.reindex(calendar.union(macro.index)).ffill().reindex(calendar).shift(lag_days)


def load_data(cfg: dict, refresh: bool = False):
    """Return (prices, macro), downloading only when the cache is missing."""
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    prices_path, macro_path = raw_dir / "prices.parquet", raw_dir / "macro.parquet"

    if refresh or not prices_path.exists():
        download_prices(cfg).to_parquet(prices_path)
    if refresh or not macro_path.exists():
        download_macro(cfg).to_parquet(macro_path)

    prices = pd.read_parquet(prices_path)
    macro = align_macro(pd.read_parquet(macro_path), prices.index, cfg["data"]["fred_lag_days"])
    return prices, macro


def make_synthetic(cfg: dict, n_days: int = 1500, seed=None):
    """Generated prices and macro series with the same layout as load_data.

    Used by the tests and by quick mode, so the pipeline can run end to end
    without a network connection.
    """
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    tickers = all_tickers(cfg)
    calendar = pd.bdate_range(cfg["data"]["start"], periods=n_days, name="date")
    n = len(tickers)

    market = rng.normal(0.0003, 0.009, size=(n_days, 1))
    rets = market + rng.normal(0.0, 0.008, size=(n_days, n))
    close = 100 * np.exp(np.cumsum(rets, axis=0))
    spread = np.abs(rng.normal(0.0, 0.006, size=(n_days, n)))
    open_ = close * np.exp(rng.normal(0.0, 0.004, size=(n_days, n)))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(15, 0.4, size=(n_days, n))

    frames = {"open": open_, "high": high, "low": low, "close": close, "adj_close": close, "volume": volume}
    prices = pd.concat(
        {f: pd.DataFrame(v, index=calendar, columns=tickers) for f, v in frames.items()},
        axis=1,
        names=["field", "ticker"],
    )

    macro = pd.DataFrame(
        {
            "DTB3": np.clip(2.0 + np.cumsum(rng.normal(0, 0.01, n_days)), 0.0, None),
            "VIX": np.clip(18 + np.cumsum(rng.normal(0, 0.5, n_days)), 9.0, 80.0),
            "TERM_SPREAD": 1.0 + np.cumsum(rng.normal(0, 0.01, n_days)),
        },
        index=calendar,
    )
    return prices, align_macro(macro, calendar, cfg["data"]["fred_lag_days"])
