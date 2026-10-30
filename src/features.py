"""31 indicators per ETF and the execution-aligned 5-day label.

Every feature on date t uses data up to and including t only. The label for a
decision made at the close of t is the return from the close of t + lag to the
close of t + lag + horizon, so a Friday decision is traded at Monday's close
and held to the following Monday's close.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252

FAMILIES = {
    "trend": "trend",
    "mom": "momentum",
    "vol": "volatility",
    "volu": "volume",
    "ctx": "market context",
}


def wilder(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(alpha=1 / window, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    return 100 - 100 / (1 + wilder(delta.clip(lower=0), window) / wilder(-delta.clip(upper=0), window))


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), window)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal and histogram, all divided by price so levels stay comparable across decades."""
    macd_line = (close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()) / close
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = compute_atr(high, low, close, window)
    plus_di = 100 * wilder(plus_dm, window) / atr
    minus_di = 100 * wilder(minus_dm, window) / atr
    return wilder(100 * (plus_di - minus_di).abs() / (plus_di + minus_di), window)


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    lowest, highest = low.rolling(k).min(), high.rolling(k).max()
    pct_k = 100 * (close - lowest) / (highest - lowest)
    return pct_k, pct_k.rolling(d).mean()


def compute_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return (multiplier * volume).rolling(window).sum() / volume.rolling(window).sum()


def compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    typical = (high + low + close) / 3
    flow = typical * volume
    pos = flow.where(typical > typical.shift(), 0.0).rolling(window).sum()
    neg = flow.where(typical < typical.shift(), 0.0).rolling(window).sum()
    # A window with no down days gives pos / 0 = inf and so MFI = 100, a valid value; it is not dropped.
    return 100 - 100 / (1 + pos / neg)


def ticker_features(prices: pd.DataFrame, ticker: str, cfg: dict) -> pd.DataFrame:
    """The 27 price and volume indicators for one ETF."""
    fc = cfg["features"]
    px = prices["adj_close"][ticker]
    # Adjusted-close scale, so dividends and splits do not distort the ranges.
    adj = px / prices["close"][ticker]
    h, l = prices["high"][ticker] * adj, prices["low"][ticker] * adj
    v = prices["volume"][ticker]
    log_ret = np.log(px).diff()

    f = pd.DataFrame(index=px.index)

    for w in fc["sma_windows"]:
        f[f"trend_sma{w}_ratio"] = px / px.rolling(w).mean() - 1
    f["trend_macd_line"], f["trend_macd_signal"], f["trend_macd_hist"] = compute_macd(px, **fc["macd"])
    f["trend_adx"] = compute_adx(h, l, px, fc["adx_window"])

    for w in fc["return_windows"]:
        f[f"mom_ret_{w}d"] = px.pct_change(w)
    f["mom_rsi"] = compute_rsi(px, fc["rsi_window"])
    f["mom_stoch_k"], f["mom_stoch_d"] = compute_stochastic(h, l, px, **fc["stochastic"])
    hh, ll = h.rolling(fc["williams_window"]).max(), l.rolling(fc["williams_window"]).min()
    f["mom_williams_r"] = -100 * (hh - px) / (hh - ll)
    f["mom_roc"] = px.pct_change(fc["roc_window"])

    for w in fc["realized_vol_windows"]:
        f[f"vol_realized_{w}d"] = log_ret.rolling(w).std() * np.sqrt(TRADING_DAYS)
    f["vol_atr_norm"] = compute_atr(h, l, px, fc["atr_window"]) / px
    bw, n_std = fc["bollinger"]["window"], fc["bollinger"]["n_std"]
    mid, sd = px.rolling(bw).mean(), px.rolling(bw).std()
    f["vol_bb_pct_b"] = (px - (mid - n_std * sd)) / (2 * n_std * sd)
    f["vol_bb_bandwidth"] = 2 * n_std * sd / mid

    ow = fc["obv_change_window"]
    obv = (np.sign(px.diff()).fillna(0) * v).cumsum()
    f["volu_obv_change"] = (obv - obv.shift(ow)) / v.rolling(ow).sum()
    zw = fc["volume_z_window"]
    log_volume = np.log(v.replace(0, np.nan))
    f["volu_zscore"] = (log_volume - log_volume.rolling(zw).mean()) / log_volume.rolling(zw).std()
    f["volu_cmf"] = compute_cmf(h, l, px, v, fc["cmf_window"])
    f["volu_mfi"] = compute_mfi(h, l, px, v, fc["mfi_window"])
    return f


def context_features(prices: pd.DataFrame, macro: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The 4 market context indicators, identical for every ETF on a given date."""
    fc = cfg["features"]
    f = pd.DataFrame(index=prices.index)
    f["ctx_vix_level"] = macro["VIX"]
    f["ctx_vix_change"] = macro["VIX"].diff(fc["vix_change_window"])
    f["ctx_benchmark_ret"] = prices["adj_close"][cfg["universe"]["benchmark"]].pct_change(
        fc["benchmark_return_window"]
    )
    f["ctx_term_spread"] = macro["TERM_SPREAD"]
    return f


def build_features(prices: pd.DataFrame, macro: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Feature panel indexed by (date, ticker) for the tradable ETFs."""
    shared = context_features(prices, macro, cfg)
    panel = {t: ticker_features(prices, t, cfg).join(shared) for t in cfg["universe"]["tradable"]}
    out = pd.concat(panel, names=["ticker", "date"]).swaplevel().sort_index()
    # A ratio over a zero range (for example a flat 14-day high-low range) is inf; treat it as missing.
    return out.replace([np.inf, -np.inf], np.nan).loc[cfg["features"]["sample_start"]:]


def build_labels(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Forward return, binary label and the date of the last price the label uses.

    label_end is what the leakage tests check against each block boundary.
    """
    lag, horizon = cfg["label"]["execution_lag_days"], cfg["label"]["horizon_days"]
    px = prices["adj_close"][cfg["universe"]["tradable"]]
    fwd = px.shift(-(lag + horizon)) / px.shift(-lag) - 1
    label_end = pd.Series(px.index, index=px.index).shift(-(lag + horizon))

    out = fwd.stack(future_stack=True).rename("fwd_ret").to_frame()
    out.index.names = ["date", "ticker"]
    out["label"] = (out["fwd_ret"] > 0).astype(float).where(out["fwd_ret"].notna())
    out["label_end"] = label_end.reindex(out.index.get_level_values("date")).to_numpy()
    return out.sort_index().loc[cfg["features"]["sample_start"]:]


def decision_dates(calendar: pd.DatetimeIndex, frequency: str = "weekly") -> pd.DatetimeIndex:
    """Dates on which a portfolio decision is made."""
    if frequency == "daily":
        return calendar
    if frequency == "weekly":
        s = pd.Series(calendar, index=calendar)
        return pd.DatetimeIndex(s.groupby(calendar.to_period("W-FRI")).last().to_numpy())
    raise ValueError(f"unknown decision frequency: {frequency}")


def build_dataset(prices: pd.DataFrame, macro: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Features joined to labels. Rows still inside an indicator warm-up are dropped,
    so an ETF enters only once its longest indicator has a value.

    Rows whose label is not yet known are kept (label is NaN) so that the most
    recent decision dates can still be scored.
    """
    features = build_features(prices, macro, cfg)
    data = features.join(build_labels(prices, cfg))
    return data.dropna(subset=list(features.columns))


def feature_columns(data: pd.DataFrame) -> list:
    return [c for c in data.columns if c not in ("fwd_ret", "label", "label_end")]


def feature_family(column: str) -> str:
    return FAMILIES[column.split("_")[0]]
