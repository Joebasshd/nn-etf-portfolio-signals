"""31 indicators per ETF and the execution-aligned 5-day label.

Every feature on date t uses data up to and including t only. The label for a
decision made at the close of t is the return from the close of t + lag to the
close of t + lag + horizon, so a Friday decision is traded at Monday's close
and held to the following Monday's close.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Column prefix -> indicator family, used for the stability measures.
FAMILIES = {
    "trend": "trend",
    "mom": "momentum",
    "vol": "volatility",
    "volu": "volume",
    "ctx": "market context",
}


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return tr.max(axis=1, skipna=False)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return true_range(high, low, close).rolling(window=window).mean()


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    smooth = dict(alpha=1 / window, min_periods=window, adjust=False)
    atr = true_range(high, low, close).ewm(**smooth).mean()
    plus_di = 100 * plus_dm.ewm(**smooth).mean() / atr
    minus_di = 100 * minus_dm.ewm(**smooth).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(**smooth).mean()


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    lowest, highest = low.rolling(k).min(), high.rolling(k).max()
    pct_k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    return pct_k, pct_k.rolling(d).mean()


def compute_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return (multiplier * volume).rolling(window).sum() / volume.rolling(window).sum()


def compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    typical = (high + low + close) / 3
    flow = typical * volume
    direction = typical.diff()
    pos = flow.where(direction > 0, 0.0).rolling(window).sum()
    neg = flow.where(direction < 0, 0.0).rolling(window).sum()
    return (100 - 100 / (1 + pos / neg.replace(0, np.nan))).where(direction.notna())


def ticker_features(prices: pd.DataFrame, ticker: str, cfg: dict) -> pd.DataFrame:
    """The 27 price and volume indicators for one ETF."""
    fc = cfg["features"]
    px = prices["adj_close"][ticker]
    # High and low are put on the adjusted-close scale so that ranges are not
    # distorted by dividends and splits.
    adj = px / prices["close"][ticker]
    h, l = prices["high"][ticker] * adj, prices["low"][ticker] * adj
    v = prices["volume"][ticker]
    ret = px.pct_change()

    f = pd.DataFrame(index=px.index)

    # Trend (8). MACD terms are divided by price to stay comparable over time.
    for w in fc["sma_windows"]:
        f[f"trend_sma{w}_ratio"] = px / px.rolling(w).mean() - 1
    macd, macd_sig, macd_hist = compute_macd(px, **fc["macd"])
    f["trend_macd_line"] = macd / px
    f["trend_macd_signal"] = macd_sig / px
    f["trend_macd_hist"] = macd_hist / px
    f["trend_adx"] = compute_adx(h, l, px, fc["adx_window"])

    # Momentum (10)
    for w in fc["return_windows"]:
        f[f"mom_ret_{w}d"] = px.pct_change(w)
    f["mom_rsi"] = compute_rsi(px, fc["rsi_window"])
    f["mom_stoch_k"], f["mom_stoch_d"] = compute_stochastic(h, l, px, **fc["stochastic"])
    hh, ll = h.rolling(fc["williams_window"]).max(), l.rolling(fc["williams_window"]).min()
    f["mom_williams_r"] = -100 * (hh - px) / (hh - ll).replace(0, np.nan)
    f["mom_roc"] = 100 * px.pct_change(fc["roc_window"])

    # Volatility (5)
    for w in fc["realized_vol_windows"]:
        f[f"vol_realized_{w}d"] = ret.rolling(w).std() * np.sqrt(TRADING_DAYS)
    f["vol_atr_norm"] = compute_atr(h, l, px, fc["atr_window"]) / px
    bw, n_std = fc["bollinger"]["window"], fc["bollinger"]["n_std"]
    mid, sd = px.rolling(bw).mean(), px.rolling(bw).std()
    f["vol_bb_pct_b"] = (px - (mid - n_std * sd)) / (2 * n_std * sd).replace(0, np.nan)
    f["vol_bb_bandwidth"] = 2 * n_std * sd / mid

    # Volume (4). The OBV change is scaled by total volume over the window.
    ow = fc["obv_change_window"]
    obv = (np.sign(ret) * v).where(ret.notna()).cumsum()
    f["volu_obv_change"] = obv.diff(ow) / v.rolling(ow).sum()
    zw = fc["volume_z_window"]
    f["volu_zscore"] = (v - v.rolling(zw).mean()) / v.rolling(zw).std()
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
    return out.loc[cfg["features"]["sample_start"]:]


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
