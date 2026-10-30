"""Volatility-scaled target weights, benchmarks and the cost-aware daily backtest.

Decisions are made at the close of the last trading day of each week and
executed execution_lag_days later. Everything that does not depend on the
signal is computed once in prepare_market, so the placebo can call backtest
many times.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import features

TRADING_DAYS = 252


@dataclass
class Market:
    dates: pd.DatetimeIndex
    tradable: list
    assets: list
    returns: pd.DataFrame
    rf: pd.Series
    realised_vol: pd.DataFrame
    close: pd.DataFrame
    decision_dates: pd.DatetimeIndex
    month_ends: pd.DatetimeIndex
    raw_returns: pd.DataFrame


def _period_ends(dates: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(dates.to_series().groupby(dates.to_period(freq)).max().to_numpy())


def prepare_market(prices: pd.DataFrame, macro: pd.DataFrame, cfg: dict, start) -> Market:
    """Returns, cash rate, volatility and decision dates from `start` (the first test date) onward."""
    pc = cfg["portfolio"]
    tradable = list(cfg["universe"]["tradable"])
    assets = tradable + [cfg["universe"]["benchmark"]]
    close = prices["adj_close"][assets]
    dates = close.dropna(how="all").index
    close = close.reindex(dates)
    raw_returns = close.pct_change()
    start = pd.Timestamp(start)

    def in_test_period(d):
        return d[(d >= start) & (d < dates[-1])]

    return Market(
        dates=dates,
        tradable=tradable,
        assets=assets,
        returns=raw_returns.fillna(0.0),
        rf=(macro[pc["cash_rate"]].reindex(dates) / 100 / TRADING_DAYS).fillna(0.0),
        realised_vol=raw_returns.rolling(pc["vol_lookback_days"]).std() * np.sqrt(TRADING_DAYS),
        close=close,
        decision_dates=in_test_period(features.decision_dates(dates, cfg["label"]["decision_frequency"])),
        month_ends=in_test_period(_period_ends(dates, "M")),
        raw_returns=raw_returns,
    )


def base_weights(market: Market, cfg: dict) -> pd.DataFrame:
    """Always-on weights per decision date: inverse volatility, scaled to the volatility target, capped per ETF.

    Uses returns up to each decision date only.
    """
    pc = cfg["portfolio"]
    rows = {}
    for t in market.decision_dates:
        vol = market.realised_vol.loc[t, market.tradable].dropna()
        w = (1 / vol) / (1 / vol).sum()
        cov = market.raw_returns.loc[:t, w.index].iloc[-pc["vol_lookback_days"]:].cov() * TRADING_DAYS
        sigma = float(np.sqrt(w @ cov @ w))
        w = w * min(pc["max_leverage"], pc["vol_target"] / sigma)
        rows[t] = w.clip(upper=pc["max_weight"]).reindex(market.assets).fillna(0.0)
    return pd.DataFrame(rows).T


def target_weights(signals: pd.Series, base: pd.DataFrame) -> pd.DataFrame:
    """Signal times base weight on each decision date; a missing signal means no position.

    signals is one model's daily signal indexed by (date, ticker).
    """
    s = signals.unstack("ticker").reindex(index=base.index, columns=base.columns).fillna(0.0)
    return (s * base).reindex(columns=base.columns).fillna(0.0)


def backtest(targets: pd.DataFrame, market: Market, cfg: dict, cost_bps=None, band=None) -> pd.DataFrame:
    """Daily net return, turnover and exposure for one strategy.

    Each target is executed at the close execution_lag_days after its decision date. Between
    trades the weights drift with prices and the cash share earns the T-bill rate. An ETF trades
    only if its target differs from its current weight by at least the band.
    """
    pc = cfg["portfolio"]
    cost_bps = pc["cost_bps_per_side"] if cost_bps is None else cost_bps
    band = pc["no_trade_band"] if band is None else band
    lag = cfg["label"]["execution_lag_days"]

    dates = market.dates
    exec_positions = dates.get_indexer(targets.index) + lag
    exec_dates = dates[exec_positions]
    target_by_date = dict(zip(exec_dates, targets[market.assets].to_numpy()))
    days = dates[dates > exec_dates[0]]
    returns, rf = market.returns.loc[days, market.assets].to_numpy(), market.rf.loc[days].to_numpy()

    w = target_by_date[exec_dates[0]].copy()
    initial_turnover = np.abs(w).sum()
    out_ret, out_turnover, out_exposure = [], [], []
    for i, d in enumerate(days):
        r = returns[i]
        port = w @ r + (1 - w.sum()) * rf[i]
        w = w * (1 + r) / (1 + port)
        turnover = 0.0
        if d in target_by_date:
            target = target_by_date[d]
            new = np.where(np.abs(target - w) >= band, target, w)
            if new.sum() > 1:
                new = new / new.sum()
            turnover = np.abs(new - w).sum()
            w = new
        out_ret.append(port - turnover * cost_bps / 1e4)
        out_turnover.append(turnover)
        out_exposure.append(w.sum())
    result = pd.DataFrame({"ret": out_ret, "turnover": out_turnover, "exposure": out_exposure}, index=days)
    result.iloc[0, result.columns.get_loc("ret")] -= initial_turnover * cost_bps / 1e4
    return result


def benchmark_targets(market: Market, base: pd.DataFrame, cfg: dict) -> dict:
    """Targets and no-trade band for each benchmark in cfg['portfolio']['benchmarks'].

    Buy-and-hold is one purchase with no band, so it pays only for that purchase.
    """
    pc = cfg["portfolio"]
    spy = cfg["universe"]["benchmark"]
    n = len(market.tradable)
    zeros = pd.DataFrame(0.0, index=market.decision_dates, columns=market.assets)

    buy_hold = pd.DataFrame(0.0, index=market.decision_dates[:1], columns=market.assets)
    buy_hold[spy] = 1.0
    equal = pd.DataFrame(0.0, index=market.month_ends, columns=market.assets)
    equal[market.tradable] = 1 / n
    vol_spy = zeros.copy()
    vol_spy[spy] = (pc["vol_target"] / market.realised_vol.loc[market.decision_dates, spy]).clip(upper=pc["max_leverage"])
    close = market.close[market.tradable]
    trend_on = (close.rolling(50).mean() > close.rolling(200).mean()).astype(float)
    sma = (trend_on.loc[market.decision_dates] / n).reindex(columns=market.assets).fillna(0.0)

    available = {
        "spy_buy_hold": (buy_hold, 0.0),
        "equal_weight_monthly": (equal, pc["no_trade_band"]),
        "vol_target_spy": (vol_spy, pc["no_trade_band"]),
        "sma_50_200": (sma, pc["no_trade_band"]),
        "always_on": (base, pc["no_trade_band"]),
    }
    return {name: available[name] for name in pc["benchmarks"]}


def run_strategies(strategies: dict, market: Market, cfg: dict, cost_bps=None) -> dict:
    """Backtest every (targets, band) strategy and trim all results to their common period."""
    results = {name: backtest(targets, market, cfg, cost_bps, band) for name, (targets, band) in strategies.items()}
    common = sorted(set.intersection(*(set(r.index) for r in results.values())))
    return {name: r.loc[common] for name, r in results.items()}


def benchmark_returns(market: Market, base: pd.DataFrame, cfg: dict, cost_bps=None) -> dict:
    """Daily backtest results for each benchmark, trimmed to their common period."""
    return run_strategies(benchmark_targets(market, base, cfg), market, cfg, cost_bps)
