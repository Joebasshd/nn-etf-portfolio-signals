import numpy as np
import pandas as pd
import pytest

from src import portfolio


@pytest.fixture(scope="module")
def market(cfg, synthetic):
    prices, macro = synthetic
    return portfolio.prepare_market(prices, macro, cfg, prices.index[300])


@pytest.fixture(scope="module")
def base(cfg, market):
    return portfolio.base_weights(market, cfg)


def all_ones(base: pd.DataFrame, tickers: list) -> pd.Series:
    index = pd.MultiIndex.from_product([base.index, tickers], names=["date", "ticker"])
    return pd.Series(1.0, index=index)


def test_weights_are_long_only_capped_and_unlevered(cfg, base):
    assert (base >= 0).all().all()
    assert (base <= cfg["portfolio"]["max_weight"] + 1e-12).all().all()
    assert (base.sum(axis=1) <= cfg["portfolio"]["max_leverage"] + 1e-12).all()
    assert (base[cfg["universe"]["benchmark"]] == 0).all()


def test_weights_at_t_ignore_later_prices(cfg, synthetic, market, base):
    prices, macro = synthetic
    cutoff = market.decision_dates[len(market.decision_dates) // 2]
    changed = prices.copy()
    later = changed.index > cutoff
    changed.loc[later] = changed.loc[later] * np.random.default_rng(3).uniform(0.5, 1.5, size=(later.sum(), prices.shape[1]))
    changed_market = portfolio.prepare_market(changed, macro, cfg, prices.index[300])
    changed_base = portfolio.base_weights(changed_market, cfg)
    pd.testing.assert_frame_equal(changed_base.loc[:cutoff], base.loc[:cutoff])


def test_a_decision_at_t_earns_nothing_before_close_t_plus_1(cfg, synthetic):
    prices, macro = synthetic
    ticker = cfg["universe"]["tradable"][0]
    t = prices.index[600]
    jumped = prices.copy()
    after_t = jumped.index > t
    for field in ("open", "high", "low", "close", "adj_close"):
        jumped.loc[after_t, (field, ticker)] *= 1.5
    market = portfolio.prepare_market(jumped, macro, cfg, prices.index[300])
    targets = pd.DataFrame(0.0, index=[t], columns=market.assets)
    targets[ticker] = 1.0

    result = portfolio.backtest(targets, market, cfg, cost_bps=0, band=0.0)
    lag = cfg["label"]["execution_lag_days"]
    executed = market.dates[market.dates.get_loc(t) + lag]
    assert market.returns.loc[market.dates[market.dates.get_loc(t) + 1], ticker] == pytest.approx(0.5, rel=0.1)
    assert result.index[0] > executed
    assert (result["ret"] < 0.4).all()
    np.testing.assert_allclose(result["ret"], market.returns.loc[result.index, ticker], atol=1e-12)


def test_costs_are_turnover_times_bps(cfg, market, base):
    gross = portfolio.backtest(base, market, cfg, cost_bps=0)
    net = portfolio.backtest(base, market, cfg, cost_bps=5)
    pd.testing.assert_series_equal(gross["turnover"], net["turnover"])
    cost = gross["ret"] - net["ret"]
    expected = gross["turnover"] * 5 / 1e4
    expected.iloc[0] += base.iloc[0].abs().sum() * 5 / 1e4
    np.testing.assert_allclose(cost, expected, atol=1e-15)
    assert portfolio.backtest(base, market, cfg, cost_bps=0)["ret"].equals(gross["ret"])


def test_spy_buy_and_hold_reproduces_spy(cfg, market, base):
    spy = cfg["universe"]["benchmark"]
    result = portfolio.benchmark_returns(market, base, cfg, cost_bps=0)["spy_buy_hold"]
    np.testing.assert_allclose(result["ret"], market.returns.loc[result.index, spy], atol=1e-14)
    assert (result["turnover"] == 0).all()


def test_always_on_equals_main_strategy_with_every_signal_1(cfg, market, base):
    targets = portfolio.target_weights(all_ones(base, market.tradable), base)
    pd.testing.assert_frame_equal(targets, base)
    always_on = portfolio.benchmark_returns(market, base, cfg)["always_on"]
    main = portfolio.backtest(targets, market, cfg).loc[always_on.index]
    pd.testing.assert_frame_equal(main, always_on)


def test_always_on_volatility_is_near_target(cfg, market, base):
    always_on = portfolio.backtest(base, market, cfg)
    realised = always_on["ret"].std() * np.sqrt(252)
    assert abs(realised - cfg["portfolio"]["vol_target"]) < 0.02


def test_no_position_earns_the_cash_rate(cfg, market, base):
    result = portfolio.backtest(base * 0.0, market, cfg)
    np.testing.assert_allclose(result["ret"], market.rf.loc[result.index], atol=1e-15)
    assert (result["exposure"] == 0).all()


def test_benchmarks_and_common_period(cfg, market, base):
    results = portfolio.benchmark_returns(market, base, cfg)
    assert list(results) == cfg["portfolio"]["benchmarks"]
    first = next(iter(results.values())).index
    assert all(r.index.equals(first) for r in results.values())
    month_end_exec = market.dates[market.dates.get_loc(market.month_ends[0]) + cfg["label"]["execution_lag_days"]]
    assert first[0] > month_end_exec
    weights = portfolio.benchmark_targets(market, base, cfg)
    assert ((weights["vol_target_spy"][0] >= 0) & (weights["vol_target_spy"][0] <= 1)).all().all()
    sma = weights["sma_50_200"][0][market.tradable]
    assert set(np.unique(sma.to_numpy().round(12))) <= {0.0, round(1 / len(market.tradable), 12)}
