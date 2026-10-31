import copy

import numpy as np
import pandas as pd
import pytest

from src import evaluate, portfolio, stats


@pytest.fixture(scope="module")
def fast_cfg(cfg):
    fast = copy.deepcopy(cfg)
    fast["stats"]["bootstrap"]["n_resamples"] = 2000
    return fast


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2010-01-01", periods=2000)
    common = rng.normal(0, 0.01, len(index))
    plain = pd.Series(common + rng.normal(0, 0.002, len(index)), index=index)
    better = pd.Series(common + rng.normal(0, 0.002, len(index)) + 0.0015, index=index)
    return plain, better


def test_identical_series_are_not_significantly_different(fast_cfg, returns):
    plain, _ = returns
    boot = stats.bootstrap_sharpe_difference(plain, plain, fast_cfg)
    assert boot["Difference"] == 0 and boot[stats.P_COLUMN] > 0.05
    lw = stats.ledoit_wolf_test(plain, plain, fast_cfg)
    assert lw["LW difference"] == 0 and lw["LW p (A > B, one-sided)"] == 0.5


def test_a_clearly_higher_sharpe_is_detected(fast_cfg, returns):
    plain, better = returns
    boot = stats.bootstrap_sharpe_difference(better, plain, fast_cfg)
    assert boot["Difference"] > 0 and boot[stats.P_COLUMN] < 0.01
    assert boot["95% CI low"] > 0
    lw = stats.ledoit_wolf_test(better, plain, fast_cfg)
    assert lw["LW difference"] == pytest.approx(boot["Difference"], rel=1e-2)
    assert lw["LW p (A > B, one-sided)"] < 0.01
    assert stats.ledoit_wolf_test(plain, better, fast_cfg)["LW p (A > B, one-sided)"] > 0.99


def test_ledoit_wolf_standard_error_reflects_autocorrelation(fast_cfg):
    rng = np.random.default_rng(3)
    n = 3000
    a_iid, b_iid = rng.normal(0.0005, 0.01, n), rng.normal(0.0, 0.01, n)
    shock = rng.normal(0, 0.01, n + 20)
    ma = np.convolve(shock, np.ones(20) / np.sqrt(20), mode="valid")[:n]
    a_ac = 0.0005 + ma
    se_iid = stats.ledoit_wolf_test(a_iid, b_iid, fast_cfg)["LW std. error"]
    se_ac = stats.ledoit_wolf_test(a_ac, b_iid, fast_cfg)["LW std. error"]
    assert se_ac > 1.5 * se_iid


def test_holm():
    np.testing.assert_allclose(stats.holm([0.01, 0.04]), [0.02, 0.04])
    np.testing.assert_allclose(stats.holm([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])
    p = np.random.default_rng(0).uniform(size=20)
    assert (stats.holm(p) >= p).all() and (stats.holm(p) <= 1).all()


def test_deflated_sharpe_falls_as_trials_rise(returns):
    _, better = returns
    few = [0.2, 0.5, 0.4, 0.6, 0.3]
    many = few * 5
    assert stats.expected_max_sharpe(many) > stats.expected_max_sharpe(few)
    assert stats.deflated_sharpe(better, many)["DSR"] < stats.deflated_sharpe(better, few)["DSR"]
    assert 0 <= stats.deflated_sharpe(better, many)["DSR"] <= 1


def test_dsr_trials_reads_15_historical_trials_from_the_frozen_file(cfg):
    names = evaluate.display_names(["lasso", "lightgbm"] + [f"lstm_{s}" for s in range(42, 52)] + ["lstm_avg"])
    main = ["lasso", "lightgbm", "lstm_42"]
    rows = [names[c] + sfx for c in main for sfx in ["", ", base-rate", ", 2p-1"]] + [names["lstm_avg"]]
    performance = pd.DataFrame({"Sharpe": np.arange(len(rows), dtype=float)}, index=rows)
    trials = stats.dsr_trials(performance, names, main, cfg)
    assert len(trials) == cfg["stats"]["deflated_sharpe_trials"] == 25
    assert list(trials.index[:10]) == rows
    historical = trials.iloc[10:]
    assert len(historical) == 15
    assert "Linear (L1 selection, ridge on union), base-rate" not in historical.index
    external = {t["name"]: t["sharpe"] for t in cfg["stats"]["dsr_external_trials"]}
    assert historical.to_dict() == pytest.approx(external)


def test_diebold_mariano():
    rng = np.random.default_rng(1)
    loss = pd.Series(rng.uniform(0.2, 0.3, 1500))
    same = stats.diebold_mariano(loss, loss, horizon=5)
    assert same["DM statistic"] == 0 and same["DM p (A lower loss, one-sided)"] == 0.5
    lower = loss - 0.01 + rng.normal(0, 0.005, len(loss))
    assert stats.diebold_mariano(lower, loss, horizon=5)["DM p (A lower loss, one-sided)"] < 0.01
    assert stats.diebold_mariano(loss, lower, horizon=5)["DM p (A lower loss, one-sided)"] > 0.99


def test_forecast_losses_are_brier_averaged_across_etfs():
    index = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=2), ["A", "B"]], names=["date", "ticker"])
    p = pd.DataFrame({"lasso": [0.6, 0.4, 0.5, 0.9]}, index=index)
    y = pd.Series([1.0, 1.0, 0.0, np.nan], index=index)
    losses = stats.forecast_losses(p, y)
    np.testing.assert_allclose(losses["lasso"], [(0.16 + 0.36) / 2, 0.25])


@pytest.fixture(scope="module")
def market(cfg, synthetic):
    prices, macro = synthetic
    return portfolio.prepare_market(prices, macro, cfg, prices.index[300])


@pytest.fixture(scope="module")
def base(cfg, market):
    return portfolio.base_weights(market, cfg)


def signal_series(matrix: pd.DataFrame) -> pd.Series:
    return matrix.rename_axis(index="date", columns="ticker").stack()


@pytest.mark.parametrize("variant", ["independent", "common"])
def test_placebo_draws_keep_each_etf_exposure(cfg, market, variant):
    rng = np.random.default_rng(0)
    signals = pd.DataFrame(rng.uniform(size=(len(market.decision_dates), len(market.tradable))),
                           index=market.decision_dates, columns=market.tradable)
    min_shift = cfg["stats"]["placebo"]["min_shift_weeks"]
    for _ in range(20):
        shifted = stats.shift_signals(signals, variant, rng, min_shift)
        for tk in market.tradable:
            assert sorted(shifted[tk]) == sorted(signals[tk])
            assert not shifted[tk].equals(signals[tk])


def _actual(signals, base, market, cfg):
    targets = (signals.reindex(columns=base.columns).fillna(0.0) * base).reindex(columns=base.columns).fillna(0.0)
    result = portfolio.backtest(targets, market, cfg)
    return result.index, evaluate.portfolio_metrics(result, market.rf)


def test_a_signal_from_future_returns_beats_every_placebo_draw(cfg, market, base):
    lag = cfg["label"]["execution_lag_days"]
    exec_close = market.close.iloc[market.dates.get_indexer(market.decision_dates) + lag]
    next_exec_close = exec_close.shift(-1)
    future = (next_exec_close.to_numpy() > exec_close.to_numpy()).astype(float)
    signals = pd.DataFrame(future, index=market.decision_dates, columns=market.assets)[market.tradable]
    common, actual = _actual(signals, base, market, cfg)
    row, sharpes = stats.placebo_test(signals, base, market, cfg, common, actual, "independent", 50,
                                      np.random.default_rng(0))
    assert (sharpes < actual["Sharpe"]).all()
    assert row["p (placebo >= actual)"] == pytest.approx(1 / 51)
    assert row["Placebo mean exposure"] == pytest.approx(row["Actual exposure"], abs=0.05)


def test_a_random_signal_gives_a_roughly_uniform_placebo_p(cfg, market, base):
    rng = np.random.default_rng(11)
    p_values = []
    for _ in range(30):
        signals = pd.DataFrame(rng.uniform(size=(len(market.decision_dates), len(market.tradable))),
                               index=market.decision_dates, columns=market.tradable)
        common, actual = _actual(signals, base, market, cfg)
        row, _ = stats.placebo_test(signals, base, market, cfg, common, actual, "independent", 19, rng)
        p_values.append(row["p (placebo >= actual)"])
    p_values = np.array(p_values)
    assert 0.35 < p_values.mean() < 0.65
    assert (p_values <= 0.1).mean() < 0.3 and (p_values >= 0.9).mean() < 0.3


def test_run_placebo_table(cfg, market, base):
    quick = copy.deepcopy(cfg)
    quick["run"]["quick_mode"] = True
    quick["stats"]["placebo"]["quick_draws"] = 5
    rng = np.random.default_rng(2)
    signals = pd.DataFrame(rng.uniform(size=(len(market.decision_dates), len(market.tradable))),
                           index=market.decision_dates, columns=market.tradable)
    main = pd.DataFrame({"lasso": signal_series(signals), "lightgbm": signal_series(1 - signals)})
    names = {"lasso": "Linear", "lightgbm": "LightGBM"}
    results = {n: portfolio.backtest(portfolio.target_weights(main[c], base), market, quick) for c, n in names.items()}
    common = results["Linear"].index
    performance = evaluate.performance_table(results, market.rf)
    table, draws = stats.run_placebo(main, base, market, quick, common, performance, names)
    assert list(table.index) == [(n, v) for v in quick["stats"]["placebo"]["variants"] for n in names.values()]
    assert draws.shape == (5, 4)
    assert (table["Draws"] == 5).all()
