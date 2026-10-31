import copy
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from src import evaluate


def series_result(returns, turnover=None, exposure=None, cost=None) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=len(returns))
    n = len(returns)
    return pd.DataFrame({"ret": returns, "turnover": turnover if turnover is not None else [0.0] * n,
                         "exposure": exposure if exposure is not None else [1.0] * n,
                         "cost": cost if cost is not None else [0.0] * n}, index=index)


def test_portfolio_metrics_match_hand_calculation():
    r = [0.01, -0.02, 0.03, 0.0]
    result = series_result(r, turnover=[0.5, 0.0, 0.25, 0.0], exposure=[1.0, 0.5, 0.5, 0.0],
                           cost=[0.0005, 0.0, 0.00025, 0.0])
    rf = pd.Series(0.0, index=result.index)
    m = evaluate.portfolio_metrics(result, rf)
    assert m["Ann. return"] == pytest.approx((1.01 * 0.98 * 1.03) ** (252 / 4) - 1)
    assert m["Sharpe"] == pytest.approx(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252))
    assert m["Sortino"] == pytest.approx(np.mean(r) * 252 / (np.sqrt(0.02 ** 2 / 4) * np.sqrt(252)))
    assert m["Max drawdown"] == pytest.approx(-0.02)
    assert m["Turnover / yr"] == pytest.approx(0.75 / (4 / 252))
    assert m["Avg exposure"] == pytest.approx(0.5)
    assert m["Total cost"] == pytest.approx(0.00075)


def test_no_position_has_no_sharpe():
    result = series_result([0.0001] * 10)
    rf = pd.Series(0.0001, index=result.index)
    assert np.isnan(evaluate.portfolio_metrics(result, rf)["Sharpe"])


def test_breakeven_cost():
    result = series_result([0.001] * 252, turnover=[0.1] * 252)
    rf = pd.Series(0.0, index=result.index)
    assert evaluate.breakeven_cost(result, rf) == pytest.approx(0.252 / 25.2 * 1e4)


def test_prediction_metrics_on_a_perfect_ranking():
    index = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=4), ["A"]], names=["date", "ticker"])
    p = pd.DataFrame({"lasso": [0.2, 0.4, 0.6, 0.8]}, index=index)
    y = pd.Series([0.0, 0.0, 1.0, 1.0], index=index)
    m = evaluate.prediction_metrics(p, y, {"lasso": "Linear"}).loc["Linear"]
    assert m["AUC"] == 1.0 and m["accuracy"] == 1.0 and m["base_rate"] == 0.5
    assert m["Brier"] == pytest.approx(np.mean([0.04, 0.16, 0.16, 0.04]))


def test_calibration_table():
    rng = np.random.default_rng(0)
    index = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=500), ["A", "B"]], names=["date", "ticker"])
    p = pd.DataFrame({"lasso": rng.uniform(0.45, 0.65, len(index))}, index=index)
    y = pd.Series((rng.uniform(size=len(index)) < p["lasso"]).astype(float), index=index)
    table = evaluate.calibration_table(p, y, {"lasso": "Linear"}, n_bins=10)
    assert len(table) == 10
    assert table["n"].sum() == len(index)
    assert table["mean_p"].is_monotonic_increasing


def selections_table(sets_by_year: dict, selector: str = "lasso") -> pd.DataFrame:
    return pd.DataFrame([{"test_year": y, "model": selector, "feature": f} for y, s in sets_by_year.items() for f in s])


def test_jaccard():
    assert evaluate.jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert evaluate.jaccard({"a"}, {"b"}) == 0.0
    same = pd.concat([selections_table({2010: ["mom_rsi", "vol_atr_norm"], 2011: ["mom_rsi", "vol_atr_norm"]}, s)
                      for s in evaluate.SELECTOR_NAMES])
    assert (evaluate.jaccard_overlap(same) == 1.0).all()
    disjoint = pd.concat([selections_table({2010: ["mom_rsi"], 2011: ["vol_atr_norm"]}, s) for s in evaluate.SELECTOR_NAMES])
    assert (evaluate.jaccard_overlap(disjoint) == 0.0).all()


def test_family_shares_sum_to_one_and_frequency():
    sel = pd.concat([selections_table({2010: ["mom_rsi", "vol_atr_norm", "trend_adx"], 2011: ["mom_rsi"]}, s)
                     for s in evaluate.SELECTOR_NAMES])
    shares = evaluate.family_share(sel)
    np.testing.assert_allclose(shares.sum(), 1.0)
    assert shares.loc["momentum", "LASSO"] == pytest.approx((1 / 3 + 1) / 2)
    freq = evaluate.selection_frequency(sel, ["mom_rsi", "vol_atr_norm", "trend_adx", "volu_cmf"])
    assert freq.loc["mom_rsi", "LASSO"] == 1.0 and freq.loc["trend_adx", "LASSO"] == 0.5 and freq.loc["volu_cmf", "LASSO"] == 0.0
    assert freq.loc["mom_rsi", "family"] == "momentum"


def test_display_names():
    names = evaluate.display_names(["lasso", "lightgbm", "lstm_42", "lstm_43", "lstm_avg"])
    assert names == {"lasso": "Linear (L1 selection, ridge on union)", "lightgbm": "LightGBM", "lstm_42": "LSTM",
                     "lstm_43": "LSTM (seed 43)", "lstm_avg": "LSTM (average of 2 seeds)"}


def test_subperiods(cfg):
    calendar = pd.bdate_range("2008-02-04", "2026-06-29")
    periods = evaluate.subperiods(cfg, calendar)
    assert list(periods) == ["GFC", "COVID", "Rate shock 2022", "Before 2018 break", "After 2018 break"]
    assert periods["Before 2018 break"][1] < periods["After 2018 break"][0] == pd.Timestamp(cfg["subperiods"]["break_2018"])


def test_results_dir_and_run_info(cfg, tmp_path):
    local = copy.deepcopy(cfg)
    local["run"]["results_dir"] = str(tmp_path)
    clock = evaluate.start_clock()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = evaluate.results_dir(local, "unit-test")
    assert out.parent == tmp_path and out.name.endswith("_unit-test")
    assert (out / "figures").is_dir() and (out / "large").is_dir()
    evaluate.write_run_info(out, local, clock, label="unit-test", data_source="synthetic")
    info = json.loads((out / "run_info.json").read_text())
    assert info["label"] == "unit-test" and info["data_source"] == "synthetic"
    assert {"commit", "runtime_minutes", "cpu_minutes", "config"} <= set(info)



def test_save_tables(tmp_path):
    (tmp_path / "large").mkdir()
    evaluate.save_tables(tmp_path, {"small": pd.DataFrame({"a": [1, 2]}), "indexed": pd.DataFrame({"a": [1]}, index=["x"])},
                         large={"big": pd.DataFrame({"b": [3]})})
    years = pd.DataFrame({"C": [0.1, 0.2]}, index=pd.RangeIndex(2008, 2010, name="test_year"))
    evaluate.save_tables(tmp_path, {"years": years})
    assert pd.read_csv(tmp_path / "years.csv").columns.tolist() == ["test_year", "C"]
    assert pd.read_csv(tmp_path / "small.csv").columns.tolist() == ["a"]
    assert pd.read_csv(tmp_path / "indexed.csv", index_col=0).index.tolist() == ["x"]
    assert (tmp_path / "large" / "big.csv").exists()
