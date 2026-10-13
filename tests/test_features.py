import numpy as np
import pandas as pd

from src import data, features


def test_31_features(cfg, synthetic):
    prices, macro = synthetic
    panel = features.build_features(prices, macro, cfg)
    assert panel.shape[1] == 31
    assert set(panel.index.get_level_values("ticker")) == set(cfg["universe"]["tradable"])
    clean = panel.dropna()
    assert len(clean) > 0
    assert np.isfinite(clean.to_numpy()).all()


def test_features_do_not_use_future_data(cfg, synthetic):
    """Changing every price and macro value after a cut-off date must leave
    all features on or before that date unchanged."""
    prices, macro = synthetic
    cutoff = prices.index[1000]
    later = prices.index > cutoff

    rng = np.random.default_rng(0)
    changed_prices = prices.copy()
    changed_prices.loc[later] = changed_prices.loc[later] * rng.uniform(0.5, 1.5, size=(later.sum(), prices.shape[1]))
    changed_macro = macro.copy()
    changed_macro.loc[later] = changed_macro.loc[later] + 5.0

    before = features.build_features(prices, macro, cfg).loc[:cutoff]
    after = features.build_features(changed_prices, changed_macro, cfg).loc[:cutoff]
    pd.testing.assert_frame_equal(before, after)


def test_macro_is_lagged_one_day(cfg):
    calendar = pd.bdate_range("2020-01-01", periods=10)
    raw = pd.DataFrame({"VIX": np.arange(10.0)}, index=calendar)
    aligned = data.align_macro(raw, calendar, lag_days=1)
    assert np.isnan(aligned["VIX"].iloc[0])
    assert (aligned["VIX"].iloc[1:].to_numpy() == raw["VIX"].iloc[:-1].to_numpy()).all()


def test_macro_alignment_never_back_fills(cfg):
    calendar = pd.bdate_range("2020-01-01", periods=10)
    raw = pd.DataFrame({"VIX": [20.0]}, index=[calendar[5]])
    aligned = data.align_macro(raw, calendar, lag_days=1)
    assert aligned["VIX"].iloc[:6].isna().all()
    assert (aligned["VIX"].iloc[6:] == 20.0).all()


def test_label_is_execution_aligned(cfg, synthetic):
    """Label at t is the return from close t+1 to close t+6."""
    prices, _ = synthetic
    labels = features.build_labels(prices, cfg)
    ticker = cfg["universe"]["tradable"][0]
    px = prices["adj_close"][ticker]
    start = px.index.get_loc(pd.Timestamp(cfg["features"]["sample_start"]))

    for i in (start, start + 37, start + 400):
        row = labels.loc[(px.index[i], ticker)]
        expected = px.iloc[i + 6] / px.iloc[i + 1] - 1
        assert np.isclose(row["fwd_ret"], expected)
        assert row["label"] == float(expected > 0)
        assert row["label_end"] == px.index[i + 6]


def test_label_unknown_at_end_of_sample(cfg, synthetic):
    prices, _ = synthetic
    labels = features.build_labels(prices, cfg)
    last_dates = prices.index[-6:]
    tail = labels[labels.index.get_level_values("date").isin(last_dates)]
    assert tail["label"].isna().all()
    assert tail["fwd_ret"].isna().all()


def test_weekly_decision_dates(cfg, synthetic):
    prices, _ = synthetic
    dates = features.decision_dates(prices.index, "weekly")
    assert dates.isin(prices.index).all()
    assert (dates.dayofweek == 4).all()  # synthetic calendar has no holidays
    assert dates.to_period("W-FRI").is_unique


def test_five_indicator_families(cfg, synthetic):
    prices, macro = synthetic
    columns = features.build_features(prices, macro, cfg).columns
    counts = pd.Series([features.feature_family(c) for c in columns]).value_counts().to_dict()
    assert counts == {"trend": 8, "momentum": 10, "volatility": 5, "volume": 4, "market context": 4}


def test_etf_enters_after_252_days_of_history(cfg, synthetic):
    """An ETF listed late has no rows until its longest indicator has a value."""
    prices, macro = synthetic
    late = cfg["universe"]["tradable"][-1]
    start = prices.index.get_loc(pd.Timestamp(cfg["features"]["sample_start"])) + 50
    prices = prices.copy()
    prices.loc[prices.index[:start], (slice(None), late)] = np.nan

    dataset = features.build_dataset(prices, macro, cfg)
    first = dataset.xs(late, level="ticker").index.min()
    assert first == prices.index[start + 252]
    other = dataset.xs(cfg["universe"]["tradable"][0], level="ticker").index.min()
    assert other < first
