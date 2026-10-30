import numpy as np
import pandas as pd
import pytest

from src import signals


def make_probabilities(values_by_year: dict) -> pd.DataFrame:
    rows = []
    for year, values in values_by_year.items():
        dates = pd.bdate_range(f"{year}-01-05", periods=len(values))
        rows += [{"date": d, "ticker": "XLK", "lasso": v, "lstm_42": v} for d, v in zip(dates, values)]
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def make_window_stats(stats_by_year: dict) -> pd.DataFrame:
    rows = [{"test_year": year, "model": model, "b": b, "p90": p90}
            for year, (b, p90) in stats_by_year.items() for model in ("lasso", "lstm_42")]
    return pd.DataFrame(rows).set_index(["test_year", "model"])


@pytest.fixture
def grid():
    return np.linspace(0, 1, 101)


def test_all_signals_are_between_0_and_1(cfg, grid):
    p = make_probabilities({2010: grid})
    stats = make_window_stats({2010: (0.55, 0.60)})
    for s in (signals.scaled_signal(p, stats, cfg), signals.base_rate_centred_signal(p, stats),
              signals.linear_signal(p)):
        assert ((s >= 0) & (s <= 1)).all().all()


def test_scaled_signal_is_0_at_b_and_1_from_p90(cfg):
    p = make_probabilities({2010: [0.40, 0.55, 0.575, 0.60, 0.70]})
    s = signals.scaled_signal(p, make_window_stats({2010: (0.55, 0.60)}), cfg)["lasso"].to_numpy()
    np.testing.assert_allclose(s, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_floor_keeps_the_whole_window_flat(cfg, grid):
    p = make_probabilities({2010: grid, 2011: grid})
    stats = make_window_stats({2010: (0.55, 0.555), 2011: (0.55, 0.56)})
    s = signals.scaled_signal(p, stats, cfg)
    years = s.index.get_level_values("date").year
    assert (s[years == 2010] == 0).all().all()
    assert (s[years == 2011] > 0).any().all()
    flat = signals.flat_windows(stats, cfg)
    assert flat.index.get_level_values("test_year").unique().tolist() == [2010]


def test_edge_equal_to_the_floor_is_not_flat(cfg):
    stats = make_window_stats({2010: (0.5, 0.5 + cfg["signals"]["floor"] + 1e-12)})
    assert not signals.signal_edges(stats, cfg)["flat"].any()


def test_each_window_uses_its_own_b_and_p90(cfg):
    p = make_probabilities({2010: [0.575], 2011: [0.575]})
    stats = make_window_stats({2010: (0.55, 0.60), 2011: (0.50, 0.65)})
    s = signals.scaled_signal(p, stats, cfg)["lasso"].to_numpy()
    np.testing.assert_allclose(s, [0.5, 0.5])


def test_base_rate_centred_signal(cfg):
    p = make_probabilities({2010: [0.30, 0.55, 0.775, 1.0]})
    s = signals.base_rate_centred_signal(p, make_window_stats({2010: (0.55, 0.60)}))["lasso"].to_numpy()
    np.testing.assert_allclose(s, [0.0, 0.0, 0.5, 1.0])


def test_linear_signal():
    p = make_probabilities({2010: [0.2, 0.5, 0.75, 1.0]})
    np.testing.assert_allclose(signals.linear_signal(p)["lasso"].to_numpy(), [0.0, 0.0, 0.5, 1.0])


def test_test_year_probabilities_do_not_change_b_or_p90(cfg, grid):
    stats = make_window_stats({2010: (0.55, 0.60)})
    before = signals.signal_edges(stats, cfg)
    p = make_probabilities({2010: grid})
    s_before = signals.scaled_signal(p, stats, cfg)
    s_after = signals.scaled_signal(1 - p, stats, cfg)
    pd.testing.assert_frame_equal(signals.signal_edges(stats, cfg), before)
    pd.testing.assert_frame_equal(stats, make_window_stats({2010: (0.55, 0.60)}))
    assert not s_after.equals(s_before)


def test_missing_window_raises(cfg):
    p = make_probabilities({2010: [0.6], 2011: [0.6]})
    with pytest.raises(KeyError):
        signals.scaled_signal(p, make_window_stats({2010: (0.55, 0.60)}), cfg)


def test_main_models(cfg):
    assert signals.main_models(cfg) == ["lasso", "lightgbm", f"lstm_{cfg['models']['lstm']['seeds'][0]}"]
