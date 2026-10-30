"""Runs the pipeline on generated prices, as far as it is implemented.

Extend this test stage by stage as models, signals, portfolio, evaluate and
stats are filled in, until it covers the whole run (quality gate 2).
"""

import copy

import pandas as pd
import pytest

from src import features, models, portfolio, signals, splits


def test_data_to_splits_on_synthetic_prices(cfg, synthetic):
    prices, macro = synthetic
    dataset = features.build_dataset(prices, macro, cfg)
    folds = splits.walk_forward_splits(prices.index, cfg)
    weekly = features.decision_dates(prices.index, cfg["label"]["decision_frequency"])
    columns = features.feature_columns(dataset)

    assert len(columns) == 31
    assert len(folds) > 0
    for s in folds:
        for block in (s.train, s.validation, s.test):
            rows = splits.select(dataset, block.intersection(weekly))
            assert len(rows) > 0
            assert rows[columns].notna().all().all()
        assert splits.select(dataset, s.train)["label"].notna().all()


def test_models_signals_and_portfolio_on_synthetic_prices(cfg, synthetic):
    pytest.importorskip("tensorflow")
    quick = copy.deepcopy(cfg)
    quick["run"]["quick_mode"] = True
    prices, macro = synthetic
    dataset = features.build_dataset(prices, macro, quick)
    folds = splits.walk_forward_splits(prices.index, quick)

    result = models.run_walk_forward(dataset, folds, quick)

    settings = models.model_settings(quick)
    columns = ["lasso", "lightgbm"] + models.lstm_columns(settings)
    years = [s.test_year for s in folds]
    test_rows = dataset[dataset.index.get_level_values("date").isin(folds[0].test.append([s.test for s in folds[1:]]))]
    assert list(result.probabilities.columns) == columns
    assert result.probabilities.index.equals(test_rows.index)
    assert result.probabilities.notna().all().all()
    assert result.window_stats.index.equals(pd.MultiIndex.from_product([years, columns], names=["test_year", "model"]))
    assert list(result.tuning.index) == years
    assert set(result.selections["model"]) == set(models.SELECTORS)
    edges = models.grid_edges(result.tuning, quick)
    assert (edges["windows"] == len(years)).all()

    main = signals.scaled_signal(result.probabilities, result.window_stats, quick)
    main_models = result.probabilities[signals.main_models(quick)]
    sensitivity = [signals.base_rate_centred_signal(main_models, result.window_stats), signals.linear_signal(main_models)]
    assert main.shape == result.probabilities.shape
    assert list(sensitivity[0].columns) == signals.main_models(quick)
    for s in [main, *sensitivity]:
        assert s.notna().all().all()
        assert ((s >= 0) & (s <= 1)).all().all()
    assert signals.signal_edges(result.window_stats, quick).index.equals(result.window_stats.index)

    market = portfolio.prepare_market(prices, macro, quick, folds[0].test[0])
    base = portfolio.base_weights(market, quick)
    strategies = portfolio.benchmark_targets(market, base, quick)
    band = quick["portfolio"]["no_trade_band"]
    for model in main.columns:
        strategies[model] = (portfolio.target_weights(main[model], base), band)
    for name, sensitivity_signal in zip(["base_rate", "linear"], sensitivity):
        for model in sensitivity_signal.columns:
            strategies[f"{model}_{name}"] = (portfolio.target_weights(sensitivity_signal[model], base), band)
    backtests = portfolio.run_strategies(strategies, market, quick)

    first = next(iter(backtests.values())).index
    assert len(backtests) == len(quick["portfolio"]["benchmarks"]) + main.shape[1] + 2 * len(signals.main_models(quick))
    for r in backtests.values():
        assert r.index.equals(first)
        assert r.notna().all().all()
        assert (r["exposure"] >= 0).all() and (r["exposure"] <= 1 + 1e-12).all()
    assert first[0] > folds[0].test[0] and first[-1] == prices.index[-1]
