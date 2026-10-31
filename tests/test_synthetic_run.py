"""Runs the pipeline on generated prices, as far as it is implemented.

Extend this test stage by stage as models, signals, portfolio, evaluate and
stats are filled in, until it covers the whole run (quality gate 2).
"""

import copy

import pandas as pd
import pytest

from src import evaluate, features, models, portfolio, signals, splits, stats


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


def test_whole_pipeline_on_synthetic_prices(cfg, synthetic):
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
    names = evaluate.display_names(result.probabilities.columns)
    strategies = evaluate.strategy_targets(main, dict(zip(["base_rate", "linear"], sensitivity)), base, market, quick, names)
    backtests = portfolio.run_strategies(strategies, market, quick)

    first = next(iter(backtests.values())).index
    assert len(backtests) == len(quick["portfolio"]["benchmarks"]) + main.shape[1] + 2 * len(signals.main_models(quick))
    for r in backtests.values():
        assert r.index.equals(first)
        assert r.notna().all().all()
        assert (r["exposure"] >= 0).all() and (r["exposure"] <= 1 + 1e-12).all()
    assert first[0] > folds[0].test[0] and first[-1] == prices.index[-1]

    labels = dataset["label"]
    rf = market.rf
    performance = evaluate.performance_table(backtests, rf)
    assert performance.index.equals(pd.Index(list(backtests)))
    assert evaluate.prediction_metrics(result.probabilities, labels, names).shape[0] == len(names)
    assert evaluate.auc_by_year(result.probabilities, labels, names).index.tolist() == years
    calibration = evaluate.calibration_table(result.probabilities, labels, names, quick["stats"]["calibration_bins"])
    assert calibration.groupby("model")["n"].sum().nunique() == 1
    feature_names = features.feature_columns(dataset)
    assert evaluate.stability_summary(result.selections, feature_names).shape[0] == 3
    costs = evaluate.cost_sensitivity({n: strategies[n] for n in names.values()}, market, quick, first)
    assert costs.shape == (len(names), len(quick["portfolio"]["cost_bps_sensitivity"]) + 1)
    sub = evaluate.subperiod_metrics({n: backtests[n] for n in names.values()}, rf, result.probabilities, labels, names, quick)
    assert set(sub["strategy"]) == set(names.values())

    tests = stats.sharpe_tests(backtests, rf, names, quick)
    n_lstm = sum(c.startswith("lstm_") for c in names)
    assert len(tests) == 2 * n_lstm + len(names)
    assert tests["p Holm (H4)"].notna().sum() == 2 * n_lstm
    placebo, draws = stats.run_placebo(main, base, market, quick, first, performance, names)
    assert len(placebo) == len(names) * len(quick["stats"]["placebo"]["variants"])
    assert (placebo["Draws"] <= quick["stats"]["placebo"]["quick_draws"]).all()
    assert len(draws) == quick["stats"]["placebo"]["quick_draws"]
    trials = stats.dsr_trials(performance, names, signals.main_models(quick), quick)
    dsr = stats.dsr_table(backtests, rf, list(names.values()), trials)
    assert len(trials) == quick["stats"]["deflated_sharpe_trials"]
    assert (dsr["Trials"] == trials.notna().sum()).all()
    dm = stats.diebold_mariano_tests(stats.forecast_losses(result.probabilities, labels), names, quick)
    assert len(dm) == 2 * n_lstm
    edges = signals.signal_edges(result.window_stats, quick).reset_index().assign(model=lambda e: e["model"].map(names))
    seeds = evaluate.lstm_seed_summary(names, evaluate.prediction_metrics(result.probabilities, labels, names),
                                       performance, edges, dsr, placebo, tests, quick["stats"]["placebo"]["variants"])
    assert names["lstm_avg"] in seeds.index
