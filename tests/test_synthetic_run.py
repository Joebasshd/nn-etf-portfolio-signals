"""Runs the pipeline on generated prices, as far as it is implemented.

Extend this test stage by stage as models, signals, portfolio, evaluate and
stats are filled in, until it covers the whole run (quality gate 2).
"""

import copy

import pandas as pd
import pytest

from src import features, models, splits


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


def test_models_on_synthetic_prices(cfg, synthetic):
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
