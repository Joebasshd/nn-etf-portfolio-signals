"""Runs the pipeline on generated prices, as far as it is implemented.

Extend this test stage by stage as models, signals, portfolio, evaluate and
stats are filled in, until it covers the whole run (quality gate 2).
"""

from src import features, splits


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
