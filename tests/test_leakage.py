import copy

import pandas as pd
import pytest

from src import features, splits


@pytest.fixture(scope="module")
def dataset(cfg, synthetic):
    prices, macro = synthetic
    return features.build_dataset(prices, macro, cfg)


@pytest.fixture(scope="module")
def folds(cfg, synthetic):
    prices, _ = synthetic
    return splits.walk_forward_splits(prices.index, cfg)


def test_one_split_per_test_year(cfg, folds):
    years = [s.test_year for s in folds]
    assert years == list(range(cfg["splits"]["first_test_year"], cfg["splits"]["last_test_year"] + 1))


def test_blocks_are_ordered_and_disjoint(folds):
    for s in folds:
        assert s.train.max() < s.validation.min()
        assert s.validation.max() < s.test.min()
        assert (s.test.year == s.test_year).all()


def test_training_is_expanding(folds):
    for prev, nxt in zip(folds, folds[1:]):
        assert nxt.train.min() == prev.train.min()
        assert nxt.train.max() > prev.train.max()


def test_no_training_label_uses_a_later_block_price(dataset, folds):
    for s in folds:
        train = splits.select(dataset, s.train)
        validation = splits.select(dataset, s.validation)
        assert train["label_end"].max() < s.validation.min()
        assert validation["label_end"].max() < s.test.min()


def test_gap_is_11_trading_days(cfg, synthetic, folds):
    prices, _ = synthetic
    calendar = prices.index
    assert splits.gap_days(cfg) == 11
    for s in folds:
        full_validation = calendar[calendar.year == s.test_year - 1]
        assert calendar.get_loc(full_validation[0]) - calendar.get_loc(s.train.max()) == 11 + 1
        assert calendar.get_loc(s.test.min()) - calendar.get_loc(s.validation.max()) == 11 + 1


def test_validation_is_last_12_months_before_test(folds):
    for s in folds:
        assert (s.validation.year == s.test_year - 1).all()
        assert s.validation.min().month == 1


def test_test_years_are_complete(synthetic, folds):
    prices, _ = synthetic
    for s in folds:
        assert s.test.equals(prices.index[prices.index.year == s.test_year])


def test_gap_follows_the_label_settings(cfg, synthetic):
    prices, _ = synthetic
    longer = copy.deepcopy(cfg)
    longer["label"]["horizon_days"] = 10
    assert splits.gap_days(longer) == 16
    dataset = features.build_dataset(prices, synthetic[1], longer)
    for s in splits.walk_forward_splits(prices.index, longer):
        assert splits.select(dataset, s.train)["label_end"].max() < s.validation.min()
