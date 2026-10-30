import copy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from src import features, models, splits  # noqa: E402


@pytest.fixture(scope="module")
def quick_cfg(cfg):
    quick = copy.deepcopy(cfg)
    quick["run"]["quick_mode"] = True
    return quick


@pytest.fixture(scope="module")
def dataset(cfg, synthetic):
    prices, macro = synthetic
    return features.build_dataset(prices, macro, cfg)


@pytest.fixture(scope="module")
def last_fold(cfg, synthetic):
    prices, _ = synthetic
    return splits.walk_forward_splits(prices.index, cfg)[-1]


@pytest.fixture(scope="module")
def result(quick_cfg, dataset, last_fold):
    return models.run_walk_forward(dataset, [last_fold], quick_cfg)


def test_probability_columns_and_range(quick_cfg, result):
    seeds = quick_cfg["models"]["lstm"]["seeds"][: quick_cfg["run"]["quick"]["lstm_seeds"]]
    expected = ["lasso", "lightgbm"] + [f"lstm_{s}" for s in seeds] + ["lstm_avg"]
    assert list(result.probabilities.columns) == expected
    p = result.probabilities.to_numpy()
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()


def test_lightgbm_selects_exactly_top_n(quick_cfg, result):
    counts = result.selections.query("model == 'lightgbm'").groupby("test_year").size()
    assert (counts == quick_cfg["models"]["lightgbm"]["top_n_features"]).all()


def test_union_is_lasso_or_lightgbm_selection(result):
    for year, rows in result.selections.groupby("test_year"):
        by_model = rows.groupby("model")["feature"].apply(set)
        assert by_model["union"] == by_model["lasso"] | by_model["lightgbm"]


def test_every_forecasting_model_uses_the_union(result):
    for year, inputs in result.inputs.items():
        union = result.selections.query("test_year == @year and model == 'union'")["feature"].tolist()
        assert set(inputs) == {"lasso", "lightgbm", *[c for c in result.probabilities if c.startswith("lstm_") and c != "lstm_avg"]}
        assert all(cols == union for cols in inputs.values())


def test_ridge_keeps_every_union_feature(quick_cfg, dataset, last_fold):
    settings = models.model_settings(quick_cfg)
    names = features.feature_columns(dataset)
    dates = dataset.index.get_level_values("date")
    known = dataset["label"].notna()
    train = dataset[dates.isin(last_fold.train) & known]
    val = dataset[dates.isin(last_fold.validation) & known]
    X_train = (train[names] - train[names].mean()) / train[names].std(ddof=0)
    X_val = (val[names] - train[names].mean()) / train[names].std(ddof=0)
    union = models.select_features(X_train.to_numpy(), train["label"].to_numpy(), X_val.to_numpy(),
                                   val["label"].to_numpy(), names, settings)["union"]
    ridge = models.fit_ridge(X_train[union].to_numpy(), train["label"].to_numpy(),
                             X_val[union].to_numpy(), val["label"].to_numpy(), settings)
    assert ridge["model"].coef_.shape[1] == len(union)
    assert (ridge["model"].coef_ != 0).all()


def test_changing_test_year_data_leaves_the_fit_unchanged(quick_cfg, dataset, last_fold, result):
    changed = dataset.copy()
    in_test = changed.index.get_level_values("date").isin(last_fold.test)
    names = features.feature_columns(dataset)
    rng = np.random.default_rng(1)
    changed.loc[in_test, names] = changed.loc[in_test, names] * rng.uniform(0.5, 1.5, size=(in_test.sum(), len(names)))
    changed.loc[in_test, "label"] = 1 - changed.loc[in_test, "label"]

    again = models.run_walk_forward(changed, [last_fold], quick_cfg)
    year = last_fold.test_year
    np.testing.assert_array_equal(again.scalers[year].mean_, result.scalers[year].mean_)
    np.testing.assert_array_equal(again.scalers[year].scale_, result.scalers[year].scale_)
    pd.testing.assert_frame_equal(again.selections, result.selections)
    pd.testing.assert_frame_equal(again.window_stats, result.window_stats)
    pd.testing.assert_frame_equal(again.tuning, result.tuning)


def test_same_seed_gives_same_probabilities(quick_cfg, dataset, last_fold, result):
    again = models.run_walk_forward(dataset, [last_fold], quick_cfg)
    pd.testing.assert_frame_equal(again.probabilities, result.probabilities)


def test_a_seed_does_not_depend_on_the_other_seeds(quick_cfg, dataset, last_fold, result):
    alone = copy.deepcopy(quick_cfg)
    alone["run"]["quick"]["lstm_seeds"] = 1
    single = models.run_walk_forward(dataset, [last_fold], alone)
    first = f"lstm_{quick_cfg['models']['lstm']['seeds'][0]}"
    assert "lstm_avg" not in single.probabilities
    pd.testing.assert_frame_equal(single.probabilities[["lasso", "lightgbm", first]],
                                  result.probabilities[["lasso", "lightgbm", first]])


def test_seed_average_is_the_mean_of_the_seeds(result):
    seeds = [c for c in result.probabilities if c.startswith("lstm_") and c != "lstm_avg"]
    expected = result.probabilities[seeds].mean(axis=1)
    np.testing.assert_allclose(result.probabilities["lstm_avg"], expected, rtol=0, atol=1e-12)
    year = next(iter(result.validation_probabilities))
    val = result.validation_probabilities[year]
    np.testing.assert_allclose(val["lstm_avg"], np.mean([val[s] for s in seeds], axis=0), rtol=0, atol=1e-12)


def test_b_and_p90_come_from_training_and_validation(quick_cfg, dataset, last_fold, result):
    dates = dataset.index.get_level_values("date")
    b = dataset.loc[dates.isin(last_fold.train), "label"].mean()
    stats = result.window_stats.loc[last_fold.test_year]
    assert np.allclose(stats["b"], b)
    for model, p_val in result.validation_probabilities[last_fold.test_year].items():
        assert np.isclose(stats.loc[model, "p90"], np.quantile(p_val, quick_cfg["signals"]["p90_quantile"]))


def test_no_sequence_uses_a_later_date():
    dates = pd.bdate_range("2020-01-01", periods=60)
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"])
    day_number = pd.DataFrame({"day": np.repeat(np.arange(60.0), 2)}, index=index)
    keep = dates[25:]
    X, seq_index = models.make_sequences(day_number, keep, length=20)
    assert len(X) == 2 * len(keep)
    for window, (date, _) in zip(X, seq_index):
        last = dates.get_loc(date)
        assert window[-1, 0] == last
        assert (window[:, 0] <= last).all()
        assert list(window[:, 0]) == list(range(last - 19, last + 1))


def test_grid_edges_counts_edge_choices(cfg):
    grid = cfg["models"]["lasso"]["C_grid"]
    ridge = cfg["models"]["lasso"]["ridge_C_grid"]
    tuning = pd.DataFrame({"lasso_C": [grid[0], grid[0], grid[-1] * 10, grid[1]],
                           "ridge_C": [ridge[0], ridge[2], ridge[-1], ridge[3]]})
    edges = models.grid_edges(tuning, cfg)
    assert edges.loc["lasso_C", ["at_lower_edge", "at_upper_edge"]].tolist() == [2, 1]
    assert edges.loc["ridge_C", ["at_lower_edge", "at_upper_edge"]].tolist() == [1, 1]
