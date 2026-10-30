"""Walk-forward selection and forecasting: LASSO and LightGBM select features; ridge, LightGBM and LSTM forecast from their union.

The linear forecasting model keeps the code key "lasso" but is a ridge (L2)
logistic regression on the union; report it as "Linear (L1 selection, ridge on union)".
"""

import copy
import os
import random
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from src import features, splits

MODEL_NAMES = ["lasso", "lightgbm", "lstm"]
SELECTORS = ["lasso", "lightgbm", "union"]
PREDICT_BATCH_SIZE = 2048


@dataclass
class WalkForwardResult:
    probabilities: pd.DataFrame
    validation_probabilities: dict
    selections: pd.DataFrame
    window_stats: pd.DataFrame
    tuning: pd.DataFrame
    inputs: dict = field(default_factory=dict)
    scalers: dict = field(default_factory=dict)


def model_settings(cfg: dict) -> dict:
    """Model settings from config.yaml, with the quick-mode overrides applied."""
    m = copy.deepcopy(cfg["models"])
    m["lightgbm"]["params"].update(random_state=cfg["seed"], verbose=-1)
    if cfg["run"]["quick_mode"]:
        quick = cfg["run"]["quick"]
        m["lstm"]["seeds"] = m["lstm"]["seeds"][: quick["lstm_seeds"]]
        m["lstm"]["max_epochs"] = quick["lstm_max_epochs"]
        m["lightgbm"]["params"]["n_estimators"] = quick["lightgbm_n_estimators"]
    return m


def lstm_columns(settings: dict) -> list:
    """Probability columns for the LSTM: one per seed, plus the seed average if enabled."""
    seeds = settings["lstm"]["seeds"]
    average = ["lstm_avg"] if settings["lstm"]["seed_average"] and len(seeds) > 1 else []
    return [f"lstm_{s}" for s in seeds] + average


def fit_lasso_selection(X_train, y_train, X_val, y_val, feature_names: list, settings: dict) -> dict:
    """L1 logistic regression with C chosen by validation log loss; keeps the non-zero features.

    A C that zeroes every coefficient is skipped; if all do, the weakest penalty x10 is used.
    """
    grid = settings["lasso"]["C_grid"]
    best = None
    for C in grid:
        model = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=5000).fit(X_train, y_train)
        coef = model.coef_.ravel()
        if not np.any(coef != 0):
            continue
        loss = log_loss(y_val, model.predict_proba(X_val)[:, 1])
        if best is None or loss < best["log_loss"]:
            best = {"log_loss": loss, "C": C, "selected": [f for f, c in zip(feature_names, coef) if c != 0]}
    if best is None:
        C = max(grid) * 10
        model = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=5000).fit(X_train, y_train)
        selected = [f for f, c in zip(feature_names, model.coef_.ravel()) if c != 0] or list(feature_names)
        best = {"log_loss": np.nan, "C": C, "selected": selected}
    return best


def fit_lightgbm(X_train, y_train, X_val, y_val, settings: dict) -> lgb.LGBMClassifier:
    """LightGBM with the fixed settings and early stopping on validation log loss."""
    gbm = settings["lightgbm"]
    stop = [lgb.early_stopping(gbm["early_stopping_rounds"], verbose=False)]
    return lgb.LGBMClassifier(**gbm["params"]).fit(
        X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=stop
    )


def fit_lightgbm_selection(X_train, y_train, X_val, y_val, feature_names: list, settings: dict) -> list:
    """The top features of a LightGBM fit on all features, by gain importance."""
    model = fit_lightgbm(X_train, y_train, X_val, y_val, settings)
    importance = pd.Series(model.booster_.feature_importance(settings["lightgbm"]["importance"]), index=feature_names)
    return list(importance.sort_values(ascending=False).index[: settings["lightgbm"]["top_n_features"]])


def select_features(X_train, y_train, X_val, y_val, feature_names: list, settings: dict) -> dict:
    """Features kept by the LASSO and LightGBM selection fits, their union (in feature order) and the LASSO C."""
    lasso = fit_lasso_selection(X_train, y_train, X_val, y_val, feature_names, settings)
    top = fit_lightgbm_selection(X_train, y_train, X_val, y_val, feature_names, settings)
    chosen = set(lasso["selected"]) | set(top)
    return {
        "lasso": lasso["selected"],
        "lightgbm": top,
        "union": [f for f in feature_names if f in chosen],
        "lasso_C": lasso["C"],
    }


def fit_ridge(X_train, y_train, X_val, y_val, settings: dict) -> dict:
    """Ridge (L2) logistic regression with C chosen by validation log loss; keeps every input feature."""
    best = None
    for C in settings["lasso"]["ridge_C_grid"]:
        model = LogisticRegression(penalty="l2", solver="lbfgs", C=C, max_iter=5000).fit(X_train, y_train)
        loss = log_loss(y_val, model.predict_proba(X_val)[:, 1])
        if best is None or loss < best["log_loss"]:
            best = {"log_loss": loss, "C": C, "model": model}
    return best


def make_sequences(scaled: pd.DataFrame, keep_dates, length: int):
    """Windows of `length` rows per ticker ending on each date in keep_dates, and their (date, ticker) index.

    Each window ends on its own date, so a sequence never uses a later row.
    """
    keep_dates = pd.DatetimeIndex(keep_dates)
    X_parts, index_parts = [], []
    for _, group in scaled.groupby(level="ticker", sort=False):
        values = group.to_numpy(dtype="float32")
        positions = np.where(group.index.get_level_values("date").isin(keep_dates))[0]
        positions = positions[positions >= length - 1]
        if len(positions) == 0:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(values, (length, values.shape[1]))[:, 0]
        X_parts.append(windows[positions - length + 1])
        index_parts.append(group.index[positions])
    return np.concatenate(X_parts), index_parts[0].append(index_parts[1:])


def _keras():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow import keras

    tf.get_logger().setLevel("ERROR")
    return keras


def build_lstm(n_features: int, settings: dict):
    keras = _keras()
    lstm = settings["lstm"]
    layers = [keras.layers.Input(shape=(lstm["sequence_length"], n_features))]
    for i in range(lstm["layers"]):
        layers.append(
            keras.layers.LSTM(
                lstm["units"], return_sequences=i < lstm["layers"] - 1, recurrent_dropout=lstm["recurrent_dropout"]
            )
        )
    layers += [keras.layers.Dropout(lstm["dropout"]), keras.layers.Dense(1, activation="sigmoid")]
    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(lstm["learning_rate"]), loss="binary_crossentropy")
    return model


def fit_predict_lstm(scaled: pd.DataFrame, labels: pd.Series, split: splits.Split, feature_names: list,
                     seed: int, settings: dict):
    """Fit one LSTM seed on the window's sequences; return validation probabilities and test probabilities."""
    keras = _keras()
    lstm = settings["lstm"]
    subset = scaled[feature_names]
    X_train, i_train = make_sequences(subset, split.train, lstm["sequence_length"])
    X_val, i_val = make_sequences(subset, split.validation, lstm["sequence_length"])
    X_test, i_test = make_sequences(subset, split.test, lstm["sequence_length"])
    y_train, y_val = labels.reindex(i_train).to_numpy(), labels.reindex(i_val).to_numpy()
    known_train, known_val = ~np.isnan(y_train), ~np.isnan(y_val)

    # tf.random.set_seed alone does not fix Keras 3's starting weights, which come from Python's
    # global random state. Seed everything Keras uses, then restore the global state so the
    # liblinear fits of later windows draw the same seeds whatever the number of LSTM seeds.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        model = build_lstm(len(feature_names), settings)
        model.fit(
            X_train[known_train], y_train[known_train],
            validation_data=(X_val[known_val], y_val[known_val]),
            epochs=lstm["max_epochs"], batch_size=lstm["batch_size"], verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=lstm["early_stopping_patience"],
                                                     restore_best_weights=True)],
        )
        p_val = model.predict(X_val, batch_size=PREDICT_BATCH_SIZE, verbose=0).ravel()
        p_test = pd.Series(model.predict(X_test, batch_size=PREDICT_BATCH_SIZE, verbose=0).ravel(), index=i_test)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return p_val, p_test


def _fit_window(data: pd.DataFrame, feature_names: list, split: splits.Split, settings: dict, p90_quantile: float):
    dates = data.index.get_level_values("date")
    labelled = data["label"].notna()
    train = data[dates.isin(split.train) & labelled]
    val = data[dates.isin(split.validation) & labelled]
    test = data[dates.isin(split.test)]

    scaler = StandardScaler().fit(train[feature_names])
    X_train, X_val, X_test = (scaler.transform(block[feature_names]) for block in (train, val, test))
    y_train, y_val = train["label"].to_numpy(), val["label"].to_numpy()

    selected = select_features(X_train, y_train, X_val, y_val, feature_names, settings)
    union = selected["union"]
    cols = [feature_names.index(f) for f in union]

    ridge = fit_ridge(X_train[:, cols], y_train, X_val[:, cols], y_val, settings)
    gbm = fit_lightgbm(X_train[:, cols], y_train, X_val[:, cols], y_val, settings)

    probabilities = pd.DataFrame(index=test.index)
    probabilities["lasso"] = ridge["model"].predict_proba(X_test[:, cols])[:, 1]
    probabilities["lightgbm"] = gbm.predict_proba(X_test[:, cols])[:, 1]
    p_val = {
        "lasso": ridge["model"].predict_proba(X_val[:, cols])[:, 1],
        "lightgbm": gbm.predict_proba(X_val[:, cols])[:, 1],
    }
    inputs = {"lasso": union, "lightgbm": union}

    seeds = settings["lstm"]["seeds"]
    if seeds:
        # Every row is scaled with this window's scaler so look-back windows are on one scale.
        scaled = pd.DataFrame(scaler.transform(data[feature_names]), index=data.index, columns=feature_names)
        for seed in seeds:
            key = f"lstm_{seed}"
            p_val[key], p_test = fit_predict_lstm(scaled, data["label"], split, union, seed, settings)
            probabilities[key] = p_test.reindex(probabilities.index)
            inputs[key] = union
        if "lstm_avg" in lstm_columns(settings):
            seed_keys = [f"lstm_{s}" for s in seeds]
            # Every seed uses the same sequences, so validation forecasts line up element by element.
            p_val["lstm_avg"] = np.mean([p_val[k] for k in seed_keys], axis=0)
            probabilities["lstm_avg"] = probabilities[seed_keys].mean(axis=1, skipna=False)

    b = float(y_train.mean())
    stats = [{"test_year": split.test_year, "model": m, "b": b, "p90": float(np.quantile(p, p90_quantile))}
             for m, p in p_val.items()]
    tuning = {"test_year": split.test_year, "lasso_C": selected["lasso_C"], "ridge_C": ridge["C"],
              "n_lasso": len(selected["lasso"]), "n_union": len(union)}
    return probabilities, p_val, selected, stats, tuning, inputs, scaler


def run_walk_forward(data: pd.DataFrame, walk_forward: list, cfg: dict) -> WalkForwardResult:
    """Fit every model in every window and collect daily test probabilities.

    data is features.build_dataset output: every daily row with a known label is used for
    training and validation; decisions are taken weekly from the daily test probabilities.
    Probability columns: lasso (ridge on union), lightgbm, lstm_<seed> per seed and lstm_avg.
    """
    settings = model_settings(cfg)
    feature_names = features.feature_columns(data)
    p90_quantile = cfg["signals"]["p90_quantile"]

    python_state, numpy_state = random.getstate(), np.random.get_state()
    # liblinear draws its shuffling seed from NumPy's global random state; seeding it once here, as
    # the prototype did, is for parity with the reference run (results/2026-10-30_363ec0b_lstm-10seeds-dsr).
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    probabilities, validation, selection_rows, stats_rows, tuning_rows, inputs, scalers = [], {}, [], [], [], {}, {}
    try:
        for split in walk_forward:
            p_test, p_val, selected, stats, tuning, window_inputs, scaler = _fit_window(
                data, feature_names, split, settings, p90_quantile
            )
            probabilities.append(p_test)
            validation[split.test_year] = p_val
            selection_rows += [{"test_year": split.test_year, "model": name, "feature": f}
                               for name in SELECTORS for f in selected[name]]
            stats_rows += stats
            tuning_rows.append(tuning)
            inputs[split.test_year] = window_inputs
            scalers[split.test_year] = scaler
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    return WalkForwardResult(
        probabilities=pd.concat(probabilities).sort_index(),
        validation_probabilities=validation,
        selections=pd.DataFrame(selection_rows),
        window_stats=pd.DataFrame(stats_rows).set_index(["test_year", "model"]),
        tuning=pd.DataFrame(tuning_rows).set_index("test_year"),
        inputs=inputs,
        scalers=scalers,
    )


def grid_edges(tuning: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """How often each tuned C lands on an edge of its grid (the x10 LASSO fallback counts as the upper edge)."""
    grids = {"lasso_C": cfg["models"]["lasso"]["C_grid"], "ridge_C": cfg["models"]["lasso"]["ridge_C_grid"]}
    rows = []
    for column, grid in grids.items():
        chosen = tuning[column].dropna()
        rows.append({
            "parameter": column,
            "grid": str(grid),
            "windows": len(chosen),
            "at_lower_edge": int((chosen <= min(grid)).sum()),
            "at_upper_edge": int((chosen >= max(grid)).sum()),
            "chosen": "; ".join(f"{c:g}: {n}" for c, n in chosen.value_counts().sort_index().items()),
        })
    return pd.DataFrame(rows).set_index("parameter")
