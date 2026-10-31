"""Significance tests: block bootstrap with Holm, exposure-matched placebo, deflated Sharpe ratio,
Ledoit-Wolf (2008) and Diebold-Mariano.

Inputs are daily excess returns over the T-bill rate unless stated otherwise.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew
from scipy.stats import t as student_t
from tqdm.auto import tqdm

from src import data, evaluate, portfolio

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649
P_COLUMN = "p (A > B, one-sided)"


def excess_returns(result: pd.DataFrame, rf: pd.Series) -> pd.Series:
    return result["ret"] - rf.loc[result.index]


def _sharpe_rows(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) / x.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)


def stationary_bootstrap_indices(n: int, n_resamples: int, mean_block: float, rng, chunk: int = 500):
    """Index arrays of the stationary bootstrap (Politis and Romano, 1994), in chunks of resamples."""
    p = 1 / mean_block
    for start in range(0, n_resamples, chunk):
        b = min(chunk, n_resamples - start)
        restart = rng.random((b, n)) < p
        restart[:, 0] = True
        starts = rng.integers(0, n, (b, n))
        pos = np.arange(n)
        last = np.maximum.accumulate(np.where(restart, pos, 0), axis=1)
        yield (np.take_along_axis(starts, last, axis=1) + (pos - last)) % n


def bootstrap_sharpe_difference(excess_a: pd.Series, excess_b: pd.Series, cfg: dict) -> dict:
    """Sharpe(a) - Sharpe(b) with a 95% interval and the one-sided p-value of a > b.

    Both series are resampled with the same indices, so their correlation is kept.
    """
    keys = ["Sharpe A", "Sharpe B", "Difference", "95% CI low", "95% CI high", P_COLUMN]
    a, b = np.asarray(excess_a, dtype=float), np.asarray(excess_b, dtype=float)
    if min(a.std(), b.std()) < 1e-10:
        return dict.fromkeys(keys, np.nan)
    boot = cfg["stats"]["bootstrap"]
    rng = np.random.default_rng(cfg["seed"])
    diffs = np.concatenate([_sharpe_rows(a[i]) - _sharpe_rows(b[i]) for i in
                            stationary_bootstrap_indices(len(a), boot["n_resamples"], boot["block_days"], rng)])
    sharpe_a, sharpe_b = _sharpe_rows(a[None])[0], _sharpe_rows(b[None])[0]
    return dict(zip(keys, [sharpe_a, sharpe_b, sharpe_a - sharpe_b, np.percentile(diffs, 2.5),
                           np.percentile(diffs, 97.5), (diffs <= 0).mean()]))


def holm(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted, running, m = np.empty(len(p_values)), 0.0, len(p_values)
    for rank, i in enumerate(np.argsort(p_values)):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def _quadratic_spectral(x: np.ndarray) -> np.ndarray:
    z = 6 * np.pi * x / 5
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 25 / (12 * np.pi ** 2 * x ** 2) * (np.sin(z) / z - np.cos(z))
    return np.where(x == 0, 1.0, k)


def _andrews_bandwidth(e: np.ndarray) -> float:
    """Automatic QS bandwidth of Andrews (1991) from AR(1) fits to each column."""
    num = den = 0.0
    for col in e.T:
        rho = np.dot(col[1:], col[:-1]) / np.dot(col[:-1], col[:-1])
        sigma2 = np.mean((col[1:] - rho * col[:-1]) ** 2)
        num += 4 * rho ** 2 * sigma2 ** 2 / (1 - rho) ** 8
        den += sigma2 ** 2 / (1 - rho) ** 4
    return 1.3221 * (num / den * len(e)) ** 0.2


def hac_covariance(v: np.ndarray, prewhiten: bool = True) -> np.ndarray:
    """Long-run covariance of the demeaned rows of v: QS kernel, automatic bandwidth, optional VAR(1) prewhitening
    (Andrews and Monahan, 1992), with the small-sample factor T / (T - k)."""
    T, k = v.shape
    A = np.zeros((k, k))
    e = v
    if prewhiten:
        A = np.linalg.lstsq(v[:-1], v[1:], rcond=None)[0].T
        e = v[1:] - v[:-1] @ A.T
    n = len(e)
    bandwidth = _andrews_bandwidth(e)
    weights = _quadratic_spectral(np.arange(1, n) / bandwidth)
    psi = e.T @ e / n
    for j, w in enumerate(weights, start=1):
        gamma = e[j:].T @ e[:-j] / n
        psi += w * (gamma + gamma.T)
    psi *= n / (n - k)
    if prewhiten:
        recolour = np.linalg.inv(np.eye(k) - A)
        psi = recolour @ psi @ recolour.T
    return psi


def ledoit_wolf_test(excess_a: pd.Series, excess_b: pd.Series, cfg: dict) -> dict:
    """HAC test of Sharpe(a) - Sharpe(b) (Ledoit and Wolf, 2008, section 3.1): delta method on the first and
    second moments, with their long-run covariance from a prewhitened quadratic-spectral kernel.

    Returns the annualised difference, its standard error, z and the one-sided p-value of a > b.
    """
    lw = cfg["stats"]["ledoit_wolf"]
    if lw["kernel"] != "quadratic_spectral":
        raise ValueError(f"unsupported kernel: {lw['kernel']}")
    a, b = np.asarray(excess_a, dtype=float), np.asarray(excess_b, dtype=float)
    keys = ["LW difference", "LW std. error", "LW z", "LW p (A > B, one-sided)"]
    if min(a.std(), b.std()) < 1e-10:
        return dict.fromkeys(keys, np.nan)
    y = np.column_stack([a, b, a ** 2, b ** 2])
    mu_a, mu_b, g_a, g_b = y.mean(axis=0)
    var_a, var_b = g_a - mu_a ** 2, g_b - mu_b ** 2
    difference = mu_a / np.sqrt(var_a) - mu_b / np.sqrt(var_b)
    gradient = np.array([g_a / var_a ** 1.5, -g_b / var_b ** 1.5,
                         -mu_a / (2 * var_a ** 1.5), mu_b / (2 * var_b ** 1.5)])
    if np.array_equal(a, b):
        se = 0.0
    else:
        psi = hac_covariance(y - y.mean(axis=0), prewhiten=lw["prewhiten"])
        se = float(np.sqrt(gradient @ psi @ gradient / len(y)))
    z = difference / se if se > 0 else 0.0
    scale = np.sqrt(TRADING_DAYS)
    return dict(zip(keys, [difference * scale, se * scale, z, norm.sf(z)]))


def sharpe_tests(results: dict, rf: pd.Series, names: dict, cfg: dict) -> pd.DataFrame:
    """Bootstrap (and Ledoit-Wolf next to it) for H4 and for each model against always-on.

    H4: every LSTM seed and the seed average against the linear model and LightGBM, Holm-corrected
    across the two. LSTM vs LightGBM is the substantive test; LSTM vs linear is reported but not
    interpreted, because the linear model finds no usable signal.
    """
    always_on = evaluate.BENCHMARK_NAMES["always_on"]
    linear, gbm = names["lasso"], names["lightgbm"]
    lstm = [n for c, n in names.items() if c.startswith("lstm_")]
    pairs = [(n, other) for n in lstm for other in (linear, gbm)] + [(n, always_on) for n in names.values()]

    rows = {}
    for a, b in tqdm(pairs, desc="Sharpe tests", unit="pair", disable=not cfg["run"]["progress"]):
        ex_a, ex_b = excess_returns(results[a], rf), excess_returns(results[b], rf)
        rows[f"{a} vs {b}"] = {**bootstrap_sharpe_difference(ex_a, ex_b, cfg), **ledoit_wolf_test(ex_a, ex_b, cfg)}
    table = pd.DataFrame(rows).T
    table["p Holm (H4)"] = np.nan
    table["Use"] = "always-on comparison"
    for n in lstm:
        h4 = [f"{n} vs {linear}", f"{n} vs {gbm}"]
        table.loc[h4, "p Holm (H4)"] = holm(table.loc[h4, P_COLUMN].to_numpy())
        table.loc[h4[0], "Use"] = "H4, reported but not interpreted (linear finds no usable signal)"
        table.loc[h4[1], "Use"] = "H4, substantive test"
    return table


def weekly_signal_matrix(signal: pd.Series, market: portfolio.Market) -> pd.DataFrame:
    """One model's signal on the decision dates (weeks x tradable ETFs); a missing signal is 0."""
    return signal.unstack("ticker").reindex(index=market.decision_dates, columns=market.tradable).fillna(0.0)


def shift_signals(signals: pd.DataFrame, variant: str, rng, min_shift: int) -> pd.DataFrame:
    """Circularly shift the weekly signals by a whole number of weeks uniform in [min_shift, W - min_shift].

    independent: one shift per ETF; common: one shift for all ETFs. Each ETF keeps its signal values,
    so its exposure, position sizes and holding lengths are unchanged; only the timing is broken.
    """
    W = len(signals)
    if W - min_shift < min_shift:
        raise ValueError(f"{W} decision weeks is too short for shifts in [{min_shift}, W - {min_shift}]")
    if variant == "common":
        k = rng.integers(min_shift, W - min_shift + 1)
        return pd.DataFrame(np.roll(signals.to_numpy(), k, axis=0), index=signals.index, columns=signals.columns)
    if variant == "independent":
        return pd.DataFrame({tk: np.roll(signals[tk].to_numpy(), rng.integers(min_shift, W - min_shift + 1))
                             for tk in signals.columns}, index=signals.index)
    raise ValueError(f"unknown placebo variant: {variant}")


def placebo_test(signals: pd.DataFrame, base: pd.DataFrame, market: portfolio.Market, cfg: dict,
                 common: pd.DatetimeIndex, actual: pd.Series, variant: str, n_draws: int, rng, bar=None):
    """Placebo distribution for one model and variant; every draw goes through portfolio.backtest.

    Returns the summary row and the Sharpe ratio of every draw.
    """
    min_shift = cfg["stats"]["placebo"]["min_shift_weeks"]
    sharpes, drawdowns, exposures = [], [], []
    for _ in range(n_draws):
        shifted = shift_signals(signals, variant, rng, min_shift)
        targets = (shifted.reindex(columns=base.columns).fillna(0.0) * base).reindex(columns=base.columns).fillna(0.0)
        stats = evaluate.portfolio_metrics(portfolio.backtest(targets, market, cfg).loc[common], market.rf)
        sharpes.append(stats["Sharpe"])
        drawdowns.append(stats["Max drawdown"])
        exposures.append(stats["Avg exposure"])
        if bar is not None:
            bar.update()
    sharpes = np.array(sharpes)
    valid = sharpes[~np.isnan(sharpes)]
    have = len(valid) > 0
    row = {
        "Actual Sharpe": actual["Sharpe"],
        "Placebo median": np.median(valid) if have else np.nan,
        "Placebo 5th pct": np.percentile(valid, 5) if have else np.nan,
        "Placebo 95th pct": np.percentile(valid, 95) if have else np.nan,
        "p (placebo >= actual)": (1 + (valid >= actual["Sharpe"]).sum()) / (len(valid) + 1) if have else np.nan,
        "Actual max drawdown": actual["Max drawdown"],
        "Placebo median max drawdown": np.median(drawdowns),
        "Actual exposure": actual["Avg exposure"],
        "Placebo mean exposure": np.mean(exposures),
        "Draws": len(valid),
    }
    return row, sharpes


def run_placebo(main_signals: pd.DataFrame, base: pd.DataFrame, market: portfolio.Market, cfg: dict,
                common: pd.DatetimeIndex, performance: pd.DataFrame, names: dict):
    """Exposure-matched placebo for every model and variant, from one generator seeded from config.

    The independent variant is drawn first for every model, so its draws do not depend on the common variant.
    Returns the summary table indexed by (model, variant) and the draws.
    """
    placebo = cfg["stats"]["placebo"]
    n_draws = placebo["quick_draws"] if cfg["run"]["quick_mode"] else placebo["n_draws"]
    rng = np.random.default_rng(cfg["seed"])
    rows, draws = {}, {}
    bar = tqdm(total=len(placebo["variants"]) * len(names) * n_draws, desc="Placebo draws", unit="draw",
               disable=not cfg["run"]["progress"])
    for variant in placebo["variants"]:
        for column, name in names.items():
            bar.set_postfix_str(f"{name} | {variant}")
            signals = weekly_signal_matrix(main_signals[column], market)
            rows[(name, variant)], draws[f"{name} | {variant}"] = placebo_test(
                signals, base, market, cfg, common, performance.loc[name], variant, n_draws, rng, bar)
    bar.close()
    return pd.DataFrame(rows).T.rename_axis(["model", "variant"]), pd.DataFrame(draws)


def expected_max_sharpe(trial_sharpes) -> float:
    """SR0: expected maximum daily Sharpe ratio of N skill-less trials with the trials' variance
    (Bailey and Lopez de Prado, 2014). trial_sharpes are annualised."""
    daily = np.array([s for s in trial_sharpes if pd.notna(s)]) / np.sqrt(TRADING_DAYS)
    n = len(daily)
    return float(np.sqrt(daily.var(ddof=1)) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n)
                                               + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e))))


def deflated_sharpe(excess: pd.Series, trial_sharpes) -> dict:
    """Probability that the Sharpe ratio exceeds SR0, the best expected by luck across the trials.

    trial_sharpes come from dsr_trials: the 10 current trials (9 main specifications and the seed
    average) are recomputed in every run; the 15 historical trials (L1 refit, the two earlier ridge
    grids, the original input design) are read from the frozen dsr_trials.csv of the reference run
    (stats.dsr_reference_trials).
    """
    x = np.asarray(excess, dtype=float)
    if x.std() < 1e-12:
        return {"Sharpe": np.nan, "Skew": np.nan, "Kurtosis": np.nan, "DSR": np.nan}
    sr0 = expected_max_sharpe(trial_sharpes)
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = skew(x), kurtosis(x, fisher=False)
    z = (sr - sr0) * np.sqrt(len(x) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return {"Sharpe": sr * np.sqrt(TRADING_DAYS), "Skew": g3, "Kurtosis": g4, "DSR": norm.cdf(z)}


def _is_current_trial(name: str) -> bool:
    current = ("Linear (L1 selection, ridge on union)", "LightGBM", "LSTM")
    return name.startswith("LSTM (average of") or any(name == c or name.startswith(c + ", ") for c in current)


def dsr_trials(performance: pd.DataFrame, names: dict, main_models: list, cfg: dict) -> pd.Series:
    """Sharpe ratios of the counted trials: the current ones from this run, the historical ones from the
    frozen reference file. Raises if the count differs from stats.deflated_sharpe_trials."""
    current = {}
    for column in main_models:
        for suffix in ["", *evaluate.SENSITIVITY_SUFFIX.values()]:
            current[f"{names[column]}{suffix}"] = performance.loc[f"{names[column]}{suffix}", "Sharpe"]
    if "lstm_avg" in names:
        current[names["lstm_avg"]] = performance.loc[names["lstm_avg"], "Sharpe"]
    frozen = pd.read_csv(data.ROOT / cfg["stats"]["dsr_reference_trials"])
    historical = frozen[~frozen["trial"].map(_is_current_trial)].set_index("trial")["Sharpe"]
    trials = pd.concat([pd.Series(current, dtype=float), historical]).rename("Sharpe").rename_axis("trial")
    if len(trials) != cfg["stats"]["deflated_sharpe_trials"]:
        raise ValueError(f"{len(trials)} trials counted, config says {cfg['stats']['deflated_sharpe_trials']}")
    return trials


def dsr_table(results: dict, rf: pd.Series, strategy_names: list, trials: pd.Series) -> pd.DataFrame:
    table = pd.DataFrame({n: deflated_sharpe(excess_returns(results[n], rf), trials) for n in strategy_names}).T
    table["SR0 (annualised)"] = expected_max_sharpe(trials) * np.sqrt(TRADING_DAYS)
    table["Trials"] = trials.notna().sum()
    return table


def forecast_losses(probabilities: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Brier loss per labelled (date, ETF) row, averaged across ETFs for each date."""
    y = labels.reindex(probabilities.index)
    known = y.notna()
    squared = probabilities[known].sub(y[known], axis=0) ** 2
    return squared.groupby(level="date").mean()


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, horizon: int, small_sample_correction: bool = True) -> dict:
    """Diebold-Mariano test of equal forecast loss; one-sided p-value that a has the lower loss.

    Newey-West variance with horizon - 1 lags, because labels of overlapping horizons are
    autocorrelated, and the Harvey, Leybourne and Newbold (1997) correction with t(T - 1).
    """
    d = (np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float))
    T, mean = len(d), d.mean()
    centred = d - mean
    variance = centred @ centred / T
    for lag in range(1, horizon):
        variance += 2 * (1 - lag / horizon) * (centred[lag:] @ centred[:-lag]) / T
    keys = ["DM mean loss difference", "DM statistic", "DM p (A lower loss, one-sided)"]
    if variance <= 0:
        return dict(zip(keys, [mean, 0.0, 0.5]))
    stat = mean / np.sqrt(variance / T)
    if small_sample_correction:
        stat *= np.sqrt((T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T)
    return dict(zip(keys, [mean, stat, student_t.cdf(stat, T - 1)]))


def diebold_mariano_tests(losses: pd.DataFrame, names: dict, cfg: dict) -> pd.DataFrame:
    """Every LSTM seed and the seed average against the linear model and LightGBM."""
    dm = cfg["stats"]["diebold_mariano"]
    if dm["loss"] != "brier":
        raise ValueError(f"unsupported loss: {dm['loss']}")
    horizon = cfg["label"]["horizon_days"]
    rows = {}
    for column, name in names.items():
        if not column.startswith("lstm_"):
            continue
        for other in ("lasso", "lightgbm"):
            rows[f"{name} vs {names[other]}"] = diebold_mariano(
                losses[column], losses[other], horizon, dm["small_sample_correction"])
    return pd.DataFrame(rows).T
