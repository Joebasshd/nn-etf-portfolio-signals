"""Prediction metrics, portfolio metrics, indicator stability, sub-periods and run folders.

Tables use the display names of the reference run (display_names), so they
can be compared with it row by row.
"""

import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from tqdm.auto import tqdm

from src import data, features, portfolio

TRADING_DAYS = 252
BENCHMARK_NAMES = {
    "spy_buy_hold": "SPY buy-and-hold",
    "equal_weight_monthly": "Equal-weight (monthly)",
    "vol_target_spy": "Vol-targeted SPY",
    "sma_50_200": "SMA 50/200 rule",
    "always_on": "Always-on (vol-scaled)",
}
SENSITIVITY_SUFFIX = {"base_rate": ", base-rate", "linear": ", 2p-1"}
SELECTOR_NAMES = {"lasso": "LASSO", "lightgbm": "LightGBM", "union": "Union"}
FAMILY_ORDER = ["trend", "momentum", "volatility", "volume", "market context"]


def display_names(model_columns) -> dict:
    """Table labels per probability column. The first LSTM seed is the main LSTM.

    The "lasso" key is the ridge forecasting fit on the union, so it is labelled as such.
    """
    names = {"lasso": "Linear (L1 selection, ridge on union)", "lightgbm": "LightGBM"}
    seeds = [c for c in model_columns if c.startswith("lstm_") and c != "lstm_avg"]
    for i, column in enumerate(seeds):
        names[column] = "LSTM" if i == 0 else f"LSTM (seed {column.split('_')[1]})"
    if "lstm_avg" in model_columns:
        names["lstm_avg"] = f"LSTM (average of {len(seeds)} seeds)"
    return {c: names[c] for c in model_columns}


def strategy_targets(main_signals: pd.DataFrame, sensitivity_signals: dict, base: pd.DataFrame,
                     market: portfolio.Market, cfg: dict, names: dict) -> dict:
    """(targets, band) per strategy, keyed by display name: the benchmarks, every model on the main
    signal and the main models on each sensitivity signal (keys of SENSITIVITY_SUFFIX)."""
    band = cfg["portfolio"]["no_trade_band"]
    strategies = {BENCHMARK_NAMES[k]: v for k, v in portfolio.benchmark_targets(market, base, cfg).items()}
    for column in main_signals:
        strategies[names[column]] = (portfolio.target_weights(main_signals[column], base), band)
        for version, signal in sensitivity_signals.items():
            if column in signal:
                strategies[names[column] + SENSITIVITY_SUFFIX[version]] = (
                    portfolio.target_weights(signal[column], base), band)
    return strategies


def _labelled(probabilities: pd.DataFrame, labels: pd.Series):
    y = labels.reindex(probabilities.index)
    known = y.notna()
    return probabilities[known], y[known]


def prediction_metrics(probabilities: pd.DataFrame, labels: pd.Series, names: dict) -> pd.DataFrame:
    """Accuracy against the base rate, AUC, Brier score and log loss on every labelled test row."""
    p_all, y = _labelled(probabilities, labels)
    rows = []
    for column, name in names.items():
        p = p_all[column].clip(1e-6, 1 - 1e-6)
        rows.append({
            "model": name, "n": len(y), "base_rate": y.mean(),
            "accuracy": accuracy_score(y, p > 0.5),
            "AUC": roc_auc_score(y, p),
            "Brier": brier_score_loss(y, p),
            "Brier (base rate)": brier_score_loss(y, np.full(len(y), y.mean())),
            "log_loss": log_loss(y, p),
            "mean_p": p.mean(), "sd_p": p.std(),
        })
    return pd.DataFrame(rows).set_index("model")


def auc_by_year(probabilities: pd.DataFrame, labels: pd.Series, names: dict) -> pd.DataFrame:
    p, y = _labelled(probabilities, labels)
    years = p.index.get_level_values("date").year
    return pd.DataFrame({
        name: {year: roc_auc_score(y[years == year], p.loc[years == year, column]) for year in sorted(set(years))}
        for column, name in names.items()
    }).rename_axis("year")


def calibration_table(probabilities: pd.DataFrame, labels: pd.Series, names: dict, n_bins: int = 10) -> pd.DataFrame:
    """Mean forecast against observed frequency in quantile bins of the forecast, per model.

    Quantile bins, because the forecasts sit in a narrow range around the base rate.
    """
    p_all, y = _labelled(probabilities, labels)
    tables = []
    for column, name in names.items():
        bins = pd.qcut(p_all[column], n_bins, labels=False, duplicates="drop")
        table = pd.DataFrame({"p": p_all[column], "y": y}).groupby(bins).agg(
            mean_p=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
        tables.append(table.rename_axis("bin").reset_index().assign(model=name))
    return pd.concat(tables)[["model", "bin", "mean_p", "observed", "n"]].reset_index(drop=True)


def portfolio_metrics(result: pd.DataFrame, rf: pd.Series) -> pd.Series:
    """Annual return and volatility, Sharpe, Sortino, max drawdown, turnover, exposure and total cost of one backtest.

    Total cost is the sum of the costs paid over the period, as a share of capital.
    """
    r = result["ret"]
    excess = r - rf.loc[r.index]
    years = len(r) / TRADING_DAYS
    equity = (1 + r).cumprod()
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean()) * np.sqrt(TRADING_DAYS)
    return pd.Series({
        "Ann. return": equity.iloc[-1] ** (1 / years) - 1,
        "Ann. volatility": r.std() * np.sqrt(TRADING_DAYS),
        "Sharpe": excess.mean() / excess.std() * np.sqrt(TRADING_DAYS) if excess.std() > 1e-10 else np.nan,
        "Sortino": excess.mean() * TRADING_DAYS / downside if downside > 1e-10 else np.nan,
        "Max drawdown": (equity / equity.cummax() - 1).min(),
        "Turnover / yr": result["turnover"].sum() / years,
        "Avg exposure": result["exposure"].mean(),
        "Total cost": result["cost"].sum(),
    })


def performance_table(results: dict, rf: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({name: portfolio_metrics(r, rf) for name, r in results.items()}).T


def breakeven_cost(gross: pd.DataFrame, rf: pd.Series) -> float:
    """Cost per side, in bps, at which net excess return over T-bills falls to zero.

    gross is the backtest at 0 bps: gross annual excess return divided by annual one-way turnover.
    """
    excess = (gross["ret"] - rf.loc[gross.index]).mean() * TRADING_DAYS
    turnover = gross["turnover"].sum() / (len(gross) / TRADING_DAYS)
    return excess / turnover * 1e4 if turnover > 0.05 else np.nan


def cost_sensitivity(strategies: dict, market: portfolio.Market, cfg: dict, common: pd.DatetimeIndex) -> pd.DataFrame:
    """Sharpe at each cost in portfolio.cost_bps_sensitivity, and the break-even cost."""
    rows = {}
    for name, (targets, band) in tqdm(strategies.items(), desc="Cost sensitivity", unit="strategy",
                                      disable=not cfg["run"]["progress"]):
        runs = {c: portfolio.backtest(targets, market, cfg, c, band).loc[common]
                for c in cfg["portfolio"]["cost_bps_sensitivity"]}
        row = {f"Sharpe @ {c} bps": portfolio_metrics(r, market.rf)["Sharpe"] for c, r in runs.items()}
        gross = runs[0] if 0 in runs else portfolio.backtest(targets, market, cfg, 0, band).loc[common]
        row["Break-even cost (bps)"] = breakeven_cost(gross, market.rf)
        rows[name] = row
    return pd.DataFrame(rows).T


def _selected_sets(selections: pd.DataFrame, selector: str) -> list:
    rows = selections[selections["model"] == selector]
    return [set(group["feature"]) for _, group in rows.groupby("test_year", sort=True)]


def selection_frequency(selections: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Share of windows in which each indicator was selected, per selection fit and the union."""
    n_windows = selections["test_year"].nunique()
    table = pd.DataFrame({
        SELECTOR_NAMES[s]: selections[selections["model"] == s]["feature"].value_counts() / n_windows
        for s in SELECTOR_NAMES
    }).reindex(feature_names).fillna(0.0)
    table.insert(0, "family", [features.feature_family(f) for f in table.index])
    return table.sort_values("LightGBM", ascending=False, kind="stable")


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else np.nan


def jaccard_overlap(selections: pd.DataFrame) -> pd.Series:
    """Average Jaccard overlap of selected sets between consecutive windows, per selection fit and the union."""
    out = {}
    for s, name in SELECTOR_NAMES.items():
        sets = _selected_sets(selections, s)
        overlaps = [jaccard(a, b) for a, b in zip(sets[:-1], sets[1:])]
        out[name] = np.nanmean(overlaps) if overlaps else np.nan
    return pd.Series(out, name="avg. overlap (Jaccard)")


def family_share(selections: pd.DataFrame) -> pd.DataFrame:
    """Average share of selected indicators in each of the five families, per selection fit and the union."""
    out = {}
    for s, name in SELECTOR_NAMES.items():
        shares = [pd.Series([features.feature_family(f) for f in chosen]).value_counts(normalize=True)
                  for chosen in _selected_sets(selections, s)]
        out[name] = pd.DataFrame(shares).fillna(0.0).mean()
    return pd.DataFrame(out).reindex(FAMILY_ORDER).fillna(0.0)


def stability_summary(selections: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    frequency = selection_frequency(selections, feature_names)
    families = family_share(selections)
    overlap = jaccard_overlap(selections)
    rows = []
    for s, name in SELECTOR_NAMES.items():
        top = frequency[name].idxmax()
        rows.append({
            "model": name,
            "avg. indicators selected": np.mean([len(x) for x in _selected_sets(selections, s)]),
            "avg. overlap (Jaccard)": overlap[name],
            "most selected family": families[name].idxmax(),
            "most selected indicator": f"{top} ({frequency.loc[top, name]:.0%} of windows)",
        })
    return pd.DataFrame(rows).set_index("model")


def subperiods(cfg: dict, calendar: pd.DatetimeIndex) -> dict:
    """Crisis windows and either side of the 2018 break, as (start, end) within the backtest calendar."""
    sp = cfg["subperiods"]
    names = {"gfc": "GFC", "covid": "COVID", "rate_shock_2022": "Rate shock 2022"}
    periods = {names.get(k, k): (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in sp["crises"].items()}
    brk = pd.Timestamp(sp["break_2018"])
    periods["Before 2018 break"] = (calendar[0], brk - pd.Timedelta(days=1))
    periods["After 2018 break"] = (brk, calendar[-1])
    return periods


def subperiod_metrics(results: dict, rf: pd.Series, probabilities: pd.DataFrame, labels: pd.Series,
                      names: dict, cfg: dict) -> pd.DataFrame:
    """Return, Sharpe, drawdown, exposure and (for models) AUC of each strategy in each sub-period."""
    calendar = next(iter(results.values())).index
    p_all, y = _labelled(probabilities, labels)
    label_dates = p_all.index.get_level_values("date")
    column_of = {name: column for column, name in names.items()}
    rows = []
    for period, (a, b) in subperiods(cfg, calendar).items():
        days = calendar[(calendar >= a) & (calendar <= b)]
        if len(days) == 0:
            continue
        in_period = (label_dates >= a) & (label_dates <= b)
        for name, result in results.items():
            stats = portfolio_metrics(result.loc[days], rf)
            row = {"period": period, "start": days[0].date(), "end": days[-1].date(), "days": len(days),
                   "strategy": name, "Ann. return": stats["Ann. return"], "Sharpe": stats["Sharpe"],
                   "Max drawdown": stats["Max drawdown"], "Avg exposure": stats["Avg exposure"], "AUC": np.nan}
            if name in column_of:
                row["AUC"] = roc_auc_score(y[in_period], p_all.loc[in_period, column_of[name]])
            rows.append(row)
    return pd.DataFrame(rows)


def subperiod_seed_summary(table: pd.DataFrame, seed_names: list) -> pd.DataFrame:
    """Median and range across LSTM seeds of Sharpe, max drawdown and AUC per sub-period."""
    order = list(dict.fromkeys(table["period"]))
    seeds = table[table["strategy"].isin(seed_names)]
    return seeds.groupby("period")[["Sharpe", "Max drawdown", "AUC"]].agg(["median", "min", "max"]).reindex(order)


def lstm_seed_summary(names: dict, prediction: pd.DataFrame, performance: pd.DataFrame, edges: pd.DataFrame,
                      dsr: pd.DataFrame, placebo: pd.DataFrame, sharpe_tests: pd.DataFrame,
                      variants: list) -> pd.DataFrame:
    """One row per LSTM seed, their median, min and max, counts of significant seeds, and the seed average."""
    p_column = "p (A > B, one-sided)"
    linear, gbm = names["lasso"], names["lightgbm"]

    def row(name):
        out = {"AUC": prediction.loc[name, "AUC"], "Sharpe": performance.loc[name, "Sharpe"],
               "Avg exposure": performance.loc[name, "Avg exposure"],
               "Max drawdown": performance.loc[name, "Max drawdown"],
               "Flat windows": int(edges.loc[edges["model"] == name, "flat"].sum()),
               "DSR": dsr.loc[name, "DSR"]}
        for v in variants:
            out[f"Placebo p ({v})"] = placebo.loc[(name, v), "p (placebo >= actual)"]
            out[f"Placebo median max DD ({v})"] = placebo.loc[(name, v), "Placebo median max drawdown"]
        out["p vs always-on"] = sharpe_tests.loc[f"{name} vs {BENCHMARK_NAMES['always_on']}", p_column]
        out["H4 p vs LightGBM (Holm)"] = sharpe_tests.loc[f"{name} vs {gbm}", "p Holm (H4)"]
        out["H4 p vs linear (Holm, not interpreted)"] = sharpe_tests.loc[f"{name} vs {linear}", "p Holm (H4)"]
        return out

    seeds = {c: n for c, n in names.items() if c.startswith("lstm_") and c != "lstm_avg"}
    table = pd.DataFrame({f"seed {c.split('_')[1]}": row(n) for c, n in seeds.items()}).T
    p_columns = [c for c in table.columns if c.startswith(("Placebo p", "p vs", "H4 p"))]
    summary = table.agg(["median", "min", "max"]).rename(index=lambda r: f"{r} across seeds")
    summary.loc["seeds with p < 0.05"] = (table[p_columns] < 0.05).sum()
    summary.loc["seeds with DSR > 0.95", "DSR"] = int((table["DSR"] > 0.95).sum())
    parts = [table, summary]
    if "lstm_avg" in names:
        parts.append(pd.DataFrame({names["lstm_avg"]: row(names["lstm_avg"])}).T)
    return pd.concat(parts)


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=data.ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def results_dir(cfg: dict, label: str) -> Path:
    """Create and return results/<date>_<git commit>_<label>/ with figures/ and large/ (git-ignored)."""
    commit = _git("rev-parse", "--short", "HEAD") or "nogit"
    if _git("status", "--porcelain", "--untracked-files=no"):
        warnings.warn("uncommitted changes: the folder name does not identify the code exactly")
    out = data.ROOT / cfg["run"]["results_dir"] / f"{pd.Timestamp.now():%Y-%m-%d}_{commit}_{label}"
    for d in (out, out / "figures", out / "large"):
        d.mkdir(parents=True, exist_ok=True)
    return out


def save_tables(out: Path, tables: dict, large: dict = None) -> None:
    """Small tables as <name>.csv in the run folder; large ones (predictions, curves, draws) in large/."""
    for folder, group in ((out, tables), (out / "large", large or {})):
        for name, table in group.items():
            # Consecutive years can be stored as a RangeIndex too, so only an unnamed one is a bare row number.
            row_numbers = isinstance(table.index, pd.RangeIndex) and table.index.name is None
            table.to_csv(folder / f"{name}.csv", index=not row_numbers)


def start_clock() -> dict:
    # monotonic() stops while the machine sleeps, so run times are not inflated by sleep.
    return {"wall": time.monotonic(), "cpu": time.process_time()}


def write_run_info(out: Path, cfg: dict, clock: dict, **details) -> dict:
    """run_info.json: label, commit, settings, data source and dates, run time and CPU minutes."""
    info = {
        "commit": _git("rev-parse", "--short", "HEAD") or "nogit",
        "run_date": f"{pd.Timestamp.now():%Y-%m-%d %H:%M}",
        **details,
        "config": cfg,
        "runtime_minutes": round((time.monotonic() - clock["wall"]) / 60, 1),
        "cpu_minutes": round((time.process_time() - clock["cpu"]) / 60, 1),
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    return info
