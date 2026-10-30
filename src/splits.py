"""Expanding walk-forward splits with an 11-day gap between blocks.

For each test year the validation block is the validation_months ending one
gap before the test year, and the training block is everything up to one gap
before validation. The model is refitted once a year.

The gap between train, validation and test is
    execution_lag_days + horizon_days + embargo_days  (1 + 5 + 5 = 11)
trading days. The first two terms remove every sample whose label uses a price
from the next block; the embargo adds a further margin. The gap is cut from
the end of the earlier block, so each test year is complete and the
out-of-sample record has no holes.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    test_year: int
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex


def gap_days(cfg: dict) -> int:
    return cfg["label"]["execution_lag_days"] + cfg["label"]["horizon_days"] + cfg["splits"]["embargo_days"]


def _cut_before(dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex, boundary: pd.Timestamp, n_days: int):
    """Dates that leave n_days trading days unused before the boundary date."""
    pos = calendar.searchsorted(boundary)
    if pos - n_days <= 0:
        return dates[:0]
    return dates[dates <= calendar[pos - n_days - 1]]


def walk_forward_splits(calendar: pd.DatetimeIndex, cfg: dict) -> list:
    """One Split per test year. calendar is the full trading calendar."""
    sc = cfg["splits"]
    gap = gap_days(cfg)
    calendar = calendar[calendar >= pd.Timestamp(cfg["features"]["sample_start"])]

    splits = []
    for year in range(sc["first_test_year"], sc["last_test_year"] + 1):
        test = calendar[calendar.year == year]
        if len(test) == 0:
            continue
        before_test = _cut_before(calendar, calendar, test[0], gap)
        if len(before_test) == 0:
            continue
        val_start = before_test[-1] - pd.DateOffset(months=sc["validation_months"])
        validation = before_test[before_test > val_start]
        train = _cut_before(calendar, calendar, validation[0], gap)
        if len(train) == 0:
            continue
        splits.append(Split(year, train, validation, test))
    return splits


def select(data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Rows of a (date, ticker) panel whose date is in dates."""
    return data[data.index.get_level_values("date").isin(dates)]
