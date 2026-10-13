"""Expanding walk-forward splits with an 11-day gap between blocks.

For each test year the validation block is the last validation_months of the
training window and the training block is everything earlier. The model is
refitted once a year.

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
        test_start = pd.Timestamp(year=year, month=1, day=1)
        val_start = test_start - pd.DateOffset(months=sc["validation_months"])
        test = calendar[calendar.year == year]
        validation = calendar[(calendar >= val_start) & (calendar < test_start)]
        train = calendar[calendar < val_start]
        if len(test) == 0 or len(validation) == 0 or len(train) == 0:
            continue
        train = _cut_before(train, calendar, validation[0], gap)
        validation = _cut_before(validation, calendar, test[0], gap)
        splits.append(Split(year, train, validation, test))
    return splits


def select(data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Rows of a (date, ticker) panel whose date is in dates."""
    return data[data.index.get_level_values("date").isin(dates)]
