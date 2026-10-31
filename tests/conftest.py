import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    cfg = data.load_config()
    # Short sample so the synthetic calendar covers several test years quickly.
    cfg["features"]["sample_start"] = "2001-01-01"
    cfg["splits"]["first_test_year"] = 2003
    cfg["splits"]["last_test_year"] = 2005
    cfg["run"]["progress"] = False
    return cfg


@pytest.fixture(scope="session")
def synthetic(cfg):
    return data.make_synthetic(cfg, n_days=1600)
