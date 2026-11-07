from datetime import date, time

import pandas as pd
import pytest

from core.calendar import (early_close, exchange_for, is_calendar_trading_date, is_full_holiday,
                           local_to_utc)
from data.segments import segment_ids, tv_bad


@pytest.mark.parametrize("cal, d, full, early", [
    ("NYSE", date(2024, 7, 4), True, None),
    ("NYSE", date(2025, 1, 9), True, None),            # national day of mourning
    ("NYSE", date(2024, 11, 29), False, time(13, 0)),
    ("NYSE", date(2024, 7, 5), False, None),
    ("XETRA", date(2024, 12, 24), True, None),
    ("XETRA", date(2025, 12, 30), False, time(14, 0)),
    ("XETRA", date(2025, 5, 1), True, None),
])
def test_holidays(cal, d, full, early):
    assert is_full_holiday(cal, d) is full
    assert early_close(cal, d) == early


def test_trading_date_rules():
    assert is_calendar_trading_date("NYSE", date(2024, 7, 5))
    assert not is_calendar_trading_date("NYSE", date(2024, 7, 6))      # Saturday
    assert not is_calendar_trading_date("NYSE", date(2024, 11, 29))    # early close
    with pytest.raises(ValueError):
        is_calendar_trading_date("NYSE", date(2028, 1, 3))


def test_exchange_config_and_local_time(cfg):
    ex = exchange_for(cfg, "NAS100")
    assert ex.calendar == "NYSE" and ex.cash_open == time(9, 30)
    # US summer time, EU winter: 09:30 New York = 13:30 UTC on 16 Mar 2026
    assert local_to_utc(date(2026, 3, 16), ex.cash_open, ex.tz) == pd.Timestamp("2026-03-16 13:30", tz="UTC")


def test_excluded_windows_dropped_on_load(cfg, tmp_path):
    from data.store import BarStore
    cfg["data_dir"] = "store"
    cfg["holdout"] = {"start": None}
    store = BarStore(cfg, root=tmp_path)
    t = pd.Series(pd.to_datetime(["2024-10-03 12:00", "2024-10-04 12:00", "2025-01-07 12:00",
                                  "2025-01-08 00:00", "2025-07-18 21:00"], utc=True))
    for key in ("US30", "GER40"):
        store.merge(key, "M15", pd.DataFrame({"time_utc": t, "close": 1.0}))
    assert store.load("US30", "M15")["time_utc"].dt.strftime("%Y-%m-%d").tolist() == \
        ["2024-10-03", "2025-01-08", "2025-07-18"]
    assert store.load("GER40", "M15")["time_utc"].dt.strftime("%Y-%m-%d").tolist() == \
        ["2024-10-03", "2025-01-08"]
    assert len(store.load("US30", "M15", raw=True)) == 5
    assert segment_ids(store.load("US30", "M15")["time_utc"]).tolist()[:2] == [0, 1]


def test_segments_and_tv_flag(cfg):
    t = pd.Series(pd.to_datetime(["2025-07-15 10:00", "2025-07-16 07:15", "2025-09-09 11:30",
                                  "2025-09-09 11:45"], utc=True))
    assert segment_ids(t).tolist() == [0, 0, 1, 1]
    flags = tv_bad(pd.Series(pd.to_datetime(["2024-10-03 23:45", "2024-10-04 00:00",
                                             "2025-01-07 23:45", "2025-01-08 00:00"], utc=True)), cfg)
    assert flags.tolist() == [False, True, True, False]
