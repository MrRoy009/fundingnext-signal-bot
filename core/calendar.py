"""Exchange calendars and local-time helpers per instrument (COMMON 1.6, 2).

Each instrument has an exchange (NYSE for US30/NAS100, XETRA for GER40), a local time zone and
a cash session, all from config.yaml `exchanges`. A valid trading date also needs the cash-open
M15 bar in the data; that check lives with the features, which see the bars.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time

import pandas as pd

from core.calendar_data import (NYSE_EARLY_CLOSES, NYSE_FULL_HOLIDAYS, XETRA_EARLY_CLOSES,
                                XETRA_FULL_HOLIDAYS)

FIRST_YEAR, LAST_YEAR = 2021, 2027
_TABLES = {
    "NYSE": (NYSE_FULL_HOLIDAYS, NYSE_EARLY_CLOSES),
    "XETRA": (XETRA_FULL_HOLIDAYS, XETRA_EARLY_CLOSES),
}


@dataclass(frozen=True)
class Exchange:
    calendar: str          # NYSE | XETRA
    tz: str                # e.g. America/New_York
    cash_open: time
    cash_close: time


def exchange_for(cfg: dict, key: str) -> Exchange:
    e = cfg["exchanges"][key]
    return Exchange(e["calendar"], e["tz"], time.fromisoformat(e["cash_open"]),
                    time.fromisoformat(e["cash_close"]))


def _check_year(d: date) -> None:
    if not FIRST_YEAR <= d.year <= LAST_YEAR:
        raise ValueError(f"No frozen exchange calendar for {d}; extend core/calendar_data.py")


def is_full_holiday(calendar: str, d: date) -> bool:
    _check_year(d)
    return d in _TABLES[calendar][0]


def early_close(calendar: str, d: date) -> time | None:
    """Local early-close time, or None on a normal day."""
    _check_year(d)
    return _TABLES[calendar][1].get(d)


def is_calendar_trading_date(calendar: str, d: date) -> bool:
    """Weekday, not a full holiday, not an early-close date (COMMON 1.6). The data check
    (cash-open bar present) is applied separately."""
    return d.weekday() < 5 and not is_full_holiday(calendar, d) and early_close(calendar, d) is None


def to_local(time_utc: pd.Series | pd.DatetimeIndex, tz: str):
    """Exchange-local wall-clock times (tz-aware) for UTC timestamps."""
    if isinstance(time_utc, pd.Series):
        return time_utc.dt.tz_convert(tz)
    return time_utc.tz_convert(tz)


def local_to_utc(d: date, t: time, tz: str) -> pd.Timestamp:
    """A local wall-clock time on a local date, as UTC."""
    return pd.Timestamp(datetime.combine(d, t)).tz_localize(tz).tz_convert("UTC")
