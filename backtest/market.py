"""Per-index M15 bid bars as numpy arrays for the backtester (COMMON 1.4, 8.1, 8.5).

Everything the fill model needs per bar is computed once, vectorised: open time in int64 ns,
OHLC (bid), the base effective spread max(spread_pts, floor_{i,h}) (c = 1; the scenario
multiplies it), the spread floor of the bar's UTC hour (slippage), the segment id (COMMON 1.4)
and the daily-break flag (COMMON 8.5, server time). Histories are ~100k bars per index, so no
per-bar Python work happens here.

Server time is the broker wall clock from config `server_timezone` (zone + shift_hours); the
daily break and the account day boundary (00:00 Europe/Athens, owner decision O8) both use it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time

import numpy as np
import pandas as pd

from core.config import tf_minutes
from data.costs import SPREAD_TO_POINTS, load_spread_floor
from data.segments import segment_ids

NS_PER_MIN = 60_000_000_000

# COMMON 8.5: "No fills during the daily break (23:45-01:00 server time)". Not in config: the
# spec states the window literally.
BREAK_START_MIN = 23 * 60 + 45
BREAK_END_MIN = 60


@dataclass(frozen=True)
class ServerClock:
    """Broker server wall clock = IANA zone + fixed shift (config server_timezone)."""
    zone: str
    shift: pd.Timedelta

    @classmethod
    def from_cfg(cls, cfg: dict) -> "ServerClock":
        tz = cfg["server_timezone"]
        return cls(tz["zone"], pd.Timedelta(hours=float(tz["shift_hours"])))

    def wall(self, utc_ns: np.ndarray) -> pd.DatetimeIndex:
        """Server wall clock (naive) for UTC int64-ns instants."""
        t = pd.DatetimeIndex(np.asarray(utc_ns, dtype="int64").view("datetime64[ns]"), tz="UTC")
        return t.tz_convert(self.zone).tz_localize(None) + self.shift

    def dates(self, utc_ns: np.ndarray) -> np.ndarray:
        """Server calendar date (datetime64[D]) of each instant."""
        return self.wall(utc_ns).to_numpy().astype("datetime64[D]")

    def date_of(self, utc_ns: int) -> date:
        return pd.Timestamp(self.wall(np.array([utc_ns]))[0]).date()

    def midnight_utc_ns(self, server_date: date) -> int:
        """UTC instant (ns) of 00:00 server time on `server_date`."""
        local = pd.Timestamp(datetime.combine(server_date, time(0))) - self.shift
        return int(local.tz_localize(self.zone).tz_convert("UTC").value)


@dataclass(frozen=True)
class IndexBars:
    key: str
    open_ns: np.ndarray        # bar open time, UTC, int64 ns, strictly increasing
    open: np.ndarray           # bid
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    spread_base: np.ndarray    # max(spread_pts, floor_{i,h}) in points (c = 1)
    floor: np.ndarray          # floor_{i,h} of the bar's UTC hour (slippage term)
    seg: np.ndarray            # segment id (COMMON 1.4)
    in_break: np.ndarray       # bar opens inside the daily break (server time)
    tf_ns: int

    @property
    def n(self) -> int:
        return len(self.open_ns)

    @property
    def close_ns(self) -> np.ndarray:
        return self.open_ns + self.tf_ns


def prepare_bars(key: str, df: pd.DataFrame, cfg: dict,
                 clock: ServerClock | None = None) -> IndexBars:
    """Arrays for one index. `df` holds bid M15 bars with time_utc (bar open, UTC), open,
    high, low, close and spread (broker points; spread_pts = spread * 0.01, COMMON 3), as
    BarStore.load returns them (holdout and excluded windows already dropped)."""
    missing = [c for c in ("time_utc", "open", "high", "low", "close", "spread")
               if c not in df.columns]
    if missing:
        raise ValueError(f"{key} bars are missing columns {missing}")
    clock = clock or ServerClock.from_cfg(cfg)
    t = pd.DatetimeIndex(df["time_utc"])
    if t.tz is None:
        raise ValueError(f"{key} time_utc must be tz-aware UTC")
    t = t.tz_convert("UTC").as_unit("ns")
    open_ns = t.asi8.copy()
    if len(open_ns) > 1 and not (np.diff(open_ns) > 0).all():
        raise ValueError(f"{key} bars must be sorted by time_utc without duplicates")
    tf_ns = tf_minutes(cfg["timeframes"]["entry"]) * NS_PER_MIN

    floors = load_spread_floor(cfg, [key])[key]
    floor_by_hour = np.array([floors[h] for h in range(24)], dtype="float64")
    floor = floor_by_hour[t.hour.to_numpy()]
    spread_pts = df["spread"].to_numpy(dtype="float64") * SPREAD_TO_POINTS
    spread_base = np.maximum(spread_pts, floor)

    # Frozen constant: a missing key must fail (KeyError), never fall back to a default.
    break_days = cfg["data_quality"]["segment_break_days"]
    seg = segment_ids(pd.Series(t), break_days).to_numpy()

    wall = clock.wall(open_ns)
    mod = wall.hour.to_numpy() * 60 + wall.minute.to_numpy()
    in_break = (mod >= BREAK_START_MIN) | (mod < BREAK_END_MIN)

    def col(name: str) -> np.ndarray:
        return df[name].to_numpy(dtype="float64").copy()

    return IndexBars(key, open_ns, col("open"), col("high"), col("low"), col("close"),
                     spread_base, floor, seg, in_break, tf_ns)
