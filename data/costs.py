"""Spread floor table (COMMON 8.2): a frozen cost constant, never a signal input.

MT5 bar spreads are probably each bar's minimum spread and some bars report zero, so the
backtest prices with max(bar spread, floor) where the floor is the 90th percentile of the
strictly positive M15 bar spreads per UTC hour over pre-holdout data. It is computed once
(compute_spread_floor) and frozen in config.yaml under costs.spread_floor.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SPREAD_TO_POINTS = 0.01          # spread_pts = spread * 0.01 (COMMON 3)
FLOOR_PERCENTILE = 90
FLOOR_START_UTC = "2023-01-01"
FLOOR_END_UTC = "2026-04-30 23:59"
HOURS = range(24)


class SpreadFloorMissing(ValueError):
    pass


def compute_spread_floor(df: pd.DataFrame, start_utc: str = FLOOR_START_UTC,
                         end_utc: str = FLOOR_END_UTC) -> dict[int, float]:
    """{utc_hour: p90 of strictly positive bar spreads in index points, 2 dp} for bars with
    time_utc in [start_utc, end_utc]. Pass pre-holdout bars only (BarStore.load)."""
    start, end = pd.Timestamp(start_utc, tz="UTC"), pd.Timestamp(end_utc, tz="UTC")
    t = df["time_utc"]
    pos = df[(t >= start) & (t <= end) & (df["spread"] > 0)]
    pts = pos["spread"].astype("float64") * SPREAD_TO_POINTS
    out = {}
    for hour, s in pts.groupby(pos["time_utc"].dt.hour):
        out[int(hour)] = round(float(np.percentile(s.to_numpy(), FLOOR_PERCENTILE,
                                                   method="linear")), 2)
    return out


def load_spread_floor(cfg: dict, keys=None) -> dict[str, dict[int, float]]:
    """The frozen floor table from config, checked for every UTC hour of every requested
    instrument (default: all configured). Raises SpreadFloorMissing instead of guessing."""
    table = (cfg.get("costs") or {}).get("spread_floor")
    if not table:
        raise SpreadFloorMissing("config.yaml costs.spread_floor is not set: compute it from "
                                 "pre-holdout data (COMMON 8.2) before running a backtest")
    keys = [i["key"] for i in cfg["instruments"]] if keys is None else list(keys)
    out = {}
    for key in keys:
        hours = table.get(key)
        if not hours:
            raise SpreadFloorMissing(f"config.yaml costs.spread_floor has no table for {key}")
        hours = {int(h): v for h, v in hours.items()}
        missing = [h for h in HOURS if hours.get(h) is None]      # absent or null
        if missing:
            raise SpreadFloorMissing(f"config.yaml costs.spread_floor.{key} is missing UTC "
                                     f"hours {missing}")
        out[key] = {h: _floor_value(key, h, hours[h]) for h in HOURS}
    return out


def _floor_value(key: str, hour: int, value) -> float:
    """A floor must be a finite positive number: NaN would turn every effective spread into
    NaN, and zero or negative would bring back the zero-spread bars the floor exists for."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not (math.isfinite(v) and v > 0):
        raise SpreadFloorMissing(f"config.yaml costs.spread_floor.{key} hour {hour} is "
                                 f"{value!r}; it must be a finite positive number of points")
    return v


def spread_floor(cfg: dict, key: str, utc_hour: int) -> float:
    """floor_{i,h} in index points for one instrument and UTC hour."""
    return load_spread_floor(cfg, [key])[key][int(utc_hour)]
