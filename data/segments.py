"""Segments and data-defect flags (COMMON 1.3-1.5).

A new segment starts where consecutive bars are more than `segment_break_days` apart. Every
rolling or lookback computation restarts at a segment start and never reaches into an earlier
segment.
"""
from __future__ import annotations

import pandas as pd


def segment_ids(time_utc: pd.Series, break_days: float = 7) -> pd.Series:
    """0, 1, 2, ... per bar; increments where the gap to the previous bar exceeds break_days."""
    gap = time_utc.diff() > pd.Timedelta(days=break_days)
    return gap.cumsum().astype("int64")


def tv_bad(time_utc: pd.Series, cfg: dict) -> pd.Series:
    """True where the bar opens inside a configured corrupt tick-volume window."""
    bad = pd.Series(False, index=time_utc.index)
    for w in (cfg.get("data_quality") or {}).get("tick_volume_bad") or ():
        start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        bad |= (time_utc >= start) & (time_utc < end)
    return bad
