"""Synthetic M15 bid bars for backtester tests (COMMON 8 rules are tested on synthetic data
only; no real history is ever priced in tests).

flat_bars builds a regular M15 series around one price with per-bar overrides, so a test can
state exactly which bar touches which level.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

M15 = pd.Timedelta(minutes=15)


def flat_bars(start: pd.Timestamp, end: pd.Timestamp, *, base: float = 40000.0,
              wiggle: float = 5.0, spread: int = 300,
              overrides: Mapping[pd.Timestamp, tuple[float, float, float, float]] | None = None,
              drop: Iterable[pd.Timestamp] = ()) -> pd.DataFrame:
    """Bars opening every 15 min from start to end inclusive: open = close = base,
    high = base + wiggle, low = base - wiggle, raw spread `spread` (points x 100), except the
    bars in `overrides` ({open time: (open, high, low, close)}) and without the bars in
    `drop`."""
    overrides = dict(overrides or {})
    dropped = set(drop)
    rows = []
    for t in pd.date_range(start, end, freq="15min"):
        if t in dropped:
            continue
        o, h, lo, c = overrides.get(t, (base, base + wiggle, base - wiggle, base))
        rows.append((t, o, h, lo, c))
    df = pd.DataFrame(rows, columns=["time_utc", "open", "high", "low", "close"])
    df["tick_volume"] = 100
    df["spread"] = spread
    return df


def concat_bars(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True).sort_values("time_utc",
                                                            kind="stable").reset_index(drop=True)
