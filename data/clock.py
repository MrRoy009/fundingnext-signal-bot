"""Infer the broker's server-clock rule from history.

The US cash open (09:30 New York) produces the biggest jump in tick volume of the day on
US index CFDs. Its server-clock time each week, compared with its known UTC time, gives
the server's UTC offset that week. Candidate rules are scored on how many weeks they
explain. Weeks when US and EU daylight saving differ are what tell the rules apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.timeutil import server_offset

CANDIDATE_RULES = [
    ("America/New_York", 7.0),   # "NY close" brokers
    ("Europe/Athens", 0.0),      # EET/EEST: UTC+2/+3 on EU dates
    ("Europe/Berlin", 0.0),
    ("Europe/London", 0.0),
    ("UTC", 0.0), ("UTC", 2.0), ("UTC", 3.0),
]


@dataclass(frozen=True)
class ClockFit:
    best: tuple[str, float]
    scores: dict[tuple[str, float], float]
    weeks: int

    def describe(self, rule: tuple[str, float]) -> str:
        return f"{rule[0]} {rule[1]:+g}h"


def weekly_open_offsets(server_time: pd.Series, tick_volume: pd.Series) -> pd.Series:
    """Observed server-minus-UTC offset (hours) per week, from the US cash-open volume jump."""
    srv = pd.to_datetime(server_time.to_numpy(), unit="s")
    df = pd.DataFrame({"srv": srv, "vol": tick_volume.to_numpy(dtype=float)})
    df = df[df["srv"].dt.day_of_week < 5]
    df["week"] = df["srv"].dt.to_period("W")
    df["slot"] = df["srv"].dt.hour * 60 + df["srv"].dt.minute
    out = {}
    for week, g in df.groupby("week"):
        if g["srv"].dt.date.nunique() < 3:
            continue
        v = g.groupby("slot")["vol"].mean()
        # Compare adjacent bars only, so the reopen after a break doesn't count as a jump.
        prev_slot = pd.Series(v.index, index=v.index).shift(1)
        step = pd.Series(v.index, index=v.index).diff().min()
        jump = (v / v.shift(1))[(v.index - prev_slot) == step]
        wednesday = week.start_time + pd.Timedelta(days=2)
        ny_open = pd.Timestamp(wednesday.date()).replace(hour=9, minute=30) \
            .tz_localize("America/New_York").tz_convert("UTC")
        open_utc = ny_open.hour * 60 + ny_open.minute
        # Plausible server offsets for MT5 brokers: UTC-1h to UTC+4h.
        jump = jump[(jump.index >= open_utc - 60) & (jump.index <= open_utc + 240)].dropna()
        if jump.empty:
            continue
        out[week] = (jump.idxmax() - open_utc) / 60
    return pd.Series(out, dtype=float)


def fit_server_clock(server_time: pd.Series, tick_volume: pd.Series,
                     configured: tuple[str, float] | None = None) -> ClockFit:
    observed = weekly_open_offsets(server_time, tick_volume)
    rules = list(CANDIDATE_RULES)
    if configured and configured not in rules:
        rules.insert(0, configured)
    scores = {}
    for rule in rules:
        expected = pd.Series({
            w: server_offset(w.start_time + pd.Timedelta(days=2, hours=14), *rule)
            .total_seconds() / 3600 for w in observed.index}, dtype=float)
        scores[rule] = float(((observed - expected).abs() < 0.2).mean()) if len(observed) else 0.0
    best = max(scores, key=scores.get)
    return ClockFit(best, scores, len(observed))
