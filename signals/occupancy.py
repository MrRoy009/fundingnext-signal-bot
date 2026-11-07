"""Reference occupancy (COMMON 7.1), shared by the backtester and the live engine.

A candidate occupies its index from its signal time T until the earliest of:
  (a) the close of the first M15 bar after T whose bid range touches SL or TP (bid prices, no
      spread, no delay);
  (b) its manual time exit;
  (c) for a limit or stop order, its EXPIRY, if the bid range never touched the limit or
      trigger price by then.
Every generated candidate occupies, suppressed ones included, so lockouts depend only on the
candidates and the bars, never on what the owner did. The span is a reference clock, not a
trade: it drives the strategies' lockouts, the live per-index block and the EXIT REMINDER.

No look-ahead: whether an index is occupied at time t depends only on bars that closed by t.
An end computed from later bars is always > t, so it answers "occupied" either way.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from strategies.base import PENDING_TYPES, STOP_ENTRY, Candidate

M15 = pd.Timedelta(minutes=15)

# Tie order when two ends fall at the same instant: a bracket touch on the bar that closes at
# the time exit means the reference trade was no longer open, so no exit reminder.
_PRIORITY = {"SL": 0, "TP": 0, "expiry": 1, "time_exit": 2}


@dataclass(frozen=True)
class Span:
    strategy_id: str
    candidate: Candidate
    suppressed: bool
    start: pd.Timestamp
    end: pd.Timestamp | None      # None: still open with the bars seen so far
    cause: str                    # SL | TP | expiry | time_exit | open

    @property
    def key(self) -> str:
        return self.candidate.key

    def occupies(self, t: pd.Timestamp) -> bool:
        return self.start <= t and (self.end is None or t < self.end)


def _entry_touched(c: Candidate, b: pd.DataFrame) -> bool:
    """Did the bid range reach the limit price (MARKET_OR_LIMIT) or the trigger (STOP_ENTRY)?
    A buy limit is reached from above, a buy stop from below; sells mirrored."""
    up = (c.order_type == STOP_ENTRY) == (c.side == "BUY")
    return bool((b["high"] >= c.entry).any() if up else (b["low"] <= c.entry).any())


def occupancy_end(c: Candidate, bars: pd.DataFrame,
                  tf: pd.Timedelta = M15) -> tuple[pd.Timestamp | None, str]:
    """(end, cause) of the candidate's reference occupancy, from the index's M15 bars.

    `bars` are bid bars with columns time_utc (bar open, UTC), high, low; any bars before T
    are ignored. Missing bars are never filled: a gap simply touches nothing.
    SPEC NOTE: "touches" includes a bar that gaps through the level (long SL: low <= SL,
    long TP: high >= TP; shorts mirrored), and (a) applies from T whether or not a limit or
    stop entry was touched first, as COMMON 7.1 lists it without that condition.
    SPEC NOTE: SL and TP touched in the same bar end the span at that bar's close with cause
    SL (COMMON 8.5 "SL first"); the end time is the same either way.
    """
    if c.signal_time is None:
        raise ValueError("candidate has no signal_time; occupancy starts at T")
    t0 = c.signal_time
    # Positional from here on: a backtester may pass segments concatenated without
    # ignore_index, so index labels are not assumed unique (or sorted by anything).
    b = bars.loc[(bars["time_utc"] >= t0).to_numpy(), ["time_utc", "high", "low"]]
    b = b.sort_values("time_utc", kind="stable").reset_index(drop=True)
    ends: list[tuple[pd.Timestamp, str]] = []

    if c.side == "BUY":
        sl_hit, tp_hit = (b["low"] <= c.stop).to_numpy(), (b["high"] >= c.take_profit).to_numpy()
    else:
        sl_hit, tp_hit = (b["high"] >= c.stop).to_numpy(), (b["low"] <= c.take_profit).to_numpy()
    hit = sl_hit | tp_hit
    if hit.any():
        i = int(hit.argmax())
        ends.append((b["time_utc"].iloc[i] + tf, "SL" if sl_hit[i] else "TP"))

    if c.time_exit_utc is not None:
        ends.append((c.time_exit_utc, "time_exit"))

    if c.order_type in PENDING_TYPES and c.expires_utc is not None:
        # Bars that can still fill the order: open in [T, EXPIRY) (COMMON 8.4 window).
        window = b[b["time_utc"] < c.expires_utc]
        if not _entry_touched(c, window):
            ends.append((c.expires_utc, "expiry"))

    if not ends:
        return None, "open"
    return min(ends, key=lambda e: (e[0], _PRIORITY[e[1]]))


@dataclass
class OccupancyTracker:
    """Spans of every candidate, answering occupied(index, strategy, t).

    Backtest: add each candidate with the index's full bar series. Live: rebuild from the
    day's signals.csv rows each cycle, or call update_bars() as new bars close.
    """
    tf: pd.Timedelta = M15
    spans: list[Span] = field(default_factory=list)

    def add(self, strategy_id: str, candidate: Candidate, bars: pd.DataFrame,
            suppressed: bool = False) -> Span:
        end, cause = occupancy_end(candidate, bars, self.tf)
        span = Span(strategy_id, candidate, suppressed, candidate.signal_time, end, cause)
        self.spans.append(span)
        return span

    def update_bars(self, key: str, bars: pd.DataFrame) -> None:
        """Recompute the spans on `key` with a longer bar series."""
        for n, s in enumerate(self.spans):
            if s.key == key:
                end, cause = occupancy_end(s.candidate, bars, self.tf)
                self.spans[n] = Span(s.strategy_id, s.candidate, s.suppressed, s.start, end, cause)

    def occupying(self, key: str, t: pd.Timestamp,
                  strategy_id: str | None = None) -> list[Span]:
        return [s for s in self.spans
                if s.key == key and (strategy_id is None or s.strategy_id == strategy_id)
                and s.occupies(t)]

    def occupied(self, key: str, strategy_id: str | None, t: pd.Timestamp) -> bool:
        """Is `key` occupied at t by a candidate of `strategy_id` (None: any strategy)?"""
        return bool(self.occupying(key, t, strategy_id))

    def open_at_time_exit(self) -> list[Span]:
        """Spans still open at their manual time exit: these get an EXIT REMINDER (ER-03)."""
        return [s for s in self.spans if s.cause == "time_exit"]
