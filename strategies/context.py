"""Shared strategy plumbing: signal windows, decision times, the regime gate per T, lockouts, and
a driver that makes a strategy's live generate() and batch scan() one code path.

Why one code path: COMMON 9 / ER-15 require that generate() on history truncated at T equals
the full-history backtest candidate at T, lockouts and daily caps included. WindowedStrategy
implements scan() as a forward replay over the decision times, and generate() as the same
replay over the current session (from local midnight) on bars closed by `now`. A strategy
module only supplies its windows, rank score, regime gate, parameters and per-cycle entry rule.

Windows (COMMON 2, 4.2.1)
    LocalWindow(start, end)                  exchange-local clock, both ends inclusive on T
    in_window(cfg, key, window, T) -> bool   T's local clock in the window on an exchange-calendar
                                             trading date; tradeable instruments only
    decision_times(cfg, key, window, start, end) -> DatetimeIndex
                                             every M15 close T in [start, end) inside the window
    window_group(cfg, key) -> str            the exchange group of an instrument (NYSE | XETRA)
    window_schedule(cfg, windows, start, end) -> DataFrame[T, group, key]   G(T) per group, all T
    window_groups(cfg, windows, T) -> tuple[tuple[str, ...], ...]           the G(T) at one T
    window_set(cfg, windows, T) -> tuple[str, ...]   every instrument in its window at T (union)

Regime (COMMON 6.7)
    regime_frame(table, times, gate, threshold) -> DataFrame (ok, passes, cause, regime,
                                             regime_prob, v_t) per T, vectorised

Lockouts (COMMON 7.1 via signals.occupancy, plus per-local-date caps)
    LockoutBook(strategy_id, bars, tf_min=15, until=None)
        .occupied(key, T) .occupying(key, T) .count(key, local_date, side=None) .add(c, date)

Driver
    WindowedStrategy(cfg, params=None, *, off_grid=False, stats=None)
        subclass sets: id, param_grid, regime_gate ('ACTIVE' | 'CALM'), rank ('rs_4h' | 'rs_day')
        subclass implements: windows(params) -> {key: LocalWindow}; decide(view) -> [Candidate]
        optional hooks: incomplete_cause(view, key), ineligible_cause(view, key)
        provides: scan(...), cycles(...) (selection per T, for diagnostics), generate(...)
    CycleView: what decide() and the hooks see at one T (selection, values at T, regime rows,
        local dates, lockout book, params, candidate_fields()).

SPEC NOTE (one G(T) per exchange group; ESCALATED TO QUANT): COMMON 4.2.1 says "the US window
and the GER40 window never overlap, so G(T) is either {US30, NAS100} or {GER40}". That is false
for MR-TWAP-01 in the DST mismatch weeks (New York on UTC-4, Berlin on UTC+1, mid-to-late March
and late October to early November): 10:30-14:30 New York and 10:00-16:00 Berlin both contain
T = 14:30, 14:45 and 15:00 UTC. A literal "all indices whose window contains T" would rank
GER40 against the US indices at those T and can flip the side the US group allows. The strategy
specs define the window sets per window ("{US30, NAS100} in the US window and {GER40} in its
window", "GER40 is ranked alone in its window") and COMMON 4.2 honest scope says GER40 is always
ranked alone, so the driver builds one G(T) per exchange group (window_group: the instrument's
exchange calendar) and runs the selection separately for each group at the same T. Two groups
at one T give two Selections and their candidates in config order (US group first). A data gap
in one group never blocks the other. Pending a Quant ruling.
SPEC NOTE (G(T) and data validity, literal COMMON 4.2.1 + 1.6): G(T) = indices whose window
contains T *on a valid trading date*, and a valid trading date needs the cash-open M15 bar
(COMMON 1.6; the strategy specs say "valid trading dates only" for every window). So an index
whose date lacks the cash-open bar is NOT in G(T): it is listed in Selection.excluded ("not a
valid trading date: cash-open bar missing") and the rest of its group is ranked without it
(a sole remaining index follows 4.2.5). If no member remains, the group has no cycle at T. The
validity is the one known at T (FeatureService.is_valid_date(d, T): the cash-open bar has closed
by every window T), and it does not depend on whether the M15 bar at T exists. An index with
no feature data at all stays in G(T) as incomplete (its validity cannot be known).
SPEC NOTE (local date of T): window membership, the valid-date check and per-day caps use T's
own exchange-local date and clock. This differs from the local date of the bar's open only at
local midnight, which no strategy window contains.
SPEC NOTE (cycle log, COMMON 4.2.8 "every cycle logs the rank scores of all members"): live
generate() logs the cycle at `now` at INFO. A batch run has ~25k cycles, so scan() logs each
Selection.log_line() at DEBUG only, and a backtest / feasibility run must keep the per-cycle
record itself: pass scan(..., on_selection=...) (or iterate cycles()) and persist
sel.log_line() for every Selection (e.g. a per-run selections file).
SPEC NOTE (gate None): regime_gate None (benchmark B3, "gate off") keeps the regime-status-OK
filter of COMMON 4.2.3 and drops only the ACTIVE / CALM condition.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import pandas as pd

from core.calendar import exchange_for, is_calendar_trading_date, local_to_utc
from core.config import tf_minutes
from signals.occupancy import OccupancyTracker, Span
from strategies.base import Candidate, ParamGrid, StrategyContext, time_exit_utc
from strategies.selection import BUY, SELL, MemberInput, Selection, config_order, select_index

log = logging.getLogger(__name__)

# (strategy id, missing context services) already warned about by generate(): once per process.
_WARNED_UNWIRED: set[tuple[str, tuple[str, ...]]] = set()

REGIME_GATES = ("ACTIVE", "CALM", None)
RANKS = ("rs_4h", "rs_day")

# Data-completeness causes (COMMON 4.2.2), in the order they are checked.
NO_FEATURES = "no features for index"
NO_BAR = "no M15 bar closing at T"
# Reason an index in its window is outside G(T) (COMMON 4.2.1 + 1.6): Selection.excluded.
INVALID_DATE = "not a valid trading date: cash-open bar missing"


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    if t.tzinfo is None:
        raise ValueError("decision times must be tz-aware UTC timestamps")
    return t.tz_convert("UTC").as_unit("ns")


def _bar_minutes(cfg: dict) -> int:
    return tf_minutes(cfg["timeframes"]["entry"])


# ---------------------------------------------------------------------------------------------
# Signal windows and decision times (COMMON 2, 4.2.1)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LocalWindow:
    """A strategy signal window in exchange-local wall-clock time, both ends inclusive on T
    (the close of the signal bar). start == end is a single decision time (SES-IBX-01)."""
    start: time
    end: time

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"window {self.start}-{self.end} crosses midnight; not supported")

    @classmethod
    def of(cls, start: str, end: str) -> LocalWindow:
        return cls(time.fromisoformat(start), time.fromisoformat(end))

    def contains(self, clock: time) -> bool:
        return self.start <= clock <= self.end

    def check_grid(self, bar_minutes: int) -> None:
        """Both ends must be M15 closes, or the window would silently lose an end."""
        for t in (self.start, self.end):
            if t.second or t.microsecond or (t.hour * 60 + t.minute) % bar_minutes:
                raise ValueError(f"window end {t} is not on the {bar_minutes}-minute bar grid")


def in_window(cfg: dict, key: str, window: LocalWindow, T) -> bool:
    """Is T in `key`'s window on an exchange-calendar trading date (COMMON 1.6 calendar part)?
    False for instruments that are not ranked (tradeable: false) and for T that is not an
    M15 close. The data part of validity (cash-open bar) needs the bars: the driver removes an
    index whose date lacks it from G(T) (Selection.excluded)."""
    if key not in config_order(cfg):
        return False
    T = _utc(T)
    bar = _bar_minutes(cfg)
    if T.value % (bar * 60 * 10**9):
        return False
    ex = exchange_for(cfg, key)
    local = T.tz_convert(ex.tz)
    return window.contains(local.time()) and is_calendar_trading_date(ex.calendar, local.date())


def decision_times(cfg: dict, key: str, window: LocalWindow, start, end) -> pd.DatetimeIndex:
    """Every decision time T (M15 close) in [start, end) whose local clock lies in `window` on an
    exchange-calendar trading date, in UTC. Local times are converted with zoneinfo date by date,
    so the UTC clock follows each market's own DST. Empty for non-ranked instruments."""
    start, end = _utc(start), _utc(end)
    empty = pd.DatetimeIndex([], tz="UTC").as_unit("ns")
    if key not in config_order(cfg) or end <= start:
        return empty
    bar = _bar_minutes(cfg)
    window.check_grid(bar)
    ex = exchange_for(cfg, key)
    d, last = start.tz_convert(ex.tz).date(), (end - pd.Timedelta(1)).tz_convert(ex.tz).date()
    dates = []
    while d <= last:
        if is_calendar_trading_date(ex.calendar, d):
            dates.append(d)
        d += timedelta(days=1)
    if not dates:
        return empty
    first = window.start.hour * 60 + window.start.minute
    offsets = np.arange(first, window.end.hour * 60 + window.end.minute + 1, bar)
    midnights = pd.DatetimeIndex([datetime.combine(x, time(0)) for x in dates]).as_unit("ns")
    naive = pd.DatetimeIndex(
        (midnights.asi8[:, None] + offsets[None, :] * 60 * 10**9).ravel()).as_unit("ns")
    # SPEC NOTE: a local clock that does not exist or is ambiguous at a DST change (02:00-03:00
    # local) is dropped; no strategy window comes near those hours.
    local = naive.tz_localize(ex.tz, ambiguous="NaT", nonexistent="NaT")
    utc = local[~local.isna()].tz_convert("UTC").as_unit("ns")
    return utc[(utc >= start) & (utc < end)]


def window_group(cfg: dict, key: str) -> str:
    """The exchange group an instrument is ranked within (its exchange calendar: NYSE for US30
    and NAS100, XETRA for GER40). See the SPEC NOTE on one G(T) per exchange group."""
    return exchange_for(cfg, key).calendar


def _group_rank(cfg: dict, order: tuple[str, ...]) -> dict[str, int]:
    """Group -> position of its first instrument in config order (groups are ordered so)."""
    out: dict[str, int] = {}
    for n, key in enumerate(order):
        out.setdefault(window_group(cfg, key), n)
    return out


def window_schedule(cfg: dict, windows: Mapping[str, LocalWindow], start, end) -> pd.DataFrame:
    """Every G(T) in [start, end): one row per (T, group, member), sorted by T, then group (by
    its first instrument in config order), then config instrument order. The rows of one
    (T, group) are one G(T). Instruments that are not ranked never appear."""
    order = config_order(cfg)
    grank = _group_rank(cfg, order)
    parts = []
    for n, key in enumerate(order):
        if key in windows:
            t = decision_times(cfg, key, windows[key], start, end)
            g = window_group(cfg, key)
            parts.append(pd.DataFrame({"T": t, "group": g, "key": key, "_g": grank[g], "_o": n}))
    if not parts:
        return pd.DataFrame({"T": pd.DatetimeIndex([], tz="UTC").as_unit("ns"),
                             "group": pd.Series([], dtype=object),
                             "key": pd.Series([], dtype=object)})
    out = pd.concat(parts, ignore_index=True).sort_values(["T", "_g", "_o"], kind="stable")
    return out.drop(columns=["_g", "_o"]).reset_index(drop=True)


def window_groups(cfg: dict, windows: Mapping[str, LocalWindow], T) -> tuple[tuple[str, ...], ...]:
    """The G(T) at one decision time, one per exchange group with a member in its window, in
    the same order as window_schedule. Usually zero or one; two in the DST mismatch weeks when
    a strategy's US and GER40 windows overlap in UTC (MR-TWAP-01)."""
    grank = _group_rank(cfg, config_order(cfg))
    groups: dict[str, list[str]] = {}
    for k in window_set(cfg, windows, T):
        groups.setdefault(window_group(cfg, k), []).append(k)
    return tuple(tuple(v) for _, v in sorted(groups.items(), key=lambda kv: grank[kv[0]]))


def window_set(cfg: dict, windows: Mapping[str, LocalWindow], T) -> tuple[str, ...]:
    """Every ranked instrument whose window contains T, in config order. This is the union of
    window_groups(); selection runs per group, never over this union."""
    return tuple(k for k in config_order(cfg) if k in windows and in_window(cfg, k, windows[k], T))


def replay_start(cfg: dict, keys, now) -> pd.Timestamp:
    """Earliest local midnight (UTC) of `now`'s local date over `keys`' exchanges: where a live
    generate() starts replaying. No lockout state crosses a local date: occupancy ends by the
    time exit on the signal's own date (COMMON 7.1 (b)) and the caps count per local date."""
    now = _utc(now)
    starts = []
    for key in keys:
        tz = exchange_for(cfg, key).tz
        starts.append(local_to_utc(now.tz_convert(tz).date(), time(0), tz))
    return min(starts).as_unit("ns") if starts else now


# ---------------------------------------------------------------------------------------------
# Regime gate per decision time (COMMON 6.7)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeRow:
    """Regime at T for one index, as the selection and Candidate fields need it."""
    ok: bool
    passes: bool
    cause: str               # "" when it passes; otherwise the log reason (COMMON 6.7, 4.2.7)
    regime: str              # Candidate.regime: '<ACTIVE|CALM>/<argmax label>' or UNAVAILABLE
    regime_prob: float
    v_t: float


def regime_frame(table: Any, times, gate: str | None, threshold: float = 0.0) -> pd.DataFrame:
    """Vectorised regime status and gate at each T in `times` (one row per T, same order).

    `table` is a regime.service.RegimeTable (or any object with `at_many(times)` returning
    status, cause, v_t, tag, label, regime_prob, and an optional `threshold`). at_many runs with
    its default live_rule=True, so a missing core H1 bar at T is UNAVAILABLE in the backtest
    exactly as live (COMMON 6.6, 9). None: every T is 'regime unavailable: no regime table'."""
    if gate not in REGIME_GATES:
        raise ValueError(f"regime gate must be one of {REGIME_GATES}, not {gate!r}")
    ts = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit("ns")
    n = len(ts)
    if table is None:
        return pd.DataFrame({"ok": np.zeros(n, bool), "passes": np.zeros(n, bool),
                             "cause": ["regime unavailable: no regime table for index"] * n,
                             "regime": ["UNAVAILABLE"] * n, "regime_prob": np.full(n, np.nan),
                             "v_t": np.full(n, np.nan)})
    if n == 0:
        return pd.DataFrame({"ok": [], "passes": [], "cause": [], "regime": [],
                             "regime_prob": [], "v_t": []})
    r = table.at_many(ts).reset_index(drop=True)
    thr = float(getattr(table, "threshold", threshold))
    ok = (r["status"] == "OK").to_numpy()
    v = pd.to_numeric(r["v_t"], errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        if gate == "ACTIVE":
            passes = ok & (v >= thr)
        elif gate == "CALM":
            passes = ok & (v < thr)
        else:
            passes = ok.copy()
    cause = np.full(n, "", dtype=object)
    for i in np.flatnonzero(~ok):
        cause[i] = f"regime unavailable: {r.at[i, 'cause'] or 'unknown cause'}"
    for i in np.flatnonzero(ok & ~passes):
        cause[i] = f"regime gate {gate} not passed: V_T {v[i]:+.4f}"
    tag = r["tag"].fillna("").astype(str).to_numpy()
    label = r["label"].fillna("").astype(str).to_numpy()
    regime = np.array([f"{t}/{lb}" if o else "UNAVAILABLE" for o, t, lb in zip(ok, tag, label)],
                      dtype=object)
    prob = pd.to_numeric(r["regime_prob"], errors="coerce").to_numpy(dtype=float)
    return pd.DataFrame({"ok": ok, "passes": passes, "cause": cause, "regime": regime,
                         "regime_prob": prob, "v_t": v})


# ---------------------------------------------------------------------------------------------
# Lockouts: reference occupancy (COMMON 7.1) and per-local-date caps
# ---------------------------------------------------------------------------------------------

class LockoutBook:
    """One strategy's lockout state during a forward replay.

    occupied(key, T): an earlier candidate of THIS strategy occupies `key` at T (COMMON 7.1,
    computed by signals.occupancy from bid M15 bars). count(key, local_date, side): candidates
    generated on that local date (side None: either side). Every generated candidate is added,
    whatever later happens to its ticket (suppressed, skipped, unfilled), so the state depends
    only on candidates and bars.

    Look-ahead: an occupancy end computed from bars after T is always > T, so it answers
    'occupied' at T either way; queries must move forward in time (ValueError otherwise).
    `until` (live) keeps only bars closed by that time, which cannot change any answer at T <=
    until but keeps generate() from reading anything after `now`.
    """

    def __init__(self, strategy_id: str, bars: Mapping[str, pd.DataFrame], tf_min: int = 15,
                 until: pd.Timestamp | None = None):
        self.strategy_id = strategy_id
        self.tf = pd.Timedelta(minutes=tf_min)
        self.tracker = OccupancyTracker(self.tf)
        self._bars = dict(bars)
        self._times: dict[str, np.ndarray] = {}
        self._until = None if until is None else _utc(until)
        self._active: dict[str, list[Span]] = {}
        self._counts: Counter = Counter()
        self._clock: pd.Timestamp | None = None

    @property
    def spans(self) -> list[Span]:
        return self.tracker.spans

    def _advance(self, T: pd.Timestamp) -> None:
        if self._clock is not None and T < self._clock:
            raise ValueError(f"lockout replay must move forward in time: {T} < {self._clock}")
        self._clock = T

    def occupying(self, key: str, T) -> list[Span]:
        T = _utc(T)
        self._advance(T)
        live = [s for s in self._active.get(key, []) if s.end is None or s.end > T]
        self._active[key] = live                  # ended spans can never occupy a later T
        return [s for s in live if s.occupies(T)]

    def occupied(self, key: str, T) -> bool:
        return bool(self.occupying(key, T))

    def count(self, key: str, local_date: date, side: str | None = None) -> int:
        return self._counts[(key, local_date, side)]

    def _slice(self, c: Candidate) -> pd.DataFrame:
        """The index's bars from T up to the time exit (later bars cannot move the end: the
        time exit always ends the span by then) and, live, closed by `until`."""
        b = self._bars.get(c.key)
        if b is None or not len(b):
            return pd.DataFrame({"time_utc": pd.DatetimeIndex([], tz="UTC"), "high": [],
                                 "low": []})
        if c.key not in self._times:
            self._times[c.key] = pd.DatetimeIndex(
                pd.to_datetime(b["time_utc"], utc=True)).as_unit("ns").asi8
        t = self._times[c.key]
        lo = int(np.searchsorted(t, _utc(c.signal_time).value, side="left"))
        hi = len(t)
        if c.time_exit_utc is not None:
            hi = int(np.searchsorted(t, _utc(c.time_exit_utc).value, side="left"))
        if self._until is not None:
            hi = min(hi, int(np.searchsorted(t, (self._until - self.tf).value, side="right")))
        return b.iloc[lo:max(lo, hi)]

    def add(self, c: Candidate, local_date: date) -> Span:
        T = _utc(c.signal_time)
        self._advance(T)
        span = self.tracker.add(self.strategy_id, c, self._slice(c))
        self._active.setdefault(c.key, []).append(span)
        self._counts[(c.key, local_date, None)] += 1
        self._counts[(c.key, local_date, c.side)] += 1
        return span


# ---------------------------------------------------------------------------------------------
# Rank scores at decision times (COMMON 4.1)
# ---------------------------------------------------------------------------------------------

def rs_4h_scores(fx: Any, times: pd.DatetimeIndex, pos: np.ndarray
                 ) -> tuple[np.ndarray, list[str | None]]:
    """RS_4h of the M15 bar closing at each T (pos = row of that bar, -1 if absent)."""
    col = fx.m15["rs_4h"].to_numpy(dtype=float)
    s = np.where(pos >= 0, col[np.maximum(pos, 0)], np.nan) if len(col) else np.full(len(pos),
                                                                                     np.nan)
    causes = [None if math.isfinite(x) else
              "RS_4h NaN (16-bar lookback incomplete or TODV undefined)" for x in s]
    return s, causes


def rs_day_scores(fx: Any, times: pd.DatetimeIndex, pos: np.ndarray
                  ) -> tuple[np.ndarray, list[str | None]]:
    """RS_day of the H1 bar chosen by the COMMON 2 alignment rule at each T."""
    if not len(fx.h1):
        return np.full(len(times), np.nan), ["no H1 bars"] * len(times)
    hpos, hcause = fx.h1_positions(times)
    col = fx.h1["rs_day"].to_numpy(dtype=float)
    s = np.where(hpos >= 0, col[np.maximum(hpos, 0)], np.nan)
    causes: list[str | None] = []
    for x, hp, hc in zip(s, hpos, hcause):
        if hp < 0:
            causes.append(f"H1 inputs unavailable: {hc}")
        elif not math.isfinite(x):
            causes.append("RS_day NaN (24-bar lookback or ATR14_H1 undefined)")
        else:
            causes.append(None)
    return s, causes


_RANK_FUNCS = {"rs_4h": rs_4h_scores, "rs_day": rs_day_scores}


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------

def _get_fx(features: Any, key: str) -> Any:
    """IndexFeatures for key from a FeatureService or a {key: IndexFeatures} mapping."""
    if features is None:
        return None
    try:
        return features.get(key) if key in features else None
    except KeyError:
        return None


@dataclass
class _KeyData:
    fx: Any
    pos: np.ndarray
    valid: np.ndarray       # T's local date is a valid trading date as known at T (COMMON 1.6)
    scores: np.ndarray
    score_causes: list
    regime: dict            # column -> numpy array (regime_frame columns)
    dates: list


@dataclass
class CycleView:
    """What a strategy sees at one decision time T. Values are read only from rows closing at
    or before T (the M15 bar closing at T is `pos[key]`)."""
    T: pd.Timestamp
    cfg: dict
    params: Mapping[str, Any]
    fx: Mapping[str, Any]                 # key -> IndexFeatures
    pos: Mapping[str, int]                # key -> row of the M15 bar closing at T (-1 absent)
    regime: Mapping[str, RegimeRow]
    local_date: Mapping[str, date]
    book: LockoutBook
    selection: Selection | None = None    # None in the hooks; set before decide()
    columns: dict = field(default_factory=dict, repr=False)   # (key, column) -> ndarray, shared
                                                              # by every view of one scan

    def column(self, key: str, column: str) -> np.ndarray:
        """Whole M15 feature column as an array (cached per scan). Index it only at rows
        closing at or before T, i.e. positions <= pos[key]."""
        arr = self.columns.get((key, column))
        if arr is None:
            arr = self.columns[(key, column)] = self.fx[key].m15[column].to_numpy()
        return arr

    def value(self, key: str, column: str) -> Any:
        """Column of the M15 feature row closing at T (the bar must be present)."""
        p = self.pos[key]
        if p < 0:
            raise ValueError(f"{key}: no M15 bar closing at {self.T}")
        return self.column(key, column)[p]

    def value_at(self, key: str, column: str, t) -> float:
        """Column of the M15 row closing exactly at t <= T; NaN if that bar is absent (e.g.
        RS-MOM-01's previous bar, 15 minutes earlier)."""
        t = _utc(t)
        if t > self.T:
            raise ValueError(f"look-ahead: {t} is after the decision time {self.T}")
        p = self.fx[key].m15_pos(t)
        return float("nan") if p < 0 else self.column(key, column)[p]

    def row(self, key: str) -> pd.Series:
        p = self.pos[key]
        if p < 0:
            raise ValueError(f"{key}: no M15 bar closing at {self.T}")
        return self.fx[key].m15.iloc[p]

    def candidate_fields(self, key: str) -> dict[str, Any]:
        """Candidate fields every strategy sets the same way: signal_time = T, regime and
        regime_prob (COMMON 6.7), rank_score rounded to 2 dp (4.2.8), and the manual time exit
        on T's local date (COMMON 2)."""
        if self.selection is None:
            raise ValueError("candidate_fields needs the selection at T")
        r = self.regime[key]
        return {"signal_time": self.T, "regime": r.regime, "regime_prob": r.regime_prob,
                "rank_score": self.selection.rank_score(key),
                "time_exit_utc": time_exit_utc(self.cfg, key, self.T)}


class WindowedStrategy:
    """Base class for strategies that select one index per side within signal windows
    (COMMON 4.2). Subclasses set the class attributes and implement windows() and decide().

    Per decision time T (window_schedule), in time order, and per exchange group at T:
      1. G(T): indices whose window contains T on a valid trading date (calendar and cash-open
         bar, as known at T; the others go to Selection.excluded). Per member, data
         completeness: M15 bar at T, finite rank score, then incomplete_cause() (strategy
         inputs NaN because of missing data). Any cause -> no candidate for the group (4.2.2).
      2. Regime status and gate (regime_frame), then ineligible_cause() for members that pass.
      3. select_index() -> LONG-SELECTED / SHORT-SELECTED.
      4. decide(view) for the selected indices; the driver checks every candidate (T, member,
         side allowed by the selection, time exit set) and adds it to the LockoutBook.

    decide() must use view.book for lockouts (occupied, count) and view.candidate_fields() for
    the shared fields; it must never look past T.
    """
    id: str = ""
    param_grid: ParamGrid
    regime_gate: str | None = "ACTIVE"
    rank: str = "rs_4h"

    def __init__(self, cfg: dict, params: Mapping[str, Any] | None = None, *,
                 off_grid: bool = False, stats: dict | None = None):
        if self.rank not in RANKS:
            raise ValueError(f"rank must be one of {RANKS}")
        if self.regime_gate not in REGIME_GATES:
            raise ValueError(f"regime_gate must be one of {REGIME_GATES}")
        self.cfg = cfg
        self.off_grid = off_grid
        self.params = self.param_grid.resolve(params, off_grid=off_grid)
        self.stats = dict(stats or {})

    # --- strategy hooks ------------------------------------------------------------------------
    def windows(self, params: Mapping[str, Any]) -> dict[str, LocalWindow]:
        raise NotImplementedError

    def incomplete_cause(self, view: CycleView, key: str) -> str | None:
        """A strategy input at T that is NaN because of missing data (COMMON 4.2.2)."""
        return None

    def ineligible_cause(self, view: CycleView, key: str) -> str | None:
        """A strategy eligibility condition that removes `key` from E(T) (e.g. SES-IBX-01's
        width filter or one-per-day rule). Called only for members whose regime gate passes."""
        return None

    def decide(self, view: CycleView) -> list[Candidate]:
        raise NotImplementedError

    # --- driver ------------------------------------------------------------------------------
    def resolve_params(self, params: Mapping[str, Any] | None) -> Mapping[str, Any]:
        return self.params if params is None else self.param_grid.resolve(
            params, off_grid=self.off_grid)

    def _key_data(self, features, regime_tables, key, times: pd.DatetimeIndex) -> _KeyData:
        n = len(times)
        thr = float(self.cfg["regime"]["gate_threshold"])
        frame = regime_frame((regime_tables or {}).get(key), times, self.regime_gate, thr)
        regime = {c: frame[c].to_numpy() for c in frame.columns}
        tz = exchange_for(self.cfg, key).tz
        dates = list(times.tz_convert(tz).date)
        fx = _get_fx(features, key)
        if fx is None:
            return _KeyData(None, np.full(n, -1), np.zeros(n, bool), np.full(n, np.nan),
                            [NO_FEATURES] * n, regime, dates)
        pos = pd.Index(pd.to_datetime(fx.m15["close_utc"], utc=True)).get_indexer(times)
        valid = self._valid_at(fx, key, times, dates)
        scores, causes = _RANK_FUNCS[self.rank](fx, times, pos)
        return _KeyData(fx, pos, valid, scores, causes, regime, dates)

    def _valid_at(self, fx: Any, key: str, times: pd.DatetimeIndex, dates: list) -> np.ndarray:
        """COMMON 1.6 validity of T's local date as known at T, per T: FeatureService.is_valid_date
        evaluated once per date at the close of its cash-open bar (from then on the answer no
        longer changes) and False for T before that close. Independent of the bar at T."""
        n = len(times)
        if not n:
            return np.zeros(0, bool)
        ex = exchange_for(self.cfg, key)
        bar = pd.Timedelta(minutes=_bar_minutes(self.cfg))
        known: dict[date, tuple[int, bool]] = {}
        for d in dict.fromkeys(dates):
            oc = local_to_utc(d, ex.cash_open, ex.tz).as_unit("ns") + bar
            known[d] = (oc.value, bool(fx.is_valid_date(d, oc)))
        oc_ns = np.fromiter((known[d][0] for d in dates), np.int64, n)
        ok = np.fromiter((known[d][1] for d in dates), bool, n)
        return ok & (times.asi8 >= oc_ns)

    def cycles(self, features: Any, regime_tables: Mapping[str, Any] | None, start, end,
               params: Mapping[str, Any] | None = None, *, until=None
               ) -> Iterator[tuple[Selection, list[Candidate]]]:
        """(Selection, candidates) for every G(T) with T in [start, end), in time order (two
        exchange groups at one T: US group first). The feasibility gate and regime diagnostics
        read the selections; scan() the candidates. A (T, group) whose members are all outside
        G(T) (no valid trading date, see the SPEC NOTE) is not a cycle and yields nothing.

        The replay starts at replay_start(start), the earliest local midnight of start's local
        date over the strategy's instruments, so a mid-session `start` sees the lockouts of that
        session's earlier candidates exactly as generate() does; cycles before `start` are run
        but not yielded. No lockout state crosses a local date (see replay_start)."""
        p = self.resolve_params(params)
        cfg = self.cfg
        order = config_order(cfg)
        start, end = _utc(start), _utc(end)
        wins = self.windows(p)
        replay_from = replay_start(cfg, [k for k in order if k in wins], start)
        sched = window_schedule(cfg, wins, replay_from, end)
        if sched.empty:
            return
        keys = [k for k in order if k in set(sched["key"])]
        data = {k: self._key_data(features, regime_tables, k,
                                  pd.DatetimeIndex(sched.loc[sched["key"] == k, "T"]))
                for k in keys}
        book = LockoutBook(self.id, {k: d.fx.m15 for k, d in data.items() if d.fx is not None},
                           _bar_minutes(cfg), until)
        fxs = {k: d.fx for k, d in data.items()}
        columns: dict = {}
        nxt = dict.fromkeys(keys, 0)
        Ts = pd.DatetimeIndex(sched["T"])
        tk = sched["key"].to_numpy()
        gk = sched["group"].to_numpy()
        ti = Ts.asi8
        bounds = np.r_[0, np.flatnonzero((ti[1:] != ti[:-1]) | (gk[1:] != gk[:-1])) + 1,
                       len(sched)]
        debug = log.isEnabledFor(logging.DEBUG)
        heads = ti[bounds[:-1]]
        shared = np.flatnonzero((heads[1:] == heads[:-1]) & (heads[1:] >= start.value))
        if len(shared):
            log.info("%s: %d decision times have two exchange groups in their windows (DST "
                     "mismatch weeks), first %s; each group is ranked separately (SPEC NOTE, "
                     "escalated to Quant)", self.id, len(shared), Ts[bounds[shared[0] + 1]])

        for a, b in zip(bounds[:-1], bounds[1:]):
            T = Ts[a]
            in_window = list(tk[a:b])
            idx = {}
            for k in in_window:
                idx[k] = nxt[k]
                nxt[k] += 1
            # COMMON 4.2.1 + 1.6: an index whose date is not a valid trading date is not in G(T).
            excluded = {k: INVALID_DATE for k in in_window
                        if data[k].fx is not None and not data[k].valid[idx[k]]}
            members = [k for k in in_window if k not in excluded]
            if not members:
                if T >= start:
                    lvl = logging.INFO if until is not None and T == until else logging.DEBUG
                    if log.isEnabledFor(lvl):
                        log.log(lvl, "%s %s UTC: G(T) empty, no cycle: %s", self.id,
                                f"{T:%Y-%m-%d %H:%M}",
                                "; ".join(f"{k}: {r}" for k, r in excluded.items()))
                continue
            pos = {k: int(data[k].pos[idx[k]]) for k in members}
            rrows = {}
            for k in members:
                r, i = data[k].regime, idx[k]
                rrows[k] = RegimeRow(bool(r["ok"][i]), bool(r["passes"][i]), str(r["cause"][i]),
                                     str(r["regime"][i]), float(r["regime_prob"][i]),
                                     float(r["v_t"][i]))
            view = CycleView(T, cfg, p, fxs, pos, rrows,
                             {k: data[k].dates[idx[k]] for k in members}, book,
                             columns=columns)

            inputs = []
            incomplete = False
            for k in members:
                d, i = data[k], idx[k]
                if d.fx is None:
                    cause = NO_FEATURES
                elif pos[k] < 0:
                    cause = NO_BAR
                else:
                    cause = d.score_causes[i] or self.incomplete_cause(view, k)
                incomplete |= cause is not None
                inputs.append(MemberInput(k, float(d.scores[i]), cause))
            if not incomplete:
                inputs = []
                for k in members:
                    r = rrows[k]
                    gate_cause = r.cause
                    passes = r.passes
                    if passes:
                        extra = self.ineligible_cause(view, k)
                        if extra:
                            passes, gate_cause = False, extra
                    inputs.append(MemberInput(k, float(data[k].scores[idx[k]]), None, r.ok,
                                              r.cause if not r.ok else "", passes, gate_cause))
            sel = select_index(T, inputs, order, self.id, excluded)
            view.selection = sel
            cands = self.decide(view) if sel.has_side else []
            for c in cands:
                self._check(c, sel)
                book.add(c, view.local_date[c.key])
            if T < start:
                continue                          # session replay before `start`
            if debug:
                log.debug(sel.log_line())
            yield sel, cands

    def _check(self, c: Candidate, sel: Selection) -> None:
        """Strategy bugs that would break COMMON 4.2.6 or the live/backtest equality."""
        if c.signal_time is None or _utc(c.signal_time) != sel.T:
            raise ValueError(f"{self.id}: candidate signal_time {c.signal_time} != T {sel.T}")
        if c.key not in sel.members:
            raise ValueError(f"{self.id}: candidate on {c.key}, not in G(T) {sel.members}")
        if c.side not in (BUY, SELL) or not sel.allows(c.key, c.side):
            raise ValueError(f"{self.id}: {c.side} on {c.key} not allowed: LONG-SELECTED "
                             f"{sel.long}, SHORT-SELECTED {sel.short} (COMMON 4.2.6)")
        if c.time_exit_utc is None:
            raise ValueError(f"{self.id}: candidate without time_exit_utc; use "
                             "view.candidate_fields() (lockout replay relies on it)")

    def scan(self, features: Any, regime_tables: Mapping[str, Any] | None, start, end,
             params: Mapping[str, Any] | None = None, *,
             on_selection: Callable[[Selection], None] | None = None) -> list[Candidate]:
        """Batch path: every candidate with T in [start, end); lockouts replayed from the
        local midnight of start's session (cycles()).

        on_selection: called with every Selection (each cycle, in order). COMMON 4.2.8 needs
        every cycle's rank scores logged; scan() logs them at DEBUG only, so a backtest or
        feasibility run passes a sink here and persists sel.log_line() (see the SPEC NOTE)."""
        out: list[Candidate] = []
        for sel, cands in self.cycles(features, regime_tables, start, end, params):
            if on_selection is not None:
                on_selection(sel)
            out.extend(cands)
        return out

    def generate(self, bars: dict[str, pd.DataFrame], regime_bars: dict[str, pd.DataFrame],
                 now: pd.Timestamp, context: StrategyContext) -> list[Candidate]:
        """Live path: the same driver as scan() over [now, now + 1 bar) on context.features /
        context.regime_tables, reading bars closed by `now` only. cycles() replays the session
        from local midnight, so lockouts match the backtest; returns the candidates at T = now
        (of every exchange group at now). `bars` and `regime_bars` are not read: the services in
        the context were built from them.

        Missing context services: [] and a WARNING (once per process per strategy), so an
        unwired run (signals/run.py must fill StrategyContext.features and .regime_tables) is
        never mistaken for "no setup" (PROMPT: no silent fallback)."""
        missing = tuple(n for n in ("features", "regime_tables")
                        if context is None or getattr(context, n, None) is None)
        if missing:
            tag = (self.id, missing)
            if tag not in _WARNED_UNWIRED:
                _WARNED_UNWIRED.add(tag)
                log.warning("%s: StrategyContext.%s not set; generate() returns no candidate "
                            "(the cycle must build the FeatureService and the RegimeService "
                            "tables as of now)", self.id, " and .".join(missing))
            return []
        now = _utc(now)
        end = now + pd.Timedelta(minutes=_bar_minutes(self.cfg))
        out: list[Candidate] = []
        for sel, cands in self.cycles(context.features, context.regime_tables, now, end,
                                      until=now):
            if sel.T == now:
                log.info(sel.log_line())
                out.extend(cands)
        return out
