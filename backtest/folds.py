"""Walk-forward folds (COMMON 10.2), trade-to-fold assignment, trading-day calendars and the
minimum trade-count checks of COMMON 10.4 (amendment A1: pro-rated fold minimum).

Why a separate module: the runner, the trial log, the statistics and the report all have to
agree on exactly which dates a fold covers, which fold a trade belongs to and which days the
daily series has. One definition here keeps them from drifting apart.

Public API
    Fold, TEST_FOLDS, fold_of, fold_data_shares    re-exported from backtest.feasibility (the
                                                   9 COMMON 10.2 test windows, defined once)
    TRAIN_MONTHS = 12
    WalkForwardFold(n, train, test)
    train_window(test, months=12) -> Fold          [test.first - 12 months, test.first)
    WALK_FORWARD: tuple[WalkForwardFold, ...]      the 9 (train, test) pairs
    SECONDARY_SAMPLE: Fold                         2023-03-01..2026-04-30 (n = 0, report only)
    window_span(windows) -> (start, end)           UTC [first 00:00, last + 1 day 00:00)
    is_contiguous(windows) -> bool
    assign_folds(times, windows) -> np.ndarray     fold number of each time, -1 = outside
    in_windows(times, windows) -> np.ndarray       bool mask
    window_label(windows) -> str
    trading_calendar(cfg, windows, keys=None) -> pd.DatetimeIndex
                                                   union of exchange trading days (naive dates)
    shares_by_fold(shares) -> dict[int, float]
    trade_count_check(trades, shares, folds=TEST_FOLDS) -> TradeCountCheck      (10.4 a, b)
    bucket_stats(trades, by, min_n=30) -> DataFrame                             (10.4 c)
    sensitivity_cell_status(n, expectancy_usd) -> str                           (10.4 d)
    competition_feasibility(trades, calendar, cfg) -> CompetitionFeasibility    (10.4 e)

SPEC NOTE (fold windows): COMMON 10.2 lists calendar dates; they are UTC dates, each window
[first 00:00 UTC, last + 1 day 00:00 UTC) (backtest.feasibility.Fold). A trade belongs to the
fold of its signal time T, never of its fill or exit (10.2).
SPEC NOTE (train window): "train 12 months" before each test window, read literally as the 12
calendar months immediately before the test window's first day: [first - 12 months, first),
e.g. fold 1 trains on 2023-02-01..2024-01-31. Train windows overlap earlier test windows by
construction (rolling walk-forward); the embargo (runner) handles the boundary.
SPEC NOTE (trading days of the daily series): COMMON 10.2 sets the calendar to "the union of
trading days of the three indices inside the test windows". 10.4 separately speaks of
"trading days with data", so a trading day is a day the exchange trades, by its frozen
calendar (COMMON 1.6): a weekday that is not a full holiday, for at least one ranked index,
whether or not the demo feed has bars that day. Early-close dates ARE trading days here (the
exchange trades that morning; MTM of a held position moves): 1.6 excludes them only from a
"valid trading date" (no ticket that day), a different notion. Days inside the 2025 hole and
the A1 excluded window are in the calendar with zero change (no position can be open there).
(fold_data_shares, backtest.feasibility, counts valid trading dates with data for the 10.4
pro-rating; it is not this calendar.) The day is the
server date (00:00 Europe/Athens, O8); every cash session of the three indices lies inside one
Athens date, so the exchange-local date and the server date coincide for these days.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from backtest.feasibility import TEST_FOLDS, Fold, fold_data_shares, fold_of
from backtest.market import ServerClock
from core.calendar import exchange_for, is_full_holiday
from strategies.selection import config_order

__all__ = [
    "Fold", "TEST_FOLDS", "fold_of", "fold_data_shares", "TRAIN_MONTHS", "WalkForwardFold",
    "train_window", "WALK_FORWARD", "SECONDARY_SAMPLE", "window_span", "is_contiguous",
    "assign_folds", "OUTSIDE", "in_windows", "window_label", "trading_calendar", "shares_by_fold",
    "MIN_TOTAL_TRADES", "MIN_PER_FULL_FOLD", "MIN_FOLDS_OK", "MIN_BUCKET",
    "MIN_SENSITIVITY_CELL", "COMP_MIN_TRADE_DAYS", "COMP_WINDOW_DAYS", "COMP_MIN_SHARE",
    "COMP_RULING_NOTE",
    "FoldCount", "TradeCountCheck", "trade_count_check", "bucket_stats", "OTHER",
    "sensitivity_cell_status", "POSITIVE", "NOT_POSITIVE", "INSUFFICIENT",
    "CompetitionFeasibility", "competition_feasibility",
]

TRAIN_MONTHS = 12                                  # COMMON 10.2
SECONDARY_SAMPLE = Fold(0, date(2023, 3, 1), date(2026, 4, 30))   # COMMON 10.2, report only

# COMMON 10.4 (run A, STRESS-d2)
MIN_TOTAL_TRADES = 150
MIN_PER_FULL_FOLD = 4
MIN_FOLDS_OK = 7
MIN_BUCKET = 30
MIN_SENSITIVITY_CELL = 100
COMP_MIN_TRADE_DAYS = 5
COMP_WINDOW_DAYS = 20
COMP_MIN_SHARE = 0.80
COMP_RULING_NOTE = ("SPEC NOTE awaiting Quant ruling: 'share'/'passed' count rolling windows "
                    "over the data hole and the A1 excluded window as failed (literal); "
                    "'share_with_data'/'passed_with_data' use only windows with data")

OTHER = "OTHER"
OUTSIDE = -1                                       # assign_folds: time in no window
POSITIVE, NOT_POSITIVE, INSUFFICIENT = "positive", "not positive", "insufficient"


# ---------------------------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class WalkForwardFold:
    """One walk-forward step: parameters are chosen on `train` and applied to `test`."""
    n: int
    train: Fold
    test: Fold


def train_window(test: Fold, months: int = TRAIN_MONTHS) -> Fold:
    """The `months` calendar months before test.first (see the SPEC NOTE). Same fold number."""
    first = (pd.Timestamp(test.first) - pd.DateOffset(months=months)).date()
    return Fold(test.n, first, test.first - timedelta(days=1))


WALK_FORWARD: tuple[WalkForwardFold, ...] = tuple(
    WalkForwardFold(f.n, train_window(f), f) for f in TEST_FOLDS)


def _windows(windows: Fold | Iterable[Fold]) -> tuple[Fold, ...]:
    out = (windows,) if isinstance(windows, Fold) else tuple(windows)
    if not out:
        raise ValueError("no windows given")
    return out


def window_span(windows: Fold | Iterable[Fold]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC [earliest start, latest end) over the windows."""
    ws = _windows(windows)
    return min(w.start for w in ws), max(w.end for w in ws)


def is_contiguous(windows: Fold | Iterable[Fold]) -> bool:
    """True if the windows, sorted by start, leave no gap and do not overlap."""
    ws = sorted(_windows(windows), key=lambda w: w.start)
    return all(a.end == b.start for a, b in zip(ws[:-1], ws[1:]))


def _check_disjoint(ws: Sequence[Fold]) -> list[Fold]:
    s = sorted(ws, key=lambda w: w.start)
    for a, b in zip(s[:-1], s[1:]):
        if b.start < a.end:
            raise ValueError(f"windows overlap: {a.label} and {b.label}; a time would belong to "
                             "two folds")
    return s


def _ns(times) -> np.ndarray:
    t = pd.DatetimeIndex(pd.to_datetime(pd.Series(times), utc=True))
    return t.as_unit("ns").asi8


def assign_folds(times, windows: Fold | Iterable[Fold] = TEST_FOLDS) -> np.ndarray:
    """Fold number (Fold.n) of each time, OUTSIDE (-1) if it lies in no window (fold 0 is the
    secondary sample). Vectorised; the windows must not overlap. NaT -> OUTSIDE."""
    ws = _check_disjoint(_windows(windows))
    raw = pd.to_datetime(pd.Series(times), utc=True)
    out = np.full(len(raw), OUTSIDE, dtype=np.int64)
    if not len(raw):
        return out
    ok = raw.notna().to_numpy()
    t = pd.DatetimeIndex(raw[ok]).as_unit("ns").asi8
    starts = np.array([w.start.value for w in ws], dtype=np.int64)
    ends = np.array([w.end.value for w in ws], dtype=np.int64)
    nums = np.array([w.n for w in ws], dtype=np.int64)
    j = np.searchsorted(starts, t, side="right") - 1
    inside = (j >= 0) & (t < ends[np.maximum(j, 0)])
    sub = np.where(inside, nums[np.maximum(j, 0)], OUTSIDE)
    out[ok] = sub
    return out


def in_windows(times, windows: Fold | Iterable[Fold] = TEST_FOLDS) -> np.ndarray:
    """True where the time lies in one of the windows (fold number 0 is allowed)."""
    ws = _check_disjoint(_windows(windows))
    t = _ns(times)
    mask = np.zeros(len(t), dtype=bool)
    for w in ws:
        mask |= (t >= w.start.value) & (t < w.end.value)
    return mask


def window_label(windows: Fold | Iterable[Fold]) -> str:
    """'2024-02-01..2026-04-30 (folds 1-9)' for contiguous windows, else each label."""
    ws = sorted(_windows(windows), key=lambda w: w.start)
    nums = [w.n for w in ws]
    if len(ws) == 1:
        return f"{ws[0].label} (fold {ws[0].n})"
    if is_contiguous(ws):
        return f"{ws[0].first.isoformat()}..{ws[-1].last.isoformat()} (folds {nums[0]}-{nums[-1]})"
    return "; ".join(f"{w.label} (fold {w.n})" for w in ws)


def trading_calendar(cfg: dict, windows: Fold | Iterable[Fold],
                     keys: Sequence[str] | None = None) -> pd.DatetimeIndex:
    """Sorted naive dates (midnight) on which at least one of `keys` (default: the ranked
    indices) trades (weekday, not a full holiday; early closes included) inside one of the
    windows (SPEC NOTE)."""
    keys = list(keys or config_order(cfg))
    cals = sorted({exchange_for(cfg, k).calendar for k in keys})
    days: set[date] = set()
    for w in _windows(windows):
        d = w.first
        while d <= w.last:
            if d.weekday() < 5 and any(not is_full_holiday(c, d) for c in cals):
                days.add(d)
            d += timedelta(days=1)
    return pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in days), name="date")


def shares_by_fold(shares: pd.DataFrame | Mapping[int, float]) -> dict[int, float]:
    """{fold: data share} from fold_data_shares() output or a mapping."""
    if isinstance(shares, pd.DataFrame):
        return {int(f): float(s) for f, s in zip(shares["fold"], shares["data_share"])}
    return {int(k): float(v) for k, v in shares.items()}


# ---------------------------------------------------------------------------------------------
# COMMON 10.4 minimum trade counts
# ---------------------------------------------------------------------------------------------

def _filled(trades: pd.DataFrame) -> pd.DataFrame:
    if "status" in trades.columns:
        return trades[trades["status"] == "filled"]
    return trades


@dataclass(frozen=True)
class FoldCount:
    fold: int
    n: int
    data_share: float
    required: float          # MIN_PER_FULL_FOLD x data_share
    ok: bool


@dataclass(frozen=True)
class TradeCountCheck:
    """COMMON 10.4 (a) total filled trades and (b) the pro-rated per-fold minimum.

    SPEC NOTE (pro-rating): a fold needs n >= 4 x share, share = the fold's "share of trading
    days with data" (fold_data_shares), unrounded; a fold with share 0 needs 0 trades. The
    "7 of 9 folds" count itself is not pro-rated (A1 pro-rates only the per-fold minimum)."""
    n_total: int
    total_ok: bool
    folds: tuple[FoldCount, ...]
    n_folds_ok: int
    folds_ok: bool

    @property
    def passed(self) -> bool:
        return self.total_ok and self.folds_ok

    def to_dict(self) -> dict[str, Any]:
        return {"n_total": self.n_total, "min_total": MIN_TOTAL_TRADES,
                "total_ok": self.total_ok, "n_folds_ok": self.n_folds_ok,
                "min_folds_ok": MIN_FOLDS_OK, "folds_ok": self.folds_ok,
                "passed": self.passed,
                "folds": [{"fold": f.fold, "n": f.n, "data_share": f.data_share,
                           "required": f.required, "ok": f.ok} for f in self.folds]}


def trade_count_check(trades: pd.DataFrame, shares: pd.DataFrame | Mapping[int, float],
                      folds: Sequence[Fold] = TEST_FOLDS) -> TradeCountCheck:
    """Filled trades (status 'filled' if the column exists) by the fold of their signal time.
    `shares`: fold_data_shares() output or {fold: share}; a fold without a share raises."""
    f = _filled(trades)
    fold_no = assign_folds(f["signal_time"], folds) if len(f) else np.zeros(0, np.int64)
    sh = shares_by_fold(shares)
    rows = []
    for w in folds:
        if w.n not in sh:
            raise ValueError(f"no data share for fold {w.n}")
        s = sh[w.n]
        n = int((fold_no == w.n).sum())
        req = MIN_PER_FULL_FOLD * s
        rows.append(FoldCount(w.n, n, s, req, bool(n >= req)))
    n_ok = sum(r.ok for r in rows)
    n_total = int((fold_no != OUTSIDE).sum())
    return TradeCountCheck(n_total, n_total >= MIN_TOTAL_TRADES, tuple(rows), n_ok,
                           n_ok >= MIN_FOLDS_OK)


def bucket_stats(trades: pd.DataFrame, by: str | Sequence[str],
                 min_n: int = MIN_BUCKET) -> pd.DataFrame:
    """$ and R statistics of filled trades per bucket (e.g. 'key', 'side', 'regime'); buckets
    with fewer than `min_n` trades are merged into one OTHER row (COMMON 10.4). Columns:
    bucket, n, pnl_usd, mean_pnl_usd, mean_r, merged (the bucket names inside OTHER)."""
    f = _filled(trades)
    cols = [by] if isinstance(by, str) else list(by)
    if not len(f):
        return pd.DataFrame(columns=["bucket", "n", "pnl_usd", "mean_pnl_usd", "mean_r",
                                     "merged"])
    names = f[cols].astype(str).agg("/".join, axis=1)
    counts = names.value_counts()
    small = set(counts[counts < min_n].index)
    label = names.where(~names.isin(small), OTHER)
    rows = []
    order = sorted(set(label) - {OTHER}) + ([OTHER] if small else [])
    for b in order:
        g = f[label == b]
        rows.append({"bucket": b, "n": int(len(g)), "pnl_usd": float(g["pnl_usd"].sum()),
                     "mean_pnl_usd": float(g["pnl_usd"].mean()),
                     "mean_r": float(g["r_multiple"].mean()),
                     "merged": ",".join(sorted(small)) if b == OTHER else ""})
    return pd.DataFrame(rows)


def sensitivity_cell_status(n: int, expectancy_usd: float) -> str:
    """COMMON 10.4 / 10.5.5: fewer than 100 trades -> 'insufficient' (a failed cell);
    otherwise 'positive' iff $ expectancy > 0."""
    if n < MIN_SENSITIVITY_CELL:
        return INSUFFICIENT
    return POSITIVE if (expectancy_usd is not None and np.isfinite(expectancy_usd)
                        and expectancy_usd > 0) else NOT_POSITIVE


@dataclass(frozen=True)
class CompetitionFeasibility:
    """COMMON 10.4 (e): >= 5 distinct trade days in >= 80% of rolling 20-trading-day windows.

    SPEC NOTE (trade day): the server date (00:00 Europe/Athens, O8) of a trade's FILL, i.e. a
    day on which a position was opened, as the competition counts days with trades. Rolling
    windows are every run of 20 consecutive days of the calendar (trading_calendar).
    SPEC NOTE (windows over the data hole, AWAITING A QUANT RULING, COMP_RULING_NOTE): windows
    over the data hole and the A1 excluded window are kept and can only fail (the literal
    reading; `share` and `passed` use them). On the test calendar this caps `share` near 0.85
    even for a fill on every day with data, i.e. about 94% of the windows with data must pass
    instead of 80%. `share_with_data` / `passed_with_data` are the same check over windows
    with at least one day with bars (given `data_days`); both readings are always reported
    side by side, and neither is chosen in code."""
    n_windows: int
    n_ok: int
    share: float
    passed: bool
    windows: pd.DataFrame = field(repr=False)     # start, end, trade_days, ok
    share_with_data: float | None = None

    @property
    def passed_with_data(self) -> bool | None:
        s = self.share_with_data
        return None if s is None or not np.isfinite(s) else bool(s >= COMP_MIN_SHARE)

    def to_dict(self) -> dict[str, Any]:
        return {"n_windows": self.n_windows, "n_ok": self.n_ok, "share": self.share,
                "min_share": COMP_MIN_SHARE, "passed": self.passed,
                "share_with_data": self.share_with_data,
                "passed_with_data": self.passed_with_data, "ruling": COMP_RULING_NOTE}


def competition_feasibility(trades: pd.DataFrame, calendar: pd.DatetimeIndex, cfg: dict, *,
                            data_days: pd.DatetimeIndex | None = None) -> CompetitionFeasibility:
    """trades: filled rows with fill_time (UTC). calendar: trading_calendar(). data_days:
    optional calendar days on which bars exist (for share_with_data only)."""
    f = _filled(trades)
    cal = pd.DatetimeIndex(calendar).normalize()
    clock = ServerClock.from_cfg(cfg)
    if len(f):
        fill = pd.DatetimeIndex(pd.to_datetime(f["fill_time"], utc=True).dropna()).as_unit("ns")
        tdays = pd.DatetimeIndex(np.unique(clock.dates(fill.asi8))).as_unit("ns")
    else:
        tdays = pd.DatetimeIndex([]).as_unit("ns")
    has = cal.as_unit("ns").isin(tdays).astype(np.int64)
    w = COMP_WINDOW_DAYS
    n = len(cal) - w + 1
    if n <= 0:
        empty = pd.DataFrame(columns=["start", "end", "trade_days", "ok"])
        return CompetitionFeasibility(0, 0, float("nan"), False, empty)
    csum = np.r_[0, np.cumsum(has)]
    counts = csum[w:] - csum[:-w]
    ok = counts >= COMP_MIN_TRADE_DAYS
    frame = pd.DataFrame({"start": cal[:n], "end": cal[w - 1:], "trade_days": counts, "ok": ok})
    share = float(ok.mean())
    swd = None
    if data_days is not None:
        dd = cal.as_unit("ns").isin(pd.DatetimeIndex(data_days).normalize().as_unit("ns"))
        dsum = np.r_[0, np.cumsum(dd.astype(np.int64))]
        with_data = (dsum[w:] - dsum[:-w]) > 0
        swd = float(ok[with_data].mean()) if with_data.any() else float("nan")
    return CompetitionFeasibility(n, int(ok.sum()), share, share >= COMP_MIN_SHARE, frame, swd)
