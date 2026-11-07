"""Competition replay (COMMON 10.5 item 9): how often the rule set would have been breached.

A competition window is 30 calendar days starting at 00:00 server time (Europe/Athens, owner
decision O8) of every Monday in the test span. Each window starts a fresh account at
rules.initial_balance (100,000) and applies the backtest's own MTM path at one contract, in two
versions (equity is the backtest's worst-intrabar MTM equity at bar closes, backtest.results
BacktestResult.equity; t0 is the last mark at or before the window start):
- raw:  E_w(t) = 100,000 + (equity(t) - equity(t0)), the backtest's $ amounts unchanged;
- norm: E_w(t) = 100,000 + s * (equity(t) - equity(t0)), s = 100,000 / bt_balance0, where
  bt_balance0 is the backtest's BALANCE at t0. Backtest lots are sized from the backtest's own
  running balance (COMMON 8.7, 0.5% of sim_balance), so the raw path of a window that starts
  while the backtest is below 100,000 shows every $ swing scaled down by bt_balance0 / 100,000:
  exactly the windows after a drawdown, whose breaches matter for 10.5.9. Under proportional
  sizing an account started at 100,000 follows the backtest's path times s (both compound by
  the same returns), so the norm path is exact apart from lot rounding (0.01 lot) and the 3-lot
  cap, and carried-in positions sized before t0.
The competition contract is a separate backtest run (run_both_contracts) and is replayed the
same way.

Per window and per path (columns *_raw and *_norm) it reports
- total-loss breach: some mark with E_w <= initial * (1 - max_loss_pct / 100);
- daily-loss breach: some server day whose lowest mark is <= reference - base * daily_loss_pct
  / 100, with base and reference from config rules.daily_loss_base / daily_loss_reference
  (initial_balance, start_of_day_balance or start_of_day_equity), exactly as risk.rules does;
- personal-stop breach: lowest mark of a day <= start-of-day balance * (1 - personal %);
- trade days: distinct server dates with a fill inside the window (>= rules.min_trading_days);
- data coverage: n_marks, no_data (n_marks == 0), missing_weekdays (Mon-Fri server dates of
  the window without a mark; holidays count) and partial_data (has marks, missing_weekdays > 0).
The verdict columns total_breach, daily_breach and personal_breach (and the *_days counts) are
the UNION of the raw and norm paths (per window, and per day for the day counts). Pass
(10.5.9): zero total-loss breaches and daily-loss breaches in at most 1% of the windows WITH
DATA (n_marks > 0), and at least one window with data.

SPEC NOTEs (most literal reading; listed in the GATE 3 tooling report)
- Raw vs balance-normalised path (QA finding on the earlier SPEC NOTE 12): COMMON 10.5.9 does
  not say whether a window replays the backtest's compounded $ amounts or a fresh 100,000
  account's. The raw reading understates breaches when the backtest is below 100,000 and the
  norm reading when lot rounding / the cap bind, so the verdict uses the union of both, which
  is at least as strict as either (COMMON 10.5: QA may add stricter criteria). Both paths are
  in the table, with bt_balance0 and s, and both counts are in summary(). A backtest balance
  <= 0 at a window start is an error.
- Windows are those whose start Monday AND last day (start + 29) lie inside [span_start,
  span_end]: a truncated window would understate breaches.
- The 10.5.9 share is over windows WITH DATA. A window lying wholly in a data hole (the A1
  excluded window, the 2025 hole) is flat and can never breach, so counting it would make the
  1% bar easier (COMMON 10.5: QA may add stricter criteria, never weaker). Such windows stay in
  the table, flagged no_data, and the all-window share and the share over fully covered
  windows (no missing weekday) are reported beside it, not used for the verdict. A replay
  whose span holds no mark at all is a caller error (ValueError), never a silent pass.
- "Breach" uses <= like risk.rules.check_ticket (reaching the floor fails the ticket).
- Start-of-day values are the window account's balance / equity at the last mark before
  00:00 server time of that day (the window's start values on its first day). A mark at time m
  belongs to the server day containing m - 1 ns, as backtest.results.daily_eod does.
- Event times follow backtest.fills.seen_ns / account.series: a fill or exit stamped at a bar
  close is known at that close; one stamped at a bar open (fill_at_open / exit_at_open) is
  known 1 ns later, i.e. at the next mark. A mark t carries a position's MTM iff
  fill_seen <= t < exit_seen, and its realised P&L is in the balance iff t >= exit_seen.
- A position open at the window start (only possible when held over a weekend, MISSED-EXIT)
  stays in the window: its MTM change after t0 counts (times s on the norm path, although a
  fresh account would not hold it at all), its unrealised P&L u0 at t0 does not.
  The window balance drops u0 at the first mark >= that position's exit_seen. With several
  carried positions exiting at DIFFERENT seen times the per-position split of u0 is not in the
  equity path, so the whole u0 is removed at the last of those exits (window balance, hence
  start-of-day balance, is approximate between the first and the last carried exit); such
  windows are flagged carry_approx. The count is reported per window (carried_in).
- Each breach type is counted independently; a window is not cut short by a breach.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Mapping

import numpy as np
import pandas as pd

from backtest.fills import AFTER_OPEN_NS
from backtest.market import ServerClock

WINDOW_DAYS = 30                 # COMMON 10.5.9
DAILY_BREACH_MAX_SHARE = 0.01    # COMMON 10.5.9: at most 1% of windows
BAR_NS = 15 * 60 * 1_000_000_000

# exit_note values (backtest.fills._exit) of exits priced at a bar OPEN (open-stamped). The
# trades frame carries fill_at_open but no exit_at_open column, so replay_backtest infers it
# from these notes; tests/test_replay.py checks they are exactly fills._exit's _OPEN events.
OPEN_EXIT_NOTES = frozenset({
    "at the clock bar open",
    "triggered in the daily break; resolved at the next open",
})

_BASES = ("initial_balance", "start_of_day_balance", "start_of_day_equity")
_REFERENCES = ("start_of_day_balance", "start_of_day_equity")


@dataclass(frozen=True)
class ReplayRules:
    """Competition rule set used by the replay (config rules / risk; COMMON 10.5.9)."""
    initial_balance: float = 100_000.0
    max_loss_pct: float = 10.0
    daily_loss_pct: float = 5.0
    daily_loss_base: str = "initial_balance"
    daily_loss_reference: str = "start_of_day_balance"
    personal_daily_stop_pct: float = 3.0
    min_trade_days: int = 5
    window_days: int = WINDOW_DAYS

    def __post_init__(self) -> None:
        if self.daily_loss_base not in _BASES:
            raise ValueError(f"daily_loss_base must be one of {_BASES}")
        if self.daily_loss_reference not in _REFERENCES:
            raise ValueError(f"daily_loss_reference must be one of {_REFERENCES}")
        if self.window_days < 1:
            raise ValueError("window_days must be >= 1")

    @classmethod
    def from_cfg(cls, cfg: dict) -> "ReplayRules":
        r, k = cfg["rules"], cfg.get("risk", {})
        return cls(initial_balance=float(r["initial_balance"]),
                   max_loss_pct=float(r["max_loss_pct"]),
                   daily_loss_pct=float(r["daily_loss_pct"]),
                   daily_loss_base=str(r["daily_loss_base"]),
                   daily_loss_reference=str(r["daily_loss_reference"]),
                   personal_daily_stop_pct=float(k.get("personal_daily_stop_pct", 3.0)),
                   min_trade_days=int(r.get("min_trading_days", 5)))


PATHS = ("raw", "norm")
_PATH_FIELDS = ("end_equity", "min_equity", "total_breach", "daily_breach",
                "daily_breach_days", "personal_breach", "personal_breach_days")
WINDOW_COLUMNS = [
    "start", "last_day", "n_marks", "no_data", "missing_weekdays", "partial_data",
    "carried_in", "carry_approx", "trade_days", "bt_balance0", "scale",
    # verdict = union of the raw and norm paths
    "total_breach", "daily_breach", "daily_breach_days", "personal_breach",
    "personal_breach_days", "min_trade_days_met",
] + [f"{f}_{p}" for p in PATHS for f in _PATH_FIELDS]


@dataclass(frozen=True)
class ReplayResult:
    contract: str
    rules: ReplayRules
    windows: pd.DataFrame          # one row per window, WINDOW_COLUMNS

    @property
    def n_windows(self) -> int:
        """All Monday windows of the span, data or not (reported)."""
        return len(self.windows)

    @property
    def _data(self) -> pd.DataFrame:
        return self.windows[~self.windows["no_data"].astype(bool)]

    @property
    def n_data_windows(self) -> int:
        """Windows with at least one mark: the 10.5.9 denominator."""
        return len(self._data)

    @property
    def n_no_data_windows(self) -> int:
        return self.n_windows - self.n_data_windows

    @property
    def n_partial_data_windows(self) -> int:
        return int(self.windows["partial_data"].sum())

    @staticmethod
    def _mean(df: pd.DataFrame, col: str) -> float:
        return float(df[col].mean()) if len(df) else float("nan")

    def _share(self, col: str) -> float:
        return self._mean(self._data, col)

    @property
    def n_total_breach(self) -> int:
        return int(self.windows["total_breach"].sum())

    @property
    def n_daily_breach(self) -> int:
        return int(self.windows["daily_breach"].sum())

    def n_breach(self, kind: str, path: str) -> int:
        """Windows with a `kind` ('total', 'daily', 'personal') breach on one path ('raw' or
        'norm'); reported beside the union verdict."""
        return int(self.windows[f"{kind}_breach_{path}"].sum())

    def breach_share(self, kind: str, path: str) -> float:
        """Share of windows WITH DATA with a `kind` breach on one path (reported)."""
        return self._share(f"{kind}_breach_{path}")

    @property
    def min_bt_balance0(self) -> float:
        """Lowest backtest balance at a window start: how far the raw path is scaled down."""
        return float(self.windows["bt_balance0"].min()) if len(self.windows) else float("nan")

    @property
    def daily_breach_share(self) -> float:
        """Share of windows WITH DATA with a daily-loss breach (the 10.5.9 figure)."""
        return self._share("daily_breach")

    @property
    def daily_breach_share_all(self) -> float:
        """Same over ALL windows, data holes included (reported, weaker, not the verdict)."""
        return self._mean(self.windows, "daily_breach")

    @property
    def daily_breach_share_full(self) -> float:
        """Same over windows with data and no missing weekday (reported, stricter)."""
        d = self._data
        return self._mean(d[~d["partial_data"].astype(bool)], "daily_breach")

    @property
    def n_personal_breach(self) -> int:
        return int(self.windows["personal_breach"].sum())

    @property
    def personal_breach_share(self) -> float:
        return self._share("personal_breach")

    @property
    def n_min_trade_days_met(self) -> int:
        return int(self.windows["min_trade_days_met"].sum())

    @property
    def min_trade_days_share(self) -> float:
        """Share of windows with data that have >= min_trade_days trade days (reported)."""
        return self._share("min_trade_days_met")

    @property
    def passes(self) -> bool:
        """COMMON 10.5.9: no total-loss breach and daily-loss breaches in <= 1% of the windows
        with data. No window with data is not a pass."""
        return (self.n_data_windows > 0 and self.n_total_breach == 0
                and self.daily_breach_share <= DAILY_BREACH_MAX_SHARE)

    def summary(self) -> dict:
        return {"contract": self.contract, "windows": self.n_windows,
                "data_windows": self.n_data_windows,
                "no_data_windows": self.n_no_data_windows,
                "partial_data_windows": self.n_partial_data_windows,
                "carry_approx_windows": int(self.windows["carry_approx"].sum()),
                "min_bt_balance0": self.min_bt_balance0,
                **{f"{kind}_loss_breaches_{p}": self.n_breach(kind, p)
                   for kind in ("total", "daily") for p in PATHS},
                **{f"daily_loss_breach_share_{p}": self.breach_share("daily", p)
                   for p in PATHS},
                "total_loss_breaches": self.n_total_breach,
                "daily_loss_breaches": self.n_daily_breach,
                "daily_loss_breach_share": self.daily_breach_share,
                "daily_loss_breach_share_all_windows": self.daily_breach_share_all,
                "daily_loss_breach_share_full_windows": self.daily_breach_share_full,
                "personal_stop_breaches": self.n_personal_breach,
                "personal_stop_breach_share": self.personal_breach_share,
                "share_windows_min_trade_days": self.min_trade_days_share,
                "passes": self.passes}

    def to_report_stats(self, usd_per_point: Mapping[str, float] | None = None):
        """validation.report.ReplayStats for this replay. Its n_windows is the 10.5.9
        denominator (windows WITH DATA), so report.py's shares match `passes`; windows in a
        data hole carry no breach, no trade day and are left out of every count. Breach
        counts are the raw / norm union (the verdict columns)."""
        from validation.report import ReplayStats   # lazy: report never imports replay
        d = self._data
        return ReplayStats(contract=self.contract, usd_per_point=usd_per_point,
                           n_windows=len(d),
                           total_loss_breach_windows=int(d["total_breach"].sum()),
                           daily_loss_breach_windows=int(d["daily_breach"].sum()),
                           personal_stop_breach_windows=int(d["personal_breach"].sum()),
                           windows_ge5_trade_days=int(d["min_trade_days_met"].sum()))


def window_starts(span_start: date, span_end: date, window_days: int = WINDOW_DAYS) -> list[date]:
    """Mondays m with span_start <= m and m + window_days - 1 <= span_end (server dates)."""
    first = span_start + timedelta(days=(7 - span_start.weekday()) % 7)
    out, m = [], first
    while m + timedelta(days=window_days - 1) <= span_end:
        out.append(m)
        m += timedelta(days=7)
    return out


def _ns(values) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    return idx.as_unit("ns").asi8 if len(idx) else np.array([], dtype="int64")


def _flag(values, n: int, name: str) -> np.ndarray:
    if values is None:
        return np.zeros(n, dtype=bool)
    a = np.asarray(values, dtype=bool)
    if a.shape != (n,):
        raise ValueError(f"{name} must have one entry per trade")
    return a


def competition_replay(mark_ns, balance, equity, fill_ns, exit_ns, span_start: date,
                       span_end: date, clock: ServerClock, rules: ReplayRules, *,
                       contract: str = "demo", start_balance: float | None = None,
                       fill_at_open=None, exit_at_open=None) -> ReplayResult:
    """Replay every 30-day window of [span_start, span_end] (server dates).

    mark_ns, balance, equity: the backtest's MTM path (UTC int64 ns bar closes, strictly
        increasing; balance and worst-intrabar equity at each mark).
    fill_ns, exit_ns: fill and exit times (UTC ns) of the FILLED trades of that run.
    fill_at_open, exit_at_open: per trade, True when that time is a bar OPEN (known 1 ns later,
        backtest.fills.seen_ns); default all False (close-stamped).
    start_balance: the backtest's own starting balance (value before the first mark);
        defaults to rules.initial_balance.
    Raises ValueError when no mark falls in the span (wrong span or wrong time unit).
    """
    t = np.asarray(mark_ns, dtype="int64")
    bal = np.asarray(balance, dtype="float64")
    eq = np.asarray(equity, dtype="float64")
    if not (len(t) == len(bal) == len(eq)):
        raise ValueError("mark_ns, balance and equity differ in length")
    if len(t) and np.any(np.diff(t) <= 0):
        raise ValueError("mark_ns must be strictly increasing")
    if not (np.all(np.isfinite(bal)) and np.all(np.isfinite(eq))):
        raise ValueError("balance / equity contain NaN or inf")
    fills = np.asarray(fill_ns, dtype="int64")
    exits = np.asarray(exit_ns, dtype="int64")
    if len(fills) != len(exits):
        raise ValueError("fill_ns and exit_ns differ in length")
    fill_seen = fills + AFTER_OPEN_NS * _flag(fill_at_open, len(fills), "fill_at_open")
    exit_seen = exits + AFTER_OPEN_NS * _flag(exit_at_open, len(exits), "exit_at_open")
    span_lo = clock.midnight_utc_ns(span_start)
    span_hi = clock.midnight_utc_ns(span_end + timedelta(days=1))
    if not np.any((t > span_lo) & (t <= span_hi)):
        raise ValueError(f"no mark falls in the span {span_start}..{span_end} (wrong span, or "
                         "mark times not in UTC ns?)")
    init = float(rules.initial_balance)
    bt_start = init if start_balance is None else float(start_balance)
    nd = rules.window_days

    mark_day = (clock.dates(t - 1).astype("int64") if len(t)
                else np.array([], dtype="int64"))
    fill_day = (clock.dates(fills).astype("int64") if len(fills)
                else np.array([], dtype="int64"))
    total_floor = init * (1 - rules.max_loss_pct / 100)
    days_k = np.arange(nd)

    rows = []
    for m in window_starts(span_start, span_end, nd):
        t0 = clock.midnight_utc_ns(m)
        t_end = clock.midnight_utc_ns(m + timedelta(days=nd))
        i0 = int(np.searchsorted(t, t0, side="right"))
        i1 = int(np.searchsorted(t, t_end, side="right"))
        bal0, eq0 = (bal[i0 - 1], eq[i0 - 1]) if i0 else (bt_start, bt_start)
        u0 = float(eq0 - bal0)

        # Positions whose MTM is in the equity at mark0 (account.series convention).
        mark0 = t[i0 - 1] if i0 else t0
        carried = (fill_seen <= mark0) & (exit_seen > mark0)
        n_carried = int(carried.sum())
        if abs(u0) > 1e-6 * max(1.0, abs(eq0)) and n_carried == 0:
            raise ValueError(f"window {m}: equity differs from balance at the start but no "
                             "filled trade is open")

        tw = t[i0:i1]
        e_w = init + (eq[i0:i1] - eq0)
        b_w = init + (bal[i0:i1] - bal0)
        carry_approx = False
        if n_carried:
            seen_c = exit_seen[carried]
            t_carry = int(seen_c.max())
            carry_approx = bool(np.any((tw >= seen_c.min()) & (tw < t_carry)))
            b_w = b_w - u0 * (tw >= t_carry)
        k = mark_day[i0:i1] - np.datetime64(m, "D").astype("int64")
        lo = np.searchsorted(k, days_k, side="left")
        hi = np.searchsorted(k, days_k, side="right")
        has = hi > lo
        if bal0 <= 0:
            raise ValueError(f"window {m}: backtest balance {bal0} <= 0 at the window start")
        scale = init / float(bal0)

        row = {"start": m, "last_day": m + timedelta(days=nd - 1)}
        hits = {}
        for path, s in (("raw", 1.0), ("norm", scale)):
            # Both paths share the rebasing; norm scales every $ increment by s.
            pe, pb = init + s * (e_w - init), init + s * (b_w - init)
            out, d_hit, p_hit = _path_checks(pe, pb, lo, hi, has, init, total_floor, rules)
            row.update({f"{f}_{path}": v for f, v in out.items()})
            hits[path] = (out["total_breach"], d_hit, p_hit)
        daily_hit = hits["raw"][1] | hits["norm"][1]
        personal_hit = hits["raw"][2] | hits["norm"][2]

        weekday = np.array([(m + timedelta(days=int(j))).weekday() < 5 for j in days_k])
        missing_wd = int((weekday & ~has).sum())
        n_marks = int(i1 - i0)
        in_w = (fills >= t0) & (fills < t_end)
        trade_days = int(len(np.unique(fill_day[in_w])))
        row.update({
            "n_marks": n_marks,
            "no_data": n_marks == 0, "missing_weekdays": missing_wd,
            "partial_data": n_marks > 0 and missing_wd > 0,
            "carried_in": n_carried, "carry_approx": carry_approx, "trade_days": trade_days,
            "bt_balance0": float(bal0), "scale": scale,
            "total_breach": bool(hits["raw"][0] or hits["norm"][0]),
            "daily_breach": bool(daily_hit.any()), "daily_breach_days": int(daily_hit.sum()),
            "personal_breach": bool(personal_hit.any()),
            "personal_breach_days": int(personal_hit.sum()),
            "min_trade_days_met": trade_days >= rules.min_trade_days,
        })
        rows.append(row)
    windows = pd.DataFrame(rows, columns=WINDOW_COLUMNS)
    return ReplayResult(contract, rules, windows)


def _path_checks(e_w: np.ndarray, b_w: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                 has: np.ndarray, init: float, total_floor: float, rules: ReplayRules):
    """Loss-floor checks of one window path. lo/hi index the window marks of each window day
    (has: the day has a mark). Returns (fields, daily_hit per day, personal_hit per day)."""
    nd = len(lo)
    # Start-of-day values: last window mark of an earlier day, else the window start.
    prev = lo - 1
    sod_b = np.where(prev >= 0, b_w[np.maximum(prev, 0)] if len(b_w) else init, init)
    sod_e = np.where(prev >= 0, e_w[np.maximum(prev, 0)] if len(e_w) else init, init)
    day_min = np.full(nd, np.inf)
    for j in np.flatnonzero(has):
        day_min[j] = e_w[lo[j]:hi[j]].min()
    base = {"initial_balance": np.full(nd, init), "start_of_day_balance": sod_b,
            "start_of_day_equity": sod_e}[rules.daily_loss_base]
    ref = {"start_of_day_balance": sod_b,
           "start_of_day_equity": sod_e}[rules.daily_loss_reference]
    daily_hit = day_min <= ref - base * rules.daily_loss_pct / 100
    personal_hit = day_min <= sod_b * (1 - rules.personal_daily_stop_pct / 100)
    fields = {
        "end_equity": float(e_w[-1]) if len(e_w) else init,
        "min_equity": float(min(init, e_w.min())) if len(e_w) else init,
        "total_breach": bool(len(e_w) and e_w.min() <= total_floor),
        "daily_breach": bool(daily_hit.any()), "daily_breach_days": int(daily_hit.sum()),
        "personal_breach": bool(personal_hit.any()),
        "personal_breach_days": int(personal_hit.sum()),
    }
    return fields, daily_hit, personal_hit


def _exit_at_open(filled: pd.DataFrame) -> np.ndarray:
    """Open-stamped exits: an explicit exit_at_open column if the trades frame has one, else
    inferred from exit_note (OPEN_EXIT_NOTES)."""
    if "exit_at_open" in filled.columns:
        return filled["exit_at_open"].eq(True).to_numpy()
    if "exit_note" not in filled.columns:
        raise ValueError("trades frame has neither exit_at_open nor exit_note")
    return filled["exit_note"].isin(OPEN_EXIT_NOTES).to_numpy()


def replay_backtest(result, span_start: date, span_end: date, cfg: dict,
                    rules: ReplayRules | None = None) -> ReplayResult:
    """Replay one backtest.results.BacktestResult (one scenario at one contract) over the test
    span. The competition contract is replayed by passing that contract's result."""
    rules = rules or ReplayRules.from_cfg(cfg)
    eq = result.equity
    filled = result.filled
    fill_open = (filled["fill_at_open"].eq(True).to_numpy()
                 if "fill_at_open" in filled.columns else None)
    return competition_replay(
        _ns(eq["time_utc"]), eq["balance"].to_numpy(), eq["equity"].to_numpy(),
        _ns(filled["fill_time"]), _ns(filled["exit_time"]), span_start, span_end,
        ServerClock.from_cfg(cfg), rules, contract=result.contract,
        start_balance=float(cfg["rules"]["initial_balance"]),
        fill_at_open=fill_open, exit_at_open=_exit_at_open(filled))
