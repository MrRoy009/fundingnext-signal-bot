"""Simulated competition account for the backtester (COMMON 8.6-8.9).

Positions carry their whole path (fill, exit, per-bar worst-price marks, commission, swap), so
the account can answer "equity at time t" for any t up to the current event time without
looking ahead: only bars that closed by t and costs charged by t enter the answer.

Equity is marked to market at each M15 bar close at the worst intrabar price (bid low for a
long, bid high + eff_spread for a short, COMMON 8.7). Balance changes when a trade closes, by
its full $P&L (COMMON 8.9), and compounds from config rules.initial_balance.
The account day starts at 00:00 server time (Europe/Athens, owner decision O8).

Every account answer uses the time an event becomes KNOWN (backtest.fills.seen_ns): a fill or
exit at the open of the bar opening at t is seen just after t, so neither a decision at T = t
nor the MTM mark at the bar close t depends on the price of a bar that has not closed by then.
Reported fill / exit times stay the real times.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from backtest.costs import SWAP_INTEREST_OPEN, swap_charge
from backtest.fills import Path, worst_marks
from backtest.market import IndexBars, ServerClock
from backtest.scenarios import Scenario
from data.mt5_adapter import SymbolSpec
from risk.state import AccountState, OpenPosition, PendingOrder


@dataclass
class Position:
    """A filled trade with everything the account needs, in USD at the run's contract."""
    entry_id: int
    key: str
    side: str
    lots: float
    usd_per_point: float
    fill_ns: int                         # real times (swap, reporting)
    exit_ns: int
    fill_seen_ns: int                    # when the account learns of the fill / exit
    exit_seen_ns: int
    fill_price: float
    exit_price: float
    planned_risk_usd: float
    commission_side_usd: float           # one side, already x lots
    mark_close_ns: np.ndarray            # own bar closes in [fill, exit seen)
    mark_upnl: np.ndarray                # unrealised $ at those closes (worst intrabar)
    swaps: list[tuple[int, date, float]] = field(default_factory=list)  # (00:00 ns, day, $)

    @property
    def sign(self) -> float:
        return 1.0 if self.side == "BUY" else -1.0

    @property
    def gross_pnl_usd(self) -> float:
        return self.lots * (self.exit_price - self.fill_price) * self.sign * self.usd_per_point

    @property
    def commission_usd(self) -> float:
        return 2.0 * self.commission_side_usd

    @property
    def swap_usd(self) -> float:
        return float(sum(s[2] for s in self.swaps))

    @property
    def pnl_usd(self) -> float:
        """COMMON 8.9: lots * (exit - fill) * side * USD/pt - commission - swap."""
        return self.gross_pnl_usd - self.commission_usd - self.swap_usd

    def mtm(self, t: np.ndarray) -> np.ndarray:
        """Unrealised $ net of costs charged so far, at instants t (fill <= t < exit on seen
        times, assumed by the caller): last own worst-price mark closed by t (0 before the
        first), minus the entry commission, minus every swap charged at a rollover < t (the
        mark closing exactly at 00:00 server time still belongs to the old day)."""
        t = np.asarray(t, dtype="int64")
        k = np.searchsorted(self.mark_close_ns, t, side="right") - 1
        up = np.where(k >= 0, self.mark_upnl[np.maximum(k, 0)] if len(self.mark_upnl)
                      else 0.0, 0.0)
        out = up - self.commission_side_usd
        for m_ns, _, amount in self.swaps:
            # charged at the rollover: the mark closing exactly at 00:00 ends the old day
            out = out - np.where(t > m_ns, amount, 0.0)
        return out


def build_position(entry_id: int, key: str, side: str, lots: float, path: Path,
                   b: IndexBars, scenario: Scenario, usd_per_point: float,
                   planned_risk_usd: float, commission_per_side: float,
                   spec: SymbolSpec, clock: ServerClock,
                   margin_ccy_usd: float | None = None) -> Position:
    """margin_ccy_usd: config costs.margin_ccy_usd[key] (swap mode 3, COMMON A4.1)."""
    buy = side == "BUY"
    sign = 1.0 if buy else -1.0
    # Marks: own bar closes before the exit is seen. An exit at the open of the bar opening at
    # t keeps the mark of the bar closing at t (the position was still held then).
    k_end = int(np.searchsorted(b.close_ns, path.exit_seen_ns, side="left"))
    lo = path.fill_bar
    hi = max(lo, k_end)
    marks = worst_marks(b, buy, scenario.spread_mult, lo, hi)
    close_ns = b.close_ns[lo:hi]
    keep = close_ns >= path.fill_ns
    upnl = lots * (marks[keep] - path.fill_price) * sign * usd_per_point
    pos = Position(entry_id, key, side, lots, usd_per_point, path.fill_ns, path.exit_ns,
                   path.fill_seen_ns, path.exit_seen_ns, path.fill_price, path.exit_price,
                   planned_risk_usd,
                   commission_per_side * lots, close_ns[keep].copy(), upnl.astype("float64"))
    pos.swaps = rollover_swaps(pos, b, spec, scenario, clock, margin_ccy_usd)
    return pos


def rollover_swaps(pos: Position, b: IndexBars, spec: SymbolSpec, scenario: Scenario,
                   clock: ServerClock, margin_ccy_usd: float | None = None
                   ) -> list[tuple[int, date, float]]:
    """Swap charged at each 00:00 server-time rollover the position is held through
    (fill < 00:00 < exit), COMMON 8.6.

    SPEC NOTE: a rollover is charged for the server day it ends, and only when that day is
    Monday to Friday; the weekend is covered by the triple day (swap_rollover3days), as in MT5.
    Without this a Friday-to-Monday hold would pay the weekend twice. Interest modes use the
    bid close of the last bar closed by the rollover (current) or the fill price (open).
    """
    first, last = clock.date_of(pos.fill_ns), clock.date_of(pos.exit_ns)
    out = []
    d = first + timedelta(days=1)
    while d <= last:
        m_ns = clock.midnight_utc_ns(d)
        day = d - timedelta(days=1)
        if pos.fill_ns < m_ns < pos.exit_ns and day.weekday() < 5:
            if spec.swap_mode == SWAP_INTEREST_OPEN:
                price = pos.fill_price
            else:
                k = int(np.searchsorted(b.close_ns, m_ns, side="right")) - 1
                price = float(b.close[max(k, 0)])
            cost = swap_charge(spec, pos.side, pos.lots, pos.usd_per_point, price, day, scenario,
                               margin_ccy_usd)
            out.append((m_ns, day, cost))
        d += timedelta(days=1)
    return out


@dataclass
class BookEntry:
    """An accepted ticket: pending (placed or waiting for placement) or open."""
    entry_id: int
    strategy_id: str
    key: str
    side: str
    lots: float
    planned_risk_usd: float
    start_ns: int                   # T
    fill_seen_ns: int | None        # when the fill is known; None: never fills
    end_seen_ns: int                # when the exit, skip or cancel is known

    def is_open(self, t: int) -> bool:
        return self.fill_seen_ns is not None and self.fill_seen_ns <= t


class Account:
    """Balance, MTM equity and the AccountState that risk.rules.check_ticket sees."""

    def __init__(self, initial_balance: float, clock: ServerClock):
        self.initial = float(initial_balance)
        self.clock = clock
        self._bal_t: list[int] = []
        self._bal_v: list[float] = []
        self.positions: list[Position] = []          # every filled trade, fill order
        self._live: list[Position] = []              # not yet pruned (exit after day start)
        self.active: dict[int, BookEntry] = {}
        self._fill_days: list[date] = []
        self._fill_ns: list[int] = []
        self.day: date | None = None
        self.day_start_ns: int | None = None
        self.sod_balance = self.initial
        self.sod_equity = self.initial

    # --- balance and equity -------------------------------------------------------------
    @property
    def balance(self) -> float:
        return self._bal_v[-1] if self._bal_v else self.initial

    def balance_at(self, t: int) -> float:
        i = bisect.bisect_right(self._bal_t, t)
        return self._bal_v[i - 1] if i else self.initial

    def equity_at(self, t: int) -> float:
        eq = self.balance_at(t)
        for p in self._live:
            if p.fill_seen_ns <= t < p.exit_seen_ns:
                eq += float(p.mtm(np.array([t]))[0])
        return eq

    # --- events ---------------------------------------------------------------------------
    def add_position(self, pos: Position) -> None:
        self.positions.append(pos)
        self._live.append(pos)

    def on_fill(self, pos: Position) -> None:
        self._fill_ns.append(pos.fill_seen_ns)
        self._fill_days.append(self.clock.date_of(pos.fill_ns))

    def on_exit(self, pos: Position) -> None:
        self._bal_t.append(pos.exit_seen_ns)
        self._bal_v.append(self.balance + pos.pnl_usd)

    def roll_day(self, t: int) -> None:
        """Start-of-day balance and equity at 00:00 server time of T's server day."""
        day = self.clock.date_of(t)
        if day == self.day:
            return
        self.day = day
        self.day_start_ns = self.clock.midnight_utc_ns(day)
        self.sod_balance = self.balance_at(self.day_start_ns)
        self.sod_equity = self.equity_at(self.day_start_ns)
        self._live = [p for p in self._live if p.exit_seen_ns > self.day_start_ns]

    # --- AccountState for check_ticket ----------------------------------------------------
    def state(self, t: int, updated_at) -> AccountState:
        """The simulated account at decision time t (all events <= t processed).
        SPEC NOTE: trades_today = fills seen in [00:00 server, t]; accepted tickets that have not
        filled yet count as pending orders (risk only, as risk.rules does for pending orders);
        open-position and pending risk is the planned risk $ at the run's contract."""
        self.roll_day(t)
        lo = bisect.bisect_left(self._fill_ns, self.day_start_ns)
        hi = bisect.bisect_right(self._fill_ns, t)
        trades_today = max(0, hi - lo)
        completed = len({d for d in self._fill_days if d < self.day})
        opens, pend = [], []
        for e in self.active.values():
            if e.is_open(t):
                opens.append(OpenPosition(e.key, e.side, e.lots, e.planned_risk_usd))
            else:
                pend.append(PendingOrder(e.key, e.side, e.lots, e.planned_risk_usd, None))
        return AccountState(updated_at=updated_at, balance=self.balance_at(t),
                            equity=self.equity_at(t), start_of_day_balance=self.sod_balance,
                            start_of_day_equity=self.sod_equity, trades_today=trades_today,
                            trading_days_completed=completed, open_positions=tuple(opens),
                            pending_orders=tuple(pend))

    # --- series ---------------------------------------------------------------------------
    def series(self, grid_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(balance, equity) at every grid instant (bar closes of all indices)."""
        grid_ns = np.asarray(grid_ns, dtype="int64")
        if self._bal_t:
            i = np.searchsorted(np.asarray(self._bal_t, dtype="int64"), grid_ns, side="right")
            vals = np.concatenate([[self.initial], np.asarray(self._bal_v, dtype="float64")])
            bal = vals[i]
        else:
            bal = np.full(len(grid_ns), self.initial)
        eq = bal.copy()
        for p in self.positions:
            a = int(np.searchsorted(grid_ns, p.fill_seen_ns, side="left"))
            z = int(np.searchsorted(grid_ns, p.exit_seen_ns, side="left"))
            if z > a:
                eq[a:z] += p.mtm(grid_ns[a:z])
        return bal, eq
