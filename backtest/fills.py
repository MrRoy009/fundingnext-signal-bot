"""Placement, entry fills and exits of one candidate on bid bars (COMMON 8.1, 8.3-8.5).

The path of a trade (where and when it is placed, filled and closed) does not depend on the
account: lots only scale the dollars. So it is computed here per candidate, from bars at or
after the placement time P only, and the account simulation (backtest.engine) decides whether
the candidate is traded at all and with how many lots.

Prices: bars are bid; ask = bid + eff_spread with eff_spread = c * max(spread_pts, floor_{i,h})
(COMMON 8.1). Buys fill on the ask, sells on the bid; triggers use the same side.

Event times (used to order events across indices; never to look ahead):
- a fill or exit at a bar's open happens at that open time;
- an intrabar fill or exit (limit, stop, SL, TP) happens at the bar's close;
- a close-price exit (time-exit fallback, data_gap_exit, end_of_data) at the bar's close.

When an event becomes KNOWN to the account ("seen" time): a close-stamped event at time t is
decided by a bar that closed at t, so a decision at T = t may see it. An open-stamped event at
time t (market fill at P, protocol skip at P, time exit at the clock-bar open, a stop or SL
filled on a gap at the open, an exit deferred to the open after the daily break) depends on the
open price of the bar that OPENS at t, which has not closed by t (COMMON 2: the forming bar is
never used). Such an event is seen at t + 1 ns: strictly after every decision at t and before
any later bar boundary. Reported times stay the real times; only the account uses seen times.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backtest.costs import slippage
from backtest.market import IndexBars
from backtest.scenarios import Scenario
from strategies.base import MARKET_BAND, MARKET_OR_LIMIT, STOP_ENTRY, Candidate

# COMMON 8.4 / 8.5: limit fills and TP exits need 0.01 of penetration. Stated literally in the
# spec (one tick at 2 digits), not in config.
PENETRATION = 0.01

FILLED, SKIPPED, CANCELLED = "filled", "skipped", "cancelled"
MARKET, LIMIT, STOP = "market", "limit", "stop"
EXIT_SL, EXIT_TP = "SL", "TP"
EXIT_TIME, EXIT_GAP, EXIT_END = "time_exit", "data_gap_exit", "end_of_data"
EXIT_TYPES = (EXIT_SL, EXIT_TP, EXIT_TIME, EXIT_GAP, EXIT_END)

# Phase of an event inside bar j: at its open, intrabar, at its close. Ties between exit
# events on the same (bar, phase) go to the lower priority number.
_OPEN, _INTRA, _CLOSE = 0, 1, 2
_PRIO = {"deferred": 0, EXIT_SL: 1, EXIT_TP: 2, EXIT_TIME: 3, EXIT_GAP: 4, EXIT_END: 5}

_FIRST_CHUNK = 256

# An open-stamped event is seen 1 ns after its time (see the module docstring). Every decision
# and every MTM instant is on a bar boundary, so 1 ns is "strictly after t, before t + 15 min".
# SPEC NOTE: COMMON 2 forbids using the forming bar at T; read literally for the account, any
# event decided by the bar opening at T (including a skip at P, and "placement beyond the
# data" at T) is unknown to decisions and MTM marks at T. Cancels at EXPIRY are decided by bars
# closed by EXPIRY and are seen at EXPIRY.
AFTER_OPEN_NS = 1


def seen_ns(t_ns: int, at_open: bool) -> int:
    """Time at which an event at t_ns becomes known to the account."""
    return int(t_ns) + (AFTER_OPEN_NS if at_open else 0)


@dataclass(frozen=True)
class Path:
    """Account-independent outcome of one candidate. Times are UTC int64 ns."""
    status: str                      # filled | skipped | cancelled
    reason: str = ""
    placement_ns: int | None = None  # P
    entry_mode: str = ""             # market | limit | stop
    resolve_ns: int | None = None    # skip / cancel time; exit time for a filled trade
    fill_ns: int | None = None
    fill_bar: int | None = None
    fill_price: float | None = None
    fill_at_open: bool = False
    exit_ns: int | None = None
    exit_bar: int | None = None
    exit_price: float | None = None
    exit_type: str = ""
    exit_reason: str = ""
    exit_at_open: bool = False       # exit priced at a bar open (open-stamped)
    resolve_at_open: bool = False    # resolve_ns is open-stamped (skips at P, exits at an open)

    @property
    def resolve_seen_ns(self) -> int | None:
        return None if self.resolve_ns is None else seen_ns(self.resolve_ns,
                                                            self.resolve_at_open)

    @property
    def fill_seen_ns(self) -> int | None:
        return None if self.fill_ns is None else seen_ns(self.fill_ns, self.fill_at_open)

    @property
    def exit_seen_ns(self) -> int | None:
        return None if self.exit_ns is None else seen_ns(self.exit_ns, self.exit_at_open)


def _first_true(fn, start: int, stop: int) -> int | None:
    """First index j in [start, stop) with fn(lo, hi)[j - lo] True, scanning in growing chunks
    so a trade that ends in a few bars never touches the rest of a 100k-bar history."""
    lo, size = start, _FIRST_CHUNK
    while lo < stop:
        hi = min(stop, lo + size)
        m = fn(lo, hi)
        if m.any():
            return lo + int(np.argmax(m))
        lo, size = hi, size * 4
    return None


def _segment_end(b: IndexBars, j: int) -> int:
    """First index after j in a different segment (or n)."""
    s = b.seg[j]
    k = _first_true(lambda lo, hi: b.seg[lo:hi] != s, j + 1, b.n)
    return b.n if k is None else k


class _Ctx:
    """Per-candidate constants for the fill model."""

    def __init__(self, c: Candidate, b: IndexBars, sc: Scenario, cfg: dict):
        self.c, self.b, self.sc, self.cfg = c, b, sc, cfg
        self.buy = c.side == "BUY"
        self.cmult = sc.spread_mult
        self.xmult = sc.exit_spread_mult

    def slip(self, j: int) -> float:
        return slippage(self.sc, self.cfg, float(self.b.floor[j]), float(self.c.atr))

    def spr(self, j: int, mult: float | None = None) -> float:
        return (self.cmult if mult is None else mult) * float(self.b.spread_base[j])


def _ns(ts) -> int:
    return int(ts.value) if hasattr(ts, "value") else int(ts)


def simulate(c: Candidate, b: IndexBars, sc: Scenario, cfg: dict, *, valid_until_ns: int,
             stops_buffer: float | None) -> Path:
    """Placement (8.3), entry (8.4) and exit (8.5) of one prepared candidate.

    `valid_until_ns` = T + valid_for_bars x 15 min (the market-ticket validity, ER-02);
    `stops_buffer` = trade_stops_level x point (STOP_ENTRY only).
    """
    x = _Ctx(c, b, sc, cfg)
    t_ns = _ns(c.signal_time)
    expiry_ns = _ns(c.expires_utc) if c.expires_utc is not None else None

    # --- 8.3 placement -----------------------------------------------------------------
    k_sig = int(np.searchsorted(b.open_ns, t_ns, side="left")) - 1   # last bar closed by T
    p = int(np.searchsorted(b.open_ns, t_ns + sc.d * b.tf_ns, side="left"))
    if p >= b.n:
        return Path(SKIPPED, "placement beyond the data", resolve_ns=t_ns,
                    resolve_at_open=True)
    p_ns = int(b.open_ns[p])
    if k_sig < 0:
        return Path(SKIPPED, "no bar closed by T", placement_ns=p_ns, resolve_ns=p_ns,
                    resolve_at_open=True)
    if b.seg[p] != b.seg[k_sig]:
        return Path(SKIPPED, "placement past a segment break", placement_ns=p_ns,
                    resolve_ns=p_ns, resolve_at_open=True)
    # SPEC NOTE: COMMON 8.3 skips "P after VALID UNTIL" (= min(T + 45 min, EXPIRY)); the
    # strategy specs add "no order is placed at or after EXPIRY" (SES-IBX-01 4, MR-TWAP-01 4),
    # so P = T + 45 min is allowed and P = EXPIRY is not.
    if p_ns > valid_until_ns or (expiry_ns is not None and p_ns >= expiry_ns):
        return Path(SKIPPED, "placement after VALID UNTIL", placement_ns=p_ns, resolve_ns=p_ns,
                    resolve_at_open=True)
    if b.in_break[p]:
        # SPEC NOTE: COMMON 8.5 allows no fills in the daily break and defers only EXITS to the
        # first bar after it; an order cannot be placed while the market is closed, so a
        # placement bar inside the break is a skip. No current strategy can reach this.
        return Path(SKIPPED, "placement in the daily break", placement_ns=p_ns,
                    resolve_ns=p_ns, resolve_at_open=True)

    # --- COMMON 7 protocol at P ------------------------------------------------------------
    bid = float(b.open[p])
    q = bid + x.spr(p) if x.buy else bid               # the owner's live quote
    mode, why = _protocol(c, q, x.buy, stops_buffer)
    if mode is None:
        return Path(SKIPPED, why, placement_ns=p_ns, resolve_ns=p_ns,
                    resolve_at_open=True)

    # --- 8.4 entry -----------------------------------------------------------------------
    if mode == MARKET:
        price = (bid + x.spr(p) + x.slip(p)) if x.buy else (bid - x.slip(p))
        f, fill_ns, at_open = p, p_ns, True
    else:
        found = _pending_fill(x, mode, p, expiry_ns)
        if isinstance(found, str):
            return Path(CANCELLED, found, placement_ns=p_ns, entry_mode=mode,
                        resolve_ns=expiry_ns if expiry_ns is not None else p_ns)
        f, price, at_open = found
        fill_ns = int(b.open_ns[f]) if at_open else int(b.open_ns[f] + b.tf_ns)

    # --- 8.5 exit ------------------------------------------------------------------------
    xb, x_ns, x_price, x_type, x_reason, x_open = _exit(x, f, price, at_open)
    return Path(FILLED, "", placement_ns=p_ns, entry_mode=mode, resolve_ns=x_ns,
                fill_ns=fill_ns, fill_bar=f, fill_price=price, fill_at_open=at_open,
                exit_ns=x_ns, exit_bar=xb, exit_price=x_price, exit_type=x_type,
                exit_reason=x_reason, exit_at_open=x_open, resolve_at_open=x_open)


def _protocol(c: Candidate, q: float, buy: bool,
              stops_buffer: float | None) -> tuple[str | None, str]:
    """COMMON 7 placement rule with quote Q: (mode, '') or (None, skip reason)."""
    sgn = 1.0 if buy else -1.0
    worse = lambda a, ref: sgn * (a - ref) > 0       # noqa: E731  a is worse than ref
    if c.order_type == MARKET_BAND:
        if worse(q, c.chase_limit):
            return None, "protocol: quote beyond CHASE LIMIT"
        if worse(c.skip_level, q):
            return None, "protocol: quote beyond SKIP LEVEL"
        if sgn * (q - c.take_profit) >= 0:
            return None, "protocol: quote at or beyond TP"
        return MARKET, ""
    if c.order_type == MARKET_OR_LIMIT:
        if worse(c.skip_level, q):
            return None, "protocol: quote beyond SKIP LEVEL"
        if sgn * (q - c.take_profit) >= 0:
            return None, "protocol: quote at or beyond TP"
        return (LIMIT, "") if worse(q, c.entry) else (MARKET, "")
    if c.order_type == STOP_ENTRY:
        if stops_buffer is None:
            raise ValueError(f"{c.key}: STOP_ENTRY needs the symbol's trade_stops_level "
                             f"(stops_buffer, COMMON 7)")
        # buy: ask < trigger - buffer; sell (mirrored with the bid): bid > trigger + buffer
        if sgn * (c.entry - q) > stops_buffer:
            return STOP, ""
        if not worse(q, c.chase_limit):                # ask <= CHASE LIMIT (buy)
            return MARKET, ""
        return None, "protocol: quote beyond CHASE LIMIT"
    raise ValueError(f"{c.key}: order_type {c.order_type!r} is not backtestable "
                     f"(COMMON 7 protocols only)")


def _pending_fill(x: _Ctx, mode: str, p: int,
                  expiry_ns: int | None) -> tuple[int, float, bool] | str:
    """First fill of a resting limit / stop order in bars [P, EXPIRY) (COMMON 8.4), or the
    cancel reason. Daily-break bars never fill (8.5). The order lives in P's segment only."""
    b, c = x.b, x.c
    if expiry_ns is None:
        raise ValueError(f"{c.key}: a resting {mode} order needs EXPIRY")
    e = _segment_end(b, p)
    stop = int(np.searchsorted(b.open_ns, expiry_ns, side="left"))
    stop = min(stop, e)
    level = c.entry
    cm = x.cmult

    def hit(lo: int, hi: int) -> np.ndarray:
        ok = ~b.in_break[lo:hi]
        if mode == LIMIT:
            m = (b.low[lo:hi] + cm * b.spread_base[lo:hi] <= level - PENETRATION) if x.buy \
                else (b.high[lo:hi] >= level + PENETRATION)
        else:
            m = (b.high[lo:hi] + cm * b.spread_base[lo:hi] >= level) if x.buy \
                else (b.low[lo:hi] <= level)
        return m & ok

    j = _first_true(hit, p, stop)
    if j is None:
        if stop >= b.n and (b.n == 0 or b.open_ns[-1] + b.tf_ns < expiry_ns):
            return f"unfilled {mode} order: data ends before EXPIRY"
        if stop == e and e < b.n and b.open_ns[e] < expiry_ns:
            return f"unfilled {mode} order: segment break before EXPIRY"
        return f"unfilled {mode} order cancelled at EXPIRY"
    if mode == LIMIT:
        # SPEC NOTE: a limit fills at E even when the bar opens through E (COMMON 8.4 "at E"),
        # so a limit fill never counts as "at the bar open" for the fill-bar TP rule.
        return j, level, False
    bid_open = float(b.open[j])
    if x.buy:
        ask_open = bid_open + x.spr(j)
        return j, max(level, ask_open) + x.slip(j), ask_open >= level
    return j, min(level, bid_open) - x.slip(j), bid_open <= level


def _exit(x: _Ctx, f: int, fill: float, at_open: bool
          ) -> tuple[int, int, float, str, str, bool]:
    """(exit bar, exit time ns, exit price, exit type, note, at the open) of a position filled
    on bar f.

    COMMON 8.5: SL / TP on bid-ask rules with gap fills; SL first on a bar that reaches both;
    on the fill bar SL counts and TP only if the fill was at the open; time exit at the open of
    the first bar >= the clock, or at the close of the last bar before it if that bar is
    missing; triggers on daily-break bars resolve at the open of the first bar after the break
    with the gap rule; segment break -> data_gap_exit at the last pre-gap close; end of data ->
    end_of_data at the last close.
    """
    b, c, sc = x.b, x.c, x.sc
    e = _segment_end(b, f)
    sl, tp = c.stop, c.take_profit
    cm, xm = x.cmult, x.xmult

    clock_ns = None
    if not sc.missed_exit and c.time_exit_utc is not None:
        clock_ns = _ns(c.time_exit_utc)

    # candidate events: (bar, phase, prio, type, note)
    events: list[tuple[int, int, int, str, str]] = []

    # time exit: first bar after the fill bar with open >= clock, inside the segment.
    # SPEC NOTE: the time exit is checked from the bar after the fill bar; a fill on a bar
    # opening at or after the clock (impossible for the current strategies) is closed at that
    # bar's close by the "missing bar" branch.
    # SPEC NOTE: 8.5 "if that bar is missing" is read as "if no bar opens EXACTLY at the
    # clock": the exit is then the close of the last bar before the clock, even when a later
    # bar exists in the segment (e.g. a half-day whose next bar is the following session). The
    # owner exits at the clock, never at a later open.
    # SPEC NOTE: when the clock falls in or after a segment-break gap, no time exit is
    # scheduled: the position is closed by the 8.5 segment-break rule (data_gap_exit at the
    # last pre-gap close, priced at c), the same as when the data ends before the clock
    # (end_of_data). Both "no bar at or after the clock in this segment" cases follow the
    # literal tag of 8.5.
    te_stop = e
    if clock_ns is not None:
        j = int(np.searchsorted(b.open_ns, clock_ns, side="left"))
        j = max(j, f + 1)
        if j < e:
            if b.open_ns[j] == clock_ns:
                events.append((j, _OPEN, _PRIO[EXIT_TIME], EXIT_TIME, "at the clock bar open"))
            else:
                events.append((j - 1, _CLOSE, _PRIO[EXIT_TIME], EXIT_TIME,
                               "clock bar missing: last close before it"))
            te_stop = j + 1

    # SL / TP: first bar in [f, te_stop) that reaches either.
    # SPEC NOTE: SES-IBX-01 Entry 6 summarises the fill-bar rule as "TP on the fill bar
    # ignored"; COMMON 8.5 adds "unless the fill was at the bar open". COMMON 8.5 is applied
    # (the strategy spec defers to "COMMON 8.4-8.5"), so a STOP_ENTRY gap fill at the open
    # (at_open True) does let TP count on the fill bar.
    def hits(lo: int, hi: int) -> np.ndarray:
        sl_m, tp_m = _sl_tp(b, lo, hi, x.buy, sl, tp, cm)
        if lo == f and not at_open:
            tp_m[0] = False                           # fill-bar rule: TP ignored
        return sl_m | tp_m

    h = _first_true(hits, f, min(te_stop, e))
    if h is not None:
        sl_m, tp_m = _sl_tp(b, h, h + 1, x.buy, sl, tp, cm)
        kind = EXIT_SL if sl_m[0] else EXIT_TP
        if not b.in_break[h]:
            events.append((h, _INTRA, _PRIO[kind], kind, ""))
        else:
            # SPEC NOTE: 8.5 "exits pending then resolve on the first bar after it with the
            # gap rule" is read as: a trigger seen in the break ALWAYS executes at the first
            # post-break open, priced by the 8.5 rule of its kind. A deferred TP fills at TP
            # even if that open is back on the wrong side of TP; a deferred SL fills at
            # SL -/+ slip (or the gapped open) even if price has recovered.
            r = _first_true(lambda lo, hi: ~b.in_break[lo:hi], h + 1, e)
            if r is not None:
                events.append((r, _OPEN, _PRIO["deferred"], kind,
                               "triggered in the daily break; resolved at the next open"))
            # else: the segment or data ends inside the break -> gap / end exit below

    if e < b.n:
        events.append((e - 1, _CLOSE, _PRIO[EXIT_GAP], EXIT_GAP, "segment break"))
    else:
        # SPEC NOTE: if the data ends before the time-exit clock, the close of the last bar is
        # tagged end_of_data (8.5 "end of pre-holdout data"), not a time exit.
        events.append((b.n - 1, _CLOSE, _PRIO[EXIT_END], EXIT_END, "end of data"))

    j, phase, _, kind, note = min(events)
    price = _exit_price(x, j, phase, kind, note)
    t_ns = int(b.open_ns[j]) if phase == _OPEN else int(b.open_ns[j] + b.tf_ns)
    return j, t_ns, price, kind, note, phase == _OPEN


def _sl_tp(b: IndexBars, lo: int, hi: int, buy: bool, sl: float, tp: float,
           cm: float) -> tuple[np.ndarray, np.ndarray]:
    if buy:
        return b.low[lo:hi] <= sl, b.high[lo:hi] >= tp + PENETRATION
    ask_hi = b.high[lo:hi] + cm * b.spread_base[lo:hi]
    ask_lo = b.low[lo:hi] + cm * b.spread_base[lo:hi]
    return ask_hi >= sl, ask_lo <= tp - PENETRATION


def _exit_price(x: _Ctx, j: int, phase: int, kind: str, note: str) -> float:
    b, c = x.b, x.c
    if kind == EXIT_TP:
        return float(c.take_profit)                   # TP exactly, gaps included, no slip
    slip = x.slip(j)
    if kind == EXIT_SL:
        # SPEC NOTE: the SL gap clause ("if the bar opens beyond SL") also applies on the fill
        # bar of an INTRABAR stop / limit fill, using that bar's open although it came before
        # the position existed (bars carry no intrabar order). Literal and conservative: e.g.
        # buy stop 40100, SL 39900, bar opens 39850 and rallies -> fill 40102, SL exit at
        # 39850 - slip. The alternative (SL -/+ slip when the fill was not at the open) needs
        # a spec decision.
        bid_open = float(b.open[j])
        if x.buy:
            return min(float(c.stop), bid_open) - slip
        # The gap test is a trigger (ask open at the scenario's c); EXIT-3X only widens the
        # price of a gap fill (SPEC NOTE at Scenario.exit_spread_mult).
        if bid_open + x.spr(j) >= float(c.stop):
            return bid_open + x.spr(j, x.xmult) + slip
        return float(c.stop) + slip
    # SPEC NOTE: COMMON 8.5 says "bid or ask, with slippage" for the segment-break close only;
    # end_of_data ("close at the last bar close") is priced the same way (bid / ask at c, with
    # slippage), the conservative reading. Time exits use the EXIT-3X multiplier.
    mult = x.xmult if kind == EXIT_TIME else x.cmult
    px = float(b.open[j]) if phase == _OPEN else float(b.close[j])
    return px - slip if x.buy else px + x.spr(j, mult) + slip


def worst_marks(b: IndexBars, buy: bool, cmult: float, lo: int, hi: int) -> np.ndarray:
    """Worst intrabar price of bars [lo, hi) for MTM (COMMON 8.7): bid low for a long, bid
    high + eff_spread for a short.
    SPEC NOTE: the fill bar is marked with its whole range, including any part before an
    intrabar fill (bars carry no intrabar order), which can only make equity lower."""
    if buy:
        return b.low[lo:hi]
    return b.high[lo:hi] + cmult * b.spread_base[lo:hi]
