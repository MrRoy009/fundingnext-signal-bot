"""RS-MOM-01: intraday relative-strength momentum ignition (M15, ACTIVE regime).

Spec: docs/specs/RS-MOM-01.md, on top of docs/specs/COMMON.md. Everything shared (signal windows,
G(T) per exchange group, data completeness, regime gate, ranking, lockout replay, live == batch)
is done by strategies.context.WindowedStrategy; this module supplies only what the spec adds:

    windows        10:30-13:30 New York (US30, NAS100), 10:00-15:00 Berlin (GER40), on T,
                   both ends inclusive, valid trading dates only            (frozen, module consts)
    rank score     RS_4h (COMMON 4.1, L = cfg features.rs4h_lookback_bars)
    regime gate    ACTIVE (COMMON 6.7: status OK and V_T >= cfg regime.gate_threshold)
    entry          LONG-SELECTED with RS_4h(t) >= theta and RS_4h(t-1) < theta; SHORT-SELECTED
                   with RS_4h(t) <= -theta and RS_4h(t-1) > -theta; t-1 = the index's own M15 bar
                   15 min earlier; NaN (or absent) RS_4h(t-1): no fire
    lockouts       no candidate while an earlier RS-MOM-01 candidate occupies the index
                   (COMMON 7.1 reference occupancy); at most 2 per index per side per local date
    levels         E = C_t (bid close); ATRH = ATR14_H1 of the H1 bar chosen by the COMMON 2
                   alignment rule; SL = E -/+ k ATRH (rounded 0.01 away from E);
                   TP = E +/- R k ATRH (rounded 0.01 toward E)
    protocol       MARKET_BAND (CHASE LIMIT / SKIP LEVEL / sizing entry via with_protocol_levels,
                   fractions from cfg protocols); no EXPIRY (market ticket; VALID UNTIL is
                   T + cfg execution.valid_for_bars bars, set by the engine)
    time exit      cfg time_exits (15:45 New York / 17:15 Berlin) on T's local date

Public API
    PARAMS                       ParamGrid: defaults theta 1.0, k 1.0, R 1.5; one-at-a-time
                                 grids; heatmap (k, R); 29 configurations
    US_WINDOW, GER40_WINDOW      LocalWindow constants (frozen)
    MAX_PER_INDEX_SIDE_DAY = 2   frozen per-day cap
    RsMom01(cfg, params=None, *, off_grid=False, stats=None)
        .generate(bars, regime_bars, now, context)   live (StrategyContext.features and
                                                     .regime_tables must be set)
        .scan(features, regime_tables, start, end, params=None, *, on_selection=None)
                                                     batch (on_selection: per-cycle Selection
                                                     sink for the COMMON 4.2.8 cycle log)
        .cycles(...)                                 per-T Selections for diagnostics
    STRATEGY_CLASS = RsMom01
    reason_line(...), round_away(...), round_toward(...)   helpers (exposed for tests)

SPEC NOTE (ATRH unavailable): the spec says "if unavailable, no candidate". COMMON 4.2.2 (which a
strategy spec may not change) makes a NaN strategy input caused by missing data an INCOMPLETE
member, so an unavailable ATRH on any member of G(T) (H1 alignment rule fails, or ATR14_H1 NaN
in its warm-up, or not > 0) gives no candidate for the whole group at T, logged
"incomplete group: <index>: ATR14_H1 unavailable: <cause>". ATRH is checked for every member,
not only the selected one.
SPEC NOTE (ATRH not > 0): a zero ATR14_H1 would put SL on E (D = 0); it is treated like NaN.
SPEC NOTE (D for the protocol levels): the spec writes D = k * ATRH; COMMON 7 defines D = |E - SL|.
with_protocol_levels uses |E - SL| with the ROUNDED SL. Rounding moves D by < 0.01 points, so
CHASE LIMIT differs from E +/- 0.20 k ATRH by at most 0.002 points and SKIP LEVEL from
E -/+ 0.50 k ATRH by at most 0.005 points; sizing covers the printed SL exactly.
SPEC NOTE (reason line, "crossed <theta>"): most literal reading: <theta> is the parameter value,
printed unsigned with 2 dp for both sides ("crossed 1.00" for a long and for a short); only
<+x.xx> (the score) carries a sign.
SPEC NOTE (output order): candidates at one T are returned in config instrument order (the
Strategy.scan contract), not side order, so live and batch risk checks see the same order.
SPEC NOTE (reason line, window set and role): "<window set>" is G(T) (COMMON 4.2.1 calls G(T)
the window set: the exchange group's members in their window on a valid trading date, e.g.
{US30, NAS100} even when only one passes the gate, but {NAS100} on a date whose US30 cash-open
bar is missing, since US30 is then outside G(T)); the role is "sole index" when |E(T)| = 1
(COMMON 4.2.5), otherwise "leader" (long) / "laggard" (short).
SPEC NOTE (ATR14_M15 on the candidate): "information only", so it is not a completeness input.
A finite RS_4h needs 17 consecutive bars in one segment, so ATR14_M15 (min_periods 14) is
always finite when a candidate fires.
SPEC NOTE (frozen constants not in config): the windows and the 2-per-day cap are frozen by the
spec but have no config key (config.yaml is not edited by the Engineer), so they are module
constants here. Every other constant (L, ATR period, protocol fractions, validity, time exits,
gate threshold) is read from cfg.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core.config import tf_minutes
from strategies.base import (MARKET_BAND, Candidate, ParamGrid, time_exit_label,
                             with_protocol_levels)
from strategies.context import CycleView, LocalWindow, WindowedStrategy
from strategies.selection import BUY, SELL, Selection, config_order

STRATEGY_ID = "RS-MOM-01"

# Parameters table of the spec: frozen defaults, one-at-a-time grids, the k x R heatmap
# (13 + 16 = 29 configurations).
PARAMS = ParamGrid(
    defaults={"theta": 1.0, "k": 1.0, "R": 1.5},
    grid={"theta": (0.8, 0.9, 1.0, 1.1, 1.2),
          "k": (0.75, 0.875, 1.0, 1.125, 1.25),
          "R": (1.0, 1.25, 1.5, 1.75, 2.0)},
    heatmap=("k", "R"),
)

# Frozen (spec "Frozen" paragraph): signal windows on T, exchange-local, both ends inclusive.
US_WINDOW = LocalWindow.of("10:30", "13:30")       # New York: 14:30-17:30 / 15:30-18:30 UTC
GER40_WINDOW = LocalWindow.of("10:00", "15:00")    # Berlin: 08:00-13:00 / 09:00-14:00 UTC
WINDOWS = {"US30": US_WINDOW, "NAS100": US_WINDOW, "GER40": GER40_WINDOW}

MAX_PER_INDEX_SIDE_DAY = 2                         # frozen: 2 per index per side per local date

_ATRH_COL = "_rsmom01_atrh"                        # view.columns cache keys (per scan)


def round_away(x: float, entry: float) -> float:
    """Round x to 0.01 away from `entry` (the stop: never tighter than k ATRH). The x * 100 is
    first rounded to 6 dp so float noise (e.g. 3900011.9999999995) does not move a whole cent."""
    c = round(x * 100.0, 6)
    return (math.floor(c) if x < entry else math.ceil(c)) / 100.0


def round_toward(x: float, entry: float) -> float:
    """Round x to 0.01 toward `entry` (the target: never further than R k ATRH)."""
    c = round(x * 100.0, 6)
    return (math.ceil(c) if x < entry else math.floor(c)) / 100.0


def reason_line(side: str, key: str, role: str, window_set: str, score: float, theta: float,
                exit_utc: pd.Timestamp, exit_label: str) -> str:
    """The spec's reason line (see the SPEC NOTEs on <theta> and the window set). `theta` is
    the parameter value, printed unsigned as the template's <theta> placeholder."""
    word = "LONG" if side == BUY else "SHORT"
    return (f"{STRATEGY_ID} {word} {key}: {role} of {window_set}, RS_4h {score:+.2f} crossed "
            f"{theta:.2f}, regime ACTIVE. MANUAL TIME EXIT {exit_utc:%H:%M} UTC "
            f"({exit_label}) if still open.")


class RsMom01(WindowedStrategy):
    """RS-MOM-01 on the shared WindowedStrategy driver (live generate() == batch scan())."""
    id = STRATEGY_ID
    param_grid = PARAMS
    regime_gate = "ACTIVE"
    rank = "rs_4h"

    def windows(self, params: Mapping[str, Any]) -> dict[str, LocalWindow]:
        return dict(WINDOWS)

    # --- ATRH at T (COMMON 2 H1 alignment) --------------------------------------------------
    def _atrh(self, view: CycleView, key: str) -> tuple[float, str]:
        """ATR14_H1 of the H1 bar the COMMON 2 rule picks at T, or (NaN, cause).

        Computed once per scan and index for every M15 close (vectorised h1_alignment), then
        read at the row of the M15 bar closing at T. The rule at a close c uses only H1 bars
        closed by c and M15 bars closed by c, so the row at T never sees anything after T."""
        cache = view.columns.get((key, _ATRH_COL))
        if cache is None:
            fx = view.fx[key]
            n = len(fx.m15)
            if n and len(fx.h1):
                hpos, cause = fx.h1_positions(pd.DatetimeIndex(fx.m15["close_utc"]))
                atr = fx.h1["atr14"].to_numpy(dtype=float)
                vals = np.where(hpos >= 0, atr[np.maximum(hpos, 0)], np.nan)
            else:
                vals, cause = np.full(n, np.nan), np.full(n, "no H1 bars", dtype=object)
            cache = view.columns[(key, _ATRH_COL)] = (vals, cause)
        vals, cause = cache
        p = view.pos[key]
        v = float(vals[p])
        if cause[p]:
            return float("nan"), str(cause[p])
        if not (math.isfinite(v) and v > 0):
            return float("nan"), "ATR14_H1 NaN or not positive"
        return v, ""

    def incomplete_cause(self, view: CycleView, key: str) -> str | None:
        _, cause = self._atrh(view, key)
        return f"ATR14_H1 unavailable: {cause}" if cause else None

    # --- entry rule ---------------------------------------------------------------------------
    def _fires(self, view: CycleView, key: str, side: str, theta: float) -> bool:
        """Fresh crossing of the index's own RS_4h (spec Entry 1-2). Rank and gate are
        filters at t only; freshness looks at the index's previous M15 bar (15 min earlier),
        whose RS_4h uses bars closed by T - 15 min. Absent or NaN previous score: no fire."""
        now = float(view.selection.scores[key])
        prev_t = view.T - pd.Timedelta(minutes=tf_minutes(view.cfg["timeframes"]["entry"]))
        prev = float(view.value_at(key, "rs_4h", prev_t))
        if not (math.isfinite(now) and math.isfinite(prev)):
            return False
        if side == BUY:
            return now >= theta and prev < theta
        return now <= -theta and prev > -theta

    def _locked(self, view: CycleView, key: str, side: str) -> bool:
        """Spec Entry 3: occupied by an earlier RS-MOM-01 candidate, or the per-day cap used."""
        if view.book.occupied(key, view.T):
            return True
        return view.book.count(key, view.local_date[key], side) >= MAX_PER_INDEX_SIDE_DAY

    def decide(self, view: CycleView) -> list[Candidate]:
        sel: Selection = view.selection
        p = view.params
        theta, k, R = float(p["theta"]), float(p["k"]), float(p["R"])
        out: list[Candidate] = []
        for side, key in ((BUY, sel.long), (SELL, sel.short)):
            if key is None or not self._fires(view, key, side, theta):
                continue
            if self._locked(view, key, side):
                continue
            atrh, cause = self._atrh(view, key)
            if cause:
                # incomplete_cause() marks the whole group incomplete before decide() runs, so
                # this is an invariant breach (e.g. a driver change to the order of its
                # completeness checks). Fail loudly: skipping would print "no setup" for a
                # fresh, selected, unlocked crossing with no log line (PROMPT: no silent
                # fallback; spec Stop: "if unavailable, no candidate" is a group outcome).
                raise AssertionError(
                    f"{self.id} {view.T}: ATRH unavailable for selected {key} ({cause}) but "
                    f"group was not marked incomplete")
            out.append(self._candidate(view, key, side, theta, k, R, atrh))
        # Config instrument order within one T (Strategy.scan contract), not side order: the
        # live path (signals/run.py) processes generate() output as returned.
        order = {key: n for n, key in enumerate(config_order(self.cfg))}
        return sorted(out, key=lambda c: order[c.key])

    def _candidate(self, view: CycleView, key: str, side: str, theta: float, k: float,
                   R: float, atrh: float) -> Candidate:
        sel = view.selection
        e = float(view.value(key, "close"))
        sg = 1.0 if side == BUY else -1.0
        d = k * atrh
        stop = round_away(e - sg * d, e)
        tp = round_toward(e + sg * R * d, e)
        fields = view.candidate_fields(key)
        role = "sole index" if sel.sole else ("leader" if side == BUY else "laggard")
        reason = reason_line(side, key, role, sel.set_text(), float(sel.scores[key]),
                             theta, fields["time_exit_utc"],
                             time_exit_label(view.cfg, key))
        c = Candidate(key, side, e, stop, tp, float(view.value(key, "atr14")), reason,
                      order_type=MARKET_BAND, expires_utc=None, **fields)
        return with_protocol_levels(c, view.cfg)


STRATEGY_CLASS = RsMom01
