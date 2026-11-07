"""MR-TWAP-01: calm-regime session-mean stretch fade (docs/specs/MR-TWAP-01.md, GATE 2).

At each decision time T (close of M15 bar t) inside the strategy window, the index that is
strongest over the last trading day (RS_day, COMMON 4.1) may be bought when its close is
stretched K ATR below today's session mean (TWAP of typical price from the cash open), and the
weakest may be sold when it is stretched K ATR above it, only while the HMM regime is CALM
(V_T < 0, COMMON 6.7). Entry E = C_t with the MARKET_OR_LIMIT protocol (never chase), SL M ATR
away, TP = TWAP_t frozen at T, a limit EXPIRY of min(T + 60 min, time exit - 30 min), and a
flagged manual time exit (15:45 New York / 17:15 Berlin).

Everything shared lives elsewhere and is reused, never duplicated:
  - windows, G(T) per exchange group, regime gate per T, lockout replay, live == batch driver:
    strategies.context.WindowedStrategy (generate() replays the session with the scan code);
  - COMMON 4.2 selection (argmax / argmin RS_day over E(T), sole-index sign rule, no promotion):
    strategies.selection;
  - TWAP, ATR14_M15, RS_day and the H1 alignment rule: features.service / features.core;
  - protocol levels (SKIP LEVEL = E -/+ 0.50 D, sizing_entry = E): strategies.base;
  - reference occupancy (COMMON 7.1): signals.occupancy through the LockoutBook.
This module only states the spec's windows, parameters, entry, stop, exit and reason line.

Public API
    STRATEGY_ID = "MR-TWAP-01"
    PARAMS: ParamGrid            defaults K = 2.0, M = 1.5; one-at-a-time grids; heatmap (K, M)
    MrTwap01(cfg, params=None, *, off_grid=False, stats=None)   a WindowedStrategy
        .windows(params) -> {key: LocalWindow}
        .generate(bars, regime_bars, now, context)  live path (inherited)
        .scan(features, regime_tables, start, end, params=None, *, on_selection=None)
                                                     batch path (inherited); a feasibility or
                                                     backtest run passes on_selection and keeps
                                                     every sel.log_line() (COMMON 4.2.8)
        .cycles(...)                                 per-(T, group) Selection + candidates
    signal_window(cfg, key) -> LocalWindow       cash open + 60 min .. time exit - 75 min
    expiry_utc(cfg, key, T) -> Timestamp         min(T + 60 min, time exit - 30 min)
    stretch(close, twap, atr) -> float           Z = (C - TWAP) / ATR (NaN unless ATR > 0)
    round_away(x, entry) / round_toward(x, entry)   0.01 rounding of SL / TP
    models_without_calm(table) -> set[str]       model ids whose states have no CALM tag

SPEC NOTE (lockouts are entry conditions, not E(T) filters): the one-candidate-per-index-per-
session rule and reference occupancy are Entry step 3, and E(T) is "members passing the CALM
gate". A locked-out index therefore still takes part in the ranking; if it is selected, there
is no candidate on that side and the other index is never promoted (COMMON 4.2.6).
SPEC NOTE (session = local date of T): "one candidate per index per session (local date)" counts
candidates by the exchange-local date of T (the driver's view.local_date).
SPEC NOTE (D for the protocol): the spec says D = M * ATR_t; the SKIP LEVEL and sizing use
COMMON 7's D = |E - SL| with SL already rounded to 0.01 (strategies.base.with_protocol_levels,
which derives levels from cfg). The two differ by less than 0.01 point.
SPEC NOTE (ATR not positive): Z is undefined when ATR14_M15 at t is 0; then neither trigger
fires for that index (no candidate on it), while a NaN ATR or TWAP is missing data and makes
the whole group incomplete (COMMON 4.2.2).
SPEC NOTE (entry not rounded): E = C_t exactly as the bid bar prints it; only SL and TP are
rounded to 0.01 (spec "Stop" and "Exit").
SPEC NOTE (reason line): the template "<strongest|weakest|sole index, RS_day +x.xx>" is read as
one of the three words followed by ", RS_day <signed 2 dp>"; "sole index" when |E(T)| = 1
(always for GER40), else "strongest" for LONG and "weakest" for SHORT. Z is printed signed to
2 dp, RS_day as Candidate.rank_score (2 dp, COMMON 4.2.8). <TWAP> is printed as the TP level
(TWAP_t rounded toward entry), not as f"{TWAP_t:.2f}": nearest rounding can differ from the
toward-entry TP by 0.01, and the owner must not see two "session mean" numbers on one ticket.
SPEC NOTE (no CALM state): a model with no CALM-tagged state has every mu_vol >= 0, so V_T >= 0
and the CALM gate never passes: the strategy is inactive for that index by construction. The
feasibility diagnostics name such models with models_without_calm().
"""
from __future__ import annotations

import math
from datetime import datetime, time, timedelta
from typing import Any, Mapping

import pandas as pd

from core.calendar import exchange_for
from strategies.base import (MARKET_OR_LIMIT, Candidate, ParamGrid, time_exit_label,
                             time_exit_utc, with_protocol_levels)
from strategies.context import CycleView, LocalWindow, WindowedStrategy
from strategies.selection import BUY, SELL, config_order

STRATEGY_ID = "MR-TWAP-01"

# Parameters table of the spec: frozen defaults, one-at-a-time grids, and the 5 x 5 K x M
# heatmap that contains the one-at-a-time grid (25 configurations).
PARAMS = ParamGrid(
    defaults={"K": 2.0, "M": 1.5},
    grid={"K": (1.6, 1.8, 2.0, 2.2, 2.4), "M": (1.0, 1.25, 1.5, 1.75, 2.0)},
    heatmap=("K", "M"),
)

# Frozen spec constants ("Frozen" paragraph of the spec); not parameters, never varied.
WINDOW_START_AFTER_OPEN = timedelta(minutes=60)    # M15 ATR lags in the first hour
WINDOW_END_BEFORE_EXIT = timedelta(minutes=75)     # every limit expires >= 30 min before exit
EXPIRY_AFTER_T = pd.Timedelta(minutes=60)
EXPIRY_BEFORE_EXIT = pd.Timedelta(minutes=30)
PRICE_STEP_DP = 2                                  # "rounded to 0.01"


# ---------------------------------------------------------------------------------------------
# Windows, expiry, stretch, rounding
# ---------------------------------------------------------------------------------------------

def _shift(clock: time, delta: timedelta) -> time:
    out = datetime.combine(datetime(2000, 1, 3).date(), clock) + delta
    if out.date() != datetime(2000, 1, 3).date():
        raise ValueError(f"window end {clock} {delta} leaves the local day")
    return out.time()


def signal_window(cfg: dict, key: str) -> LocalWindow:
    """Exchange-local window on T, both ends inclusive: cash open + 60 min to time exit - 75 min.
    Read from config (exchanges.cash_open, time_exits), so it is 10:30-14:30 New York and
    10:00-16:00 Berlin with the current values; zoneinfo turns it into UTC per date."""
    ex = exchange_for(cfg, key)
    clock = (cfg.get("time_exits") or {}).get(key)
    if clock is None:
        raise ValueError(f"{STRATEGY_ID}: no time exit configured for {key} (config time_exits)")
    return LocalWindow(_shift(ex.cash_open, WINDOW_START_AFTER_OPEN),
                       _shift(time.fromisoformat(clock), -WINDOW_END_BEFORE_EXIT))


def expiry_utc(cfg: dict, key: str, T: pd.Timestamp) -> pd.Timestamp:
    """EXPIRY = min(T + 60 min, time exit - 30 min), the time exit on T's local date."""
    te = time_exit_utc(cfg, key, T)
    if te is None:
        raise ValueError(f"{STRATEGY_ID}: no time exit configured for {key}")
    return min(T + EXPIRY_AFTER_T, te - EXPIRY_BEFORE_EXIT)


def stretch(close: float, twap: float, atr: float) -> float:
    """Z_t = (C_t - TWAP_t) / ATR_t; NaN unless every input is finite and ATR > 0."""
    if not (math.isfinite(close) and math.isfinite(twap) and math.isfinite(atr)) or atr <= 0:
        return float("nan")
    return (close - twap) / atr


def _scaled(x: float) -> float:
    # round(..., 6) first so 39000.12 * 100 = 3900011.9999999995 is not floored to ...11.
    return round(x * 10 ** PRICE_STEP_DP, 6)


def round_away(x: float, entry: float) -> float:
    """Round x to 0.01 away from entry (the stop: the risk is never understated)."""
    f = math.floor if x < entry else math.ceil
    return f(_scaled(x)) / 10 ** PRICE_STEP_DP


def round_toward(x: float, entry: float) -> float:
    """Round x to 0.01 toward entry (the take profit: the reward is never overstated)."""
    f = math.ceil if x < entry else math.floor
    return f(_scaled(x)) / 10 ** PRICE_STEP_DP


def models_without_calm(table: Any) -> set[str]:
    """Model ids (regime.service.RegimeTable records) whose states carry no CALM volatility tag:
    the CALM gate cannot pass while such a model serves (spec "Regime condition"). Tables
    without fitted records (stand-ins) give an empty set."""
    out = set()
    for rec in getattr(table, "records", None) or []:
        m = getattr(rec, "model", None)
        if m is not None and "CALM" not in list(m.vol_tags):
            out.add(m.model_id)
    return out


# ---------------------------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------------------------

class MrTwap01(WindowedStrategy):
    """MR-TWAP-01 on the shared WindowedStrategy driver: RS_day ranking, CALM gate."""
    id = STRATEGY_ID
    param_grid = PARAMS
    regime_gate = "CALM"
    rank = "rs_day"

    def windows(self, params: Mapping[str, Any]) -> dict[str, LocalWindow]:
        # US30, NAS100, GER40 (config order of tradeable instruments; US500 is never ranked).
        return {k: signal_window(self.cfg, k) for k in config_order(self.cfg)}

    def incomplete_cause(self, view: CycleView, key: str) -> str | None:
        """TWAP_t or ATR14_M15 NaN at T is a strategy input missing because of data (COMMON
        4.2.2): the whole group gets no candidate."""
        if not math.isfinite(float(view.value(key, "twap"))):
            return "TWAP NaN (cash-open bar missing in the session)"
        if not math.isfinite(float(view.value(key, "atr14"))):
            return "ATR14_M15 NaN (14-bar lookback incomplete in the segment)"
        return None

    def decide(self, view: CycleView) -> list[Candidate]:
        sel, p, T = view.selection, view.params, view.T
        K, M = float(p["K"]), float(p["M"])
        out: list[Candidate] = []
        for side, key in ((BUY, sel.long), (SELL, sel.short)):
            if key is None:
                continue
            # Entry step 3: one per index per session (either side), and no new candidate
            # while an earlier MR-TWAP-01 candidate occupies the index (COMMON 7.1).
            if view.book.count(key, view.local_date[key]) >= 1 or view.book.occupied(key, T):
                continue
            close = float(view.value(key, "close"))
            twap = float(view.value(key, "twap"))
            atr = float(view.value(key, "atr14"))
            z = stretch(close, twap, atr)
            if not math.isfinite(z):
                continue
            if (side == BUY and not z <= -K) or (side == SELL and not z >= K):
                continue
            out.append(self._candidate(view, key, side, close, twap, atr, z, M))
        order = {k: n for n, k in enumerate(config_order(self.cfg))}
        return sorted(out, key=lambda c: order[c.key])

    def _candidate(self, view: CycleView, key: str, side: str, e: float, twap: float,
                   atr: float, z: float, M: float) -> Candidate:
        sg = 1.0 if side == BUY else -1.0
        stop = round_away(e - sg * M * atr, e)
        tp = round_toward(twap, e)
        fields = view.candidate_fields(key)
        expires = expiry_utc(self.cfg, key, view.T)
        sel = view.selection
        rel = "sole index" if sel.sole else ("strongest" if side == BUY else "weakest")
        reason = (f"{STRATEGY_ID} {'LONG' if side == BUY else 'SHORT'} {key}: Z {z:+.2f} from "
                  f"session mean {tp:.2f}, {rel}, RS_day {fields['rank_score']:+.2f}, regime "
                  f"CALM. Limit at entry if price is worse; expires {expires:%H:%M} UTC. "
                  f"MANUAL TIME EXIT {fields['time_exit_utc']:%H:%M} UTC "
                  f"({time_exit_label(self.cfg, key)}) if still open.")
        c = Candidate(key, side, e, stop, tp, atr, reason, order_type=MARKET_OR_LIMIT,
                      expires_utc=expires, **fields)
        return with_protocol_levels(c, self.cfg)
