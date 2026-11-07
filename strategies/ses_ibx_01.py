"""SES-IBX-01: initial-balance extension with resting stop orders (ACTIVE regime).

Spec: docs/specs/SES-IBX-01.md, on top of docs/specs/COMMON.md. This module only supplies what
the spec adds; selection, the regime gate, lockouts and the live/batch driver are the shared
strategies.context.WindowedStrategy, so generate() (live) and scan() (batch) are one code path.

Rule summary (spec section in brackets)
    Decision time T  close of the last IB bar: cash open + ib_minutes, exchange-local
                     (default 10:30 New York, 10:00 Berlin), on exchange-calendar trading dates
                     [Instruments]. One decision time per index per local date.
    Window set       {US30, NAS100} at the US T, {GER40} at the GER40 T [Index selection]
    Rank score       RS_4h at T; regime gate ACTIVE (V_T >= 0) [Regime condition]
    Valid date       an index whose local date lacks the cash-open bar (COMMON 1.6) is outside
                     G(T) (Selection.excluded, strategies.context); the rest of its group is
                     ranked without it (a lone US index follows the 4.2.5 sign rule)
    Completeness     M15 bar at T, RS_4h, IB complete, DATR finite: any missing -> no candidate
                     for the whole group (COMMON 4.2.2; SPEC NOTE 1)
    Eligibility      W <= max_ib_datr * DATR; no config restricted window overlaps [T, EXPIRY];
                     no earlier SES-IBX-01 candidate on the index that local date [Entry 1]
    Side             LONG-SELECTED -> BUY STOP only if RS_4h > 0; SHORT-SELECTED -> SELL STOP
                     only if RS_4h < 0 (two US tickets of opposite sign = a pair day)
    Levels           b = 0.05 W; trigger IBH + b (long) / IBL - b (short); SL = IB midpoint;
                     TP = IBH + tp_ib_mult W (long) / IBL - tp_ib_mult W (short) [Entry 3, Stop,
                     Exit]; all absolute prices fixed at T
    Protocol         STOP_ENTRY; CHASE LIMIT = trigger +/- 0.20 D and sizing_entry = CHASE LIMIT
                     from cfg `protocols` (with_protocol_levels); EXPIRY 12:00 New York /
                     11:30 Berlin on T's local date; manual time exit from cfg `time_exits`
                     (15:45 New York / 17:15 Berlin) [Entry 4-5, Sizing]

Why every level is fixed at T: the owner places the ticket 1-3 bars later and must never
re-anchor anything to the fill (COMMON 7). Why the EXPIRY is a fixed clock and not a parameter:
QA found the old order_life interacted with ib_minutes; the spec freezes it.

Public API
    STRATEGY_ID, PARAMS (ParamGrid: frozen defaults, one-at-a-time grid, heatmap pair)
    TRIGGER_BUFFER_FRAC (b = 0.05 W), EXPIRY_LOCAL ({exchange calendar: local clock})
    SesIbx01(cfg, params=None, *, off_grid=False, stats=None)
        .scan(features, regime_tables, start, end, params=None, *, on_selection=None)
                                                                          -> [Candidate]   batch
        .generate(bars, regime_bars, now, context) -> [Candidate]                live
        .cycles(...) -> (Selection, [Candidate]) per G(T), for diagnostics and setup counts
        .windows(params) -> {key: LocalWindow(T, T)}
    decision_clock(cfg, key, ib_minutes) -> time     T's exchange-local clock
    expiry_utc(cfg, key, T) -> Timestamp             EXPIRY on T's local date, in UTC
    restricted_overlap(windows, T, expiry) -> str | None
    w_datr_text(w, datr) -> str                      W/DATR to 2 dp, "n/a" when DATR <= 0
    pair_days(candidates) -> [(T, (long key, short key))]   for QA's separate pair-day report

SPEC NOTEs (each also marked where it applies)
1. IB incomplete and DATR NaN are listed under "Eligibility" in the spec, but both are "NaN
   strategy input because of missing data", which COMMON 4.2.2 (which a strategy spec may not
   change) turns into "incomplete group": no candidate for the whole group at T. W >
   max_ib_datr * DATR, a restricted window and the one-per-day rule remove only that index from
   E(T) (strategies.context ineligible_cause, applied after the regime gate).
2. Restricted windows are config `rules.restricted_windows_utc` entries {start, end, reason},
   read as [start, end) like risk.rules does; [T, EXPIRY] is closed as written in the spec. So
   they overlap iff start <= EXPIRY and end > T.
3. Reason line: the spec gives the BUY form only. A SELL reads "SELL STOP" and "bottom" where
   the BUY reads "top"; "sole" when E(T) has one index (Selection.sole). "{<set>}" is the window
   set G(T) (the spec's own term "Window set", as RS-MOM-01's reason line); an index outside
   G(T) because its date is not valid (Selection.excluded) is not printed, so a NAS100 ranked
   alone on a day US30 lacks its cash-open bar reads "sole of {NAS100}". "regime ACTIVE" is
   the volatility tag of Candidate.regime (always ACTIVE under the ACTIVE gate). Prices, W and
   W/DATR are printed to 2 dp; the ticket prints the levels at the symbol's precision.
4. W = 0 (IBH = IBL) is not guarded: the spec removed the minimum-width guard as inert. Such a
   candidate has SL = trigger = TP (D = 0) and is suppressed by the ticket engine (sizing:
   "stop distance is zero"); it still consumes the index's one candidate that day.
5. A decision clock at or after EXPIRY (impossible on the spec grid; possible only with an
   off-grid ib_minutes in a synthetic test) raises ValueError: the spec says an order placed at
   or after EXPIRY is skipped, so such a configuration could never trade.
6. DATR = 0 (finite: 14 prior flat sessions; impossible after the A1 exclusion) is not a
   data-completeness cause: COMMON 4.2.2 covers only NaN inputs from missing data, and the
   filter W <= max_ib_datr * DATR is well defined at 0 (W > 0 is ineligible, W = 0 passes and
   falls under SPEC NOTE 4). Only the printed ratio W/DATR is undefined; it is printed "n/a"
   (w_datr_text) so a data edge never aborts the cycle. No eligibility rule is changed.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from core.calendar import exchange_for, local_to_utc
from core.config import tf_minutes
from strategies.base import (STOP_ENTRY, Candidate, ParamGrid, time_exit_label,
                             with_protocol_levels)
from strategies.context import CycleView, LocalWindow, WindowedStrategy
from strategies.selection import BUY, SELL, config_order

STRATEGY_ID = "SES-IBX-01"

# Parameters table of the spec: frozen defaults, one-at-a-time grid, 2-D heatmap
# ib_minutes x tp_ib_mult. 13 one-at-a-time + 16 more heatmap cells = 29 configurations.
PARAMS = ParamGrid(
    defaults={"ib_minutes": 60, "tp_ib_mult": 1.0, "max_ib_datr": 0.6},
    grid={"ib_minutes": (30, 45, 60, 75, 90),
          "tp_ib_mult": (0.5, 0.75, 1.0, 1.25, 1.5),
          "max_ib_datr": (0.5, 0.55, 0.6, 0.65, 0.7)},
    heatmap=("ib_minutes", "tp_ib_mult"),
)

# Frozen spec constants (not parameters; a change is a new trial).
TRIGGER_BUFFER_FRAC = 0.05                                  # b = 0.05 W
EXPIRY_LOCAL = {"NYSE": time(12, 0), "XETRA": time(11, 30)}  # 12:00 New York / 11:30 Berlin

# Data-completeness causes (COMMON 4.2.2, SPEC NOTE 1).
DATR_NAN = "DATR NaN (fewer than 14 prior valid dates with >= 90% of session bars in segment)"


def _city(tz: str) -> str:
    return tz.split("/")[-1].replace("_", " ")


def w_datr_text(w: float, datr: float) -> str:
    """W/DATR to 2 dp for the reason line and the width-filter cause; "n/a" when DATR <= 0
    (SPEC NOTE 6: the ratio is undefined, the filter itself is not)."""
    return f"{w / datr:.2f}" if datr > 0 else "n/a"


def decision_clock(cfg: dict, key: str, ib_minutes: int) -> time:
    """Exchange-local clock of T: cash open + ib_minutes (close of the last IB bar)."""
    bar = tf_minutes(cfg["timeframes"]["entry"])
    if ib_minutes <= 0 or ib_minutes % bar:
        raise ValueError(f"ib_minutes must be a positive multiple of {bar}, got {ib_minutes}")
    ex = exchange_for(cfg, key)
    return (datetime.combine(date(2000, 1, 3), ex.cash_open)
            + timedelta(minutes=int(ib_minutes))).time()


def expiry_clock(cfg: dict, key: str) -> time:
    """EXPIRY as an exchange-local clock (frozen: 12:00 New York, 11:30 Berlin)."""
    return EXPIRY_LOCAL[exchange_for(cfg, key).calendar]


def expiry_utc(cfg: dict, key: str, T) -> pd.Timestamp:
    """EXPIRY on T's exchange-local date, in UTC (zoneinfo, so it follows each market's DST:
    US 16:00 / 17:00 UTC, GER40 09:30 / 10:30 UTC)."""
    ex = exchange_for(cfg, key)
    d = pd.Timestamp(T).tz_convert(ex.tz).date()
    return local_to_utc(d, expiry_clock(cfg, key), ex.tz).as_unit("ns")


def parse_restricted(cfg: dict) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """config rules.restricted_windows_utc as (start, end, reason) in UTC."""
    out = []
    for w in (cfg.get("rules") or {}).get("restricted_windows_utc") or ():
        s, e = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        s = s.tz_localize("UTC") if s.tzinfo is None else s.tz_convert("UTC")
        e = e.tz_localize("UTC") if e.tzinfo is None else e.tz_convert("UTC")
        out.append((s, e, str(w.get("reason") or "restricted")))
    return out


def restricted_overlap(windows: Iterable[tuple[pd.Timestamp, pd.Timestamp, str]],
                       T: pd.Timestamp, expiry: pd.Timestamp) -> str | None:
    """Reason of the first restricted window [start, end) that overlaps [T, EXPIRY], else None.
    SPEC NOTE 2: overlap iff start <= EXPIRY and end > T."""
    for s, e, why in windows:
        if s <= expiry and e > T:
            return why
    return None


def pair_days(candidates: Sequence[Candidate]) -> list[tuple[pd.Timestamp, tuple[str, str]]]:
    """Decision times with a BUY on one index and a SELL on another (spec: QA reports pair days
    separately). Returns (T, (BUY key, SELL key)) in time order."""
    by_t: dict[pd.Timestamp, dict[str, str]] = {}
    for c in candidates:
        by_t.setdefault(pd.Timestamp(c.signal_time), {})[c.side] = c.key
    return [(t, (s[BUY], s[SELL])) for t, s in sorted(by_t.items())
            if BUY in s and SELL in s and s[BUY] != s[SELL]]


class SesIbx01(WindowedStrategy):
    """SES-IBX-01 on the shared WindowedStrategy driver (see the module docstring)."""
    id = STRATEGY_ID
    param_grid = PARAMS
    regime_gate = "ACTIVE"
    rank = "rs_4h"

    def __init__(self, cfg: dict, params: Mapping[str, Any] | None = None, *,
                 off_grid: bool = False, stats: dict | None = None):
        super().__init__(cfg, params, off_grid=off_grid, stats=stats)
        self._restricted = parse_restricted(cfg)

    # --- windows -------------------------------------------------------------------------------
    def windows(self, params: Mapping[str, Any]) -> dict[str, LocalWindow]:
        """One decision time per index per date: the close of the last IB bar."""
        out = {}
        for key in config_order(self.cfg):
            t = decision_clock(self.cfg, key, int(params["ib_minutes"]))
            if t >= expiry_clock(self.cfg, key):
                # SPEC NOTE 5: no order may be placed at or after EXPIRY.
                raise ValueError(f"{key}: decision time {t} with ib_minutes "
                                 f"{params['ib_minutes']} is not before EXPIRY "
                                 f"{expiry_clock(self.cfg, key)}")
            out[key] = LocalWindow(t, t)
        return out

    # --- per-index inputs at T -----------------------------------------------------------------
    def _ib(self, view: CycleView, key: str):
        """IB of T's local date as known at T (features.service: NaN + cause unless complete
        and closed by T)."""
        return view.fx[key].ib(view.local_date[key], int(view.params["ib_minutes"]), view.T)

    def _datr(self, view: CycleView, key: str) -> float:
        """DATR of T's local date: prior valid dates only, so known at T."""
        return float(view.fx[key].datr(view.local_date[key]))

    def incomplete_cause(self, view: CycleView, key: str) -> str | None:
        # SPEC NOTE 1: IB incomplete and DATR NaN block the whole group (COMMON 4.2.2).
        ib = self._ib(view, key)
        if not ib.available:
            return ib.cause or "IB incomplete"
        if not math.isfinite(self._datr(view, key)):
            return DATR_NAN
        return None

    def ineligible_cause(self, view: CycleView, key: str) -> str | None:
        """Entry 1 conditions that remove `key` from E(T); called only when its regime gate
        passes and its inputs are complete."""
        p = view.params
        ib, datr = self._ib(view, key), self._datr(view, key)
        if not ib.w <= p["max_ib_datr"] * datr:
            # SPEC NOTE 6: DATR = 0 is evaluated literally; only the printed ratio is guarded.
            return (f"IB width W {ib.w:.2f} = {w_datr_text(ib.w, datr)} DATR > max_ib_datr "
                    f"{p['max_ib_datr']}")
        why = restricted_overlap(self._restricted, view.T, expiry_utc(self.cfg, key, view.T))
        if why is not None:
            return f"restricted window overlaps [T, EXPIRY]: {why}"
        if view.book.count(key, view.local_date[key]) >= 1:
            return f"one {self.id} candidate per index per day: already issued on this date"
        return None

    # --- entry ---------------------------------------------------------------------------------
    def decide(self, view: CycleView) -> list[Candidate]:
        s = view.selection
        out: list[Candidate] = []
        for key in s.members:                                  # config order
            score = s.scores[key]
            for side, chosen, ok in ((BUY, s.long, score > 0), (SELL, s.short, score < 0)):
                # The one-per-day rule already removed an index with an earlier candidate
                # from E(T) (ineligible_cause); on an exact tie (long == short) only one sign
                # can hold, so an index never gets two candidates here.
                if key == chosen and ok:
                    out.append(self._candidate(view, key, side))
                    break
        return out

    def _candidate(self, view: CycleView, key: str, side: str) -> Candidate:
        cfg, p, s = self.cfg, view.params, view.selection
        ib, datr = self._ib(view, key), self._datr(view, key)
        w = ib.w
        # SPEC NOTE 4: W = 0 is not guarded (the spec removed the minimum-width guard); the
        # levels collapse to SL = trigger = TP and the ticket engine's sizing suppresses it.
        b = TRIGGER_BUFFER_FRAC * w
        sl = (ib.ibh + ib.ibl) / 2.0
        if side == BUY:
            trigger, tp = ib.ibh + b, ib.ibh + p["tp_ib_mult"] * w
        else:
            trigger, tp = ib.ibl - b, ib.ibl - p["tp_ib_mult"] * w
        fields = view.candidate_fields(key)
        expires = expiry_utc(cfg, key, view.T)
        tz = exchange_for(cfg, key).tz
        # SPEC NOTE 3: the spec gives only the BUY form. SELL reads "SELL STOP" and "bottom";
        # "sole" when E(T) has one index; {<set>} is G(T) (Selection.excluded not printed);
        # prices, W and W/DATR to 2 dp (SPEC NOTE 6: W/DATR "n/a" when DATR <= 0).
        place = "sole" if s.sole else ("top" if side == BUY else "bottom")
        vol_tag = fields["regime"].split("/")[0]
        te = fields["time_exit_utc"]
        reason = (f"{self.id} {side} STOP {key} at {trigger:.2f} (IB {ib.ibl:.2f}-{ib.ibh:.2f}, "
                  f"W {w:.2f} = {w_datr_text(w, datr)} DATR), RS_4h {s.scores[key]:+.2f} {place} of "
                  f"{s.set_text()}, regime {vol_tag}. Order expires {expires:%H:%M} UTC "
                  f"({expiry_clock(cfg, key):%H:%M} {_city(tz)}). MANUAL TIME EXIT "
                  f"{te:%H:%M} UTC ({time_exit_label(cfg, key)}) if filled and still open.")
        c = Candidate(key, side, float(trigger), float(sl), float(tp),
                      float(view.value(key, "atr14")), reason, order_type=STOP_ENTRY,
                      expires_utc=expires, **fields)
        return with_protocol_levels(c, cfg)
