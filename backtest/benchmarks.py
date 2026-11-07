"""COMMON 10.6 benchmarks for a run-A configuration: B1 placebo, B2 random index, B3 gate off.

Why they exist: run A can look profitable because the market drifted, because the bracket
geometry and costs happen to suit the period, or because one index carried the book. Each
benchmark keeps everything of run A except the one ingredient the strategy claims as its edge,
and reruns it through the same backtester (backtest.engine.run_backtest) under STRESS-d2:
    B1 placebo       the signal TIME   -> random time in the strategy window, same date, index,
                                          side, geometry in the strategy's own units, protocol
                                          and costs; 200 draws, seeds 1..200 (pass criterion 7:
                                          run A total $ P&L > 95th percentile of the draws)
    B2 random index  the INDEX choice  -> a uniformly random other index whose window contains
                                          T, same T and side (report: if the strategy does not
                                          beat B2, relative-strength selection adds nothing)
    B3 gate off      the REGIME GATE   -> the same strategy rules with an always-pass gate
                                          (report: if B3 >= run A in $ expectancy, the gate
                                          adds nothing; the gate is never changed in response)
Benchmarks are stage `benchmark`: never selectable, logged, not counted in N (COMMON 10.6,
10.7). This module never writes docs/trial_log.csv itself; every result has
trial_log_fields() for the runner, which owns the log. Each fields dict maps onto one
backtest.trials.TrialRecord: stage -> stage, variant (B1/B2/B3 name) -> variant (so the three
benchmarks of one configuration get distinct config keys), scenario_key -> the key of
`scenarios` with metrics as its value, extra -> extra; result_summary and notes are the text.

Strategy-specific placebo / B2 geometry lives in one class per strategy (PlaceboGeometry);
MR-TWAP-01 was DROPPED by amendment A3 and has none, so it can never be run here.

Public API
    STAGE = "benchmark"; PLACEBO_SEEDS = range(1, 201); BENCHMARK_SCENARIO = "STRESS-d2"
    SES_PLACEBO_CLOSE, SES_PLACEBO_SHIFT (SES_PLACEBO_MODES); SES_PLACEBO_RULING = None
    BenchmarkInputs(cfg, strategy, features, regime_tables, bars, specs, scenario=STRESS-d2,
                    contract=None, contract_name="demo", ses_placebo=None)
                                                          everything a backtest of run A needs
    geometry_for(inputs) -> PlaceboGeometry               RsMom01Geometry | SesIbx01Geometry
    PlaceboGeometry
        .strategy_levels(key, side, T) -> (Candidate | None, cause)   the strategy's own levels
        .placebo_levels(key, side, T, source=None) -> (Candidate | None, cause)  B1 geometry
        .placebo_times(c) -> DatetimeIndex                            the B1 window of c
    placebo_options(inputs, run_a) -> list[PlaceboOptions]            eligible T' per candidate
    placebo_draw(options, seed) -> PlaceboDraw                         one seeded draw (no P&L)
    run_placebo(inputs, run_a, seeds=PLACEBO_SEEDS, *, keep_results=False)
        -> PlaceboResult (.totals, .p95, .seeds, .complete, .above_p95(total),
                          .passes(run_a_total, allow_partial=False), .draws frame with
                          per-status counts, .contract,
                          .trial_log_fields(run_a_result, allow_partial=False))
    random_index_candidates(inputs, run_a, seed=1) -> (candidates, RandomIndexMap frame)
    run_random_index(inputs, run_a, seed=1) -> RandomIndexResult (.mapping, .n_kept, .swapped,
                          .swapped_stats(run_a_result), .trial_log_fields(run_a_result))
    GateOffTable(table, gate); gate_off_tables(tables, gate) -> {key: GateOffTable}
    gate_off_candidates(inputs, spans) -> list[Candidate]
    run_gate_off(inputs, spans) -> GateOffResult
    BenchmarkRun                                     one backtest: .total_pnl_usd, .n_filled,
                                                     .expectancy_usd, .expectancy_r, .stats
    result_stats(res) -> dict                        n_candidates, n_filled, n_<status>, $, R
    trades_stats(trades); select_trades(trades, idents)   the same on a subset of trades
    check_run_a(inputs, run_a, run_a_result, spans)  run A, its result and spans belong together
    criterion7(run_a_result, placebo, *, allow_partial=False) -> Criterion7
    b2_verdict(run_a_result, b2) -> Comparison (.swapped_only); b3_verdict(...) -> Comparison
    run_benchmarks(inputs, run_a, run_a_result, spans, seeds=PLACEBO_SEEDS, b2_seed=1, *,
                   allow_partial=False) -> BenchmarkReport (.trial_log_rows(), .lines())

Guards: STRESS-d2 only; MR-TWAP-01 refused; every bars frame and feature frame must end before
cfg holdout.start (PermissionError at BenchmarkInputs construction, as the runner's
_check_pre_holdout), and no candidate or span may reach it; criterion 7 (criterion7,
PlaceboResult.passes / trial_log_fields with run A, run_benchmarks) refuses any seed set
other than 1..200 unless allow_partial=True (tests / tooling), whose verdict then reads
"NOT criterion 7 (partial seeds ...)" and is logged with criterion7_passed None.

SPEC NOTEs
1. "For each run-A trade" (B1) is read as each run-A CANDIDATE: run A's total $ P&L is the
   backtest of every candidate the strategy generated, so the placebo replicates every
   candidate and lets the same protocol skips, cancels, blocks (COMMON 8.8) and risk
   suppressions act on the placebo tickets. Taking only filled run-A trades would condition
   the placebo set on fills, which depend on prices after the real T.
2. "Uniformly from the same strategy window on the same local date": uniform over the M15
   decision times of that window on the candidate's exchange-local date at which the placebo
   levels are computable (M15 bar closing at T' present, strategy inputs finite). Times where
   no level exists are not drawn (they would be a no-ticket, which is not a random entry). A
   candidate with no eligible time is kept out of the draw and counted in `unmatched`.
3. RS-MOM-01 placebo geometry = the strategy's own: E = bid close at T', SL = E -/+ k ATRH(T')
   (ATR14_H1 by the COMMON 2 alignment at T'), TP = E +/- R k ATRH(T'), with the strategy's own
   rounding (rs_mom_01.round_away / round_toward), MARKET_BAND, time exit on the same date.
   The window is the strategy's window (10:30-13:30 New York, 10:00-15:00 Berlin).
4. SES-IBX-01 has a single decision time per date (the IB end), so its "window" is read as the
   order life of that date: T' from the IB end up to the last M15 close with T' + d bars <
   EXPIRY (d = the scenario delay, so the placebo can still be placed; COMMON 8.3 and the spec
   skip a placement at or after EXPIRY). STOP_ENTRY with the same fixed EXPIRY (12:00 New York
   / 11:30 Berlin) and time exit; every distance in the strategy's unit, the IB width W of that
   date (known from the IB end on). The placement of the bracket is an OPEN QUESTION escalated
   to the Quant (QA found the first reading skips far more placebo orders than run A, which
   thins the placebo distribution and makes criterion 7 easier). Two geometries exist and the
   placebo REFUSES to run until one is chosen (BenchmarkInputs.ses_placebo, or the module
   constant SES_PLACEBO_RULING once the Quant has ruled); nothing here picks one by default:
     SES_PLACEBO_CLOSE  the IB extreme is replaced by C' = bid close at T': trigger = C' +/-
                        0.05 W, SL = C' -/+ 0.5 W (D = 0.55 W), TP = C' +/- tp_ib_mult W. The
                        trigger is only 0.05 W from price, so at d = 2 many orders are skipped
                        (quote beyond CHASE LIMIT) where run A's needed a breakout first.
     SES_PLACEBO_SHIFT  the run-A bracket of that date translated by the price move C' - C_T
                        (C_T = bid close at the run-A T): trigger, midpoint stop and target keep
                        run A's distances to the reference price, in W units (trigger' = C' +
                        sg o W with o = sg (trigger_A - C_T) / W, SL' = trigger' -/+ 0.55 W,
                        TP' = trigger' +/- (tp_ib_mult - 0.05) W). The T' = IB-end draw equals
                        run A, as T' = T does for RS-MOM-01.
5. The 95th percentile is numpy's default (linear interpolation) over the 200 totals;
   criterion 7 needs run A's total strictly above it. Totals are $ at the scenario's contract
   (demo unless `contract` is given), summed over filled trades (0 for a draw with none).
6. B2: "whose window contains T" = the strategy's window for that index contains T on a valid
   trading date as known at T (COMMON 4.2.1 window set), and the strategy's own levels are
   computable on that index at T (B2 changes only the index, so the levels follow the
   strategy's rules on the new index: RS-MOM-01 E / ATRH of the new index; SES-IBX-01 the new
   index's IB, trigger, midpoint stop, target and EXPIRY). The strategy's filters (rank, gate,
   width, lockouts) are NOT re-applied: only the index is replaced. With no such index (GER40 is
   ranked alone in every window) the original candidate is kept and flagged. The draw is seeded
   (default 1); with one alternative (US30 <-> NAS100) it is deterministic.
7. "B2's mean" is read as B2's mean $ P&L per filled trade ($ expectancy); the comparison also
   shows totals and R. "Beat" = run A expectancy strictly greater. The literal verdict is over
   all candidates, kept ones included (GER40 is about half of RS-MOM-01's setups and is the
   same trade on both sides); the report and the B2 trial-log row add, for information only,
   the same $ expectancy on the swapped candidates (map flag "") where B2 actually differs.
8. B3 gate off: COMMON 4.2.3 forms E(T) from members whose regime status is OK AND that pass
   the gate; only the gate is removed. GateOffTable passes every status-OK row (threshold
   -inf for ACTIVE, +inf for CALM, read by strategies.context.regime_frame) and keeps
   UNAVAILABLE rows unavailable (COMMON 6.7: never guess). The strategy files are not edited;
   the result equals a strategy with regime_gate None (strategies.context SPEC NOTE).
9. Every benchmark candidate carries the run-A configuration's strategy_id so the engine runs
   it in single-strategy mode with the same concurrency rule; benchmark tickets have regime
   "n/a" (not used by the backtester) and a reason line naming the benchmark.
10. B1 draws each run-A candidate's T' independently (the literal "for each run-A trade").
   Two run-A candidates on one index and date (RS-MOM-01 allows two per side per day) can
   draw the same or overlapping T', and the engine then blocks the second ticket
   (blocked_open_position, COMMON 8.8) where run A had none. The draws are not altered; the
   per-draw filled / skipped / blocked / cancelled / suppressed counts are recorded next to
   run A's in the draws frame, the trial-log fields and the report, so a thinner placebo is
   visible.
11. B1 and B2 replicate the run-A candidates given; B3 rescans run A's spans. run_benchmarks
   therefore first checks (check_run_a) that a rescan of the spans with the real regime
   tables reproduces run A by (key, side, T), that run A's result holds exactly those
   candidates under the run-A strategy_id, and that the result is at the benchmarks' scenario
   and contract (COMMON 8.7: demo and competition $ are never mixed).
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, time, timedelta
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import (SINGLE, STATUS_BLOCKED, STATUS_CANCELLED, STATUS_FILLED,
                             STATUS_SKIPPED, STATUS_SUPPRESSED, run_backtest)
from backtest.results import BacktestResult
from backtest.trials import json_safe
from backtest.scenarios import SCENARIOS, Scenario
from core.calendar import exchange_for, local_to_utc
from core.config import tf_minutes
from data.mt5_adapter import SymbolSpec
from strategies.base import (MARKET_BAND, STOP_ENTRY, Candidate, time_exit_utc,
                             with_protocol_levels)
from strategies.context import _get_fx, decision_times, in_window
from strategies.rs_mom_01 import round_away, round_toward
from strategies.selection import BUY, config_order
from strategies.ses_ibx_01 import TRIGGER_BUFFER_FRAC, decision_clock, expiry_utc

STAGE = "benchmark"
BENCHMARK_SCENARIO = "STRESS-d2"
PLACEBO_SEEDS = tuple(range(1, 201))          # COMMON 10.6: 200 draws, seeds 1..200
PLACEBO_PERCENTILE = 95.0                     # COMMON 10.5 criterion 7
DROPPED_STRATEGIES = ("MR-TWAP-01",)          # amendment A3: never run
B1, B2, B3 = "B1-placebo", "B2-random-index", "B3-gate-off"
STATUSES = (STATUS_FILLED, STATUS_SKIPPED, STATUS_CANCELLED, STATUS_BLOCKED, STATUS_SUPPRESSED)

# SPEC NOTE 4: the SES-IBX-01 placebo bracket awaits the Quant's ruling. None = B1 for
# SES-IBX-01 refuses to run unless BenchmarkInputs.ses_placebo names a mode explicitly.
SES_PLACEBO_CLOSE = "close-reference"
SES_PLACEBO_SHIFT = "shifted-run-a"
SES_PLACEBO_MODES = (SES_PLACEBO_CLOSE, SES_PLACEBO_SHIFT)
SES_PLACEBO_RULING: str | None = SES_PLACEBO_SHIFT   # COMMON 12D, amendment A4.2


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    if t.tzinfo is None:
        raise ValueError("times must be tz-aware UTC")
    return t.tz_convert("UTC").as_unit("ns")


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkInputs:
    """Everything a backtest of run A needs, shared by B1-B3.

    strategy: the run-A configuration (a WindowedStrategy instance with run A's params).
    features: FeatureService (or {key: IndexFeatures}) built on the same bars.
    regime_tables: {key: RegimeTable} as used by run A (B3 wraps them).
    bars: per-index M15 bid bars as given to run_backtest (BarStore.load output).
    scenario: STRESS-d2 (COMMON 10.6 benchmarks run there; another scenario raises).
    contract: USD/pt table (None = demo, as run_backtest); contract_name labels it and must
        match run A's result (COMMON 8.7).
    ses_placebo: SES-IBX-01 B1 geometry (SPEC NOTE 4); None = SES_PLACEBO_RULING."""
    cfg: dict
    strategy: Any
    features: Any
    regime_tables: Mapping[str, Any]
    bars: Mapping[str, pd.DataFrame]
    specs: Mapping[str, SymbolSpec]
    scenario: Scenario = SCENARIOS[BENCHMARK_SCENARIO]
    contract: Mapping[str, float] | None = None
    contract_name: str = "demo"
    ses_placebo: str | None = None

    def __post_init__(self) -> None:
        sid = getattr(self.strategy, "id", "")
        if sid in DROPPED_STRATEGIES:
            raise ValueError(f"{sid} was DROPPED by amendment A3 (COMMON 12C) and is never run")
        if self.scenario != SCENARIOS[BENCHMARK_SCENARIO]:
            raise ValueError(f"benchmarks run under {BENCHMARK_SCENARIO} (task / COMMON 10.6), "
                             f"not {self.scenario.name}")
        if self.ses_placebo is not None and self.ses_placebo not in SES_PLACEBO_MODES:
            raise ValueError(f"ses_placebo must be one of {SES_PLACEBO_MODES}, "
                             f"not {self.ses_placebo!r}")
        if self.contract is None and self.contract_name != "demo":
            raise ValueError(f"contract_name {self.contract_name!r} needs its USD/pt table; "
                             "contract None is the demo contract")
        self._check_frames_pre_holdout()

    def _holdout_start(self) -> pd.Timestamp | None:
        start = (self.cfg.get("holdout") or {}).get("start")
        if start is None:
            return None
        h = pd.Timestamp(start)
        return _utc(h.tz_localize("UTC") if h.tzinfo is None else h)

    def _check_frames_pre_holdout(self) -> None:
        """The holdout stays locked until GATE 3 (PROMPT.md MANDATORY VALIDATION 1): the bars
        backtested on and the feature frames the levels are read from must all end before
        holdout.start, as backtest.runner._check_pre_holdout requires of the runner's bars.
        A frame that only reaches the holdout could still feed exits, end-of-data handling or
        MTM even when every ticket is pre-holdout, so it is refused outright."""
        hs = self._holdout_start()
        if hs is None:
            return
        frames: list[tuple[str, pd.DataFrame]] = [(f"{k} M15 bars", df)
                                                  for k, df in self.bars.items()]
        keys = set(self.bars) | {i["key"] for i in self.cfg.get("instruments", [])}
        for k in sorted(keys):
            fx = self.fx(k)
            for tf in ("m15", "h1"):
                df = getattr(fx, tf, None) if fx is not None else None
                if isinstance(df, pd.DataFrame):
                    frames.append((f"{k} {tf.upper()} features", df))
        for what, df in frames:
            if len(df) and "time_utc" in df and \
                    pd.Timestamp(pd.to_datetime(df["time_utc"], utc=True).max()) >= hs:
                raise PermissionError(f"{what} reach the final holdout (from {hs}); benchmarks "
                                      "only run on pre-holdout data (GATE 3 not approved)")

    @property
    def strategy_id(self) -> str:
        return self.strategy.id

    @property
    def bar_minutes(self) -> int:
        return tf_minutes(self.cfg["timeframes"]["entry"])

    def fx(self, key: str) -> Any:
        return _get_fx(self.features, key)

    def check_pre_holdout(self, times: Iterable) -> None:
        """The holdout stays locked until GATE 3: no benchmark ticket may be on or after it."""
        h = self._holdout_start()
        if h is None:
            return
        late =[t for t in times if t is not None and _utc(t) >= h]
        if late:
            raise ValueError(f"benchmark input at {late[0]} is inside the locked holdout "
                             f"(from {h}); the holdout is never used here")

    def backtest(self, candidates: Sequence[Candidate]) -> BacktestResult:
        self.check_pre_holdout(c.signal_time for c in candidates)
        return run_backtest([(self.strategy_id, c) for c in candidates], self.bars, self.cfg,
                            self.scenario, self.specs, contract=self.contract,
                            contract_name=self.contract_name, mode=SINGLE)


def _check_run_a(inputs: BenchmarkInputs, run_a: Sequence[Candidate]) -> list[Candidate]:
    out = list(run_a)
    for c in out:
        if c.signal_time is None or c.order_type is None:
            raise ValueError(f"run-A candidate {c.key} {c.side} lacks signal_time/order_type")
    inputs.check_pre_holdout(c.signal_time for c in out)
    return out


# ---------------------------------------------------------------------------------------------
# Per-strategy geometry
# ---------------------------------------------------------------------------------------------

Levels = tuple[Candidate | None, str]


class PlaceboGeometry:
    """Strategy-specific level rules for B1 and B2 (see SPEC NOTEs 3, 4, 6)."""
    strategy_id = ""

    def __init__(self, inputs: BenchmarkInputs):
        self.inputs = inputs
        self.cfg = inputs.cfg
        self.params = inputs.strategy.params
        self.windows = inputs.strategy.windows(self.params)
        self.bar = pd.Timedelta(minutes=inputs.bar_minutes)

    # hooks
    def strategy_levels(self, key: str, side: str, T: pd.Timestamp) -> Levels:
        raise NotImplementedError

    def placebo_levels(self, key: str, side: str, T: pd.Timestamp,
                       source: Candidate | None = None) -> Levels:
        """B1 ticket at T'; `source` is the run-A candidate it replaces (needed by geometries
        defined relative to run A)."""
        raise NotImplementedError

    def placebo_times(self, c: Candidate) -> pd.DatetimeIndex:
        raise NotImplementedError

    def check_placebo_ready(self) -> None:
        """Raise when B1 cannot run yet (a pending ruling); B2 and B3 are unaffected."""

    # shared helpers
    def local_date(self, key: str, T) -> date:
        return _utc(T).tz_convert(exchange_for(self.cfg, key).tz).date()

    def day_bounds(self, key: str, d: date) -> tuple[pd.Timestamp, pd.Timestamp]:
        tz = exchange_for(self.cfg, key).tz
        a = local_to_utc(d, time(0), tz).as_unit("ns")
        b = local_to_utc(d + timedelta(days=1), time(0), tz).as_unit("ns")
        return a, b

    def window_valid(self, key: str, T: pd.Timestamp) -> bool:
        """SPEC NOTE 6: key's strategy window contains T on a valid trading date known at T."""
        w = self.windows.get(key)
        if w is None or not in_window(self.cfg, key, w, T):
            return False
        fx = self.inputs.fx(key)
        return fx is not None and bool(fx.is_valid_date(self.local_date(key, T), T))

    def _close(self, key: str, T: pd.Timestamp) -> tuple[float, float, str]:
        """(bid close, ATR14_M15) of the M15 bar closing at T, or NaNs + cause."""
        fx = self.inputs.fx(key)
        if fx is None:
            return math.nan, math.nan, "no features for index"
        p = fx.m15_pos(T)
        if p < 0:
            return math.nan, math.nan, "no M15 bar closing at T"
        row = fx.m15.iloc[p]
        e = float(row["close"])
        return e, float(row["atr14"]), "" if math.isfinite(e) else "close NaN"

    def _finish(self, c: Candidate) -> Candidate:
        return with_protocol_levels(c, self.cfg)


class RsMom01Geometry(PlaceboGeometry):
    """SPEC NOTE 3: the placebo levels are the strategy's levels at T'."""
    strategy_id = "RS-MOM-01"

    def _atrh(self, key: str, T: pd.Timestamp) -> tuple[float, str]:
        al = self.inputs.fx(key).h1_at(T)
        if not al.available:
            return math.nan, f"ATR14_H1 unavailable: {al.cause}"
        v = float(al.row["atr14"])
        if not (math.isfinite(v) and v > 0):
            return math.nan, "ATR14_H1 NaN or not positive"
        return v, ""

    def strategy_levels(self, key: str, side: str, T: pd.Timestamp, *,
                        tag: str = "") -> Levels:
        T = _utc(T)
        e, atr15, cause = self._close(key, T)
        if cause:
            return None, cause
        atrh, cause = self._atrh(key, T)
        if cause:
            return None, cause
        k, R = float(self.params["k"]), float(self.params["R"])
        sg = 1.0 if side == BUY else -1.0
        d = k * atrh
        stop = round_away(e - sg * d, e)
        tp = round_toward(e + sg * R * d, e)
        c = Candidate(key, side, e, stop, tp, atr15,
                      f"{tag or 'benchmark'} {self.strategy_id} {side} {key} at {T:%Y-%m-%d %H:%M}"
                      f" UTC (k {k}, R {R}, ATRH {atrh:.2f})",
                      signal_time=T, order_type=MARKET_BAND, expires_utc=None,
                      time_exit_utc=time_exit_utc(self.cfg, key, T))
        return self._finish(c), ""

    def placebo_levels(self, key: str, side: str, T: pd.Timestamp,
                       source: Candidate | None = None) -> Levels:
        return self.strategy_levels(key, side, T, tag=B1)

    def placebo_times(self, c: Candidate) -> pd.DatetimeIndex:
        a, b = self.day_bounds(c.key, self.local_date(c.key, c.signal_time))
        return decision_times(self.cfg, c.key, self.windows[c.key], a, b)


class SesIbx01Geometry(PlaceboGeometry):
    """SPEC NOTE 4 (placebo: SES_PLACEBO_CLOSE or SES_PLACEBO_SHIFT, pending the Quant's
    ruling) and SPEC NOTE 6 (B2: the strategy's own IB levels on the other index)."""
    strategy_id = "SES-IBX-01"

    def __init__(self, inputs: BenchmarkInputs):
        super().__init__(inputs)
        self.mode = inputs.ses_placebo or SES_PLACEBO_RULING

    def check_placebo_ready(self) -> None:
        if self.mode not in SES_PLACEBO_MODES:
            raise ValueError("SES-IBX-01 B1 placebo geometry awaits the Quant's ruling (SPEC "
                             f"NOTE 4): set BenchmarkInputs.ses_placebo to one of "
                             f"{SES_PLACEBO_MODES} as ruled; no default is picked here")

    def _ib(self, key: str, T: pd.Timestamp):
        fx = self.inputs.fx(key)
        if fx is None:
            return None, "no features for index"
        ib = fx.ib(self.local_date(key, T), int(self.params["ib_minutes"]), T)
        if not ib.available:
            return None, ib.cause or "IB incomplete"
        if not math.isfinite(ib.w):
            return None, "IB width NaN"
        return ib, ""

    def _make(self, key: str, side: str, T: pd.Timestamp, ref_hi: float, ref_lo: float,
              w: float, atr15: float, reason: str) -> Candidate:
        b = TRIGGER_BUFFER_FRAC * w
        sl = (ref_hi + ref_lo) / 2.0
        m = float(self.params["tp_ib_mult"])
        if side == BUY:
            trigger, tp = ref_hi + b, ref_hi + m * w
        else:
            trigger, tp = ref_lo - b, ref_lo - m * w
        c = Candidate(key, side, float(trigger), float(sl), float(tp), atr15, reason,
                      signal_time=T, order_type=STOP_ENTRY,
                      expires_utc=expiry_utc(self.cfg, key, T),
                      time_exit_utc=time_exit_utc(self.cfg, key, T))
        return self._finish(c)

    def strategy_levels(self, key: str, side: str, T: pd.Timestamp) -> Levels:
        T = _utc(T)
        _, atr15, cause = self._close(key, T)
        if cause:
            return None, cause
        ib, cause = self._ib(key, T)
        if cause:
            return None, cause
        return self._make(key, side, T, ib.ibh, ib.ibl, ib.w, atr15,
                          f"benchmark {self.strategy_id} {side} STOP {key} at "
                          f"{T:%Y-%m-%d %H:%M} UTC (IB {ib.ibl:.2f}-{ib.ibh:.2f})"), ""

    def placebo_levels(self, key: str, side: str, T: pd.Timestamp,
                       source: Candidate | None = None) -> Levels:
        self.check_placebo_ready()
        T = _utc(T)
        e, atr15, cause = self._close(key, T)
        if cause:
            return None, cause
        ib, cause = self._ib(key, T)
        if cause:
            return None, cause
        if self.mode == SES_PLACEBO_CLOSE:
            # The IB extreme is replaced by C'; the IB of width W around it is [C' - W, C'] for
            # a long (extreme = high) and [C', C' + W] for a short (extreme = low), so trigger,
            # midpoint stop and target keep the strategy's offsets in units of W.
            hi, lo = (e, e - ib.w) if side == BUY else (e + ib.w, e)
            note = f"reference {e:.2f}, W {ib.w:.2f}"
        else:
            # SES_PLACEBO_SHIFT: the run-A IB of that date moved by the price change C' - C_T,
            # so every level keeps run A's distance to the reference close (in W units).
            if source is None or source.signal_time is None:
                return None, "shifted geometry needs the run-A candidate"
            if source.key != key or self.local_date(key, source.signal_time) != \
                    self.local_date(key, T):
                return None, "run-A candidate is on another index or date"
            c0, _, cause = self._close(key, _utc(source.signal_time))
            if cause:
                return None, f"run-A reference close: {cause}"
            shift = e - c0
            hi, lo = ib.ibh + shift, ib.ibl + shift
            note = f"run-A IB {ib.ibl:.2f}-{ib.ibh:.2f} shifted {shift:+.2f}, W {ib.w:.2f}"
        return self._make(key, side, T, hi, lo, ib.w, atr15,
                          f"{B1} {self.strategy_id} {side} STOP {key} at {T:%Y-%m-%d %H:%M} "
                          f"UTC ({self.mode}: {note})"), ""

    def placebo_times(self, c: Candidate) -> pd.DatetimeIndex:
        d = self.local_date(c.key, c.signal_time)
        tz = exchange_for(self.cfg, c.key).tz
        first = local_to_utc(d, decision_clock(self.cfg, c.key, int(self.params["ib_minutes"])),
                             tz).as_unit("ns")
        expiry = expiry_utc(self.cfg, c.key, c.signal_time)
        last = expiry - self.inputs.scenario.d * self.bar      # T' + d bars < EXPIRY
        if last <= first:
            return pd.DatetimeIndex([], tz="UTC").as_unit("ns")
        return pd.date_range(first, last, freq=self.bar, inclusive="left").as_unit("ns")


GEOMETRIES: dict[str, type[PlaceboGeometry]] = {
    RsMom01Geometry.strategy_id: RsMom01Geometry,
    SesIbx01Geometry.strategy_id: SesIbx01Geometry,
}


def geometry_for(inputs: BenchmarkInputs) -> PlaceboGeometry:
    sid = inputs.strategy_id
    if sid not in GEOMETRIES:
        raise ValueError(f"no benchmark geometry for {sid!r}; defined for {sorted(GEOMETRIES)}")
    return GEOMETRIES[sid](inputs)


# ---------------------------------------------------------------------------------------------
# Results shared by B1-B3
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkRun:
    """One benchmark backtest."""
    benchmark: str
    seed: int | None
    candidates: tuple[Candidate, ...]
    result: BacktestResult

    @property
    def filled(self) -> pd.DataFrame:
        return self.result.filled

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def n_filled(self) -> int:
        return int(len(self.filled))

    @property
    def total_pnl_usd(self) -> float:
        return float(self.filled["pnl_usd"].sum()) if self.n_filled else 0.0

    @property
    def expectancy_usd(self) -> float:
        return float(self.filled["pnl_usd"].mean()) if self.n_filled else math.nan

    @property
    def expectancy_r(self) -> float:
        return float(self.filled["r_multiple"].mean()) if self.n_filled else math.nan

    @property
    def stats(self) -> dict[str, float]:
        return result_stats(self.result)


def result_stats(res: BacktestResult) -> dict[str, float]:
    """Trade counts by engine status (n_filled, n_skipped, n_cancelled, n_blocked,
    n_suppressed; n_candidates = all rows) and the $ / R figures of the filled trades."""
    return trades_stats(res.trades)


def trades_stats(trades: pd.DataFrame) -> dict[str, float]:
    """result_stats on any subset of a result's trades frame."""
    f = trades[trades["status"] == STATUS_FILLED]
    n = int(len(f))
    st = trades["status"].value_counts()
    out: dict[str, float] = {"n_candidates": int(len(trades))}
    out.update({f"n_{s}": int(st.get(s, 0)) for s in STATUSES})
    out.update({"total_pnl_usd": float(f["pnl_usd"].sum()) if n else 0.0,
                "expectancy_usd": float(f["pnl_usd"].mean()) if n else math.nan,
                "expectancy_r": float(f["r_multiple"].mean()) if n else math.nan})
    return out


def select_trades(trades: pd.DataFrame, idents: Iterable[tuple[str, str, Any]]) -> pd.DataFrame:
    """The rows of a trades frame whose (key, side, signal_time) is in `idents`, matched as a
    multiset (each ident takes at most as many rows as it occurs). The engine sorts trades by
    time, so rows are found by identity, not by the candidate order."""
    want = _ident_set(idents)
    keep = []
    for i, ident in enumerate(zip(trades["key"], trades["side"],
                                  (_utc(t).value for t in trades["signal_time"]))):
        if want[ident] > 0:
            want[ident] -= 1
            keep.append(i)
    return trades.iloc[keep]


def _usd(x: float) -> str:
    return f"${x:.2f}" if math.isfinite(x) else "n/a (no fills)"


def _counts_text(s: Mapping[str, float]) -> str:
    return (f"{int(s['n_filled'])} filled / {int(s['n_candidates'])} candidates, "
            f"{int(s['n_skipped'])} skipped, {int(s['n_cancelled'])} cancelled, "
            f"{int(s['n_blocked'])} blocked, {int(s['n_suppressed'])} suppressed")


def _scenario_key(scenario: str, contract: str) -> str:
    """As the runner keys TrialRecord.scenarios: the name at demo, 'name@contract' otherwise."""
    return scenario if contract == "demo" else f"{scenario}@{contract}"


def _log_fields(name: str, scenario: str, contract: str, summary: str, notes: str,
                metrics: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """One stage-`benchmark` row for backtest.trials.TrialRecord (module docstring)."""
    return {"stage": STAGE, "variant": name, "scenario": scenario, "contract": contract,
            "scenario_key": _scenario_key(scenario, contract), "result_summary": summary,
            "notes": f"{notes}; not counted in N (COMMON 10.6)",
            "metrics": json_safe(dict(metrics)), "extra": json_safe(dict(extra))}


# ---------------------------------------------------------------------------------------------
# B1 placebo
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PlaceboOptions:
    """The eligible placebo tickets for one run-A candidate (one per eligible T')."""
    source: Candidate
    options: tuple[Candidate, ...]
    window_times: int                 # decision times in the window before the level check


def placebo_options(inputs: BenchmarkInputs, run_a: Sequence[Candidate]
                    ) -> list[PlaceboOptions]:
    """SPEC NOTEs 1-4: per run-A candidate, every computable placebo ticket on its date."""
    geo = geometry_for(inputs)
    geo.check_placebo_ready()
    cache: dict[tuple, tuple[Candidate, ...]] = {}
    out = []
    for c in _check_run_a(inputs, run_a):
        times = geo.placebo_times(c)
        k = (c.key, c.side, _utc(c.signal_time).value, tuple(times.asi8))
        if k not in cache:
            opts = []
            for t in times:
                pc, _ = geo.placebo_levels(c.key, c.side, t, source=c)
                if pc is not None:
                    opts.append(pc)
            cache[k] = tuple(opts)
        out.append(PlaceboOptions(c, cache[k], len(times)))
    return out


@dataclass(frozen=True)
class PlaceboDraw:
    seed: int
    candidates: tuple[Candidate, ...]
    unmatched: int                    # run-A candidates without an eligible placebo time


def placebo_draw(options: Sequence[PlaceboOptions], seed: int) -> PlaceboDraw:
    """One seeded draw: for each run-A candidate in order, a uniform choice among its options
    (numpy default_rng(seed); a candidate without options consumes no random number)."""
    rng = np.random.default_rng(int(seed))
    picks, unmatched = [], 0
    for o in options:
        if not o.options:
            unmatched += 1
            continue
        picks.append(o.options[int(rng.integers(len(o.options)))])
    return PlaceboDraw(int(seed), tuple(picks), unmatched)


DRAW_COLUMNS = ["seed", "n_candidates", *[f"n_{s}" for s in STATUSES], "total_pnl_usd",
                "expectancy_usd", "expectancy_r", "unmatched"]


@dataclass(frozen=True)
class PlaceboResult:
    """Distribution of B1 total $ P&L over the seeds (SPEC NOTE 5), with each draw's trade
    counts by status (SPEC NOTE 10) so a thinner placebo than run A is visible."""
    strategy_id: str
    scenario: str
    contract: str
    draws: pd.DataFrame        # DRAW_COLUMNS, one row per seed
    n_run_a: int
    geometry: str = ""         # SES-IBX-01: the ruled SES_PLACEBO_* mode
    runs: tuple[BenchmarkRun, ...] = field(default=(), repr=False)

    @property
    def totals(self) -> np.ndarray:
        return self.draws["total_pnl_usd"].to_numpy(dtype=float)

    @property
    def p95(self) -> float:
        return float(np.percentile(self.totals, PLACEBO_PERCENTILE))

    def quantiles(self, qs: Sequence[float] = (5, 50, 95)) -> dict[float, float]:
        return {q: float(np.percentile(self.totals, q)) for q in qs}

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.draws["seed"])

    @property
    def complete(self) -> bool:
        """The draws are exactly the pre-registered seeds 1..200 in order (COMMON 10.6 B1)."""
        return self.seeds == PLACEBO_SEEDS

    def check_complete(self, allow_partial: bool = False) -> None:
        """Criterion 7 is defined on the 200 pre-registered draws only; any other seed set
        raises unless the caller (tests, tooling) passes allow_partial=True, and its verdict
        is then labelled 'NOT criterion 7 (partial seeds)', never PASS / FAIL."""
        if not self.complete and not allow_partial:
            s = self.seeds
            raise ValueError(f"criterion 7 needs the pre-registered placebo seeds 1..200 "
                             f"(COMMON 10.6 B1); got {len(s)} seeds "
                             f"{s[:3]}{'...' if len(s) > 3 else ''}. Pass allow_partial=True "
                             "only outside the Run phase")

    def above_p95(self, run_a_total: float) -> bool:
        """The raw comparison: run A total $ P&L strictly above the 95th percentile."""
        return bool(run_a_total > self.p95)

    def passes(self, run_a_total: float, *, allow_partial: bool = False) -> bool:
        """COMMON 10.5 criterion 7 (raises on a seed set other than 1..200 unless
        allow_partial; a partial result is a comparison, not the criterion)."""
        self.check_complete(allow_partial)
        return self.above_p95(run_a_total)

    def verdict_word(self, run_a_total: float, allow_partial: bool = False) -> str:
        if self.complete:
            return "PASS" if self.above_p95(run_a_total) else "FAIL"
        self.check_complete(allow_partial)
        side = "above" if self.above_p95(run_a_total) else "not above"
        return f"NOT criterion 7 (partial seeds: {len(self.draws)} of {len(PLACEBO_SEEDS)}; " \
               f"run A {side} p95)"

    def count_summary(self) -> dict[str, dict[str, float]]:
        """Per status count: mean, min, median, max over the draws."""
        out = {}
        for col in ["n_candidates", *[f"n_{s}" for s in STATUSES]]:
            x = self.draws[col].to_numpy(dtype=float)
            out[col] = {"mean": float(x.mean()), "min": float(x.min()),
                        "median": float(np.median(x)), "max": float(x.max())}
        return out

    def counts_text(self, run_a_stats: Mapping[str, float] | None = None) -> str:
        cs = self.count_summary()

        def rng(col: str) -> str:
            c = cs[col]
            return f"{c['mean']:.1f} ({c['min']:.0f}-{c['max']:.0f})"
        s = (f"placebo per draw: filled {rng('n_filled')}, skipped {rng('n_skipped')}, "
             f"cancelled {rng('n_cancelled')}, blocked {rng('n_blocked')}, suppressed "
             f"{rng('n_suppressed')} of {rng('n_candidates')} tickets")
        if run_a_stats is not None:
            s += f"; run A: {_counts_text(run_a_stats)}"
        return s

    def trial_log_fields(self, run_a_result: BacktestResult | None = None, *,
                         allow_partial: bool = False) -> dict[str, Any]:
        """The B1 trial-log row; with run_a_result, the criterion 7 verdict and run A's counts
        (the result must be at this scenario and contract, and the draws the seeds 1..200
        unless allow_partial, whose verdict reads 'NOT criterion 7 (partial seeds ...)')."""
        q = self.quantiles()
        s = (f"{B1} {self.strategy_id} {self.scenario}: {len(self.draws)} draws, total $ "
             f"p5/p50/p95 {q[5]:.2f}/{q[50]:.2f}/{q[95]:.2f}")
        a = None
        if run_a_result is not None:
            _check_same(run_a_result, self.scenario, self.contract)
            a = result_stats(run_a_result)
            word = self.verdict_word(a["total_pnl_usd"], allow_partial)
            s += (f"; run A total {a['total_pnl_usd']:.2f} -> "
                  f"{word if not self.complete else 'criterion 7 ' + word}")
        um = int(self.draws["unmatched"].max()) if len(self.draws) else 0
        geo = f", geometry {self.geometry}" if self.geometry else ""
        notes = (f"seeds {int(self.draws['seed'].min())}..{int(self.draws['seed'].max())}, "
                 f"{self.n_run_a} run-A candidates, {um} without an eligible placebo time{geo}; "
                 f"{self.counts_text(a)}")
        metrics = {"scenario": self.scenario, "contract": self.contract,
                   "n_draws": int(len(self.draws)), "p95_total_usd": self.p95,
                   "quantiles_total_usd": {str(k): v for k, v in q.items()},
                   "mean_total_usd": float(self.totals.mean()),
                   "counts": self.count_summary()}
        extra = {"benchmark": B1, "seeds": [int(x) for x in self.draws["seed"]],
                 "percentile": PLACEBO_PERCENTILE, "n_run_a": self.n_run_a,
                 "geometry": self.geometry, "draws": self.draws.to_dict(orient="list")}
        extra["complete_seeds"] = self.complete
        if a is not None:
            # None = not criterion 7 (partial seeds); the raw comparison is kept beside it
            extra.update(run_a=a, run_a_above_p95=self.above_p95(a["total_pnl_usd"]),
                         criterion7_passed=(self.passes(a["total_pnl_usd"])
                                            if self.complete else None))
        return _log_fields(B1, self.scenario, self.contract, s, notes, metrics, extra)


def run_placebo(inputs: BenchmarkInputs, run_a: Sequence[Candidate],
                seeds: Iterable[int] = PLACEBO_SEEDS, *,
                keep_results: bool = False) -> PlaceboResult:
    """B1: one backtest per seed under STRESS-d2 (COMMON 10.6)."""
    opts = placebo_options(inputs, run_a)
    rows, runs = [], []
    for s in seeds:
        dr = placebo_draw(opts, s)
        res = inputs.backtest(dr.candidates)
        st = result_stats(res)
        rows.append({"seed": dr.seed, **st, "unmatched": dr.unmatched})
        if keep_results:
            runs.append(BenchmarkRun(B1, dr.seed, dr.candidates, res))
    draws = pd.DataFrame(rows, columns=DRAW_COLUMNS)
    if draws.empty:
        raise ValueError("no placebo seeds given")
    geo = geometry_for(inputs)
    return PlaceboResult(inputs.strategy_id, inputs.scenario.name, inputs.contract_name, draws,
                         len(opts), getattr(geo, "mode", "") or "", tuple(runs))


# ---------------------------------------------------------------------------------------------
# B2 random index
# ---------------------------------------------------------------------------------------------

def random_index_candidates(inputs: BenchmarkInputs, run_a: Sequence[Candidate],
                            seed: int = 1) -> tuple[list[Candidate], pd.DataFrame]:
    """SPEC NOTE 6. Returns the B2 candidates and a map frame (signal_time, side, key,
    b2_key, n_alternatives, flag); flag is "" when replaced, else why the original was kept."""
    geo = geometry_for(inputs)
    rng = np.random.default_rng(int(seed))
    order = config_order(inputs.cfg)
    out, rows = [], []
    for c in _check_run_a(inputs, run_a):
        T = _utc(c.signal_time)
        alts: list[Candidate] = []
        for k in order:
            if k == c.key or not geo.window_valid(k, T):
                continue
            nc, _ = geo.strategy_levels(k, c.side, T)
            if nc is not None:
                alts.append(replace(nc, reason=f"{B2} ({c.key} -> {k}) " + nc.reason))
        if alts:
            pick = alts[int(rng.integers(len(alts)))]
            out.append(pick)
            rows.append((T, c.side, c.key, pick.key, len(alts), ""))
        else:
            out.append(c)
            rows.append((T, c.side, c.key, c.key, 0,
                         "kept: no other index whose window contains T with computable levels"))
    frame = pd.DataFrame(rows, columns=["signal_time", "side", "key", "b2_key",
                                        "n_alternatives", "flag"])
    return out, frame


@dataclass(frozen=True)
class RandomIndexResult:
    run: BenchmarkRun
    mapping: pd.DataFrame

    @property
    def n_kept(self) -> int:
        return int((self.mapping["flag"] != "").sum())

    @property
    def swapped(self) -> pd.DataFrame:
        """Mapping rows whose index was replaced (flag == ""): the only rows on which B2
        differs from run A, so the only part that tests the relative-strength selection."""
        m = self.mapping
        if m.empty or "flag" not in m:
            return m.iloc[0:0]
        return m[m["flag"] == ""]

    def swapped_stats(self, run_a_result: BacktestResult | None = None
                      ) -> dict[str, dict[str, float]]:
        """result_stats restricted to the swapped candidates: B2 on (b2_key, side, T) and, with
        run_a_result, run A on the original (key, side, T) of the same rows. Kept candidates
        (GER40 always, a US index with no level on the other) are identical trades on both
        sides and are left out of this figure only (the literal B2 verdict keeps them)."""
        sw = self.swapped
        rows = list(zip(sw.get("signal_time", []), sw.get("side", []), sw.get("key", []),
                        sw.get("b2_key", [])))
        out = {"b2": trades_stats(select_trades(self.run.result.trades,
                                                ((b, s, t) for t, s, _, b in rows)))}
        if run_a_result is not None:
            out["run_a"] = trades_stats(select_trades(run_a_result.trades,
                                                      ((k, s, t) for t, s, k, _ in rows)))
        return out

    def trial_log_fields(self, run_a_result: BacktestResult | None = None) -> dict[str, Any]:
        """The B2 trial-log row; with run_a_result (same scenario and contract) the swapped-only
        comparison with run A is added to the summary and to extra['swapped_only']."""
        r = self.run
        st = r.stats
        res = r.result
        if run_a_result is not None:
            _check_same(run_a_result, res.scenario.name, res.contract)
        sw = self.swapped_stats(run_a_result)
        s = (f"{B2} seed {r.seed}: {r.n_filled} filled, total $ {r.total_pnl_usd:.2f}, "
             f"expectancy $ {r.expectancy_usd:.2f}, R {r.expectancy_r:.3f}")
        if "run_a" in sw:
            s += (f"; swapped only ({len(self.swapped)} candidates): run A expectancy "
                  f"{_usd(sw['run_a']['expectancy_usd'])} vs B2 "
                  f"{_usd(sw['b2']['expectancy_usd'])}")
        return _log_fields(B2, res.scenario.name, res.contract, s,
                           f"{len(self.mapping)} run-A candidates, {self.n_kept} kept on their "
                           f"own index (no other index in window); {_counts_text(st)}",
                           st, {"benchmark": B2, "seed": r.seed, "n_kept": self.n_kept,
                                "n_swapped": int(len(self.swapped)), "swapped_only": sw,
                                "swaps": self.mapping.groupby(["key", "b2_key"]).size()
                                .rename("n").reset_index().to_dict(orient="records")})


def run_random_index(inputs: BenchmarkInputs, run_a: Sequence[Candidate],
                     seed: int = 1) -> RandomIndexResult:
    cands, frame = random_index_candidates(inputs, run_a, seed)
    res = inputs.backtest(cands)
    return RandomIndexResult(BenchmarkRun(B2, int(seed), tuple(cands), res), frame)


# ---------------------------------------------------------------------------------------------
# B3 gate off
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GateOffTable:
    """A regime table whose gate always passes for status-OK rows (SPEC NOTE 8). at_many is
    the wrapped table's, unchanged (status, cause, V_T, tag, label, probability); only the
    threshold that strategies.context.regime_frame compares V_T with is moved to -inf (ACTIVE
    gate, V_T >= threshold) or +inf (CALM gate, V_T < threshold)."""
    table: Any
    gate: str

    def __post_init__(self) -> None:
        if self.gate not in ("ACTIVE", "CALM"):
            raise ValueError(f"gate must be ACTIVE or CALM, not {self.gate!r}")

    @property
    def threshold(self) -> float:
        return -math.inf if self.gate == "ACTIVE" else math.inf

    def at_many(self, times, live_rule: bool = True) -> pd.DataFrame:
        return self.table.at_many(times, live_rule)

    def __getattr__(self, name: str) -> Any:          # anything else: the wrapped table's
        if name in ("table", "gate"):
            raise AttributeError(name)
        return getattr(self.table, name)


def gate_off_tables(tables: Mapping[str, Any], gate: str) -> dict[str, Any]:
    """Every table wrapped; a missing table (None) stays None: no regime, no candidate."""
    return {k: (None if t is None else GateOffTable(t, gate)) for k, t in tables.items()}


def gate_off_candidates(inputs: BenchmarkInputs,
                        spans: Sequence[tuple[Any, Any]]) -> list[Candidate]:
    """The run-A strategy's scan with the gate off, over run A's spans [start, end) (the
    concatenated test windows), in time order."""
    strat = inputs.strategy
    gate = getattr(strat, "regime_gate", None)
    if gate is None:
        raise ValueError(f"{strat.id} has no regime gate to switch off")
    tables = gate_off_tables(inputs.regime_tables, gate)
    out: list[Candidate] = []
    for a, b in spans:
        a, b = _utc(a), _utc(b)
        inputs.check_pre_holdout([b - pd.Timedelta(1)])
        out.extend(strat.scan(inputs.features, tables, a, b))
    return out


@dataclass(frozen=True)
class GateOffResult:
    run: BenchmarkRun

    spans: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = ()

    def trial_log_fields(self) -> dict[str, Any]:
        r = self.run
        st = r.stats
        res = r.result
        return _log_fields(B3, res.scenario.name, res.contract,
                           f"{B3}: {r.n_candidates} candidates, {r.n_filled} filled, total "
                           f"$ {r.total_pnl_usd:.2f}, expectancy $ {r.expectancy_usd:.2f}, "
                           f"R {r.expectancy_r:.3f}",
                           "regime gate always passes for status-OK rows; status filter kept; "
                           f"{_counts_text(st)}",
                           st, {"benchmark": B3, "spans": [[a.isoformat(), b.isoformat()]
                                                           for a, b in self.spans]})


def _spans(spans: Sequence[tuple[Any, Any]]) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    return tuple((_utc(a), _utc(b)) for a, b in spans)


def run_gate_off(inputs: BenchmarkInputs, spans: Sequence[tuple[Any, Any]]) -> GateOffResult:
    cands = gate_off_candidates(inputs, spans)
    return GateOffResult(BenchmarkRun(B3, None, tuple(cands), inputs.backtest(cands)),
                         _spans(spans))


# ---------------------------------------------------------------------------------------------
# Comparisons with run A (COMMON 10.5 criterion 7, 10.6 report sentences)
# ---------------------------------------------------------------------------------------------

def _check_same(run_a_result: BacktestResult, scenario: str, contract: str) -> None:
    """Run A and the benchmark must be at the same scenario AND contract (COMMON 8.7)."""
    if run_a_result.scenario.name != scenario:
        raise ValueError(f"run A result is {run_a_result.scenario.name}; benchmarks compare "
                         f"with run A under {scenario}")
    if run_a_result.contract != contract:
        raise ValueError(f"run A result is at the {run_a_result.contract} contract, the "
                         f"benchmark at {contract}; $ figures of two contracts are never "
                         "compared (COMMON 8.7)")


def _ident_set(rows: Iterable[tuple[str, str, Any]]) -> Counter:
    return Counter((k, s, _utc(t).value) for k, s, t in rows)


def check_run_a(inputs: BenchmarkInputs, run_a: Sequence[Candidate],
                run_a_result: BacktestResult, spans: Sequence[tuple[Any, Any]]) -> None:
    """SPEC NOTE 11: run_a, run_a_result and spans describe one run A. Raises otherwise."""
    _check_same(run_a_result, inputs.scenario.name, inputs.contract_name)
    want = _ident_set((c.key, c.side, c.signal_time) for c in run_a)
    tr = run_a_result.trades
    got = _ident_set(zip(tr["key"], tr["side"], tr["signal_time"]))
    if got != want:
        raise ValueError(f"run A result holds {sum(got.values())} candidates that differ from "
                         f"the {sum(want.values())} run-A candidates given "
                         f"({sum((got - want).values())} extra, "
                         f"{sum((want - got).values())} missing)")
    sids = set(tr["strategy_id"])
    if sids and sids != {inputs.strategy_id}:
        raise ValueError(f"run A result is for {sorted(sids)}, not {inputs.strategy_id}")
    strat = inputs.strategy
    rescan: list[Candidate] = []
    for a, b in _spans(spans):
        inputs.check_pre_holdout([b - pd.Timedelta(1)])
        rescan.extend(strat.scan(inputs.features, inputs.regime_tables, a, b))
    again = _ident_set((c.key, c.side, c.signal_time) for c in rescan)
    if again != want:
        raise ValueError(f"the spans do not reproduce run A: a rescan with the run-A tables "
                         f"gives {sum(again.values())} candidates, run A has "
                         f"{sum(want.values())} ({sum((again - want).values())} extra, "
                         f"{sum((want - again).values())} missing); B3 would differ from run A "
                         "in its sample, not only in the gate")


@dataclass(frozen=True)
class Criterion7:
    run_a_total_usd: float
    placebo_p95_usd: float
    passed: bool | None        # None: partial seeds, not criterion 7
    counts: str = ""           # placebo vs run A trade counts (SPEC NOTE 10)
    n_draws: int = len(PLACEBO_SEEDS)

    @property
    def partial(self) -> bool:
        return self.passed is None

    @property
    def text(self) -> str:
        cmp = (f"run A total ${self.run_a_total_usd:.2f} vs placebo 95th percentile "
               f"${self.placebo_p95_usd:.2f}")
        if self.partial:
            side = "above" if self.run_a_total_usd > self.placebo_p95_usd else "not above"
            return (f"NOT criterion 7 (partial seeds: {self.n_draws} of {len(PLACEBO_SEEDS)} "
                    f"placebo draws): {cmp}: {side}")
        return f"Criterion 7 (placebo): {cmp}: {'PASS' if self.passed else 'FAIL'}"


def criterion7(run_a_result: BacktestResult, placebo: PlaceboResult, *,
               allow_partial: bool = False) -> Criterion7:
    """COMMON 10.5 criterion 7 on the 200 pre-registered draws. Another seed set raises unless
    allow_partial (tests / tooling only), and then gives a 'NOT criterion 7' comparison."""
    _check_same(run_a_result, placebo.scenario, placebo.contract)
    placebo.check_complete(allow_partial)
    a = result_stats(run_a_result)
    total = a["total_pnl_usd"]
    passed = placebo.passes(total) if placebo.complete else None
    return Criterion7(total, placebo.p95, passed,
                      f"B1 trade counts: {placebo.counts_text(a)}", int(len(placebo.draws)))


@dataclass(frozen=True)
class Comparison:
    benchmark: str
    run_a: dict
    bench: dict
    adds_nothing: bool         # the report must state the ingredient adds nothing
    text: str
    # B2 only: the same figures on the swapped candidates ({"run_a": ..., "b2": ..., "n": ...})
    swapped_only: dict | None = None


def b2_verdict(run_a_result: BacktestResult, b2: RandomIndexResult) -> Comparison:
    """SPEC NOTE 7: strategy beats B2 iff run A $ expectancy > B2 $ expectancy (the literal
    verdict, over all candidates). A second figure on the swapped candidates only (the rows
    where B2 differs from run A) is added to the text for information; it is not a verdict."""
    _check_same(run_a_result, b2.run.result.scenario.name, b2.run.result.contract)
    a, b = result_stats(run_a_result), result_stats(b2.run.result)
    beats = bool(np.nan_to_num(a["expectancy_usd"], nan=-np.inf)
                 > np.nan_to_num(b["expectancy_usd"], nan=-np.inf))
    text = (f"B2 random index: run A expectancy ${a['expectancy_usd']:.2f} vs B2 "
            f"${b['expectancy_usd']:.2f} per trade.")
    if not beats:
        text += " The strategy does not beat B2: the relative-strength selection adds nothing."
    sw = b2.swapped_stats(run_a_result)
    n_sw = int(len(b2.swapped))
    text += (f" Swapped candidates only ({n_sw} of {len(b2.mapping)}; kept ones are identical "
             f"on both sides): run A {_usd(sw['run_a']['expectancy_usd'])} "
             f"({int(sw['run_a']['n_filled'])} filled) vs B2 "
             f"{_usd(sw['b2']['expectancy_usd'])} ({int(sw['b2']['n_filled'])} filled) "
             "per trade.")
    return Comparison(B2, a, b, not beats, text, {**sw, "n": n_sw})


def b3_verdict(run_a_result: BacktestResult, b3: GateOffResult) -> Comparison:
    """COMMON 10.6: if B3 >= the gated run in $ expectancy, the gate adds nothing."""
    _check_same(run_a_result, b3.run.result.scenario.name, b3.run.result.contract)
    a, b = result_stats(run_a_result), result_stats(b3.run.result)
    nothing = bool(np.nan_to_num(b["expectancy_usd"], nan=-np.inf)
                   >= np.nan_to_num(a["expectancy_usd"], nan=-np.inf))
    text = (f"B3 gate off: expectancy ${b['expectancy_usd']:.2f} vs gated run A "
            f"${a['expectancy_usd']:.2f} per trade.")
    if nothing:
        text += " B3 >= the gated run: the regime gate adds nothing (the gate is not changed)."
    return Comparison(B3, a, b, nothing, text)


@dataclass(frozen=True)
class BenchmarkReport:
    strategy_id: str
    placebo: PlaceboResult
    random_index: RandomIndexResult
    gate_off: GateOffResult
    criterion7: Criterion7
    b2: Comparison
    b3: Comparison

    run_a_result: BacktestResult | None = field(default=None, repr=False)
    allow_partial: bool = False

    def trial_log_rows(self) -> list[dict[str, Any]]:
        """Three stage-`benchmark` rows for the runner's trial log (not counted in N), one per
        variant B1, B2, B3 (module docstring: mapping onto TrialRecord)."""
        return [self.placebo.trial_log_fields(self.run_a_result,
                                              allow_partial=self.allow_partial),
                self.random_index.trial_log_fields(self.run_a_result),
                self.gate_off.trial_log_fields()]

    def lines(self) -> list[str]:
        return [self.criterion7.text, self.criterion7.counts, self.b2.text, self.b3.text]


def run_benchmarks(inputs: BenchmarkInputs, run_a: Sequence[Candidate],
                   run_a_result: BacktestResult, spans: Sequence[tuple[Any, Any]],
                   seeds: Iterable[int] = PLACEBO_SEEDS, b2_seed: int = 1, *,
                   allow_partial: bool = False) -> BenchmarkReport:
    """B1, B2 and B3 for one run-A configuration, compared with its STRESS-d2 result at the
    same contract. spans: run A's windows [start, end) (the runner's Fold.start / Fold.end).
    seeds other than 1..200 raise before anything runs unless allow_partial (tests / tooling
    only; the B1 verdict then reads 'NOT criterion 7 (partial seeds)')."""
    seeds = tuple(int(s) for s in seeds)
    if seeds != PLACEBO_SEEDS and not allow_partial:
        raise ValueError(f"run_benchmarks needs the pre-registered placebo seeds 1..200 "
                         f"(COMMON 10.6 B1), got {len(seeds)}; allow_partial=True is for tests "
                         "and tooling only")
    check_run_a(inputs, run_a, run_a_result, spans)
    pl = run_placebo(inputs, run_a, seeds)
    b2 = run_random_index(inputs, run_a, b2_seed)
    b3 = run_gate_off(inputs, spans)
    return BenchmarkReport(inputs.strategy_id, pl, b2, b3,
                           criterion7(run_a_result, pl, allow_partial=allow_partial),
                           b2_verdict(run_a_result, b2), b3_verdict(run_a_result, b3),
                           run_a_result, allow_partial)
