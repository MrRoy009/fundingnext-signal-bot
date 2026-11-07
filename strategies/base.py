"""Interface every strategy implements. Specs live in docs/specs/<strategy_id>.md.

A strategy returns Candidates: absolute price levels fixed at the decision time T plus the
execution protocol the owner follows when placing the ticket by hand (COMMON 7). The engine
turns candidates into sized, risk-checked tickets; the backtester and the live engine read
the same fields, so a candidate means the same thing in both.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import time
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import pandas as pd

from core.calendar import exchange_for, local_to_utc

# Execution protocols (COMMON 7). None on a Candidate means the legacy symmetric
# "skip if +/- skip_if_atr_mult x ATR" rule, which COMMON 7 keeps only as the fallback if the
# owner rejects the protocols (O1 was approved, so every strategy in docs/specs uses one).
MARKET_BAND = "MARKET_BAND"
MARKET_OR_LIMIT = "MARKET_OR_LIMIT"
STOP_ENTRY = "STOP_ENTRY"
ORDER_TYPES = (MARKET_BAND, MARKET_OR_LIMIT, STOP_ENTRY)
PENDING_TYPES = (MARKET_OR_LIMIT, STOP_ENTRY)     # may rest as a limit / stop order with EXPIRY


@dataclass(frozen=True)
class Candidate:
    key: str
    side: str            # BUY / SELL
    entry: float         # E (MARKET_BAND, MARKET_OR_LIMIT) or the trigger (STOP_ENTRY)
    stop: float          # absolute price fixed at T, never re-anchored to the fill
    take_profit: float   # absolute price fixed at T
    atr: float           # ATR14_M15 at T: information only (legacy skip rule uses it)
    reason: str
    regime: str = "n/a"
    regime_prob: float = 0.0
    rank_score: float = 0.0
    signal_time: pd.Timestamp | None = None    # T = close of the signal bar (UTC)
    order_type: str | None = None              # one of ORDER_TYPES; None = legacy rule
    chase_limit: float | None = None           # MARKET_BAND, STOP_ENTRY
    skip_level: float | None = None            # MARKET_BAND, MARKET_OR_LIMIT
    expires_utc: pd.Timestamp | None = None    # EXPIRY of a resting limit / stop order
    time_exit_utc: pd.Timestamp | None = None  # manual time exit on the signal's local date
    sizing_entry: float | None = None          # worst acceptable entry; lots are sized on it

    @property
    def trigger(self) -> float | None:
        """The stop-entry trigger (COMMON 7 calls a stop ticket's reference the trigger)."""
        return self.entry if self.order_type == STOP_ENTRY else None

    @property
    def d(self) -> float:
        """D = |E - SL| (market / limit) or |trigger - SL| (stop): the same expression."""
        return abs(self.entry - self.stop)


@dataclass(frozen=True)
class ProtocolLevels:
    chase_limit: float | None
    skip_level: float | None
    sizing_entry: float


def protocol_levels(order_type: str, side: str, entry: float, stop: float,
                    cfg: dict) -> ProtocolLevels:
    """CHASE LIMIT, SKIP LEVEL and sizing entry for one protocol (COMMON 7).

    D = |entry - stop|. 'Worse' for a buy is a higher price, so the chase limit sits above E
    for a BUY and below it for a SELL; the skip level is on the stop side.
    sizing_entry = CHASE LIMIT (MARKET_BAND, STOP_ENTRY) or E (MARKET_OR_LIMIT), so a fill the
    protocol permits never risks more than the printed RISK $.
    SPEC NOTE: the levels are not rounded (COMMON 7 gives the fractions only); the ticket prints
    them at the symbol's precision.
    """
    if order_type not in ORDER_TYPES:
        raise ValueError(f"unknown order_type {order_type!r}; use one of {ORDER_TYPES}")
    if side not in ("BUY", "SELL"):
        raise ValueError(f"side must be BUY or SELL, not {side!r}")
    p = cfg["protocols"]
    sgn = 1.0 if side == "BUY" else -1.0
    d = abs(entry - stop)
    chase = entry + sgn * p["chase_frac"] * d
    skip = entry - sgn * p["skip_frac"] * d
    if order_type == MARKET_BAND:
        return ProtocolLevels(chase, skip, chase)
    if order_type == MARKET_OR_LIMIT:
        return ProtocolLevels(None, skip, entry)      # never chases
    return ProtocolLevels(chase, None, chase)         # STOP_ENTRY


def _same_level(given: float | None, computed: float | None) -> bool:
    """A strategy may restate a level it computed itself; only float noise is tolerated."""
    if given is None:
        return True                      # left blank: the computed level is filled in
    if computed is None:
        return False                     # a level this protocol does not have
    return math.isclose(given, computed, rel_tol=1e-9, abs_tol=1e-9)


def with_protocol_levels(c: Candidate, cfg: dict) -> Candidate:
    """chase_limit / skip_level / sizing_entry of a protocol candidate, always from cfg.

    The fractions are frozen (COMMON 11) and sizing_entry must be the worst fill the printed
    levels permit (COMMON 7), so the levels are derived here, never taken from the strategy. A
    strategy that sets a level which differs from the computed one (or that its protocol does
    not have) raises ValueError instead of being silently kept or overwritten: a mismatch is a
    strategy bug, and keeping it would print a CHASE LIMIT that RISK $ does not cover.
    A legacy candidate (order_type None) is sized on its entry, as before the protocols.
    """
    if c.order_type is None:
        return c if c.sizing_entry is not None else replace(c, sizing_entry=c.entry)
    lv = protocol_levels(c.order_type, c.side, c.entry, c.stop, cfg)
    for name in ("chase_limit", "skip_level", "sizing_entry"):
        given, computed = getattr(c, name), getattr(lv, name)
        if not _same_level(given, computed):
            raise ValueError(f"{c.key} {c.side} {c.order_type}: {name} {given} differs from "
                             f"the protocol level {computed} (config protocols)")
    return replace(c, chase_limit=lv.chase_limit, skip_level=lv.skip_level,
                   sizing_entry=lv.sizing_entry)


def time_exit_utc(cfg: dict, key: str, signal_time: pd.Timestamp) -> pd.Timestamp | None:
    """The manual time exit (config `time_exits`, exchange-local) on T's local date, in UTC.

    zoneinfo does the conversion, so the UTC clock moves with each market's own DST
    (US 19:45 / 20:45 UTC, GER40 15:15 / 16:15 UTC). None if the index has no time exit.
    """
    clock = (cfg.get("time_exits") or {}).get(key)
    if clock is None:
        return None
    tz = exchange_for(cfg, key).tz
    local_date = signal_time.tz_convert(tz).date()
    return local_to_utc(local_date, time.fromisoformat(clock), tz)


def time_exit_label(cfg: dict, key: str) -> str:
    """Local clock and market for the ticket line, e.g. '15:45 New York'."""
    clock = (cfg.get("time_exits") or {}).get(key, "?")
    city = exchange_for(cfg, key).tz.split("/")[-1].replace("_", " ")
    return f"{clock} {city}"


@dataclass(frozen=True)
class StrategyContext:
    """Per-cycle inputs computed once and shared by every strategy (ER-06).

    features: the FeatureService output for this cycle (COMMON 3-4), or None until it exists.
    regime:   the RegimeService output for this cycle (COMMON 6), or None.
    now:      the decision time T (close of the latest M15 bar); never wall-clock time.
    cfg:      the loaded config (frozen constants live there; strategies never hard-code them).
    regime_tables: {key: RegimeTable} built from bars closed by `now` (RegimeService.run_batch
              with as_of=now), or anything with the same `at_many(times)` / `threshold`. A live
              generate() replays the session's earlier cycles (COMMON 9: lockouts and daily
              caps are rebuilt by deterministic replay), which needs the regime at those earlier
              T as well, not only the snapshot at `now`. None: strategies return no candidate.
    Strategies must use only values with close time <= now.
    """
    features: Any = None
    regime: Any = None
    now: pd.Timestamp | None = None
    cfg: dict | None = None
    regime_tables: Any = None


# ---------------------------------------------------------------------------------------------
# Strategy parameters (each spec's "Parameters" table)
# ---------------------------------------------------------------------------------------------

def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
    return a == b


@dataclass(frozen=True)
class ParamGrid:
    """A strategy's tunable parameters: frozen defaults, the spec's one-at-a-time sensitivity
    grid per parameter, and the pair forming the spec's 2-D heatmap.

    Each strategy module writes its own values from its spec, e.g.
        PARAMS = ParamGrid(defaults={"theta": 1.0, "k": 1.0, "R": 1.5},
                           grid={"theta": (0.8, 0.9, 1.0, 1.1, 1.2), ...}, heatmap=("k", "R"))
    Frozen spec constants (windows, caps, protocol fractions) are NOT parameters: they never
    vary and stay module constants or config values.

    Why the grid check in `resolve`: every configuration run on pre-holdout data is a trial
    (COMMON 10.7), and the trial budget counts only grid cells. An off-grid value is refused
    unless the caller says it is a synthetic-data test (`off_grid=True`).
    """
    defaults: Mapping[str, Any]
    grid: Mapping[str, tuple]
    heatmap: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        defaults = MappingProxyType(dict(self.defaults))
        grid = MappingProxyType({k: tuple(v) for k, v in self.grid.items()})
        if set(grid) != set(defaults):
            raise ValueError(f"grid parameters {sorted(grid)} differ from defaults "
                             f"{sorted(defaults)}")
        for k, v in defaults.items():
            if not any(_same_value(v, g) for g in grid[k]):
                raise ValueError(f"default {k}={v!r} is not in its grid {grid[k]}")
            if len(set(grid[k])) != len(grid[k]):
                raise ValueError(f"grid for {k} has duplicate values: {grid[k]}")
        if self.heatmap is not None:
            a, b = self.heatmap
            if a == b or a not in defaults or b not in defaults:
                raise ValueError(f"heatmap {self.heatmap} must name two different parameters")
        object.__setattr__(self, "defaults", defaults)
        object.__setattr__(self, "grid", grid)

    def resolve(self, overrides: Mapping[str, Any] | None = None, *,
                off_grid: bool = False) -> Mapping[str, Any]:
        """Defaults with `overrides` applied, as a read-only mapping. Unknown names raise;
        values must be grid values unless off_grid (synthetic tests only)."""
        out = dict(self.defaults)
        for k, v in (overrides or {}).items():
            if k not in out:
                raise ValueError(f"unknown parameter {k!r}; parameters are {list(out)}")
            match = [g for g in self.grid[k] if _same_value(v, g)]
            if match:
                v = match[0]             # the grid's own value, so equal configs compare equal
            elif not off_grid:
                raise ValueError(f"{k}={v!r} is not in its sensitivity grid {self.grid[k]}")
            out[k] = v
        return MappingProxyType(out)

    def one_at_a_time(self) -> list[dict[str, Any]]:
        """The default first, then each parameter's other grid values with the rest at their
        defaults (parameter order of `defaults`, grid order within)."""
        out = [dict(self.defaults)]
        for k, values in self.grid.items():
            for v in values:
                if not _same_value(v, self.defaults[k]):
                    out.append({**self.defaults, k: v})
        return out

    def heatmap_cells(self) -> list[dict[str, Any]]:
        """Every cell of the 2-D heatmap (row-major: first parameter outer), others default."""
        if self.heatmap is None:
            return []
        a, b = self.heatmap
        return [{**self.defaults, a: va, b: vb} for va in self.grid[a] for vb in self.grid[b]]

    def configurations(self) -> list[dict[str, Any]]:
        """One-at-a-time grid plus the heatmap cells not already in it, each once. Its length
        is the spec's configuration count (e.g. 13 + 16 = 29; a 5 x 5 heatmap containing the
        one-at-a-time grid gives 25)."""
        out: list[dict[str, Any]] = []
        for p in self.one_at_a_time() + self.heatmap_cells():
            if not any(all(_same_value(p[k], q[k]) for k in p) for q in out):
                out.append(p)
        return out

    @staticmethod
    def label(params: Mapping[str, Any]) -> str:
        """Stable text form for logs and the trial log, e.g. 'theta=1.0,k=1.0,R=1.5'."""
        return ",".join(f"{k}={v}" for k, v in params.items())


class Strategy(Protocol):
    """What the live engine, the backtester and the feasibility gate call.

    generate() is the live path; scan() is the batch path over [start, end). For every T in
    [start, end), generate() at T (with features and regime tables built from bars closed by T)
    returns exactly the candidates of scan() with signal_time == T, lockouts (reference
    occupancy, COMMON 7.1) and daily caps included (COMMON 9, ER-15). The usual implementation
    is strategies.context.WindowedStrategy, where generate() replays the current session with
    the same scan code.

    params: the strategy's resolved parameters (defaults unless overridden at construction);
    param_grid: its ParamGrid. scan(params=None) uses `params`.
    """
    id: str
    stats: dict          # QA-validated out-of-sample stats: oos_win_rate, expectancy_r,
                         # max_dd_pct, n_trades
    params: Mapping[str, Any]
    param_grid: ParamGrid

    def generate(self, bars: dict[str, pd.DataFrame], regime_bars: dict[str, pd.DataFrame],
                 now: pd.Timestamp, context: StrategyContext) -> list[Candidate]:
        """Return candidates using closed bars only (close time <= now)."""
        ...

    def scan(self, features: Any, regime_tables: Mapping[str, Any], start: pd.Timestamp,
             end: pd.Timestamp, params: Mapping[str, Any] | None = None, *,
             on_selection: Any = None) -> list[Candidate]:
        """Every candidate with signal time T in [start, end), in time order (config instrument
        order within one T). `features` is a FeatureService (or {key: IndexFeatures});
        `regime_tables` maps key -> RegimeTable. Lockouts are replayed from the local midnight
        of start's session, so a mid-session `start` gives the same candidates as a scan from
        an earlier start (and as generate()).

        on_selection: optional callable given every cycle's Selection. COMMON 4.2.8 requires
        every cycle's rank scores to be logged; a backtest or feasibility run must pass a sink
        and persist sel.log_line() (scan() itself logs them at DEBUG only)."""
        ...
