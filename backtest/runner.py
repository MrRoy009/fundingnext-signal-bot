"""Trial runner (COMMON 10.2-10.4, 10.7; ER-14): runs configurations through the backtester on
the walk-forward windows and logs every configuration in docs/trial_log.csv.

Why one runner: outcome statistics on pre-holdout data may only be produced for pre-registered
configurations and must all be counted (COMMON 10.7). Every function here that runs a
configuration either logs it (backtest.trials) or is a building block the logging functions
call; nothing here reads the holdout (BarStore.load is never given final_holdout).

Pipeline
  1. MarketData (load_market / market_from_frames): pre-holdout M15 + H1 bars, the
     FeatureService and the regime tables, built ONCE per process and cached by the data hash
     and the shared settings hash (the regime batch takes minutes per index). Every
     configuration of the process reuses them; strategy scans are cached on them as well.
  2. scan_config: strategy.scan() per window with an on_selection sink, so every cycle's
     selection line (COMMON 4.2.8, A3.5) is kept and saved with the run.
  3. run_candidates / run_config: backtest.engine.run_backtest for one scenario (COMMON 10.3,
     backtest.scenarios.SCENARIOS) at one contract; trades get the fold of their signal time T;
     the daily series is the end-of-day MTM equity change on the trading-day calendar
     (backtest.folds.trading_calendar), zero on days without a position.
  4. Protocols: run_a (defaults, concatenated test windows, every scenario, secondary sample
     reported only), run_sensitivity (every grid configuration, STRESS-d2), run_b (walk-forward
     selection with embargo), run_portfolio (combined book, COMMON 8.8, report only).

Real data is logged or not run: on a MarketData from load_market (source REAL_DATA_SOURCE)
run_candidates / run_config raise PermissionError outside a logging protocol call, and the
protocol functions refuse log=False. Unlogged runs are for synthetic frames only.

Contracts: the demo contract always; the competition contract only when config
rules.competition_usd_per_point is set. It is null now, so every result is at the demo contract
only and the notes say so (CONTRACT_NOTE).

Public API
    STRATEGIES, DROPPED_STRATEGIES, PASS_SCENARIO ('STRESS-d2'), SELECT_SCENARIO, RUN_A_SCENARIOS
    strategy_class(sid), param_grid(sid), entry_protocol(sid), make_strategy(cfg, sid, params)
    contracts(cfg) -> tuple[str, ...];  CONTRACT_NOTE
    MarketData                              bars, features, regime tables, specs, hashes
    market_from_frames(cfg, m15, h1, specs, *, regime_factory=None, regime_tag=...) -> MarketData
    load_market(cfg, specs, *, store=None, regime_factory=None) -> MarketData
    clear_market_cache()
    ScanResult; scan_config(market, sid, params, windows) -> ScanResult
    ConfigRun (.trades, .daily, .equity, .filled, .metrics())
    run_candidates(market, pairs, scenario, windows, *, contract, mode, embargo, ...) -> ConfigRun
    run_config(strategy_id, params, scenario, windows=TEST_FOLDS, *, market, contract='demo',
               embargo=False) -> ConfigRun
    daily_series(daily_eod, calendar, initial) -> DataFrame (date, equity, pnl_usd, ret)
    daily_sharpe(pnl) -> float;  max_drawdown(values, start_value) -> (usd, pct)
    run_metrics(run) -> dict
    evaluate_configuration(market, sid, params, *, stage, scenarios, ...) -> Evaluation
    ensure_logged(market, sid, params, *, stage, windows, log_path, runs_dir, rerun_note) -> str
    REAL_DATA_SOURCE; MarketData.is_real; MarketData.cfg_digest (scan cache key)
    run_a(market, sid, ...) -> RunAResult
    sensitivity_plan(grid) -> DataFrame;  criterion5(table) -> dict
    run_sensitivity(market, sid, ...) -> SensitivityResult
    grid_distance(params, grid) -> int;  select_params(points, sharpes, grid) -> int
    run_b(market, sid, ...) -> RunBResult
    run_portfolio(market, strategy_ids, ...) -> PortfolioResult
    save_specs(specs, path), load_specs(path), fetch_specs(cfg)   (read-only symbol_info)
    main(argv)                              CLI for the Run phase (real pre-holdout data)

SPEC NOTE (daily series): COMMON 10.2 "end-of-day MTM equity change": the engine's end-of-day
equity (worst-intrabar MTM, last mark of each 00:00 Europe/Athens server day) as of each
calendar day, differenced; the first day is differenced against the equity at the end of the
server day before it (the initial balance if nothing earlier). A change on a server date that
is not in the calendar (e.g. a held position over a common holiday) lands on the next calendar
day, so the series sums to the equity change over the windows. Sharpe, DSR, PBO and the
bootstrap use the $ change ('pnl_usd'); 'ret' (change / previous equity) is stored as well.
SPEC NOTE (daily Sharpe): mean / standard deviation (ddof 1) of the daily $ series,
non-annualised (COMMON 10.7 SR_n); NaN with fewer than 2 days or zero deviation.
SPEC NOTE (concatenated windows): the 9 test windows are contiguous, so a concatenated run is
ONE backtest over [2024-02-01, 2026-05-01) with the candidates of every window; the balance
compounds across folds and a position may cross a fold boundary (it belongs to its T's fold).
Run B is one backtest of the union of each test window's candidates, each scanned with the
parameters chosen for its fold.
SPEC NOTE (run B selection scenario): "highest daily-return Sharpe" in each train window is
computed under STRESS-d2, the PASS scenario whose STATS the tickets carry; the run-B
out-of-sample result (pass criterion 3) is reported under the same scenario.
SPEC NOTE (embargo): "train trades whose exit falls after the train window end are dropped from
selection". The train backtest runs on bars up to train end + EMBARGO_PAD_DAYS; every filled
trade with exit_time > train end is dropped by removing its candidate and re-running, until no
filled trade exits after the end (dropping one can free its index for a later candidate). The
train Sharpe is then on the train-window daily calendar, with no MTM of a dropped trade in it.
Bars after end + pad cannot change which trades exit by the end: a trade still open at the
last bar ends there (end_of_data) and is dropped either way.
SPEC NOTE (ties): "closest to default" = smallest sum over parameters of |grid position -
default's grid position| (grid order of the spec table); equal distance (e.g. theta 0.9 vs 1.1)
-> the earlier point of ParamGrid.one_at_a_time() (parameter order of the spec table, grid
order within). A Sharpe tie is equality within 1e-12 relative. A grid point with NaN Sharpe (no
trade or zero variance in the train window) is never chosen over a finite one; if every point
is NaN the default is chosen.
SPEC NOTE (sensitivity cells): every ParamGrid.configurations() cell (13 one-at-a-time + 16
heatmap = 29) is a cell; a cell with fewer than 100 trades is 'insufficient' and failed
(COMMON 10.4). Immediate neighbours of each default are the adjacent grid values of each
parameter with the others at default (one-at-a-time cells).
"""
from __future__ import annotations

import argparse
import contextvars
import dataclasses
import functools
import hashlib
import importlib
import inspect
import json
import logging
import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from backtest import trials as tl
from backtest.engine import COMBINED, COMBINED_PRIORITY, SINGLE, StrategyCandidate, run_backtest
from backtest.feasibility import _frames_sha256
from backtest.folds import (COMP_RULING_NOTE, OUTSIDE, SECONDARY_SAMPLE, TEST_FOLDS,
                            WALK_FORWARD, Fold,
                            WalkForwardFold, assign_folds, competition_feasibility,
                            fold_data_shares, in_windows, is_contiguous, sensitivity_cell_status,
                            trade_count_check, trading_calendar, window_label, window_span,
                            POSITIVE)
from backtest.market import ServerClock
from backtest.results import BacktestResult
from backtest.scenarios import SCENARIOS, Scenario, contract_table
from data.mt5_adapter import SymbolSpec
from strategies.base import MARKET_BAND, STOP_ENTRY, Candidate, ParamGrid
from strategies.selection import config_order

logger = logging.getLogger(__name__)

# strategy_id -> (module, class, entry protocol from its spec)
STRATEGIES: dict[str, tuple[str, str, str]] = {
    "RS-MOM-01": ("strategies.rs_mom_01", "RsMom01", MARKET_BAND),
    "SES-IBX-01": ("strategies.ses_ibx_01", "SesIbx01", STOP_ENTRY),
}
DROPPED_STRATEGIES = {"MR-TWAP-01": "dropped without a backtest (COMMON 12C, amendment A3: "
                                    "190 candidates < 200); it is never run"}

PASS_SCENARIO = "STRESS-d2"          # COMMON 10.3: ticket STATS and the 10.4 counts
SELECT_SCENARIO = "STRESS-d2"        # run B train selection (SPEC NOTE)
RUN_A_SCENARIOS = tuple(SCENARIOS)   # BASE, STRESS-d2, STRESS-d3, STRESS-d1, MISSED-EXIT, EXIT-3X
BAR_PAD_BEFORE_DAYS = 3              # bars kept before a window (signal bar, placement)
EMBARGO_PAD_DAYS = 14                # bars kept after a train window (SPEC NOTE, embargo)
CONTRACT_NOTE = ("competition contract not set (config rules.competition_usd_per_point is null):"
                 " results at the demo contract only")
SHARPE_TIE_REL = 1e-12
REAL_DATA_SOURCE = "BarStore.load pre-holdout"     # MarketData.source of load_market

# True while a logging protocol call is running (see _protocol): backtests of a real-data
# market are only allowed inside one, so every outcome on real data reaches the trial log.
_LOGGED_CALL: contextvars.ContextVar[bool] = contextvars.ContextVar("runner_logged_call",
                                                                    default=False)


def _require_logged(market: "MarketData", what: str) -> None:
    """COMMON 10.7 / PROMPT MANDATORY VALIDATION 7: outcome statistics on real pre-holdout
    data only for logged configurations. Synthetic markets are unrestricted (tests, tooling)."""
    if market.is_real and not _LOGGED_CALL.get():
        raise PermissionError(
            f"{what} on real pre-holdout data ({market.source}) outside a logging protocol: "
            "real-data outcomes are produced only by run_a / run_sensitivity / run_b / "
            "run_portfolio / evaluate_configuration with log=True (COMMON 10.7)")


def _protocol(fn: Callable) -> Callable:
    """Decorator for the protocol functions (arguments `market` and `log`): log=False is
    refused on a real-data market unless called inside a logging protocol (run_a's own
    unlogged evaluation before it logs one row), and a logged call opens the context in which
    run_config / run_candidates may backtest real data."""
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        b = sig.bind(*args, **kwargs)
        b.apply_defaults()
        market, log = b.arguments["market"], bool(b.arguments.get("log", True))
        if not log and market.is_real and not _LOGGED_CALL.get():
            raise PermissionError(
                f"{fn.__name__}(log=False) on real pre-holdout data ({market.source}): every "
                "configuration run on real data is logged (COMMON 10.7, 8.10)")
        token = _LOGGED_CALL.set(_LOGGED_CALL.get() or log)
        try:
            return fn(*args, **kwargs)
        finally:
            _LOGGED_CALL.reset(token)
    return wrapper


# ---------------------------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------------------------

def _entry(strategy_id: str) -> tuple[str, str, str]:
    if strategy_id in DROPPED_STRATEGIES:
        raise ValueError(f"{strategy_id} is {DROPPED_STRATEGIES[strategy_id]}")
    try:
        return STRATEGIES[strategy_id]
    except KeyError:
        raise ValueError(f"unknown strategy {strategy_id!r}; runnable: {sorted(STRATEGIES)}")


def strategy_class(strategy_id: str) -> type:
    mod, cls, _ = _entry(strategy_id)
    return getattr(importlib.import_module(mod), cls)


def param_grid(strategy_id: str) -> ParamGrid:
    return strategy_class(strategy_id).param_grid


def entry_protocol(strategy_id: str) -> str:
    return _entry(strategy_id)[2]


def make_strategy(cfg: dict, strategy_id: str, params: Mapping[str, Any] | None = None):
    """The strategy at grid parameters (off-grid values are refused: every configuration run
    on pre-holdout data is a trial and the budget counts grid cells only)."""
    return strategy_class(strategy_id)(cfg, params)


def contracts(cfg: dict) -> tuple[str, ...]:
    """'demo' always; 'competition' only when config rules.competition_usd_per_point is set."""
    return ("demo",) if contract_table(cfg, "competition") is None else ("demo", "competition")


def _scenario(s: Scenario | str) -> Scenario:
    if isinstance(s, Scenario):
        return s
    try:
        return SCENARIOS[s]
    except KeyError:
        raise ValueError(f"unknown scenario {s!r}; COMMON 10.3 scenarios: {list(SCENARIOS)}")


def _bracket(params: Mapping[str, Any]) -> str:
    """'[theta=1.0 k=1.0 R=1.5]': a parameter vector as one value of a composite row."""
    return "[" + " ".join(f"{k}={v}" for k, v in params.items()) + "]"


def _scen_key(name: str, contract: str) -> str:
    return name if contract == "demo" else f"{name}@{contract}"


# ---------------------------------------------------------------------------------------------
# Market data (built once per process)
# ---------------------------------------------------------------------------------------------

RegimeFactory = Callable[[dict, Mapping[str, pd.DataFrame], Mapping[str, pd.DataFrame]],
                         Mapping[str, Any]]
DEFAULT_REGIME_TAG = "regime.service.RegimeService.run_batch"


def default_regime_factory(model_dir: Path | None = None,
                           refit_log: Path | None = None) -> RegimeFactory:
    """RegimeService.run_batch per index (model cache and refit log as in the feasibility CLI)."""
    def build(cfg: dict, m15: Mapping[str, pd.DataFrame], h1: Mapping[str, pd.DataFrame]):
        from regime.service import MODEL_DIR, REFIT_LOG, RegimeService
        rs = RegimeService(cfg, model_dir=model_dir or MODEL_DIR, refit_log=refit_log or REFIT_LOG)
        out = {}
        for k in m15:
            t0 = _time.time()
            out[k] = rs.run_batch(k, h1[k], m15[k])
            logger.info("regime table %s built (%.0fs)", k, _time.time() - t0)
        return out
    return build


@dataclass
class _Services:
    features: Any
    tables: Mapping[str, Any]
    scans: dict = field(default_factory=dict)


_SERVICES: dict[tuple[str, str, str], _Services] = {}


def clear_market_cache() -> None:
    _SERVICES.clear()


@dataclass
class MarketData:
    """Everything a configuration run reads. Build with load_market / market_from_frames."""
    cfg: dict
    keys: tuple[str, ...]
    m15: dict[str, pd.DataFrame]
    h1: dict[str, pd.DataFrame]
    features: Any
    tables: Mapping[str, Any]
    specs: Mapping[str, SymbolSpec]
    data_hash: str
    source: str
    scans: dict = field(default_factory=dict, repr=False)

    @property
    def shared_hash(self) -> str:
        return tl.shared_settings_hash(self.cfg)

    @property
    def config_hash(self) -> str:
        return tl.config_hash(self.cfg, self.specs)

    @property
    def is_real(self) -> bool:
        """Real pre-holdout bars (load_market): backtests only inside a logging protocol."""
        return self.source == REAL_DATA_SOURCE

    @property
    def cfg_digest(self) -> str:
        """Hash of the WHOLE config: the scan cache key, since a strategy may read blocks
        outside the shared settings (SES-IBX-01 reads rules.restricted_windows_utc)."""
        return hashlib.sha256(json.dumps(tl.json_safe(self.cfg), sort_keys=True)
                              .encode("utf-8")).hexdigest()[:16]

    @property
    def initial_balance(self) -> float:
        return float(self.cfg["rules"]["initial_balance"])

    def data_days(self) -> pd.DatetimeIndex:
        """Server dates (naive) with at least one M15 bar of a ranked index."""
        clock = ServerClock.from_cfg(self.cfg)
        days = set()
        for k in self.keys:
            t = pd.DatetimeIndex(pd.to_datetime(self.m15[k]["time_utc"], utc=True)).as_unit("ns")
            days.update(np.unique(clock.dates(t.asi8)).tolist())
        return pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in days))


def data_hash(cfg: dict, m15: Mapping[str, pd.DataFrame], h1: Mapping[str, pd.DataFrame],
              keys: Sequence[str]) -> str:
    """SHA-256 (16 hex) of the M15 and H1 frames of `keys`, the same digest as the
    feasibility gate's provenance line."""
    entry, regime_tf = cfg["timeframes"]["entry"], cfg["timeframes"]["regime"]
    return _frames_sha256([(f"{k}/{tf}", d) for k in keys
                           for tf, d in ((entry, m15[k]), (regime_tf, h1[k]))])[:16]


def _check_pre_holdout(cfg: dict, frames: Mapping[str, pd.DataFrame], what: str) -> None:
    start = (cfg.get("holdout") or {}).get("start")
    if not start:
        return
    hs = pd.Timestamp(start, tz="UTC")
    for k, df in frames.items():
        if len(df) and pd.Timestamp(pd.to_datetime(df["time_utc"], utc=True).max()) >= hs:
            raise PermissionError(f"{k} {what} bars reach the final holdout ({start}); the "
                                  "runner only runs on pre-holdout data (GATE 3 not approved)")


def market_from_frames(cfg: dict, m15: Mapping[str, pd.DataFrame],
                       h1: Mapping[str, pd.DataFrame], specs: Mapping[str, SymbolSpec], *,
                       regime_factory: RegimeFactory | None = None,
                       regime_tag: str = DEFAULT_REGIME_TAG,
                       source: str = "frames") -> MarketData:
    """MarketData from bar frames. Features and regime tables are built once per (data hash,
    shared settings hash, regime_tag) per process and reused (the scan cache with them)."""
    from features.service import FeatureService
    keys = tuple(k for k in config_order(cfg) if k in m15)
    if not keys:
        raise ValueError("no bars for any ranked index")
    m15 = {k: m15[k] for k in keys}
    h1 = {k: h1[k] for k in keys}
    _check_pre_holdout(cfg, m15, "M15")
    _check_pre_holdout(cfg, h1, "H1")
    dh = data_hash(cfg, m15, h1, keys)
    ck = (dh, tl.shared_settings_hash(cfg), regime_tag)
    svc = _SERVICES.get(ck)
    if svc is None:
        t0 = _time.time()
        fs = FeatureService(cfg)
        for k in keys:
            fs.build(k, m15[k], h1[k])
        factory = regime_factory or default_regime_factory()
        tables = dict(factory(cfg, m15, h1))
        svc = _Services(fs, tables)
        _SERVICES[ck] = svc
        logger.info("features and regime tables built for data %s (%.0fs)", dh, _time.time() - t0)
    return MarketData(cfg, keys, dict(m15), dict(h1), svc.features, svc.tables, dict(specs), dh,
                      source, svc.scans)


def load_market(cfg: dict, specs: Mapping[str, SymbolSpec], *, store: Any = None,
                regime_factory: RegimeFactory | None = None) -> MarketData:
    """Pre-holdout research data via BarStore.load (holdout and A1 excluded windows dropped;
    final_holdout is never passed)."""
    from data.store import BarStore
    store = store or BarStore(cfg)
    entry, regime_tf = cfg["timeframes"]["entry"], cfg["timeframes"]["regime"]
    keys = config_order(cfg)
    m15 = {k: store.load(k, entry) for k in keys}
    h1 = {k: store.load(k, regime_tf) for k in keys}
    return market_from_frames(cfg, m15, h1, specs, regime_factory=regime_factory,
                              source=REAL_DATA_SOURCE)


# ---------------------------------------------------------------------------------------------
# Scans
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ScanResult:
    strategy_id: str
    params: Mapping[str, Any]
    windows: tuple[Fold, ...]
    candidates: tuple[Candidate, ...]
    folds: tuple[int, ...]                              # window number of each candidate
    selection_lines: tuple[tuple[int, str], ...] = field(repr=False)

    def pairs(self) -> list[StrategyCandidate]:
        return [StrategyCandidate(self.strategy_id, c) for c in self.candidates]


def _win_key(windows: Sequence[Fold]) -> tuple:
    return tuple((w.n, w.first.isoformat(), w.last.isoformat()) for w in windows)


def scan_config(market: MarketData, strategy_id: str, params: Mapping[str, Any] | None,
                windows: Fold | Sequence[Fold]) -> ScanResult:
    """strategy.scan() over each window [start, end) with an on_selection sink (COMMON 4.2.8).
    Per-window scans give the same candidates as one long scan (scan() replays lockouts from
    the local midnight of start; no lockout crosses a local date). Cached per market data
    and full config (MarketData.cfg_digest): markets with the same bars and shared settings
    share the cache, but a different rules block (e.g. restricted windows) never reuses a
    scan."""
    grid = param_grid(strategy_id)
    p = dict(grid.resolve(params))
    ws = tuple(sorted((windows,) if isinstance(windows, Fold) else tuple(windows),
                      key=lambda w: w.start))
    key = (market.cfg_digest, strategy_id, tl.params_text(tl.canonical_params(p)), _win_key(ws))
    hit = market.scans.get(key)
    if hit is not None:
        return hit
    strat = make_strategy(market.cfg, strategy_id, p)
    cands: list[Candidate] = []
    nums: list[int] = []
    lines: list[tuple[int, str]] = []
    for w in ws:
        sels: list = []
        got = strat.scan(market.features, market.tables, w.start, w.end,
                         on_selection=sels.append)
        for c in got:
            if not w.contains(c.signal_time):
                raise AssertionError(f"{strategy_id}: candidate at {c.signal_time} outside "
                                     f"window {w.label}")
        cands.extend(got)
        nums.extend([w.n] * len(got))
        lines.extend((w.n, s.log_line()) for s in sels)
    out = ScanResult(strategy_id, p, ws, tuple(cands), tuple(nums), tuple(lines))
    market.scans[key] = out
    return out


# ---------------------------------------------------------------------------------------------
# Daily series and metrics
# ---------------------------------------------------------------------------------------------

def daily_series(daily_eod: pd.DataFrame, calendar: pd.DatetimeIndex,
                 initial: float) -> pd.DataFrame:
    """End-of-day MTM equity change on `calendar` (see the SPEC NOTE). daily_eod: the engine's
    BacktestResult.daily (date, balance, equity per server date)."""
    cal = pd.DatetimeIndex(calendar).normalize().as_unit("ns")
    if len(cal) == 0:
        return pd.DataFrame({"date": cal, "equity": [], "pnl_usd": [], "ret": []})
    if len(daily_eod):
        d = pd.DatetimeIndex(pd.to_datetime(pd.Series(daily_eod["date"]).astype(str))
                             ).as_unit("ns").asi8
        v = daily_eod["equity"].to_numpy(dtype="float64")
    else:
        d, v = np.zeros(0, np.int64), np.zeros(0)
    j = np.searchsorted(d, cal.asi8, side="right") - 1
    eod = np.where(j >= 0, v[np.maximum(j, 0)] if len(v) else initial, initial)
    j0 = int(np.searchsorted(d, cal.asi8[0], side="left")) - 1
    prev0 = float(v[j0]) if j0 >= 0 else float(initial)
    prev = np.r_[prev0, eod[:-1]]
    pnl = eod - prev
    return pd.DataFrame({"date": cal, "equity": eod, "pnl_usd": pnl, "ret": pnl / prev})


def daily_sharpe(pnl) -> float:
    """Non-annualised mean / std (ddof 1); NaN with < 2 values or zero deviation."""
    x = np.asarray(pnl, dtype="float64")
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    if not sd > 0:
        return float("nan")
    return float(np.mean(x) / sd)


def max_drawdown(values, start_value: float) -> tuple[float, float]:
    """(max drawdown in $ as a non-positive number, as a fraction of the running peak)."""
    e = np.r_[float(start_value), np.asarray(values, dtype="float64")]
    peak = np.maximum.accumulate(e)
    dd = e - peak
    i = int(np.argmin(dd))
    return float(dd[i]), float(dd[i] / peak[i]) if peak[i] else float("nan")


@dataclass(frozen=True)
class ConfigRun:
    """One configuration (or candidate set) under one scenario at one contract.

    trades: every candidate row of the engine (filled / skipped / suppressed / blocked /
    cancelled) plus 'fold' = the window number of its signal time T (-1 outside the windows).
    daily: date, equity, pnl_usd, ret on the trading-day calendar of the windows.
    equity: the M15 MTM marks inside the windows' span (time_utc, balance, equity).
    embargoed: filled train trades dropped by the embargo (empty without embargo)."""
    strategy_id: str
    params: Mapping[str, Any]
    scenario: Scenario
    contract: str
    windows: tuple[Fold, ...]
    result: BacktestResult = field(repr=False)
    trades: pd.DataFrame = field(repr=False)
    daily: pd.DataFrame = field(repr=False)
    equity: pd.DataFrame = field(repr=False)
    start_equity: float
    selection_lines: tuple[tuple[int, str], ...] = field(default=(), repr=False)
    embargoed: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    @property
    def filled(self) -> pd.DataFrame:
        t = self.trades
        return t[(t["status"] == "filled") & (t["fold"] != OUTSIDE)]

    @property
    def sharpe(self) -> float:
        return daily_sharpe(self.daily["pnl_usd"])

    def metrics(self) -> dict[str, Any]:
        return run_metrics(self)


def run_metrics(run: ConfigRun) -> dict[str, Any]:
    """Per-scenario metrics for the trial log and the reports (JSON-safe values)."""
    t = run.trades[run.trades["fold"] != OUTSIDE]
    f = run.filled
    n = int(len(f))
    dd_usd, dd_pct = max_drawdown(run.equity["equity"], run.start_equity)
    per_fold = {}
    for w in run.windows:
        g = f[f["fold"] == w.n]
        per_fold[str(w.n)] = {"n": int(len(g)), "pnl_usd": float(g["pnl_usd"].sum())}

    def by(col: str) -> dict:
        out = {}
        for k, g in f.groupby(col, sort=True):
            out[str(k)] = {"n": int(len(g)), "pnl_usd": float(g["pnl_usd"].sum()),
                           "mean_r": float(g["r_multiple"].mean())}
        return out
    exit_shares = (f["exit_type"].value_counts(normalize=True).sort_index().to_dict()
                   if n else {})
    return tl.json_safe({
        "scenario": run.scenario.name, "contract": run.contract,
        "n_candidates": int(len(t)), "status": t["status"].value_counts().sort_index().to_dict(),
        "n_trades": n,
        "expectancy_usd": float(f["pnl_usd"].mean()) if n else float("nan"),
        "expectancy_r": float(f["r_multiple"].mean()) if n else float("nan"),
        "total_usd": float(f["pnl_usd"].sum()),
        "sharpe_daily": run.sharpe,
        "n_days": int(len(run.daily)), "daily_total_usd": float(run.daily["pnl_usd"].sum()),
        "max_dd_usd": dd_usd, "max_dd_pct": dd_pct,
        "folds": per_fold, "index": by("key"), "side": by("side"),
        "exit_shares": exit_shares, "n_embargoed": int(len(run.embargoed)),
    })


# ---------------------------------------------------------------------------------------------
# Running candidates through the backtester
# ---------------------------------------------------------------------------------------------

def _trim(frames: Mapping[str, pd.DataFrame], start: pd.Timestamp,
          end: pd.Timestamp | None) -> dict[str, pd.DataFrame]:
    out = {}
    for k, df in frames.items():
        t = pd.to_datetime(df["time_utc"], utc=True)
        keep = t >= start
        if end is not None:
            keep &= t < end
        out[k] = df[keep.to_numpy()].reset_index(drop=True)
    return out


def _ident(sid: str, key: str, signal_time, side: str) -> tuple:
    return (sid, key, pd.Timestamp(signal_time).value, side)


def run_candidates(market: MarketData, pairs: Iterable[StrategyCandidate | tuple[str, Candidate]],
                   scenario: Scenario | str, windows: Fold | Sequence[Fold], *,
                   contract: str = "demo", mode: str = SINGLE, embargo: bool = False,
                   strategy_id: str | None = None, params: Mapping[str, Any] | None = None,
                   selection_lines: Sequence[tuple[int, str]] = ()) -> ConfigRun:
    """Backtest a candidate set for one scenario and contract over `windows`.

    Building block for run_config and for tools that make their own candidates (benchmarks).
    The account starts at the initial balance at the first bar of the windows' span (bars from
    BAR_PAD_BEFORE_DAYS earlier are loaded for the signal bar). embargo=True (train windows):
    bars end EMBARGO_PAD_DAYS after the span and trades exiting after the span end are removed
    (SPEC NOTE). On a real-data market (load_market) it runs only inside a logging protocol
    (PermissionError otherwise)."""
    _require_logged(market, "run_candidates / run_config")
    cfg = market.cfg
    sc = _scenario(scenario)
    ws = tuple(sorted((windows,) if isinstance(windows, Fold) else tuple(windows),
                      key=lambda w: w.start))
    start, end = window_span(ws)
    table = contract_table(cfg, contract)
    if table is None:
        raise ValueError(f"{contract} contract is not set in config; {CONTRACT_NOTE}")
    items = [p if isinstance(p, StrategyCandidate) else StrategyCandidate(str(p[0]), p[1])
             for p in pairs]
    sids = sorted({p.strategy_id for p in items})
    sid = strategy_id or (sids[0] if len(sids) == 1 else "+".join(sids) or "none")
    bars = _trim(market.m15, start - pd.Timedelta(days=BAR_PAD_BEFORE_DAYS),
                 end + pd.Timedelta(days=EMBARGO_PAD_DAYS) if embargo else None)

    dropped: list[pd.DataFrame] = []
    current = list(items)
    while True:
        res = run_backtest(current, bars, cfg, sc, market.specs, contract=table,
                           contract_name=contract, mode=mode)
        if not embargo:
            break
        tr = res.trades
        late = (tr["status"] == "filled") & (pd.to_datetime(tr["exit_time"], utc=True) > end)
        if not late.any():
            break
        gone = {_ident(r.strategy_id, r.key, r.signal_time, r.side)
                for r in tr[late].itertuples()}
        dropped.append(tr[late])
        current = [p for p in current if _ident(p.strategy_id, p.candidate.key,
                                                p.candidate.signal_time, p.candidate.side)
                   not in gone]

    trades = res.trades.copy()
    trades["fold"] = assign_folds(trades["signal_time"], ws)
    calendar = trading_calendar(cfg, ws, market.keys)
    daily = daily_series(res.daily, calendar, market.initial_balance)
    eq = res.equity
    te = pd.to_datetime(eq["time_utc"], utc=True)
    before = eq[te <= start]
    start_eq = float(before["equity"].iloc[-1]) if len(before) else market.initial_balance
    equity = eq[((te > start) & (te <= end)).to_numpy()].reset_index(drop=True)
    emb = pd.concat(dropped, ignore_index=True) if dropped else trades.iloc[0:0].copy()
    return ConfigRun(sid, dict(params or {}), sc, contract, ws, res, trades, daily, equity,
                     start_eq, tuple(selection_lines), emb)


def run_config(strategy_id: str, params: Mapping[str, Any] | None, scenario: Scenario | str,
               windows: Fold | Sequence[Fold] = TEST_FOLDS, *, market: MarketData,
               contract: str = "demo", embargo: bool = False) -> ConfigRun:
    """One configuration (grid parameters; None = defaults) under one scenario over the
    windows: scan with the selection sink, then the backtester. Not logged by itself: the
    protocol functions (run_a, run_sensitivity, run_b) log every configuration they run, and
    on a real-data market it runs only inside one of them (PermissionError otherwise)."""
    scan = scan_config(market, strategy_id, params, windows)
    return run_candidates(market, scan.pairs(), scenario, scan.windows, contract=contract,
                          mode=SINGLE, embargo=embargo, strategy_id=strategy_id,
                          params=scan.params, selection_lines=scan.selection_lines)


# ---------------------------------------------------------------------------------------------
# Logging a configuration
# ---------------------------------------------------------------------------------------------

@dataclass
class Evaluation:
    """A configuration's runs over the concatenated test windows and its trial-log entry."""
    strategy_id: str
    params: dict[str, Any]
    runs: dict[str, ConfigRun]                 # scenario key ('STRESS-d2', 'BASE@competition')
    trial: tl.LogResult | None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def canonical(self) -> ConfigRun:
        return self.runs[PASS_SCENARIO]


def _record(market: MarketData, strategy_id: str, params: Mapping[str, Any], stage: str,
            windows: Sequence[Fold], scenarios: Mapping[str, Mapping[str, Any]],
            extra: Mapping[str, Any], *, protocol: str | None = None, variant: str = "",
            canonical: str | None = PASS_SCENARIO) -> tl.TrialRecord:
    ex = dict(extra)
    ex.setdefault("contracts", list(contracts(market.cfg)))
    if "competition" not in contracts(market.cfg):
        ex.setdefault("contract_note", CONTRACT_NOTE)
    ex.setdefault("data_source", market.source)
    return tl.TrialRecord(
        strategy_id=strategy_id, params=dict(params), stage=stage,
        protocol=protocol or entry_protocol(strategy_id), data_window=window_label(windows),
        shared_hash=market.shared_hash, config_hash=market.config_hash,
        data_hash=market.data_hash, scenarios=dict(scenarios), extra=ex, variant=variant,
        canonical=canonical)


def _write_and_log(rec: tl.TrialRecord, runs: Mapping[str, ConfigRun], *, log_path: Path,
                   runs_dir: Path, rerun_note: str | None = None,
                   sub_runs: Mapping[str, Mapping[str, ConfigRun]] | None = None
                   ) -> tl.LogResult:
    """Log the record and write its run files. A configuration already logged is merged
    first (raising TrialConflict before any file is touched), then only the new scenarios'
    files are written into its directory; the canonical files are never overwritten. A
    merge writes into the trial's EXISTING folder in any layout of trials.trial_locations()
    (e.g. runs_dir/RS-MOM-01/T0001), found before the log is touched, so it never creates a
    second flat runs_dir/<trial_id> copy; a new row's files go to trials.trial_dir()."""
    if tl.will_append(rec.key, log_path, rerun_note):
        tid = tl.reserve_trial_id(log_path)
        new = set(runs)
        new_sub = {k: set(v) for k, v in (sub_runs or {}).items()}
    else:
        tid = None
        new = set()
        new_sub = {}
    if tid is not None:
        _write_files(tl.trial_dir(tid, runs_dir), rec, runs, new, sub_runs or {}, new_sub)
        res = tl.log_trial(rec, path=log_path, rerun_note=rerun_note, trial_id=tid)
    else:
        target = str(tl.find_trial(rec.key, log_path)["trial_id"])
        d = tl.find_trial_dir(target, runs_dir, rec.strategy_id) or tl.trial_dir(target, runs_dir)
        res = tl.log_trial(rec, path=log_path, rerun_note=rerun_note)
        if res.trial_id != target:      # log_trial merges into the newest row of the key
            raise RuntimeError(f"merged into {res.trial_id}, expected {target}")
        merged = set(res.merged)
        new = {s for s in runs if s in merged}
        new_sub = {k: set(v) for k, v in (sub_runs or {}).items() if k in merged}
        _write_files(d, rec, runs, new, sub_runs or {}, new_sub)
    return res


def _write_files(d: Path, rec: tl.TrialRecord, runs: Mapping[str, ConfigRun], new: set,
                 sub_runs: Mapping[str, Mapping[str, ConfigRun]], new_sub: Mapping[str, set]
                 ) -> None:
    canon = rec.canonical
    if canon in runs and not (d / "daily.parquet").exists():
        r = runs[canon]
        tl.save_run_files(d, daily=r.daily, trades=r.trades,
                          selection=(rec.strategy_id, r.selection_lines))
    for name in sorted(new):
        r = runs[name]
        tl.save_run_files(d / "scenarios" / name, daily=r.daily, trades=r.trades,
                          equity=r.equity)
    for group, names in new_sub.items():
        for name, r in sub_runs[group].items():
            tl.save_run_files(d / group / name, daily=r.daily, trades=r.trades, equity=r.equity)


@_protocol
def evaluate_configuration(market: MarketData, strategy_id: str,
                           params: Mapping[str, Any] | None, *, stage: str,
                           scenarios: Sequence[str] = (PASS_SCENARIO,),
                           windows: Sequence[Fold] = TEST_FOLDS, log: bool = True,
                           log_path: Path = tl.TRIAL_LOG, runs_dir: Path = tl.RUNS_DIR,
                           rerun_note: str | None = None,
                           extra: Mapping[str, Any] | None = None,
                           sub_runs: Mapping[str, Mapping[str, ConfigRun]] | None = None
                           ) -> Evaluation:
    """Run a configuration under `scenarios` (every contract) over the concatenated test
    windows and log it (one row per configuration; scenarios in the row)."""
    if PASS_SCENARIO not in scenarios:
        raise ValueError(f"every logged configuration includes {PASS_SCENARIO}: its daily series "
                         "is the trial's series in the trial matrix")
    p = dict(param_grid(strategy_id).resolve(params))
    runs: dict[str, ConfigRun] = {}
    for c in contracts(market.cfg):
        for s in scenarios:
            runs[_scen_key(s, c)] = run_config(strategy_id, p, s, windows, market=market,
                                               contract=c)
    metrics = {k: r.metrics() for k, r in runs.items()}
    ex = dict(extra or {})
    for group, rs in (sub_runs or {}).items():
        ex[group] = {k: r.metrics() for k, r in rs.items()}
    trial = None
    if log:
        rec = _record(market, strategy_id, p, stage, windows, metrics, ex)
        trial = _write_and_log(rec, runs, log_path=log_path, runs_dir=runs_dir,
                               rerun_note=rerun_note, sub_runs=sub_runs)
        logger.info("%s %s %s -> %s%s", stage, strategy_id, tl.params_text(p), trial.trial_id,
                 " (new row)" if trial.created else f" (merged {list(trial.merged)})")
    return Evaluation(strategy_id, p, runs, trial, ex)


def ensure_logged(market: MarketData, strategy_id: str, params: Mapping[str, Any], *,
                  stage: str, windows: Sequence[Fold] = TEST_FOLDS,
                  log_path: Path = tl.TRIAL_LOG, runs_dir: Path = tl.RUNS_DIR,
                  rerun_note: str | None = None) -> str:
    """trial_id of the configuration, running and logging its canonical evaluation (test
    windows, STRESS-d2) with `stage` unless its newest row is current: same config and data
    hash, STRESS-d2 present and, with a rerun_note, written in that rerun session
    (trials.rerun_tag). Otherwise the evaluation is logged (a stale row without rerun_note
    raises TrialConflict; with one, a rerun row is appended and the earlier row kept)."""
    p = dict(param_grid(strategy_id).resolve(params))
    key = tl.config_key(strategy_id, p, market.shared_hash, entry_protocol(strategy_id), stage)
    row = tl.find_trial(key, log_path)
    if row is not None:
        n = row["notes_json"]
        current = (n.get("config_hash") == market.config_hash
                   and n.get("data_hash") == market.data_hash
                   and PASS_SCENARIO in n.get("scenarios", {}))
        this_session = rerun_note is None or tl.rerun_tag(n) == tl.rerun_text(rerun_note)
        if current and this_session:
            return str(row["trial_id"])
    ev = evaluate_configuration(market, strategy_id, p, stage=stage, windows=windows,
                                log_path=log_path, runs_dir=runs_dir, rerun_note=rerun_note)
    return ev.trial.trial_id


# ---------------------------------------------------------------------------------------------
# Run A
# ---------------------------------------------------------------------------------------------

@dataclass
class RunAResult:
    strategy_id: str
    params: dict[str, Any]
    runs: dict[str, ConfigRun]              # test windows, every scenario (and contract)
    secondary: dict[str, ConfigRun]         # 2023-03-01..2026-04-30, reported only
    shares: pd.DataFrame
    counts: Any                             # folds.TradeCountCheck (STRESS-d2, demo)
    competition: Any                        # folds.CompetitionFeasibility (STRESS-d2, demo)
    trial: tl.LogResult | None


@_protocol
def run_a(market: MarketData, strategy_id: str, *, scenarios: Sequence[str] = RUN_A_SCENARIOS,
          secondary: bool = True, windows: Sequence[Fold] = TEST_FOLDS,
          secondary_window: Fold = SECONDARY_SAMPLE, log: bool = True,
          log_path: Path = tl.TRIAL_LOG, runs_dir: Path = tl.RUNS_DIR,
          rerun_note: str | None = None) -> RunAResult:
    """Run A (COMMON 10.2): defaults over the concatenated test windows under every scenario,
    plus the secondary sample (reported only, in the row's extra), the 10.4 trade counts and
    the competition feasibility, logged as one row (stage run_a)."""
    grid = param_grid(strategy_id)
    p = dict(grid.defaults)
    sec: dict[str, ConfigRun] = {}
    if secondary:
        for c in contracts(market.cfg):
            for s in scenarios:
                sec[_scen_key(s, c)] = run_config(strategy_id, p, s, (secondary_window,),
                                                  market=market, contract=c)
    shares = fold_data_shares(market.cfg, market.features, tuple(windows), market.keys)
    ev = evaluate_configuration(market, strategy_id, p, stage="run_a", scenarios=scenarios,
                                windows=windows, log=False)
    # Counts and feasibility come from STRESS-d2 at the demo contract (COMMON 10.4).
    canon = ev.canonical
    counts = trade_count_check(canon.filled, shares, tuple(windows))
    comp = competition_feasibility(canon.filled, canon.daily["date"], market.cfg,
                                   data_days=market.data_days())
    extra = {"trade_counts": counts.to_dict(), "competition_feasibility": comp.to_dict(),
             "fold_data_shares": shares.to_dict(orient="records"),
             "secondary_window": secondary_window.label}
    if sec:
        extra["secondary"] = {k: r.metrics() for k, r in sec.items()}
    ev.trial = None
    if log:
        rec = _record(market, strategy_id, p, "run_a", windows,
                      {k: r.metrics() for k, r in ev.runs.items()}, extra)
        ev.trial = _write_and_log(rec, ev.runs, log_path=log_path, runs_dir=runs_dir,
                                  rerun_note=rerun_note, sub_runs={"secondary": sec} if sec
                                  else None)
    return RunAResult(strategy_id, p, ev.runs, sec, shares, counts, comp, ev.trial)


# ---------------------------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------------------------

def sensitivity_plan(grid: ParamGrid) -> pd.DataFrame:
    """Every grid cell (ParamGrid.configurations order) with flags: is_default, in_oat
    (one-at-a-time grid), in_heatmap, neighbour_of (the parameter whose default it neighbours,
    '' if none)."""
    oat = grid.one_at_a_time()
    heat = grid.heatmap_cells()

    def member(p, group) -> bool:
        return any(all(tl.canonical_params(p)[k] == tl.canonical_params(q)[k] for k in p)
                   for q in group)
    defaults = dict(grid.defaults)
    rows = []
    for order, p in enumerate(grid.configurations()):
        nb = ""
        diff = [k for k in p if tl.canonical_params({k: p[k]}) != tl.canonical_params(
            {k: defaults[k]})]
        if len(diff) == 1:
            k = diff[0]
            vals = list(grid.grid[k])
            i = [j for j, v in enumerate(vals) if tl.canonical_params({k: v}) ==
                 tl.canonical_params({k: p[k]})][0]
            d = [j for j, v in enumerate(vals) if tl.canonical_params({k: v}) ==
                 tl.canonical_params({k: defaults[k]})][0]
            if abs(i - d) == 1:
                nb = k
        rows.append({"order": order, "label": tl.params_text(p), "params": p,
                     "is_default": not diff, "in_oat": member(p, oat),
                     "in_heatmap": member(p, heat), "neighbour_of": nb, **p})
    return pd.DataFrame(rows)


def criterion5(table: pd.DataFrame) -> dict[str, Any]:
    """COMMON 10.5.5 on a sensitivity table (columns status, neighbour_of): every immediate
    neighbour of each default positive (with >= 100 trades), and >= 80% of all cells positive
    (insufficient cells count as failed)."""
    nb = table[table["neighbour_of"] != ""]
    nb_ok = bool(len(nb)) and bool((nb["status"] == POSITIVE).all())
    share = float((table["status"] == POSITIVE).mean()) if len(table) else float("nan")
    return {"neighbours": nb[["label", "status"]].to_dict(orient="records"),
            "neighbours_positive": nb_ok, "share_positive": share, "min_share": 0.8,
            "n_cells": int(len(table)),
            "n_insufficient": int((table["status"] == "insufficient").sum()),
            "passed": nb_ok and share >= 0.8}


@dataclass
class SensitivityResult:
    strategy_id: str
    scenario: str
    table: pd.DataFrame                     # one row per cell (sensitivity_plan + results)
    grid: ParamGrid
    runs: dict[str, ConfigRun] = field(repr=False)

    @property
    def criterion5(self) -> dict[str, Any]:
        return criterion5(self.table)

    def heatmap(self, value: str = "expectancy_usd") -> pd.DataFrame:
        """The 2-D heatmap (first heatmap parameter as rows) of `value`."""
        if self.grid.heatmap is None:
            raise ValueError(f"{self.strategy_id} has no heatmap pair")
        a, b = self.grid.heatmap
        h = self.table[self.table["in_heatmap"]]
        return h.pivot(index=a, columns=b, values=value)


@_protocol
def run_sensitivity(market: MarketData, strategy_id: str, *, scenario: str = PASS_SCENARIO,
                    windows: Sequence[Fold] = TEST_FOLDS, log: bool = True,
                    log_path: Path = tl.TRIAL_LOG, runs_dir: Path = tl.RUNS_DIR,
                    rerun_note: str | None = None) -> SensitivityResult:
    """Every grid configuration (one-at-a-time + 2-D heatmap) over the concatenated test
    windows under STRESS-d2, each logged (stage sensitivity; a configuration already logged by
    run_a or run_b_train is merged, not re-counted)."""
    if scenario != PASS_SCENARIO:
        raise ValueError(f"COMMON 10.4/10.5: sensitivity runs under {PASS_SCENARIO}")
    grid = param_grid(strategy_id)
    plan = sensitivity_plan(grid)
    out_rows, runs = [], {}
    for row in plan.to_dict(orient="records"):
        ev = evaluate_configuration(market, strategy_id, row["params"], stage="sensitivity",
                                    scenarios=(scenario,), windows=windows, log=log,
                                    log_path=log_path, runs_dir=runs_dir, rerun_note=rerun_note)
        r = ev.runs[scenario]
        m = r.metrics()
        runs[row["label"]] = r
        exp = m["expectancy_usd"]
        row.update(n=m["n_trades"], expectancy_usd=exp, expectancy_r=m["expectancy_r"],
                   sharpe_daily=m["sharpe_daily"], total_usd=m["total_usd"],
                   status=sensitivity_cell_status(m["n_trades"],
                                                  float("nan") if exp is None else exp),
                   trial_id=ev.trial.trial_id if ev.trial else None,
                   **{f"share_{k}": v for k, v in m["exit_shares"].items()})
        out_rows.append(row)
    table = pd.DataFrame(out_rows)
    for c in table.columns:
        if c.startswith("share_"):
            table[c] = table[c].fillna(0.0)
    return SensitivityResult(strategy_id, scenario, table, grid, runs)


# ---------------------------------------------------------------------------------------------
# Run B
# ---------------------------------------------------------------------------------------------

def _grid_pos(grid: ParamGrid, k: str, v: Any) -> int:
    vals = list(grid.grid[k])
    for i, g in enumerate(vals):
        if tl.canonical_params({k: g}) == tl.canonical_params({k: v}):
            return i
    raise ValueError(f"{k}={v!r} is not in its grid {vals}")


def grid_distance(params: Mapping[str, Any], grid: ParamGrid) -> int:
    """Sum over parameters of |grid position - default's grid position| (SPEC NOTE, ties)."""
    return sum(abs(_grid_pos(grid, k, params[k]) - _grid_pos(grid, k, grid.defaults[k]))
               for k in grid.defaults)


def select_params(points: Sequence[Mapping[str, Any]], sharpes: Sequence[float],
                  grid: ParamGrid) -> int:
    """Index of the chosen grid point: highest finite Sharpe; ties (within 1e-12 relative)
    -> smallest grid_distance -> earliest in `points`. All NaN -> the default point."""
    s = np.asarray(sharpes, dtype="float64")
    if len(s) != len(points) or not len(points):
        raise ValueError("points and sharpes must have the same non-zero length")
    fin = np.isfinite(s)
    if not fin.any():
        for i, p in enumerate(points):
            if grid_distance(p, grid) == 0:
                return i
        raise ValueError("every Sharpe is NaN and the default is not among the points")
    best = float(s[fin].max())
    tied = [i for i in range(len(points)) if fin[i]
            and (s[i] == best or abs(s[i] - best) <= SHARPE_TIE_REL * abs(best))]
    return min(tied, key=lambda i: (grid_distance(points[i], grid), i))


@dataclass
class FoldSelection:
    fold: int
    train: Fold
    test: Fold
    table: pd.DataFrame                     # order, label, sharpe, n_trades, n_embargoed, ...
    chosen: dict[str, Any]

    @property
    def chosen_label(self) -> str:
        return tl.params_text(self.chosen)

    def to_dict(self) -> dict[str, Any]:
        return {"fold": self.fold, "train": self.train.label, "test": self.test.label,
                "chosen": self.chosen_label,
                "grid": self.table.drop(columns=["params"]).to_dict(orient="records")}


@dataclass
class RunBResult:
    strategy_id: str
    selections: list[FoldSelection]
    runs: dict[str, ConfigRun]              # out-of-sample, per scenario key
    trial: tl.LogResult | None


@_protocol
def run_b(market: MarketData, strategy_id: str, *,
          folds: Sequence[WalkForwardFold] = WALK_FORWARD, scenario: str = SELECT_SCENARIO,
          canonical_windows: Sequence[Fold] = TEST_FOLDS, log: bool = True,
          log_path: Path = tl.TRIAL_LOG, runs_dir: Path = tl.RUNS_DIR,
          rerun_note: str | None = None) -> RunBResult:
    """Run B (COMMON 10.2). In each fold's train window every one-at-a-time grid point is
    backtested with the embargo and scored by daily Sharpe; the best (ties: closest to
    default) is applied to the next test window; the test windows' candidates are backtested
    as one run. Each grid point counts toward N_raw as its configuration: when logging, any
    point not yet in the log gets its canonical evaluation logged with stage run_b_train. The
    composite out-of-sample run is logged with stage run_b (not counted). canonical_windows:
    the windows of a configuration's canonical evaluation (the test windows; tests only
    shorten them). rerun_note (a bug-fix rerun): every grid point whose newest row was not
    written in this rerun session is re-evaluated and logged as a rerun row (earlier rows
    kept, N_raw unchanged), so the trial matrix reads the post-fix daily series."""
    grid = param_grid(strategy_id)
    points = grid.one_at_a_time()
    if log:
        for p in points:
            ensure_logged(market, strategy_id, p, stage="run_b_train",
                          windows=canonical_windows, log_path=log_path, runs_dir=runs_dir,
                          rerun_note=rerun_note)
    selections: list[FoldSelection] = []
    for wf in folds:
        rows, sharpes = [], []
        for i, p in enumerate(points):
            r = run_config(strategy_id, p, scenario, (wf.train,), market=market, embargo=True)
            sh = r.sharpe
            sharpes.append(sh)
            rows.append({"order": i, "label": tl.params_text(p), "params": p,
                         "distance": grid_distance(p, grid), "sharpe": sh,
                         "n_trades": int(len(r.filled)), "n_embargoed": int(len(r.embargoed)),
                         "total_usd": float(r.filled["pnl_usd"].sum())})
        k = select_params(points, sharpes, grid)
        table = pd.DataFrame(rows)
        table["chosen"] = table["order"] == k
        selections.append(FoldSelection(wf.n, wf.train, wf.test, table, dict(points[k])))
        logger.info("run B %s fold %d: chose %s (train Sharpe %.4f)", strategy_id, wf.n,
                 tl.params_text(points[k]), sharpes[k])

    tests = tuple(wf.test for wf in folds)
    pairs: list[StrategyCandidate] = []
    lines: list[tuple[int, str]] = []
    for sel in selections:
        sc = scan_config(market, strategy_id, sel.chosen, (sel.test,))
        pairs.extend(sc.pairs())
        lines.extend(sc.selection_lines)
    chosen = {f"fold{s.fold}": _bracket(s.chosen) for s in selections}
    runs: dict[str, ConfigRun] = {}
    for c in contracts(market.cfg):
        runs[_scen_key(scenario, c)] = run_candidates(
            market, pairs, scenario, tests, contract=c, strategy_id=strategy_id, params=chosen,
            selection_lines=lines)
    trial = None
    if log:
        metrics = {k: r.metrics() for k, r in runs.items()}
        rec = _record(market, strategy_id, chosen, "run_b", tests, metrics,
                      {"selections": [s.to_dict() for s in selections],
                       "selection_scenario": scenario},
                      variant=f"walk-forward selected, one-at-a-time grid, {scenario} train "
                              "Sharpe, embargo", canonical=scenario)
        trial = _write_and_log(rec, runs, log_path=log_path, runs_dir=runs_dir,
                               rerun_note=rerun_note)
    return RunBResult(strategy_id, selections, runs, trial)


# ---------------------------------------------------------------------------------------------
# Combined book (report only)
# ---------------------------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    strategy_ids: tuple[str, ...]
    runs: dict[str, ConfigRun]
    competition: dict[str, Any]             # per strategy: CompetitionFeasibility (STRESS-d2)
    trial: tl.LogResult | None


@_protocol
def run_portfolio(market: MarketData, strategy_ids: Sequence[str], *,
                  params: Mapping[str, Mapping[str, Any]] | None = None,
                  scenarios: Sequence[str] = (PASS_SCENARIO,),
                  windows: Sequence[Fold] = TEST_FOLDS, log: bool = True,
                  log_path: Path = tl.TRIAL_LOG, runs_dir: Path = tl.RUNS_DIR,
                  rerun_note: str | None = None) -> PortfolioResult:
    """Combined book (COMMON 8.8): the approved strategies (defaults unless `params`) share
    one book, one position per index, same-T priority SES-IBX-01, RS-MOM-01. Report only:
    logged with stage portfolio, never counted. Competition feasibility (10.4) per strategy."""
    for sid in strategy_ids:
        _entry(sid)
    ids = tuple(sorted(dict.fromkeys(strategy_ids), key=COMBINED_PRIORITY.index))
    if not ids:
        raise ValueError("no strategies for the combined book")
    pairs: list[StrategyCandidate] = []
    lines: list[tuple[int, str]] = []
    used: dict[str, str] = {}
    for sid in ids:
        sc = scan_config(market, sid, (params or {}).get(sid), windows)
        pairs.extend(sc.pairs())
        lines.extend(sc.selection_lines)
        used[sid] = _bracket(sc.params)
    name = "+".join(ids)
    runs: dict[str, ConfigRun] = {}
    for c in contracts(market.cfg):
        for s in scenarios:
            runs[_scen_key(s, c)] = run_candidates(market, pairs, s, windows, contract=c,
                                                   mode=COMBINED, strategy_id=name,
                                                   params=used, selection_lines=lines)
    comp = {}
    if PASS_SCENARIO in runs:
        r = runs[PASS_SCENARIO]
        for sid in ids:
            f = r.filled[r.filled["strategy_id"] == sid]
            comp[sid] = competition_feasibility(f, r.daily["date"], market.cfg,
                                                data_days=market.data_days())
    trial = None
    if log:
        metrics = {k: r.metrics() for k, r in runs.items()}
        rec = _record(market, name, used, "portfolio", windows, metrics,
                      {"competition_feasibility": {k: v.to_dict() for k, v in comp.items()}},
                      protocol="+".join(entry_protocol(s) for s in ids),
                      variant="combined book (COMMON 8.8)",
                      canonical=PASS_SCENARIO if PASS_SCENARIO in runs else None)
        trial = _write_and_log(rec, runs, log_path=log_path, runs_dir=runs_dir,
                               rerun_note=rerun_note)
    return PortfolioResult(ids, runs, comp, trial)


# ---------------------------------------------------------------------------------------------
# Symbol specs (read-only snapshot)
# ---------------------------------------------------------------------------------------------

def save_specs(specs: Mapping[str, SymbolSpec], path: Path) -> None:
    Path(path).write_text(json.dumps({k: dataclasses.asdict(v) for k, v in specs.items()},
                                     indent=2, sort_keys=True), encoding="utf-8")


def load_specs(path: Path) -> dict[str, SymbolSpec]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: SymbolSpec(**v) for k, v in raw.items()}


def fetch_specs(cfg: dict) -> dict[str, SymbolSpec]:
    """Current demo SymbolSpecs via the read-only adapter (symbol_info only)."""
    from core.config import load_credentials
    from data.mt5_adapter import MT5Client
    client = MT5Client(cfg)
    client.connect(load_credentials())
    try:
        names = client.resolve_symbols()
        return {k: client.spec(names[k]) for k in config_order(cfg)}
    finally:
        client.shutdown()


# ---------------------------------------------------------------------------------------------
# CLI (Run phase: real pre-holdout data)
# ---------------------------------------------------------------------------------------------

STEPS = ("run_a", "sensitivity", "run_b")


def main(argv: Sequence[str] | None = None) -> int:
    from core.config import load_config
    ap = argparse.ArgumentParser(description="COMMON 10 trial runner (pre-holdout data only)")
    ap.add_argument("--strategy", action="append", default=[],
                    help="strategy id (repeatable): RS-MOM-01, SES-IBX-01")
    ap.add_argument("--steps", default=",".join(STEPS),
                    help=f"comma-separated subset of {STEPS}")
    ap.add_argument("--portfolio", default=None,
                    help="comma-separated approved strategy ids for the combined book")
    ap.add_argument("--specs", default=None, help="SymbolSpec snapshot JSON (save_specs)")
    ap.add_argument("--snapshot-specs", default=None,
                    help="read the demo SymbolSpecs (read-only) into this JSON and exit")
    ap.add_argument("--rerun-note", default=None,
                    help="'<bug fix description>' for a rerun of logged configurations")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    if args.snapshot_specs:
        save_specs(fetch_specs(cfg), Path(args.snapshot_specs))
        print(f"SymbolSpecs written to {args.snapshot_specs}")
        return 0
    if not args.specs:
        ap.error("--specs is required (create it with --snapshot-specs)")
    specs = load_specs(Path(args.specs))
    market = load_market(cfg, specs)
    steps = [s for s in args.steps.split(",") if s]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        ap.error(f"unknown steps {bad}")
    for sid in args.strategy:
        _entry(sid)
        if "run_a" in steps:
            ra = run_a(market, sid, rerun_note=args.rerun_note)
            m = ra.runs[PASS_SCENARIO].metrics()
            print(f"{sid} run A ({ra.trial.trial_id if ra.trial else '-'}): "
                  f"{tl.summary_text({PASS_SCENARIO: m})}; counts passed {ra.counts.passed}")
            cf = ra.competition
            print(f"{sid} competition feasibility: share {cf.share:.3f} (all windows, passed "
                  f"{cf.passed}); share_with_data {cf.share_with_data} (windows with data); "
                  f"{COMP_RULING_NOTE}")
        if "sensitivity" in steps:
            se = run_sensitivity(market, sid, rerun_note=args.rerun_note)
            print(f"{sid} sensitivity: criterion 5 passed {se.criterion5['passed']} "
                  f"(share positive {se.criterion5['share_positive']:.2f})")
        if "run_b" in steps:
            rb = run_b(market, sid, rerun_note=args.rerun_note)
            m = rb.runs[_scen_key(SELECT_SCENARIO, 'demo')].metrics()
            print(f"{sid} run B ({rb.trial.trial_id if rb.trial else '-'}): "
                  f"{tl.summary_text({SELECT_SCENARIO: m})}")
    if args.portfolio:
        pr = run_portfolio(market, [s for s in args.portfolio.split(",") if s],
                           rerun_note=args.rerun_note)
        print(f"combined book {'+'.join(pr.strategy_ids)} logged as "
              f"{pr.trial.trial_id if pr.trial else '-'}")
    print(f"N_raw = {tl.n_raw()}")
    if "competition" not in contracts(cfg):
        print(CONTRACT_NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
