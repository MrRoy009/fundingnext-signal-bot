"""Pre-backtest feasibility gate (COMMON 10.1): setup counts and regime-gate diagnostics.

What it does, per strategy at its DEFAULT parameters, over the 9 walk-forward test windows of
COMMON 10.2:
  - runs the strategy's own scan() per test window and counts the candidates (setups) in total,
    by index, by side and by fold;
  - computes the COMMON 6.8 regime diagnostics for the strategy's gate: the share of in-window
    cycles the gate blocks (per index, per fold, per refit), flagged NON-FUNCTIONAL below 5% or
    above 95%, plus the per-refit model diagnostics of regime.diagnostics (K, label distribution,
    churn, |V_T| <= 0.05 share, agreement with sign(x2), seed stability);
  - applies the rule: fewer than 200 candidates in total -> DROPPED without a backtest.

Why it exists before any P&L code: the gate is a count of setup conditions, not an outcome. A
strategy that cannot produce enough setups cannot reach the trade minimum (COMMON 10.4), and a
gate that blocks almost nothing or almost everything makes every regime-conditional claim void
(COMMON 6.8). Deciding both from counts first keeps the decision free of outcome information.

What it never does: read a price to judge a candidate, simulate a fill, or compute P&L, returns,
win rate, expectancy or any other outcome statistic. Candidates come from scan(), which reads
only bars closed by each decision time T (its lockout replay, COMMON 7.1, reads bars up to each
later decision time only, never past it). Nothing here writes docs/trial_log.csv: COMMON 10.7
counts a trial from a configuration run with P&L, and setup counts are not one.

Public API
    Fold, TEST_FOLDS                      COMMON 10.2 test windows (9 folds)
    MIN_CANDIDATES                        COMMON 10.1 (200)
    GATE_BLOCK_MIN, GATE_BLOCK_MAX        COMMON 6.8 non-functional bounds (0.05, 0.95)
    default_strategies(cfg)               RS-MOM-01, SES-IBX-01, MR-TWAP-01 at default params
    scan_folds(strategy, features, tables, folds) -> FoldScan
    gate_cycles(scan, tables, gate, threshold) -> DataFrame  (one row per member-cycle)
    gate_summary(gc, by) -> DataFrame     block shares + flags
    gate_flag(share) -> str
    verdict(n) -> str                     PROCEED | DROPPED
    fold_data_shares(cfg, features, folds, keys) -> DataFrame
    models_without_tag(table, tag) -> list[str]
    refit_diagnostics_in_span(service, table, start, end, stability_seeds) -> DataFrame
    run_feasibility(cfg, features, tables, ...) -> FeasibilityReport
    FeasibilityReport.to_markdown() -> str
    write_selection_log(report, path)     COMMON 4.2.8 per-cycle log (gzip text)
    main(argv)                            CLI: real pre-holdout data, writes the markdown report

SPEC NOTE (fold windows): COMMON 10.2 gives the test windows as calendar dates; they are read
as UTC dates, [first 00:00 UTC, last + 1 day 00:00 UTC), and a candidate belongs to the fold of
its signal time T (10.2: "Trades belong to the fold of their signal time T"). Every strategy
window lies between 08:00 and 20:00 UTC, so a local-date reading gives the same assignment.
They are module constants because config.yaml does not carry them (and is not ours to edit).
SPEC NOTE (200 is not pro-rated): COMMON 10.1 drops a strategy with "fewer than 200 candidates
in total". Amendment A1 (12A.1) pro-rates only "the fold minimum (10.4)" by the share of
trading days with data, and the 10.4 total of 150 trades the 200 is meant to protect is not
pro-rated either. The verdict therefore uses the literal, unscaled 200. The report still shows
each fold's data share, a per-full-fold equivalent count and, for information only, what a
pro-rated threshold would be, so Quant can rule on it.
SPEC NOTE (data share of a fold): "share of trading days with data" = (dates in the fold that
are exchange-calendar trading dates for at least one ranked index AND are valid trading dates,
cash-open M15 bar present, for at least one such index) / (calendar trading dates in the fold,
union over the ranked indices). Per-index shares are reported alongside.
SPEC NOTE (in-window cycles): a member-cycle is (T, index) with the index in G(T) (its window
contains T on a valid trading date, as the strategy's own cycles() builds it); indices outside
G(T) because the cash-open bar is missing are not cycles. Incomplete-group cycles (COMMON
4.2.2) are kept: the gate is a regime property and is evaluated at every in-window cycle.
SPEC NOTE (what "blocks" means): the gate passes iff regime status OK and the V_T condition
holds (COMMON 6.7), so a cycle with status UNAVAILABLE is blocked. The flag of 6.8 uses that
literal share. The share among status-OK cycles (V_T condition alone) is reported next to it
and flagged the same way, so a gate that only "works" because the model is often unavailable
is visible.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import io
import logging
import math
import re
import subprocess
import sys
import time as _time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from core.calendar import exchange_for, is_calendar_trading_date
from strategies.base import Candidate
from strategies.context import _get_fx, regime_frame
from strategies.selection import BUY, SELL, Selection, config_order

log = logging.getLogger(__name__)

MIN_CANDIDATES = 200                   # COMMON 10.1
GATE_BLOCK_MIN, GATE_BLOCK_MAX = 0.05, 0.95   # COMMON 6.8
PROCEED, DROPPED = "PROCEED", "DROPPED"
NO_MODEL = "no model"


# ---------------------------------------------------------------------------------------------
# Folds (COMMON 10.2, amendment A1)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    """One walk-forward test window: calendar dates first..last, both inclusive."""
    n: int
    first: date
    last: date

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.first).tz_localize("UTC").as_unit("ns")

    @property
    def end(self) -> pd.Timestamp:
        """Exclusive end: 00:00 UTC of the day after `last`."""
        return pd.Timestamp(self.last + timedelta(days=1)).tz_localize("UTC").as_unit("ns")

    @property
    def label(self) -> str:
        return f"{self.first.isoformat()}..{self.last.isoformat()}"

    def contains(self, t) -> bool:
        t = pd.Timestamp(t)
        return self.start <= t < self.end


_FOLD_DATES = (("2024-02-01", "2024-04-30"), ("2024-05-01", "2024-07-31"),
               ("2024-08-01", "2024-10-31"), ("2024-11-01", "2025-01-31"),
               ("2025-02-01", "2025-04-30"), ("2025-05-01", "2025-07-31"),
               ("2025-08-01", "2025-10-31"), ("2025-11-01", "2026-01-31"),
               ("2026-02-01", "2026-04-30"))
TEST_FOLDS: tuple[Fold, ...] = tuple(
    Fold(n, date.fromisoformat(a), date.fromisoformat(b)) for n, (a, b) in enumerate(_FOLD_DATES, 1))


def fold_of(t, folds: Sequence[Fold] = TEST_FOLDS) -> int | None:
    for f in folds:
        if f.contains(t):
            return f.n
    return None


def fold_data_shares(cfg: dict, features: Any, folds: Sequence[Fold] = TEST_FOLDS,
                     keys: Sequence[str] | None = None) -> pd.DataFrame:
    """Share of trading days with data per fold (see the SPEC NOTE). Whole-date validity
    (is_valid_date_eod) is fine here: this is a report on finished dates, never a decision."""
    keys = list(keys or config_order(cfg))
    rows = []
    for f in folds:
        days = [f.first + timedelta(days=i) for i in range((f.last - f.first).days + 1)]
        cal_any: set[date] = set()
        data_any: set[date] = set()
        row: dict[str, Any] = {"fold": f.n, "window": f.label}
        for k in keys:
            ex = exchange_for(cfg, k)
            fx = _get_fx(features, k)
            cal = [d for d in days if is_calendar_trading_date(ex.calendar, d)]
            have = [d for d in cal if fx is not None and fx.is_valid_date_eod(d)]
            cal_any.update(cal)
            data_any.update(have)
            row[f"share_{k}"] = len(have) / len(cal) if cal else float("nan")
        row["trading_days"] = len(cal_any)
        row["days_with_data"] = len(data_any)
        row["data_share"] = len(data_any) / len(cal_any) if cal_any else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Strategies at default parameters
# ---------------------------------------------------------------------------------------------

def default_strategies(cfg: dict) -> list:
    """The three GATE 2 strategies at their default parameters."""
    from strategies.mr_twap_01 import MrTwap01
    from strategies.rs_mom_01 import RsMom01
    from strategies.ses_ibx_01 import SesIbx01
    return [RsMom01(cfg), SesIbx01(cfg), MrTwap01(cfg)]


def _check_defaults(strategy) -> None:
    """COMMON 10.1 runs each strategy at its default parameters, nothing else."""
    grid = strategy.param_grid
    if dict(strategy.params) != dict(grid.resolve(None)):
        raise ValueError(f"{strategy.id}: the feasibility gate runs DEFAULT parameters only "
                         f"({dict(grid.defaults)}), got {dict(strategy.params)}")


# ---------------------------------------------------------------------------------------------
# Scan per fold
# ---------------------------------------------------------------------------------------------

@dataclass
class FoldScan:
    """Candidates and per-cycle selection records of one strategy over the test folds."""
    strategy_id: str
    gate: str | None
    params: Mapping[str, Any]
    folds: tuple[Fold, ...]
    candidates: list[Candidate]
    cand_fold: list[int]
    cycles: pd.DataFrame            # T, fold, group, key, incomplete (one row per member-cycle)
    excluded: pd.DataFrame          # T, fold, key, reason (in window, outside G(T))
    selection_lines: list[tuple[int, str]] = field(repr=False)   # (fold, Selection.log_line())
    n_group_cycles: int = 0
    n_incomplete_group_cycles: int = 0

    def candidate_frame(self) -> pd.DataFrame:
        """key, side, fold, signal_time of each candidate: no price field on purpose."""
        return pd.DataFrame({"key": [c.key for c in self.candidates],
                             "side": [c.side for c in self.candidates],
                             "fold": self.cand_fold,
                             "signal_time": [c.signal_time for c in self.candidates]})


def scan_folds(strategy, features: Any, tables: Mapping[str, Any] | None,
               folds: Sequence[Fold] = TEST_FOLDS) -> FoldScan:
    """strategy.scan() over each fold [start, end), with every Selection kept (COMMON 4.2.8).

    One scan per fold rather than one long scan: scan() replays lockouts from the local
    midnight of `start`, and no lockout state crosses a local date, so this gives exactly the
    candidates of a full-history scan inside each window (checked by the tests)."""
    _check_defaults(strategy)
    cands: list[Candidate] = []
    cand_fold: list[int] = []
    rows, excl, lines = [], [], []
    n_groups = n_incomplete = 0
    for f in folds:
        sels: list[Selection] = []
        got = strategy.scan(features, tables, f.start, f.end, on_selection=sels.append)
        for c in got:
            if not f.contains(c.signal_time):
                raise AssertionError(f"{strategy.id}: candidate at {c.signal_time} outside "
                                     f"fold {f.n} {f.label}")
        cands.extend(got)
        cand_fold.extend([f.n] * len(got))
        for s in sels:
            n_groups += 1
            n_incomplete += bool(s.incomplete)
            lines.append((f.n, s.log_line()))
            group = "+".join(s.members)
            for k in s.members:
                rows.append((s.T, f.n, group, k, bool(s.incomplete)))
            for k, why in s.excluded.items():
                excl.append((s.T, f.n, k, why))
    cyc = pd.DataFrame(rows, columns=["T", "fold", "group", "key", "incomplete"])
    if len(cyc):
        cyc["T"] = pd.DatetimeIndex(pd.to_datetime(cyc["T"], utc=True)).as_unit("ns")
    return FoldScan(strategy.id, strategy.regime_gate, dict(strategy.params), tuple(folds),
                    cands, cand_fold, cyc,
                    pd.DataFrame(excl, columns=["T", "fold", "key", "reason"]), lines,
                    n_groups, n_incomplete)


# ---------------------------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------------------------

def verdict(n_candidates: int) -> str:
    """COMMON 10.1: fewer than MIN_CANDIDATES in total -> DROPPED without a backtest."""
    return DROPPED if n_candidates < MIN_CANDIDATES else PROCEED


def count_tables(scan: FoldScan, keys: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Candidate counts: by index x side, and by fold x (index, side)."""
    cf = scan.candidate_frame()
    sides = [BUY, SELL]
    by_key = pd.DataFrame(0, index=list(keys), columns=sides, dtype=int)
    for (k, s), n in cf.groupby(["key", "side"]).size().items():
        by_key.loc[k, s] = int(n)
    by_key["total"] = by_key[sides].sum(axis=1)
    by_key.loc["ALL"] = by_key.sum(axis=0)
    fold_rows = []
    for f in scan.folds:
        part = cf[cf["fold"] == f.n]
        row = {"fold": f.n, "window": f.label, "candidates": len(part)}
        for k in keys:
            row[k] = int((part["key"] == k).sum())
        for s in sides:
            row[s] = int((part["side"] == s).sum())
        fold_rows.append(row)
    return {"by_key": by_key, "by_fold": pd.DataFrame(fold_rows)}


# ---------------------------------------------------------------------------------------------
# Regime gate diagnostics (COMMON 6.8)
# ---------------------------------------------------------------------------------------------

_COUNT_SUFFIX = re.compile(r"\s*\(\d[\d,]*\)$")


def _cause_kind(cause: str) -> str:
    """Regime cause without the per-refit row count, so causes aggregate."""
    return _COUNT_SUFFIX.sub("", str(cause or ""))


def _fit_times(table: Any, times: pd.DatetimeIndex) -> np.ndarray:
    """Fit time of the model whose filter answers at each T (NaT: no model). Stand-in tables
    without a fit_time column give NaT everywhere."""
    if table is None or not len(times):
        return np.full(len(times), np.datetime64("NaT", "ns"))
    out = table.at_many(times)
    if "fit_time" not in out.columns:
        return np.full(len(times), np.datetime64("NaT", "ns"))
    ft = pd.to_datetime(out["fit_time"], utc=True)
    return ft.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")


def gate_cycles(scan: FoldScan, tables: Mapping[str, Any] | None, gate: str | None,
                threshold: float) -> pd.DataFrame:
    """Regime status and gate at every member-cycle of the scan, via the strategy driver's own
    regime_frame (at_many with the live rule, COMMON 6.6), so the diagnostic sees exactly the
    gate the strategy saw."""
    parts = []
    for key, part in scan.cycles.groupby("key", sort=False):
        times = pd.DatetimeIndex(part["T"])
        table = (tables or {}).get(key)
        rf = regime_frame(table, times, gate, threshold)
        df = part.reset_index(drop=True).copy()
        df["regime_ok"] = rf["ok"].to_numpy(dtype=bool)
        df["gate_pass"] = rf["passes"].to_numpy(dtype=bool)
        df["cause"] = [_cause_kind(c) for c in rf["cause"]]
        df["v_t"] = rf["v_t"].to_numpy(dtype=float)
        df["fit_time"] = pd.to_datetime(_fit_times(table, times), utc=True)
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["T", "fold", "group", "key", "incomplete", "regime_ok",
                                     "gate_pass", "cause", "v_t", "fit_time"])
    return pd.concat(parts, ignore_index=True).sort_values(["T", "key"], kind="stable") \
        .reset_index(drop=True)


def gate_flag(share: float) -> str:
    """COMMON 6.8: a gate blocking < 5% or > 95% of in-window cycles is non-functional."""
    if share is None or not math.isfinite(share):
        return "n/a"
    if share < GATE_BLOCK_MIN:
        return "NON-FUNCTIONAL (< 5%)"
    if share > GATE_BLOCK_MAX:
        return "NON-FUNCTIONAL (> 95%)"
    return "functional"


def _summ(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    n_ok = int(df["regime_ok"].sum())
    n_pass = int(df["gate_pass"].sum())
    share = (n - n_pass) / n if n else float("nan")
    share_ok = (n_ok - n_pass) / n_ok if n_ok else float("nan")
    return {"cycles": n, "status_ok": n_ok, "unavailable": n - n_ok,
            "blocked_by_vt": n_ok - n_pass, "blocked": n - n_pass, "block_share": share,
            "flag": gate_flag(share), "block_share_ok": share_ok,
            "flag_ok": gate_flag(share_ok),
            "unavailable_share": (n - n_ok) / n if n else float("nan")}


def gate_summary(gc: pd.DataFrame, by: str | Sequence[str] | None = "key",
                 total: bool = True) -> pd.DataFrame:
    """Block shares per group of member-cycles, plus an ALL row (by=None: one ALL row)."""
    rows = []
    if by is not None:
        for name, part in gc.groupby(by, sort=True, dropna=False):
            r = {"group": name}
            r.update(_summ(part))
            rows.append(r)
    if total or by is None:
        r = {"group": "ALL"}
        r.update(_summ(gc))
        rows.append(r)
    return pd.DataFrame(rows)


def gate_by_refit(gc: pd.DataFrame) -> pd.DataFrame:
    """Block share per (index, model fit time) over the in-window cycles that model served."""
    rows = []
    g = gc.assign(fit=gc["fit_time"].astype(object).where(gc["fit_time"].notna(), NO_MODEL))
    for (key, fit), part in g.groupby(["key", "fit"], sort=False, dropna=False):
        r = {"key": key, "fit_time": fit}
        r.update(_summ(part))
        rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out["_o"] = [pd.Timestamp.max.tz_localize("UTC") if f == NO_MODEL else pd.Timestamp(f)
                     for f in out["fit_time"]]
        out = out.sort_values(["key", "_o"], kind="stable").drop(columns="_o")
    return out.reset_index(drop=True)


def unavailable_causes(gc: pd.DataFrame) -> pd.Series:
    """Member-cycles with regime status not OK, by cause (row counts stripped)."""
    bad = gc[~gc["regime_ok"]]
    return bad.groupby(["key", "cause"]).size() if len(bad) else pd.Series(dtype=int)


def models_without_tag(table: Any, tag: str, start=None, end=None) -> list[str]:
    """Model ids with no state of volatility tag `tag` (ACTIVE / CALM): the gate cannot pass
    while such a model serves (MR-TWAP-01 spec "Regime condition"; for CALM this is
    strategies.mr_twap_01.models_without_calm). Restricted to intervals overlapping
    [start, end) when given."""
    out = []
    for iv in getattr(table, "intervals", None) or []:
        m = iv.model
        if m is None:
            continue
        if start is not None and iv.end is not None and iv.end <= start:
            continue
        if end is not None and iv.start >= end:
            continue
        if tag not in list(m.vol_tags) and m.model_id not in out:
            out.append(m.model_id)
    return out


def refit_diagnostics_in_span(service: Any, table: Any, start: pd.Timestamp, end: pd.Timestamp,
                              stability_seeds=None) -> pd.DataFrame:
    """regime.diagnostics.refit_diagnostics restricted to the refits serving [start, end).

    The latest model before the span is kept in the run (then dropped from the output) so the
    first in-span refit still gets its label churn. `service` provides .rc and .fit; pass a
    RegimeService with its own model_dir (or None) so the seed-stability refits never write to
    the production cache."""
    from regime.diagnostics import refit_diagnostics
    ivs = list(getattr(table, "intervals", None) or [])
    keep, before = [], None
    for iv in ivs:
        if iv.model is None or iv.fallback or iv.model.fit_time != iv.start:
            continue
        if iv.end is not None and iv.end <= start:
            before = iv
            continue
        if iv.start >= end:
            continue
        keep.append(iv)
    run = ([before] if before is not None else []) + keep
    sub = dataclasses.replace(table, intervals=run)
    out = refit_diagnostics(service, sub, stability_seeds)
    if before is not None and len(out):
        out = out[pd.to_datetime(out["fit_time_utc"], utc=True) != before.start]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------

@dataclass
class StrategyFeasibility:
    strategy_id: str
    gate: str | None
    params: Mapping[str, Any]
    scan: FoldScan
    counts: dict[str, pd.DataFrame]
    gc: pd.DataFrame
    gate_by_key: pd.DataFrame
    gate_by_fold: pd.DataFrame
    gate_by_refit: pd.DataFrame
    causes: pd.Series
    models_without_gate_state: dict[str, list[str]]

    @property
    def total(self) -> int:
        return len(self.scan.candidates)

    @property
    def verdict(self) -> str:
        return verdict(self.total)

    @property
    def gate_share(self) -> float:
        all_row = self.gate_by_key[self.gate_by_key["group"] == "ALL"]
        return float(all_row["block_share"].iloc[0]) if len(all_row) else float("nan")

    @property
    def gate_flag(self) -> str:
        return gate_flag(self.gate_share)


@dataclass
class FeasibilityReport:
    keys: tuple[str, ...]
    folds: tuple[Fold, ...]
    shares: pd.DataFrame
    strategies: list[StrategyFeasibility]
    refit_diag: dict[str, pd.DataFrame]      # key -> per-refit regime diagnostics (span only)
    provenance: dict[str, str] = field(default_factory=dict)
    seed_stability: bool = True

    def strategy(self, sid: str) -> StrategyFeasibility:
        for s in self.strategies:
            if s.strategy_id == sid:
                return s
        raise KeyError(sid)

    def to_markdown(self) -> str:
        return render_markdown(self)


def run_feasibility(cfg: dict, features: Any, tables: Mapping[str, Any], *,
                    strategies: Sequence | None = None, folds: Sequence[Fold] = TEST_FOLDS,
                    stability_service: Any = None, stability_seeds=None,
                    refit_diag: bool = True,
                    provenance: Mapping[str, str] | None = None) -> FeasibilityReport:
    """The whole gate on already-built services (FeatureService, {key: RegimeTable}).

    stability_service: object with .rc and .fit used for the per-refit diagnostics (a
    RegimeService with a scratch model_dir and refit_log None). refit_diag False skips those
    diagnostics (stand-in tables have no models); stability_seeds () skips only the seed refits."""
    keys = config_order(cfg)
    folds = tuple(folds)
    thr = float(cfg["regime"]["gate_threshold"])
    strategies = list(strategies) if strategies is not None else default_strategies(cfg)
    span_start, span_end = min(f.start for f in folds), max(f.end for f in folds)
    out = []
    for strat in strategies:
        t0 = _time.time()
        sc = scan_folds(strat, features, tables, folds)
        gc = gate_cycles(sc, tables, strat.regime_gate, thr)
        missing = {}
        if strat.regime_gate in ("ACTIVE", "CALM"):
            for k in keys:
                missing[k] = models_without_tag((tables or {}).get(k), strat.regime_gate,
                                                span_start, span_end)
        out.append(StrategyFeasibility(
            strat.id, strat.regime_gate, dict(strat.params), sc, count_tables(sc, keys), gc,
            gate_summary(gc, "key"), gate_summary(gc, "fold"), gate_by_refit(gc),
            unavailable_causes(gc), missing))
        log.info("%s: %d candidates in %d folds (%.1fs)", strat.id, len(sc.candidates),
                 len(folds), _time.time() - t0)
    diag: dict[str, pd.DataFrame] = {}
    if refit_diag and stability_service is not None:
        for k in keys:
            table = (tables or {}).get(k)
            if table is None or not hasattr(table, "intervals"):
                continue
            t0 = _time.time()
            diag[k] = refit_diagnostics_in_span(stability_service, table, span_start, span_end,
                                                stability_seeds)
            log.info("%s: regime refit diagnostics (%.1fs)", k, _time.time() - t0)
    seeds_on = stability_seeds is None or len(tuple(stability_seeds)) > 0
    return FeasibilityReport(keys, folds, fold_data_shares(cfg, features, folds, keys), out,
                             diag, dict(provenance or {}), seeds_on and bool(diag))


def write_selection_log(report: FeasibilityReport, path: Path) -> int:
    """Persist every cycle's Selection.log_line() (COMMON 4.2.8), gzip text, one line per cycle:
    '<strategy>\\t<fold>\\t<line>'. Returns the number of lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    # mtime=0: the gzip header carries no clock, so the file (and its SHA-256 in the report)
    # is identical across reruns on the same data.
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, \
            io.TextIOWrapper(gz, encoding="utf-8", newline="\n") as f:
        for s in report.strategies:
            for fold, line in s.scan.selection_lines:
                f.write(f"{s.strategy_id}\t{fold}\t{line}\n")
                n += 1
    return n


# ---------------------------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------------------------

def _pct(x: Any) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(x) else f"{100 * x:.1f}%"


def _cell(x: Any) -> str:
    if isinstance(x, pd.Timestamp):
        return f"{x:%Y-%m-%d %H:%M}"
    if isinstance(x, float):
        return "n/a" if not math.isfinite(x) else f"{x:.2f}"
    return str(x).replace("|", "/")


def _md(df: pd.DataFrame, cols: Mapping[str, str], fmt: Mapping[str, Any] | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols.values()) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for _, r in df.iterrows():
        body.append("| " + " | ".join(fmt.get(c, _cell)(r[c]) for c in cols) + " |")
    return "\n".join([head, sep] + body)


def _gate_table(df: pd.DataFrame, first: str) -> str:
    return _md(df, {"group": first, "cycles": "In-window cycles", "status_ok": "Status OK",
                    "unavailable_share": "Unavailable share", "blocked": "Blocked",
                    "block_share": "Block share", "flag": "5-95% check",
                    "block_share_ok": "Block share among OK", "flag_ok": "Check (OK only)"},
               {"unavailable_share": _pct, "block_share": _pct, "block_share_ok": _pct})


def render_markdown(rep: FeasibilityReport) -> str:
    keys = list(rep.keys)
    L: list[str] = []
    a = L.append
    a("# Feasibility gate (COMMON 10.1)")
    a("")
    a("Descriptive pre-backtest gate, run before any P&L code. Each strategy at its DEFAULT "
      "parameters, over the 9 walk-forward test windows of COMMON 10.2: setup counts "
      "(candidates from the strategy's own `scan()`) and the COMMON 6.8 regime diagnostics of "
      "its gate. No fills, no prices after T, no P&L, returns, win rate, expectancy or any other "
      "outcome statistic was computed. The rule: fewer than "
      f"{MIN_CANDIDATES} candidates in total -> DROPPED without a backtest.")
    a("")
    a("Generated by `backtest/feasibility.py` (`python -m backtest.feasibility`).")
    if rep.provenance:
        a("")
        for k, v in rep.provenance.items():
            a(f"- {k}: {v}")
    a("")
    a("## Verdicts")
    a("")
    vrows = pd.DataFrame([{
        "sid": s.strategy_id, "params": ", ".join(f"{k}={v}" for k, v in s.params.items()),
        "gate": s.gate or "none", "total": s.total, "verdict": s.verdict,
        "share": s.gate_share, "flag": s.gate_flag} for s in rep.strategies])
    a(_md(vrows, {"sid": "Strategy", "params": "Default parameters", "gate": "Gate",
                  "total": "Candidates (9 test windows)", "verdict": "Verdict (>= 200)",
                  "share": "Gate block share", "flag": "COMMON 6.8 gate flag"},
          {"share": _pct}))
    a("")
    for s in rep.strategies:
        if s.verdict == DROPPED:
            a(f"- **{s.strategy_id}: DROPPED without a backtest** ({s.total} < {MIN_CANDIDATES} "
              "candidates in the 9 test windows).")
        else:
            a(f"- **{s.strategy_id}: PROCEEDS to the backtest** ({s.total} >= {MIN_CANDIDATES}).")
        if s.gate_flag.startswith("NON-FUNCTIONAL"):
            a(f"  - Its {s.gate} gate is **non-functional** ({s.gate_flag}): no "
              "regime-conditional claim may be made for this strategy (COMMON 6.8).")
        idx_flags = s.gate_by_key[(s.gate_by_key["group"] != "ALL")
                                  & s.gate_by_key["flag"].str.startswith("NON")]
        for _, r in idx_flags.iterrows():
            a(f"  - Per-index flag: {r['group']} {r['flag']} ({_pct(r['block_share'])}).")
        ok_flags = s.gate_by_key[s.gate_by_key["flag_ok"].str.startswith("NON")]
        for _, r in ok_flags.iterrows():
            a(f"  - Among status-OK cycles only: {r['group']} {r['flag_ok']} "
              f"({_pct(r['block_share_ok'])}).")
    a("")
    a("Nothing was added to `docs/trial_log.csv`: COMMON 10.7 counts a trial from a "
      "configuration run with P&L on pre-holdout data, and these are setup counts only.")
    a("")

    # --- data per fold ------------------------------------------------------------------------
    a("## Test windows and data coverage (COMMON 10.2, amendment A1)")
    a("")
    a("Data share = trading days with data / calendar trading days in the fold (union over "
      f"{', '.join(keys)}; a day has data when at least one index has its cash-open M15 bar). "
      "A1 drops 2024-10-04 to 2025-01-08 (frozen prices), so fold 4 has no data and fold 3 "
      "loses October; the 2025 data hole cuts folds 6 and 7.")
    a("")
    cols = {"fold": "Fold", "window": "Window", "trading_days": "Trading days",
            "days_with_data": "Days with data", "data_share": "Data share"}
    cols.update({f"share_{k}": f"{k} share" for k in keys})
    a(_md(rep.shares, cols, {"data_share": _pct, **{f"share_{k}": _pct for k in keys}}))
    a("")
    mean_share = float(np.nanmean(rep.shares["data_share"])) if len(rep.shares) else float("nan")
    a(f"Mean data share over the {len(rep.folds)} folds: {_pct(mean_share)}. For information "
      f"only (see SPEC NOTE 2), a threshold pro-rated by that share would be "
      f"{MIN_CANDIDATES * mean_share:.0f}; the verdicts above use the literal {MIN_CANDIDATES}.")
    a("")

    # --- per strategy -------------------------------------------------------------------------
    shares = rep.shares.set_index("fold")["data_share"]
    for s in rep.strategies:
        sc = s.scan
        a(f"## {s.strategy_id}")
        a("")
        a(f"Gate {s.gate}; parameters {', '.join(f'{k}={v}' for k, v in s.params.items())}. "
          f"**{s.total} candidates: {s.verdict}.** {sc.n_group_cycles} group cycles, of which "
          f"{sc.n_incomplete_group_cycles} incomplete (COMMON 4.2.2); {len(sc.cycles)} "
          f"member-cycles; {len(sc.excluded)} index-cycles outside G(T) (cash-open bar missing).")
        a("")
        a("### Candidates by index and side")
        a("")
        bk = s.counts["by_key"].reset_index().rename(columns={"index": "key"})
        a(_md(bk, {"key": "Index", BUY: "BUY", SELL: "SELL", "total": "Total"}))
        a("")
        a("### Candidates by fold")
        a("")
        bf = s.counts["by_fold"].copy()
        bf["data_share"] = [shares.get(n, float("nan")) for n in bf["fold"]]
        bf["per_full_fold"] = [c / sh if sh and math.isfinite(sh) and sh > 0 else float("nan")
                               for c, sh in zip(bf["candidates"], bf["data_share"])]
        cols = {"fold": "Fold", "window": "Window", "data_share": "Data share",
                "candidates": "Candidates"}
        cols.update({k: k for k in keys})
        cols.update({BUY: "BUY", SELL: "SELL", "per_full_fold": "Per full-fold equivalent"})
        a(_md(bf, cols, {"data_share": _pct,
                         "per_full_fold": lambda x: "n/a" if not math.isfinite(x)
                         else f"{x:.1f}"}))
        a("")
        a(f"### Regime gate ({s.gate}) at in-window cycles (COMMON 6.8)")
        a("")
        a("Per index. Blocked = the literal gate fails (status OK and the V_T condition, so "
          "UNAVAILABLE blocks). The COMMON 6.8 verdict for the strategy is the ALL row "
          f"(**{s.gate_flag}**, {_pct(s.gate_share)}); the per-index and per-fold checks are "
          "descriptive.")
        a("")
        a(_gate_table(s.gate_by_key, "Index"))
        a("")
        a("Per fold (a fold whose cycles are all regime-UNAVAILABLE shows 100%: no model or "
          "warm-up, not a V_T decision; those folds have no candidate):")
        a("")
        a(_gate_table(s.gate_by_fold, "Fold"))
        a("")
        if len(s.causes):
            a("Regime status not OK at in-window cycles, by cause:")
            a("")
            cz = s.causes.reset_index()
            cz.columns = ["key", "cause", "n"]
            a(_md(cz, {"key": "Index", "cause": "Cause", "n": "Member-cycles"}))
            a("")
        if s.models_without_gate_state:
            bad = {k: v for k, v in s.models_without_gate_state.items() if v}
            if bad:
                for k, v in bad.items():
                    a(f"- {k}: models with no {s.gate}-tagged state in the test span (the "
                      f"strategy is inactive for {k} while they serve): {', '.join(v)}")
            else:
                a(f"Every model serving the test span has at least one {s.gate}-tagged state "
                  f"on every index, so the {s.gate} gate can pass under each of them.")
            a("")

    # --- per refit ----------------------------------------------------------------------------
    a("## Per-index, per-refit regime diagnostics (COMMON 6.8)")
    a("")
    a("Model diagnostics over the core observations each refit served (regime.diagnostics; "
      "the warm-up observations of amendment A2 are left out), restricted to refits serving the "
      "test span, followed by each strategy's block share over the in-window cycles that refit "
      "served. Label distribution, |V_T| <= 0.05 and agreement with sign(x2) are the same for "
      "the ACTIVE and the CALM gate (CALM is the complement of ACTIVE; agreement is symmetric). "
      "Churn = share of the served observations whose argmax label differs from the previous "
      "model's. Seed stability = gate agreement of the seeds 10..19 refit with the main model. "
      "A refit with 0 core obs was fitted but served only excluded-window, data-hole or "
      "warm-up time (amendments A1, A2); its in-window cycles are all UNAVAILABLE.")
    if not rep.seed_stability:
        a("")
        a("Seed-stability refits were skipped in this run (column n/a).")
    a("")
    for k in keys:
        d = rep.refit_diag.get(k)
        a(f"### {k}")
        a("")
        if d is None or not len(d):
            a("No refit diagnostics (no fitted model served the test span, or diagnostics "
              "were not run).")
            a("")
            continue
        d = d.copy()
        d["fit"] = pd.to_datetime(d["fit_time_utc"], utc=True)
        cols = {"fit": "Refit", "k": "K", "n_core_obs": "Core obs",
                "n_warmup_obs": "Warm-up obs", "label_dist": "Labels",
                "share_active": "ACTIVE share", "share_v_near_zero": "abs(V_T) <= 0.05",
                "agree_sign_x2": "Agree sign(x2)", "label_churn": "Churn",
                "seed_stability": "Seed stability"}
        fmt: dict[str, Any] = {c: _pct for c in ("share_active", "share_v_near_zero",
                                                 "agree_sign_x2", "label_churn",
                                                 "seed_stability")}
        fmt["label_dist"] = _labels
        for s in rep.strategies:
            br = s.gate_by_refit[s.gate_by_refit["key"] == k]
            m = {pd.Timestamp(f): (n, sh) for f, n, sh in
                 zip(br["fit_time"], br["cycles"], br["block_share"]) if f != NO_MODEL}
            col = f"blk_{s.strategy_id}"
            d[col] = [m.get(f, (0, float("nan"))) for f in d["fit"]]
            cols[col] = f"{s.strategy_id} block (n)"
            fmt[col] = lambda v: "n/a (0)" if not v[0] else f"{_pct(v[1])} ({v[0]})"
        a(_md(d, cols, fmt))
        a("")
        for s in rep.strategies:
            nm = s.gate_by_refit[(s.gate_by_refit["key"] == k)
                                 & (s.gate_by_refit["fit_time"] == NO_MODEL)]
            if len(nm):
                a(f"- {s.strategy_id}: {int(nm['cycles'].iloc[0])} in-window {k} cycles with "
                  "no model (all blocked: regime unavailable).")
        a("")

    a("## SPEC NOTEs (engineering readings, also in the module docstring)")
    a("")
    a("1. Test windows are UTC dates [first 00:00, last + 1 day 00:00); a candidate belongs to "
      "the fold of its signal time T. Every strategy window lies between 08:00 and 20:00 UTC, "
      "so a local-date reading assigns the same fold.")
    a(f"2. The {MIN_CANDIDATES}-candidate threshold is applied unscaled. COMMON 10.1 says "
      f"\"fewer than {MIN_CANDIDATES} candidates in total\"; amendment A1 pro-rates only the "
      "per-fold minimum of 10.4 by the data share, and the 10.4 total of 150 trades, which the "
      "200 protects, is not pro-rated either. Escalated to Quant: if Quant rules that the "
      "threshold is pro-rated, the information line under the coverage table gives the "
      "pro-rated value.")
    a("3. Data share of a fold: trading days with data (cash-open M15 bar present on at least "
      "one index) over calendar trading days (union over the ranked indices).")
    a("4. In-window cycles are member-cycles (T, index) with the index in G(T), as the "
      "strategy's own driver builds G(T); incomplete-group cycles are included, indices outside "
      "G(T) are not.")
    a("5. The 6.8 flag uses the literal gate (status OK and the V_T condition), so UNAVAILABLE "
      "counts as blocked; the share among status-OK cycles is shown and flagged next to it.")
    a("6. The per-cycle selection log of COMMON 4.2.8 is written by the CLI to the "
      "`--selections` file (gzip text, one line per cycle); see provenance.")
    a("")
    return "\n".join(L) + "\n"


def _labels(js: Any) -> str:
    """'{"ACTIVE_FLAT": 0.68, ...}' -> 'ACTIVE_FLAT 68%, ...'."""
    import json
    try:
        d = json.loads(js) if isinstance(js, str) else dict(js)
    except (TypeError, ValueError):
        return str(js)
    return ", ".join(f"{k} {100 * float(v):.0f}%" for k, v in
                     sorted(d.items(), key=lambda kv: -float(kv[1])))


# ---------------------------------------------------------------------------------------------
# CLI (real pre-holdout data)
# ---------------------------------------------------------------------------------------------

def _frames_sha256(frames: Iterable[tuple[str, pd.DataFrame]]) -> str:
    h = hashlib.sha256()
    for name, df in frames:
        h.update(name.encode())
        h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _git_sha(root: Path) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "no commit yet"
    except (OSError, subprocess.SubprocessError):
        return "git unavailable"


def main(argv: Sequence[str] | None = None) -> int:
    from core.config import ROOT, load_config
    from data.store import BarStore
    from features.service import FeatureService
    from regime.service import MODEL_DIR, REFIT_LOG, RegimeService

    ap = argparse.ArgumentParser(description="COMMON 10.1 feasibility gate (counts only)")
    ap.add_argument("--doc", default=str(ROOT / "docs" / "feasibility_gate.md"))
    ap.add_argument("--selections", default=None,
                    help="gzip file for the per-cycle selection log (COMMON 4.2.8)")
    ap.add_argument("--model-dir", default=str(MODEL_DIR),
                    help="regime model cache read (and written on a cache miss) by run_batch")
    ap.add_argument("--refit-log", default=str(REFIT_LOG))
    ap.add_argument("--stability-model-dir", default=None,
                    help="cache for the seeds 10..19 refits; default none (refit every run, "
                         "never written to the production cache)")
    ap.add_argument("--no-seed-stability", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    t_all = _time.time()
    cfg = load_config()
    keys = config_order(cfg)
    store = BarStore(cfg)
    entry, regime_tf = cfg["timeframes"]["entry"], cfg["timeframes"]["regime"]
    m15 = {k: store.load(k, entry) for k in keys}       # pre-holdout, excluded windows dropped
    h1 = {k: store.load(k, regime_tf) for k in keys}
    features = FeatureService(cfg)
    for k in keys:
        features.build(k, m15[k], h1[k])
    refit_log = Path(args.refit_log)
    n_log = _line_count(refit_log)
    rs = RegimeService(cfg, model_dir=Path(args.model_dir), refit_log=refit_log)
    tables = {k: rs.run_batch(k, h1[k], m15[k]) for k in keys}
    n_new = _line_count(refit_log) - n_log
    stab = RegimeService(cfg, model_dir=(Path(args.stability_model_dir)
                                         if args.stability_model_dir else None),
                         refit_log=None)
    cfg_path = ROOT / "config.yaml"
    prov = {
        "Run (UTC)": f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M}",
        "Data": "BarStore.load pre-holdout (bars before "
                f"{cfg['holdout']['start']} UTC; A1 excluded windows dropped); final_holdout "
                "never passed",
        "Data SHA-256 (M15 + H1, all ranked indices)":
            _frames_sha256([(f"{k}/{tf}", d) for k in keys
                            for tf, d in ((entry, m15[k]), (regime_tf, h1[k]))])[:16],
        "config.yaml SHA-256": hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:16],
        "Git": _git_sha(ROOT),
        "Regime model cache": f"{args.model_dir} (refit log {args.refit_log}: {n_new} rows "
                              "appended by this run; 0 = every model loaded from the cache)",
        "Seed-stability refits": "skipped" if args.no_seed_stability else
            f"seeds {list(cfg['regime']['stability_seeds'])}, cache "
            f"{args.stability_model_dir or 'none'} (never the production cache, never logged)",
    }
    rep = run_feasibility(cfg, features, tables, stability_service=stab,
                          stability_seeds=() if args.no_seed_stability else None,
                          provenance=prov)
    if args.selections:
        n = write_selection_log(rep, Path(args.selections))
        digest = hashlib.sha256(Path(args.selections).read_bytes()).hexdigest()[:16]
        rep.provenance["Per-cycle selection log (COMMON 4.2.8)"] = \
            f"`{args.selections}` ({n} lines, SHA-256 {digest})"
    rep.provenance["Runtime"] = f"{_time.time() - t_all:.0f} s"
    Path(args.doc).write_text(rep.to_markdown(), encoding="utf-8")
    for s in rep.strategies:
        print(f"{s.strategy_id}: {s.total} candidates -> {s.verdict}; gate {s.gate} block "
              f"share {_pct(s.gate_share)} ({s.gate_flag})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
