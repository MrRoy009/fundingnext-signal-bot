"""Trial log (COMMON 10.7, 8.10; ER-14): one row per configuration in docs/trial_log.csv, the
raw trial count N_raw and the global trial matrix of daily series (T_days x N_raw).

Why it exists: the deflated Sharpe ratio and PBO are only honest if every configuration ever run
on pre-holdout data is counted, failures included. This module is the only writer of the trial
log, so the count cannot be bypassed by a script that forgets to log.

What a configuration is (COMMON 10.7): (strategy_id, full parameter vector, shared-component
settings, entry protocol). Its identity is `config_key`: a hash of those four, where the
shared-component settings are the config blocks that define the shared components
(SHARED_BLOCKS: features, regime, ...). Scenarios (COMMON 10.3) of the same configuration go
into the same row's notes; they are never new rows.

Rows
    trial_id,date_utc,strategy_id,params,data_window,stage,result_summary,notes
    notes = JSON: config_key, config_hash (every frozen config block + the SymbolSpecs), shared
    hash, data hash, git SHA ('-dirty' when the work tree has uncommitted changes), entry
    protocol, stages, per-scenario metrics (trades, expectancy $ and R, daily Sharpe, total $,
    max DD, ...) and stage-specific extras (secondary sample, run-B selections, ...).

Logging rules (log_trial)
  - A configuration not yet logged: a new row (new trial_id).
  - Logged already with the same config hash and data hash: no new row. Scenarios or extras the
    row lacks are merged into that row's notes (the CSV is rewritten atomically; no row is ever
    removed). A scenario already present must reproduce its metrics (the backtest is
    deterministic, COMMON 8.10); a mismatch raises TrialConflict.
  - Logged with a different config hash or data hash, or a metric mismatch: TrialConflict,
    unless `rerun_note` is given (a rerun after a code bug fix, or after a data change). Then a
    NEW row is appended, its notes say "rerun after bug fix: <note>" and name the earlier
    trial_id; the earlier row is kept. The newest row of a config_key is its current result.
    Later logs of the same configuration with the same rerun note merge into that rerun row
    (so run_a and sensitivity rerunning the default make one rerun row, not two). A
    configuration logged for the FIRST time under a rerun note gets a normal new row (no
    'note', no 'rerun_of': nothing was rerun) whose notes carry 'rerun_session' = the rerun
    text, so later logs in the same session merge into it instead of appending a phantom
    rerun row.

Stages: run_a, run_b_train, sensitivity, shared (counted in N_raw); run_b, stress, benchmark,
portfolio (not counted).
How N_raw is counted (n_raw): the number of DISTINCT config_keys among rows whose stage is in
COUNTED_STAGES, all strategies. A run-B train-window evaluation of a grid point is the same
configuration as that grid point anywhere else (the window it is evaluated on is not part of a
configuration), so it counts once, whether it was first logged by run_b_train, sensitivity or
run_a; nine folds of train evaluations of one grid point are one trial. A rerun after a bug fix
is the same configuration and does not add to N_raw. Benchmarks and the combined book are
logged but never counted (COMMON 10.6, 8.8). MR-TWAP-01 is never run (A3), so it adds nothing.
SPEC NOTE (stage run_b): the walk-forward-selected run B result is a composite of configurations
that are already counted (each train evaluation is its grid point), so it is logged with stage
`run_b` and not counted. SPEC NOTE (stage stress): 10.7 puts scenarios in the configuration's
row; `stress` is accepted for a scenario-only run that has no configuration row of its own and
is never counted.
SPEC NOTE (rules block, awaiting a Quant ruling): `rules` is not a shared-component block, so
rules.restricted_windows_utc (read by SES-IBX-01.scan, O5) is NOT part of config_key; it is
in config_hash (FROZEN_BLOCKS). Filling or changing the restricted windows therefore changes a
logged configuration's result without changing its identity: log_trial raises TrialConflict
(config hash changed) and the configuration can only be logged again as a rerun row with
rerun_note (N_raw unchanged). The runner's scan cache is keyed on the full config, so it never
serves scans made under other restricted windows.

Per trial files (backtest/runs/<trial_id>/ as the runner writes them by default, or under a
strategy folder: backtest/runs/<strategy>/<trial_id>/ (RS-MOM-01) or
backtest/runs/<strategy>/trials/<trial_id>/ (SES-IBX-01)): daily.parquet (the canonical daily
series: date, equity, pnl_usd, ret), trades.parquet, selection.log.gz, and
scenarios/<SCENARIO>/{daily,trades,equity}.parquet for every scenario run. Readers
(trial_locations, find_trial_dir, load_daily, load_trades, trial_matrix) search all three
layouts in every strategy folder (never a 'staging' folder); a trial id with more than one
folder, or a folder under another strategy, is an error. The trial matrix (trial_matrix) is
built from each counted configuration's newest daily.parquet, all strategies by default.

Public API
    TRIAL_LOG, RUNS_DIR, COLUMNS, STAGES, COUNTED_STAGES, SHARED_BLOCKS, FROZEN_BLOCKS
    TrialConflict
    canonical_params(params) -> dict;  params_text(params) -> str
    shared_settings_hash(cfg) -> str;  config_hash(cfg, specs=None) -> str
    specs_hash(specs) -> str;  git_state(root=ROOT) -> str
    config_key(strategy_id, params, shared_hash, protocol, stage=None, variant="") -> str
    TrialRecord(...)                     what a caller logs
    LogResult(trial_id, created, merged)
    read_log(path=TRIAL_LOG) -> DataFrame (notes parsed into a 'notes_json' column)
    find_trial(key, path=TRIAL_LOG) -> pd.Series | None    (newest row of that config_key)
    log_trial(record, *, path=TRIAL_LOG, rerun_note=None, now=None, trial_id=None) -> LogResult
    will_append(key, path=TRIAL_LOG, rerun_note=None) -> bool
    rerun_text(note) -> str;  rerun_tag(notes) -> str | None
    reserve_trial_id(path) -> str
    counted_trials(path=TRIAL_LOG, strategy_id=None) -> DataFrame
    n_raw(path=TRIAL_LOG, strategy_id=None) -> int
    trial_dir(trial_id, runs_dir=RUNS_DIR) -> Path      (where a writer puts a trial: flat)
    trial_locations(runs_dir=RUNS_DIR) -> {trial_id: [(folder, owner strategy or None)]}
    find_trial_dir(trial_id, runs_dir=RUNS_DIR, strategy_id=None) -> Path | None
    save_run_files(directory, *, daily=None, trades=None, equity=None, selection=None) -> Path
    write_selection_log(path, strategy_id, lines) -> int
    parquet_safe(df) -> DataFrame
    load_daily(trial_id, runs_dir=RUNS_DIR, scenario=None, strategy_id=None) -> DataFrame
    load_trades(trial_id, runs_dir=RUNS_DIR, scenario=None, strategy_id=None) -> DataFrame
    trial_matrix(path=TRIAL_LOG, runs_dir=RUNS_DIR, column="pnl_usd", strategy_id=None)
        -> DataFrame (index: date, one column per counted trial_id)
    json_safe(obj) -> JSON-serialisable copy (NaN -> None, numpy -> python)
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from core.config import ROOT

TRIAL_LOG = ROOT / "docs" / "trial_log.csv"
RUNS_DIR = ROOT / "backtest" / "runs"
COLUMNS = ["trial_id", "date_utc", "strategy_id", "params", "data_window", "stage",
           "result_summary", "notes"]

COUNTED_STAGES = ("run_a", "run_b_train", "sensitivity", "shared")
UNCOUNTED_STAGES = ("run_b", "stress", "benchmark", "portfolio")
STAGES = COUNTED_STAGES + UNCOUNTED_STAGES

# Config blocks that define the shared components of a configuration (COMMON 10.7: rank lookback,
# HMM window, Kmax, refit cadence, ... live in features / regime), plus everything else that
# changes which candidates exist or how the protocol reads them.
SHARED_BLOCKS = ("instruments", "exchanges", "time_exits", "timeframes", "server_timezone",
                 "holdout", "data_quality", "features", "regime", "protocols", "execution")
# Every frozen block that can change a backtest result (ER-11: "all hashed into each trial-log
# row"). Owner flags (owner_confirmed, gates), paths and download settings are left out.
FROZEN_BLOCKS = SHARED_BLOCKS + ("costs", "risk", "rules", "sessions_utc")

SUMMARY_SCENARIOS = ("STRESS-d2", "STRESS-d3", "MISSED-EXIT")
STRATEGY_TRIALS_DIR = "trials"   # runs/<strategy>/trials/<trial_id> (SES-IBX-01 layout)
STAGING_DIR = "staging"          # staged copies under staged ids: never read as trials
_TOL = 1e-9


class TrialConflict(RuntimeError):
    """A logged configuration would change without an explicit rerun note."""


# ---------------------------------------------------------------------------------------------
# Hashes and identity
# ---------------------------------------------------------------------------------------------

def json_safe(obj: Any) -> Any:
    """A JSON-serialisable copy: numpy scalars -> python, NaN/inf -> None, timestamps -> ISO,
    tuples -> lists, mapping keys -> str."""
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        seq = sorted(obj, key=str) if isinstance(obj, set) else obj
        return [json_safe(v) for v in seq]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        t = pd.Timestamp(obj)
        return None if pd.isna(t) else t.isoformat()
    if obj is None or isinstance(obj, str):
        return obj
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    return str(obj)


def _dumps(obj: Any) -> str:
    return json.dumps(json_safe(obj), sort_keys=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Parameter values as floats where numeric (1 == 1.0), keys in their given order."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            v = float(v)
        out[str(k)] = v
    return out


def params_text(params: Mapping[str, Any]) -> str:
    """The trial log's params column, e.g. 'theta=1.0,k=1.0,R=1.5' (ParamGrid.label form)."""
    return ",".join(f"{k}={v}" for k, v in params.items())


def shared_settings_hash(cfg: dict) -> str:
    return _sha(_dumps({b: cfg.get(b) for b in SHARED_BLOCKS}))[:16]


def specs_hash(specs: Mapping[str, Any] | None) -> str:
    """Hash of the read-only SymbolSpecs (lot step, swap, stops level change results)."""
    if not specs:
        return "none"
    rows = {}
    for k, s in sorted(specs.items()):
        rows[k] = {f: getattr(s, f) for f in getattr(s, "__dataclass_fields__", {})} \
            if hasattr(s, "__dataclass_fields__") else s
    return _sha(_dumps(rows))[:16]


def config_hash(cfg: dict, specs: Mapping[str, Any] | None = None) -> str:
    """Hash of every frozen config block (FROZEN_BLOCKS) and the SymbolSpecs."""
    return _sha(_dumps({"cfg": {b: cfg.get(b) for b in FROZEN_BLOCKS},
                        "specs": specs_hash(specs)}))[:16]


def git_state(root: Path = ROOT) -> str:
    """`git rev-parse HEAD`, with '-dirty' if `git status --porcelain` lists any change.
    'no commit yet' / 'git unavailable' when it cannot be read."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                           text=True, timeout=20)
        sha = r.stdout.strip()
        if r.returncode != 0 or not sha:
            return "no commit yet"
        s = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                           text=True, timeout=60)
        if s.returncode != 0:
            return f"{sha}-unknown-tree"
        return f"{sha}-dirty" if s.stdout.strip() else sha
    except (OSError, subprocess.SubprocessError):
        return "git unavailable"


def config_key(strategy_id: str, params: Mapping[str, Any], shared_hash: str, protocol: str,
               stage: str | None = None, variant: str = "") -> str:
    """Identity of a configuration (COMMON 10.7). For counted stages it is (strategy_id,
    params, shared settings, entry protocol) only, so the same grid point logged by run_a,
    sensitivity and run_b_train is one trial. Uncounted stages (benchmarks, combined book,
    run B composite, stress) add the stage and a variant name, so they never collide with a
    configuration or with each other."""
    ident: dict[str, Any] = {"strategy_id": strategy_id, "params": canonical_params(params),
                             "shared": shared_hash, "protocol": protocol}
    if stage is not None and stage not in COUNTED_STAGES:
        ident.update(stage=stage, variant=variant)
    return _sha(_dumps(ident))[:20]


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TrialRecord:
    """One configuration run, as the runner (or a benchmark / report tool) logs it.

    scenarios: {scenario name (e.g. 'STRESS-d2', or 'BASE@competition'): metrics dict}.
    extra: stage-specific JSON (secondary sample, run-B selection tables, benchmark settings).
    canonical: the scenario whose daily series is the trial's daily.parquet (the series used in
    the trial matrix), normally 'STRESS-d2' over the concatenated test windows.
    """
    strategy_id: str
    params: Mapping[str, Any]
    stage: str
    protocol: str
    data_window: str
    shared_hash: str
    config_hash: str
    data_hash: str
    scenarios: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)
    variant: str = ""
    canonical: str | None = "STRESS-d2"
    git: str | None = None

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, not {self.stage!r}")
        if self.stage in COUNTED_STAGES and self.variant:
            raise ValueError("a counted configuration has no variant name: its identity is "
                             "(strategy_id, params, shared settings, protocol)")

    @property
    def key(self) -> str:
        return config_key(self.strategy_id, self.params, self.shared_hash, self.protocol,
                          self.stage, self.variant)


@dataclass(frozen=True)
class LogResult:
    trial_id: str
    created: bool                 # a new row was appended
    merged: tuple[str, ...] = ()  # scenario / extra names added to an existing row


def summary_text(scenarios: Mapping[str, Mapping[str, Any]]) -> str:
    """result_summary: the pass scenarios first (STRESS-d2, STRESS-d3, MISSED-EXIT), else
    every scenario, one clause each."""
    names = [s for s in SUMMARY_SCENARIOS if s in scenarios] or sorted(scenarios)
    parts = []
    for s in names:
        m = scenarios[s]

        def f(key: str, fmt: str) -> str:
            v = m.get(key)
            return "nan" if v is None else format(v, fmt)
        parts.append(f"{s}: n={m.get('n_trades', 0)} E$={f('expectancy_usd', '.2f')} "
                     f"ER={f('expectancy_r', '.3f')} SR={f('sharpe_daily', '.4f')} "
                     f"tot$={f('total_usd', '.0f')} DD$={f('max_dd_usd', '.0f')}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------------------------
# Log file
# ---------------------------------------------------------------------------------------------

def _read_rows(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{path} has columns {reader.fieldnames}, expected {COLUMNS}")
        return [dict(r) for r in reader]


def _write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    """Atomic rewrite (temp file + replace), so a crash never leaves half a log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".trial_log.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in COLUMNS})
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _append_row(path: Path, row: Mapping[str, str]) -> None:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        _write_rows(path, [row])
        return
    _read_rows(path)                    # header check
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n").writerow(
            {c: row.get(c, "") for c in COLUMNS})


def _notes(row: Mapping[str, str]) -> dict[str, Any]:
    try:
        out = json.loads(row.get("notes") or "{}")
    except ValueError:
        return {}
    return out if isinstance(out, dict) else {}


def read_log(path: Path = TRIAL_LOG) -> pd.DataFrame:
    """The trial log as a DataFrame (all columns as text) plus 'notes_json' (parsed) and
    'config_key'."""
    rows = _read_rows(path)
    df = pd.DataFrame(rows, columns=COLUMNS)
    notes = [_notes(r) for r in rows]
    df["notes_json"] = pd.Series(notes, index=df.index, dtype=object)
    df["config_key"] = [n.get("config_key", "") for n in notes]
    return df


def _newest_by_key(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, r in enumerate(rows):
        k = _notes(r).get("config_key")
        if k:
            out[k] = i                  # file order = time order; the last row wins
    return out


def find_trial(key: str, path: Path = TRIAL_LOG) -> pd.Series | None:
    """The newest row of a config_key, or None."""
    df = read_log(path)
    hit = df[df["config_key"] == key]
    return None if hit.empty else hit.iloc[-1]


def _next_id(rows: Sequence[Mapping[str, str]]) -> str:
    n = 0
    for r in rows:
        tid = r.get("trial_id", "")
        if tid.startswith("T") and tid[1:].isdigit():
            n = max(n, int(tid[1:]))
    return f"T{n + 1:04d}"


def reserve_trial_id(path: Path = TRIAL_LOG) -> str:
    """The trial_id the next appended row will get (for writing run files first)."""
    return _next_id(_read_rows(path))


def _same(a: Any, b: Any) -> bool:
    """Equal JSON values, floats within a relative 1e-9 (deterministic reruns)."""
    a, b = json_safe(a), json_safe(b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=_TOL, abs_tol=_TOL)
    return a == b


def rerun_text(note: str) -> str:
    return f"rerun after bug fix: {note}"


def rerun_tag(notes: Mapping[str, Any]) -> str | None:
    """The rerun session a row was written in: its 'note' (a rerun row) or its
    'rerun_session' (a configuration first logged during a rerun session), else None. Later
    logs of the configuration under the same rerun note merge into that row."""
    return notes.get("note") or notes.get("rerun_session")


def will_append(key: str, path: Path = TRIAL_LOG, rerun_note: str | None = None) -> bool:
    """True if log_trial would append a new row for this config_key (not logged yet, or a
    rerun note whose rerun row does not exist yet); False if it would merge."""
    rows = _read_rows(path)
    idx = _newest_by_key(rows).get(key)
    if idx is None:
        return True
    return rerun_note is not None and rerun_tag(_notes(rows[idx])) != rerun_text(rerun_note)


def _new_notes(rec: TrialRecord, git: str, rerun_of: str | None, rerun_note: str | None
               ) -> dict[str, Any]:
    n: dict[str, Any] = {
        "config_key": rec.key, "config_hash": rec.config_hash, "shared_hash": rec.shared_hash,
        "data_hash": rec.data_hash, "git": git, "protocol": rec.protocol,
        "params": canonical_params(rec.params), "stages": [rec.stage],
        "canonical_scenario": rec.canonical, "scenarios": json_safe(dict(rec.scenarios)),
        "extra": json_safe(dict(rec.extra)),
    }
    if rec.variant:
        n["variant"] = rec.variant
    if rerun_of is not None:
        n["rerun_of"] = rerun_of
        n["note"] = rerun_text(rerun_note)
    elif rerun_note is not None:
        # First log of this configuration, made during a rerun session: there is no earlier
        # row, so it is not a rerun ('note' stays unset), but later logs under the same note
        # merge into it.
        n["rerun_session"] = rerun_text(rerun_note)
    return n


def log_trial(rec: TrialRecord, *, path: Path = TRIAL_LOG, rerun_note: str | None = None,
              now: pd.Timestamp | None = None, trial_id: str | None = None) -> LogResult:
    """Append or merge one configuration row (see the module docstring for the rules).

    trial_id: the id reserved with reserve_trial_id() for a NEW row (so run files can be
    written before the row); it must equal the next free id. Ignored when the configuration is
    already logged and is merged instead."""
    path = Path(path)
    rows = _read_rows(path)
    newest = _newest_by_key(rows)
    git = rec.git or git_state()
    stamp = (now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    idx = newest.get(rec.key)
    if idx is not None and rerun_note is not None and \
            rerun_tag(_notes(rows[idx])) == rerun_text(rerun_note):
        rerun_note = None               # this rerun session's row exists: merge into it

    if idx is not None and rerun_note is None:
        row = rows[idx]
        notes = _notes(row)
        why = []
        if notes.get("config_hash") != rec.config_hash:
            why.append(f"config hash {notes.get('config_hash')} -> {rec.config_hash}")
        if notes.get("data_hash") != rec.data_hash:
            why.append(f"data hash {notes.get('data_hash')} -> {rec.data_hash}")
        scen = notes.setdefault("scenarios", {})
        extra = notes.setdefault("extra", {})
        merged = []
        for name, m in rec.scenarios.items():
            if name in scen:
                if not _same(scen[name], m):
                    why.append(f"scenario {name} metrics differ from {row['trial_id']}")
            else:
                scen[name] = json_safe(dict(m))
                merged.append(name)
        for name, v in rec.extra.items():
            if name in extra:
                if not _same(extra[name], v):
                    why.append(f"extra {name!r} differs from {row['trial_id']}")
            else:
                extra[name] = json_safe(v)
                merged.append(name)
        if why:
            raise TrialConflict(f"{rec.strategy_id} {params_text(rec.params)} is already logged "
                                f"as {row['trial_id']}: " + "; ".join(why) +
                                ". A changed result needs rerun_note='<bug fix or data change>'"
                                " (a new row; the earlier row is kept)")
        stages = notes.setdefault("stages", [])
        if rec.stage not in stages:
            stages.append(rec.stage)
            merged.append(f"stage:{rec.stage}")
        if merged:
            notes.setdefault("merged", []).append({"date_utc": stamp, "git": git,
                                                   "added": merged})
            row["notes"] = _dumps(notes)
            row["result_summary"] = summary_text(scen)
            _write_rows(path, rows)
        return LogResult(row["trial_id"], False, tuple(merged))

    tid = _next_id(rows)
    if trial_id is not None and trial_id != tid:
        raise ValueError(f"reserved trial_id {trial_id} is no longer the next id ({tid})")
    rerun_of = rows[idx]["trial_id"] if idx is not None else None
    notes = _new_notes(rec, git, rerun_of, rerun_note)
    _append_row(path, {
        "trial_id": tid, "date_utc": stamp, "strategy_id": rec.strategy_id,
        "params": params_text(rec.params), "data_window": rec.data_window, "stage": rec.stage,
        "result_summary": summary_text(rec.scenarios), "notes": _dumps(notes)})
    return LogResult(tid, True, tuple(rec.scenarios))


# ---------------------------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------------------------

def counted_trials(path: Path = TRIAL_LOG, strategy_id: str | None = None) -> pd.DataFrame:
    """The newest row of every counted configuration (see n_raw), in first-logged order.
    A configuration counts if ANY of its rows has a counted stage (its notes' stages list
    included, so a configuration first logged by run_a and merged by sensitivity counts once)."""
    df = read_log(path)
    if df.empty:
        return df
    stages = [set([s]) | set(n.get("stages", [])) for s, n in zip(df["stage"], df["notes_json"])]
    counted = pd.Series([bool(st & set(COUNTED_STAGES)) for st in stages], index=df.index)
    df = df[counted & (df["config_key"] != "")]
    if strategy_id is not None:
        df = df[df["strategy_id"] == strategy_id]
    first = df.drop_duplicates("config_key", keep="first")["config_key"].tolist()
    newest = df.drop_duplicates("config_key", keep="last").set_index("config_key")
    return newest.loc[first].reset_index()


def n_raw(path: Path = TRIAL_LOG, strategy_id: str | None = None) -> int:
    """N_raw (COMMON 10.7): distinct counted configurations, all strategies by default."""
    return int(len(counted_trials(path, strategy_id)))


# ---------------------------------------------------------------------------------------------
# Run files
# ---------------------------------------------------------------------------------------------

def trial_dir(trial_id: str, runs_dir: Path = RUNS_DIR) -> Path:
    return Path(runs_dir) / trial_id


def parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """A copy pyarrow can write: object columns of timestamps -> UTC datetime, of bools ->
    nullable boolean, of numbers -> float, anything else -> nullable string."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if s.dtype != object:
            continue
        vals = s.dropna()
        if vals.empty:
            out[c] = s.astype("string")
        elif all(isinstance(v, (pd.Timestamp, np.datetime64)) for v in vals):
            out[c] = pd.to_datetime(s, utc=True)
        elif all(isinstance(v, (bool, np.bool_)) for v in vals):
            out[c] = s.astype("boolean")
        elif all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                 for v in vals):
            out[c] = pd.to_numeric(s, errors="coerce").astype("float64")
        else:
            out[c] = s.map(lambda v: None if v is None or (isinstance(v, float) and math.isnan(v))
                           else str(v)).astype("string")
    return out


def write_selection_log(path: Path, strategy_id: str,
                        lines: Iterable[tuple[int, str]]) -> int:
    """COMMON 4.2.8 / A3.5 per-cycle selection log, gzip text '<strategy>\\t<fold>\\t<line>'.
    mtime 0 and no stored file name keep the file byte-identical across reruns and paths.
    Returns the line count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb",
                                                mtime=0) as gz, \
            io.TextIOWrapper(gz, encoding="utf-8", newline="\n") as f:
        for fold, line in lines:
            f.write(f"{strategy_id}\t{fold}\t{line}\n")
            n += 1
    return n


def save_run_files(directory: Path, *, daily: pd.DataFrame | None = None,
                   trades: pd.DataFrame | None = None, equity: pd.DataFrame | None = None,
                   selection: tuple[str, Iterable[tuple[int, str]]] | None = None) -> Path:
    """Write daily.parquet / trades.parquet / equity.parquet / selection.log.gz into
    `directory` (created). selection = (strategy_id, [(fold, line), ...])."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    if daily is not None:
        parquet_safe(daily).to_parquet(d / "daily.parquet", index=False)
    if trades is not None:
        parquet_safe(trades).to_parquet(d / "trades.parquet", index=False)
    if equity is not None:
        parquet_safe(equity).to_parquet(d / "equity.parquet", index=False)
    if selection is not None:
        write_selection_log(d / "selection.log.gz", selection[0], selection[1])
    return d


def _is_trial_id(name: str) -> bool:
    return len(name) >= 5 and name[0] == "T" and name[1:].isdigit()


def trial_locations(runs_dir: Path = RUNS_DIR) -> dict[str, list[tuple[Path, str | None]]]:
    """Every trial folder under runs_dir: {trial_id: [(folder, owner strategy or None)]}.

    Layouts found (a strategy folder is any sub-folder of runs_dir that is not a trial id):
      runs_dir/<trial_id>/                         flat (runner default, owner None)
      runs_dir/<strategy>/<trial_id>/              e.g. backtest/runs/RS-MOM-01/T0001
      runs_dir/<strategy>/trials/<trial_id>/       e.g. backtest/runs/SES-IBX-01/trials/T0034
    Folders named STAGING_DIR (staged copies under their staged ids) are never searched, and
    only names T<digits> are trial folders (a half-renamed '_tmp_T0034' is not)."""
    runs_dir = Path(runs_dir)
    out: dict[str, list[tuple[Path, str | None]]] = {}
    if not runs_dir.is_dir():
        return out

    def scan(folder: Path, owner: str | None) -> None:
        for d in sorted(folder.iterdir()):
            if d.is_dir() and _is_trial_id(d.name):
                out.setdefault(d.name, []).append((d, owner))

    scan(runs_dir, None)
    for s in sorted(runs_dir.iterdir()):
        if not s.is_dir() or _is_trial_id(s.name) or s.name == STAGING_DIR:
            continue
        scan(s, s.name)
        sub = s / STRATEGY_TRIALS_DIR
        if sub.is_dir():
            scan(sub, s.name)
    return out


def _resolve(trial_id: str, found: list[tuple[Path, str | None]], runs_dir: Path,
             strategy_id: str | None) -> Path | None:
    if len(found) > 1:
        raise ValueError(f"trial {trial_id} has {len(found)} folders under {runs_dir}: "
                         f"{[str(p) for p, _ in found]}; a trial id must be unique")
    if not found:
        return None
    folder, owner = found[0]
    if strategy_id is not None and owner is not None and owner != strategy_id:
        raise ValueError(f"trial {trial_id} ({strategy_id}) was found in the folder of "
                         f"{owner}: {folder}")
    return folder


def find_trial_dir(trial_id: str, runs_dir: Path = RUNS_DIR, strategy_id: str | None = None
                   ) -> Path | None:
    """The run folder of a trial in any layout of trial_locations(), or None. ValueError if
    the id has several folders, or a folder under another strategy than `strategy_id`."""
    return _resolve(trial_id, trial_locations(runs_dir).get(trial_id, []), Path(runs_dir),
                    strategy_id)


def _declared_run_files(notes: Mapping[str, Any]) -> str | None:
    """The run folder a batch-appended row names in its notes ('run_files', or
    'staged_as.run_files'), as a path string, or None."""
    staged = notes.get("staged_as")
    v = notes.get("run_files") or (staged.get("run_files") if isinstance(staged, Mapping)
                                   else None)
    return v if isinstance(v, str) and v else None


def _check_declared(trial_id: str, folder: Path, runs_dir: Path, declared: str | None) -> None:
    """A row that names its run folder ('backtest/runs/<...>') must be found exactly there
    (relative to runs_dir); a folder found elsewhere is a stale copy, not this trial."""
    if declared is None:
        return
    parts = Path(declared.replace("\\", "/")).parts
    if "runs" not in parts:
        return
    want = parts[len(parts) - list(reversed(parts)).index("runs"):]
    got = folder.relative_to(runs_dir).parts
    if tuple(want) != tuple(got):
        raise ValueError(f"trial {trial_id}: notes name run files {declared!r} but the folder "
                         f"found is {folder}")


def _run_path(trial_id: str, runs_dir: Path, scenario: str | None, name: str,
              strategy_id: str | None = None) -> Path:
    base = find_trial_dir(trial_id, runs_dir, strategy_id) or trial_dir(trial_id, runs_dir)
    return (base / name) if scenario is None else (base / "scenarios" / scenario / name)


def load_daily(trial_id: str, runs_dir: Path = RUNS_DIR, scenario: str | None = None,
               strategy_id: str | None = None) -> pd.DataFrame:
    """A trial's daily series (canonical, or one scenario's), from its folder in any layout of
    trial_locations()."""
    return pd.read_parquet(_run_path(trial_id, runs_dir, scenario, "daily.parquet",
                                     strategy_id))


def load_trades(trial_id: str, runs_dir: Path = RUNS_DIR, scenario: str | None = None,
                strategy_id: str | None = None) -> pd.DataFrame:
    return pd.read_parquet(_run_path(trial_id, runs_dir, scenario, "trades.parquet",
                                     strategy_id))


def trial_matrix(path: Path = TRIAL_LOG, runs_dir: Path = RUNS_DIR, column: str = "pnl_usd",
                 strategy_id: str | None = None) -> pd.DataFrame:
    """The global trial matrix (COMMON 10.7): index = date (the shared daily calendar), one
    column per counted configuration (its newest trial_id), values = `column` of its canonical
    daily.parquet. All strategies by default; run folders are found in every strategy
    sub-folder of runs_dir (trial_locations). Every counted trial must have exactly one folder
    (in its own strategy's sub-folder, or flat; where its notes name one, exactly there), its
    daily.parquet and the same calendar (the canonical series of every configuration covers the
    concatenated test windows), otherwise ValueError: a matrix with a silently missing trial
    would understate N. The column count equals n_raw(path, strategy_id)."""
    trials = counted_trials(path, strategy_id)
    runs_dir = Path(runs_dir)
    where = trial_locations(runs_dir)
    cols: dict[str, pd.Series] = {}
    calendar: pd.DatetimeIndex | None = None
    missing = []
    for _, row in trials.iterrows():
        tid = row["trial_id"]
        folder = _resolve(tid, where.get(tid, []), runs_dir, row["strategy_id"])
        if folder is None or not (folder / "daily.parquet").exists():
            missing.append(f"{tid} ({row['strategy_id']})")
            continue
        _check_declared(tid, folder, runs_dir, _declared_run_files(row["notes_json"]))
        p = folder / "daily.parquet"
        d = pd.read_parquet(p)
        idx = pd.DatetimeIndex(pd.to_datetime(d["date"])).as_unit("ns")
        if calendar is None:
            calendar = idx
        elif not idx.equals(calendar):
            raise ValueError(f"trial {tid}: daily calendar differs from the first trial's "
                             f"({len(idx)} vs {len(calendar)} days); the trial matrix needs "
                             "one calendar")
        cols[tid] = pd.Series(d[column].to_numpy(dtype="float64"), index=idx)
    if missing:
        raise ValueError(f"counted trials without daily.parquet in {runs_dir}: {missing}")
    if len(cols) != len(trials) or len(cols) != trials["config_key"].nunique():
        raise ValueError(f"trial matrix has {len(cols)} columns for {len(trials)} counted "
                         f"configurations")
    if calendar is None:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    out = pd.DataFrame(cols, index=calendar)
    out.index.name = "date"
    return out
