"""backtest.trials (COMMON 10.7, ER-14) on synthetic records only: row format, one row per
configuration with scenarios merged, conflicts and reruns after a bug fix, N_raw counting,
hashes, git state and the trial matrix. Every test writes to tmp_path; the real
docs/trial_log.csv is only read (its header)."""
from __future__ import annotations

import csv
import json
import math
import subprocess
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from backtest import trials as tl
from data.mt5_adapter import SymbolSpec

DEFAULT = {"theta": 1.0, "k": 1.0, "R": 1.5}


def metrics(n=120, exp=10.0, sr=0.05):
    return {"n_trades": n, "expectancy_usd": exp, "expectancy_r": 0.1, "sharpe_daily": sr,
            "total_usd": n * exp, "max_dd_usd": -500.0, "max_dd_pct": -0.005}


def rec(params=None, stage="sensitivity", scenarios=None, **over):
    base = dict(strategy_id="RS-MOM-01", params=dict(params or DEFAULT), stage=stage,
                protocol="MARKET_BAND", data_window="2024-02-01..2026-04-30 (folds 1-9)",
                shared_hash="sh1", config_hash="cf1", data_hash="dh1",
                scenarios={"STRESS-d2": metrics()} if scenarios is None else scenarios,
                git="abc123")
    base.update(over)
    return tl.TrialRecord(**base)


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


NOW = pd.Timestamp("2026-11-05 12:00", tz="UTC")


def test_columns_match_the_committed_trial_log_header():
    with open(tl.TRIAL_LOG, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == tl.COLUMNS == ["trial_id", "date_utc", "strategy_id", "params",
                                    "data_window", "stage", "result_summary", "notes"]


def test_new_row_format(tmp_path):
    p = tmp_path / "log.csv"
    res = tl.log_trial(rec(), path=p, now=NOW)
    assert res == tl.LogResult("T0001", True, ("STRESS-d2",))
    (r,) = rows(p)
    assert r["trial_id"] == "T0001" and r["date_utc"] == "2026-11-05T12:00:00Z"
    assert r["strategy_id"] == "RS-MOM-01" and r["params"] == "theta=1.0,k=1.0,R=1.5"
    assert r["stage"] == "sensitivity" and r["data_window"].startswith("2024-02-01")
    assert r["result_summary"].startswith("STRESS-d2: n=120 E$=10.00")
    n = json.loads(r["notes"])
    for k in ("config_key", "config_hash", "shared_hash", "data_hash", "git", "protocol",
              "params", "stages", "scenarios", "extra", "canonical_scenario"):
        assert k in n
    assert n["git"] == "abc123" and n["protocol"] == "MARKET_BAND"
    assert n["scenarios"]["STRESS-d2"]["n_trades"] == 120
    assert n["stages"] == ["sensitivity"]


def test_same_configuration_is_one_row_and_scenarios_merge(tmp_path):
    p = tmp_path / "log.csv"
    tl.log_trial(rec(), path=p)
    again = tl.log_trial(rec(), path=p)
    assert again == tl.LogResult("T0001", False, ())
    assert len(rows(p)) == 1
    # run_a adds the other scenarios of the same configuration to the same row
    more = rec(stage="run_a", scenarios={"STRESS-d2": metrics(), "BASE": metrics(exp=12.0),
                                         "MISSED-EXIT": metrics(exp=3.0)},
               extra={"secondary": {"STRESS-d2": metrics(n=150)}})
    res = tl.log_trial(more, path=p)
    assert res.trial_id == "T0001" and not res.created
    assert set(res.merged) == {"BASE", "MISSED-EXIT", "secondary", "stage:run_a"}
    (r,) = rows(p)
    n = json.loads(r["notes"])
    assert set(n["scenarios"]) == {"STRESS-d2", "BASE", "MISSED-EXIT"}
    assert n["stages"] == ["sensitivity", "run_a"]
    assert n["extra"]["secondary"]["STRESS-d2"]["n_trades"] == 150
    assert "MISSED-EXIT" in r["result_summary"]
    # float noise is not a conflict (deterministic reruns)
    noisy = metrics()
    noisy["expectancy_usd"] *= 1 + 1e-12
    assert not tl.log_trial(rec(scenarios={"STRESS-d2": noisy}), path=p).created


@pytest.mark.parametrize("change", [
    {"scenarios": {"STRESS-d2": metrics(exp=11.0)}},
    {"data_hash": "dh2"},
    {"config_hash": "cf2"},
])
def test_changed_result_needs_a_rerun_note(tmp_path, change):
    p = tmp_path / "log.csv"
    tl.log_trial(rec(), path=p)
    before = p.read_bytes()
    with pytest.raises(tl.TrialConflict, match="rerun_note"):
        tl.log_trial(rec(**change), path=p)
    assert p.read_bytes() == before                      # nothing written


def test_rerun_after_bug_fix_keeps_earlier_row(tmp_path):
    p = tmp_path / "log.csv"
    tl.log_trial(rec(), path=p)
    fixed = rec(scenarios={"STRESS-d2": metrics(exp=8.0)})
    key = fixed.key
    assert tl.will_append(key, p, "swap sign") and not tl.will_append(key, p)
    res = tl.log_trial(fixed, path=p, rerun_note="swap sign")
    assert res == tl.LogResult("T0002", True, ("STRESS-d2",))
    r1, r2 = rows(p)
    assert json.loads(r1["notes"])["scenarios"]["STRESS-d2"]["expectancy_usd"] == 10.0
    n2 = json.loads(r2["notes"])
    assert n2["rerun_of"] == "T0001" and n2["note"] == "rerun after bug fix: swap sign"
    assert tl.n_raw(p) == 1                              # same configuration
    assert tl.find_trial(key, p)["trial_id"] == "T0002"  # the newest row is current
    # A second stage rerunning the same configuration with the same note merges into T0002.
    assert not tl.will_append(key, p, "swap sign")
    res = tl.log_trial(rec(stage="run_a", scenarios={"STRESS-d2": metrics(exp=8.0),
                                                     "BASE": metrics()}),
                       path=p, rerun_note="swap sign")
    assert res.trial_id == "T0002" and not res.created and "BASE" in res.merged
    assert len(rows(p)) == 2


def test_first_log_in_a_rerun_session_merges_later_logs(tmp_path):
    """A configuration first logged under a rerun note is a plain row (nothing was rerun);
    the next log of it under the same note merges instead of adding a phantom rerun row."""
    p = tmp_path / "log.csv"
    first = rec(stage="run_a")
    res = tl.log_trial(first, path=p, rerun_note="fix Y")
    assert res == tl.LogResult("T0001", True, ("STRESS-d2",))
    n = json.loads(rows(p)[0]["notes"])
    assert "note" not in n and "rerun_of" not in n
    assert n["rerun_session"] == "rerun after bug fix: fix Y"
    assert tl.rerun_tag(n) == tl.rerun_text("fix Y")
    assert not tl.will_append(first.key, p, "fix Y") and tl.will_append(first.key, p, "fix Z")
    res = tl.log_trial(rec(stage="sensitivity"), path=p, rerun_note="fix Y")
    assert res == tl.LogResult("T0001", False, ("stage:sensitivity",))
    assert len(rows(p)) == 1 and tl.n_raw(p) == 1
    # a different session is a real rerun of T0001
    res = tl.log_trial(rec(stage="sensitivity"), path=p, rerun_note="fix Z")
    assert res.created and json.loads(rows(p)[1]["notes"])["rerun_of"] == "T0001"


def test_n_raw_counts_distinct_configurations(tmp_path):
    p = tmp_path / "log.csv"
    assert tl.n_raw(p) == 0
    tl.log_trial(rec(stage="run_b_train"), path=p)        # train evaluation of the default
    tl.log_trial(rec(stage="sensitivity"), path=p)        # same configuration: merged
    tl.log_trial(rec(stage="run_a"), path=p)
    assert tl.n_raw(p) == 1 and len(rows(p)) == 1
    tl.log_trial(rec({"theta": 0.9, "k": 1.0, "R": 1.5}, stage="run_b_train"), path=p)
    tl.log_trial(rec({"theta": 1.0, "k": 0.75, "R": 1.5}), path=p)
    assert tl.n_raw(p) == 3
    # a shared-component variant is a new configuration (COMMON 10.7)
    tl.log_trial(rec(stage="shared", shared_hash="sh-L12"), path=p)
    assert tl.n_raw(p) == 4
    # not counted: benchmarks, combined book, run B composite, scenario-only stress rows
    tl.log_trial(rec(stage="benchmark", variant="B1 placebo seeds 1..200"), path=p)
    tl.log_trial(rec(stage="benchmark", variant="B3 gate off"), path=p)
    tl.log_trial(rec(stage="portfolio", strategy_id="SES-IBX-01+RS-MOM-01",
                     variant="combined book"), path=p)
    tl.log_trial(rec({"fold1": "[theta=0.9 k=1.0 R=1.5]"}, stage="run_b",
                     variant="walk-forward"), path=p)
    tl.log_trial(rec(stage="stress", variant="d3 only"), path=p)
    assert tl.n_raw(p) == 4 and len(rows(p)) == 9
    tl.log_trial(rec(strategy_id="SES-IBX-01", protocol="STOP_ENTRY",
                     params={"ib_minutes": 60, "tp_ib_mult": 1.0, "max_ib_datr": 0.6}), path=p)
    assert tl.n_raw(p) == 5
    assert tl.n_raw(p, "RS-MOM-01") == 4 and tl.n_raw(p, "SES-IBX-01") == 1
    assert list(tl.counted_trials(p)["trial_id"]) == ["T0001", "T0002", "T0003", "T0004",
                                                      "T0010"]


def test_config_key_identity():
    k = tl.config_key("RS-MOM-01", {"theta": 1, "k": 1.0, "R": 1.5}, "sh", "MARKET_BAND")
    assert k == tl.config_key("RS-MOM-01", {"theta": 1.0, "k": 1, "R": 1.5}, "sh",
                              "MARKET_BAND", stage="run_a")
    assert k == tl.config_key("RS-MOM-01", {"theta": 1.0, "k": 1.0, "R": 1.5}, "sh",
                              "MARKET_BAND", stage="sensitivity")
    assert k != tl.config_key("RS-MOM-01", {"theta": 1.1, "k": 1.0, "R": 1.5}, "sh",
                              "MARKET_BAND")
    assert k != tl.config_key("RS-MOM-01", {"theta": 1.0, "k": 1.0, "R": 1.5}, "sh2",
                              "MARKET_BAND")
    assert k != tl.config_key("RS-MOM-01", {"theta": 1.0, "k": 1.0, "R": 1.5}, "sh",
                              "MARKET_OR_LIMIT")
    b1 = tl.config_key("RS-MOM-01", DEFAULT, "sh", "MARKET_BAND", "benchmark", "B1")
    assert b1 != k and b1 != tl.config_key("RS-MOM-01", DEFAULT, "sh", "MARKET_BAND",
                                           "benchmark", "B2")


def test_record_validation():
    with pytest.raises(ValueError, match="stage"):
        rec(stage="final")
    with pytest.raises(ValueError, match="variant"):
        rec(stage="sensitivity", variant="x")


def test_hashes(cfg):
    sh, ch = tl.shared_settings_hash(cfg), tl.config_hash(cfg)
    c2 = json.loads(json.dumps(cfg))
    c2["costs"]["commission_stress_per_lot_side"]["US30"] = 2.0
    assert tl.shared_settings_hash(c2) == sh and tl.config_hash(c2) != ch
    c3 = json.loads(json.dumps(cfg))
    c3["features"]["rs4h_lookback_bars"] = 12          # a shared-component variant
    assert tl.shared_settings_hash(c3) != sh
    c4 = json.loads(json.dumps(cfg))
    c4["owner_confirmed"] = not cfg.get("owner_confirmed")     # not a frozen block
    assert tl.config_hash(c4) == ch
    spec = SymbolSpec("US30", 2, 0.01, 0.01, 0.01, 0.1, 250.0, 0.1, "USD", swap_long=-5.0)
    a = tl.config_hash(cfg, {"US30": spec})
    assert a != ch and a != tl.config_hash(cfg, {"US30": replace(spec, swap_long=-6.0)})


def test_git_state(monkeypatch):
    calls = []

    def fake(args, **kw):
        calls.append(args)
        out = "deadbeef\n" if args[1] == "rev-parse" else state["status"]
        return subprocess.CompletedProcess(args, 0, out, "")
    state = {"status": ""}
    monkeypatch.setattr(tl.subprocess, "run", fake)
    assert tl.git_state() == "deadbeef"
    state["status"] = " M backtest/runner.py\n"
    assert tl.git_state() == "deadbeef-dirty"
    assert calls[0][:3] == ["git", "rev-parse", "HEAD"]

    def fail(args, **kw):
        return subprocess.CompletedProcess(args, 128, "", "not a repo")
    monkeypatch.setattr(tl.subprocess, "run", fail)
    assert tl.git_state() == "no commit yet"


def test_json_safe():
    out = tl.json_safe({"a": np.float64("nan"), "b": np.int64(3), 1: (np.bool_(True), math.inf),
                        "t": pd.Timestamp("2024-02-01", tz="UTC")})
    assert out == {"a": None, "b": 3, "1": [True, None], "t": "2024-02-01T00:00:00+00:00"}
    json.dumps(out)


def _daily(days, values):
    return pd.DataFrame({"date": days, "equity": 100000 + np.cumsum(values),
                         "pnl_usd": values, "ret": np.asarray(values) / 100000})


def test_trial_matrix(tmp_path):
    p, runs = tmp_path / "log.csv", tmp_path / "runs"
    days = pd.date_range("2024-02-01", periods=5, freq="B")
    ids = []
    for i, theta in enumerate((1.0, 0.9, 1.1)):
        r = tl.log_trial(rec({"theta": theta, "k": 1.0, "R": 1.5}), path=p)
        tl.save_run_files(tl.trial_dir(r.trial_id, runs), daily=_daily(days, [i, 0, 1, 0, -1]))
        ids.append(r.trial_id)
    b = tl.log_trial(rec(stage="benchmark", variant="B2"), path=p)       # never in the matrix
    tl.save_run_files(tl.trial_dir(b.trial_id, runs), daily=_daily(days, [9, 9, 9, 9, 9]))
    m = tl.trial_matrix(p, runs)
    assert m.shape == (5, 3) == (len(days), tl.n_raw(p))
    assert list(m.columns) == ids and m.index.name == "date"
    assert m[ids[1]].tolist() == [1, 0, 1, 0, -1]
    # a rerun after a bug fix replaces its configuration's column
    r = tl.log_trial(rec({"theta": 0.9, "k": 1.0, "R": 1.5},
                         scenarios={"STRESS-d2": metrics(exp=1.0)}), path=p, rerun_note="fix")
    tl.save_run_files(tl.trial_dir(r.trial_id, runs), daily=_daily(days, [5, 5, 5, 5, 5]))
    m = tl.trial_matrix(p, runs)
    assert list(m.columns) == [ids[0], r.trial_id, ids[2]] and m[r.trial_id].sum() == 25
    assert tl.load_daily(r.trial_id, runs)["pnl_usd"].sum() == 25
    # a counted trial without its daily series, or on another calendar, is an error
    late = tl.log_trial(rec({"theta": 1.2, "k": 1.0, "R": 1.5}), path=p).trial_id
    with pytest.raises(ValueError, match="without daily.parquet"):
        tl.trial_matrix(p, runs)
    tl.save_run_files(tl.trial_dir(late, runs), daily=_daily(days[:4], [1, 1, 1, 1]))
    with pytest.raises(ValueError, match="calendar"):
        tl.trial_matrix(p, runs)


def _set_notes(p, trial_id, **fields):
    rs = tl._read_rows(p)
    for r in rs:
        if r["trial_id"] == trial_id:
            n = json.loads(r["notes"])
            n.update(fields)
            r["notes"] = tl._dumps(n)
    tl._write_rows(p, rs)


def test_trial_matrix_finds_every_strategy_folder(tmp_path):
    """RS-MOM-01 layout runs/<sid>/<tid>, SES-IBX-01 layout runs/<sid>/trials/<tid>; staged
    copies and half-renamed folders are ignored; N_raw counts distinct configurations."""
    p, runs = tmp_path / "log.csv", tmp_path / "runs"
    days = pd.date_range("2024-02-01", periods=5, freq="B")
    rs_dir, ses_dir = runs / "RS-MOM-01", runs / "SES-IBX-01" / "trials"
    ids = {}
    for theta in (1.0, 0.9):
        r = tl.log_trial(rec({"theta": theta, "k": 1.0, "R": 1.5}), path=p)
        tl.save_run_files(rs_dir / r.trial_id, daily=_daily(days, [theta] * 5))
        ids[r.trial_id] = "RS-MOM-01"
    for a in (0.5, 1.0):
        r = tl.log_trial(rec({"alpha": a}, strategy_id="SES-IBX-01", protocol="STOP_BRACKET"),
                         path=p)
        tl.save_run_files(ses_dir / r.trial_id, daily=_daily(days, [a * 10] * 5))
        ids[r.trial_id] = "SES-IBX-01"
    # the same SES configuration logged again by run_a: one trial, no new column
    tl.log_trial(rec({"alpha": 0.5}, strategy_id="SES-IBX-01", protocol="STOP_BRACKET",
                     stage="run_a"), path=p)
    # an uncounted benchmark with its own folder never enters the matrix
    b = tl.log_trial(rec(stage="benchmark", variant="B1", strategy_id="SES-IBX-01",
                         protocol="STOP_BRACKET"), path=p)
    tl.save_run_files(ses_dir / b.trial_id, daily=_daily(days, [9] * 5))
    # staged copies under staged ids and a half-renamed folder are not trial folders
    stray = _daily(days, [7] * 5)
    tl.save_run_files(runs / "RS-MOM-01" / "staging" / "runs" / "T0001", daily=stray)
    tl.save_run_files(runs / "RS-MOM-01" / "staging" / "T0002", daily=stray)
    tl.save_run_files(runs / "staging" / "T0005", daily=stray)
    tl.save_run_files(ses_dir / "_tmp_T0003", daily=stray)
    (runs / "RS-MOM-01" / "artifacts").mkdir()

    where = tl.trial_locations(runs)
    assert sorted(where) == sorted(list(ids) + [b.trial_id])
    assert all(len(v) == 1 for v in where.values())
    assert tl.find_trial_dir("T0001", runs) == rs_dir / "T0001"
    assert tl.find_trial_dir("T0003", runs, "SES-IBX-01") == ses_dir / "T0003"
    assert tl.find_trial_dir("T0999", runs) is None

    m = tl.trial_matrix(p, runs)                                   # default: all strategies
    assert list(m.columns) == list(ids) and m.shape == (5, 4) == (5, tl.n_raw(p))
    assert tl.n_raw(p, "RS-MOM-01") == 2 and tl.n_raw(p, "SES-IBX-01") == 2
    assert m["T0002"].tolist() == [0.9] * 5 and m["T0004"].tolist() == [10.0] * 5
    assert list(tl.trial_matrix(p, runs, strategy_id="SES-IBX-01").columns) == ["T0003", "T0004"]
    assert tl.load_daily("T0003", runs)["pnl_usd"].tolist() == [5.0] * 5
    assert tl.load_daily("T0003", runs, strategy_id="SES-IBX-01")["pnl_usd"].sum() == 25.0

    # notes that name the run folder must match where it is found
    _set_notes(p, "T0001", run_files="backtest/runs/RS-MOM-01/T0001")
    _set_notes(p, "T0003", staged_as={"trial_id": "T0001",
                                      "run_files": "backtest/runs/SES-IBX-01/trials/T0003"})
    assert tl.trial_matrix(p, runs).shape == (5, 4)
    _set_notes(p, "T0004", run_files="backtest/runs/SES-IBX-01/T0004")
    with pytest.raises(ValueError, match="notes name run files"):
        tl.trial_matrix(p, runs)
    _set_notes(p, "T0004", run_files="backtest/runs/SES-IBX-01/trials/T0004")

    # a counted SES trial without its daily.parquet is an error naming the strategy
    (ses_dir / "T0004" / "daily.parquet").unlink()
    with pytest.raises(ValueError, match=r"without daily.parquet.*T0004 \(SES-IBX-01\)"):
        tl.trial_matrix(p, runs)
    tl.save_run_files(ses_dir / "T0004", daily=_daily(days, [10.0] * 5))
    # a trial folder under the other strategy, or two folders for one id, is an error
    tl.save_run_files(rs_dir / "T0004", daily=_daily(days, [1] * 5))
    with pytest.raises(ValueError, match="unique"):
        tl.trial_matrix(p, runs)
    import shutil
    shutil.rmtree(ses_dir / "T0004")
    with pytest.raises(ValueError, match="folder of RS-MOM-01"):
        tl.trial_matrix(p, runs)


def test_flat_layout_still_found_next_to_strategy_folders(tmp_path):
    p, runs = tmp_path / "log.csv", tmp_path / "runs"
    days = pd.date_range("2024-02-01", periods=3, freq="B")
    a = tl.log_trial(rec(), path=p).trial_id
    tl.save_run_files(tl.trial_dir(a, runs), daily=_daily(days, [1, 2, 3]))      # flat
    b = tl.log_trial(rec({"alpha": 1.0}, strategy_id="SES-IBX-01"), path=p).trial_id
    tl.save_run_files(runs / "SES-IBX-01" / "trials" / b, daily=_daily(days, [4, 5, 6]))
    m = tl.trial_matrix(p, runs)
    assert list(m.columns) == [a, b] and m[b].sum() == 15
    assert tl.trial_locations(tmp_path / "nowhere") == {}


def test_selection_log_is_deterministic(tmp_path):
    lines = [(1, "RS-MOM-01 2024-02-05 14:45 UTC G(T) {US30 +0.10}"), (2, "x")]
    a, b = tmp_path / "a.log.gz", tmp_path / "b.log.gz"
    assert tl.write_selection_log(a, "RS-MOM-01", lines) == 2
    tl.write_selection_log(b, "RS-MOM-01", lines)
    assert a.read_bytes() == b.read_bytes()
    import gzip
    assert gzip.decompress(a.read_bytes()).decode().splitlines()[0].startswith(
        "RS-MOM-01\t1\tRS-MOM-01 2024-02-05")


def test_parquet_safe_roundtrip(tmp_path):
    df = pd.DataFrame({"t": [pd.Timestamp("2024-02-01", tz="UTC"), None],
                       "b": [True, None], "x": [1.5, None], "s": ["a", None],
                       "mixed": [1, "a"]}, dtype=object)
    tl.parquet_safe(df).to_parquet(tmp_path / "x.parquet")
    back = pd.read_parquet(tmp_path / "x.parquet")
    assert str(back["t"].dtype).startswith("datetime64") and back["x"].iloc[0] == 1.5
    assert back["mixed"].tolist() == ["1", "a"]


def _fake_run(values):
    from types import SimpleNamespace
    days = pd.date_range("2024-02-01", periods=len(values), freq="B").date
    return SimpleNamespace(daily=_daily(days, values), trades=pd.DataFrame({"x": values}),
                           equity=pd.DataFrame({"x": values}), selection_lines=((1, "s"),))


def test_runner_merge_writes_into_the_existing_strategy_folder(tmp_path):
    """A later scenario merged into an already-logged trial (e.g. MISSED-EXIT into T0001)
    lands in that trial's existing folder (runs/<strategy>/T0001), not in a new flat
    runs/T0001 duplicate, whichever runs_dir the caller passes."""
    from backtest import runner
    p, runs = tmp_path / "log.csv", tmp_path / "runs"
    first = rec(stage="run_a", scenarios={"STRESS-d2": metrics()})
    tl.log_trial(first, path=p)
    home = runs / "RS-MOM-01" / "T0001"
    tl.save_run_files(home, daily=_fake_run([1, 2]).daily)
    later = rec(stage="run_a", scenarios={"STRESS-d2": metrics(), "MISSED-EXIT": metrics(exp=2)})
    res = runner._write_and_log(later, {"STRESS-d2": _fake_run([1, 2]),
                                        "MISSED-EXIT": _fake_run([3, 4])},
                                log_path=p, runs_dir=runs)
    assert res == tl.LogResult("T0001", False, ("MISSED-EXIT",))
    assert (home / "scenarios" / "MISSED-EXIT" / "daily.parquet").exists()
    assert not (runs / "T0001").exists() and len(rows(p)) == 1
    assert tl.find_trial_dir("T0001", runs, "RS-MOM-01") == home
    # the strategy folder passed as runs_dir (the trial is then 'flat') resolves the same way
    more = rec(stage="run_a", scenarios={"STRESS-d2": metrics(), "BASE": metrics(exp=5)})
    runner._write_and_log(more, {"STRESS-d2": _fake_run([1, 2]), "BASE": _fake_run([5, 6])},
                          log_path=p, runs_dir=runs / "RS-MOM-01")
    assert (home / "scenarios" / "BASE" / "daily.parquet").exists()
    # a trial with no folder yet still gets the writer's flat path; a second folder of the
    # same id is refused before the log is touched
    tl.save_run_files(runs / "SES-IBX-01" / "trials" / "T0001", daily=_fake_run([0]).daily)
    before = p.read_text(encoding="utf-8")
    third = rec(stage="run_a", scenarios={"STRESS-d2": metrics(), "COST-2X": metrics(exp=1)})
    with pytest.raises(ValueError, match="2 folders"):
        runner._write_and_log(third, {"STRESS-d2": _fake_run([1, 2]),
                                      "COST-2X": _fake_run([7])}, log_path=p, runs_dir=runs)
    assert p.read_text(encoding="utf-8") == before
    fresh = rec(params={**DEFAULT, "k": 2.0}, stage="run_a")
    res = runner._write_and_log(fresh, {"STRESS-d2": _fake_run([9])}, log_path=p,
                                runs_dir=runs)
    assert res.created and (runs / res.trial_id / "daily.parquet").exists()