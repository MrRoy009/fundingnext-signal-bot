"""Tests for validation.report with fabricated inputs only (no market data, no real outcomes)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validation import report as rp

ROOT = Path(__file__).resolve().parent.parent
SHARES = {1: 1.0, 2: 1.0, 3: 0.697, 4: 0.286, 5: 1.0, 6: 0.803, 7: 0.591, 8: 1.0, 9: 1.0}
DEMO = {"US30": 1.0, "NAS100": 1.0, "GER40": 1.12}
CFG_CONFIRMED = {"costs": {"usd_per_point_demo": DEMO},
                 "rules": {"competition_usd_per_point": dict(DEMO)}}
CFG_UNCONFIRMED = {"costs": {"usd_per_point_demo": DEMO},
                   "rules": {"competition_usd_per_point": None}}
N_LOGGED = 58
_LOG: dict[str, Path] = {}


def _write_trial_log(path: Path, n: int, strategy_id: str = "RS-MOM-01") -> Path:
    """A fabricated trial log with n counted configurations (no outcomes in it)."""
    import json
    from backtest.trials import COLUMNS, _write_rows
    rows = [dict(trial_id=f"T{i:04d}", date_utc="2026-11-05T00:00:00Z", strategy_id=strategy_id,
                 params=f"theta={i}", data_window="synthetic", stage="run_a", result_summary="",
                 notes=json.dumps({"config_key": f"key{i}", "stages": ["run_a"]}))
            for i in range(1, n + 1)]
    _write_rows(path, rows)
    assert list(rows[0]) == COLUMNS
    return path


def _log58() -> Path:
    """A fabricated 58-configuration trial log in a temp dir (created once, also outside pytest
    so QA probes can reuse the fixtures)."""
    if "58" not in _LOG:
        import tempfile
        _LOG["58"] = _write_trial_log(Path(tempfile.mkdtemp(prefix="report_log_"))
                                      / "trial_log.csv", N_LOGGED)
    return _LOG["58"]


# --------------------------------------------------------------------------------------------
# Fabricated inputs
# --------------------------------------------------------------------------------------------

def _trades(n_by: dict[tuple[str, str, str], int], pnl: float = 10.0, seed: int = 0,
            regime: str = "ACTIVE/ACTIVE_FLAT") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2024-02-05 15:00", tz="UTC")
    i = 0
    for (key, side, ex), n in n_by.items():
        for _ in range(n):
            p = pnl + float(rng.normal(0, 1))
            rows.append(dict(key=key, side=side, exit_type=ex, status="filled", pnl_usd=p,
                             r_multiple=p / 300.0, regime=regime,
                             signal_time=t0 + pd.Timedelta(days=i)))
            i += 1
    rows.append(dict(key="US30", side="BUY", exit_type=None, status="skipped", pnl_usd=np.nan,
                     r_multiple=np.nan, regime=regime, signal_time=t0))
    return pd.DataFrame(rows)


def _scen(name: str, exp: float = 5.0, r: float = 0.02, sh: float = 0.05, n: int = 200,
          total: float | None = None) -> rp.ScenarioStats:
    return rp.ScenarioStats(name, n, exp, r, sh, exp * n if total is None else total, 800.0,
                            0.008, n_days=500, win_rate=0.55)


def _folds(pnl=(10.0,) * 9, trades=(30, 30, 20, 2, 30, 20, 8, 30, 30)) -> list[rp.FoldStats]:
    return [rp.FoldStats(i, f"w{i}", SHARES[i], trades[i - 1], pnl[i - 1]) for i in range(1, 10)]


def _grid(exp_fn=lambda p: 5.0, n_fn=lambda p: 150) -> rp.Sensitivity:
    defaults = {"theta": 1.0, "k": 1.0, "R": 1.5}
    grids = {"theta": [0.8, 0.9, 1.0, 1.1, 1.2], "k": [0.75, 0.875, 1.0, 1.125, 1.25],
             "R": [1.0, 1.25, 1.5, 1.75, 2.0]}
    cells = []
    for p, vals in grids.items():                     # one at a time (13 distinct)
        for v in vals:
            pv = dict(defaults, **{p: v})
            cells.append(rp.SensitivityCell(pv, n_fn(pv), exp_fn(pv), 0.01,
                                            {"SL": 0.45, "TP": 0.35, "time_exit": 0.2}))
    for k in grids["k"]:                              # heatmap k x R (16 new)
        for r in grids["R"]:
            pv = dict(defaults, k=k, R=r)
            cells.append(rp.SensitivityCell(pv, n_fn(pv), exp_fn(pv), 0.01,
                                            {"SL": 0.5, "TP": 0.3, "time_exit": 0.2}))
    return rp.Sensitivity(defaults, cells, "k", "R")


def _replay(contract="demo", total=0, daily=0, n=100) -> rp.ReplayStats:
    return rp.ReplayStats(contract, dict(DEMO), n, total, daily, 3, 90)


RUN_B_PARAMS = [{"theta": 0.9, "k": 1.0, "R": 1.5}, {"theta": 1.0, "k": 1.0, "R": 1.5},
                {"theta": 1.0, "k": 1.125, "R": 1.5}, {"theta": 1.0, "k": 1.0, "R": 1.75},
                {"theta": 1.2, "k": 1.0, "R": 1.5}, {"theta": 1.0, "k": 0.75, "R": 1.5},
                {"theta": 1.0, "k": 1.0, "R": 1.0}, {"theta": 1.1, "k": 1.0, "R": 1.5},
                {"theta": 1.0, "k": 1.0, "R": 1.5}]


def _run_b(selections=None, trades: int = 180, total: float = 540.0) -> rp.RunBStats:
    """Nine one-at-a-time selections, 20 trades / $60 each, summing to the totals."""
    sel = selections if selections is not None else [
        rp.RunBSelection(i, RUN_B_PARAMS[i - 1], 0.04, 20, 60.0, n_embargoed=i % 3)
        for i in range(1, 10)]
    return rp.RunBStats(sel, trades, total / trades, 0.01, total, 0.03)


def _with_counts(inp: rp.ValidationInputs, n: int | None = None) -> rp.ValidationInputs:
    """Make STRESS-d2's trade count equal the fold sum (or n) and drop the trades table, so a
    test that varies fold counts is not caught by the count cross-check."""
    n = sum(f.trades for f in inp.folds) if n is None else n
    sc = [replace(s, trades=n) if s.name == "STRESS-d2" else s for s in inp.scenarios]
    return replace(inp, scenarios=sc, trades=None)


def _passing_inputs(**over) -> rp.ValidationInputs:
    prov = rp.Provenance("RS-MOM-01", "abc123", "data9ab5", "cfg54c4",
                         {"theta": 1.0, "k": 1.0, "R": 1.5}, data_end=date(2026, 4, 30),
                         trial_ids=("T001",), generated_utc=datetime(2026, 11, 5, tzinfo=timezone.utc),
                         synthetic=True)
    base = dict(
        provenance=prov,
        scenarios=[_scen(n) for n in rp.SCENARIO_ORDER],
        folds=_folds(),
        trades=_trades({("US30", "BUY", "SL"): 40, ("NAS100", "SELL", "TP"): 40,
                        ("GER40", "BUY", "time_exit"): 115, ("GER40", "SELL", "SL"): 5}),
        run_b=_run_b(),
        sensitivity=_grid(),
        bootstrap=rp.BootstrapStats({5: 100.0, 50: 900.0, 95: 1700.0},
                                    {5: 300.0, 50: 600.0, 95: 1200.0},
                                    scenario="STRESS-d2", n_obs=500),
        regime_bootstrap=rp.BootstrapStats({5: 50.0, 50: 900.0, 95: 1800.0},
                                           {5: 300.0, 50: 650.0, 95: 1300.0},
                                           mean_block_length=None, bucket_counts={"ACTIVE": 200},
                                           method="regime_conditional",
                                           scenario="STRESS-d2", n_obs=200),
        shuffle=rp.ShuffleStats(7.0, 1500.0),
        benchmarks=rp.BenchmarkStats(b1_p95_total_usd=600.0, b1_mean_total_usd=0.0,
                                     b2_mean_usd=1.0, b2_trades=150, b3_expectancy_usd=2.0,
                                     b3_trades=400, b3_total_usd=800.0,
                                     b1_seeds=tuple(range(1, 201))),
        replay=[_replay("demo"), _replay("competition")],
        dsr=rp.DsrStats(0.05, 500, 0.1, 3.5, 58, 0.97, 0.001, 12, 0.98, 0.0012),
        pbo=rp.PboStats(0.1, 58),
        regime=rp.RegimeDiagnostics("ACTIVE", 0.578, "functional"),
        trade_day_share=0.9,
        config=CFG_CONFIRMED,
        trial_log=_log58(),
    )
    base.update(over)
    return rp.ValidationInputs(**base)


def _row(rows, crit):
    return next(r for r in rows if r.criterion == crit)


# --------------------------------------------------------------------------------------------
# Criteria and verdict
# --------------------------------------------------------------------------------------------

def test_all_pass_gives_pass():
    rows = rp.evaluate_criteria(_passing_inputs())
    judged = [r for r in rows if r.status != rp.INFO]
    assert {r.criterion for r in judged} == {"10.4 trades", "10.4 folds", "1", "2", "3", "4",
                                             "5", "6", "7", "8", "9"}
    assert all(r.status == rp.PASS for r in judged), [(r.criterion, r.status) for r in judged]
    assert rp.overall_verdict(rows) == rp.VERDICT_PASS


@pytest.mark.parametrize("name,field,value,crit", [
    ("STRESS-d2", "expectancy_usd", -1.0, "1"),
    ("STRESS-d3", "expectancy_r", -0.01, "1"),
    ("STRESS-d3", "daily_sharpe", 0.0, "1"),
    ("MISSED-EXIT", "expectancy_usd", 0.0, "2"),
])
def test_scenario_criteria_fail(name, field, value, crit):
    inp = _passing_inputs()
    sc = [replace(s, **{field: value}) if s.name == name else s for s in inp.scenarios]
    rows = rp.evaluate_criteria(replace(inp, scenarios=sc))
    assert _row(rows, crit).status == rp.FAIL
    assert rp.overall_verdict(rows) == rp.VERDICT_FAIL


def test_report_only_scenarios_do_not_affect_verdict():
    inp = _passing_inputs()
    sc = [replace(s, expectancy_usd=-50.0) if s.name in ("BASE", "STRESS-d1", "EXIT-3X") else s
          for s in inp.scenarios]
    assert rp.overall_verdict(rp.evaluate_criteria(replace(inp, scenarios=sc))) == rp.VERDICT_PASS


def test_total_trade_minimum():
    inp = _passing_inputs(folds=())
    r = _row(rp.evaluate_criteria(_with_counts(inp, 149)), "10.4 trades")
    assert r.status == rp.FAIL
    assert _row(rp.evaluate_criteria(_with_counts(inp, 150)), "10.4 trades").status == rp.PASS


def test_prorated_fold_minimum():
    f = rp.FoldStats(4, "w4", 0.286, 1, 5.0)
    assert f.prorated_minimum == pytest.approx(1.144)
    assert not f.meets_minimum
    assert replace(f, trades=2).meets_minimum
    assert rp.FoldStats(7, "w7", 0.0, 0, 0.0).meets_minimum          # share 0 -> minimum 0
    assert rp.FoldStats(7, "w7", float("nan"), 3, 0.0).prorated_minimum == 4.0   # unknown: full
    # Two folds below the minimum -> 7 of 9 still pass; three -> fail.
    tr = [30, 30, 20, 1, 30, 20, 2, 30, 30]
    rows = rp.evaluate_criteria(_with_counts(_passing_inputs(folds=_folds(trades=tr))))
    assert _row(rows, "10.4 folds").status == rp.PASS
    tr[0] = 3
    rows = rp.evaluate_criteria(_with_counts(_passing_inputs(folds=_folds(trades=tr))))
    assert _row(rows, "10.4 folds").status == rp.FAIL
    assert _row(rows, "10.4 folds").measured == "6 of 9"


def test_positive_folds_criterion_not_prorated():
    pnl = [10.0] * 6 + [0.0, -1.0, -2.0]
    assert _row(rp.evaluate_criteria(_passing_inputs(folds=_folds(pnl=pnl))), "4").status == rp.PASS
    pnl[0] = 0.0                                    # zero P&L (e.g. no trades) is not positive
    r = _row(rp.evaluate_criteria(_passing_inputs(folds=_folds(pnl=pnl))), "4")
    assert r.status == rp.FAIL and r.measured == "5 of 9"


def test_missing_folds_not_evaluated():
    rows = rp.evaluate_criteria(_passing_inputs(folds=_folds()[:8]))
    assert _row(rows, "4").status == rp.NOT_EVALUATED
    assert rp.overall_verdict(rows) == rp.VERDICT_INCOMPLETE


def test_run_b_criterion():
    inp = _passing_inputs()
    rows = rp.evaluate_criteria(replace(inp, run_b=replace(inp.run_b, expectancy_usd=-0.5)))
    assert _row(rows, "3").status == rp.FAIL
    assert _row(rp.evaluate_criteria(replace(inp, run_b=None)), "3").status == rp.NOT_EVALUATED


def test_sensitivity_distinct_cells_and_neighbours():
    ss = rp.sensitivity_summary(_grid())
    assert ss["n_cells"] == 29                    # 13 one-at-a-time + 16 heatmap, deduplicated
    assert len(ss["neighbours"]) == 6             # 2 per parameter; = heatmap axis neighbours
    assert ss["neighbours_ok"] and ss["share"] == 1.0


def test_sensitivity_neighbour_negative_fails():
    sens = _grid(exp_fn=lambda p: -1.0 if (p["k"] == 1.125 and p["R"] == 1.5
                                            and p["theta"] == 1.0) else 5.0)
    r = _row(rp.evaluate_criteria(_passing_inputs(sensitivity=sens)), "5")
    assert r.status == rp.FAIL
    assert "neighbours 5 of 6" in r.measured


def test_sensitivity_far_cell_negative_and_share_threshold():
    # A non-neighbour negative cell alone still passes (28/29 = 96.6%).
    sens = _grid(exp_fn=lambda p: -1.0 if (p["k"] == 0.75 and p["R"] == 1.0) else 5.0)
    assert _row(rp.evaluate_criteria(_passing_inputs(sensitivity=sens)), "5").status == rp.PASS
    # Six bad heatmap corners/edges (non-neighbours) -> 23/29 = 79.3% < 80% -> fail.
    bad = {(0.75, 1.0), (0.75, 2.0), (1.25, 1.0), (1.25, 2.0), (0.75, 1.25), (1.25, 1.25)}
    sens = _grid(exp_fn=lambda p: -1.0 if (p["k"], p["R"]) in bad else 5.0)
    r = _row(rp.evaluate_criteria(_passing_inputs(sensitivity=sens)), "5")
    assert r.status == rp.FAIL and "23 of 29" in r.measured


def test_sensitivity_insufficient_cell_counts_as_failed():
    sens = _grid(n_fn=lambda p: 99 if (p["theta"] == 0.9) else 150)   # theta neighbour
    r = _row(rp.evaluate_criteria(_passing_inputs(sensitivity=sens)), "5")
    assert r.status == rp.FAIL and "1 insufficient" in r.measured


def test_bootstrap_placebo_dsr_pbo():
    inp = _passing_inputs()
    bs = replace(inp.bootstrap, total_pnl_q={5: -1.0, 50: 900.0, 95: 1700.0})
    assert _row(rp.evaluate_criteria(replace(inp, bootstrap=bs)), "6").status == rp.FAIL
    # Run A total (STRESS-d2) is 5 * 200 = 1000; placebo p95 equal -> fail (strictly greater).
    bm = replace(inp.benchmarks, b1_p95_total_usd=1000.0)
    assert _row(rp.evaluate_criteria(replace(inp, benchmarks=bm)), "7").status == rp.FAIL
    for d, p, st in [(replace(inp.dsr, dsr_eff=0.949), inp.pbo, rp.FAIL),
                     (inp.dsr, replace(inp.pbo, pbo=0.5), rp.FAIL),
                     (inp.dsr, replace(inp.pbo, pbo=0.3), rp.PASS),
                     (inp.dsr, replace(inp.pbo, pbo=float("nan")), rp.NOT_EVALUATED)]:
        r = _row(rp.evaluate_criteria(replace(inp, dsr=d, pbo=p)), "8")
        assert r.status == st
    r = _row(rp.evaluate_criteria(replace(inp, pbo=replace(inp.pbo, pbo=0.3))), "8")
    assert "flagged" in r.measured


def test_replay_criterion_and_unconfirmed_contract():
    inp = _passing_inputs()
    r = _row(rp.evaluate_criteria(replace(inp, replay=[_replay("demo", total=1),
                                                       _replay("competition")])), "9")
    assert r.status == rp.FAIL
    r = _row(rp.evaluate_criteria(replace(inp, replay=[_replay("demo", daily=2)])), "9")
    assert r.status == rp.FAIL                     # 2% > 1%, even with competition missing
    rows = rp.evaluate_criteria(replace(inp, replay=[_replay("demo", daily=1)]))
    assert _row(rows, "9").status == rp.NOT_EVALUATED   # 1% passes demo; competition pending
    assert rp.overall_verdict(rows) == rp.VERDICT_INCOMPLETE
    text = rp.render_report(replace(inp, replay=[_replay("demo")], config=CFG_UNCONFIRMED))
    assert rp.COMPETITION_UNCONFIRMED in text


def test_info_rows_never_affect_verdict():
    inp = _passing_inputs(shuffle=None, trade_day_share=0.1)
    rows = rp.evaluate_criteria(inp)
    assert {r.status for r in rows if r.criterion.endswith("(report)")} == {rp.INFO}
    assert rp.overall_verdict(rows) == rp.VERDICT_PASS


def test_empty_inputs_are_incomplete_not_pass():
    prov = rp.Provenance("SES-IBX-01", "x", "y", "z", {})
    rows = rp.evaluate_criteria(rp.ValidationInputs(provenance=prov))
    assert rp.overall_verdict(rows) == rp.VERDICT_INCOMPLETE
    assert rp.PASS not in {r.status for r in rows}


# --------------------------------------------------------------------------------------------
# Tables and helpers
# --------------------------------------------------------------------------------------------

def test_bucket_rule_merges_small_buckets():
    tr = _trades({("US30", "BUY", "SL"): 40, ("NAS100", "SELL", "TP"): 20,
                  ("GER40", "BUY", "time_exit"): 15, ("GER40", "SELL", "end_of_data"): 3})
    bt = rp.bucket_table(tr, "key")
    assert list(bt["bucket"]) == ["US30", "OTHER"]
    other = bt.iloc[1]
    assert other["n"] == 38 and other["merged"] == "GER40, NAS100" and not other["withheld"]
    assert bt["n"].sum() == 78                   # skipped row excluded
    bt = rp.bucket_table(tr, "exit_type")             # TP 20 + time_exit 15 + end 3 = 38
    assert list(bt["bucket"]) == ["SL", "OTHER"] and bt["n"].iloc[1] == 38
    bt = rp.bucket_table(tr, "side")                  # BUY 55, SELL 23 -> OTHER withheld
    other = bt[bt["bucket"] == "OTHER"].iloc[0]
    assert other["n"] == 23 and other["withheld"] and np.isnan(other["mean_usd"])


def test_bucket_regime_uses_volatility_prefix():
    tr = pd.concat([_trades({("US30", "BUY", "SL"): 31}, regime="ACTIVE/ACTIVE_FLAT"),
                    _trades({("US30", "BUY", "SL"): 31}, regime="ACTIVE/CALM_UP")])
    bt = rp.bucket_table(tr, "regime")
    assert list(bt["bucket"]) == ["ACTIVE"] and bt["n"].iloc[0] == 62


def test_scenario_stats_from_backtest_frames():
    tr = pd.DataFrame(dict(status=["filled", "filled", "skipped"], pnl_usd=[100.0, -40.0, np.nan],
                           r_multiple=[0.5, -0.2, np.nan]))
    daily = pd.DataFrame(dict(date=[date(2024, 2, d) for d in (1, 2, 3, 4)],
                              balance=[1e5] * 4, equity=[100_100.0, 100_000.0, 100_060.0,
                                                         100_060.0]))
    s = rp.scenario_stats("BASE", tr, daily)
    assert s.trades == 2 and s.total_usd == 60.0 and s.expectancy_usd == 30.0
    assert s.expectancy_r == pytest.approx(0.15)
    d = np.array([100.0, -100.0, 60.0, 0.0])
    assert s.daily_sharpe == pytest.approx(d.mean() / d.std(ddof=1))
    assert s.max_dd_usd == pytest.approx(100.0)
    assert s.max_dd_pct == pytest.approx(100.0 / 100_100.0)


def test_fold_stats_assigns_by_signal_time():
    from backtest.feasibility import TEST_FOLDS
    tr = pd.DataFrame(dict(status=["filled"] * 3,
                           signal_time=pd.to_datetime(["2024-02-01 09:00", "2024-04-30 23:59",
                                                       "2024-05-01 00:00"], utc=True),
                           pnl_usd=[1.0, 2.0, -5.0], r_multiple=[0.1, 0.2, -0.5]))
    fs = rp.fold_stats(tr, SHARES, TEST_FOLDS)
    assert [f.trades for f in fs] == [2, 1, 0, 0, 0, 0, 0, 0, 0]
    assert fs[0].pnl_usd == 3.0 and fs[1].pnl_usd == -5.0 and fs[3].data_share == 0.286


def test_rolling_trade_day_share():
    cal = [date(2024, 1, 1) + pd.Timedelta(days=i) for i in range(40)]
    cal = [d.date() if hasattr(d, "date") else d for d in cal]
    traded = cal[::4]                              # 1 trade day in 4 -> 5 per 20-day window
    assert rp.rolling_trade_day_share(traded, cal) == 1.0
    assert rp.rolling_trade_day_share(cal[:3], cal) == 0.0
    assert np.isnan(rp.rolling_trade_day_share([], cal[:10]))


def test_parse_feasibility_report():
    text = (ROOT / "docs" / "feasibility_gate.md").read_text(encoding="utf-8")
    shares = rp.parse_fold_shares(text)
    assert len(shares) == 9 and shares[4] == pytest.approx(0.286) and shares[1] == 1.0
    rg = rp.parse_regime_flags(text, "RS-MOM-01")
    assert rg.gate == "ACTIVE" and rg.flag == "functional" and rg.functional
    assert rg.block_share == pytest.approx(0.578)
    assert set(rg.per_index) == {"US30", "NAS100", "GER40"}
    assert list(rg.non_functional_folds) == [4, 7]
    with pytest.raises(ValueError):
        rp.parse_regime_flags(text, "NOPE-01")


def test_stats_adapters_duck_typed():
    from types import SimpleNamespace as NS
    q = NS(p5=1.0, p50=2.0, p95=3.0)
    b = rp.bootstrap_from(NS(total_pnl=q, max_drawdown=q, n_resamples=10, seed=1, mean_block=5.0,
                             buckets=None, method="regime_conditional", n_obs=42), "STRESS-d2")
    assert b.total_pnl_q == {5: 1.0, 50: 2.0, 95: 3.0} and b.bucket_counts == {}
    assert b.method == "regime_conditional" and b.scenario == "STRESS-d2" and b.n_obs == 42
    s = rp.shuffle_from(NS(worst_streak=q, max_drawdown=q, n_permutations=10, seed=1))
    assert s.worst_streak_p95 == 3.0
    res = NS(sr_hat=0.1, t_days=100, skew=0.0, kurtosis=3.0, n_trials=58, dsr=0.96,
             sr_variance=0.001)
    d = rp.dsr_from(NS(raw=res, eff=NS(**{**vars(res), "n_trials": 10, "dsr": 0.97})))
    assert d.n_raw == 58 and d.n_eff == 10 and d.dsr_eff == 0.97
    p = rp.pbo_from(NS(pbo=0.1, n_trials=58, n_splits=12870, n_blocks=16))
    assert p.pbo == 0.1


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------

def test_render_contains_every_section_and_mandatory_lines():
    text = rp.render_report(_passing_inputs())
    for needle in ("## Provenance", "Git SHA: abc123", "Data SHA-256: data9ab5",
                   "config.yaml SHA-256: cfg54c4", "## Scenarios", "STRESS-d3", "MISSED-EXIT",
                   "EXIT-3X", "## Walk-forward folds", "Pro-rated minimum", "### Per index",
                   "### Per side", "### Per exit type", "## Run B", "### One at a time",
                   "### Heatmap k x R", "exits k=0.75", "## Bootstrap", "Regime-conditional",
                   "## Shuffle test", "## Benchmarks", "B1 placebo", "## Competition replay",
                   "N_raw", "N_eff", "PBO", "## Regime diagnostics", "## Pass / fail",
                   "**Overall verdict: PASS**", rp.DEMO_NOTE, rp.HOLDOUT_NOTE, "SYNTHETIC DATA",
                   "## SPEC NOTEs"):
        assert needle in text, needle
    assert rp.SENTENCE_RS not in text and rp.SENTENCE_GATE not in text
    assert "**+5.00 (n=150)**" in text            # default heatmap cell in bold
    # 10 criterion rows + 2 minimums + 2 info rows in the pass/fail table.
    table = text.split("## Pass / fail")[1].split("**Overall verdict")[0]
    assert table.count("| **") == 13


def test_render_heatmap_grid_shape():
    text = rp.render_report(_passing_inputs())
    block = text.split("### Heatmap k x R")[1].split("Distinct grid cells")[0]
    rows = [ln for ln in block.splitlines() if ln.startswith("| R=")]
    assert [r.split("|")[1].strip() for r in rows] == ["R=2", "R=1.75", "R=1.5", "R=1.25", "R=1"]
    assert all(r.count("|") == 12 for r in rows)   # label + 5 values + 5 exit-share columns


def test_mandated_benchmark_sentences():
    inp = _passing_inputs()
    bm = replace(inp.benchmarks, b2_mean_usd=5.0, b3_expectancy_usd=5.0)   # ties -> sentences
    text = rp.render_report(replace(inp, benchmarks=bm))
    assert rp.SENTENCE_RS in text and rp.SENTENCE_GATE in text


def test_non_functional_gate_blocks_regime_claims():
    inp = _passing_inputs(regime=rp.RegimeDiagnostics("ACTIVE", 0.97, "NON-FUNCTIONAL (> 95%)"))
    text = rp.render_report(inp)
    assert "no regime-conditional claim may be made" in text


def test_insufficient_cell_rendered():
    sens = _grid(n_fn=lambda p: 99 if p["k"] == 0.75 else 150)
    assert "insufficient (n=99)" in rp.render_report(_passing_inputs(sensitivity=sens))


def test_holdout_guard():
    inp = _passing_inputs()
    with pytest.raises(ValueError):
        rp.render_report(replace(inp, provenance=replace(inp.provenance, final_holdout_used=True)))
    with pytest.raises(ValueError):
        rp.render_report(replace(inp, provenance=replace(inp.provenance,
                                                         data_end=date(2026, 5, 1))))


def test_write_report(tmp_path):
    path = rp.write_report(_passing_inputs(), tmp_path)
    assert path == tmp_path / "RS-MOM-01.md"
    assert path.read_text(encoding="utf-8").startswith("# Validation report: RS-MOM-01")


def test_default_output_dir():
    assert rp.DEFAULT_OUT_DIR == ROOT / "docs" / "validation"


# --------------------------------------------------------------------------------------------
# QA round 2: nothing a caller can omit or change makes a criterion PASS
# --------------------------------------------------------------------------------------------

def _sub_grid(keep) -> rp.Sensitivity:
    g = _grid()
    return replace(g, cells=[c for c in g.cells if keep(c.params)])


def test_sensitivity_missing_cells_never_pass():
    # Default + its 6 neighbours, all positive: 7 of 29 run -> NOT EVALUATED, not PASS.
    d = {"theta": 1.0, "k": 1.0, "R": 1.5}
    nbrs = [dict(d, theta=0.9), dict(d, theta=1.1), dict(d, k=0.875), dict(d, k=1.125),
            dict(d, R=1.25), dict(d, R=1.75), d]
    sens = _sub_grid(lambda p: any(dict(p) == n for n in nbrs))
    r = _row(rp.evaluate_criteria(_passing_inputs(sensitivity=sens)), "5")
    assert r.status == rp.NOT_EVALUATED
    assert "cells 7 of 29 positive" in r.measured and "22 missing" in r.measured
    # The full grid with its 6 failing heatmap cells dropped: 23 of 29, not "23 of 23".
    bad = {(0.75, 1.0), (0.75, 2.0), (1.25, 1.0), (1.25, 2.0), (0.75, 1.25), (1.25, 1.25)}
    sens = _sub_grid(lambda p: (p["k"], p["R"]) not in bad)
    ss = rp.sensitivity_summary(sens, "RS-MOM-01")
    assert ss["n_expected"] == 29 and ss["n_cells"] == 23 and len(ss["missing"]) == 6
    r = _row(rp.evaluate_criteria(_passing_inputs(sensitivity=sens)), "5")
    assert r.status == rp.NOT_EVALUATED and "23 of 29" in r.measured
    text = rp.render_report(_passing_inputs(sensitivity=sens))
    assert "Cells not run (counted as not positive)" in text and "k=0.75, R=1" in text
    assert "**Overall verdict: PASS**" not in text


def test_sensitivity_missing_cells_with_certain_failure_is_fail():
    # A failing neighbour fails the criterion whatever the missing cells would show.
    sens = _grid(exp_fn=lambda p: -1.0 if (p["theta"] == 0.9) else 5.0)
    sens = replace(sens, cells=[c for c in sens.cells if c.params["k"] != 0.75])
    assert _row(rp.evaluate_criteria(_passing_inputs(sensitivity=sens)), "5").status == rp.FAIL


def test_sensitivity_grid_must_be_registered():
    g = _grid()
    # An off-grid cell (an unregistered configuration) -> NOT EVALUATED.
    extra = rp.SensitivityCell({"theta": 1.0, "k": 1.0, "R": 3.0}, 150, 5.0)
    r = _row(rp.evaluate_criteria(_passing_inputs(sensitivity=replace(
        g, cells=list(g.cells) + [extra]))), "5")
    assert r.status == rp.NOT_EVALUATED and "off the registered grid" in r.measured
    # A supplied grid different from the RS-MOM-01 ParamGrid -> NOT EVALUATED.
    r = _row(rp.evaluate_criteria(_passing_inputs(sensitivity=replace(
        g, grid={"theta": (1.0,), "k": (1.0,), "R": (1.5,)}))), "5")
    assert r.status == rp.NOT_EVALUATED and "differs from the registered" in r.measured
    # Wrong defaults or heatmap pair -> NOT EVALUATED.
    for s in (replace(g, defaults={"theta": 1.1, "k": 1.0, "R": 1.5}),
              replace(g, heatmap_x="theta")):
        assert _row(rp.evaluate_criteria(_passing_inputs(sensitivity=s)), "5").status \
            == rp.NOT_EVALUATED
    # Unregistered strategy id: the grid must be supplied.
    prov = replace(_passing_inputs().provenance, strategy_id="SYN-01")
    r = _row(rp.evaluate_criteria(_passing_inputs(provenance=prov, sensitivity=g)), "5")
    assert r.status == rp.NOT_EVALUATED and "no registered or supplied grid" in r.measured
    grid = {"theta": [0.8, 0.9, 1.0, 1.1, 1.2], "k": [0.75, 0.875, 1.0, 1.125, 1.25],
            "R": [1.0, 1.25, 1.5, 1.75, 2.0]}
    r = _row(rp.evaluate_criteria(_passing_inputs(provenance=prov,
                                                  sensitivity=replace(g, grid=grid))), "5")
    assert r.status == rp.PASS


def test_ses_registered_grid_has_29_cells():
    pg = rp.registered_grid("SES-IBX-01")
    cells = [rp.SensitivityCell(p, 150, 5.0) for p in pg.configurations()]
    sens = rp.Sensitivity(dict(pg.defaults), cells, *pg.heatmap)
    ss = rp.sensitivity_summary(sens, "SES-IBX-01")
    assert ss["n_expected"] == 29 and not ss["missing"] and ss["status"] == rp.PASS
    assert len(ss["neighbours"]) == 6
    assert rp.registered_grid("MR-TWAP-01") is None


@pytest.mark.parametrize("over", [dict(n_resamples=50), dict(seed=1),
                                  dict(mean_block_length=1.0), dict(mean_block_length=None),
                                  dict(method="regime_conditional")])
def test_bootstrap_settings_must_be_frozen(over):
    inp = _passing_inputs()
    r = _row(rp.evaluate_criteria(replace(inp, bootstrap=replace(inp.bootstrap, **over))), "6")
    assert r.status == rp.NOT_EVALUATED and "NOT EVALUATED" in r.measured


@pytest.mark.parametrize("over", [dict(b1_draws=3), dict(b1_seeds=None),
                                  dict(b1_seeds=tuple(range(2, 202))),
                                  dict(b1_seeds=tuple(range(1, 4)))])
def test_placebo_settings_must_be_frozen(over):
    inp = _passing_inputs()
    r = _row(rp.evaluate_criteria(replace(inp, benchmarks=replace(inp.benchmarks, **over))), "7")
    assert r.status == rp.NOT_EVALUATED


@pytest.mark.parametrize("over", [dict(blocks=4), dict(n_splits=6), dict(n_trials=57)])
def test_pbo_settings_must_be_frozen(over):
    inp = _passing_inputs()
    r = _row(rp.evaluate_criteria(replace(inp, pbo=replace(inp.pbo, **over))), "8")
    assert r.status == rp.NOT_EVALUATED


def test_shuffle_settings_warned_but_info():
    inp = _passing_inputs(shuffle=rp.ShuffleStats(7.0, 1500.0, n_permutations=10, seed=1))
    rows = rp.evaluate_criteria(inp)
    r = _row(rows, "10 (report)")
    assert r.status == rp.INFO and "settings not frozen" in r.measured
    assert rp.overall_verdict(rows) == rp.VERDICT_PASS
    assert "shuffle test not at the frozen settings" in rp.render_report(inp)


def test_regime_bootstrap_settings_warned():
    inp = _passing_inputs()
    inp = replace(inp, regime_bootstrap=replace(inp.regime_bootstrap, n_resamples=10))
    assert "Regime-conditional bootstrap of trades not at the frozen settings" \
        in rp.render_report(inp)


def test_trial_accounting_against_trial_log(tmp_path):
    log5 = _write_trial_log(tmp_path / "trial_log.csv", 5)
    inp = _passing_inputs(trial_log=log5, dsr=replace(_passing_inputs().dsr, n_raw=5, n_eff=3),
                          pbo=rp.PboStats(0.1, 5))
    r = _row(rp.evaluate_criteria(inp), "8")
    assert r.status == rp.PASS and "trial log 5" in r.measured
    for d, p in [(replace(inp.dsr, n_raw=1, n_eff=1), rp.PboStats(0.1, 1)),   # < logged
                 (replace(inp.dsr, n_raw=58), rp.PboStats(0.1, 58)),           # > logged
                 (replace(inp.dsr, n_eff=12), inp.pbo),                        # N_eff > N_raw
                 (replace(inp.dsr, n_eff=0), inp.pbo),                         # N_eff < 1
                 (inp.dsr, rp.PboStats(0.1, 4))]:                             # PBO trials
        r = _row(rp.evaluate_criteria(replace(inp, dsr=d, pbo=p)), "8")
        assert r.status == rp.NOT_EVALUATED, (d, p)
    # An empty log: N_raw cannot be confirmed.
    empty = tmp_path / "empty.csv"
    r = _row(rp.evaluate_criteria(replace(inp, trial_log=empty)), "8")
    assert r.status == rp.NOT_EVALUATED
    text = rp.render_report(replace(inp, dsr=replace(inp.dsr, n_raw=1, n_eff=1)))
    assert "Trial log: 5 counted configurations" in text and "criterion 8 NOT EVALUATED" in text


def test_competition_replay_ignored_while_o4_unconfirmed():
    inp = _passing_inputs(config=CFG_UNCONFIRMED)       # both replays present and passing
    rows = rp.evaluate_criteria(inp)
    r = _row(rows, "9")
    assert r.status == rp.NOT_EVALUATED and "NOT confirmed (O4)" in r.measured
    assert rp.overall_verdict(rows) == rp.VERDICT_INCOMPLETE
    text = rp.render_report(inp)
    head = text.split("## Provenance")[0]
    assert rp.COMPETITION_UNCONFIRMED in head
    assert rp.COMPETITION_IGNORED in text and "competition (IGNORED" in text
    # The demo replay still fails the criterion on its own.
    r = _row(rp.evaluate_criteria(replace(inp, replay=[_replay("demo", total=1),
                                                       _replay("competition")])), "9")
    assert r.status == rp.FAIL


def test_real_config_has_o4_unconfirmed():
    # config.yaml today: rules.competition_usd_per_point is null -> criterion 9 cannot pass.
    from core.config import load_config
    assert not rp.competition_confirmed(load_config())


def test_replay_contract_must_match_config():
    inp = _passing_inputs()
    other = replace(_replay("competition"), usd_per_point={"US30": 0.5, "NAS100": 1.0,
                                                           "GER40": 1.12})
    r = _row(rp.evaluate_criteria(replace(inp, replay=[_replay("demo"), other])), "9")
    assert r.status == rp.NOT_EVALUATED and "differs from config" in r.measured


def test_real_report_reads_config_and_log_from_disk():
    inp = _passing_inputs()
    real = replace(inp, provenance=replace(inp.provenance, synthetic=False))
    with pytest.raises(ValueError, match="config.yaml"):
        rp.evaluate_criteria(replace(real, trial_log=None))
    with pytest.raises(ValueError, match="counts trials in"):
        rp.evaluate_criteria(replace(real, config=None))


def test_benchmark_and_run_b_scenario_must_be_stress_d2():
    inp = _passing_inputs()
    sc = [replace(s, total_usd=-500.0) if s.name == "STRESS-d2" else s for s in inp.scenarios]
    bm = replace(inp.benchmarks, scenario="BASE")          # BASE total 1000 > p95 600
    rows = rp.evaluate_criteria(replace(inp, scenarios=sc, benchmarks=bm))
    assert _row(rows, "7").status == rp.NOT_EVALUATED
    rows = rp.evaluate_criteria(replace(inp, scenarios=sc))
    r7 = _row(rows, "7")
    assert r7.status == rp.FAIL and "-500.00" in r7.measured    # always STRESS-d2's total
    r = _row(rp.evaluate_criteria(replace(inp, run_b=replace(inp.run_b, scenario="BASE"))), "3")
    assert r.status == rp.NOT_EVALUATED
    r = _row(rp.evaluate_criteria(replace(inp, sensitivity=replace(inp.sensitivity,
                                                                   scenario="BASE"))), "5")
    assert r.status == rp.NOT_EVALUATED
    text = rp.render_report(replace(inp, benchmarks=bm))
    assert "B2 / B3: not evaluable" in text and rp.SENTENCE_RS not in text


def test_real_report_needs_data_end_and_trial_ids():
    inp = _passing_inputs(config=None, trial_log=None)
    prov = replace(inp.provenance, synthetic=False)
    with pytest.raises(ValueError, match="data_end"):
        rp.render_report(replace(inp, provenance=replace(prov, data_end=None)))
    with pytest.raises(ValueError, match="trial-log row ids"):
        rp.render_report(replace(inp, provenance=replace(prov, trial_ids=())))
    with pytest.raises(ValueError, match="not in the trial log"):
        rp.render_report(replace(inp, provenance=replace(prov, trial_ids=("T9999x",))))
    with pytest.raises(ValueError, match="registered"):
        rp.render_report(replace(inp, provenance=replace(prov, strategy_id="SYN-01")))
    with pytest.raises(ValueError, match="dropped"):
        rp.render_report(replace(inp, provenance=replace(prov, strategy_id="MR-TWAP-01")))
    # A synthetic report may omit data_end (it is bannered as synthetic).
    rp.render_report(replace(_passing_inputs(), provenance=replace(
        _passing_inputs().provenance, data_end=None)))


def test_max_dd_pct_is_the_largest_relative_drawdown():
    tr = pd.DataFrame(dict(status=["filled"], pnl_usd=[89_000.0], r_multiple=[1.0]))
    daily = pd.DataFrame(dict(equity=[90_000.0, 200_000.0, 189_000.0]))
    s = rp.scenario_stats("BASE", tr, daily)
    assert s.max_dd_usd == pytest.approx(11_000.0)
    assert s.max_dd_pct == pytest.approx(0.10)


@pytest.mark.parametrize("ns", [[1] * 9, [1, 2, 3, 4, 5, 6, 7, 8, 8], list(range(2, 11))])
def test_fold_numbers_must_be_exactly_1_to_9(ns):
    folds = [rp.FoldStats(n, f"w{n}", 1.0, 30, 10.0) for n in ns]
    rows = rp.evaluate_criteria(_passing_inputs(folds=folds))
    assert _row(rows, "4").status == rp.NOT_EVALUATED
    assert _row(rows, "10.4 folds").status == rp.NOT_EVALUATED
    assert "need exactly 1..9" in _row(rows, "4").measured


# --------------------------------------------------------------------------------------------
# QA round 3: run A defaults, run B evidence, ties to STRESS-d2, counts, pair days, win rate
# --------------------------------------------------------------------------------------------

RUN_A_CRITS = ("10.4 trades", "10.4 folds", "1", "2", "4", "6", "7", "8")


def test_run_a_must_be_registered_defaults():
    inp = _passing_inputs()
    bad = replace(inp, provenance=replace(inp.provenance,
                                          params={"theta": 1.2, "k": 1.0, "R": 1.5}))
    rows = rp.evaluate_criteria(bad)
    for c in RUN_A_CRITS:
        r = _row(rows, c)
        assert r.status == rp.NOT_EVALUATED and "not the registered defaults" in r.measured, c
    assert rp.overall_verdict(rows) != rp.VERDICT_PASS
    # A failing run-A row on the wrong parameters is no evidence either.
    sc = [replace(s, expectancy_usd=-1.0) if s.name == "STRESS-d2" else s for s in bad.scenarios]
    assert _row(rp.evaluate_criteria(replace(bad, scenarios=sc)), "1").status == rp.NOT_EVALUATED
    text = rp.render_report(bad)
    assert "every run-A row NOT EVALUATED" in text and "**Overall verdict: PASS**" not in text
    # Same values in another key order are the defaults.
    ok = replace(inp, provenance=replace(inp.provenance, params={"R": 1.5, "k": 1.0, "theta": 1}))
    assert rp.overall_verdict(rp.evaluate_criteria(ok)) == rp.VERDICT_PASS
    # A real report on non-default parameters is refused.
    real = _passing_inputs(config=None, trial_log=None)
    real = replace(real, provenance=replace(bad.provenance, synthetic=False))
    with pytest.raises(ValueError, match="defaults fixed ex ante"):
        rp.render_report(real)


@pytest.mark.parametrize("case", ["zero", "partial", "off_grid", "heatmap_only", "trade_sum",
                                  "pnl_sum", "no_embargo", "duplicate_fold"])
def test_run_b_needs_full_on_grid_consistent_selections(case):
    inp = _passing_inputs()
    sel = list(inp.run_b.selections)
    rb = {
        "zero": _run_b([], trades=180),
        "partial": _run_b(sel[:1], trades=20, total=60.0),
        "off_grid": _run_b([replace(sel[0], params={"theta": 7.0, "k": 1.0, "R": 1.5})]
                           + sel[1:]),
        # k x R heatmap cell: on the sensitivity grid but not one-at-a-time.
        "heatmap_only": _run_b([replace(sel[0], params={"theta": 1.0, "k": 0.75, "R": 1.0})]
                               + sel[1:]),
        "trade_sum": _run_b(sel, trades=200),
        "pnl_sum": replace(_run_b(sel), total_usd=541.0),
        "no_embargo": _run_b([replace(sel[3], n_embargoed=None)] + sel[:3] + sel[4:]),
        "duplicate_fold": _run_b([replace(sel[0], fold=2)] + sel[1:]),
    }[case]
    r = _row(rp.evaluate_criteria(replace(inp, run_b=rb)), "3")
    assert r.status == rp.NOT_EVALUATED, r.measured
    assert "WARNING (criterion 3 NOT EVALUATED)" in rp.render_report(replace(inp, run_b=rb))


def test_run_b_table_shows_embargo_counts():
    text = rp.render_report(_passing_inputs())
    block = text.split("## Run B")[1].split("## Parameter sensitivity")[0]
    assert "Embargoed train trades" in block
    assert sum(ln.startswith("| ") and ln[2].isdigit() for ln in block.splitlines()) == 9
    assert "WARNING" not in block


@pytest.mark.parametrize("over", [dict(scenario="BASE"), dict(scenario=None),
                                  dict(n_obs=None), dict(n_obs=200)])
def test_bootstrap_must_be_of_stress_d2_daily_series(over):
    inp = _passing_inputs()
    r = _row(rp.evaluate_criteria(replace(inp, bootstrap=replace(inp.bootstrap, **over))), "6")
    assert r.status == rp.NOT_EVALUATED


def test_bootstrap_n_obs_needs_stress_d2_days():
    inp = _passing_inputs()
    sc = [replace(s, n_days=None) if s.name == "STRESS-d2" else s for s in inp.scenarios]
    assert _row(rp.evaluate_criteria(replace(inp, scenarios=sc)), "6").status == rp.NOT_EVALUATED


@pytest.mark.parametrize("over", [dict(sr_hat=0.9), dict(t_days=3), dict(sr_hat=0.0500001)])
def test_dsr_must_be_of_stress_d2_daily_series(over):
    inp = _passing_inputs()
    d = replace(inp.dsr, **over)
    r = _row(rp.evaluate_criteria(replace(inp, dsr=d)), "8")
    assert r.status == rp.NOT_EVALUATED
    assert "criterion 8 NOT EVALUATED" in rp.render_report(replace(inp, dsr=d))
    # Within float rounding it is the same Sharpe.
    r = _row(rp.evaluate_criteria(replace(inp, dsr=replace(inp.dsr, sr_hat=0.05 + 1e-15))), "8")
    assert r.status == rp.PASS


def test_regime_bootstrap_buckets_must_be_active_calm():
    inp = _passing_inputs()
    hmm = replace(inp.regime_bootstrap, bucket_counts={"ACTIVE/ACTIVE_UP": 120,
                                                       "ACTIVE/ACTIVE_FLAT": 80})
    assert "are not within ACTIVE/CALM" in rp.render_report(replace(inp, regime_bootstrap=hmm))
    short = replace(inp.regime_bootstrap, bucket_counts={"ACTIVE": 150, "CALM": 20})
    assert "bucket counts sum to 170" in rp.render_report(replace(inp, regime_bootstrap=short))
    # The report-only regime bootstrap never changes the verdict.
    assert rp.overall_verdict(rp.evaluate_criteria(replace(inp, regime_bootstrap=hmm))) \
        == rp.VERDICT_PASS
    # regime_labels: the ACTIVE / CALM prefix of every filled trade (skipped row excluded).
    tr = pd.concat([_trades({("US30", "BUY", "SL"): 3}, regime="ACTIVE/ACTIVE_UP"),
                    _trades({("US30", "BUY", "SL"): 2}, regime="CALM/CALM_FLAT")])
    assert list(rp.regime_labels(tr)) == ["ACTIVE"] * 3 + ["CALM"] * 2


def test_trade_counts_must_agree():
    inp = _passing_inputs()
    short = inp.trades.iloc[15:]                              # 185 filled vs 200
    rows = rp.evaluate_criteria(replace(inp, trades=short))
    for c in ("10.4 trades", "10.4 folds", "4"):
        assert _row(rows, c).status == rp.NOT_EVALUATED and "185 filled" in _row(rows, c).measured
    assert _row(rows, "1").status == rp.PASS
    rows = rp.evaluate_criteria(replace(inp, folds=_folds(trades=(30, 30, 20, 2, 30, 20, 10,
                                                                    30, 30))))
    assert _row(rows, "10.4 folds").status == rp.NOT_EVALUATED
    assert "fold trades sum to 202" in _row(rows, "10.4 trades").measured
    assert "WARNING (10.4 rows and criterion 4 NOT EVALUATED)" in rp.render_report(
        replace(inp, trades=short))


def test_outside_fold_rows_are_out_of_scope():
    inp = _passing_inputs()
    extra = inp.trades.iloc[:1].assign(fold=-1, signal_time=pd.Timestamp("2024-01-20", tz="UTC"))
    tr = pd.concat([inp.trades.assign(fold=1), extra], ignore_index=True)
    assert len(rp._filled(tr)) == 200
    assert _row(rp.evaluate_criteria(replace(inp, trades=tr)), "10.4 trades").status == rp.PASS


@pytest.mark.parametrize("col", ["signal_time", "exit_time"])
def test_holdout_trade_rows_are_refused(col):
    inp = _passing_inputs()
    tr = inp.trades.copy()
    if col not in tr.columns:
        tr[col] = tr["signal_time"]
    tr[col] = tr[col].astype(object)
    tr.loc[tr.index[-1], col] = pd.Timestamp("2026-06-01", tz="UTC")   # even a skipped row
    with pytest.raises(ValueError, match="holdout"):
        rp.render_report(replace(inp, trades=tr))


def test_fold_stats_raises_on_trade_outside_every_window():
    from backtest.feasibility import TEST_FOLDS
    tr = pd.DataFrame(dict(status=["filled"] * 4,
                           signal_time=pd.to_datetime(["2024-02-01 09:00", "2024-04-30 23:59",
                                                       "2024-05-01 00:00", "2026-05-01 00:00"],
                                                      utc=True),
                           pnl_usd=[1.0, 2.0, -5.0, 3.0], r_multiple=[0.1, 0.2, -0.5, 0.3]))
    with pytest.raises(ValueError, match="outside every test window"):
        rp.fold_stats(tr, SHARES, TEST_FOLDS)
    # The runner's OUTSIDE rows (fold -1) are out of scope, not an error.
    fs = rp.fold_stats(tr.assign(fold=[1, 1, 2, -1]), SHARES, TEST_FOLDS)
    assert sum(f.trades for f in fs) == 3


def _ses_trades() -> pd.DataFrame:
    """35 pair days with both legs filled, 1 pair day with one leg cancelled, 1 same-side day
    (not a pair day), 40 GER40 trades. On the first pair day the NAS100 leg is at 02:00 UTC of
    the next UTC date, still the same New York date."""
    rows = []
    day0 = pd.Timestamp("2024-02-05 15:00", tz="UTC")

    def add(t, key, side, status="filled", pnl=10.0):
        rows.append(dict(key=key, side=side, status=status, exit_type="SL",
                         pnl_usd=pnl if status == "filled" else np.nan,
                         r_multiple=pnl / 300 if status == "filled" else np.nan,
                         regime="ACTIVE/ACTIVE_UP", signal_time=t))
    for i in range(35):
        t = day0 + pd.Timedelta(days=i)
        add(t, "US30", "BUY", pnl=-5.0)
        add(t + (pd.Timedelta(hours=11) if i == 0 else pd.Timedelta(0)), "NAS100", "SELL",
            pnl=-5.0)
        add(t - pd.Timedelta(hours=6), "GER40", "BUY", pnl=20.0)
    t = day0 + pd.Timedelta(days=40)
    add(t, "US30", "SELL")
    add(t, "NAS100", "BUY", status="cancelled")               # issued, not filled: pair day
    t = day0 + pd.Timedelta(days=41)
    add(t, "US30", "BUY")
    add(t, "NAS100", "BUY")                                   # same side: not a pair day
    for i in range(5):
        add(day0 + pd.Timedelta(days=50 + i), "GER40", "SELL", pnl=20.0)
    return pd.DataFrame(rows)


def test_pair_days():
    tr = _ses_trades()
    assert len(rp.pair_days(tr)) == 36
    ps = rp.pair_day_summary(tr)
    assert ps["n_pair_days"] == 36 and ps["n_both_filled"] == 35
    t = ps["table"].set_index("bucket")
    assert t.loc["pair-day leg", "n"] == 71                    # 70 + the filled leg of day 40
    assert t.loc["pair-day leg", "mean_usd"] == pytest.approx((70 * -5.0 + 10.0) / 71)
    assert t.loc["single ticket", "n"] == 42                   # 40 GER40 + 2 same-side legs
    inp = _passing_inputs()
    ses = replace(inp, provenance=replace(inp.provenance, strategy_id="SES-IBX-01"), trades=tr)
    text = rp.render_report(ses)
    assert "### Pair days" in text and "Pair days: 36; with both legs filled: 35" in text
    assert "### Pair days" not in rp.render_report(inp)        # RS-MOM-01: no pair days


def test_pair_day_small_bucket_withheld():
    tr = _ses_trades()
    late = tr["key"].isin(rp.US_PAIR) & (tr["signal_time"] > pd.Timestamp("2024-02-15",
                                                                          tz="UTC"))
    t = rp.pair_day_summary(tr[~late])["table"]
    other = t[t["bucket"] == rp.OTHER].iloc[0]
    assert other["merged"] == "pair-day leg" and other["withheld"]


def test_win_rate_in_scenario_stats_and_table():
    tr = pd.DataFrame(dict(status=["filled", "filled", "filled", "skipped"],
                           pnl_usd=[100.0, -40.0, 0.0, np.nan],
                           r_multiple=[0.5, -0.2, 0.0, np.nan]))
    daily = pd.DataFrame(dict(equity=[100_100.0, 100_060.0, 100_060.0]))
    s = rp.scenario_stats("STRESS-d2", tr, daily)
    assert s.win_rate == pytest.approx(1 / 3) and s.n_days == 3      # zero P&L is not a win
    text = rp.render_report(_passing_inputs())
    assert "| Scenario | Trades | Win rate |" in text
    assert "| STRESS-d2 | 200 | 55.0% |" in text


def test_max_dd_pct_convention_differs_from_trial_log():
    """The report's Max DD % is the largest relative drawdown; backtest.runner.max_drawdown
    (the trial log) measures % at the largest $ drawdown, negative. Both pinned here."""
    from backtest.runner import max_drawdown
    eq = np.array([90_000.0, 200_000.0, 189_000.0])
    usd, pct = max_drawdown(eq, 100_000.0)
    assert usd == pytest.approx(-11_000.0) and pct == pytest.approx(-0.055)
    s = rp.scenario_stats("BASE", pd.DataFrame(dict(status=["filled"], pnl_usd=[89_000.0],
                                                    r_multiple=[1.0])),
                          pd.DataFrame(dict(equity=eq)))
    assert s.max_dd_usd == pytest.approx(11_000.0) and s.max_dd_pct == pytest.approx(0.10)
    assert "max_dd_pct is measured at the largest $ drawdown" in rp.render_report(
        _passing_inputs())
