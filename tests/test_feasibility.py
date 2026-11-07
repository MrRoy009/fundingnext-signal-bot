"""COMMON 10.1 feasibility gate (backtest/feasibility.py). Synthetic data only.

What is pinned here:
  - the 9 COMMON 10.2 test windows, the 200-candidate rule and the 6.8 non-functional bounds;
  - per-fold scans give exactly the candidates of one scan over the whole span (lockout replay
    from local midnight; no state crosses a date), each in its own fold;
  - the gate diagnostic sees the gate the strategy saw (same blocks as the Selection reasons,
    same share as regime.diagnostics.gate_block_share);
  - counts at or before a cut do not change when bars after the cut change (look-ahead);
  - the data share per fold follows the cash-open bars, and an empty fold has share 0;
  - the module never reaches P&L code and never writes the trial log.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from datetime import date, time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import backtest.feasibility as fz
from backtest.feasibility import (DROPPED, MIN_CANDIDATES, NO_MODEL, PROCEED, TEST_FOLDS, Fold,
                                  count_tables, fold_data_shares, gate_by_refit, gate_cycles,
                                  gate_flag, gate_summary, models_without_tag,
                                  refit_diagnostics_in_span, run_feasibility, scan_folds, verdict,
                                  write_selection_log)
from core.calendar import exchange_for, is_calendar_trading_date, local_to_utc
from regime.diagnostics import gate_block_share
from regime.service import latest_refit_at_or_before
from strategies.mr_twap_01 import MrTwap01, models_without_calm
from strategies.rs_mom_01 import RsMom01
from strategies.ses_ibx_01 import SesIbx01
from tests.strategy_testkit import (assert_candidates_equal, build_services, make_market,
                                    proxy_regime_tables)

F1 = Fold(1, date(2024, 2, 5), date(2024, 3, 3))
F2 = Fold(2, date(2024, 3, 4), date(2024, 4, 12))
FOLDS = (F1, F2)


@pytest.fixture(scope="module")
def market():
    return make_market()


@pytest.fixture
def services(cfg, market):
    return build_services(cfg, market)


def _strategies(cfg):
    return [RsMom01(cfg), SesIbx01(cfg), MrTwap01(cfg)]


# --- constants -------------------------------------------------------------------------------

def test_test_folds_are_common_10_2(cfg):
    assert len(TEST_FOLDS) == 9
    assert [f.label for f in TEST_FOLDS] == [
        "2024-02-01..2024-04-30", "2024-05-01..2024-07-31", "2024-08-01..2024-10-31",
        "2024-11-01..2025-01-31", "2025-02-01..2025-04-30", "2025-05-01..2025-07-31",
        "2025-08-01..2025-10-31", "2025-11-01..2026-01-31", "2026-02-01..2026-04-30"]
    for a, b in zip(TEST_FOLDS, TEST_FOLDS[1:]):
        assert a.end == b.start                       # contiguous, no day lost or doubled
    assert TEST_FOLDS[-1].end <= pd.Timestamp(cfg["holdout"]["start"], tz="UTC")
    assert TEST_FOLDS[0].start == pd.Timestamp("2024-02-01", tz="UTC")
    assert F1.contains(pd.Timestamp("2024-03-03 23:45", tz="UTC"))
    assert not F1.contains(F1.end)
    assert fz.fold_of(pd.Timestamp("2024-11-15 15:00", tz="UTC")) == 4
    assert fz.fold_of(pd.Timestamp("2026-05-01 00:00", tz="UTC")) is None


def test_verdict_threshold_is_200_unscaled():
    assert MIN_CANDIDATES == 200
    assert verdict(199) == DROPPED
    assert verdict(200) == PROCEED
    assert verdict(0) == DROPPED


@pytest.mark.parametrize("share,flag", [(0.0, "NON-FUNCTIONAL (< 5%)"),
                                        (0.0499, "NON-FUNCTIONAL (< 5%)"),
                                        (0.05, "functional"), (0.5, "functional"),
                                        (0.95, "functional"),
                                        (0.9501, "NON-FUNCTIONAL (> 95%)"),
                                        (1.0, "NON-FUNCTIONAL (> 95%)"),
                                        (float("nan"), "n/a")])
def test_gate_flag_bounds(share, flag):
    assert gate_flag(share) == flag


# --- scans per fold --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [RsMom01, SesIbx01, MrTwap01])
def test_fold_scans_equal_one_scan(cfg, services, cls):
    svc, tabs = services
    strat = cls(cfg)
    sels = []
    full = strat.scan(svc, tabs, F1.start, F2.end, on_selection=sels.append)
    sc = scan_folds(strat, svc, tabs, FOLDS)
    assert_candidates_equal(sc.candidates, full, f"{strat.id} folds vs one scan")
    assert sc.cand_fold == [1 if c.signal_time < F1.end else 2 for c in full]
    assert sc.n_group_cycles == len(sels) == len(sc.selection_lines)
    assert len(sc.cycles) == sum(len(s.members) for s in sels)
    assert sc.n_incomplete_group_cycles == sum(bool(s.incomplete) for s in sels)
    assert len(sc.excluded) == sum(len(s.excluded) for s in sels)
    assert [line for _, line in sc.selection_lines] == [s.log_line() for s in sels]


def test_counts(cfg, services):
    svc, tabs = services
    sc = scan_folds(RsMom01(cfg), svc, tabs, FOLDS)
    keys = ["US30", "NAS100", "GER40"]
    t = count_tables(sc, keys)
    bk, bf = t["by_key"], t["by_fold"]
    n = len(sc.candidates)
    assert n > 0
    assert int(bk.loc["ALL", "total"]) == n
    for k in keys:
        for side in ("BUY", "SELL"):
            assert bk.loc[k, side] == sum(c.key == k and c.side == side for c in sc.candidates)
    assert int(bf["candidates"].sum()) == n
    assert (bf[keys].sum(axis=1) == bf["candidates"]).all()
    assert (bf[["BUY", "SELL"]].sum(axis=1) == bf["candidates"]).all()
    # no price field reaches the count frame
    assert set(sc.candidate_frame().columns) == {"key", "side", "fold", "signal_time"}


def test_default_parameters_only(cfg, services):
    svc, tabs = services
    with pytest.raises(ValueError, match="DEFAULT parameters"):
        scan_folds(RsMom01(cfg, {"theta": 0.8}), svc, tabs, FOLDS)


# --- gate diagnostics ------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [RsMom01, SesIbx01, MrTwap01])
def test_gate_cycles_match_the_strategy_gate(cfg, services, cls):
    svc, tabs = services
    strat = cls(cfg)
    sels = []
    strat.scan(svc, tabs, F1.start, F2.end, on_selection=sels.append)
    sc = scan_folds(strat, svc, tabs, FOLDS)
    thr = float(cfg["regime"]["gate_threshold"])
    gc = gate_cycles(sc, tabs, strat.regime_gate, thr)
    assert len(gc) == len(sc.cycles)
    look = {(r.T, r.key): r for r in gc.itertuples()}
    checked = 0
    for s in sels:
        if s.incomplete:
            continue
        for k in s.members:
            r = look[(s.T, k)]
            reason = s.reasons.get(k, "")
            assert (not r.gate_pass) == reason.startswith("regime"), (s.log_line(), r)
            assert r.regime_ok == (not reason.startswith("regime unavailable"))
            checked += 1
    assert checked > 50
    blocked_any = (~gc["gate_pass"]).any() and gc["gate_pass"].any()
    assert blocked_any                                   # the synthetic gate does both
    summ = gate_summary(gc, "key")
    for k in ("US30", "NAS100", "GER40"):
        times = pd.DatetimeIndex(gc.loc[gc["key"] == k, "T"])
        row = summ[summ["group"] == k].iloc[0]
        assert row["block_share"] == pytest.approx(
            gate_block_share(tabs[k], times, strat.regime_gate))
        assert row["cycles"] == len(times)
        assert row["blocked"] == row["unavailable"] + row["blocked_by_vt"]
    all_row = summ[summ["group"] == "ALL"].iloc[0]
    assert all_row["cycles"] == len(gc)
    assert all_row["block_share"] == pytest.approx((~gc["gate_pass"]).mean())
    assert all_row["block_share_ok"] == pytest.approx(
        1 - gc.loc[gc["regime_ok"], "gate_pass"].mean())
    assert all_row["flag"] == gate_flag(all_row["block_share"])


@dataclass
class _FitTimeTable:
    """Stand-in table that also reports the fit time of the (pretend) monthly model."""
    inner: object
    threshold: float = 0.0

    def at_many(self, times, live_rule: bool = True):
        out = self.inner.at_many(times, live_rule)
        out["fit_time"] = [latest_refit_at_or_before(t) for t in out["time_utc"]]
        return out


def test_gate_by_refit(cfg, market):
    svc, tabs = build_services(cfg, market)
    wrapped = {k: _FitTimeTable(v, v.threshold) for k, v in tabs.items()}
    strat = RsMom01(cfg)
    sc = scan_folds(strat, svc, wrapped, FOLDS)
    gc = gate_cycles(sc, wrapped, "ACTIVE", 0.0)
    br = gate_by_refit(gc)
    assert int(br["cycles"].sum()) == len(gc)
    assert set(br["fit_time"]) - {NO_MODEL}             # real fit times present
    for k in ("US30", "NAS100", "GER40"):
        part = br[br["key"] == k]
        assert int(part["blocked"].sum()) == int((~gc.loc[gc["key"] == k, "gate_pass"]).sum())
    # stand-in without fit_time: everything is "no model"
    plain = gate_by_refit(gate_cycles(sc, tabs, "ACTIVE", 0.0))
    assert set(plain["fit_time"]) == {NO_MODEL}


def test_no_lookahead_in_fold_counts(cfg, market):
    """Bars after the end of fold 1 cannot change fold 1's candidates or gate cycles."""
    cut = F1.end
    svc, tabs = build_services(cfg, market)
    pert = market.perturbed_after(cut, seed=3)
    svc2, tabs2 = build_services(cfg, pert)
    for strat in _strategies(cfg):
        a = scan_folds(strat, svc, tabs, (F1,))
        b = scan_folds(strat, svc2, tabs2, (F1,))
        assert_candidates_equal(a.candidates, b.candidates, f"{strat.id} fold 1")
        ga = gate_cycles(a, tabs, strat.regime_gate, 0.0)
        gb = gate_cycles(b, tabs2, strat.regime_gate, 0.0)
        pd.testing.assert_frame_equal(ga, gb)
        # ...while fold 2 does change (the test is not vacuous)
    a2 = scan_folds(RsMom01(cfg), svc, tabs, (F2,))
    b2 = scan_folds(RsMom01(cfg), svc2, tabs2, (F2,))
    assert [c.signal_time for c in a2.candidates] != [c.signal_time for c in b2.candidates]


# --- data share ------------------------------------------------------------------------------

def test_fold_data_shares(cfg, market):
    svc, _ = build_services(cfg, market)
    empty = Fold(3, date(2024, 6, 3), date(2024, 6, 28))   # after the synthetic history
    df = fold_data_shares(cfg, svc, (F1, F2, empty)).set_index("fold")
    for f in (F1, F2):
        cal_any, data_any = set(), set()
        for k in ("US30", "NAS100", "GER40"):
            ex = exchange_for(cfg, k)
            m15 = market.m15[k]
            opens = set(pd.DatetimeIndex(m15["time_utc"]).asi8)
            d = f.first
            cal = have = 0
            while d <= f.last:
                if is_calendar_trading_date(ex.calendar, d):
                    cal += 1
                    cal_any.add(d)
                    if local_to_utc(d, ex.cash_open, ex.tz).value in opens:
                        have += 1
                        data_any.add(d)
                d = d + pd.Timedelta(days=1).to_pytimedelta()
            assert df.at[f.n, f"share_{k}"] == pytest.approx(have / cal)
        assert df.at[f.n, "trading_days"] == len(cal_any)
        assert df.at[f.n, "days_with_data"] == len(data_any)
        assert df.at[f.n, "data_share"] == pytest.approx(len(data_any) / len(cal_any))
    # the GER40 hole (2024-02-26..03-05) and the missing US30 cash-open bar (2024-03-05) show
    assert df.at[1, "share_GER40"] < 1.0
    assert df.at[2, "share_US30"] < 1.0
    assert df.at[3, "data_share"] == 0.0
    assert df.at[3, "trading_days"] > 0


# --- models without a gate-tagged state ------------------------------------------------------

def test_models_without_tag_matches_mr_twap_helper():
    t0 = pd.Timestamp("2024-01-06 12:00", tz="UTC")
    mods = [SimpleNamespace(model_id=f"m{i}", vol_tags=tags, fit_time=t0 + pd.Timedelta(days=28 * i))
            for i, tags in enumerate([("CALM", "ACTIVE"), ("ACTIVE", "ACTIVE"),
                                      ("CALM", "CALM", "CALM"), ("ACTIVE",)])]
    ivs = [SimpleNamespace(start=m.fit_time, end=m.fit_time + pd.Timedelta(days=28), model=m,
                           fallback=False) for m in mods]
    ivs.insert(1, SimpleNamespace(start=ivs[0].end, end=ivs[0].end, model=None, fallback=False))
    table = SimpleNamespace(intervals=ivs, records=[SimpleNamespace(model=m) for m in mods])
    assert set(models_without_tag(table, "CALM")) == models_without_calm(table) == {"m1", "m3"}
    assert models_without_tag(table, "ACTIVE") == ["m2"]
    # span restriction: only m0 and m1 overlap the first 40 days
    assert models_without_tag(table, "CALM", t0, t0 + pd.Timedelta(days=40)) == ["m1"]
    assert models_without_tag(None, "CALM") == []


@dataclass
class _IvTable:
    intervals: list


def test_refit_diagnostics_in_span_keeps_prior_model_for_churn(monkeypatch):
    import regime.diagnostics as rd
    seen = {}

    def fake(service, table, stability_seeds=None):
        seen["starts"] = [iv.start for iv in table.intervals]
        seen["seeds"] = stability_seeds
        return pd.DataFrame({"fit_time_utc": [iv.start for iv in table.intervals]})

    monkeypatch.setattr(rd, "refit_diagnostics", fake)
    t = [pd.Timestamp(s, tz="UTC") for s in ("2023-12-02 12:00", "2024-01-06 12:00",
                                             "2024-02-03 12:00", "2024-03-02 12:00",
                                             "2024-04-06 12:00")]

    def iv(a, b, fallback=False, model=True):
        m = SimpleNamespace(fit_time=a) if model else None
        return SimpleNamespace(start=a, end=b, model=m, fallback=fallback)

    ivs = [iv(t[0], t[1]), iv(t[1], t[2]), iv(t[2], t[3]),
           iv(t[3], t[4], model=False),                       # no model
           iv(t[4], None)]
    out = refit_diagnostics_in_span(None, _IvTable(ivs), pd.Timestamp("2024-02-01", tz="UTC"),
                                    pd.Timestamp("2024-04-01", tz="UTC"), ())
    # the model serving across the span start (fit 2024-01-06) is in the span; 2023-12-02 is
    # the prior model run for churn and dropped from the output; 2024-04-06 is after the span
    assert seen["starts"] == [t[0], t[1], t[2]]
    assert seen["seeds"] == ()
    assert list(out["fit_time_utc"]) == [t[1], t[2]]


# --- whole report ----------------------------------------------------------------------------

def test_run_feasibility_report_and_selection_log(cfg, market, tmp_path):
    svc, tabs = build_services(cfg, market)
    rep = run_feasibility(cfg, svc, tabs, folds=FOLDS, refit_diag=False,
                          provenance={"Data": "synthetic"})
    assert [s.strategy_id for s in rep.strategies] == ["RS-MOM-01", "SES-IBX-01", "MR-TWAP-01"]
    for s in rep.strategies:
        assert s.total == len(s.scan.candidates)
        assert s.verdict == verdict(s.total)
        assert s.gate in ("ACTIVE", "CALM")
        assert set(s.models_without_gate_state) == {"US30", "NAS100", "GER40"}
    md = rep.to_markdown()
    assert md.startswith("# Feasibility gate (COMMON 10.1)")
    for s in rep.strategies:
        assert f"## {s.strategy_id}" in md
        assert f"**{s.total} candidates: {s.verdict}.**" in md
    assert "Data: synthetic" in md
    assert "trial_log.csv" in md
    assert "Per-index, per-refit regime diagnostics" in md
    p = tmp_path / "sel.txt.gz"
    n = write_selection_log(rep, p)
    assert n == sum(s.scan.n_group_cycles for s in rep.strategies)
    with gzip.open(p, "rt", encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert len(lines) == n
    assert lines[0].startswith("RS-MOM-01\t1\tRS-MOM-01 ")
    first = p.read_bytes()
    write_selection_log(rep, p)                 # deterministic bytes: hash is reproducible
    assert p.read_bytes() == first


def test_module_has_no_outcome_code():
    """The gate runs before any P&L code: it must not import the fill / account / results
    modules or write the trial log."""
    src = open(fz.__file__, encoding="utf-8").read()
    for mod in ("backtest.fills", "backtest.engine", "backtest.account", "backtest.results",
                "backtest.costs", "backtest.market"):
        assert mod not in src
    assert "trial_log.csv\"" not in src and "open(\"docs/trial_log" not in src
    assert "final_holdout=True" not in src
