"""SES-IBX-01 (docs/specs/SES-IBX-01.md): every rule on synthetic data, look-ahead perturbation,
live generate() == batch scan(), DST edges of its decision times, EXPIRY and time exits.

Synthetic data only (tests.strategy_testkit); never real bars. Levels are re-derived here from
the raw M15 bars, not from the feature service, so a wrong IB in either place shows up.
"""
from __future__ import annotations

import copy
import dataclasses
import math
import re
from datetime import date, time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.calendar import exchange_for, local_to_utc
from core.config import ROOT, load_config
from risk.sizing import size_position
from signals.engine import prepare_candidate
from strategies.base import STOP_ENTRY, StrategyContext, time_exit_utc
from strategies.context import (CycleView, INVALID_DATE, LockoutBook, decision_times,
                                window_schedule)
from strategies.selection import BUY, SELL
from strategies.ses_ibx_01 import (DATR_NAN, EXPIRY_LOCAL, PARAMS, STRATEGY_ID,
                                   TRIGGER_BUFFER_FRAC, SesIbx01, decision_clock, expiry_utc,
                                   pair_days, restricted_overlap, w_datr_text)

from tests.strategy_testkit import (ProxyRegimeTable, assert_candidates_equal,
                                    assert_generate_matches_scan, assert_no_lookahead,
                                    build_services, make_market, sample_times, ts)

M15 = pd.Timedelta(minutes=15)


@pytest.fixture(scope="module")
def mcfg():
    return load_config(ROOT / "config.yaml")


@pytest.fixture(scope="module")
def market():
    return make_market()


@pytest.fixture(scope="module")
def services(mcfg, market):
    return build_services(mcfg, market)


@pytest.fixture(scope="module")
def default_run(mcfg, market, services):
    strat = SesIbx01(mcfg)
    cycles = list(strat.cycles(*services, market.scan_start, market.end))
    return strat, cycles, [c for _, cs in cycles for c in cs]


def local(T, key, cfg):
    return ts(T).tz_convert(exchange_for(cfg, key).tz)


def raw_ib(market, cfg, key, d: date, ib_minutes: int):
    """IB straight from the raw bars: the ib_minutes/15 bars from the local cash open."""
    ex = exchange_for(cfg, key)
    b = market.m15[key]
    lt = b["time_utc"].dt.tz_convert(ex.tz)
    start = local_to_utc(d, ex.cash_open, ex.tz)
    want = [start + k * M15 for k in range(ib_minutes // 15)]
    rows = b[b["time_utc"].isin(want)]
    assert (lt[rows.index].dt.date == d).all()
    if len(rows) != len(want):
        return None
    return float(rows["high"].max()), float(rows["low"].min())


# ---------------------------------------------------------------------------------------------
# Parameters (spec Parameters table)
# ---------------------------------------------------------------------------------------------

def test_parameters_and_grid():
    assert dict(PARAMS.defaults) == {"ib_minutes": 60, "tp_ib_mult": 1.0, "max_ib_datr": 0.6}
    assert PARAMS.grid["ib_minutes"] == (30, 45, 60, 75, 90)
    assert PARAMS.grid["tp_ib_mult"] == (0.5, 0.75, 1.0, 1.25, 1.5)
    assert PARAMS.grid["max_ib_datr"] == (0.5, 0.55, 0.6, 0.65, 0.7)
    assert PARAMS.heatmap == ("ib_minutes", "tp_ib_mult")
    assert len(PARAMS.one_at_a_time()) == 13
    assert len(PARAMS.configurations()) == 29          # 13 + 16 heatmap cells
    assert TRIGGER_BUFFER_FRAC == 0.05
    assert EXPIRY_LOCAL == {"NYSE": time(12, 0), "XETRA": time(11, 30)}


def test_off_grid_values_refused(mcfg):
    with pytest.raises(ValueError, match="sensitivity grid"):
        SesIbx01(mcfg, {"max_ib_datr": 0.8})
    assert SesIbx01(mcfg, {"max_ib_datr": 0.8}, off_grid=True).params["max_ib_datr"] == 0.8
    strat = SesIbx01(mcfg, {"tp_ib_mult": 1.25})
    assert strat.id == STRATEGY_ID and strat.regime_gate == "ACTIVE" and strat.rank == "rs_4h"


# ---------------------------------------------------------------------------------------------
# Decision times, EXPIRY and time exit across DST
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ib", PARAMS.grid["ib_minutes"])
def test_decision_clock_on_grid(mcfg, ib):
    us, ger = decision_clock(mcfg, "US30", ib), decision_clock(mcfg, "GER40", ib)
    assert (us.hour * 60 + us.minute) == 9 * 60 + 30 + ib
    assert (ger.hour * 60 + ger.minute) == 9 * 60 + ib
    assert time(10, 0) <= us <= time(11, 0) and time(9, 30) <= ger <= time(10, 30)
    wins = SesIbx01(mcfg, {"ib_minutes": ib}).windows({"ib_minutes": ib})
    assert set(wins) == {"US30", "NAS100", "GER40"}             # US500 never
    assert wins["NAS100"].start == wins["NAS100"].end == us
    assert wins["GER40"].start == wins["GER40"].end == ger


def test_decision_clock_errors(mcfg):
    with pytest.raises(ValueError, match="multiple of 15"):
        decision_clock(mcfg, "US30", 20)
    strat = SesIbx01(mcfg, {"ib_minutes": 150}, off_grid=True)     # US T = 12:00 = EXPIRY
    with pytest.raises(ValueError, match="not before EXPIRY"):
        strat.windows(strat.params)


@pytest.mark.parametrize("key,day,ib,utc", [
    ("US30", "2024-03-08", 60, "15:30"),       # EST
    ("US30", "2024-03-11", 60, "14:30"),       # EDT, Berlin still CET (mismatch week)
    ("NAS100", "2024-11-01", 60, "14:30"),     # EDT, Berlin already CET (mismatch week)
    ("NAS100", "2024-11-04", 60, "15:30"),     # EST
    ("GER40", "2024-03-28", 60, "09:00"),      # CET
    ("GER40", "2024-04-02", 60, "08:00"),      # CEST
    ("GER40", "2024-10-25", 60, "08:00"),      # CEST
    ("GER40", "2024-10-28", 60, "09:00"),      # CET
    ("US30", "2024-03-11", 30, "14:00"),
    ("US30", "2024-01-10", 90, "16:00"),
    ("GER40", "2024-07-10", 90, "08:30"),
    ("GER40", "2024-01-10", 30, "08:30"),
])
def test_decision_times_follow_local_dst(mcfg, key, day, ib, utc):
    strat = SesIbx01(mcfg, {"ib_minutes": ib})
    win = strat.windows(strat.params)[key]
    got = decision_times(mcfg, key, win, ts(day), ts(day) + pd.Timedelta(days=1))
    assert list(got) == [ts(f"{day} {utc}")]


@pytest.mark.parametrize("key,day", [
    ("US30", "2024-07-04"), ("US30", "2024-07-03"),             # NYSE holiday, early close
    ("NAS100", "2024-11-29"), ("GER40", "2024-12-24"),          # early closes
    ("GER40", "2024-04-01"), ("US30", "2024-03-29"),            # Easter Monday, Good Friday
    ("GER40", "2024-03-30"),                                    # Saturday
])
def test_no_decision_time_off_calendar(mcfg, key, day):
    strat = SesIbx01(mcfg)
    win = strat.windows(strat.params)[key]
    assert len(decision_times(mcfg, key, win, ts(day), ts(day) + pd.Timedelta(days=1))) == 0


@pytest.mark.parametrize("ib", PARAMS.grid["ib_minutes"])
def test_decision_times_inside_config_sessions(mcfg, ib):
    """Spec: across the grid, T lies inside config sessions_utc in both DST states."""
    strat = SesIbx01(mcfg, {"ib_minutes": ib})
    sched = window_schedule(mcfg, strat.windows(strat.params), ts("2024-01-01"),
                            ts("2025-01-01"))
    for key, grp in sched.groupby("key"):
        (a, b), = mcfg["sessions_utc"][key]
        clocks = {t.strftime("%H:%M") for t in grp["T"]}
        assert len(clocks) == 2                                 # one per DST state
        assert all(a <= c < b for c in clocks), (key, clocks)
    # One G(T) per T: the US and GER40 decision times never coincide on the grid.
    assert not sched.duplicated(["T", "key"]).any()
    assert sched.groupby("T")["group"].nunique().max() == 1


@pytest.mark.parametrize("key,T,expiry,exit_", [
    ("US30", "2024-03-08 15:30", "2024-03-08 17:00", "2024-03-08 20:45"),
    ("US30", "2024-03-11 14:30", "2024-03-11 16:00", "2024-03-11 19:45"),
    ("NAS100", "2024-11-01 14:30", "2024-11-01 16:00", "2024-11-01 19:45"),
    ("NAS100", "2024-11-04 15:30", "2024-11-04 17:00", "2024-11-04 20:45"),
    ("GER40", "2024-03-28 09:00", "2024-03-28 10:30", "2024-03-28 16:15"),
    ("GER40", "2024-04-02 08:00", "2024-04-02 09:30", "2024-04-02 15:15"),
    ("GER40", "2024-10-28 09:00", "2024-10-28 10:30", "2024-10-28 16:15"),
])
def test_expiry_and_time_exit_dst(mcfg, key, T, expiry, exit_):
    assert expiry_utc(mcfg, key, ts(T)) == ts(expiry)
    assert time_exit_utc(mcfg, key, ts(T)) == ts(exit_)


def test_expiry_after_t_on_whole_grid(mcfg):
    for ib in PARAMS.grid["ib_minutes"]:
        strat = SesIbx01(mcfg, {"ib_minutes": ib})
        sched = window_schedule(mcfg, strat.windows(strat.params), ts("2024-01-01"),
                                ts("2025-01-01"))
        gap = [expiry_utc(mcfg, k, t) - t for t, k in zip(sched["T"], sched["key"])]
        assert min(gap) >= pd.Timedelta(minutes=60)             # spec: 60 min at ib 90


# ---------------------------------------------------------------------------------------------
# Candidate fields and levels (Entry 3-5, Stop, Exit, Sizing)
# ---------------------------------------------------------------------------------------------

REASON = re.compile(
    r"^SES-IBX-01 (BUY|SELL) STOP (\w+) at (-?[\d.]+) \(IB ([\d.]+)-([\d.]+), W ([\d.]+) = "
    r"([\d.]+) DATR\), RS_4h ([+-][\d.]+) (top|bottom|sole) of \{([\w, ]+)\}, regime ACTIVE\. "
    r"Order expires (\d\d:\d\d) UTC \((12:00 New York|11:30 Berlin)\)\. MANUAL TIME EXIT "
    r"(\d\d:\d\d) UTC \((15:45 New York|17:15 Berlin)\) if filled and still open\.$")

PARAM_CASES = [{}, {"ib_minutes": 30, "tp_ib_mult": 0.5},
               {"ib_minutes": 90, "tp_ib_mult": 1.5, "max_ib_datr": 0.7}]


@pytest.mark.parametrize("params", PARAM_CASES)
def test_candidates_follow_every_rule(mcfg, market, services, params):
    svc, tabs = services
    strat = SesIbx01(mcfg, params)
    p = strat.params
    cycles = list(strat.cycles(svc, tabs, market.scan_start, market.end))
    cands = [c for _, cs in cycles for c in cs]
    assert len(cands) >= 15, "synthetic market should produce candidates"
    assert {c.key for c in cands} == {"US30", "NAS100", "GER40"}
    assert {c.side for c in cands} == {BUY, SELL}
    sels = {(s.T, k): s for s, _ in cycles for k in s.members}
    seen_days = set()
    for c in cands:
        T, key, ex = ts(c.signal_time), c.key, exchange_for(mcfg, c.key)
        lt = local(T, key, mcfg)
        d = lt.date()
        assert lt.time() == decision_clock(mcfg, key, p["ib_minutes"])
        assert (key, d) not in seen_days                            # one per index per day
        seen_days.add((key, d))
        hi, lo = raw_ib(market, mcfg, key, d, p["ib_minutes"])
        fx = svc.get(key)
        datr = fx.datr(d)
        w = hi - lo
        assert w <= p["max_ib_datr"] * datr                          # width filter
        b = 0.05 * w
        mid = (hi + lo) / 2
        if c.side == BUY:
            trig, tp = hi + b, hi + p["tp_ib_mult"] * w
        else:
            trig, tp = lo - b, lo - p["tp_ib_mult"] * w
        assert c.order_type == STOP_ENTRY and c.trigger == pytest.approx(trig)
        assert c.entry == pytest.approx(trig) and c.stop == pytest.approx(mid)
        assert c.take_profit == pytest.approx(tp)
        assert c.d == pytest.approx(0.55 * w)
        sg = 1 if c.side == BUY else -1
        assert c.chase_limit == pytest.approx(trig + sg * 0.20 * 0.55 * w)
        assert c.sizing_entry == pytest.approx(c.chase_limit) and c.skip_level is None
        assert c.expires_utc == local_to_utc(d, EXPIRY_LOCAL[ex.calendar], ex.tz)
        assert c.time_exit_utc == time_exit_utc(mcfg, key, T)
        assert c.expires_utc > T and c.time_exit_utc > c.expires_utc
        pos = fx.m15_pos(T)
        assert c.atr == pytest.approx(float(fx.m15["atr14"].iloc[pos]))
        score = float(fx.m15["rs_4h"].iloc[pos])
        assert (score > 0) if c.side == BUY else (score < 0)
        assert c.rank_score == round(score, 2)
        r = tabs[key].at_many([T]).iloc[0]
        assert r["status"] == "OK" and r["v_t"] >= 0                  # ACTIVE gate
        assert c.regime == f"{r['tag']}/{r['label']}" and c.regime.startswith("ACTIVE/")
        assert c.regime_prob == pytest.approx(r["regime_prob"])
        s = sels[(T, key)]
        assert s.allows(key, c.side)
        m = REASON.match(c.reason)
        assert m, c.reason
        assert m.group(1) == c.side and m.group(2) == key
        assert float(m.group(3)) == pytest.approx(trig, abs=0.006)
        assert float(m.group(7)) == pytest.approx(w / datr, abs=0.006)
        assert m.group(9) == ("sole" if len(s.eligible) == 1 else
                              "top" if c.side == BUY else "bottom")
        assert "{" + m.group(10) + "}" == s.set_text()
        assert m.group(11) == f"{c.expires_utc:%H:%M}" and m.group(13) == f"{c.time_exit_utc:%H:%M}"
        assert prepare_candidate(c, mcfg, T) == c                     # engine-ready, idempotent


def test_selection_and_side_rules(mcfg, services, default_run):
    """Per cycle: E(T) = complete members passing ACTIVE and the width filter; candidates are
    exactly LONG-SELECTED with RS_4h > 0 (BUY STOP) and SHORT-SELECTED with RS_4h < 0
    (SELL STOP); the next index is never promoted."""
    svc, tabs = services
    _, cycles, _ = default_run
    n_pair = n_complete = 0
    for s, cands in cycles:
        got = {(c.key, c.side) for c in cands}
        if s.incomplete:
            assert not got
            continue
        n_complete += 1
        expected_e = []
        for k in s.members:
            fx = svc.get(k)
            d = local(s.T, k, mcfg).date()
            ib = fx.ib(d, 60, s.T)
            r = tabs[k].at_many([s.T]).iloc[0]
            if r["status"] == "OK" and r["v_t"] >= 0 and ib.w <= 0.6 * fx.datr(d):
                expected_e.append(k)
            elif r["status"] == "OK" and r["v_t"] >= 0:
                assert s.reasons[k].startswith("IB width W")
        assert list(s.eligible) == expected_e
        want = set()
        if s.long and s.scores[s.long] > 0:
            want.add((s.long, BUY))
        if s.short and s.scores[s.short] < 0:
            want.add((s.short, SELL))
        assert got == want
        if len(s.eligible) == 2:
            assert s.long == max(s.eligible, key=lambda k: s.scores[k])
            assert s.short == min(s.eligible, key=lambda k: s.scores[k])
        n_pair += len(want) == 2
    assert n_complete > 40 and n_pair >= 1


def test_pair_days(default_run):
    _, _, cands = default_run
    pairs = pair_days(cands)
    assert pairs
    for T, (buy, sell) in pairs:
        at = {c.side: c for c in cands if ts(c.signal_time) == T}
        assert at[BUY].key == buy and at[SELL].key == sell
        assert {buy, sell} == {"US30", "NAS100"}
        assert " top of {US30, NAS100}" in at[BUY].reason
        assert " bottom of {US30, NAS100}" in at[SELL].reason
        assert at[BUY].rank_score > 0 > at[SELL].rank_score


def test_ger40_ranked_alone_by_own_sign(default_run):
    _, cycles, cands = default_run
    for s, _ in cycles:
        if s.excluded:
            # A US index outside G(T) (invalid date): the other is ranked alone (4.2.1 + 1.6).
            assert set(s.members) | set(s.excluded) == {"US30", "NAS100"}
            assert len(s.members) == 1
        else:
            assert s.members in (("US30", "NAS100"), ("GER40",))
    ger = [c for c in cands if c.key == "GER40"]
    assert ger and all("sole of {GER40}" in c.reason for c in ger)
    assert all((c.side == BUY) == (c.rank_score > 0) for c in ger)


def test_no_ger40_cycle_inside_its_data_hole(mcfg, default_run):
    """GER40's hole (2024-02-26..03-05): no cash-open bar, so no valid date and no cycle."""
    _, cycles, _ = default_run
    ger = [s.T for s, _ in cycles if "GER40" in s.members or "GER40" in s.excluded]
    assert ger
    assert not [T for T in ger if ts("2024-02-26") <= T < ts("2024-03-05")]


# ---------------------------------------------------------------------------------------------
# Data completeness (COMMON 4.2.2, SPEC NOTE 1)
# ---------------------------------------------------------------------------------------------

def _sel_at(cycles, T):
    out = [s for s, _ in cycles if s.T == ts(T)]
    assert len(out) == 1
    return out[0]


def test_missing_ib_bar_blocks_us_group(mcfg):
    T = "2024-02-27 15:30"                     # 10:30 New York (EST)
    clean = make_market(drops={}, holes={})
    gap = make_market(drops={"NAS100": ["2024-02-27 14:45"]}, holes={})   # 09:45 NY IB bar
    strat = SesIbx01(mcfg)
    a = list(strat.cycles(*build_services(mcfg, clean), clean.scan_start, clean.end))
    b = list(strat.cycles(*build_services(mcfg, gap), gap.scan_start, gap.end))
    assert _sel_at(a, T).incomplete is None
    s = _sel_at(b, T)
    # A missing IB bar is also inside RS_4h's 16-bar lookback, which the driver checks first;
    # either way the whole US group has no candidate (COMMON 4.2.2).
    assert s.incomplete.startswith("incomplete group: NAS100: ")
    assert "US30" not in s.incomplete
    assert not [c for x, cs in b for c in cs if x.T == ts(T)]
    # The strategy's own hook names the IB.
    svc, _ = build_services(mcfg, gap)
    strat_view = CycleView(ts(T), mcfg, strat.params, {"NAS100": svc.get("NAS100")},
                           {"NAS100": svc.get("NAS100").m15_pos(ts(T))}, {},
                           {"NAS100": date(2024, 2, 27)}, LockoutBook(strat.id, {}))
    assert strat.incomplete_cause(strat_view, "NAS100") == "IB incomplete: missing IB bar"
    # The GER40 group the same day is unaffected.
    ga = [cs for x, cs in a if x.T == ts("2024-02-27 09:00")]
    gb = [cs for x, cs in b if x.T == ts("2024-02-27 09:00")]
    assert_candidates_equal(ga[0], gb[0])


def test_missing_cash_open_bar_excludes_index(mcfg, default_run, services):
    """US30's 09:30 New York bar is dropped on 2024-03-05: not a valid trading date for US30, so
    US30 is outside G(T) and NAS100 is ranked alone by its own sign (COMMON 4.2.1 + 1.6, 4.2.5)."""
    svc, tabs = services
    _, cycles, _ = default_run
    s = _sel_at(cycles, "2024-03-05 15:30")
    assert s.members == ("NAS100",) and s.incomplete is None
    assert dict(s.excluded) == {"US30": INVALID_DATE}
    assert f"outside G(T): US30: {INVALID_DATE}" in s.log_line()
    got = [c for x, cs in cycles for c in cs if x.T == s.T]
    assert not [c for c in got if c.key == "US30"]
    score = float(svc.get("NAS100").m15["rs_4h"].iloc[svc.get("NAS100").m15_pos(s.T)])
    if s.eligible == ("NAS100",):
        assert (s.long, s.short) == (("NAS100", None) if score > 0 else (None, "NAS100"))
        assert len(got) == 1 and got[0].key == "NAS100"
        assert got[0].side == (BUY if score > 0 else SELL)
        assert "sole of {NAS100}" in got[0].reason
    else:
        assert not got and "NAS100" in s.reasons


def test_missing_cash_open_bar_live_matches_scan(mcfg, market, default_run):
    strat, _, cands = default_run
    assert assert_generate_matches_scan(strat, mcfg, market, [ts("2024-03-05 15:30")],
                                        full=cands) == 1


def test_missing_regime_h1_bar_at_ger40_t(mcfg, default_run):
    """At 10:00 Berlin the regime observation is the core 09:00-10:00 H1 bar, which closes
    exactly at T; if it is missing the regime is UNAVAILABLE at T (COMMON 6.6 live rule), in
    scan() and in generate() alike."""
    strat, _, cands = default_run
    c = next(x for x in cands if x.key == "GER40")
    T = ts(c.signal_time)
    assert local(T, "GER40", mcfg).time() == time(10, 0)
    gap = make_market(h1_drops={"GER40": [T - pd.Timedelta(hours=1)]})
    svc = build_services(mcfg, gap)
    cyc = list(strat.cycles(*svc, gap.scan_start, gap.end))
    s = _sel_at(cyc, T)
    assert s.reasons["GER40"] == "regime unavailable: regime H1 bar missing"
    assert not [x for y, cs in cyc for x in cs if y.T == T]
    full = [x for _, cs in cyc for x in cs]
    assert assert_generate_matches_scan(strat, mcfg, gap, [T, T + pd.Timedelta(days=1)],
                                        full=full) >= 0


def test_scan_selection_sink(mcfg, market, services, default_run):
    """COMMON 4.2.8: scan(on_selection=) hands every cycle's Selection to the caller in order."""
    strat, cycles, cands = default_run
    got = []
    out = strat.scan(*services, market.scan_start, market.end, on_selection=got.append)
    assert_candidates_equal(out, cands)
    assert [s.log_line() for s in got] == [s.log_line() for s, _ in cycles]


def test_datr_nan_after_segment_break_blocks_ger40(mcfg, default_run, services):
    """The GER40 hole (2024-02-26..03-05) restarts every lookback. Once RS_4h is back (10 dates
    of TODV) DATR still needs 14 dates, so a few dates are incomplete because of DATR alone."""
    svc, _ = services
    _, cycles, cands = default_run
    after = [s for s, _ in cycles if s.members == ("GER40",)
             and ts("2024-03-05") <= s.T < ts("2024-04-01")]
    assert after
    datr_only = [s for s in after if s.incomplete == f"incomplete group: GER40: {DATR_NAN}"]
    assert datr_only
    for s in datr_only:
        d = local(s.T, "GER40", mcfg).date()
        assert math.isnan(svc.get("GER40").datr(d))
        assert math.isfinite(s.scores["GER40"])
    assert not [c for c in cands if c.key == "GER40" and ts("2024-02-26") <= c.signal_time
                and any(c.signal_time == s.T for s in datr_only)]


# ---------------------------------------------------------------------------------------------
# Eligibility: width, regime gate, restricted windows, one per index per day
# ---------------------------------------------------------------------------------------------

def test_width_filter_binds(mcfg, market, services):
    tight = SesIbx01(mcfg, {"max_ib_datr": 1e-6}, off_grid=True)
    cyc = list(tight.cycles(*services, market.scan_start, market.end))
    assert not [c for _, cs in cyc for c in cs]
    width = [r for s, _ in cyc for r in s.reasons.values() if r.startswith("IB width W")]
    assert width and all("> max_ib_datr 1e-06" in r for r in width)
    wide = SesIbx01(mcfg, {"max_ib_datr": 100.0}, off_grid=True)
    wcyc = list(wide.cycles(*services, market.scan_start, market.end))
    assert not [r for s, _ in wcyc for r in s.reasons.values() if r.startswith("IB width")]


def test_regime_gate_and_unavailable(mcfg, market):
    def never(m, cfg):
        return {k: ProxyRegimeTable(k, m.h1[k], threshold=1e9) for k in m.keys}

    def outage(m, cfg):
        return {k: ProxyRegimeTable(k, m.h1[k], unavailable=(("2024-01-01", "2025-01-01"),))
                for k in m.keys}

    strat = SesIbx01(mcfg)
    for factory, text in ((never, "regime gate ACTIVE not passed"),
                          (outage, "regime unavailable: test outage")):
        cyc = list(strat.cycles(*build_services(mcfg, market, factory), market.scan_start,
                                market.end))
        assert not [c for _, cs in cyc for c in cs]
        rs = [r for s, _ in cyc for r in s.reasons.values()]
        assert rs and all(r.startswith(text) for r in rs)


def test_restricted_overlap_rule():
    T, exp = ts("2024-03-11 14:30"), ts("2024-03-11 16:00")
    w = lambda a, b: [(ts(a), ts(b), "CPI")]        # noqa: E731
    assert restricted_overlap(w("2024-03-11 16:00", "2024-03-11 16:30"), T, exp) == "CPI"
    assert restricted_overlap(w("2024-03-11 12:00", "2024-03-11 14:45"), T, exp) == "CPI"
    assert restricted_overlap(w("2024-03-11 15:00", "2024-03-11 15:15"), T, exp) == "CPI"
    assert restricted_overlap(w("2024-03-11 12:00", "2024-03-11 14:30"), T, exp) is None
    assert restricted_overlap(w("2024-03-11 16:15", "2024-03-11 17:00"), T, exp) is None
    assert restricted_overlap([], T, exp) is None


def test_restricted_window_removes_index(mcfg, market, services, default_run):
    _, _, cands = default_run
    c = next(x for x in cands if x.key == "US30")
    T = ts(c.signal_time)
    for start, end, removed in ((c.expires_utc, c.expires_utc + M15, True),      # at EXPIRY
                                (T - pd.Timedelta(hours=1), T, False),          # ends at T
                                (c.expires_utc + M15, c.expires_utc + 2 * M15, False)):
        cfg2 = copy.deepcopy(mcfg)
        cfg2["rules"]["restricted_windows_utc"] = [
            {"start": start.isoformat(), "end": end.isoformat(), "reason": "FOMC"}]
        strat = SesIbx01(cfg2)
        cyc = list(strat.cycles(*services, market.scan_start, market.end))
        s = _sel_at(cyc, T)
        got = [x for _, cs in cyc for x in cs if ts(x.signal_time) == T and x.key == "US30"]
        if removed:
            assert not got
            assert s.reasons["US30"] == "restricted window overlaps [T, EXPIRY]: FOMC"
            # NAS100 shares US30's T and EXPIRY, so it is removed too.
            assert "NAS100" not in s.eligible
        else:
            assert_candidates_equal(got, [c])


def test_one_candidate_per_index_per_day(mcfg, services, default_run):
    """With one decision time per date the rule cannot bind in a scan; check the hook."""
    svc, tabs = services
    strat, _, cands = default_run
    c = next(x for x in cands if x.key == "NAS100")
    T, fx = ts(c.signal_time), svc.get("NAS100")
    d = local(T, "NAS100", mcfg).date()

    def view(book):
        return CycleView(T, mcfg, strat.params, {"NAS100": fx}, {"NAS100": fx.m15_pos(T)}, {},
                         {"NAS100": d}, book)

    empty = LockoutBook(strat.id, {"NAS100": fx.m15})
    assert strat.ineligible_cause(view(empty), "NAS100") is None
    used = LockoutBook(strat.id, {"NAS100": fx.m15})
    used.add(c, d)
    assert strat.ineligible_cause(view(used), "NAS100") == \
        "one SES-IBX-01 candidate per index per day: already issued on this date"
    other = LockoutBook(strat.id, {"NAS100": fx.m15})
    other.add(c, d - pd.Timedelta(days=1))
    assert strat.ineligible_cause(view(other), "NAS100") is None


def test_zero_width_ib_not_guarded(mcfg, services, default_run, monkeypatch):
    """SPEC NOTE 4: the spec removed the minimum-width guard, so W = 0 still gives a candidate
    (SL = trigger = TP); the ticket engine suppresses it (sizing: 'stop distance is zero'), and
    it consumes the index's one candidate that day."""
    svc, tabs = services
    _, _, cands = default_run
    c = next(x for x in cands if x.key == "GER40")
    T = ts(c.signal_time)
    orig = SesIbx01._ib

    def flat(self, view, key):
        ib = orig(self, view, key)
        if view.T == T and key == "GER40":
            return dataclasses.replace(ib, ibl=ib.ibh, w=0.0)
        return ib

    monkeypatch.setattr(SesIbx01, "_ib", flat)
    got = SesIbx01(mcfg).scan(svc, tabs, T, T + M15)
    assert len(got) == 1 and got[0].key == "GER40" and got[0].side == c.side
    g = got[0]
    assert g.entry == g.stop == g.take_profit == g.chase_limit == g.sizing_entry
    assert g.d == 0.0 and "W 0.00 = 0.00 DATR" in g.reason
    # signals.engine.build_ticket sizes on sizing_entry and suppresses a ticket without lots.
    size = size_position(100_000.0, 0.5, g.sizing_entry, g.stop,
                         SimpleNamespace(point=0.01), 3.0)
    assert size.lots is None and size.note == "stop distance is zero"


def test_w_datr_text():
    assert w_datr_text(30.0, 100.0) == "0.30"
    assert w_datr_text(0.0, 0.0) == "n/a" and w_datr_text(5.0, 0.0) == "n/a"


def test_zero_datr_never_crashes(mcfg, services, default_run, monkeypatch):
    """SPEC NOTE 6: a finite DATR of 0 is not a completeness cause and the width filter is read
    literally (W > 0 fails W <= max_ib_datr * 0, W = 0 passes); only the printed ratio becomes
    "n/a", so neither branch raises ZeroDivisionError and aborts the cycle."""
    svc, tabs = services
    _, _, cands = default_run
    c = next(x for x in cands if x.key == "GER40")
    T = ts(c.signal_time)
    orig_datr, orig_ib = SesIbx01._datr, SesIbx01._ib

    def zero_datr(self, view, key):
        return 0.0 if (view.T == T and key == "GER40") else orig_datr(self, view, key)

    monkeypatch.setattr(SesIbx01, "_datr", zero_datr)
    # W > 0: ineligible through the width filter, group not blocked, no exception.
    cyc = list(SesIbx01(mcfg).cycles(svc, tabs, T, T + M15))
    assert not [x for _, cs in cyc for x in cs if x.key == "GER40"]
    reasons = [s.reasons.get("GER40") for s, _ in cyc if "GER40" in s.reasons]
    assert reasons and reasons[0].startswith("IB width W") and "= n/a DATR" in reasons[0]

    # W = 0: passes 0 <= 0.6 * 0, so the candidate is built (SPEC NOTE 4) with "n/a" printed.
    def flat(self, view, key):
        ib = orig_ib(self, view, key)
        if view.T == T and key == "GER40":
            return dataclasses.replace(ib, ibl=ib.ibh, w=0.0)
        return ib

    monkeypatch.setattr(SesIbx01, "_ib", flat)
    got = SesIbx01(mcfg).scan(svc, tabs, T, T + M15)
    assert len(got) == 1 and got[0].key == "GER40" and "W 0.00 = n/a DATR" in got[0].reason


# ---------------------------------------------------------------------------------------------
# Look-ahead and live == batch (COMMON 9, ER-15)
# ---------------------------------------------------------------------------------------------

DST_EDGE_TIMES = ["2024-03-08 15:30", "2024-03-11 14:30", "2024-03-28 09:00",
                  "2024-04-02 08:00", "2024-03-12 14:30", "2024-04-03 08:00",
                  "2024-03-05 15:30"]            # US30 outside G(T): NAS100 ranked alone


def test_no_lookahead(mcfg, market, default_run):
    strat, _, cands = default_run
    cand_t = sorted({ts(c.signal_time) for c in cands})
    cuts = [cand_t[3], cand_t[len(cand_t) // 2], ts("2024-03-11 14:30"),
            ts("2024-03-11 14:00")]                 # mid-IB (09:30-10:30 New York)
    full = assert_no_lookahead(strat, mcfg, market, cuts, seeds=(1, 2))
    assert_candidates_equal(full, cands)


def test_generate_equals_scan(mcfg, market, default_run):
    strat, _, cands = default_run
    times = sample_times(strat, mcfg, cands, market.scan_start, market.end, n_candidates=6,
                         n_other=3)
    times = sorted(set(times) | {ts(t) for t in DST_EDGE_TIMES})
    n = assert_generate_matches_scan(strat, mcfg, market, times, full=cands)
    assert n >= 6


def test_generate_outside_decision_time_is_empty(mcfg, market):
    strat = SesIbx01(mcfg)
    T = ts("2024-03-12 14:15")                      # 10:15 New York: inside the IB
    trunc = market.truncated(T)
    svc, tabs = build_services(mcfg, trunc)
    ctx = StrategyContext(features=svc, now=T, cfg=mcfg, regime_tables=tabs)
    assert strat.generate(trunc.m15, trunc.h1, T, ctx) == []


@pytest.mark.parametrize("params", [{"ib_minutes": 30}, {"ib_minutes": 90}])
def test_grid_extremes_live_equals_scan(mcfg, market, services, params):
    strat = SesIbx01(mcfg, params)
    full = strat.scan(*services, market.scan_start, market.end)
    assert full
    times = sample_times(strat, mcfg, full, market.scan_start, market.end, n_candidates=3,
                         n_other=1)
    assert_generate_matches_scan(strat, mcfg, market, times, full=full)


def test_scan_window_slicing_is_stable(mcfg, market, services, default_run):
    """A scan started mid-history gives exactly the later part of the full scan."""
    strat, _, cands = default_run
    mid = ts("2024-03-11 12:00")
    part = strat.scan(*services, mid, market.end)
    assert_candidates_equal(part, [c for c in cands if ts(c.signal_time) >= mid])
    assert np.all(np.diff([ts(c.signal_time).value for c in part]) >= 0)
