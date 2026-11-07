"""COMMON 10.6 benchmarks (backtest.benchmarks) on SYNTHETIC markets only: B1 placebo geometry,
windows, seeding and the criterion 7 comparison; B2 random index; B3 gate off through an
always-pass regime-table wrapper (no strategy file edited); guards (STRESS-d2 only, MR-TWAP-01
dropped, holdout locked). No real bar is ever priced here.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

import backtest.benchmarks as bm
from backtest import runner as R
from backtest import trials as tl
from backtest.engine import run_backtest
from backtest.folds import Fold
from backtest.scenarios import SCENARIOS
from core.calendar import exchange_for
from core.config import load_config
from data.mt5_adapter import SymbolSpec
from strategies.base import MARKET_BAND, STOP_ENTRY, Candidate
from strategies.rs_mom_01 import RsMom01
from strategies.selection import BUY, SELL
from strategies.ses_ibx_01 import SesIbx01, expiry_utc
from tests.conftest import ROOT
from tests.strategy_testkit import (SyntheticMarket, build_services, make_market,
                                    proxy_regime_tables)

STRESS = SCENARIOS["STRESS-d2"]
FEW_SEEDS = (1, 2, 3, 4, 5)
CLOSE, SHIFT = bm.SES_PLACEBO_CLOSE, bm.SES_PLACEBO_SHIFT
# runner-style test windows inside the synthetic market (features from ~2024-02-05)
W = (Fold(1, date(2024, 2, 5), date(2024, 2, 29)), Fold(2, date(2024, 3, 1), date(2024, 3, 31)),
     Fold(3, date(2024, 4, 1), date(2024, 4, 12)))
W_SPANS = [(w.start, w.end) for w in W]
LEVEL_FIELDS = ("key", "side", "entry", "stop", "take_profit", "chase_limit", "skip_level",
                "sizing_entry", "expires_utc", "time_exit_utc", "order_type", "signal_time")


def _cfg() -> dict:
    c = load_config(ROOT / "config.yaml")
    c["owner_confirmed"] = True
    c["sessions_utc"] = {}
    c["rules"]["competition_end_utc"] = None
    return c


def _spec(k: str) -> SymbolSpec:
    return SymbolSpec(k, 2, 0.01, 0.01, 0.01, 0.1, 250.0, 0.1, "USD", swap_long=-5.0,
                      swap_short=-5.0, swap_mode=1, swap_rollover3days=3, trade_stops_level=100)


@pytest.fixture(scope="module")
def world():
    cfg = _cfg()
    market = make_market()
    svc, tables = build_services(cfg, market)
    specs = {k: _spec(k) for k in market.keys}
    return cfg, market, svc, tables, specs


def _inputs(world, strategy_cls, **over) -> bm.BenchmarkInputs:
    cfg, market, svc, tables, specs = world
    kw = dict(cfg=cfg, strategy=strategy_cls(cfg), features=svc, regime_tables=tables,
              bars=market.m15, specs=specs)
    kw.update(over)
    return bm.BenchmarkInputs(**kw)


@pytest.fixture(scope="module")
def rs(world):
    inp = _inputs(world, RsMom01)
    _, market, *_ = world
    run_a = inp.strategy.scan(inp.features, inp.regime_tables, market.scan_start, market.end)
    assert len(run_a) >= 10
    return inp, run_a


@pytest.fixture(scope="module")
def ses(world):
    inp = _inputs(world, SesIbx01)
    _, market, *_ = world
    run_a = inp.strategy.scan(inp.features, inp.regime_tables, market.scan_start, market.end)
    assert len(run_a) >= 10
    return inp, run_a


def _mode(inp: bm.BenchmarkInputs, mode: str | None) -> bm.BenchmarkInputs:
    """The SES-IBX-01 placebo geometry is pending a Quant ruling: tests name it explicitly."""
    return replace(inp, ses_placebo=mode)


def _levels(c: Candidate) -> tuple:
    return tuple(getattr(c, f) for f in LEVEL_FIELDS)


def _local(cfg, c: Candidate):
    return pd.Timestamp(c.signal_time).tz_convert(exchange_for(cfg, c.key).tz)


# --- geometry: the strategy's own levels are reproduced ------------------------------------
@pytest.mark.parametrize("which", ["rs", "ses"])
def test_strategy_levels_reproduce_run_a(which, request):
    inp, run_a = request.getfixturevalue(which)
    geo = bm.geometry_for(inp)
    for c in run_a:
        nc, cause = geo.strategy_levels(c.key, c.side, c.signal_time)
        assert cause == "" and nc is not None
        assert _levels(nc) == _levels(c)


def test_rs_mom_placebo_window_and_geometry(rs):
    inp, run_a = rs
    cfg = inp.cfg
    k, R = inp.strategy.params["k"], inp.strategy.params["R"]
    opts = bm.placebo_options(inp, run_a)
    assert len(opts) == len(run_a)
    win = inp.strategy.windows(inp.strategy.params)
    for o in opts:
        c = o.source
        assert o.options, "every synthetic run-A candidate has eligible placebo times"
        assert len(o.options) <= o.window_times
        assert c.signal_time in {p.signal_time for p in o.options}   # T itself can be drawn
        for p in o.options:
            assert (p.key, p.side, p.order_type) == (c.key, c.side, MARKET_BAND)
            assert _local(cfg, p).date() == _local(cfg, c).date()
            assert win[c.key].contains(_local(cfg, p).time())
            fx = inp.fx(p.key)
            assert p.entry == pytest.approx(float(fx.m15_at(p.signal_time)["close"]))
            atrh = float(fx.h1_at(p.signal_time).row["atr14"])
            sg = 1 if p.side == BUY else -1
            # stop k ATRH away (rounded 0.01 away), target R k ATRH (rounded toward)
            assert sg * (p.entry - p.stop) == pytest.approx(k * atrh, abs=0.011)
            assert sg * (p.take_profit - p.entry) == pytest.approx(R * k * atrh, abs=0.011)
            assert sg * (p.entry - p.stop) >= k * atrh - 1e-9
            assert p.time_exit_utc == c.time_exit_utc
            assert p.reason.startswith(bm.B1)
        same_t = [p for p in o.options if p.signal_time == c.signal_time][0]
        assert _levels(same_t) == _levels(c)


def test_ses_placebo_window_and_geometry_close_reference(ses):
    inp, run_a = ses
    inp = _mode(inp, CLOSE)
    cfg = inp.cfg
    m = inp.strategy.params["tp_ib_mult"]
    bar = pd.Timedelta(minutes=15)
    for o in bm.placebo_options(inp, run_a):
        c = o.source
        exp = expiry_utc(cfg, c.key, c.signal_time)
        times = [p.signal_time for p in o.options]
        assert times and min(times) == c.signal_time            # starts at the IB end
        assert all(t + STRESS.d * bar < exp for t in times)      # still placeable at d = 2
        # window = IB end .. last close placeable before EXPIRY; a missing bar drops a T'
        assert len(times) <= o.window_times == (exp - c.signal_time) // bar - STRESS.d
        w = c.d / 0.55                                          # run-A D = 0.55 W
        for p in o.options:
            assert (p.key, p.side, p.order_type) == (c.key, c.side, STOP_ENTRY)
            assert p.expires_utc == exp and p.time_exit_utc == c.time_exit_utc
            ref = float(inp.fx(p.key).m15_at(p.signal_time)["close"])
            sg = 1 if p.side == BUY else -1
            assert sg * (p.entry - ref) == pytest.approx(0.05 * w)
            assert sg * (ref - p.stop) == pytest.approx(0.5 * w)
            assert p.d == pytest.approx(0.55 * w)
            assert sg * (p.take_profit - ref) == pytest.approx(m * w)
            assert p.chase_limit == pytest.approx(p.entry + sg * 0.20 * p.d)


def test_ses_placebo_shifted_run_a_geometry(ses):
    """SES_PLACEBO_SHIFT: the run-A bracket moved by C' - C_T; T' = IB end equals run A."""
    inp, run_a = ses
    inp = _mode(inp, SHIFT)
    m = inp.strategy.params["tp_ib_mult"]
    for o in bm.placebo_options(inp, run_a):
        c = o.source
        fx = inp.fx(c.key)
        c0 = float(fx.m15_at(c.signal_time)["close"])
        sg = 1 if c.side == BUY else -1
        w = c.d / 0.55
        off = sg * (c.entry - c0) / w                          # run A's trigger offset, in W
        assert o.options and o.options[0].signal_time == c.signal_time
        assert _levels(o.options[0]) == _levels(c)             # T' = T reproduces run A
        for p in o.options:
            assert (p.key, p.side, p.order_type) == (c.key, c.side, STOP_ENTRY)
            assert p.expires_utc == c.expires_utc and p.time_exit_utc == c.time_exit_utc
            ref = float(fx.m15_at(p.signal_time)["close"])
            shift = ref - c0
            for f in ("entry", "stop", "take_profit"):
                assert getattr(p, f) == pytest.approx(getattr(c, f) + shift)
            assert sg * (p.entry - ref) == pytest.approx(off * w)
            assert p.d == pytest.approx(0.55 * w)
            assert sg * (p.take_profit - p.entry) == pytest.approx((m - 0.05) * w)
            assert SHIFT in p.reason


def test_ses_placebo_ruling_is_shift():
    """COMMON 12D, amendment A4.2: the Quant ruled for the shifted run-A bracket."""
    assert bm.SES_PLACEBO_RULING == bm.SES_PLACEBO_SHIFT


def test_ses_placebo_refuses_without_ruling(ses, world, monkeypatch):
    """SPEC NOTE 4: without a ruling no SES geometry is picked; B2 and B3 are unaffected."""
    monkeypatch.setattr(bm, "SES_PLACEBO_RULING", None)
    inp, run_a = ses
    assert inp.ses_placebo is None
    with pytest.raises(ValueError, match="Quant's ruling"):
        bm.placebo_options(inp, run_a)
    with pytest.raises(ValueError, match="Quant's ruling"):
        bm.geometry_for(inp).placebo_levels(run_a[0].key, run_a[0].side, run_a[0].signal_time)
    with pytest.raises(ValueError, match="ses_placebo"):
        _mode(inp, "somewhere-else")
    assert bm.random_index_candidates(inp, run_a[:3])[0]
    _, market, *_ = world
    assert bm.gate_off_candidates(inp, [(market.scan_start, market.end)])


def test_ses_placebo_counts_by_geometry(ses):
    """The QA finding behind the escalation, pinned on the synthetic market: the
    close-reference bracket skips far more placebo orders than run A (trigger 0.05 W from
    price); the shifted bracket keeps run A's skip share. Counts are recorded per draw."""
    inp, run_a = ses
    a = bm.result_stats(inp.backtest(run_a))
    share = {}
    for mode in (CLOSE, SHIFT):
        pl = bm.run_placebo(_mode(inp, mode), run_a, range(1, 11))
        assert pl.geometry == mode
        d = pl.draws
        assert (d[[f"n_{s}" for s in bm.STATUSES]].sum(axis=1) == d["n_candidates"]).all()
        share[mode] = float((d["n_skipped"] / d["n_candidates"]).mean())
    a_share = a["n_skipped"] / a["n_candidates"]
    assert abs(share[SHIFT] - a_share) < 0.10
    assert share[CLOSE] - a_share > 0.15


def test_ses_placebo_window_follows_scenario_delay(ses):
    inp, run_a = ses
    c = run_a[0]
    exp = expiry_utc(inp.cfg, c.key, c.signal_time)
    slots = (exp - c.signal_time) // pd.Timedelta(minutes=15)
    for d in (1, 2, 3):
        alt = replace(inp)
        object.__setattr__(alt, "scenario", replace(STRESS, d=d))   # bypass the guard: test only
        assert len(bm.geometry_for(alt).placebo_times(c)) == slots - d


# --- B1 draws ------------------------------------------------------------------------------
def test_placebo_draw_seeded_and_uniform(rs):
    inp, run_a = rs
    opts = bm.placebo_options(inp, run_a)
    a, b = bm.placebo_draw(opts, 7), bm.placebo_draw(opts, 7)
    assert a == b and a.seed == 7 and a.unmatched == 0
    assert len(a.candidates) == len(run_a)
    for p, o in zip(a.candidates, opts):
        assert p in o.options
    draws = [bm.placebo_draw(opts, s) for s in bm.PLACEBO_SEEDS]
    assert len({tuple(d.candidates) for d in draws}) > 150
    # uniform choice: every option of the first candidate is drawn, frequencies balanced
    o0 = opts[0].options
    cnt = Counter(d.candidates[0].signal_time for d in draws)
    assert set(cnt) == {p.signal_time for p in o0}
    exp = len(draws) / len(o0)
    chi2 = sum((v - exp) ** 2 / exp for v in cnt.values())
    assert chi2 < 3 * len(o0) + 10


def test_placebo_seeds_are_1_to_200():
    assert bm.PLACEBO_SEEDS == tuple(range(1, 201))


def test_placebo_unmatched_candidate_is_counted(rs):
    inp, run_a = rs
    opts = bm.placebo_options(inp, run_a)
    empty = bm.PlaceboOptions(run_a[0], (), 13)
    dr = bm.placebo_draw([empty] + list(opts), 3)
    assert dr.unmatched == 1
    assert dr.candidates == bm.placebo_draw(opts, 3).candidates   # no random number consumed


@pytest.mark.parametrize("which,mode", [("rs", None), ("ses", CLOSE), ("ses", SHIFT)])
def test_run_placebo_matches_direct_backtests(which, mode, request):
    inp, run_a = request.getfixturevalue(which)
    inp = _mode(inp, mode)
    res = bm.run_placebo(inp, run_a, FEW_SEEDS, keep_results=True)
    assert list(res.draws["seed"]) == list(FEW_SEEDS) and res.scenario == "STRESS-d2"
    opts = bm.placebo_options(inp, run_a)
    for s, total, run in zip(FEW_SEEDS, res.totals, res.runs):
        dr = bm.placebo_draw(opts, s)
        direct = run_backtest([(inp.strategy_id, c) for c in dr.candidates], inp.bars,
                              inp.cfg, STRESS, inp.specs)
        assert total == pytest.approx(float(direct.filled["pnl_usd"].sum()))
        row = res.draws[res.draws["seed"] == s].iloc[0]
        st = direct.trades["status"].value_counts()
        for name in bm.STATUSES:
            assert row[f"n_{name}"] == st.get(name, 0)
        assert row["n_candidates"] == len(dr.candidates)
        assert run.result.scenario == STRESS and run.seed == s
        assert set(run.result.trades["strategy_id"]) == {inp.strategy_id}
    again = bm.run_placebo(inp, run_a, FEW_SEEDS)
    np.testing.assert_array_equal(again.totals, res.totals)          # deterministic
    assert not again.runs
    assert res.p95 == pytest.approx(np.percentile(res.totals, 95))
    assert len({round(t, 6) for t in res.totals}) > 1


def test_criterion7_strictly_above_p95(rs):
    inp, run_a = rs
    draws = pd.DataFrame({"seed": range(1, 201), "n_candidates": 1, "n_filled": 1,
                          "n_skipped": 0, "n_cancelled": 0, "n_blocked": 0, "n_suppressed": 0,
                          "total_pnl_usd": np.arange(200, dtype=float), "expectancy_usd": 0.0,
                          "expectancy_r": 0.0, "unmatched": 0})[bm.DRAW_COLUMNS]
    pl = bm.PlaceboResult("RS-MOM-01", "STRESS-d2", "demo", draws, 1)
    p95 = np.percentile(np.arange(200.0), 95)
    assert pl.p95 == pytest.approx(p95) == pytest.approx(189.05)
    assert pl.passes(pl.p95 + 1e-6) and not pl.passes(pl.p95) and not pl.passes(10.0)
    row = pl.trial_log_fields()
    assert row["stage"] == "benchmark" and row["variant"] == bm.B1
    assert "not counted in N" in row["notes"] and "filled 1.0 (1-1)" in row["notes"]
    res = inp.backtest(run_a)
    total = float(res.filled["pnl_usd"].sum())
    row = pl.trial_log_fields(res)
    assert ("PASS" if total > pl.p95 else "FAIL") in row["result_summary"]
    assert row["extra"]["run_a"]["n_filled"] == len(res.filled)
    assert "run A: " in row["notes"]


def test_criterion7_refuses_partial_seeds(world, rs):
    """QA: criterion 7 is defined on the pre-registered seeds 1..200 only. Fewer seeds, other
    seeds or another order raise unless allow_partial, which never prints PASS / FAIL."""
    inp, run_a = rs
    _, market, *_ = world
    res = inp.backtest(run_a)
    pl = bm.run_placebo(inp, run_a, FEW_SEEDS)
    assert pl.seeds == FEW_SEEDS and not pl.complete
    for call in (lambda: bm.criterion7(res, pl), lambda: pl.passes(1e9),
                 lambda: pl.trial_log_fields(res)):
        with pytest.raises(ValueError, match="seeds 1..200"):
            call()
    with pytest.raises(ValueError, match="seeds 1..200"):          # before any backtest
        bm.run_benchmarks(inp, run_a, res, [(market.scan_start, market.end)], seeds=FEW_SEEDS)
    assert "criterion 7" not in pl.trial_log_fields()["result_summary"]   # no run A: no verdict
    c7 = bm.criterion7(res, pl, allow_partial=True)
    assert c7.passed is None and c7.partial and c7.n_draws == 5
    assert c7.text.startswith("NOT criterion 7 (partial seeds: 5 of 200")
    assert "PASS" not in c7.text and "FAIL" not in c7.text
    row = pl.trial_log_fields(res, allow_partial=True)
    assert "NOT criterion 7 (partial seeds" in row["result_summary"]
    assert row["extra"]["criterion7_passed"] is None and row["extra"]["complete_seeds"] is False
    # 200 draws but not 1..200 in order: still not the criterion
    base = pd.DataFrame({"seed": range(1, 201), "n_candidates": 1, "n_filled": 1,
                         "n_skipped": 0, "n_cancelled": 0, "n_blocked": 0, "n_suppressed": 0,
                         "total_pnl_usd": np.arange(200, dtype=float), "expectancy_usd": 0.0,
                         "expectancy_r": 0.0, "unmatched": 0})[bm.DRAW_COLUMNS]
    full = bm.PlaceboResult("RS-MOM-01", "STRESS-d2", "demo", base, 1)
    assert full.complete
    assert bm.criterion7(res, full).passed in (True, False)
    for bad in (base.iloc[::-1].reset_index(drop=True), base.assign(seed=base["seed"] + 1)):
        other = bm.PlaceboResult("RS-MOM-01", "STRESS-d2", "demo", bad, 1)
        with pytest.raises(ValueError, match="seeds 1..200"):
            bm.criterion7(res, other)


# --- B2 random index -----------------------------------------------------------------------
def test_b2_swapped_only_figure(rs):
    """QA: beside the literal B2 verdict (all candidates), the $ expectancy of run A and B2 on
    the swapped candidates only, found independently by merging trades with the map."""
    inp, run_a = rs
    res = inp.backtest(run_a)
    b2 = bm.run_random_index(inp, run_a, seed=1)
    mp = b2.mapping
    sw = mp[mp["flag"] == ""]
    assert 0 < len(sw) < len(mp) and len(sw) + b2.n_kept == len(mp)
    assert (sw["key"] != "GER40").all() and (mp.loc[mp["key"] == "GER40", "flag"] != "").all()

    def merged(trades, key_col):
        m = sw[["signal_time", "side", key_col]].rename(columns={key_col: "key"})
        t = trades.assign(signal_time=pd.to_datetime(trades["signal_time"], utc=True))
        return t.merge(m, on=["key", "side", "signal_time"], how="inner")
    ra, rb = merged(res.trades, "key"), merged(b2.run.result.trades, "b2_key")
    assert len(ra) == len(rb) == len(sw)
    ea = ra.loc[ra["status"] == "filled", "pnl_usd"].mean()
    eb = rb.loc[rb["status"] == "filled", "pnl_usd"].mean()
    v = bm.b2_verdict(res, b2)
    assert v.swapped_only["n"] == len(sw)
    assert v.swapped_only["run_a"]["expectancy_usd"] == pytest.approx(ea)
    assert v.swapped_only["b2"]["expectancy_usd"] == pytest.approx(eb)
    assert v.swapped_only["run_a"]["n_candidates"] == len(sw)
    assert f"Swapped candidates only ({len(sw)} of {len(mp)}" in v.text
    # the literal verdict is still over all candidates
    a, b = bm.result_stats(res), bm.result_stats(b2.run.result)
    assert v.adds_nothing == (not a["expectancy_usd"] > b["expectancy_usd"])
    row = b2.trial_log_fields(res)
    assert row["extra"]["swapped_only"]["b2"]["expectancy_usd"] == pytest.approx(eb)
    assert row["extra"]["n_swapped"] == len(sw) and "swapped only (" in row["result_summary"]
    json.dumps(row["extra"], allow_nan=False)
    assert "run_a" not in b2.trial_log_fields()["extra"]["swapped_only"]
    with pytest.raises(ValueError, match="contract"):
        b2.trial_log_fields(replace(res, contract="competition"))


def test_select_trades_is_a_multiset_match(rs):
    inp, run_a = rs
    tr = inp.backtest(run_a[:4]).trades
    c = run_a[0]
    one = bm.select_trades(tr, [(c.key, c.side, c.signal_time)] * 3)   # occurs once in trades
    assert len(one) == 1 and one.iloc[0]["key"] == c.key
    assert bm.select_trades(tr, []).empty
    assert bm.trades_stats(tr) == pytest.approx(bm.result_stats(inp.backtest(run_a[:4])),
                                                nan_ok=True)
@pytest.mark.parametrize("which", ["rs", "ses"])
def test_random_index_swaps_us_and_keeps_ger40(which, request):
    inp, run_a = request.getfixturevalue(which)
    geo = bm.geometry_for(inp)
    cands, mp = bm.random_index_candidates(inp, run_a, seed=1)
    assert len(cands) == len(mp) == len(run_a)
    assert set(mp["key"]) >= {"GER40"} and set(mp["key"]) & {"US30", "NAS100"}
    swapped = 0
    for c, b, (_, r) in zip(run_a, cands, mp.iterrows()):
        assert b.signal_time == c.signal_time and b.side == c.side
        if c.key == "GER40":
            assert b is c and r["flag"].startswith("kept") and r["n_alternatives"] == 0
            continue
        other = {"US30": "NAS100", "NAS100": "US30"}[c.key]
        ref, cause = geo.strategy_levels(other, c.side, c.signal_time)
        if r["flag"]:                       # other US index without a level at T (data gap)
            assert b is c and r["n_alternatives"] == 0
            assert ref is None or not geo.window_valid(other, c.signal_time)
            continue
        swapped += 1
        assert b.key == r["b2_key"] == other and r["n_alternatives"] == 1
        assert _levels(b) == _levels(ref) and b.reason.startswith(bm.B2)
    assert swapped >= 5
    again, _ = bm.random_index_candidates(inp, run_a, seed=1)
    assert again == cands


def test_random_index_kept_when_other_index_has_no_bar(rs):
    inp, _ = rs
    # NAS100's bar opening 2024-02-22 15:30 UTC is missing (testkit DEFAULT_DROPS), so at
    # T = 15:45 (10:45 New York, inside both US windows) NAS100 has no level: keep US30.
    T = pd.Timestamp("2024-02-22 15:45", tz="UTC")
    c, cause = bm.geometry_for(inp).strategy_levels("US30", BUY, T)
    assert c is not None and cause == ""
    cands, mp = bm.random_index_candidates(inp, [c], seed=1)
    assert cands == [c] and mp.loc[0, "flag"].startswith("kept")
    nc, cause = bm.geometry_for(inp).strategy_levels("NAS100", BUY, T)
    assert nc is None and "no M15 bar" in cause


def test_random_index_draw_is_uniform_among_several(rs, monkeypatch):
    inp, run_a = rs
    us = [c for c in run_a if c.key != "GER40"][:1]
    geo = bm.geometry_for(inp)
    monkeypatch.setattr(bm.PlaceboGeometry, "window_valid", lambda self, k, T: True)

    def levels(self, key, side, T):            # every index gets a level (synthetic stand-in)
        return replace(us[0], key=key), ""
    monkeypatch.setattr(type(geo), "strategy_levels", levels)
    picks = Counter(bm.random_index_candidates(inp, us, seed=s)[0][0].key for s in range(1, 401))
    assert set(picks) == {"US30", "NAS100", "GER40"} - {us[0].key}
    assert min(picks.values()) > 150


def test_run_random_index_backtests_under_stress(rs):
    inp, run_a = rs
    r = bm.run_random_index(inp, run_a, seed=1)
    assert r.run.result.scenario == STRESS and r.run.n_candidates == len(run_a)
    _, mp = bm.random_index_candidates(inp, run_a, seed=1)
    assert r.n_kept == int((mp["flag"] != "").sum()) >= sum(c.key == "GER40" for c in run_a)
    assert r.trial_log_fields()["stage"] == "benchmark"


# --- B3 gate off ---------------------------------------------------------------------------
def test_gate_off_wrapper_equals_strategy_without_gate(world, rs):
    inp, run_a = rs
    _, market, svc, tables, _ = world
    spans = [(market.scan_start, pd.Timestamp("2024-03-01", tz="UTC")),
             (pd.Timestamp("2024-03-01", tz="UTC"), market.end)]
    off = bm.gate_off_candidates(inp, spans)
    NoGate = type("NoGate", (RsMom01,), {"regime_gate": None})
    ref = NoGate(inp.cfg).scan(svc, tables, market.scan_start, market.end)
    assert [_levels(c) for c in off] == [_levels(c) for c in ref]
    assert RsMom01.regime_gate == "ACTIVE" and inp.strategy.regime_gate == "ACTIVE"
    tags = Counter(c.regime.split("/")[0] for c in off)
    assert tags["CALM"] > 0 and "UNAVAILABLE" not in tags     # gate gone, status kept
    assert all(c.regime.split("/")[0] == "ACTIVE" for c in run_a)
    assert len(off) > len(run_a)


def test_gate_off_ses(world, ses):
    inp, run_a = ses
    _, market, *_ = world
    r = bm.run_gate_off(inp, [(market.scan_start, market.end)])
    assert r.run.result.scenario == STRESS
    assert any(c.regime.startswith("CALM") for c in r.run.candidates)
    assert r.trial_log_fields()["stage"] == "benchmark"


def test_gate_off_table_thresholds_and_delegation(world):
    _, market, _, tables, _ = world
    t = tables["US30"]
    on, calm = bm.GateOffTable(t, "ACTIVE"), bm.GateOffTable(t, "CALM")
    assert on.threshold == -math.inf and calm.threshold == math.inf
    times = pd.date_range("2024-02-05 15:00", periods=20, freq="15min", tz="UTC")
    pd.testing.assert_frame_equal(on.at_many(times), t.at_many(times))
    assert on.key == "US30"
    with pytest.raises(ValueError):
        bm.GateOffTable(t, None)
    wrapped = bm.gate_off_tables({"US30": t, "X": None}, "ACTIVE")
    assert wrapped["X"] is None and isinstance(wrapped["US30"], bm.GateOffTable)


# --- comparisons and the full set ----------------------------------------------------------
def test_run_benchmarks_end_to_end(world, ses):
    inp, run_a = ses
    inp = _mode(inp, SHIFT)
    _, market, *_ = world
    run_a_res = inp.backtest(run_a)
    rep = bm.run_benchmarks(inp, run_a, run_a_res, [(market.scan_start, market.end)],
                            seeds=FEW_SEEDS, allow_partial=True)
    assert rep.criterion7.run_a_total_usd == pytest.approx(
        float(run_a_res.filled["pnl_usd"].sum()))
    assert rep.criterion7.passed is None and rep.criterion7.partial     # 5 seeds: no verdict
    assert rep.lines()[0].startswith("NOT criterion 7 (partial seeds: 5 of 200")
    rows = rep.trial_log_rows()
    assert "NOT criterion 7 (partial seeds" in rows[0]["result_summary"]
    assert "PASS" not in rows[0]["result_summary"] and "FAIL" not in rows[0]["result_summary"]
    assert rows[0]["extra"]["criterion7_passed"] is None
    assert rows[0]["extra"]["run_a_above_p95"] == (rep.criterion7.run_a_total_usd
                                                   > rep.placebo.p95)
    assert rows[1]["extra"]["swapped_only"]["run_a"] == rep.b2.swapped_only["run_a"]
    assert [r["variant"] for r in rows] == [bm.B1, bm.B2, bm.B3]
    assert all(r["stage"] == "benchmark" for r in rows)
    assert all(r["scenario_key"] == "STRESS-d2" and r["contract"] == "demo" for r in rows)
    # each row maps onto one TrialRecord; the three get distinct config keys (variant)
    recs = [tl.TrialRecord(inp.strategy_id, dict(inp.strategy.params), r["stage"], "proto",
                           "synthetic", "s", "c", "d", {r["scenario_key"]: r["metrics"]},
                           r["extra"], variant=r["variant"]) for r in rows]
    assert len({x.key for x in recs}) == 3
    json.dumps([(r["metrics"], r["extra"]) for r in rows], allow_nan=False)
    a, b3 = rep.b3.run_a, rep.b3.bench
    assert rep.b3.adds_nothing == (b3["expectancy_usd"] >= a["expectancy_usd"])
    assert rep.b2.adds_nothing == (not rep.b2.run_a["expectancy_usd"]
                                   > rep.b2.bench["expectancy_usd"])
    assert len(rep.lines()) == 4 and "run A: " in rep.lines()[1]
    base = run_backtest([(inp.strategy_id, c) for c in run_a], inp.bars, inp.cfg,
                        SCENARIOS["BASE"], inp.specs)
    with pytest.raises(ValueError, match="STRESS-d2"):
        bm.criterion7(base, rep.placebo)


def test_verdict_texts(rs):
    inp, run_a = rs
    res = inp.backtest(run_a)
    b3 = bm.GateOffResult(bm.BenchmarkRun(bm.B3, None, tuple(run_a), res))
    v = bm.b3_verdict(res, b3)                    # equal expectancy: B3 >= gated
    assert v.adds_nothing and "adds nothing" in v.text
    b2 = bm.RandomIndexResult(bm.BenchmarkRun(bm.B2, 1, tuple(run_a), res),
                              pd.DataFrame({"flag": []}))
    v = bm.b2_verdict(res, b2)                    # equal: not beaten
    assert v.adds_nothing and "relative-strength selection adds nothing" in v.text


# --- guards --------------------------------------------------------------------------------
def test_only_stress_d2(world):
    with pytest.raises(ValueError, match="STRESS-d2"):
        _inputs(world, RsMom01, scenario=SCENARIOS["BASE"])


def test_mr_twap_is_never_run(world):
    class Dropped:
        id = "MR-TWAP-01"
        params: dict = {}
    with pytest.raises(ValueError, match="DROPPED"):
        _inputs(world, lambda cfg: Dropped())


def _with_holdout(inp: bm.BenchmarkInputs, start: str) -> dict:
    cfg = dict(inp.cfg)
    cfg["holdout"] = {"start": start}
    return cfg


def test_bars_reaching_holdout_are_refused(world, rs):
    """QA: the bars (and feature frames) themselves must end before holdout.start, as the
    runner's _check_pre_holdout requires; the synthetic bars run to 2024-04-12 23:45 UTC."""
    inp, _ = rs
    _, market, *_ = world
    last = max(pd.to_datetime(df["time_utc"], utc=True).max() for df in market.m15.values())
    assert last == pd.Timestamp("2024-04-12 23:45", tz="UTC")
    for start in ("2024-03-15", "2024-04-12 23:45"):
        with pytest.raises(PermissionError, match="final holdout"):
            replace(inp, cfg=_with_holdout(inp, start))
    # bars cut before the holdout, features still built on the full market: still refused
    cut = {k: df[pd.to_datetime(df["time_utc"], utc=True) < pd.Timestamp("2024-03-15", tz="UTC")]
           for k, df in market.m15.items()}
    with pytest.raises(PermissionError, match="features reach the final holdout"):
        replace(inp, cfg=_with_holdout(inp, "2024-03-15"), bars=cut)
    ok = replace(inp, cfg=_with_holdout(inp, "2024-04-13"))       # everything strictly before
    assert ok.cfg["holdout"]["start"] == "2024-04-13"


def test_holdout_is_locked(world, rs):
    """The candidate / span guard behind the frame check (the frame check is bypassed here so
    the second line of defence is exercised on its own)."""
    inp, run_a = rs
    locked = replace(inp)
    object.__setattr__(locked, "cfg", _with_holdout(inp, "2024-03-01"))   # test only
    late = [c for c in run_a if c.signal_time >= pd.Timestamp("2024-03-01", tz="UTC")]
    assert late
    with pytest.raises(ValueError, match="holdout"):
        bm.placebo_options(locked, run_a)
    with pytest.raises(ValueError, match="holdout"):
        bm.gate_off_candidates(locked, [(pd.Timestamp("2024-02-01", tz="UTC"),
                                         pd.Timestamp("2024-03-02", tz="UTC"))])


def test_no_final_holdout_flag_in_module():
    src = (ROOT / "backtest" / "benchmarks.py").read_text(encoding="utf-8")
    assert "final_holdout" not in src


@pytest.mark.parametrize("mode", [CLOSE, SHIFT])
def test_sell_side_placebo_mirrors(ses, mode):
    inp, run_a = ses
    sells = [c for c in run_a if c.side == SELL]
    assert sells
    geo = bm.geometry_for(_mode(inp, mode))
    c = sells[0]
    p, cause = geo.placebo_levels(c.key, SELL, c.signal_time + pd.Timedelta(minutes=15),
                                  source=c)
    assert cause == "" and p.stop > p.entry > p.take_profit and p.chase_limit < p.entry


# --- run A consistency, contract, runner parity --------------------------------------------
def test_contract_must_match(world, rs):
    inp, run_a = rs
    res = inp.backtest(run_a)
    pl = bm.run_placebo(inp, run_a, (1, 2))
    assert pl.contract == "demo" == res.contract
    other = replace(res, contract="competition")
    with pytest.raises(ValueError, match="contract"):
        bm.criterion7(other, pl)
    with pytest.raises(ValueError, match="contract"):
        pl.trial_log_fields(other)
    b3 = bm.GateOffResult(bm.BenchmarkRun(bm.B3, None, tuple(run_a), res))
    with pytest.raises(ValueError, match="contract"):
        bm.b3_verdict(other, b3)
    b2 = bm.RandomIndexResult(bm.BenchmarkRun(bm.B2, 1, tuple(run_a), res),
                              pd.DataFrame({"flag": []}))
    with pytest.raises(ValueError, match="contract"):
        bm.b2_verdict(other, b2)
    with pytest.raises(ValueError, match="contract"):
        _inputs(world, RsMom01, contract_name="competition")


def test_check_run_a_refuses_mismatched_spans_and_results(world, rs):
    inp, run_a = rs
    _, market, *_ = world
    res = inp.backtest(run_a)
    full = [(market.scan_start, market.end)]
    bm.check_run_a(inp, run_a, res, full)                     # consistent: no error
    short = [(pd.Timestamp("2024-03-04", tz="UTC"), pd.Timestamp("2024-03-07", tz="UTC"))]
    with pytest.raises(ValueError, match="spans do not reproduce run A"):
        bm.check_run_a(inp, run_a, res, short)
    with pytest.raises(ValueError, match="spans do not reproduce run A"):
        bm.run_benchmarks(inp, run_a, res, short, seeds=(1,), allow_partial=True)
    with pytest.raises(ValueError, match="differ from the"):
        bm.check_run_a(inp, run_a[1:], res, full)             # result of another candidate set
    part = inp.backtest(run_a[1:])
    with pytest.raises(ValueError, match="differ from the"):
        bm.check_run_a(inp, run_a, part, full)


def _proxy_factory(cfg, m15, h1):
    mk = SyntheticMarket(dict(m15), dict(h1), pd.Timestamp("2024-01-08", tz="UTC"),
                         pd.Timestamp("2024-04-13", tz="UTC"))
    return proxy_regime_tables(mk, cfg)


@pytest.mark.parametrize("sid", ["RS-MOM-01", "SES-IBX-01"])
def test_runner_run_a_total_equals_benchmark_backtest(world, sid):
    """Criterion 7 compares the runner's run-A total (bars trimmed to start - pad days) with
    the benchmarks' backtest path (full bars): the two must agree."""
    cfg, market, _, _, specs = world
    mk = R.market_from_frames(cfg, market.m15, market.h1, specs,
                              regime_factory=_proxy_factory, regime_tag="proxy-benchmarks")
    run = R.run_config(sid, None, "STRESS-d2", W, market=mk)
    scan = R.scan_config(mk, sid, None, W)
    inp = bm.BenchmarkInputs(mk.cfg, R.make_strategy(mk.cfg, sid, scan.params), mk.features,
                             mk.tables, mk.m15, mk.specs, ses_placebo=SHIFT)
    mine = inp.backtest(scan.candidates)
    assert len(scan.candidates) >= 10
    assert float(mine.filled["pnl_usd"].sum()) == pytest.approx(
        float(run.filled["pnl_usd"].sum()), abs=1e-9)
    assert len(mine.filled) == len(run.filled)
    bm.check_run_a(inp, scan.candidates, run.result, W_SPANS)   # runner spans reproduce run A
    c7 = bm.criterion7(run.result, bm.run_placebo(inp, scan.candidates, (1,)),
                       allow_partial=True)
    assert c7.run_a_total_usd == pytest.approx(float(run.filled["pnl_usd"].sum()))
