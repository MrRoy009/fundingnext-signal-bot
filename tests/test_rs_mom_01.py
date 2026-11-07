"""RS-MOM-01 (docs/specs/RS-MOM-01.md) on synthetic data only: every entry, stop, exit, lockout,
cap, window, protocol and reason-line rule, DST edges of the windows and time exits, the
look-ahead perturbation check and live generate() == batch scan()."""
from __future__ import annotations

import math
from datetime import date, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import strategies.rs_mom_01 as rs
from core.config import load_config
from strategies.base import MARKET_BAND, time_exit_label, time_exit_utc
from strategies.context import (LockoutBook, decision_times, window_groups, window_schedule)
from strategies.rs_mom_01 import (GER40_WINDOW, MAX_PER_INDEX_SIDE_DAY, PARAMS, US_WINDOW,
                                  RsMom01, reason_line, round_away, round_toward)
from strategies.selection import BUY, SELL, MemberInput, config_order, select_index
from tests.strategy_testkit import (ProxyRegimeTable, SyntheticMarket, assert_candidates_equal,
                                    assert_generate_matches_scan, assert_no_lookahead,
                                    build_services, candidate_tuple, make_market, sample_times,
                                    ts)

M15 = pd.Timedelta(minutes=15)
H1 = pd.Timedelta(hours=1)
TZ = {"US30": "America/New_York", "NAS100": "America/New_York", "GER40": "Europe/Berlin"}


@pytest.fixture(scope="module")
def mcfg():
    return load_config(Path(__file__).resolve().parent.parent / "config.yaml")


@pytest.fixture(scope="module")
def market():
    return make_market()


@pytest.fixture(scope="module")
def services(market, mcfg):
    return build_services(mcfg, market)


@pytest.fixture(scope="module")
def full(market, mcfg, services):
    """(cycles, candidates) at default parameters over the whole synthetic market."""
    cyc = list(RsMom01(mcfg).cycles(*services, market.scan_start, market.end))
    return cyc, [c for _, cs in cyc for c in cs]


@pytest.fixture(scope="module")
def loose(market, mcfg, services):
    """Grid value theta = 0.8: more candidates, so lockouts and caps bind more often."""
    cyc = list(RsMom01(mcfg, {"theta": 0.8}).cycles(*services, market.scan_start, market.end))
    return cyc, [c for _, cs in cyc for c in cs]


def local_date(c) -> date:
    return ts(c.signal_time).tz_convert(TZ[c.key]).date()


# ---------------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------------

def test_parameters_and_grid():
    assert dict(PARAMS.defaults) == {"theta": 1.0, "k": 1.0, "R": 1.5}
    assert PARAMS.grid["theta"] == (0.8, 0.9, 1.0, 1.1, 1.2)
    assert PARAMS.grid["k"] == (0.75, 0.875, 1.0, 1.125, 1.25)
    assert PARAMS.grid["R"] == (1.0, 1.25, 1.5, 1.75, 2.0)
    assert PARAMS.heatmap == ("k", "R")
    assert len(PARAMS.one_at_a_time()) == 13
    assert len(PARAMS.heatmap_cells()) == 25
    assert len(PARAMS.configurations()) == 29          # spec: 13 + 16 = 29


def test_params_resolve(mcfg):
    s = RsMom01(mcfg)
    assert s.id == "RS-MOM-01" and s.regime_gate == "ACTIVE" and s.rank == "rs_4h"
    assert dict(s.params) == {"theta": 1.0, "k": 1.0, "R": 1.5}
    assert RsMom01(mcfg, {"k": 1.25, "R": 2.0}).params["R"] == 2.0
    with pytest.raises(ValueError):
        RsMom01(mcfg, {"theta": 0.95})                  # off-grid on real data is refused
    with pytest.raises(ValueError):
        RsMom01(mcfg, {"L": 12})                        # L is frozen, not a parameter
    assert MAX_PER_INDEX_SIDE_DAY == 2


# ---------------------------------------------------------------------------------------------
# Windows and time exits across DST
# ---------------------------------------------------------------------------------------------

def _hm(t) -> str:
    return f"{t:%H:%M}"


@pytest.mark.parametrize("key,day,first,last", [
    ("US30", "2024-03-08", "15:30", "18:30"),     # Friday before US DST (EST)
    ("NAS100", "2024-03-11", "14:30", "17:30"),   # Monday after US DST (EDT), mismatch week
    ("US30", "2024-03-28", "14:30", "17:30"),     # last day of the March mismatch week
    ("US30", "2024-11-01", "14:30", "17:30"),     # November mismatch week (EDT)
    ("NAS100", "2024-11-04", "15:30", "18:30"),   # Monday after US DST end (EST)
    ("GER40", "2024-03-28", "09:00", "14:00"),    # last CET day before EU DST (Good Friday next)
    ("GER40", "2024-04-02", "08:00", "13:00"),    # first Xetra day in CEST (Easter Monday shut)
    ("GER40", "2024-10-25", "08:00", "13:00"),    # Friday before EU DST end (CEST)
    ("GER40", "2024-10-28", "09:00", "14:00"),    # Monday after EU DST end (CET)
])
def test_window_decision_times_dst(mcfg, key, day, first, last):
    w = US_WINDOW if key != "GER40" else GER40_WINDOW
    t = decision_times(mcfg, key, w, ts(day), ts(day) + pd.Timedelta(days=1))
    assert (_hm(t[0]), _hm(t[-1])) == (first, last)
    assert len(t) == (13 if key != "GER40" else 21)          # both ends inclusive, every M15
    assert (np.diff(t.asi8) == M15.value).all()


@pytest.mark.parametrize("key,day", [("US30", "2024-01-15"), ("NAS100", "2024-03-29"),
                                     ("US30", "2024-11-29"), ("GER40", "2024-04-01"),
                                     ("GER40", "2024-12-24"), ("US30", "2024-03-09")])
def test_no_window_on_holidays_early_closes_weekends(mcfg, key, day):
    w = US_WINDOW if key != "GER40" else GER40_WINDOW
    assert len(decision_times(mcfg, key, w, ts(day), ts(day) + pd.Timedelta(days=1))) == 0


def test_windows_inside_sessions_and_groups_never_mix(mcfg):
    """Spec: both windows lie inside config sessions_utc (start <= T < end) in both DST states,
    and G(T) is {US30, NAS100} or {GER40}, never both at one T (also in the mismatch weeks)."""
    s = RsMom01(mcfg)
    sched = window_schedule(mcfg, s.windows(s.params), ts("2023-01-01"), ts("2027-01-01"))
    assert set(sched["key"]) == {"US30", "NAS100", "GER40"}
    for key, rows in sched.groupby("key"):
        mins = pd.DatetimeIndex(rows["T"]).hour * 60 + pd.DatetimeIndex(rows["T"]).minute
        for a, b in mcfg["sessions_utc"][key]:
            lo = int(a[:2]) * 60 + int(a[3:])
            hi = int(b[:2]) * 60 + int(b[3:])
            assert ((mins >= lo) & (mins < hi)).all()
    shapes = set(sched.groupby("T")["key"].apply(tuple))
    assert shapes == {("US30", "NAS100"), ("GER40",)}
    assert window_groups(mcfg, s.windows(s.params), ts("2024-03-13 15:00")) == (
        ("US30", "NAS100"),)
    assert window_groups(mcfg, s.windows(s.params), ts("2024-03-13 13:00")) == (("GER40",),)


@pytest.mark.parametrize("key,T,exit_utc", [
    ("US30", "2024-03-08 18:30", "2024-03-08 20:45"),
    ("NAS100", "2024-03-11 14:30", "2024-03-11 19:45"),
    ("US30", "2024-11-01 17:30", "2024-11-01 19:45"),
    ("US30", "2024-11-04 15:30", "2024-11-04 20:45"),
    ("GER40", "2024-03-28 14:00", "2024-03-28 16:15"),
    ("GER40", "2024-04-02 08:00", "2024-04-02 15:15"),
    ("GER40", "2024-10-28 09:00", "2024-10-28 16:15"),
])
def test_time_exit_dst(mcfg, key, T, exit_utc):
    assert time_exit_utc(mcfg, key, ts(T)) == ts(exit_utc)
    # The window ends 2 h 15 min before the time exit in every DST state.
    assert ts(exit_utc) - ts(T) >= pd.Timedelta(hours=2, minutes=15)


# ---------------------------------------------------------------------------------------------
# Helpers: rounding and the reason line
# ---------------------------------------------------------------------------------------------

def test_rounding_directions():
    assert round_away(100.123, 101.0) == 100.12           # long SL below E: down
    assert round_away(100.121, 99.0) == 100.13            # short SL above E: up
    assert round_toward(100.129, 99.0) == 100.12          # long TP above E: down
    assert round_toward(100.121, 101.0) == 100.13         # short TP below E: up
    assert round_away(0.1 + 0.2, 1.0) == 0.3              # float noise never moves a cent
    assert round_toward(39000.12, 38000.0) == 39000.12
    assert round_away(39000.12, 40000.0) == 39000.12


def test_reason_line_format():
    # <theta> is the parameter value, unsigned on both sides; only the score carries a sign.
    line = reason_line(BUY, "US30", "leader", "{US30, NAS100}", 1.0349, 1.0,
                       ts("2024-03-11 19:45"), "15:45 New York")
    assert line == ("RS-MOM-01 LONG US30: leader of {US30, NAS100}, RS_4h +1.03 crossed 1.00, "
                    "regime ACTIVE. MANUAL TIME EXIT 19:45 UTC (15:45 New York) if still open.")
    line = reason_line(SELL, "GER40", "sole index", "{GER40}", -1.2, 0.9,
                       ts("2024-03-11 16:15"), "17:15 Berlin")
    assert line == ("RS-MOM-01 SHORT GER40: sole index of {GER40}, RS_4h -1.20 crossed 0.90, "
                    "regime ACTIVE. MANUAL TIME EXIT 16:15 UTC (17:15 Berlin) if still open.")


# ---------------------------------------------------------------------------------------------
# Freshness rule (Entry 1-2) on a stub view
# ---------------------------------------------------------------------------------------------

class _StubView:
    def __init__(self, cfg, now, prev):
        self.T = ts("2024-03-12 15:00")
        self.cfg = cfg
        self.selection = SimpleNamespace(scores={"US30": now})
        self._prev = prev
        self.asked = []

    def value_at(self, key, column, t):
        self.asked.append((key, column, ts(t)))
        return self._prev


@pytest.mark.parametrize("side,now,prev,fires", [
    (BUY, 1.0, 0.99, True),          # RS_4h(t) >= theta, inclusive
    (BUY, 1.3, 0.2, True),
    (BUY, 1.3, 1.0, False),          # previous bar already at theta: not fresh
    (BUY, 1.3, 1.1, False),
    (BUY, 0.999, 0.5, False),
    (BUY, 1.3, float("nan"), False),  # previous RS_4h NaN (or bar absent): no fire
    (SELL, -1.0, -0.99, True),
    (SELL, -1.5, 0.3, True),
    (SELL, -1.5, -1.0, False),       # previous bar already at -theta
    (SELL, -0.999, 0.0, False),
    (SELL, -1.5, float("nan"), False),
])
def test_freshness_rule(mcfg, side, now, prev, fires):
    v = _StubView(mcfg, now, prev)
    assert RsMom01(mcfg)._fires(v, "US30", side, 1.0) is fires
    assert v.asked == [("US30", "rs_4h", v.T - M15)]       # own previous bar, 15 min earlier


# ---------------------------------------------------------------------------------------------
# Every candidate against the rules, recomputed from the features
# ---------------------------------------------------------------------------------------------

def _rs_at(fx, t) -> float:
    p = fx.m15_pos(t)
    return float("nan") if p < 0 else float(fx.m15["rs_4h"].iat[p])


def _check_candidate(c, sel, cfg, fx, params):
    T = ts(c.signal_time)
    theta, k, R = params["theta"], params["k"], params["R"]
    sg = 1 if c.side == BUY else -1
    # Entry: selected index, fresh crossing of its own score.
    assert sel.allows(c.key, c.side)
    now, prev = _rs_at(fx, T), _rs_at(fx, T - M15)
    assert now == sel.scores[c.key]
    if c.side == BUY:
        assert now >= theta and prev < theta
    else:
        assert now <= -theta and prev > -theta
    # E = bid close of the bar closing at T; ATR14_M15 at t for information.
    row = fx.m15_at(T)
    assert c.entry == row["close"] and c.atr == row["atr14"] and math.isfinite(c.atr)
    # ATRH: ATR14_H1 of the latest H1 bar closed by T (never after T, at most 1 h old).
    h1c = pd.DatetimeIndex(fx.h1["close_utc"])
    j = int(np.searchsorted(h1c.asi8, T.value, side="right")) - 1
    assert j >= 0 and h1c[j] <= T and h1c[j] > T - H1
    if T.minute == 0:
        assert h1c[j] == T                         # full hour: the H1 bar closing at T
    atrh = float(fx.h1["atr14"].iat[j])
    assert c.stop == round_away(c.entry - sg * k * atrh, c.entry)
    assert c.take_profit == round_toward(c.entry + sg * R * k * atrh, c.entry)
    assert sg * (c.entry - c.stop) >= k * atrh - 1e-9                     # away from entry
    assert sg * (c.take_profit - c.entry) <= R * k * atrh + 1e-9          # toward entry
    assert abs(sg * (c.entry - c.stop) - k * atrh) < 0.01 + 1e-9
    assert abs(sg * (c.take_profit - c.entry) - R * k * atrh) < 0.01 + 1e-9
    # Protocol MARKET_BAND: chase 0.20 D, skip 0.50 D, sized on the chase limit; no EXPIRY.
    d = abs(c.entry - c.stop)
    pc = cfg["protocols"]
    assert c.order_type == MARKET_BAND and c.expires_utc is None
    assert math.isclose(c.chase_limit, c.entry + sg * pc["chase_frac"] * d, abs_tol=1e-9)
    assert math.isclose(c.skip_level, c.entry - sg * pc["skip_frac"] * d, abs_tol=1e-9)
    assert c.sizing_entry == c.chase_limit
    # Time exit: 15:45 New York / 17:15 Berlin on T's local date, >= 2 h 15 min after T.
    tz = TZ[c.key]
    ex = ts(c.time_exit_utc).tz_convert(tz)
    assert ex.date() == T.tz_convert(tz).date()
    assert ex.time() == time.fromisoformat(cfg["time_exits"][c.key])
    assert ts(c.time_exit_utc) - T >= pd.Timedelta(hours=2, minutes=15)
    # Regime fields (COMMON 6.7) and rank score (4.2.8).
    assert c.regime.startswith("ACTIVE/") and 0.0 <= c.regime_prob <= 1.0
    assert c.rank_score == round(now, 2)
    # Reason line.
    role = "sole index" if len(sel.eligible) == 1 else ("leader" if c.side == BUY else "laggard")
    assert c.reason == reason_line(c.side, c.key, role, sel.set_text(), now, theta,
                                   ts(c.time_exit_utc), time_exit_label(cfg, c.key))
    assert ("LONG" if c.side == BUY else "SHORT") in c.reason
    assert f" crossed {theta:.2f}, " in c.reason                 # unsigned <theta>


@pytest.mark.parametrize("which", ["full", "loose"])
def test_every_candidate_follows_the_rules(request, mcfg, services, which):
    cyc, cands = request.getfixturevalue(which)
    params = PARAMS.resolve({} if which == "full" else {"theta": 0.8})
    assert len(cands) >= 20, "the strategy must fire on the synthetic market"
    assert {c.key for c in cands} == {"US30", "NAS100", "GER40"}
    assert {c.side for c in cands} == {BUY, SELL}
    svc = services[0]
    for sel, cs in cyc:
        for c in cs:
            _check_candidate(c, sel, mcfg, svc.get(c.key), params)
    # Both roles and both group shapes occur.
    reasons = " ".join(c.reason for c in cands)
    assert "leader of {US30, NAS100}" in reasons and "sole index of {GER40}" in reasons


@pytest.mark.parametrize("which", ["full", "loose"])
def test_no_missed_fire(request, mcfg, services, which):
    """Exhaustive: at every cycle, a candidate exists exactly when the selected index crosses
    freshly and is neither occupied nor capped (lockouts replayed independently)."""
    cyc, _ = request.getfixturevalue(which)
    theta = 1.0 if which == "full" else 0.8
    svc = services[0]
    book = LockoutBook("REF", {k: svc.get(k).m15 for k in ("US30", "NAS100", "GER40")})
    n_locked = 0
    for sel, cs in cyc:
        expected = []
        for side, key in ((BUY, sel.long), (SELL, sel.short)):
            if key is None:
                continue
            fx = svc.get(key)
            now, prev = _rs_at(fx, sel.T), _rs_at(fx, sel.T - M15)
            fresh = (now >= theta and prev < theta) if side == BUY else \
                (now <= -theta and prev > -theta)
            if not fresh:
                continue
            d = sel.T.tz_convert(TZ[key]).date()
            if book.occupied(key, sel.T) or book.count(key, d, side) >= 2:
                n_locked += 1
                continue
            expected.append((key, side))
        order = config_order(mcfg)                  # config instrument order within one T
        expected.sort(key=lambda ks: order.index(ks[0]))
        assert [(c.key, c.side) for c in cs] == expected, sel.log_line()
        for c in cs:
            book.add(c, local_date(c))
    if which == "loose":
        assert n_locked > 0                    # the lockouts actually bind in this data


def test_caps_and_occupancy(mcfg, services, loose):
    _, cands = loose
    per_day = pd.Series([(c.key, c.side, local_date(c)) for c in cands]).value_counts()
    assert per_day.max() <= 2
    svc = services[0]
    book = LockoutBook("X", {k: svc.get(k).m15 for k in ("US30", "NAS100", "GER40")})
    for c in cands:
        assert not book.occupied(c.key, c.signal_time)
        book.add(c, local_date(c))


def test_occupancy_and_cap_bind(mcfg, market, services, monkeypatch):
    """Removing the occupancy lockout adds candidates (it binds); a cap of 1 removes some."""
    class NoOccupancy(RsMom01):
        def _locked(self, view, key, side):
            return view.book.count(key, view.local_date[key], side) >= rs.MAX_PER_INDEX_SIDE_DAY

    base = RsMom01(mcfg, {"theta": 0.8}).scan(*services, market.scan_start, market.end)
    free = NoOccupancy(mcfg, {"theta": 0.8}).scan(*services, market.scan_start, market.end)
    assert len(free) > len(base)
    per_day = pd.Series([(c.key, c.side, local_date(c)) for c in free]).value_counts()
    assert per_day.max() == 2                                    # cap of 2 still holds
    monkeypatch.setattr(rs, "MAX_PER_INDEX_SIDE_DAY", 1)
    capped = NoOccupancy(mcfg, {"theta": 0.8}).scan(*services, market.scan_start, market.end)
    assert len(capped) < len(free)
    assert pd.Series([(c.key, c.side, local_date(c)) for c in capped]).value_counts().max() == 1


def test_parameters_move_levels_not_signals(mcfg, market, services, full):
    """k and R change only SL / TP (and the protocol levels), never which candidates fire...
    except through the occupancy lockout, which depends on SL / TP. Compare the first
    candidate of each index, which no earlier candidate can lock out."""
    _, base = full
    other = RsMom01(mcfg, {"k": 1.25, "R": 2.0}).scan(*services, market.scan_start, market.end)
    firsts = {}
    for c in base:
        firsts.setdefault(c.key, c)
    for key, c in firsts.items():
        o = next(x for x in other if x.key == key)
        assert (o.signal_time, o.side, o.entry) == (c.signal_time, c.side, c.entry)
        assert abs(o.entry - o.stop) > abs(c.entry - c.stop)
    # theta moves the trigger. (The count need not fall: a higher theta also changes which
    # crossing is fresh and which candidates lock others out.)
    higher = RsMom01(mcfg, {"theta": 1.2}).scan(*services, market.scan_start, market.end)
    assert higher and all(abs(c.rank_score) >= 1.2 - 0.005 for c in higher)
    assert any(abs(c.rank_score) < 1.2 for c in base)


# ---------------------------------------------------------------------------------------------
# Data completeness and regime gate
# ---------------------------------------------------------------------------------------------

def test_missing_h1_bar_makes_group_incomplete(mcfg, market):
    """ATRH unavailable under the COMMON 2 H1 rule -> incomplete group (COMMON 4.2.2)."""
    h1 = dict(market.h1)
    h1["US30"] = h1["US30"][h1["US30"]["time_utc"] != ts("2024-03-12 15:00")]
    m = SyntheticMarket(dict(market.m15), h1, market.scan_start, market.end, dict(market.meta))
    sels = {s.T: s for s, _ in RsMom01(mcfg).cycles(*build_services(mcfg, m),
                                                    ts("2024-03-12"), ts("2024-03-13"))
            if "US30" in s.members}
    for t in ("16:00", "16:15", "16:30", "16:45"):
        s = sels[ts(f"2024-03-12 {t}")]
        assert "incomplete group: US30: ATR14_H1 unavailable: H1 bar missing" in s.incomplete
        assert not s.has_side
    for t in ("15:45", "17:00"):
        assert "ATR14_H1" not in (sels[ts(f"2024-03-12 {t}")].incomplete or "")


@pytest.mark.parametrize("value", [0.0, float("nan"), -1.0])
def test_atrh_nan_or_not_positive_makes_group_incomplete(mcfg, market, full, value):
    """Spec Stop 'if unavailable, no candidate' + SPEC NOTE 'ATRH not > 0': ATR14_H1 NaN,
    zero or negative on the (aligned) H1 bar a known candidate uses -> incomplete group,
    no side, no candidate at that T."""
    _, cands = full
    c = next(x for x in cands if x.key == "US30")
    T = ts(c.signal_time)
    svc, tabs = build_services(mcfg, market)
    fx = svc.get(c.key)
    hc = pd.DatetimeIndex(fx.h1["close_utc"])
    j = int(np.searchsorted(hc.asi8, T.value, side="right")) - 1
    fx.h1.loc[fx.h1.index[j], "atr14"] = value
    cyc = list(RsMom01(mcfg).cycles(svc, tabs, T, T + M15))
    s, cs = next((s, cs) for s, cs in cyc if s.T == T)
    assert s.incomplete and ("incomplete group: US30: ATR14_H1 unavailable: "
                             "ATR14_H1 NaN or not positive") in s.incomplete
    assert not s.has_side and cs == []


def test_atrh_unavailable_without_incomplete_group_raises(mcfg, market, full, monkeypatch):
    """No silent fallback (PROMPT): if a driver change ever let a group with an unavailable
    ATRH reach decide(), the fresh selected crossing must fail loudly, not vanish as 'no
    setup'."""
    _, cands = full
    c = next(x for x in cands if x.key == "US30")
    T = ts(c.signal_time)
    svc, tabs = build_services(mcfg, market)
    fx = svc.get(c.key)
    j = int(np.searchsorted(pd.DatetimeIndex(fx.h1["close_utc"]).asi8, T.value,
                            side="right")) - 1
    fx.h1.loc[fx.h1.index[j], "atr14"] = float("nan")
    monkeypatch.setattr(RsMom01, "incomplete_cause", lambda self, view, key: None)
    with pytest.raises(AssertionError, match="ATRH unavailable for selected US30"):
        list(RsMom01(mcfg).cycles(svc, tabs, T, T + M15))


def test_atrh_h1_bar_from_earlier_segment(mcfg, services):
    """COMMON 1.4 via the COMMON 2 H1 rule: at the first M15 close after GER40's data hole the
    latest H1 bar closed before the hole, so ATRH is unavailable with that cause."""
    fx = services[0].get("GER40")
    seg = fx.m15["segment"].to_numpy()
    p = int(np.flatnonzero(seg[1:] != seg[:-1])[0]) + 1          # first bar after the hole
    assert ts(fx.m15["time_utc"].iat[p]) >= ts("2024-03-05")
    view = SimpleNamespace(columns={}, fx={"GER40": fx}, pos={"GER40": p})
    s = RsMom01(mcfg)
    v, cause = s._atrh(view, "GER40")
    assert math.isnan(v) and cause.startswith("H1 bar from earlier segment")
    assert s.incomplete_cause(view, "GER40") == f"ATR14_H1 unavailable: {cause}"


def test_same_t_candidates_in_config_order(mcfg):
    """Strategy.scan contract: candidates at one T in config instrument order, not side
    order. LONG-SELECTED NAS100 and SHORT-SELECTED US30 both fire: US30 comes first."""
    T = ts("2024-03-12 15:00")
    sel = select_index(T, [MemberInput("US30", -1.2), MemberInput("NAS100", 1.3)],
                       config_order(mcfg))
    assert (sel.long, sel.short) == ("NAS100", "US30")

    class Probe(RsMom01):
        def _fires(self, view, key, side, theta):
            return True

        def _locked(self, view, key, side):
            return False

        def _atrh(self, view, key):
            return 10.0, ""

        def _candidate(self, view, key, side, theta, k, R, atrh):
            return SimpleNamespace(key=key, side=side)

    s = Probe(mcfg)
    out = s.decide(SimpleNamespace(selection=sel, params=s.params, cfg=mcfg))
    assert [(c.key, c.side) for c in out] == [("US30", SELL), ("NAS100", BUY)]
    assert config_order(mcfg).index("US30") < config_order(mcfg).index("NAS100")


def test_incomplete_group_blocks_both_us_indices(full):
    """A missing NAS100 bar (synthetic defect 2024-02-22 15:30 UTC) blocks US30 too, while
    the bar is missing and while it sits inside NAS100's 16-bar RS_4h lookback."""
    cyc, cands = full
    sels = {s.T: s for s, _ in cyc if "NAS100" in s.members}
    s = sels[ts("2024-02-22 15:45")]
    assert s.incomplete == "incomplete group: NAS100: no M15 bar closing at T"
    assert not s.has_side
    s = sels[ts("2024-02-22 16:00")]
    assert s.incomplete.startswith("incomplete group: NAS100: RS_4h NaN") and not s.has_side
    blocked = [t for t, s in sels.items() if t.date() == date(2024, 2, 22) and s.incomplete]
    assert not [c for c in cands if c.key in ("US30", "NAS100") and ts(c.signal_time) in blocked]


def test_missing_cash_open_bar_excludes_index(mcfg, market, full):
    """US30's cash-open bar is missing on 2024-03-05 (synthetic defect): not a valid trading
    date for US30, so US30 is outside G(T) (COMMON 4.2.1 + 1.6, Selection.excluded) at all 13
    US decision times and NAS100 is ranked alone under the sign rule (4.2.5). A NAS100
    candidate that day is the 'sole index of {NAS100}'. No US30 candidate that day."""
    from strategies.context import INVALID_DATE
    cyc, cands = full
    day = [(s, cs) for s, cs in cyc if s.T.date() == date(2024, 3, 5)
           and ("US30" in s.members or "NAS100" in s.members)]
    assert len(day) == 13
    for s, cs in day:
        assert s.members == ("NAS100",) and dict(s.excluded) == {"US30": INVALID_DATE}
        assert f"outside G(T): US30: {INVALID_DATE}" in s.log_line()
        assert all(c.key == "NAS100" for c in cs)
    got = [c for s, cs in day for c in cs]
    assert got, "the synthetic market fires NAS100 alone on 2024-03-05"
    for c in got:
        assert "sole index of {NAS100}" in c.reason
        assert (c.side == BUY) == (c.rank_score > 0)
    # Live at those T equals the batch scan (the exclusion uses only bars closed by T).
    assert_generate_matches_scan(RsMom01(mcfg), mcfg, market,
                                 sorted({ts(c.signal_time) for c in got})
                                 + [ts("2024-03-05 15:30")], full=cands)


def test_no_ger40_cycle_inside_its_data_hole(full):
    cyc, _ = full
    assert not [s for s, _ in cyc if "GER40" in s.members
                and ts("2024-02-26") <= s.T < ts("2024-03-05")]
    s = next(s for s, _ in cyc if s.T == ts("2024-03-06 10:00"))
    assert s.members == ("GER40",) and s.incomplete.startswith("incomplete group: GER40")


def test_regime_h1_bar_missing_live_rule(mcfg):
    """An H1 bar absent while its M15 bars exist (H1 feed lag): at the T it should close,
    both ATRH (COMMON 2) and the regime (6.6 live rule) are unavailable; the group is
    incomplete for US30 and NAS100 at T, and generate() == scan() there and one hour later."""
    mk = make_market("2024-01-08", "2024-02-24", drops={}, holes={},
                     h1_drops={"US30": ["2024-02-21 16:00"]})
    strat = RsMom01(mcfg, {"theta": 0.8})
    svc = build_services(mcfg, mk)
    sels = {s.T: s for s, _ in strat.cycles(*svc, ts("2024-02-21"), ts("2024-02-22"))
            if "US30" in s.members}
    s = sels[ts("2024-02-21 17:00")]
    assert s.incomplete and "incomplete group: US30: ATR14_H1 unavailable" in s.incomplete
    assert not s.has_side
    assert sels[ts("2024-02-21 16:45")].incomplete is None
    assert_generate_matches_scan(strat, mcfg, mk, [ts("2024-02-21 17:00"),
                                                   ts("2024-02-21 18:00")],
                                 start=mk.scan_start)


def test_scan_selection_sink(mcfg, market, services, full):
    """COMMON 4.2.8: scan(on_selection=...) receives every cycle's Selection in order, so a
    backtest can persist every member's rank score; candidates are unchanged."""
    cyc, cands = full
    got = []
    out = RsMom01(mcfg).scan(*services, market.scan_start, market.end, on_selection=got.append)
    assert_candidates_equal(out, cands, "scan with a selection sink")
    assert [s.log_line() for s in got] == [s.log_line() for s, _ in cyc]
    line = next(s for s, cs in cyc if cs).log_line()
    assert line.startswith("RS-MOM-01 ") and "LONG-SELECTED" in line


def test_regime_unavailable_and_gate(mcfg, market, full):
    _, cands = full
    target = next(c for c in cands if c.key == "GER40")
    d = ts(target.signal_time).normalize()

    def tables(mk, cfg):
        thr = float(cfg["regime"]["gate_threshold"])
        out = {k: ProxyRegimeTable(k, mk.h1[k], thr) for k in mk.keys}
        out["GER40"] = ProxyRegimeTable("GER40", mk.h1["GER40"], thr,
                                        unavailable=((d, d + pd.Timedelta(days=1)),))
        return out

    svc, tabs = build_services(mcfg, market, tables)
    cyc = list(RsMom01(mcfg).cycles(svc, tabs, d, d + pd.Timedelta(days=1)))
    ger = [s for s, _ in cyc if s.members == ("GER40",) and s.incomplete is None]
    assert ger and all(s.reasons["GER40"] == "regime unavailable: test outage" for s in ger)
    assert not any(c.key == "GER40" for _, cs in cyc for c in cs)
    # Every default candidate passed the ACTIVE gate: V_T >= 0 at T.
    full_tabs = build_services(mcfg, market)[1]
    for c in cands:
        r = full_tabs[c.key].at_many([c.signal_time]).iloc[0]
        assert r["status"] == "OK" and r["v_t"] >= 0


def test_freshness_needs_previous_bar(mcfg, market, full):
    """Mutating only RS_4h(t-1) of a fired candidate removes it: NaN or already past theta."""
    _, cands = full
    c = cands[len(cands) // 2]
    for value in (float("nan"), 1.05 if c.side == BUY else -1.05):
        svc, tabs = build_services(mcfg, market)
        fx = svc.get(c.key)
        p = fx.m15_pos(ts(c.signal_time) - M15)
        fx.m15.loc[fx.m15.index[p], "rs_4h"] = value
        got = RsMom01(mcfg).scan(svc, tabs, ts(c.signal_time).normalize(),
                                 ts(c.signal_time) + M15)
        assert not any((x.key, x.side, x.signal_time) == (c.key, c.side, c.signal_time)
                       for x in got)


def test_generate_without_context_returns_nothing(mcfg):
    from strategies.base import StrategyContext
    T = ts("2024-03-12 15:00")
    assert RsMom01(mcfg).generate({}, {}, T, StrategyContext(now=T, cfg=mcfg)) == []


# ---------------------------------------------------------------------------------------------
# Look-ahead and live == batch (testkit)
# ---------------------------------------------------------------------------------------------

def test_no_lookahead(mcfg, market):
    """Perturbing every M15 / H1 bar after the cut leaves candidates up to the cut unchanged:
    one cut in the March mismatch week (New York EDT, Berlin CET), one after EU DST."""
    full = assert_no_lookahead(RsMom01(mcfg, {"theta": 0.8}), mcfg, market,
                               cuts=["2024-03-13 16:00", "2024-04-03 09:30"],
                               end=ts("2024-04-06"), seeds=(1,))
    assert any(ts(c.signal_time) <= ts("2024-03-13 16:00") for c in full)
    assert any(ts("2024-03-11") <= ts(c.signal_time) <= ts("2024-03-13 16:00") for c in full)


def test_generate_matches_scan(mcfg, market, services, full):
    strat = RsMom01(mcfg)
    _, cands = full
    start = market.scan_start + pd.Timedelta(days=2)
    times = [t for t in sample_times(strat, mcfg, cands, start, market.end, n_candidates=6,
                                     n_other=3) if t >= start]
    # A second candidate on the same index and local date (lockout / cap replay).
    same = [b for a, b in zip(cands, cands[1:]) if a.key == b.key
            and local_date(a) == local_date(b)]
    times += [ts(same[0].signal_time)] if same else []
    # DST edges: first window T after US DST, mismatch week, first CEST window T.
    times += [ts("2024-03-11 14:30"), ts("2024-03-13 15:00"), ts("2024-04-02 08:00"),
              ts("2024-03-08 18:30")]
    times += [ts(c.signal_time) for c in cands
              if ts("2024-03-11") <= ts(c.signal_time) < ts("2024-04-06")][:3]
    n = assert_generate_matches_scan(strat, mcfg, market, sorted(set(times)), full=cands)
    assert n >= 6


@pytest.fixture(scope="module")
def autumn():
    """Defect-free market across the November 2024 mismatch week (EU on CET from 2024-10-27,
    US on EST from 2024-11-03)."""
    return make_market("2024-09-23", "2024-11-09", drops={}, holes={})


def test_november_mismatch_week(mcfg, autumn):
    strat = RsMom01(mcfg, {"theta": 0.8})
    full = assert_no_lookahead(strat, mcfg, autumn, cuts=["2024-10-30 16:00"],
                               start=autumn.scan_start, seeds=(2,))
    svc = build_services(mcfg, autumn)
    cyc = list(strat.cycles(*svc, autumn.scan_start, autumn.end))
    for sel, cs in cyc:
        for c in cs:
            _check_candidate(c, sel, mcfg, svc[0].get(c.key), strat.params)
    assert len({s.T for s, _ in cyc}) == len(cyc)          # never two groups at one T
    week = [c for c in full if ts("2024-10-28") <= ts(c.signal_time) < ts("2024-11-09")]
    times = [ts("2024-10-28 09:00"), ts("2024-10-28 14:30"), ts("2024-11-01 17:30"),
             ts("2024-11-04 15:30"), ts("2024-11-04 13:00")]
    times += [ts(c.signal_time) for c in week[:4]]
    assert_generate_matches_scan(strat, mcfg, autumn, sorted(set(times)), full=full)


def _h1_perturbed_after(market: SyntheticMarket, cut, seed: int,
                        drop_frac: float = 0.2) -> SyntheticMarket:
    """Only H1 bars with close > cut change (range scaled around the mid, close moved inside
    the new range, about `drop_frac` dropped); every M15 bar stays. This is the case
    SyntheticMarket.perturbed_after cannot show, because it rebuilds H1 from the perturbed
    M15: an ATRH (or regime input) read from an H1 bar that closes after T, including the
    bar still open at T, would change while every M15 input stays the same."""
    cut = ts(cut)
    rng = np.random.default_rng(seed)
    h = {}
    for k, df in market.h1.items():
        after = (df["time_utc"] + H1 > cut).to_numpy()
        out = df.copy()
        n = int(after.sum())
        if n:
            rows = out.loc[after]
            mid = (rows["high"] + rows["low"]) / 2
            half = (rows["high"] - rows["low"]) / 2 * rng.uniform(0.2, 4.0, n)
            out.loc[after, "high"] = mid + half
            out.loc[after, "low"] = mid - half
            out.loc[after, "open"] = mid + half * rng.uniform(-1, 1, n)
            out.loc[after, "close"] = mid + half * rng.uniform(-1, 1, n)
            keep = ~after | (rng.uniform(0, 1, len(out)) >= drop_frac)
            out = out[keep].reset_index(drop=True)
        h[k] = out
    return SyntheticMarket(dict(market.m15), h, market.scan_start, market.end,
                           dict(market.meta))


@pytest.mark.parametrize("which", ["march", "november"])
def test_no_lookahead_h1_only(request, mcfg, which):
    """ER-15 / spec look-ahead safeguards: perturbing or dropping only H1 bars that close
    after the cut (M15 unchanged) leaves every candidate with signal_time <= cut unchanged.
    Cuts at a candidate T and at T - 15 min (when the H1 bar holding T is still open), in
    the March (US EDT, EU CET) and November (EU CET, US EDT) mismatch weeks."""
    if which == "march":
        mkt, week = request.getfixturevalue("market"), ("2024-03-11", "2024-03-30")
    else:
        mkt, week = request.getfixturevalue("autumn"), ("2024-10-28", "2024-11-02")
    strat = RsMom01(mcfg, {"theta": 0.8})
    full = strat.scan(*build_services(mcfg, mkt), mkt.scan_start, mkt.end)
    inweek = [c for c in full if ts(week[0]) <= ts(c.signal_time) < ts(week[1])]
    assert inweek, f"no candidate in the {which} mismatch week"
    T = ts(inweek[0].signal_time)
    for cut, seed in ((T, 3), (T - M15, 4)):
        pert = _h1_perturbed_after(mkt, cut, seed)
        got = strat.scan(*build_services(mcfg, pert), mkt.scan_start, mkt.end)
        assert_candidates_equal([c for c in got if ts(c.signal_time) <= cut],
                                [c for c in full if ts(c.signal_time) <= cut],
                                f"after perturbing H1 bars closing after {cut}")
        # The perturbation has teeth: later candidates (ATRH levels, regime gate) do change.
        assert [candidate_tuple(c) for c in got] != [candidate_tuple(c) for c in full]


def test_scan_from_mid_session_matches_full(mcfg, market, services, loose):
    _, cands = loose
    strat = RsMom01(mcfg, {"theta": 0.8})
    cuts = [ts(a.signal_time) + M15 for a, b in zip(cands, cands[1:])
            if a.key == b.key and local_date(a) == local_date(b)][:3]
    assert cuts
    for cut in cuts:
        part = strat.scan(*services, cut, market.end)
        assert_candidates_equal(part, [c for c in cands if ts(c.signal_time) >= cut],
                                f"scan from {cut}")
