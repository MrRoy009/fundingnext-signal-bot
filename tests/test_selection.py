"""COMMON 4.2 index selection, signal windows, parameters, lockouts and the scan/generate driver.

Synthetic data only. The toy strategies below exist to exercise the shared plumbing; they are not
any spec's strategy.
"""
from __future__ import annotations

import logging
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import strategies.context as ctx_mod
from core.config import load_config
from strategies.base import MARKET_BAND, MARKET_OR_LIMIT, Candidate, ParamGrid, StrategyContext
from strategies.context import (INVALID_DATE, NO_BAR, LocalWindow, LockoutBook,
                                WindowedStrategy, decision_times, in_window, regime_frame,
                                replay_start, window_group, window_groups, window_schedule,
                                window_set)
from strategies.selection import (BUY, SELL, TIE_TOL, MemberInput, config_order,
                                  select_index)
from tests.strategy_testkit import (ProxyRegimeTable, assert_candidates_equal,
                                    assert_generate_matches_scan, assert_no_lookahead,
                                    build_services, make_market, sample_times, ts)

ORDER = ("US30", "NAS100", "GER40")
T0 = ts("2024-03-12 15:00")


def sel(*members, order=ORDER):
    return select_index(T0, list(members), order, "TEST")


# ---------------------------------------------------------------------------------------------
# select_index: COMMON 4.2 steps 2-8
# ---------------------------------------------------------------------------------------------

def test_config_order_excludes_untradeable(cfg):
    assert config_order(cfg) == ORDER          # US500 is tradeable: false (O6)


def test_untradeable_member_is_refused():
    with pytest.raises(ValueError, match="not a ranked instrument"):
        sel(MemberInput("US30", 1.0), MemberInput("US500", 2.0))


def test_duplicate_member_is_refused():
    with pytest.raises(ValueError, match="duplicate"):
        sel(MemberInput("US30", 1.0), MemberInput("US30", 2.0))


def test_argmax_long_argmin_short():
    s = sel(MemberInput("NAS100", -0.4), MemberInput("US30", 1.3))
    assert s.members == ("US30", "NAS100")      # config order, whatever the input order
    assert s.eligible == ("US30", "NAS100")
    assert (s.long, s.short) == ("US30", "NAS100")
    assert not s.sole and s.incomplete is None


def test_two_members_no_sign_condition():
    """With two passing indices the leader may get a BUY although both scores are negative."""
    s = sel(MemberInput("US30", -0.2), MemberInput("NAS100", -1.5))
    assert (s.long, s.short) == ("US30", "NAS100")


def test_tie_within_tolerance_goes_to_config_order():
    a = 0.75
    s = sel(MemberInput("US30", a), MemberInput("NAS100", a + 0.5 * TIE_TOL))
    # Earlier index is higher for the long AND lower for the short (literal 4.2.4).
    assert (s.long, s.short) == ("US30", "US30")
    s = sel(MemberInput("NAS100", a), MemberInput("US30", a + 0.5 * TIE_TOL))
    assert (s.long, s.short) == ("US30", "US30")


def test_difference_beyond_tolerance_is_not_a_tie():
    s = sel(MemberInput("US30", 0.75), MemberInput("NAS100", 0.75 + 3 * TIE_TOL))
    assert (s.long, s.short) == ("NAS100", "US30")


def test_tie_against_extreme_with_three_members():
    s = sel(MemberInput("US30", 1.0), MemberInput("NAS100", 1.0 + 0.8e-12),
            MemberInput("GER40", 1.0 + 1.6e-12))
    # Within 1e-12 of the max (GER40): NAS100 and GER40; NAS100 is earlier in config order.
    assert s.long == "NAS100"
    assert s.short == "US30"


def test_incomplete_member_blocks_whole_group():
    s = sel(MemberInput("US30", 2.0), MemberInput("NAS100", 0.1, incomplete="no M15 bar closing at T"))
    assert s.incomplete == "incomplete group: NAS100: no M15 bar closing at T"
    assert s.long is None and s.short is None and not s.has_side
    assert s.eligible == ()


def test_nan_score_is_incomplete_even_without_cause():
    s = sel(MemberInput("US30", float("nan")), MemberInput("NAS100", 0.5))
    assert s.incomplete == "incomplete group: US30: rank score NaN"
    assert not s.has_side


def test_several_incomplete_members_all_logged():
    s = sel(MemberInput("US30", 1.0, incomplete="a"), MemberInput("NAS100", 1.0, incomplete="b"))
    assert s.incomplete == "incomplete group: US30: a; incomplete group: NAS100: b"


def test_completeness_is_checked_before_the_regime_filter():
    s = sel(MemberInput("US30", 2.0),
            MemberInput("NAS100", 0.0, incomplete="x", regime_ok=False, regime_cause="r"))
    assert s.incomplete == "incomplete group: NAS100: x" and not s.has_side


def test_regime_filter_then_rank():
    """O7 default: the strongest index that passes the filter, not the strongest overall."""
    s = sel(MemberInput("US30", 2.0, gate_pass=False, gate_cause="regime gate ACTIVE not passed"),
            MemberInput("NAS100", 0.4))
    assert s.eligible == ("NAS100",) and s.sole
    assert (s.long, s.short) == ("NAS100", None)
    assert s.reasons == {"US30": "regime gate ACTIVE not passed"}


@pytest.mark.parametrize("score, long, short", [(0.3, "GER40", None), (-0.3, None, "GER40"),
                                                (0.0, None, None)])
def test_single_index_sign_rule(score, long, short):
    s = sel(MemberInput("GER40", score))
    assert (s.long, s.short) == (long, short)
    assert s.sole
    if score == 0:
        assert "rank score 0" in s.note


def test_single_index_after_regime_filter_uses_sign_rule():
    s = sel(MemberInput("US30", 1.0, regime_ok=False, regime_cause="regime unavailable: stale"),
            MemberInput("NAS100", -0.7))
    assert (s.long, s.short) == (None, "NAS100")


def test_empty_eligible_set_logs_reason_per_index():
    s = sel(MemberInput("US30", 1.0, regime_ok=False, regime_cause="regime unavailable: stale"),
            MemberInput("NAS100", -0.7, gate_pass=False, gate_cause="regime gate ACTIVE not passed"))
    assert s.eligible == () and not s.has_side and s.incomplete is None
    assert s.reasons == {"US30": "regime unavailable: stale",
                         "NAS100": "regime gate ACTIVE not passed"}
    line = s.log_line()
    assert "E(T) empty" in line and "regime unavailable: stale" in line


def test_never_promote_next_index():
    s = sel(MemberInput("US30", 1.5), MemberInput("NAS100", 0.2))
    assert s.allows("US30", BUY) and not s.allows("NAS100", BUY)
    assert s.allows("NAS100", SELL) and not s.allows("US30", SELL)
    with pytest.raises(ValueError):
        s.allows("US30", "LONG")


def test_rank_score_rounded_to_two_dp():
    s = sel(MemberInput("US30", 1.23678), MemberInput("NAS100", -0.40412))
    assert s.rank_score("US30") == 1.24 and s.rank_score("NAS100") == -0.40


def test_log_line_has_every_member_score():
    s = sel(MemberInput("US30", 1.23456), MemberInput("NAS100", -0.5,
                                                      gate_pass=False, gate_cause="g"))
    line = s.log_line()
    assert "US30 +1.2346" in line and "NAS100 -0.5000" in line
    assert "LONG-SELECTED US30" in line and "SHORT-SELECTED -" in line   # sole, score > 0
    assert "not eligible: NAS100: g" in line
    inc = sel(MemberInput("US30", 1.0), MemberInput("NAS100", float("nan"), incomplete="c"))
    assert "NAS100 nan" in inc.log_line() and "incomplete group: NAS100: c" in inc.log_line()


def test_empty_window_set():
    s = sel()
    assert s.members == () and not s.has_side


# ---------------------------------------------------------------------------------------------
# Windows and decision times (COMMON 2, 4.2.1)
# ---------------------------------------------------------------------------------------------

US_W = LocalWindow.of("10:30", "13:30")
GER_W = LocalWindow.of("10:00", "15:00")


@pytest.mark.parametrize("key, w, t, expect", [
    # US winter (EST): 10:30 New York = 15:30 UTC; both ends inclusive.
    ("US30", US_W, "2024-03-08 15:15", False),
    ("US30", US_W, "2024-03-08 15:30", True),
    ("US30", US_W, "2024-03-08 18:30", True),
    ("US30", US_W, "2024-03-08 18:45", False),
    # US DST from 2024-03-10 (mismatch weeks: Berlin still on CET until 2024-03-31).
    ("NAS100", US_W, "2024-03-11 14:30", True),
    ("NAS100", US_W, "2024-03-11 17:30", True),
    ("NAS100", US_W, "2024-03-11 18:30", False),
    # Berlin: 10:00 = 09:00 UTC (CET), 08:00 UTC (CEST from 2024-03-31).
    ("GER40", GER_W, "2024-03-28 09:00", True),
    ("GER40", GER_W, "2024-03-28 08:00", False),
    ("GER40", GER_W, "2024-04-02 08:00", True),
    ("GER40", GER_W, "2024-04-02 13:00", True),
    ("GER40", GER_W, "2024-04-02 13:15", False),
    # Holidays and early closes are not valid trading dates.
    ("US30", US_W, "2024-03-29 15:30", False),      # Good Friday (NYSE)
    ("GER40", GER_W, "2024-03-29 09:00", False),    # Good Friday (Xetra)
    ("GER40", GER_W, "2024-04-01 08:00", False),    # Easter Monday (Xetra)
    ("US30", US_W, "2024-04-01 14:30", True),       # NYSE open on Easter Monday
    ("US30", US_W, "2024-07-03 14:30", False),      # NYSE early close
    ("US30", US_W, "2024-03-09 15:30", False),      # Saturday
    # Not an M15 close.
    ("US30", US_W, "2024-03-08 15:37", False),
])
def test_in_window(cfg, key, w, t, expect):
    assert in_window(cfg, key, w, ts(t)) is expect


def test_untradeable_never_in_window(cfg):
    assert not in_window(cfg, "US500", US_W, ts("2024-03-08 15:30"))
    assert len(decision_times(cfg, "US500", US_W, ts("2024-03-04"), ts("2024-03-09"))) == 0


def test_decision_times_follow_dst_and_calendar(cfg):
    t = decision_times(cfg, "US30", US_W, ts("2024-03-07"), ts("2024-03-12"))
    by_day = pd.Series(t).groupby(pd.Series(t).dt.date).agg(["first", "last", "size"])
    assert list(by_day["size"]) == [13, 13, 13]                         # Thu, Fri, Mon
    assert by_day.loc[date(2024, 3, 8), "first"] == ts("2024-03-08 15:30")
    assert by_day.loc[date(2024, 3, 11), "first"] == ts("2024-03-11 14:30")
    assert by_day.loc[date(2024, 3, 11), "last"] == ts("2024-03-11 17:30")
    # Good Friday and Easter Monday for GER40; [start, end) is half-open on T.
    g = decision_times(cfg, "GER40", GER_W, ts("2024-03-28"), ts("2024-04-03"))
    assert sorted({x.date() for x in g}) == [date(2024, 3, 28), date(2024, 4, 2)]
    assert len(decision_times(cfg, "GER40", GER_W, ts("2024-04-02 08:00"),
                              ts("2024-04-02 08:15"))) == 1
    assert len(decision_times(cfg, "GER40", GER_W, ts("2024-04-02 08:15"),
                              ts("2024-04-02 08:15"))) == 0
    # A single-point window (SES-IBX-01 style).
    one = decision_times(cfg, "US30", LocalWindow.of("10:30", "10:30"), ts("2024-03-04"),
                         ts("2024-03-16"))
    assert len(one) == 10 and all(in_window(cfg, "US30", US_W, x) for x in one)


def test_decision_times_agree_with_in_window(cfg):
    start, end = ts("2024-03-01"), ts("2024-04-06")
    t = decision_times(cfg, "NAS100", US_W, start, end)
    grid = pd.date_range(start, end, freq="15min", inclusive="left")
    assert list(t) == [x for x in grid if in_window(cfg, "NAS100", US_W, x)]


def test_window_validation():
    with pytest.raises(ValueError, match="midnight"):
        LocalWindow.of("22:00", "01:00")
    with pytest.raises(ValueError, match="grid"):
        LocalWindow.of("10:05", "11:00").check_grid(15)


def test_window_schedule_and_window_set(cfg):
    wins = {"US30": US_W, "NAS100": US_W, "US500": US_W, "GER40": GER_W}
    sched = window_schedule(cfg, wins, ts("2024-03-11"), ts("2024-03-12"))
    assert "US500" not in set(sched["key"])
    groups = sched.groupby("T")["key"].apply(tuple)
    assert set(groups) == {("US30", "NAS100"), ("GER40",)}
    assert window_set(cfg, wins, ts("2024-03-11 15:00")) == ("US30", "NAS100")
    assert window_set(cfg, wins, ts("2024-03-11 10:00")) == ("GER40",)
    assert window_set(cfg, wins, ts("2024-03-11 20:00")) == ()
    assert (sched["T"].diff().dropna() >= pd.Timedelta(0)).all()
    assert window_group(cfg, "US30") == window_group(cfg, "NAS100") != window_group(cfg, "GER40")


@pytest.mark.parametrize("key, w, t, expect", [
    # November mismatch week 2024-10-28..11-01: Berlin on CET (UTC+1), New York still on EDT.
    ("US30", US_W, "2024-10-29 14:30", True),       # 10:30 New York
    ("US30", US_W, "2024-10-29 14:15", False),
    ("NAS100", US_W, "2024-11-01 17:30", True),     # 13:30 New York
    ("NAS100", US_W, "2024-11-01 17:45", False),
    ("GER40", GER_W, "2024-10-28 09:00", True),     # 10:00 Berlin
    ("GER40", GER_W, "2024-10-28 08:45", False),
    ("GER40", GER_W, "2024-10-31 14:00", True),     # 15:00 Berlin
    ("GER40", GER_W, "2024-10-31 14:15", False),
    # Both on winter time again from 2024-11-04.
    ("US30", US_W, "2024-11-04 15:30", True),
    ("US30", US_W, "2024-11-04 14:30", False),
    ("GER40", GER_W, "2024-10-25 08:00", True),     # last EU summer-time Friday
    ("GER40", GER_W, "2024-10-25 07:45", False),
])
def test_in_window_november_mismatch(cfg, key, w, t, expect):
    assert in_window(cfg, key, w, ts(t)) is expect


def test_decision_times_november_mismatch(cfg):
    us = decision_times(cfg, "US30", US_W, ts("2024-10-28"), ts("2024-11-09"))
    first = pd.Series(us).groupby(pd.Series(us).dt.date).first()
    assert first.loc[date(2024, 11, 1)] == ts("2024-11-01 14:30")
    assert first.loc[date(2024, 11, 4)] == ts("2024-11-04 15:30")
    ger = decision_times(cfg, "GER40", GER_W, ts("2024-10-21"), ts("2024-11-02"))
    gfirst = pd.Series(ger).groupby(pd.Series(ger).dt.date).first()
    assert gfirst.loc[date(2024, 10, 25)] == ts("2024-10-25 08:00")
    assert gfirst.loc[date(2024, 10, 28)] == ts("2024-10-28 09:00")


# The signal windows of the three specs (SES-IBX-01: T = end of IB, default and grid extremes).
SPEC_WINDOWS = {
    "RS-MOM-01": ({"US": ("10:30", "13:30"), "GER": ("10:00", "15:00")}),
    "MR-TWAP-01": ({"US": ("10:30", "14:30"), "GER": ("10:00", "16:00")}),
    "SES-IBX-01 ib 60": ({"US": ("10:30", "10:30"), "GER": ("10:00", "10:00")}),
    "SES-IBX-01 ib 30": ({"US": ("10:00", "10:00"), "GER": ("09:30", "09:30")}),
    "SES-IBX-01 ib 90": ({"US": ("11:00", "11:00"), "GER": ("10:30", "10:30")}),
}


def spec_windows(name):
    w = SPEC_WINDOWS[name]
    us, ger = LocalWindow.of(*w["US"]), LocalWindow.of(*w["GER"])
    return {"US30": us, "NAS100": us, "US500": us, "GER40": ger}


@pytest.mark.parametrize("name", list(SPEC_WINDOWS))
def test_window_groups_over_a_year(cfg, name):
    """COMMON 4.2.1: every G(T) is {US30, NAS100} or {GER40}. MR-TWAP-01's windows share 14:30,
    14:45 and 15:00 UTC in the DST mismatch weeks; the schedule then holds two groups at one T,
    never one mixed group (SPEC NOTE: one G(T) per exchange group)."""
    wins = spec_windows(name)
    sched = window_schedule(cfg, wins, ts("2024-01-01"), ts("2025-01-01"))
    g = sched.groupby(["T", "group"], sort=False)["key"].apply(tuple)
    assert set(g) == {("US30", "NAS100"), ("GER40",)}
    groups_per_T = g.groupby(level=0).size()
    shared = groups_per_T[groups_per_T > 1].index
    if name != "MR-TWAP-01":
        assert len(shared) == 0
        return
    mismatch = [date(2024, 3, d) for d in (11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 25, 26, 27,
                                           28)] + [date(2024, 10, d) for d in (28, 29, 30, 31)] \
        + [date(2024, 11, 1)]
    assert sorted({t.date() for t in shared}) == mismatch
    assert sorted({t.strftime("%H:%M") for t in shared}) == ["14:30", "14:45", "15:00"]
    assert len(shared) == 3 * len(mismatch)
    T = ts("2024-10-29 14:45")
    assert window_groups(cfg, wins, T) == (("US30", "NAS100"), ("GER40",))
    assert window_set(cfg, wins, T) == ("US30", "NAS100", "GER40")
    assert window_groups(cfg, wins, ts("2024-10-29 15:15")) == (("US30", "NAS100"),)
    assert window_groups(cfg, wins, ts("2024-10-29 20:00")) == ()
    # The schedule's rows of one (T, group) are contiguous: US group first, then GER40.
    rows = sched[sched["T"] == T]
    assert list(rows["key"]) == ["US30", "NAS100", "GER40"]


def test_replay_start_is_local_midnight(cfg):
    now = ts("2024-03-12 15:00")
    assert replay_start(cfg, ["US30"], now) == ts("2024-03-12 04:00")       # EDT midnight
    assert replay_start(cfg, ["GER40", "US30"], now) == ts("2024-03-11 23:00")  # CET midnight


# ---------------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------------

def three_param_grid() -> ParamGrid:
    return ParamGrid(defaults={"theta": 1.0, "k": 1.0, "R": 1.5},
                     grid={"theta": (0.8, 0.9, 1.0, 1.1, 1.2), "k": (0.75, 0.875, 1.0, 1.125, 1.25),
                           "R": (1.0, 1.25, 1.5, 1.75, 2.0)}, heatmap=("k", "R"))


def test_param_grid_counts():
    g = three_param_grid()
    assert len(g.one_at_a_time()) == 13
    assert g.one_at_a_time()[0] == {"theta": 1.0, "k": 1.0, "R": 1.5}
    assert len(g.heatmap_cells()) == 25
    assert len(g.configurations()) == 29
    two = ParamGrid(defaults={"K": 2.0, "M": 1.5},
                    grid={"K": (1.6, 1.8, 2.0, 2.2, 2.4), "M": (1.0, 1.25, 1.5, 1.75, 2.0)},
                    heatmap=("K", "M"))
    assert len(two.one_at_a_time()) == 9 and len(two.configurations()) == 25
    configs = two.configurations()
    assert len({tuple(sorted(c.items())) for c in configs}) == 25


def test_param_grid_resolve():
    g = three_param_grid()
    p = g.resolve({"k": 1.25})
    assert dict(p) == {"theta": 1.0, "k": 1.25, "R": 1.5}
    with pytest.raises(TypeError):
        p["k"] = 2.0                                              # read-only
    with pytest.raises(ValueError, match="unknown parameter"):
        g.resolve({"L": 16})
    with pytest.raises(ValueError, match="sensitivity grid"):
        g.resolve({"theta": 0.5})
    assert g.resolve({"theta": 0.5}, off_grid=True)["theta"] == 0.5
    assert g.resolve({"theta": 1})["theta"] == 1.0
    assert g.defaults["theta"] == 1.0 and ParamGrid.label(p) == "theta=1.0,k=1.25,R=1.5"


def test_param_grid_validation():
    with pytest.raises(ValueError, match="not in its grid"):
        ParamGrid(defaults={"a": 3}, grid={"a": (1, 2)})
    with pytest.raises(ValueError, match="differ from defaults"):
        ParamGrid(defaults={"a": 1}, grid={"a": (1,), "b": (2,)})
    with pytest.raises(ValueError, match="two different"):
        ParamGrid(defaults={"a": 1}, grid={"a": (1,)}, heatmap=("a", "a"))


# ---------------------------------------------------------------------------------------------
# Regime gate frame (COMMON 6.7)
# ---------------------------------------------------------------------------------------------

class TableStub:
    threshold = 0.0

    def __init__(self, rows):
        self.rows = rows

    def at_many(self, times, live_rule=True):
        assert live_rule is True
        return pd.DataFrame(self.rows)


def test_regime_frame_gates():
    rows = {"status": ["OK", "OK", "OK", "UNAVAILABLE"], "cause": ["", "", "", "stale"],
            "v_t": [0.2, 0.0, -0.1, np.nan], "tag": ["ACTIVE", "ACTIVE", "CALM", ""],
            "label": ["ACTIVE_UP", "ACTIVE_FLAT", "CALM_FLAT", ""],
            "regime_prob": [0.9, 0.5, 0.8, np.nan]}
    times = pd.date_range("2024-03-11 15:00", periods=4, freq="15min", tz="UTC")
    act = regime_frame(TableStub(rows), times, "ACTIVE")
    assert list(act["passes"]) == [True, True, False, False]       # V_T >= 0 passes
    assert list(act["regime"]) == ["ACTIVE/ACTIVE_UP", "ACTIVE/ACTIVE_FLAT", "CALM/CALM_FLAT",
                                   "UNAVAILABLE"]
    assert act["cause"][3] == "regime unavailable: stale"
    assert act["cause"][2].startswith("regime gate ACTIVE not passed")
    calm = regime_frame(TableStub(rows), times, "CALM")
    assert list(calm["passes"]) == [False, False, True, False]
    off = regime_frame(TableStub(rows), times, None)
    assert list(off["passes"]) == [True, True, True, False]       # status OK still required
    none = regime_frame(None, times, "ACTIVE")
    assert not none["ok"].any() and "no regime table" in none["cause"][0]
    with pytest.raises(ValueError):
        regime_frame(None, times, "UP")


# ---------------------------------------------------------------------------------------------
# Lockout book (COMMON 7.1 and daily caps)
# ---------------------------------------------------------------------------------------------

def bars_frame(rows):
    return pd.DataFrame(rows, columns=["time_utc", "high", "low"]).assign(
        time_utc=lambda d: pd.to_datetime(d["time_utc"], utc=True))


def test_lockout_book_occupancy_and_counts():
    bars = bars_frame([("2024-03-11 15:00", 101, 99), ("2024-03-11 15:15", 101, 99),
                       ("2024-03-11 15:30", 101, 94), ("2024-03-11 15:45", 101, 99)])
    book = LockoutBook("TOY", {"US30": bars})
    c = Candidate("US30", BUY, 100.0, 95.0, 110.0, 2.0, "t", signal_time=ts("2024-03-11 15:00"),
                  order_type=MARKET_BAND, time_exit_utc=ts("2024-03-11 19:45"))
    d = date(2024, 3, 11)
    span = book.add(c, d)
    assert (span.end, span.cause) == (ts("2024-03-11 15:45"), "SL")   # close of the 15:30 bar
    assert book.occupied("US30", ts("2024-03-11 15:30"))
    assert not book.occupied("NAS100", ts("2024-03-11 15:30"))
    assert not book.occupied("US30", ts("2024-03-11 15:45"))
    assert book.count("US30", d) == 1 and book.count("US30", d, BUY) == 1
    assert book.count("US30", d, SELL) == 0
    with pytest.raises(ValueError, match="forward"):
        book.occupied("US30", ts("2024-03-11 15:00"))


def test_lockout_book_until_hides_later_bars():
    bars = bars_frame([("2024-03-11 15:00", 101, 99), ("2024-03-11 15:15", 101, 94)])
    c = Candidate("US30", BUY, 100.0, 95.0, 110.0, 2.0, "t", signal_time=ts("2024-03-11 15:00"),
                  order_type=MARKET_BAND, time_exit_utc=ts("2024-03-11 19:45"))
    live = LockoutBook("TOY", {"US30": bars}, until=ts("2024-03-11 15:15")).add(c, date(2024, 3, 11))
    full = LockoutBook("TOY", {"US30": bars}).add(c, date(2024, 3, 11))
    # Live has not seen the SL bar yet, so only the (known) time exit ends its span.
    assert (live.end, live.cause) == (ts("2024-03-11 19:45"), "time_exit")
    assert (full.end, full.cause) == (ts("2024-03-11 15:30"), "SL")
    assert live.occupies(ts("2024-03-11 15:15")) and full.occupies(ts("2024-03-11 15:15"))


# ---------------------------------------------------------------------------------------------
# Toy strategies on the driver
# ---------------------------------------------------------------------------------------------

class ToyMomentum(WindowedStrategy):
    """RS_4h, ACTIVE gate, US and GER40 windows; at most 2 per index per side per local date and
    no new candidate while its own earlier candidate occupies the index."""
    id = "TOY-MOM"
    param_grid = ParamGrid(defaults={"theta": 0.8, "k": 1.0},
                           grid={"theta": (0.6, 0.8, 1.0), "k": (0.5, 1.0, 1.5)},
                           heatmap=("theta", "k"))
    regime_gate = "ACTIVE"
    rank = "rs_4h"

    def windows(self, params):
        # US500 is listed on purpose: it must never enter G(T).
        return {"US30": US_W, "NAS100": US_W, "US500": US_W, "GER40": GER_W}

    def incomplete_cause(self, view, key):
        return None if math.isfinite(view.value(key, "atr14")) else "ATR14_M15 NaN"

    def decide(self, view):
        s, p, out = view.selection, view.params, []
        for side, key in ((BUY, s.long), (SELL, s.short)):
            if key is None:
                continue
            score = s.scores[key]
            if (side == BUY and score < p["theta"]) or (side == SELL and score > -p["theta"]):
                continue
            if view.book.occupied(key, view.T) or \
                    view.book.count(key, view.local_date[key], side) >= 2:
                continue
            e, a = float(view.value(key, "close")), float(view.value(key, "atr14"))
            sg = 1 if side == BUY else -1
            out.append(Candidate(key, side, e, e - sg * p["k"] * a, e + sg * 1.5 * p["k"] * a, a,
                                 f"TOY {side} {key}", order_type=MARKET_BAND,
                                 **view.candidate_fields(key)))
        return out


class ToyFade(WindowedStrategy):
    """RS_day, CALM gate, MARKET_OR_LIMIT with an EXPIRY; one candidate per index per date."""
    id = "TOY-FADE"
    param_grid = ParamGrid(defaults={"z": 0.5}, grid={"z": (0.25, 0.5, 1.0)})
    regime_gate = "CALM"
    rank = "rs_day"

    def windows(self, params):
        return {"US30": LocalWindow.of("10:30", "14:30"), "NAS100": LocalWindow.of("10:30", "14:30"),
                "GER40": LocalWindow.of("10:00", "16:00")}

    def incomplete_cause(self, view, key):
        for col in ("twap", "atr14"):
            if not math.isfinite(view.value(key, col)):
                return f"{col} NaN"
        return None

    def decide(self, view):
        s, out = view.selection, []
        for side, key in ((BUY, s.long), (SELL, s.short)):
            if key is None or view.book.occupied(key, view.T) or \
                    view.book.count(key, view.local_date[key]) >= 1:
                continue
            e, a = float(view.value(key, "close")), float(view.value(key, "atr14"))
            tw = float(view.value(key, "twap"))
            z = (e - tw) / a
            if (side == BUY and z > -view.params["z"]) or (side == SELL and z < view.params["z"]):
                continue
            sg = 1 if side == BUY else -1
            out.append(Candidate(key, side, e, e - sg * 1.5 * a, tw, a, f"TOY {side} {key}",
                                 order_type=MARKET_OR_LIMIT,
                                 expires_utc=view.T + pd.Timedelta(minutes=60),
                                 **view.candidate_fields(key)))
        return out


@pytest.fixture(scope="module")
def market():
    return make_market()


@pytest.fixture(scope="module")
def mod_cfg():
    """Module-scoped config (the conftest `cfg` fixture is per test; nothing here edits it)."""
    return load_config(Path(__file__).resolve().parent.parent / "config.yaml")


@pytest.fixture(scope="module")
def services(market, mod_cfg):
    return build_services(mod_cfg, market)


def test_scan_candidates_respect_selection(market, mod_cfg, services):
    strat = ToyMomentum(mod_cfg)
    cycles = list(strat.cycles(*services, market.scan_start, market.end))
    cands = [c for _, cs in cycles for c in cs]
    assert len(cands) > 20                        # the toy must actually fire
    by_T_key = {(s.T, k): s for s, _ in cycles for k in s.members}
    for c in cands:
        s = by_T_key[(ts(c.signal_time), c.key)]
        assert s.allows(c.key, c.side)
        assert c.rank_score == round(s.scores[c.key], 2)
        assert c.regime.startswith("ACTIVE/")
        assert c.key != "US500"
    assert all("US500" not in s.members for s, _ in cycles)
    assert {c.key for c in cands} == {"US30", "NAS100", "GER40"}
    # scan() is the candidates of cycles() in order.
    assert_candidates_equal(strat.scan(*services, market.scan_start, market.end), cands)


def test_scan_caps_and_occupancy(market, mod_cfg, services):
    strat = ToyMomentum(mod_cfg, {"theta": 0.6})
    cands = strat.scan(*services, market.scan_start, market.end)
    per_day = pd.Series(1, index=pd.MultiIndex.from_tuples(
        [(c.key, c.side, c.signal_time.tz_convert(mod_cfg["exchanges"][c.key]["tz"]).date())
         for c in cands])).groupby(level=[0, 1, 2]).sum()
    assert per_day.max() <= 2
    # No candidate starts while an earlier one on the same index still occupies it.
    book = LockoutBook("X", {k: services[0].get(k).m15 for k in ("US30", "NAS100", "GER40")})
    for c in cands:
        assert not book.occupied(c.key, c.signal_time)
        book.add(c, c.signal_time.date())


def test_incomplete_group_causes(market, mod_cfg, services):
    strat = ToyMomentum(mod_cfg)
    sels = {s.T: s for s, _ in strat.cycles(*services, market.scan_start, market.end)}
    # NAS100 bar 15:30-15:45 UTC missing on 2024-02-22.
    s = sels[ts("2024-02-22 15:45")]
    assert s.incomplete == f"incomplete group: NAS100: {NO_BAR}" and not s.has_side
    # ... and its RS_4h is NaN while the missing bar is inside the 16-bar lookback.
    s = sels[ts("2024-02-22 16:00")]
    assert s.incomplete.startswith("incomplete group: NAS100: RS_4h NaN")
    # US30 cash-open bar missing on 2024-03-05: not a valid trading date for US30, so US30 is
    # not in G(T) (COMMON 4.2.1 + 1.6) and NAS100 is ranked alone under the sign rule (4.2.5).
    day = [s for t, s in sels.items() if t.date() == date(2024, 3, 5)
           and ("US30" in s.members or "NAS100" in s.members)]
    assert len(day) == 13                                   # 10:30..13:30 New York
    for s in day:
        assert s.members == ("NAS100",) and dict(s.excluded) == {"US30": INVALID_DATE}
        assert f"outside G(T): US30: {INVALID_DATE}" in s.log_line()
    ranked = [s for s in day if s.incomplete is None]
    assert ranked, "NAS100 must be ranked on its own that day"
    for s in ranked:
        sc = s.scores["NAS100"]
        if s.eligible == ("NAS100",):
            assert s.sole and (s.long, s.short) == (("NAS100", None) if sc > 0 else
                                                    (None, "NAS100") if sc < 0 else (None, None))
    # GER40 inside its > 7-day hole: no bars, so no valid trading date and no cycle at all.
    assert not [s for t, s in sels.items() if ts("2024-02-26") <= t < ts("2024-03-05")
                and "GER40" in s.members]
    # GER40 after the hole: lookbacks restart, so the group is incomplete for a while.
    s = sels[ts("2024-03-06 10:00")]
    assert s.members == ("GER40",) and s.incomplete.startswith("incomplete group: GER40")


def test_excluded_member_validation_and_log():
    s = select_index(T0, [MemberInput("NAS100", -0.4)], ORDER, "TEST",
                     excluded={"US30": INVALID_DATE})
    assert s.members == ("NAS100",) and s.short == "NAS100" and s.long is None and s.sole
    assert dict(s.excluded) == {"US30": INVALID_DATE}
    assert s.log_line().startswith(f"TEST 2024-03-12 15:00 UTC G(T) {{NAS100 -0.4000}}; "
                                   f"outside G(T): US30: {INVALID_DATE}; E(T) {{NAS100}}")
    with pytest.raises(ValueError, match="excluded"):
        select_index(T0, [MemberInput("US30", 0.1)], ORDER, excluded={"US30": "x"})
    with pytest.raises(ValueError, match="excluded"):
        select_index(T0, [MemberInput("US30", 0.1)], ORDER, excluded={"US500": "x"})
    assert dict(sel(MemberInput("US30", 0.1)).excluded) == {}


def test_excluded_member_live_matches_scan(market, mod_cfg, services):
    """The literal G(T) on a date without US30's cash-open bar: live == batch, and the
    candidates of that day are on NAS100 only."""
    strat = ToyMomentum(mod_cfg, {"theta": 0.6})
    start = ts("2024-03-04")
    full = strat.scan(*services, start, market.end)
    day = [c for c in full if c.signal_time.date() == date(2024, 3, 5) and c.key != "GER40"]
    assert all(c.key == "NAS100" for c in day)
    times = [ts(c.signal_time) for c in day[:3]] + [ts("2024-03-05 15:00"),
                                                    ts("2024-03-05 17:30")]
    assert_generate_matches_scan(strat, mod_cfg, market, times, start=start, full=full)


def test_scan_selection_sink(market, mod_cfg, services):
    """COMMON 4.2.8: scan() hands every cycle's Selection to the sink, in cycle order."""
    strat = ToyMomentum(mod_cfg)
    start, end = ts("2024-03-11"), ts("2024-03-16")
    got: list = []
    cands = strat.scan(*services, start, end, on_selection=got.append)
    ref = list(strat.cycles(*services, start, end))
    assert [s.log_line() for s in got] == [s.log_line() for s, _ in ref]
    assert_candidates_equal(cands, [c for _, cs in ref for c in cs])
    assert all("G(T) {" in s.log_line() for s in got)


def test_strategy_hook_incomplete_cause(market, mod_cfg, services):
    class NeedsVolume(ToyMomentum):
        def incomplete_cause(self, view, key):
            return "made-up input NaN" if key == "NAS100" else None
    sels = [s for s, _ in NeedsVolume(mod_cfg).cycles(*services, ts("2024-03-11"),
                                                       ts("2024-03-12"))]
    us = [s for s in sels if "US30" in s.members]
    assert us and all(s.incomplete == "incomplete group: NAS100: made-up input NaN" for s in us)


def test_strategy_hook_ineligible_cause(market, mod_cfg, services):
    class OnlyNasdaq(ToyMomentum):
        def ineligible_cause(self, view, key):
            return "test filter" if key == "US30" else None
    sels = [s for s, _ in OnlyNasdaq(mod_cfg).cycles(*services, market.scan_start, market.end)]
    us = [s for s in sels if "US30" in s.members and s.incomplete is None]
    assert us and all("US30" not in s.eligible for s in us)
    assert any(s.reasons.get("US30") == "test filter" for s in us)


def test_driver_rejects_promoting_the_next_index(market, mod_cfg, services):
    class Promoter(ToyMomentum):
        def decide(self, view):
            s = view.selection
            other = [k for k in s.members if k != s.long]
            if s.long and other:
                e = float(view.value(other[0], "close"))
                return [Candidate(other[0], BUY, e, e - 10, e + 10, 1.0, "bad",
                                  order_type=MARKET_BAND, **view.candidate_fields(other[0]))]
            return []
    with pytest.raises(ValueError, match="4.2.6"):
        Promoter(mod_cfg).scan(*services, market.scan_start, market.end)


def test_generate_without_services_warns_once(mod_cfg, caplog):
    """An unwired live context must not look like 'no setup': [] plus one WARNING."""
    ctx_mod._WARNED_UNWIRED.clear()
    strat = ToyMomentum(mod_cfg)
    T = ts("2024-03-12 15:00")
    with caplog.at_level(logging.WARNING, logger="strategies.context"):
        assert strat.generate({}, {}, T, StrategyContext(now=T, cfg=mod_cfg)) == []
        assert strat.generate({}, {}, T, StrategyContext(now=T, cfg=mod_cfg)) == []
    warn = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn) == 1
    assert "TOY-MOM" in warn[0] and "features and .regime_tables" in warn[0]


class ToyFadeUS(ToyFade):
    def windows(self, params):
        return {k: w for k, w in super().windows(params).items() if k != "GER40"}


class ToyFadeGER(ToyFade):
    def windows(self, params):
        return {"GER40": super().windows(params)["GER40"]}


def group_cycles(cycles, keys):
    return [(s.log_line(), cs) for s, cs in cycles if set(s.members) <= set(keys)]


def assert_groups_independent(mod_cfg, services, start, end, n_shared):
    """MR-TWAP-style windows in a DST mismatch week: two G(T) at one T, each ranked exactly as
    its exchange group alone would be (SPEC NOTE: one G(T) per exchange group)."""
    both = list(ToyFade(mod_cfg).cycles(*services, start, end))
    assert {s.members for s, _ in both} == {("US30", "NAS100"), ("GER40",)}
    assert len(both) - len({s.T for s, _ in both}) == n_shared
    for alone, keys in ((ToyFadeUS, ("US30", "NAS100")), (ToyFadeGER, ("GER40",))):
        ref = list(alone(mod_cfg).cycles(*services, start, end))
        got = group_cycles(both, keys)
        assert [x for x, _ in got] == [s.log_line() for s, _ in ref]
        assert_candidates_equal([c for _, cs in got for c in cs],
                                [c for _, cs in ref for c in cs], f"group {keys}")
    return both


def test_overlapping_windows_rank_each_exchange_group_separately(market, mod_cfg, services):
    both = assert_groups_independent(mod_cfg, services, ts("2024-03-11"), ts("2024-03-29"),
                                     n_shared=3 * 14)
    at = [s for s, _ in both if s.T == ts("2024-03-11 14:30")]
    assert [s.members for s in at] == [("US30", "NAS100"), ("GER40",)]
    for s, cs in both:
        for c in cs:
            assert c.key in s.members and s.allows(c.key, c.side)
    # live == batch at the shared T (both groups' candidates at T, US group first).
    shared = sorted({s.T for s, _ in both if s.T.strftime("%H:%M") in ("14:30", "14:45",
                                                                        "15:00")})
    with_c = [T for T in shared if any(cs for s, cs in both if s.T == T)]
    times = (with_c[:3] + shared[:2]) if with_c else shared[:3]
    full = ToyFade(mod_cfg).scan(*services, ts("2024-03-11"), market.end)
    assert_generate_matches_scan(ToyFade(mod_cfg), mod_cfg, market, times, full=full,
                                 start=ts("2024-03-11"))


@pytest.fixture(scope="module")
def autumn():
    """Defect-free market across the November 2024 mismatch week (EU back on CET 2024-10-27,
    US on 2024-11-03)."""
    return make_market("2024-09-23", "2024-11-09", drops={}, holes={})


def test_overlapping_windows_november_mismatch(autumn, mod_cfg):
    svc = build_services(mod_cfg, autumn)
    both = assert_groups_independent(mod_cfg, svc, ts("2024-10-28"), ts("2024-11-09"),
                                     n_shared=3 * 5)
    assert all(s.T.date() <= date(2024, 11, 1) for s, _ in both
               if sum(x.T == s.T for x, _ in both) > 1)


def test_scan_from_mid_session_matches_full_scan(market, mod_cfg, services):
    """scan() from a mid-session start replays that session's lockouts (as generate() does)."""
    strat = ToyMomentum(mod_cfg, {"theta": 0.6})
    full = strat.scan(*services, market.scan_start, market.end)
    cuts = [ts("2024-03-13 15:00")]
    cuts += [ts(a.signal_time) + pd.Timedelta(minutes=15) for a, b in zip(full, full[1:])
             if a.key == b.key and a.signal_time.date() == b.signal_time.date()][:4]
    assert len(cuts) > 1
    for cut in cuts:
        part = strat.scan(*services, cut, market.end)
        assert_candidates_equal(part, [c for c in full if ts(c.signal_time) >= cut],
                                f"scan from {cut}")
        first = next(strat.cycles(*services, cut, market.end))[0]
        assert first.T >= cut                        # the replay before `cut` is not yielded


def test_params_override_and_off_grid(mod_cfg):
    assert ToyMomentum(mod_cfg).params["theta"] == 0.8
    assert ToyMomentum(mod_cfg, {"theta": 1.0}).params["theta"] == 1.0
    with pytest.raises(ValueError):
        ToyMomentum(mod_cfg, {"theta": 0.7})
    assert ToyMomentum(mod_cfg, {"theta": 0.7}, off_grid=True).params["theta"] == 0.7


# ---------------------------------------------------------------------------------------------
# Testkit: look-ahead safety and live == batch
# ---------------------------------------------------------------------------------------------

def test_toy_momentum_no_lookahead(market, mod_cfg):
    strat = ToyMomentum(mod_cfg, {"theta": 0.6})
    full = assert_no_lookahead(strat, mod_cfg, market,
                               cuts=["2024-03-13 16:00", "2024-04-03 09:30"],
                               end=ts("2024-04-06"), seeds=(1,))
    assert any(c.signal_time <= ts("2024-03-13 16:00") for c in full)


def test_toy_momentum_generate_matches_scan(market, mod_cfg, services):
    strat = ToyMomentum(mod_cfg, {"theta": 0.6})
    full = strat.scan(*services, market.scan_start, market.end)
    times = sample_times(strat, mod_cfg, full, market.scan_start + pd.Timedelta(days=2),
                         market.end, n_candidates=5, n_other=2)
    times = [t for t in times if t >= market.scan_start + pd.Timedelta(days=2)]
    # Include a T right after an earlier candidate on the same index and date (lockout replay).
    same_day = [b for a, b in zip(full, full[1:]) if a.key == b.key
                and a.signal_time.date() == b.signal_time.date()]
    if same_day:
        times.append(ts(same_day[0].signal_time))
    assert assert_generate_matches_scan(strat, mod_cfg, market, times, full=full) > 0


def test_toy_fade_generate_matches_scan_and_no_lookahead(market, mod_cfg):
    strat = ToyFade(mod_cfg)
    full = assert_no_lookahead(strat, mod_cfg, market, cuts=["2024-03-20 15:00"],
                               end=ts("2024-03-28"), seeds=(2,))
    assert full, "the toy must fire"
    times = [ts(full[len(full) // 2].signal_time), ts(full[-1].signal_time)]
    assert assert_generate_matches_scan(strat, mod_cfg, market, times, full=full) >= 2


def test_testkit_catches_lookahead(market, mod_cfg):
    """A strategy that peeks at the next bar must fail the look-ahead check."""
    class Peeker(ToyMomentum):
        def decide(self, view):
            s = view.selection
            key = s.long
            if key is None:
                return []
            m15 = view.fx[key].m15
            p = view.pos[key] + 1
            if p >= len(m15) or m15["close"].iat[p] <= m15["close"].iat[p - 1]:
                return []
            e = float(view.value(key, "close"))
            return [Candidate(key, BUY, e, e - 10, e + 10, 1.0, "peek", order_type=MARKET_BAND,
                              **view.candidate_fields(key))]
    with pytest.raises(AssertionError, match="perturbing"):
        assert_no_lookahead(Peeker(mod_cfg), mod_cfg, market, cuts=["2024-03-20 16:00"],
                            end=ts("2024-03-25"), seeds=(3,))


def test_testkit_market_shapes(market):
    T = ts("2024-03-11 15:00")
    tr = market.truncated(T)
    assert (tr.m15["US30"]["time_utc"] + pd.Timedelta(minutes=15) <= T).all()
    assert (tr.h1["US30"]["time_utc"] + pd.Timedelta(hours=1) <= T).all()
    pert = market.perturbed_after(T, seed=5)
    for k in market.keys:
        a = market.m15[k][market.m15[k]["time_utc"] < T].reset_index(drop=True)
        b = pert.m15[k][pert.m15[k]["time_utc"] < T].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        ha = market.h1[k][market.h1[k]["time_utc"] + pd.Timedelta(hours=1) <= T]
        hb = pert.h1[k][pert.h1[k]["time_utc"] + pd.Timedelta(hours=1) <= T]
        pd.testing.assert_frame_equal(ha.reset_index(drop=True), hb.reset_index(drop=True))
        assert len(pert.m15[k]) < len(market.m15[k])           # some later bars dropped
    # Default defects are present.
    assert not (market.m15["NAS100"]["time_utc"] == ts("2024-02-22 15:30")).any()
    gap = market.m15["GER40"]["time_utc"].diff().max()
    assert gap > pd.Timedelta(days=7)


def test_proxy_regime_is_causal(market, mod_cfg):
    T = ts("2024-03-20 15:00")
    times = pd.date_range(ts("2024-03-18"), T, freq="15min")
    full = ProxyRegimeTable("US30", market.h1["US30"]).at_many(times)
    trunc = ProxyRegimeTable("US30", market.truncated(T).h1["US30"]).at_many(times)
    pd.testing.assert_frame_equal(full, trunc)
    assert set(full["status"]) <= {"OK", "UNAVAILABLE"} and (full["status"] == "OK").any()


# US30 H1 bar 16:00-17:00 UTC (12:00-13:00 New York, a core hour) missing, M15 bars intact.
H1_GAP_OPEN, H1_GAP_T = ts("2024-03-12 16:00"), ts("2024-03-12 17:00")


@pytest.fixture(scope="module")
def h1_gap():
    return make_market("2024-02-12", "2024-03-16", h1_drops={"US30": [H1_GAP_OPEN]})


def test_proxy_regime_live_rule(h1_gap, mod_cfg):
    """COMMON 6.6 live rule in the stand-in: UNAVAILABLE exactly at the close of the missing
    core H1 bar, the previous value otherwise; causal on truncated history."""
    tab = build_services(mod_cfg, h1_gap)[1]["US30"]
    times = pd.DatetimeIndex([H1_GAP_T - pd.Timedelta(minutes=15), H1_GAP_T,
                              H1_GAP_T + pd.Timedelta(minutes=15)])
    live = tab.at_many(times)
    assert list(live["status"]) == ["OK", "UNAVAILABLE", "OK"]
    assert live["cause"][1] == "regime H1 bar missing"
    assert (tab.at_many(times, live_rule=False)["status"] == "OK").all()
    trunc = build_services(mod_cfg, h1_gap.truncated(H1_GAP_T))[1]["US30"]
    pd.testing.assert_frame_equal(trunc.at_many(times[:2]), live.iloc[:2].reset_index(drop=True))
    # Only that slot: the other indices and every other core slot keep their H1 bars.
    other = build_services(mod_cfg, h1_gap)[1]["NAS100"].at_many([H1_GAP_T])
    assert other["status"][0] == "OK"
    assert list(tab._miss) == [H1_GAP_T.value]


def test_missing_regime_h1_bar_live_matches_scan(h1_gap, mod_cfg):
    """A missing core H1 bar at a window T: US30 is not eligible ("regime unavailable: regime
    H1 bar missing", COMMON 6.7) in the scan, and live generate() agrees."""
    strat = ToyMomentum(mod_cfg, {"theta": 0.6})
    svc = build_services(mod_cfg, h1_gap)
    start = h1_gap.scan_start
    sels = {s.T: s for s, _ in strat.cycles(*svc, start, h1_gap.end) if "US30" in s.members}
    s = sels[H1_GAP_T]
    assert s.incomplete is None                    # RS_4h is M15-based: the group is complete
    assert s.reasons["US30"] == "regime unavailable: regime H1 bar missing"
    assert "US30" not in s.eligible
    assert "US30" not in sels[H1_GAP_T + pd.Timedelta(minutes=15)].reasons or \
        "H1 bar missing" not in sels[H1_GAP_T + pd.Timedelta(minutes=15)].reasons["US30"]
    full = strat.scan(*svc, start, h1_gap.end)
    assert_generate_matches_scan(strat, mod_cfg, h1_gap,
                                 [H1_GAP_T, H1_GAP_T + pd.Timedelta(minutes=15)],
                                 start=start, full=full)
