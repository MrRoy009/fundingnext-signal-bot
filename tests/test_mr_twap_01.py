"""MR-TWAP-01 (docs/specs/MR-TWAP-01.md): every rule on synthetic data, the look-ahead
perturbation check, live generate() == batch scan(), DST edges of the windows and time exits.

Synthetic data only (tests.strategy_testkit); no real bars are read here. Off-grid K values
(off_grid=True, synthetic tests only) are used where a test needs more setups than the
defaults give on three months of random bars; the rules do not depend on K's value.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.config import load_config
from strategies.base import MARKET_OR_LIMIT, StrategyContext, protocol_levels
from strategies.context import (INVALID_DATE, NO_BAR, CycleView, LocalWindow, LockoutBook,
                                RegimeRow, decision_times, in_window, window_groups)
from strategies.mr_twap_01 import (PARAMS, STRATEGY_ID, MrTwap01, expiry_utc,
                                   models_without_calm, round_away, round_toward,
                                   signal_window, stretch)
from strategies.selection import BUY, SELL, MemberInput, select_index
from tests.strategy_testkit import (assert_candidates_equal, assert_generate_matches_scan,
                                    assert_no_lookahead, build_services, make_market,
                                    sample_times, ts)

KEYS = ("US30", "NAS100", "GER40")
LOW_K = {"K": 1.0}          # off-grid, synthetic tests only: more setups per test market


@pytest.fixture(scope="module")
def mcfg():
    return load_config(Path(__file__).resolve().parent.parent / "config.yaml")


@pytest.fixture(scope="module")
def market():
    return make_market()


@pytest.fixture(scope="module")
def autumn():
    """Covers the November mismatch week (Berlin on CET from 2024-10-27, New York on EDT until
    2024-11-03) and both DST ends."""
    return make_market(start="2024-09-16", end="2024-11-23", drops={}, holes={})


@pytest.fixture(scope="module")
def services(mcfg, market):
    return build_services(mcfg, market)


# ---------------------------------------------------------------------------------------------
# Regime stand-ins with the RegimeTable at_many interface
# ---------------------------------------------------------------------------------------------

@dataclass
class ConstantRegime:
    """Status OK and a fixed V_T at every T (CALM if v < threshold)."""
    v: float
    threshold: float = 0.0
    unavailable: bool = False

    def at_many(self, times, live_rule: bool = True) -> pd.DataFrame:
        t = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit("ns")
        n = len(t)
        tag = "CALM" if self.v < self.threshold else "ACTIVE"
        if self.unavailable:
            return pd.DataFrame({"time_utc": t, "status": ["UNAVAILABLE"] * n,
                                 "cause": ["test outage"] * n, "v_t": np.full(n, np.nan),
                                 "tag": [""] * n, "label": [""] * n,
                                 "regime_prob": np.full(n, np.nan)})
        return pd.DataFrame({"time_utc": t, "status": ["OK"] * n, "cause": [""] * n,
                             "v_t": np.full(n, self.v), "tag": [tag] * n,
                             "label": [f"{tag}_FLAT"] * n, "regime_prob": np.full(n, 0.9)})


def const_regime(**per_key):
    """Factory for build_services: per-key ConstantRegime (default CALM, V_T = -0.5)."""
    def make(market, cfg):
        return {k: per_key.get(k, ConstantRegime(-0.5)) for k in market.keys}
    return make


def calm_services(cfg, market, **per_key):
    return build_services(cfg, market, const_regime(**per_key))


# ---------------------------------------------------------------------------------------------
# Parameters (spec "Parameters")
# ---------------------------------------------------------------------------------------------

def test_parameters_defaults_grid_and_heatmap():
    assert dict(PARAMS.defaults) == {"K": 2.0, "M": 1.5}
    assert PARAMS.grid["K"] == (1.6, 1.8, 2.0, 2.2, 2.4)
    assert PARAMS.grid["M"] == (1.0, 1.25, 1.5, 1.75, 2.0)
    assert PARAMS.heatmap == ("K", "M")
    assert len(PARAMS.one_at_a_time()) == 9
    assert len(PARAMS.heatmap_cells()) == 25
    assert len(PARAMS.configurations()) == 25       # the heatmap contains the 1-D grid


def test_parameters_refuse_off_grid_values(mcfg):
    assert MrTwap01(mcfg).params == {"K": 2.0, "M": 1.5}
    assert MrTwap01(mcfg, {"K": 2.2, "M": 1.25}).params == {"K": 2.2, "M": 1.25}
    with pytest.raises(ValueError, match="sensitivity grid"):
        MrTwap01(mcfg, {"K": 1.0})
    with pytest.raises(ValueError, match="unknown parameter"):
        MrTwap01(mcfg, {"P_MIN": 0.6})                # removed by the QA revision
    s = MrTwap01(mcfg)
    assert s.id == STRATEGY_ID and s.regime_gate == "CALM" and s.rank == "rs_day"


# ---------------------------------------------------------------------------------------------
# Windows and DST (spec "Instruments, timeframes, sessions")
# ---------------------------------------------------------------------------------------------

def test_signal_windows_from_config(mcfg):
    assert signal_window(mcfg, "US30") == LocalWindow.of("10:30", "14:30")
    assert signal_window(mcfg, "NAS100") == LocalWindow.of("10:30", "14:30")
    assert signal_window(mcfg, "GER40") == LocalWindow.of("10:00", "16:00")
    assert set(MrTwap01(mcfg).windows(PARAMS.defaults)) == set(KEYS)   # US500 never ranked


def _first_last(cfg, key, day):
    t = decision_times(cfg, key, signal_window(cfg, key), ts(day), ts(day) + pd.Timedelta(days=1))
    return f"{t[0]:%H:%M}", f"{t[-1]:%H:%M}", len(t)


@pytest.mark.parametrize("key, day, first, last", [
    ("US30", "2024-03-08", "15:30", "19:30"),     # last US winter day before 2024-03-10
    ("US30", "2024-03-11", "14:30", "18:30"),     # first US summer day
    ("NAS100", "2024-11-01", "14:30", "18:30"),   # last US summer day before 2024-11-03
    ("NAS100", "2024-11-04", "15:30", "19:30"),   # first US winter day
    ("GER40", "2024-03-28", "09:00", "15:00"),    # last EU winter day before 2024-03-31
    ("GER40", "2024-04-02", "08:00", "14:00"),    # first EU summer trading day (after Easter)
    ("GER40", "2024-10-25", "08:00", "14:00"),    # last EU summer day before 2024-10-27
    ("GER40", "2024-10-28", "09:00", "15:00"),    # first EU winter day
])
def test_window_utc_follows_each_markets_dst(mcfg, key, day, first, last):
    f, l, n = _first_last(mcfg, key, day)
    assert (f, l) == (first, last)
    assert n == (17 if key != "GER40" else 25)    # both ends inclusive, 15-minute steps


def test_window_edges_inclusive(mcfg):
    w = signal_window(mcfg, "US30")
    assert in_window(mcfg, "US30", w, ts("2024-03-12 14:30"))       # 10:30 New York (EDT)
    assert in_window(mcfg, "US30", w, ts("2024-03-12 18:30"))       # 14:30 New York
    assert not in_window(mcfg, "US30", w, ts("2024-03-12 14:15"))
    assert not in_window(mcfg, "US30", w, ts("2024-03-12 18:45"))
    g = signal_window(mcfg, "GER40")
    assert in_window(mcfg, "GER40", g, ts("2024-03-12 09:00"))      # 10:00 Berlin (CET)
    assert in_window(mcfg, "GER40", g, ts("2024-03-12 15:00"))      # 16:00 Berlin
    assert not in_window(mcfg, "GER40", g, ts("2024-03-12 15:15"))
    assert not in_window(mcfg, "US30", w, ts("2024-07-04 15:00"))   # NYSE holiday
    assert not in_window(mcfg, "US30", w, ts("2024-11-29 15:00"))   # NYSE early close


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("day", ["2024-01-10", "2024-03-13", "2024-07-10", "2024-10-30"])
def test_window_inside_config_sessions_both_dst_states(mcfg, key, day):
    """Spec: the windows lie inside config sessions_utc in both DST states (start <= T < end,
    ER-12), including the mismatch weeks (2024-03-13, 2024-10-30)."""
    t = decision_times(mcfg, key, signal_window(mcfg, key), ts(day), ts(day) + pd.Timedelta(days=1))
    assert len(t)
    spans = [(time.fromisoformat(a), time.fromisoformat(b)) for a, b in mcfg["sessions_utc"][key]]
    for T in t:
        assert any(a <= T.time() < b for a, b in spans), (key, T)


def test_mismatch_weeks_give_two_groups_at_shared_T(mcfg):
    wins = MrTwap01(mcfg).windows(PARAMS.defaults)
    for T in ("2024-03-12 14:30", "2024-03-12 15:00", "2024-10-29 14:45"):
        assert window_groups(mcfg, wins, ts(T)) == (("US30", "NAS100"), ("GER40",))
    assert window_groups(mcfg, wins, ts("2024-03-12 15:15")) == (("US30", "NAS100"),)
    assert window_groups(mcfg, wins, ts("2024-04-09 14:30")) == (("US30", "NAS100"),)
    assert window_groups(mcfg, wins, ts("2024-11-05 14:30")) == (("GER40",),)   # 15:30 CET, 09:30 EST
    assert window_groups(mcfg, wins, ts("2024-11-05 15:15")) == ()


# ---------------------------------------------------------------------------------------------
# Time exit and EXPIRY (spec "Entry" step 4, "Exit")
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key, T, exit_utc, expiry", [
    ("US30", "2024-03-08 15:30", "2024-03-08 20:45", "2024-03-08 16:30"),   # EST, T + 60
    ("US30", "2024-03-08 19:30", "2024-03-08 20:45", "2024-03-08 20:15"),   # last T: exit - 30
    ("US30", "2024-03-08 19:15", "2024-03-08 20:45", "2024-03-08 20:15"),   # both equal
    ("US30", "2024-03-08 19:00", "2024-03-08 20:45", "2024-03-08 20:00"),
    ("NAS100", "2024-03-11 18:30", "2024-03-11 19:45", "2024-03-11 19:15"),  # EDT
    ("NAS100", "2024-11-04 19:30", "2024-11-04 20:45", "2024-11-04 20:15"),  # EST again
    ("GER40", "2024-03-28 15:00", "2024-03-28 16:15", "2024-03-28 15:45"),   # CET
    ("GER40", "2024-04-02 14:00", "2024-04-02 15:15", "2024-04-02 14:45"),   # CEST
    ("GER40", "2024-04-02 08:00", "2024-04-02 15:15", "2024-04-02 09:00"),
    ("GER40", "2024-10-28 15:00", "2024-10-28 16:15", "2024-10-28 15:45"),   # CET
])
def test_time_exit_and_expiry_on_dst_edges(mcfg, key, T, exit_utc, expiry):
    from strategies.base import time_exit_utc
    assert time_exit_utc(mcfg, key, ts(T)) == ts(exit_utc)
    assert expiry_utc(mcfg, key, ts(T)) == ts(expiry)


def test_every_window_T_expires_at_least_30_min_before_exit(mcfg):
    """Spec: End = time exit - 75 min, so every limit expires >= 30 min before the time exit."""
    from strategies.base import time_exit_utc
    for key in KEYS:
        w = signal_window(mcfg, key)
        for T in decision_times(mcfg, key, w, ts("2024-03-04"), ts("2024-04-06")):
            te = time_exit_utc(mcfg, key, T)
            e = expiry_utc(mcfg, key, T)
            assert e == min(T + pd.Timedelta(minutes=60), te - pd.Timedelta(minutes=30))
            assert e <= te - pd.Timedelta(minutes=30) and T < e


# ---------------------------------------------------------------------------------------------
# Helpers: stretch and rounding (spec "Entry" step 1, "Stop", "Exit")
# ---------------------------------------------------------------------------------------------

def test_stretch():
    assert stretch(100.0, 104.0, 2.0) == -2.0
    assert stretch(108.0, 104.0, 2.0) == 2.0
    assert math.isnan(stretch(100.0, 104.0, 0.0))
    assert math.isnan(stretch(100.0, float("nan"), 2.0))
    assert math.isnan(stretch(100.0, 104.0, float("nan")))


def test_rounding_away_and_toward_entry():
    assert round_away(97.123, 100.0) == 97.12          # long SL below entry: down
    assert round_away(102.871, 100.0) == 102.88        # short SL above entry: up
    assert round_toward(104.009, 100.0) == 104.0       # long TP above entry: down
    assert round_toward(95.991, 100.0) == 96.0         # short TP below entry: up
    assert round_away(39000.12, 39100.0) == 39000.12   # exact 2 dp is kept (float noise)
    assert round_toward(39000.12, 38000.0) == 39000.12
    assert round_away(97.12, 100.0) == 97.12


# ---------------------------------------------------------------------------------------------
# decide() unit tests on a hand-built CycleView (exact thresholds, lockouts, fields, reason)
# ---------------------------------------------------------------------------------------------

class _Fx:
    def __init__(self, close, twap, atr):
        self.m15 = pd.DataFrame({"close": [close], "twap": [twap], "atr14": [atr]})


class _Book:
    def __init__(self, counted=(), occupied=()):
        self.counted, self.busy = set(counted), set(occupied)

    def count(self, key, local_date, side=None):
        return int(key in self.counted)

    def occupied(self, key, T):
        return key in self.busy


def _view(cfg, T, members, book=None, params=None, regime_v=-0.3):
    """members: {key: (score, close, twap, atr)}; all members pass the CALM gate."""
    T = ts(T)
    sel = select_index(T, [MemberInput(k, v[0]) for k, v in members.items()],
                       ("US30", "NAS100", "GER40"), STRATEGY_ID)
    tz = {k: cfg["exchanges"][k]["tz"] for k in members}
    return CycleView(
        T, cfg, PARAMS.resolve(params, off_grid=True),
        {k: _Fx(*v[1:]) for k, v in members.items()}, {k: 0 for k in members},
        {k: RegimeRow(True, True, "", "CALM/CALM_FLAT", 0.8, regime_v) for k in members},
        {k: T.tz_convert(tz[k]).date() for k in members}, book or _Book(), selection=sel)


def test_long_trigger_at_exact_threshold_and_fields(mcfg):
    # US group, both pass: US30 strongest (LONG-SELECTED), NAS100 weakest.
    v = _view(mcfg, "2024-03-12 15:00", {"US30": (0.8, 100.0, 104.0, 2.0),
                                          "NAS100": (-0.3, 200.0, 200.0, 1.0)})
    [c] = MrTwap01(mcfg).decide(v)
    assert (c.key, c.side) == ("US30", BUY)
    assert c.entry == 100.0 and c.stop == 97.0 and c.take_profit == 104.0 and c.atr == 2.0
    assert c.order_type == MARKET_OR_LIMIT and c.chase_limit is None
    assert c.skip_level == 100.0 - 0.5 * 3.0 and c.sizing_entry == 100.0
    assert c.signal_time == ts("2024-03-12 15:00")
    assert c.expires_utc == ts("2024-03-12 16:00")
    assert c.time_exit_utc == ts("2024-03-12 19:45")
    assert c.rank_score == 0.8 and c.regime == "CALM/CALM_FLAT" and c.regime_prob == 0.8
    assert c.reason == ("MR-TWAP-01 LONG US30: Z -2.00 from session mean 104.00, strongest, "
                        "RS_day +0.80, regime CALM. Limit at entry if price is worse; expires "
                        "16:00 UTC. MANUAL TIME EXIT 19:45 UTC (15:45 New York) if still open.")


def test_short_trigger_and_reason_weakest(mcfg):
    v = _view(mcfg, "2024-01-10 19:30", {"US30": (0.8, 100.0, 100.0, 2.0),
                                          "NAS100": (-0.3, 203.0, 200.0, 1.234567)})
    [c] = MrTwap01(mcfg).decide(v)
    assert (c.key, c.side) == ("NAS100", SELL)
    assert c.stop == round_away(203.0 + 1.5 * 1.234567, 203.0) == 204.86
    assert c.take_profit == 200.0
    assert c.skip_level == pytest.approx(203.0 + 0.5 * (204.86 - 203.0))
    assert c.expires_utc == ts("2024-01-10 20:15")             # time exit 20:45 - 30
    assert c.time_exit_utc == ts("2024-01-10 20:45")
    assert "SHORT NAS100: Z +2.43 from session mean 200.00, weakest, RS_day -0.30" in c.reason
    assert "MANUAL TIME EXIT 20:45 UTC (15:45 New York)" in c.reason


def test_reason_prints_the_tp_not_nearest_rounded_twap(mcfg):
    # TWAP 104.0075: nearest rounding prints 104.01, TP rounded toward entry (100) is 104.00.
    # The ticket must show one session-mean number, the TP.
    v = _view(mcfg, "2024-03-12 15:00", {"US30": (0.8, 100.0, 104.0075, 2.0),
                                          "NAS100": (-0.3, 200.0, 200.0, 1.0)})
    [c] = MrTwap01(mcfg).decide(v)
    assert f"{104.0075:.2f}" == "104.01" and c.take_profit == 104.0
    assert "from session mean 104.00, strongest" in c.reason
    # SHORT side: TWAP below entry rounds up toward entry.
    v = _view(mcfg, "2024-01-10 19:30", {"US30": (0.8, 100.0, 100.0, 2.0),
                                          "NAS100": (-0.3, 203.0, 199.9925, 1.0)})
    [c] = MrTwap01(mcfg).decide(v)
    assert f"{199.9925:.2f}" == "199.99" and c.take_profit == 200.0
    assert "from session mean 200.00, weakest" in c.reason


def test_just_inside_threshold_gives_nothing(mcfg):
    v = _view(mcfg, "2024-03-12 15:00", {"US30": (0.8, 100.01, 104.0, 2.0),
                                          "NAS100": (-0.3, 203.99, 200.0, 2.0)})
    assert MrTwap01(mcfg).decide(v) == []


def test_side_follows_selection_not_stretch(mcfg):
    """The strongest index stretched UP and the weakest stretched DOWN give nothing: only
    LONG-SELECTED may buy a dip and only SHORT-SELECTED may sell a rally (no promotion)."""
    v = _view(mcfg, "2024-03-12 15:00", {"US30": (0.8, 110.0, 100.0, 2.0),
                                          "NAS100": (-0.3, 190.0, 200.0, 2.0)})
    assert MrTwap01(mcfg).decide(v) == []


def test_both_sides_fire_on_different_indices(mcfg):
    v = _view(mcfg, "2024-03-12 15:00", {"US30": (-0.1, 100.0, 105.0, 2.0),
                                          "NAS100": (0.4, 210.0, 200.0, 2.0)})
    # NAS100 is the strongest (long) but stretched up; US30 the weakest but stretched down.
    assert MrTwap01(mcfg).decide(v) == []
    # Now NAS100 stretched down and US30 stretched up: both fire, in config order.
    v = _view(mcfg, "2024-03-12 15:00", {"US30": (-0.1, 110.0, 105.0, 2.0),
                                          "NAS100": (0.4, 190.0, 200.0, 2.0)})
    cs = MrTwap01(mcfg).decide(v)
    assert [(c.key, c.side) for c in cs] == [("US30", SELL), ("NAS100", BUY)]  # config order
    assert all("strongest" in c.reason or "weakest" in c.reason for c in cs)


def test_two_passing_us_indices_no_sign_condition(mcfg):
    """Spec: with two passing US indices the stronger may be bought even if both fell."""
    v = _view(mcfg, "2024-03-12 15:00", {"US30": (-0.2, 100.0, 104.0, 2.0),
                                          "NAS100": (-0.9, 200.0, 200.0, 2.0)})
    [c] = MrTwap01(mcfg).decide(v)
    assert (c.key, c.side, c.rank_score) == ("US30", BUY, -0.2)


def test_sole_index_sign_rule_and_reason(mcfg):
    T = "2024-03-12 13:00"                                   # GER40 window, CET
    [c] = MrTwap01(mcfg).decide(_view(mcfg, T, {"GER40": (0.5, 100.0, 104.0, 2.0)}))
    assert c.side == BUY and "LONG GER40" in c.reason and "sole index, RS_day +0.50" in c.reason
    assert "MANUAL TIME EXIT 16:15 UTC (17:15 Berlin)" in c.reason
    assert c.expires_utc == ts("2024-03-12 14:00")
    # Falling GER40 stretched down: the sole index may only be SOLD, so no candidate.
    assert MrTwap01(mcfg).decide(_view(mcfg, T, {"GER40": (-0.5, 100.0, 104.0, 2.0)})) == []
    [c] = MrTwap01(mcfg).decide(_view(mcfg, T, {"GER40": (-0.5, 108.0, 104.0, 2.0)}))
    assert c.side == SELL and "sole index, RS_day -0.50" in c.reason
    # Score exactly 0: neither side.
    assert MrTwap01(mcfg).decide(_view(mcfg, T, {"GER40": (0.0, 100.0, 104.0, 2.0)})) == []


def test_one_per_session_and_occupancy_block_without_promotion(mcfg):
    m = {"US30": (0.8, 100.0, 104.0, 2.0), "NAS100": (-0.3, 196.0, 200.0, 2.0)}
    # NAS100 is stretched down but is SHORT-SELECTED: never promoted when US30 is blocked.
    for book in (_Book(counted={"US30"}), _Book(occupied={"US30"})):
        assert MrTwap01(mcfg).decide(_view(mcfg, "2024-03-12 15:00", m, book)) == []
    assert len(MrTwap01(mcfg).decide(_view(mcfg, "2024-03-12 15:00", m,
                                           _Book(counted={"NAS100"})))) == 1


def test_zero_atr_gives_no_trigger(mcfg):
    v = _view(mcfg, "2024-03-12 15:00", {"US30": (0.8, 100.0, 104.0, 0.0),
                                          "NAS100": (-0.3, 200.0, 200.0, 2.0)})
    assert MrTwap01(mcfg).decide(v) == []


def test_params_k_and_m_change_trigger_and_stop(mcfg):
    m = {"US30": (0.8, 100.0, 103.5, 2.0), "NAS100": (-0.3, 200.0, 200.0, 2.0)}   # Z = -1.75
    assert MrTwap01(mcfg).decide(_view(mcfg, "2024-03-12 15:00", m)) == []
    [c] = MrTwap01(mcfg).decide(_view(mcfg, "2024-03-12 15:00", m, params={"K": 1.6, "M": 1.0}))
    assert c.stop == 98.0 and c.take_profit == 103.5


def test_incomplete_cause_twap_and_atr(mcfg):
    s = MrTwap01(mcfg)
    v = _view(mcfg, "2024-03-12 15:00", {"US30": (0.8, 100.0, float("nan"), 2.0)})
    assert s.incomplete_cause(v, "US30").startswith("TWAP NaN")
    v = _view(mcfg, "2024-03-12 15:00", {"US30": (0.8, 100.0, 101.0, float("nan"))})
    assert s.incomplete_cause(v, "US30").startswith("ATR14_M15 NaN")
    v = _view(mcfg, "2024-03-12 15:00", {"US30": (0.8, 100.0, 101.0, 2.0)})
    assert s.incomplete_cause(v, "US30") is None


# ---------------------------------------------------------------------------------------------
# Scan-level rules on synthetic markets
# ---------------------------------------------------------------------------------------------

REASON = re.compile(
    r"^MR-TWAP-01 (LONG|SHORT) (US30|NAS100|GER40): Z ([+-]\d+\.\d\d) from session mean "
    r"(-?\d+\.\d\d), (strongest|weakest|sole index), RS_day ([+-]\d+\.\d\d), regime CALM\. "
    r"Limit at entry if price is worse; expires (\d\d:\d\d) UTC\. MANUAL TIME EXIT (\d\d:\d\d) "
    r"UTC \((15:45 New York|17:15 Berlin)\) if still open\.$")


def _check_scan(cfg, svc, cycles, params):
    """Replays the cycles and checks every candidate against the spec AND that every setup the
    spec allows produced a candidate (completeness). Returns the candidates."""
    K, M = params["K"], params["M"]
    fs = svc[0]
    book = LockoutBook("CHECK", {k: fs.get(k).m15 for k in KEYS})
    out = []
    for sel, cands in cycles:
        T = sel.T
        got = {(c.key, c.side): c for c in cands}
        assert len(got) == len(cands)
        assert [c.key for c in cands] == sorted((c.key for c in cands), key=KEYS.index)
        expected = set()
        if sel.has_side:
            for side, key in ((BUY, sel.long), (SELL, sel.short)):
                if key is None:
                    continue
                fx = fs.get(key)
                row = fx.m15_at(T)
                ld = T.tz_convert(cfg["exchanges"][key]["tz"]).date()
                if book.count(key, ld) or book.occupied(key, T):
                    continue
                z = (row["close"] - row["twap"]) / row["atr14"]
                if (side == BUY and z <= -K) or (side == SELL and z >= K):
                    expected.add((key, side))
        assert set(got) == expected, (T, got.keys(), expected)
        for c in cands:
            fx = fs.get(c.key)
            row = fx.m15_at(T)
            # RS_day of the aligned H1 bar is the rank score (COMMON 4.1, 2).
            hpos, _ = fx.h1_positions(pd.DatetimeIndex([T]))
            assert sel.scores[c.key] == fx.h1["rs_day"].iloc[hpos[0]]
            assert c.rank_score == round(sel.scores[c.key], 2)
            assert c.entry == row["close"] and c.atr == row["atr14"]
            sg = 1 if c.side == BUY else -1
            assert c.stop == round_away(c.entry - sg * M * c.atr, c.entry)
            assert c.take_profit == round_toward(row["twap"], c.entry)
            assert sg * (c.take_profit - c.entry) >= K * c.atr - 0.01
            assert 0 <= sg * (c.entry - c.stop) - M * c.atr < 0.01
            lv = protocol_levels(MARKET_OR_LIMIT, c.side, c.entry, c.stop, cfg)
            assert (c.chase_limit, c.skip_level, c.sizing_entry) == \
                (None, lv.skip_level, c.entry)
            assert c.order_type == MARKET_OR_LIMIT
            assert c.expires_utc == expiry_utc(cfg, c.key, T)
            assert c.regime.startswith("CALM/") and sel.allows(c.key, c.side)
            m = REASON.match(c.reason)
            assert m, c.reason
            assert m.group(1) == ("LONG" if c.side == BUY else "SHORT") and m.group(2) == c.key
            word = "sole index" if sel.sole else ("strongest" if c.side == BUY else "weakest")
            assert m.group(5) == word
            assert m.group(4) == f"{c.take_profit:.2f}"            # same mean as the TP
            assert m.group(7) == f"{c.expires_utc:%H:%M}" and m.group(8) == \
                f"{c.time_exit_utc:%H:%M}"
            book.add(c, T.tz_convert(cfg["exchanges"][c.key]["tz"]).date())
            out.append(c)
    return out


def test_scan_rules_and_completeness_proxy_regime(mcfg, market, services):
    s = MrTwap01(mcfg, LOW_K, off_grid=True)
    cycles = list(s.cycles(*services, market.scan_start, market.end))
    cands = _check_scan(mcfg, services, cycles, s.params)
    assert len(cands) >= 10
    assert {c.side for c in cands} == {BUY, SELL} and {c.key for c in cands} == set(KEYS)


def test_scan_rules_and_completeness_always_calm(mcfg, market):
    svc = calm_services(mcfg, market)
    s = MrTwap01(mcfg, {"K": 0.5}, off_grid=True)
    cycles = list(s.cycles(*svc, market.scan_start, market.end))
    cands = _check_scan(mcfg, svc, cycles, s.params)
    assert len(cands) >= 40
    # One per index per session (local date), whichever side: and it binds (some selected,
    # stretched index got nothing later on the same date because of it).
    per = Counter((c.key, c.signal_time.tz_convert(mcfg["exchanges"][c.key]["tz"]).date())
                  for c in cands)
    assert max(per.values()) == 1
    blocked = 0
    fs = svc[0]
    for sel, cs in cycles:
        for side, key in ((BUY, sel.long), (SELL, sel.short)):
            if key is None or any(c.key == key for c in cs):
                continue
            row = fs.get(key).m15_at(sel.T)
            z = (row["close"] - row["twap"]) / row["atr14"]
            ld = sel.T.tz_convert(mcfg["exchanges"][key]["tz"]).date()
            if ((side == BUY and z <= -0.5) or (side == SELL and z >= 0.5)) and per[(key, ld)]:
                blocked += 1
    assert blocked > 0


def test_default_parameters_scan_is_consistent(mcfg, market, services):
    s = MrTwap01(mcfg)
    cycles = list(s.cycles(*services, market.scan_start, market.end))
    _check_scan(mcfg, services, cycles, s.params)


def test_regime_gate_calm_only(mcfg, market):
    s = MrTwap01(mcfg, {"K": 0.5}, off_grid=True)
    active = calm_services(mcfg, market, **{k: ConstantRegime(0.2) for k in KEYS})
    assert s.scan(*active, market.scan_start, market.end) == []
    off = calm_services(mcfg, market, **{k: ConstantRegime(-1, unavailable=True) for k in KEYS})
    cycles = list(s.cycles(*off, market.scan_start, market.end))
    assert not any(cs for _, cs in cycles)
    reasons = [r for sel, _ in cycles if not sel.incomplete for r in sel.reasons.values()]
    assert reasons and all(r.startswith("regime unavailable") for r in reasons)
    # V_T exactly 0 is ACTIVE (V_T < 0 is CALM).
    zero = calm_services(mcfg, market, **{k: ConstantRegime(0.0) for k in KEYS})
    assert s.scan(*zero, market.scan_start, market.end) == []


def test_us_sole_index_when_other_fails_gate(mcfg, market):
    """NAS100 ACTIVE, US30 CALM: E(T) = {US30} so US30 is the sole index (sign rule) and NAS100
    never gets a candidate."""
    svc = calm_services(mcfg, market, NAS100=ConstantRegime(0.3))
    s = MrTwap01(mcfg, {"K": 0.5}, off_grid=True)
    cycles = list(s.cycles(*svc, market.scan_start, market.end))
    cands = [c for _, cs in cycles for c in cs]
    us = [c for c in cands if c.key != "GER40"]
    assert us and all(c.key == "US30" for c in us)
    by_T = {(sel.T, k): sel for sel, _ in cycles for k in sel.members}
    for c in us:
        sel = by_T[(c.signal_time, "US30")]
        assert sel.eligible == ("US30",) and "sole index" in c.reason
        assert (sel.scores["US30"] > 0) == (c.side == BUY)


def test_incomplete_groups_from_data_defects(mcfg, market, services):
    s = MrTwap01(mcfg, LOW_K, off_grid=True)
    sels = {(sel.T, sel.members): sel for sel, _ in s.cycles(*services, market.scan_start,
                                                              market.end)}
    us = ("US30", "NAS100")
    # NAS100 bar 15:30-15:45 UTC missing on 2024-02-22: incomplete US group at 15:45 only.
    a = sels[(ts("2024-02-22 15:45"), us)]
    assert a.incomplete == f"incomplete group: NAS100: {NO_BAR}" and not a.has_side
    assert not sels[(ts("2024-02-22 16:00"), us)].incomplete
    # US30 cash-open bar missing on 2024-03-05: not a valid trading date for US30, so US30 is
    # outside G(T) (COMMON 4.2.1 + 1.6, Selection.excluded) and NAS100 is ranked alone under
    # the sole-index sign rule (4.2.5) at all 17 US decision times of that date.
    day = [sel for (T, g), sel in sels.items()
           if "NAS100" in g and T.tz_convert("America/New_York").date() == date(2024, 3, 5)]
    assert len(day) == 17
    for sel in day:
        assert sel.members == ("NAS100",) and dict(sel.excluded) == {"US30": INVALID_DATE}
        assert "outside G(T)" in sel.log_line()
        assert sel.long in (None, "NAS100") and sel.short in (None, "NAS100")
        if not sel.incomplete and sel.eligible == ("NAS100",):
            sc = sel.scores["NAS100"]
            assert (sel.long == "NAS100") == (sc > 0) and (sel.short == "NAS100") == (sc < 0)
    # GER40 has no cycle inside its > 7-day data hole (no cash-open bar: no valid date).
    assert not [T for (T, g) in sels if g == ("GER40",)
                and ts("2024-02-26") <= T < ts("2024-03-05")]
    # GER40 segment restart after the hole: RS_day needs 24 H1 bars inside the segment.
    ger = sels[(ts("2024-03-05 09:00"), ("GER40",))]
    assert ger.incomplete and "RS_day" in ger.incomplete


def test_missing_h1_bar_makes_rs_day_unavailable_live_matches_scan(mcfg):
    """US30's core H1 bar 10:00-11:00 New York (14:00-15:00 UTC, EDT) never arrives while its
    M15 bars do. RS_day is H1-based, so the COMMON 2 alignment rule makes it UNAVAILABLE from
    T = 15:00 until the next H1 bar closes at 16:00: the whole US group is incomplete (4.2.2,
    no partial ranking), which takes precedence over the regime's "H1 bar missing" status at
    15:00. Live generate() agrees with scan() at those T."""
    mk = make_market("2024-02-12", "2024-03-16", h1_drops={"US30": ["2024-03-12 14:00"]})
    s = MrTwap01(mcfg, {"K": 0.5}, off_grid=True)
    svc = build_services(mcfg, mk)
    sels = {sel.T: sel for sel, _ in s.cycles(*svc, mk.scan_start, mk.end)
            if "US30" in sel.members}
    for m in (0, 15, 30, 45):
        sel = sels[ts("2024-03-12 15:00") + pd.Timedelta(minutes=m)]
        assert sel.incomplete and sel.incomplete.startswith("incomplete group: US30: H1 inputs "
                                                            "unavailable")
        assert not sel.has_side
    assert not sels[ts("2024-03-12 16:00")].incomplete
    assert not sels[ts("2024-03-12 14:45")].incomplete
    full = s.scan(*svc, mk.scan_start, mk.end)
    assert_generate_matches_scan(s, mcfg, mk, [ts("2024-03-12 15:00"), ts("2024-03-12 15:15"),
                                               ts("2024-03-12 16:00")], full=full)


def test_scan_selection_sink_gets_every_cycle(mcfg, market, services):
    """COMMON 4.2.8: a feasibility / backtest run receives every cycle's Selection."""
    s = MrTwap01(mcfg)
    got = []
    cands = s.scan(*services, market.scan_start, market.end, on_selection=got.append)
    sels = [sel for sel, _ in s.cycles(*services, market.scan_start, market.end)]
    assert [x.log_line() for x in got] == [x.log_line() for x in sels] and len(got) > 1000
    assert all(x.strategy_id == STRATEGY_ID for x in got)
    assert all(any(c.signal_time == x.T for x in got) for c in cands)


def test_groups_ranked_separately_in_mismatch_weeks(mcfg, market, autumn):
    """GER40 is ranked alone in its window, so its candidates are those of a GER40-only run and
    the US candidates those of a US-only run, including the shared 14:30-15:00 UTC times."""
    class UsOnly(MrTwap01):
        def windows(self, params):
            return {k: w for k, w in super().windows(params).items() if k != "GER40"}

    class GerOnly(MrTwap01):
        def windows(self, params):
            return {"GER40": super().windows(params)["GER40"]}

    for mk in (market, autumn):
        svc = calm_services(mcfg, mk)
        p = {"K": 0.5}
        full = MrTwap01(mcfg, p, off_grid=True).scan(*svc, mk.scan_start, mk.end)
        us = UsOnly(mcfg, p, off_grid=True).scan(*svc, mk.scan_start, mk.end)
        ger = GerOnly(mcfg, p, off_grid=True).scan(*svc, mk.scan_start, mk.end)
        assert_candidates_equal([c for c in full if c.key != "GER40"], us, "US group")
        assert_candidates_equal([c for c in full if c.key == "GER40"], ger, "GER40 alone")
        shared = [c for c in full if c.signal_time.time() in (time(14, 30), time(14, 45),
                                                              time(15, 0))
                  and len(window_groups(mcfg, MrTwap01(mcfg).windows(PARAMS.defaults),
                                        c.signal_time)) == 2]
        assert shared, "the test market must have a candidate at a shared T"


def test_time_exit_fields_follow_dst(mcfg, market, autumn):
    for mk in (market, autumn):
        svc = calm_services(mcfg, mk)
        for c in MrTwap01(mcfg, {"K": 0.5}, off_grid=True).scan(*svc, mk.scan_start, mk.end):
            tz = mcfg["exchanges"][c.key]["tz"]
            local = c.time_exit_utc.tz_convert(tz)
            assert local.time() == time.fromisoformat(mcfg["time_exits"][c.key])
            assert local.date() == c.signal_time.tz_convert(tz).date()
            assert c.expires_utc <= c.time_exit_utc - pd.Timedelta(minutes=30)
            assert c.expires_utc <= c.signal_time + pd.Timedelta(minutes=60)


def test_candidates_pass_the_ticket_engine_unchanged(mcfg, market):
    """The engine's protocol check (signals.engine.sanitize_candidate) accepts every candidate
    as is: no level is recomputed and nothing is suppressed for invalid protocol levels. The
    placement window VALID UNTIL = min(T + 45 min, EXPIRY) (COMMON 7, ER-02)."""
    from signals.engine import sanitize_candidate, valid_until
    svc = calm_services(mcfg, market)
    cands = MrTwap01(mcfg, {"K": 0.5}, off_grid=True).scan(*svc, market.scan_start, market.end)
    assert cands
    for c in cands:
        out, why = sanitize_candidate(c, mcfg, c.signal_time)
        assert why == "" and out == c
        assert valid_until(c, mcfg, 15) == min(c.signal_time + pd.Timedelta(minutes=45),
                                               c.expires_utc)


# ---------------------------------------------------------------------------------------------
# Look-ahead and live == batch (testkit)
# ---------------------------------------------------------------------------------------------

def test_no_lookahead(mcfg, market):
    s = MrTwap01(mcfg, LOW_K, off_grid=True)
    full = s.scan(*build_services(mcfg, market), market.scan_start, market.end)
    times = sorted({c.signal_time for c in full})
    cuts = [times[1], times[len(times) // 2], ts("2024-03-13 14:45")]   # last: mismatch week
    got = assert_no_lookahead(s, mcfg, market, cuts)
    assert_candidates_equal(got, full)


def test_no_lookahead_autumn_always_calm(mcfg, autumn):
    s = MrTwap01(mcfg, {"K": 0.5}, off_grid=True)
    full = s.scan(*calm_services(mcfg, autumn), autumn.scan_start, autumn.end)
    times = sorted({c.signal_time for c in full})
    assert_no_lookahead(s, mcfg, autumn, [times[len(times) // 3], ts("2024-10-29 15:00")],
                        seeds=(3,), regime=const_regime())


def test_generate_matches_scan(mcfg, market):
    s = MrTwap01(mcfg, LOW_K, off_grid=True)
    full = s.scan(*build_services(mcfg, market), market.scan_start, market.end)
    times = sample_times(s, mcfg, full, market.scan_start, market.end, n_candidates=5)
    # Also the bar right after a candidate (same session, one-per-session replay) and a shared
    # mismatch-week T.
    times += [full[2].signal_time + pd.Timedelta(minutes=15), ts("2024-03-13 15:00")]
    n = assert_generate_matches_scan(s, mcfg, market, times, full=full)
    assert n >= 5


def test_generate_matches_scan_always_calm_including_blocked_times(mcfg, autumn):
    s = MrTwap01(mcfg, {"K": 0.5}, off_grid=True)
    svc = calm_services(mcfg, autumn)
    full = s.scan(*svc, autumn.scan_start, autumn.end)
    times = sample_times(s, mcfg, full, autumn.scan_start, autumn.end, n_candidates=4, seed=2)
    times += [c.signal_time + pd.Timedelta(minutes=30) for c in full[:3]]
    n = assert_generate_matches_scan(s, mcfg, autumn, times, regime=const_regime(), full=full)
    assert n >= 4


def test_mid_session_scan_start_equals_full_scan(mcfg, market):
    s = MrTwap01(mcfg, {"K": 0.5}, off_grid=True)
    svc = calm_services(mcfg, market)
    full = s.scan(*svc, market.scan_start, market.end)
    for c in full[1:6]:
        start = c.signal_time + pd.Timedelta(minutes=15)
        part = s.scan(*svc, start, market.end)
        assert_candidates_equal(part, [x for x in full if x.signal_time >= start], str(start))


def test_generate_without_services_returns_nothing(mcfg):
    T = ts("2024-03-12 15:00")
    assert MrTwap01(mcfg).generate({}, {}, T, StrategyContext(now=T, cfg=mcfg)) == []


def test_models_without_calm():
    @dataclass
    class M:
        vol_tags: list
        model_id: str

    @dataclass
    class R:
        model: object

    @dataclass
    class Tab:
        records: list

    t = Tab([R(M(["CALM", "ACTIVE"], "a")), R(M(["ACTIVE", "ACTIVE"], "b")), R(None)])
    assert models_without_calm(t) == {"b"}
    assert models_without_calm(ConstantRegime(-1)) == set()
