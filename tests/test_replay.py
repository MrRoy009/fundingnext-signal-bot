"""validation.replay on SYNTHETIC equity paths: windows, loss floors, Athens day boundary,
trade days, carried positions and edge cases."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.market import ServerClock
from validation.replay import (OPEN_EXIT_NOTES, ReplayRules, competition_replay,
                               replay_backtest, window_starts)

CLOCK = ServerClock("Europe/Athens", pd.Timedelta(0))
SPAN = (date(2024, 2, 1), date(2024, 4, 30))
RULES = ReplayRules()


def ns(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").value)


def hourly(start="2024-01-25", end="2024-05-05"):
    return pd.date_range(start, end, freq="1h", tz="UTC").as_unit("ns").asi8


def run(t, bal, eq, fills=(), exits=(), rules=RULES, span=SPAN, **kw):
    return competition_replay(t, bal, eq, np.asarray(fills, dtype="int64"),
                              np.asarray(exits, dtype="int64"), span[0], span[1], CLOCK, rules,
                              **kw)


def test_window_starts_mondays_full_windows_only():
    w = window_starts(*SPAN)
    assert w[0] == date(2024, 2, 5) and w[-1] == date(2024, 4, 1) and len(w) == 9
    assert all(d.weekday() == 0 for d in w)
    assert window_starts(date(2024, 2, 5), date(2024, 3, 5))[0] == date(2024, 2, 5)
    assert window_starts(date(2024, 2, 5), date(2024, 3, 4)) == []     # 29 days: no window


def test_flat_path_no_breach_and_passes():
    t = hourly()
    flat = np.full(len(t), 100_000.0)
    r = run(t, flat, flat)
    assert r.n_windows == 9 and r.n_total_breach == 0 and r.n_daily_breach == 0
    assert r.n_data_windows == 9 and r.n_no_data_windows == 0
    assert r.passes and r.min_trade_days_share == 0.0
    assert (r.windows["missing_weekdays"] == 0).all() and r.n_partial_data_windows == 0
    assert (r.windows["end_equity_raw"] == 100_000.0).all()
    assert r.windows["n_marks"].iloc[0] == 30 * 24


def test_rebasing_uses_window_start_equity():
    t = hourly()
    eq = 100_000.0 + 50.0 * np.arange(len(t))           # steady growth from a different base
    r = run(t, eq, eq)
    w0 = r.windows.iloc[0]
    t0 = CLOCK.midnight_utc_ns(date(2024, 2, 5))
    t_end = CLOCK.midnight_utc_ns(date(2024, 3, 6))
    i0 = np.searchsorted(t, t0, "right") - 1
    i1 = np.searchsorted(t, t_end, "right") - 1
    assert w0["end_equity_raw"] == pytest.approx(100_000 + eq[i1] - eq[i0])
    assert w0["bt_balance0"] == eq[i0] and w0["scale"] == pytest.approx(100_000 / eq[i0])
    assert w0["end_equity_norm"] == pytest.approx(100_000 + (eq[i1] - eq[i0]) * 1e5 / eq[i0])
    assert not r.windows["daily_breach"].any()


def test_balance_normalised_path_after_drawdown():
    """QA case: backtest balance 80,000 at the window start; the window loses $9,000 at that
    sizing. Raw minimum 91,000 (no 10% breach); at 100k sizing the loss is 11,250 (breach).
    The verdict is the union, so the window fails; the raw / norm columns show both. A daily
    dip of 4,500 at 80k sizing (5,625 at 100k) likewise breaches only on the norm path."""
    t = hourly()
    span = (date(2024, 2, 5), date(2024, 3, 5))                  # exactly one window
    bal = np.full(len(t), 80_000.0)
    loss_at = ns("2024-02-20 12:00")
    bal[t >= loss_at] -= 9_000.0
    r = run(t, bal, bal.copy(), span=span, start_balance=80_000.0)
    w = r.windows.iloc[0]
    assert w["bt_balance0"] == 80_000.0 and w["scale"] == pytest.approx(1.25)
    assert w["min_equity_raw"] == pytest.approx(91_000.0) and not w["total_breach_raw"]
    assert w["min_equity_norm"] == pytest.approx(88_750.0) and w["total_breach_norm"]
    assert w["total_breach"] and r.n_total_breach == 1 and not r.passes
    assert r.n_breach("total", "raw") == 0 and r.n_breach("total", "norm") == 1
    s = r.summary()
    assert s["total_loss_breaches_raw"] == 0 and s["total_loss_breaches_norm"] == 1
    assert s["min_bt_balance0"] == 80_000.0
    assert r.to_report_stats({"US30": 1.0}).total_loss_breach_windows == 1

    eq = np.full(len(t), 80_000.0)
    eq[t == loss_at] -= 4_500.0
    r = run(t, eq.copy(), eq, span=span, start_balance=80_000.0)
    w = r.windows.iloc[0]
    assert not w["daily_breach_raw"] and w["daily_breach_norm"] and w["daily_breach"]
    assert w["daily_breach_days"] == 1 and r.breach_share("daily", "raw") == 0.0
    assert r.daily_breach_share == 1.0 and not r.passes
    # Above 100,000 the raw path is the stricter one, and the union keeps it.
    up = np.full(len(t), 125_000.0)
    up[t == loss_at] -= 5_000.0
    w = run(t, np.full(len(t), 125_000.0), up, span=span).windows.iloc[0]
    assert w["daily_breach_raw"] and not w["daily_breach_norm"] and w["daily_breach"]
    with pytest.raises(ValueError, match="balance"):
        run(t, np.zeros(len(t)), np.zeros(len(t)), span=span, start_balance=0.0)


def _dip(t, at: str, amount: float):
    eq = np.full(len(t), 100_000.0)
    eq[t == ns(at)] -= amount
    return eq


def test_total_loss_floor_inclusive_and_window_count():
    t = hourly()
    bal = np.full(len(t), 100_000.0)
    r = run(t, bal, _dip(t, "2024-03-06 15:00", 10_000.0))
    # Windows containing 2024-03-06: starts Feb 12, 19, 26, Mar 4.
    hit = r.windows.loc[r.windows["total_breach"], "start"].tolist()
    assert hit == [date(2024, 2, 12), date(2024, 2, 19), date(2024, 2, 26), date(2024, 3, 4)]
    assert not r.passes
    r2 = run(t, bal, _dip(t, "2024-03-06 15:00", 9_999.0))
    assert r2.n_total_breach == 0 and r2.n_daily_breach == 4


def test_daily_and_personal_floors():
    t = hourly()
    bal = np.full(len(t), 100_000.0)
    r = run(t, bal, _dip(t, "2024-03-06 15:00", 5_000.0))        # floor 95,000: inclusive
    assert r.n_daily_breach == 4 and r.n_personal_breach == 4
    assert r.daily_breach_share == pytest.approx(4 / 9) and not r.passes
    r = run(t, bal, _dip(t, "2024-03-06 15:00", 3_500.0))
    assert r.n_daily_breach == 0 and r.n_personal_breach == 4
    assert r.passes                                              # personal stop is report only
    assert r.windows["personal_breach_days"].max() == 1


def test_daily_loss_base_choice_matters():
    """Window account grows to 120,000; a 5,500 intraday dip breaches with base =
    initial_balance (5,000 limit) but not with base = start_of_day_balance (6,000)."""
    t = hourly()
    t0 = CLOCK.midnight_utc_ns(date(2024, 2, 5))
    grow_at = ns("2024-02-06 12:00")
    bal = np.where(t >= grow_at, 120_000.0, 100_000.0)
    eq = bal.copy()
    eq[t == ns("2024-02-08 12:00")] -= 5_500.0
    span = (date(2024, 2, 5), date(2024, 3, 5))                  # exactly one window
    a = run(t, bal, eq, span=span)
    b = run(t, bal, eq, rules=ReplayRules(daily_loss_base="start_of_day_balance"), span=span)
    c = run(t, bal, eq, rules=ReplayRules(daily_loss_base="start_of_day_equity",
                                          daily_loss_reference="start_of_day_equity"), span=span)
    assert a.n_windows == 1 and t0 < grow_at
    assert a.n_daily_breach == 1 and b.n_daily_breach == 0 and c.n_daily_breach == 0
    assert a.n_personal_breach == 1 and b.n_personal_breach == 1   # 3% of 120k = 3,600


def test_day_boundary_is_athens_midnight():
    """Summer: 21:30 UTC is 00:30 Athens of the NEXT day, whose start-of-day balance already
    includes the 4,000 realised loss, so the extra 2,000 dip does not breach the 5% floor.
    Counted on the UTC day it would (min 94,000 <= 95,000)."""
    t = pd.date_range("2024-05-25", "2024-07-10", freq="30min", tz="UTC").as_unit("ns").asi8
    loss_at, dip_at = ns("2024-06-04 20:00"), ns("2024-06-04 21:30")
    bal = np.where(t >= loss_at, 96_000.0, 100_000.0)
    eq = bal.copy()
    eq[t == dip_at] -= 2_000.0
    span = (date(2024, 6, 3), date(2024, 7, 2))
    r = run(t, bal, eq, span=span)
    w = r.windows.iloc[0]
    assert r.n_windows == 1
    assert not w["daily_breach"] and not w["total_breach"]
    assert w["personal_breach"] and w["personal_breach_days"] == 1  # day 1: 96k < 97k
    assert w["min_equity_raw"] == 94_000.0
    # Same dip one hour earlier (23:30 Athens, still day 1) does breach.
    eq2 = bal.copy()
    eq2[t == ns("2024-06-04 20:30")] -= 2_000.0
    assert run(t, bal, eq2, span=span).windows.iloc[0]["daily_breach"]


def test_trade_days_counted_on_athens_dates_inside_window():
    t = hourly()
    flat = np.full(len(t), 100_000.0)
    fills = [ns(s) for s in ("2024-02-04 23:00",       # Monday 01:00 Athens (in window)
                             "2024-02-04 21:00",       # Sunday 23:00 Athens (before window)
                             "2024-02-06 10:00", "2024-02-06 14:00",   # same day once
                             "2024-02-07 10:00", "2024-02-08 10:00", "2024-02-09 10:00",
                             "2024-03-06 10:00")]       # after window 1 ends (Mar 5)
    exits = [f + 3_600_000_000_000 for f in fills]
    span = (date(2024, 2, 5), date(2024, 3, 5))
    r = run(t, flat, flat, fills, exits, span=span)
    w = r.windows.iloc[0]
    assert w["trade_days"] == 5 and w["min_trade_days_met"]
    assert r.min_trade_days_share == 1.0


def _carry_path(t, fill, exit_seen, u0, realised):
    """account.series convention: MTM at marks in [fill, exit_seen), balance moved at marks
    >= exit_seen. Unrealised is u0 throughout."""
    bal = np.where(t >= exit_seen, 100_000.0 + realised, 100_000.0)
    eq = bal.copy()
    eq[(t >= fill) & (t < exit_seen)] += u0
    return bal, eq


def test_carried_position_only_post_start_change_counts():
    t = hourly()
    t0 = CLOCK.midnight_utc_ns(date(2024, 2, 5))
    fill, exit_ = ns("2024-02-02 15:00"), ns("2024-02-05 15:00")
    bal, eq = _carry_path(t, fill, exit_, -1_000.0, -1_500.0)
    span = (date(2024, 2, 5), date(2024, 3, 5))
    r = run(t, bal, eq, [fill], [exit_], span=span)
    w = r.windows.iloc[0]
    assert t0 > fill and w["carried_in"] == 1 and not w["carry_approx"]
    assert w["end_equity_raw"] == pytest.approx(99_500.0)        # only the post-start -500
    assert not w["daily_breach"]
    with pytest.raises(ValueError, match="no filled trade is open"):
        run(t, bal, eq, span=span)                           # open P&L but no trade given
    # A trade whose exit is seen at the last pre-window mark (mark0 = t0) is not carried.
    assert ns("2024-02-04 22:00") == t0                      # 00:00 Athens in winter
    b2, e2 = _carry_path(t, fill, t0, -1_000.0, -1_500.0)
    assert run(t, b2, e2, [fill], [t0], span=span).windows.iloc[0]["carried_in"] == 0


def test_carried_close_stamped_exit_on_last_mark_of_day():
    """QA case: -2,000 carried over the weekend, SL stamped at the close of Monday's last bar
    (the 00:00 Athens mark, which belongs to Monday). The window balance is 100,000 again from
    that mark, so Tuesday's start-of-day balance is 100,000: personal floor 97,000 and daily
    floor 95,000. Switching one bar late (the old exit + 15 min rule) gives 98,000, floors
    95,060 / 93,000: a Tuesday low of 96,000 then misses the personal stop and 95,000 misses
    the daily floor."""
    t = pd.date_range("2024-01-25", "2024-03-10", freq="15min", tz="UTC").as_unit("ns").asi8
    span = (date(2024, 2, 5), date(2024, 3, 5))
    fill = ns("2024-02-02 15:00")
    sl = CLOCK.midnight_utc_ns(date(2024, 2, 6))             # 00:00 Athens close mark
    assert sl == ns("2024-02-05 22:00")
    bal, eq = _carry_path(t, fill, sl, -2_000.0, -2_000.0)
    for dip, col in ((4_000.0, "personal_breach"), (5_000.0, "daily_breach")):
        e = eq.copy()
        e[t == ns("2024-02-06 10:00")] -= dip
        for rules in (RULES, ReplayRules(daily_loss_base="start_of_day_balance")):
            w = run(t, bal, e, [fill], [sl], span=span, rules=rules).windows.iloc[0]
            assert w["carried_in"] == 1 and w["min_equity_raw"] == 100_000.0 - dip
            assert w[col] and not w["total_breach"]
        late = run(t, bal, e, [fill], [sl], span=span, exit_at_open=[True]).windows.iloc[0]
        assert not late[col]                                 # one bar late: missed


def test_open_stamped_exit_switches_at_next_mark():
    """Time exit at Tuesday's 00:00 Athens open, open-stamped: the 00:00 mark (Monday's last)
    still carries the MTM and the old balance, so Monday's closing window balance is 100,000.
    Read as close-stamped it would be 102,000 and a Tuesday low of 98,500 would wrongly fall
    under the personal floor (98,940)."""
    t = pd.date_range("2024-01-25", "2024-03-10", freq="15min", tz="UTC").as_unit("ns").asi8
    span = (date(2024, 2, 5), date(2024, 3, 5))
    fill = ns("2024-02-02 15:00")
    x = CLOCK.midnight_utc_ns(date(2024, 2, 6))
    bal, eq = _carry_path(t, fill, x + 900_000_000_000, -2_000.0, -2_000.0)
    eq[t == ns("2024-02-06 10:00")] -= 1_500.0
    good = run(t, bal, eq, [fill], [x], span=span, exit_at_open=[True]).windows.iloc[0]
    assert good["carried_in"] == 1 and not good["personal_breach"]
    bad = run(t, bal, eq, [fill], [x], span=span).windows.iloc[0]
    assert bad["personal_breach"]                            # what the wrong stamp gives
    with pytest.raises(ValueError, match="one entry per trade"):
        run(t, bal, eq, [fill], [x], span=span, exit_at_open=[True, False])


def test_multi_carry_flagged_when_exits_differ():
    t = hourly()
    span = (date(2024, 2, 5), date(2024, 3, 5))
    f1, f2 = ns("2024-02-02 15:00"), ns("2024-02-02 16:00")
    x1, x2 = ns("2024-02-05 10:00"), ns("2024-02-06 10:00")
    b1, e1 = _carry_path(t, f1, x1, -500.0, -500.0)
    b2, e2 = _carry_path(t, f2, x2, -700.0, -700.0)
    bal, eq = b1 + b2 - 100_000.0, e1 + e2 - 100_000.0
    w = run(t, bal, eq, [f1, f2], [x1, x2], span=span).windows.iloc[0]
    assert w["carried_in"] == 2 and w["carry_approx"]
    assert w["end_equity_raw"] == pytest.approx(100_000.0)
    b3, e3 = _carry_path(t, f2, x1, -700.0, -700.0)
    bal, eq = b1 + b3 - 100_000.0, e1 + e3 - 100_000.0
    same = run(t, bal, eq, [f1, f2], [x1, x1], span=span).windows.iloc[0]
    assert same["carried_in"] == 2 and not same["carry_approx"]
    assert run(t, bal, eq, [f1, f2], [x1, x1], span=span).summary()["carry_approx_windows"] == 0


def test_open_exit_notes_match_fills_open_events():
    """replay_backtest infers exit_at_open from exit_note: the notes must be exactly those of
    backtest.fills._exit's open-phase events."""
    src = (Path(__file__).resolve().parents[1] / "backtest" / "fills.py").read_text("utf-8")
    found = set(re.findall(r'_OPEN,\s*_PRIO\[.*?\],\s*\w+,\s*"([^"]+)"', src, re.S))
    assert found == set(OPEN_EXIT_NOTES)


def test_no_marks_and_empty_inputs():
    empty = np.array([], dtype="int64")
    with pytest.raises(ValueError, match="no mark falls in the span"):
        run(empty, np.array([]), np.array([]))
    # Mark times in microseconds land in 1970: no mark in the span -> error, not a PASS.
    t = hourly()
    flat = np.full(len(t), 100_000.0)
    with pytest.raises(ValueError, match="no mark falls in the span"):
        run(t // 1_000, flat, flat)
    with pytest.raises(ValueError, match="no mark falls in the span"):
        run(t, flat, flat, span=(date(2025, 2, 1), date(2025, 4, 30)))
    # Marks in the span but none inside any window: every window is no_data -> not a pass.
    early = (t > CLOCK.midnight_utc_ns(date(2024, 2, 1))) & (
        t < CLOCK.midnight_utc_ns(date(2024, 2, 4)))
    r = run(t[early], flat[early], flat[early])
    assert r.n_windows == 9 and r.n_data_windows == 0 and r.windows["no_data"].all()
    assert not r.passes and np.isnan(r.daily_breach_share)
    short = run(t, flat, flat, span=(date(2024, 2, 1), date(2024, 2, 20)))
    assert short.n_windows == 0 and not short.passes
    with pytest.raises(ValueError):
        run(np.array([2, 1], dtype="int64"), np.ones(2), np.ones(2))


def test_share_uses_windows_with_data_only():
    """One breach among N < 100 data windows fails 10.5.9 even though k empty windows in a
    data hole would bring the all-window share to <= 1%."""
    span = (date(2022, 1, 3), date(2024, 1, 28))
    t = pd.date_range("2022-01-01", "2024-02-05", freq="1h", tz="UTC").as_unit("ns").asi8
    t = t[(t < ns("2023-06-01")) | (t >= ns("2023-11-20"))]          # hole Jun - mid-Nov
    bal = np.full(len(t), 100_000.0)
    eq = bal.copy()
    eq[t == ns("2022-01-03 12:00")] -= 5_000.0      # first day: only the first window has it
    r = run(t, bal, eq, span=span)
    w = r.windows
    assert r.n_daily_breach == 1 and r.n_total_breach == 0
    assert r.n_data_windows < 100 <= r.n_windows and r.n_no_data_windows > 0
    assert r.daily_breach_share_all <= 0.01 < r.daily_breach_share
    assert r.daily_breach_share == pytest.approx(1 / r.n_data_windows)
    assert not r.passes
    assert (w.loc[w["no_data"], "n_marks"] == 0).all()
    assert r.n_partial_data_windows > 0                       # windows touching the hole
    assert w.loc[w["partial_data"], "missing_weekdays"].min() > 0
    s = r.summary()
    assert s["data_windows"] == r.n_data_windows and s["passes"] is False
    assert s["daily_loss_breach_share_all_windows"] == r.daily_breach_share_all
    st = r.to_report_stats({"US30": 1.0})
    assert st.n_windows == r.n_data_windows and st.daily_loss_breach_windows == 1
    assert st.share(st.daily_loss_breach_windows) == pytest.approx(r.daily_breach_share)


def test_rules_from_cfg_and_validation(cfg):
    r = ReplayRules.from_cfg(cfg)
    assert r.initial_balance == 100_000 and r.max_loss_pct == 10 and r.daily_loss_pct == 5
    assert r.daily_loss_base == cfg["rules"]["daily_loss_base"]
    assert r.daily_loss_reference == cfg["rules"]["daily_loss_reference"]
    assert r.personal_daily_stop_pct == 3 and r.min_trade_days == 5 and r.window_days == 30
    with pytest.raises(ValueError):
        ReplayRules(daily_loss_base="equity")
    with pytest.raises(ValueError):
        ReplayRules(daily_loss_reference="initial_balance")


def test_replay_backtest_adapter(cfg):
    t = hourly()
    eq = _dip(t, "2024-03-06 15:00", 6_000.0)
    equity = pd.DataFrame({"time_utc": pd.to_datetime(t, utc=True),
                           "balance": np.full(len(t), 100_000.0), "equity": eq})
    trades = pd.DataFrame({"status": ["filled", "skipped"],
                           "fill_time": [pd.Timestamp("2024-03-06 14:00", tz="UTC"), pd.NaT],
                           "fill_at_open": [False, None],
                           "exit_time": [pd.Timestamp("2024-03-06 16:00", tz="UTC"), pd.NaT],
                           "exit_note": ["at the clock bar open", None]})
    res = SimpleNamespace(equity=equity, trades=trades, contract="competition",
                          filled=trades[trades["status"] == "filled"])
    r = replay_backtest(res, *SPAN, cfg)
    assert r.contract == "competition" and r.n_daily_breach == 4
    assert r.windows["trade_days"].sum() == 4
    s = r.summary()
    assert s["windows"] == 9 and s["daily_loss_breaches"] == 4 and s["passes"] is False
    no_note = SimpleNamespace(equity=equity, contract="demo",
                              filled=res.filled.drop(columns=["exit_note"]))
    with pytest.raises(ValueError, match="exit_at_open nor exit_note"):
        replay_backtest(no_note, *SPAN, cfg)
