"""COMMON 2 / 8.5 / 8.6 / 8.7 server-day and DST handling of the backtester (ER-15), on
SYNTHETIC bars only.

The other backtest tests use July 2024 (US and EU on summer time). Here the same checks run on
four Wednesdays whose UTC offsets differ:

| date       | New York | Athens (server) | time exit 15:45 NY | 00:00 server | break (UTC)  |
|------------|----------|-----------------|--------------------|--------------|--------------|
| 2024-07-10 | UTC-4    | UTC+3           | 19:45              | 21:00        | 20:45-22:00  |
| 2024-01-10 | UTC-5    | UTC+2           | 20:45              | 22:00        | 21:45-23:00  |
| 2024-03-13 | UTC-4    | UTC+2 (US-only) | 19:45              | 22:00        | 21:45-23:00  |
| 2024-10-30 | UTC-4    | UTC+2 (EU-only) | 19:45              | 22:00        | 21:45-23:00  |

The March and October weeks are the mismatch weeks where only one side has switched. Each case
is built so that applying the wrong offset moves the result.
"""

import numpy as np
import pandas as pd
import pytest

import backtest.engine as engine
from backtest.engine import run_backtest
from backtest.market import prepare_bars
from backtest.scenarios import SCENARIOS, Scenario
from backtest.synthetic import M15, flat_bars
from data.mt5_adapter import SymbolSpec
from strategies.base import MARKET_BAND, Candidate


def _spec(name):
    return SymbolSpec(name, 2, 0.01, 0.01, 0.01, 0.1, 250.0, 0.1, "USD", swap_long=-500.0,
                      swap_short=100.0, swap_mode=1, swap_rollover3days=3,
                      trade_stops_level=100)


SPECS = {k: _spec(k) for k in ("US30", "NAS100", "GER40")}
BASE = SCENARIOS["BASE"]
HOLD = Scenario(name="hold", missed_exit=True)

CASES = {
    #                (day,          time exit, 00:00 server, break start, break end)
    "summer":        ("2024-07-10", "19:45",   "21:00",      "20:45",     "22:00"),
    "winter":        ("2024-01-10", "20:45",   "22:00",      "21:45",     "23:00"),
    "march-us-only": ("2024-03-13", "19:45",   "22:00",      "21:45",     "23:00"),
    "oct-eu-only":   ("2024-10-30", "19:45",   "22:00",      "21:45",     "23:00"),
}


class Day:
    def __init__(self, name):
        d, te, mid, bs, be = CASES[name]
        ts = lambda hm: pd.Timestamp(f"{d} {hm}", tz="UTC")          # noqa: E731
        self.t = ts("14:00")
        self.time_exit, self.midnight = ts(te), ts(mid)
        self.break_start, self.break_end = ts(bs), ts(be)
        self.next_day = self.t + pd.Timedelta(days=1)
        self.end = ts("23:45") + pd.Timedelta(days=1)

    def bars(self, overrides=None, **kw):
        return flat_bars(self.t - M15, self.end, overrides=overrides, **kw)

    def at(self, minutes):
        return self.t + pd.Timedelta(minutes=minutes)


def cand(t, key="US30", side="BUY"):
    sgn = 1.0 if side == "BUY" else -1.0
    return Candidate(key=key, side=side, entry=40000.0, stop=40000.0 - sgn * 400.0,
                     take_profit=40000.0 + sgn * 300.0, atr=100.0, reason="t", signal_time=t,
                     order_type=MARKET_BAND)


def run(cands, bar_map, cfg, scenario=BASE):
    pairs = [("RS-MOM-01", c) for c in cands]
    return run_backtest(pairs, bar_map, cfg, scenario, SPECS)


@pytest.fixture(params=list(CASES))
def dd(request):
    return Day(request.param)


def test_daily_break_flag_follows_server_time(cfg, dd):
    df = dd.bars()
    b = prepare_bars("US30", df, cfg)
    t = pd.DatetimeIndex(df["time_utc"])
    expected = (t >= dd.break_start) & (t < dd.break_end)
    expected |= (t >= dd.break_start + pd.Timedelta(days=1)) & \
        (t < dd.break_end + pd.Timedelta(days=1))
    np.testing.assert_array_equal(b.in_break, np.asarray(expected))
    assert b.in_break.sum() == 2 * 5                    # 23:45-01:00 = five M15 bars a day


def test_time_exit_clock_follows_new_york_time(cfg, dd):
    r = run([cand(dd.t)], {"US30": dd.bars()}, cfg).trades.iloc[0]
    assert r.status == "filled"
    assert r.exit_type == "time_exit" and r.exit_time == dd.time_exit
    assert r.exit_price == pytest.approx(40000 - 2)


def test_sl_touched_in_the_daily_break_resolves_after_it(cfg, dd):
    bars = dd.bars({dd.midnight: (40000.0, 40005.0, 39500.0, 40000.0)})
    r = run([cand(dd.t)], {"US30": bars}, cfg, HOLD).trades.iloc[0]
    assert r.exit_type == "SL" and r.exit_time == dd.break_end
    assert r.exit_price == pytest.approx(39600 - 2)


def test_tp_just_before_server_midnight_pays_no_swap(cfg, dd):
    tp_bar = dd.midnight - 2 * M15                      # closes 15 min before 00:00 server
    bars = dd.bars({tp_bar: (40000.0, 40400.0, 39995.0, 40300.0)})
    r = run([cand(dd.t)], {"US30": bars}, cfg, HOLD).trades.iloc[0]
    assert (r.exit_type, r.exit_time) == ("TP", tp_bar + M15)
    assert r.rollovers == 0 and r.swap_usd == 0.0


def test_swap_rollover_daily_series_and_mtm_at_server_midnight(cfg, dd):
    # the bar closing exactly at 00:00 server dips (SL not hit): it ends day 1
    last = dd.midnight - M15
    bars = dd.bars({last: (40000.0, 40005.0, 39900.0, 40000.0),
                    dd.next_day: (40000.0, 40400.0, 39990.0, 40300.0)})
    res = run([cand(dd.t)], {"US30": bars}, cfg, HOLD)
    r = res.trades.iloc[0]
    assert r.exit_type == "TP" and r.rollovers == 1
    assert r.swap_usd == pytest.approx(15.0)            # -500 pts x 0.01 x 1 lot, Wed triple
    eq = res.equity.set_index("time_utc")
    assert eq.loc[dd.midnight, "equity"] == pytest.approx(100000 + (39900 - 40005))
    assert eq.loc[dd.midnight + M15, "equity"] == pytest.approx(100000 + (39995 - 40005) - 15)
    d = res.daily.set_index("date")
    day1, day2 = dd.t.date(), dd.next_day.date()
    assert d.loc[day1, "equity"] == pytest.approx(100000 + (39900 - 40005))
    assert d.loc[day1, "balance"] == 100000.0
    assert d.loc[day2, "balance"] == pytest.approx(100000 + r.pnl_usd)


def test_account_state_day_roll_at_server_midnight(cfg, dd, monkeypatch):
    seen = []
    real = engine.check_ticket
    monkeypatch.setattr(engine, "check_ticket", lambda **kw: seen.append(kw) or real(**kw))
    us = dd.bars({dd.at(60): (40000.0, 40005.0, 39600.0, 39700.0)})   # US30: SL, day 1
    before, after = dd.midnight - 2 * M15, dd.break_end
    res = run([cand(dd.t), cand(before, "NAS100"), cand(after, "GER40")],
              {"US30": us, "NAS100": dd.bars(), "GER40": dd.bars()}, cfg)
    loss = res.trades.iloc[0].pnl_usd
    assert res.trades.iloc[0].exit_type == "SL" and loss < 0
    assert len(seen) == 3
    s1, s2 = seen[1]["state"], seen[2]["state"]
    assert seen[1]["now"] == before and seen[2]["now"] == after
    assert s1.trades_today == 1 and s1.start_of_day_balance == 100000.0   # still day 1
    assert s2.trades_today == 0                                            # new server day
    assert s2.start_of_day_balance == pytest.approx(100000 + loss)
