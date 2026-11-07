"""COMMON 8.1-8.5 fill model on SYNTHETIC bars only: delay and placement, the three protocols,
market / limit / stop fills, SL / TP bid-ask rules, gaps, SL first, the fill-bar rule, time
exits, the daily break, segment breaks, end of data and the stress scenarios.

Setup used throughout: US30 on Wednesday 2024-07-10 (US summer). T = 14:00 UTC, manual time
exit 15:45 New York = 19:45 UTC. Server time is Athens (UTC+3), so the daily break 23:45-01:00
server is 20:45-22:00 UTC and server midnight is 21:00 UTC. Raw spread 300 = 3.0 points, above
every US30 floor, so eff_spread = 3.0 x c. ATR 100 makes slip = s x max(0.5 floor, 2.0) = 2 s.
"""
from dataclasses import replace

import pandas as pd
import pytest

from backtest.engine import run_backtest
from backtest.scenarios import SCENARIOS, Scenario
from backtest.synthetic import M15, concat_bars, flat_bars
from data.mt5_adapter import SymbolSpec
from strategies.base import MARKET_BAND, MARKET_OR_LIMIT, STOP_ENTRY, Candidate

T = pd.Timestamp("2024-07-10 14:00", tz="UTC")
EXIT = pd.Timestamp("2024-07-10 19:45", tz="UTC")
SPEC = SymbolSpec("US30", 2, 0.01, 0.01, 0.01, 0.1, 250.0, 0.1, "USD", swap_long=-500.0,
                  swap_short=100.0, swap_mode=1, swap_rollover3days=3, trade_stops_level=100)
BASE = SCENARIOS["BASE"]
SID = "RS-MOM-01"


def at(minutes: int) -> pd.Timestamp:
    return T + pd.Timedelta(minutes=minutes)


def day(overrides=None, base=40000.0, end=pd.Timestamp("2024-07-10 23:00", tz="UTC"), **kw):
    """Bars from the signal bar (open T - 15 min) to `end`."""
    return flat_bars(T - M15, end, base=base, overrides=overrides, **kw)


def long_band(**over):
    c = dict(key="US30", side="BUY", entry=40000.0, stop=39600.0, take_profit=40300.0,
             atr=100.0, reason="t", signal_time=T, order_type=MARKET_BAND)
    c.update(over)
    return Candidate(**c)


def short_band(**over):
    return long_band(side="SELL", stop=40400.0, take_profit=39700.0, **over)


def run(c, bars, scenario=BASE, cfg=None, spec=SPEC, sid=SID):
    res = run_backtest([(sid, c)], {"US30": bars}, cfg, scenario, {"US30": spec})
    assert len(res.trades) == 1
    return res.trades.iloc[0]


# --- 8.3 delay, placement and validity ---------------------------------------------------
def test_market_buy_fills_at_ask_plus_slip_at_P(cfg):
    r = run(long_band(), day(), cfg=cfg)
    assert r.status == "filled" and r.entry_mode == "market"
    assert r.placement_time == at(30) and r.fill_time == at(30)       # d = 2
    assert r.fill_price == pytest.approx(40000 + 3 + 2)
    assert r.exit_type == "time_exit" and r.exit_time == EXIT
    assert r.exit_price == pytest.approx(40000 - 2)                    # sell at bid - slip


def test_market_sell_fills_at_bid_minus_slip(cfg):
    r = run(short_band(), day(), cfg=cfg)
    assert r.fill_price == pytest.approx(40000 - 2)
    assert r.exit_type == "time_exit"
    assert r.exit_price == pytest.approx(40000 + 3 + 2)                # buy at ask + slip


@pytest.mark.parametrize("name, p_min", [("STRESS-d1", 15), ("STRESS-d2", 30),
                                         ("STRESS-d3", 45)])
def test_delay_d_and_stressed_spread_and_slip(cfg, name, p_min):
    r = run(long_band(), day(), SCENARIOS[name], cfg=cfg)
    assert r.placement_time == at(p_min)
    assert r.fill_price == pytest.approx(40000 + 2 * 3 + 2 * 2)        # c = 2, s = 2


def test_placement_is_the_first_existing_bar_after_the_delay(cfg):
    r = run(long_band(), day(drop=[at(30)]), cfg=cfg)
    assert r.status == "filled" and r.placement_time == at(45)         # T + 45 is still valid


def test_placement_after_valid_until_is_skipped(cfg):
    r = run(long_band(), day(drop=[at(30), at(45)]), cfg=cfg)
    assert r.status == "skipped" and "VALID UNTIL" in r.reason
    assert r.placement_time == at(60)


def test_placement_at_or_after_expiry_is_skipped(cfg):
    c = long_band(order_type=MARKET_OR_LIMIT, expires_utc=at(30))
    r = run(c, day(), cfg=cfg)                                         # P = T + 30 = EXPIRY
    assert r.status == "skipped" and "VALID UNTIL" in r.reason


def test_placement_beyond_the_data_is_skipped(cfg):
    r = run(long_band(), flat_bars(T - M15, at(15)), cfg=cfg)
    assert r.status == "skipped" and "beyond the data" in r.reason


def test_placement_past_a_segment_break_is_skipped(cfg):
    later = flat_bars(T + pd.Timedelta(days=8), T + pd.Timedelta(days=8, hours=2))
    r = run(long_band(), concat_bars(flat_bars(T - M15, T - M15), later), cfg=cfg)
    assert r.status == "skipped" and "segment break" in r.reason


# --- COMMON 7 protocols at P -------------------------------------------------------------
@pytest.mark.parametrize("bid_open, status", [
    (40077.0, "filled"),      # ask 40080 = CHASE LIMIT: allowed
    (40078.0, "skipped"),     # ask 40081 beyond the chase limit
    (39797.0, "filled"),      # ask 39800 = SKIP LEVEL: allowed
    (39796.0, "skipped"),     # ask 39799: more than 0.50 D better
])
def test_market_band_chase_and_skip_levels(cfg, bid_open, status):
    bars = day({at(30): (bid_open, bid_open + 1, bid_open - 1, bid_open)})
    r = run(long_band(), bars, cfg=cfg)
    assert r.status == status
    if status == "filled":
        assert r.fill_price == pytest.approx(bid_open + 3 + 2)


def test_market_band_skips_when_quote_at_tp(cfg):
    c = long_band(take_profit=40050.0)
    r = run(c, day({at(30): (40047.0, 40048.0, 40046.0, 40047.0)}), cfg=cfg)
    assert r.status == "skipped" and "TP" in r.reason


def test_market_band_sell_uses_the_bid(cfg):
    # SELL chase limit 39920: bid 39919 is beyond it even though the ask (39922) is not
    r = run(short_band(), day({at(30): (39919.0, 39920.0, 39918.0, 39919.0)}), cfg=cfg)
    assert r.status == "skipped" and "CHASE" in r.reason


def limit_long(**over):
    return long_band(order_type=MARKET_OR_LIMIT, expires_utc=at(60), **over)


def test_market_or_limit_market_when_quote_at_or_better_than_e(cfg):
    r = run(limit_long(), day({at(30): (39990.0, 39995.0, 39985.0, 39990.0)}), cfg=cfg)
    assert r.entry_mode == "market" and r.fill_price == pytest.approx(39990 + 3 + 2)


def test_buy_limit_fills_at_e_with_penetration_and_no_slip(cfg):
    bars = day({at(30): (40010.0, 40015.0, 39996.0, 40000.0)}, base=40010.0)
    r = run(limit_long(), bars, cfg=cfg)
    assert r.entry_mode == "limit" and r.status == "filled"
    assert r.fill_price == 40000.0                                     # at E, no slippage
    assert r.fill_time == at(45) and not r.fill_at_open                # intrabar: bar close


def test_buy_limit_needs_one_cent_of_penetration(cfg):
    # ask low = 39996.995 + 3 = 39999.995 > E - 0.01: no fill; quiet bars at 40010 never fill
    bars = day({at(30): (40010.0, 40015.0, 39996.995, 40000.0)}, base=40010.0)
    r = run(limit_long(), bars, cfg=cfg)
    assert r.status == "cancelled" and "EXPIRY" in r.reason
    assert r.entry_mode == "limit"


def test_limit_window_ends_at_expiry(cfg):
    # touch in the bar opening at EXPIRY (T + 60) is outside [P, EXPIRY)
    bars = day({at(60): (40010.0, 40015.0, 39990.0, 40000.0)}, base=40010.0)
    assert run(limit_long(), bars, cfg=cfg).status == "cancelled"
    bars = day({at(45): (40010.0, 40015.0, 39990.0, 40000.0)}, base=40010.0)
    assert run(limit_long(), bars, cfg=cfg).status == "filled"


def test_sell_limit_fills_on_bid_high(cfg):
    c = short_band(order_type=MARKET_OR_LIMIT, expires_utc=at(60))
    bars = day({at(45): (39990.0, 40000.02, 39985.0, 39990.0)}, base=39990.0)
    r = run(c, bars, cfg=cfg)
    assert r.entry_mode == "limit" and r.fill_price == 40000.0
    bars = day({at(45): (39990.0, 40000.0, 39985.0, 39990.0)}, base=39990.0)
    assert run(c, bars, cfg=cfg).status == "cancelled"


def stop_long(**over):
    c = dict(key="US30", side="BUY", entry=40100.0, stop=39900.0, take_profit=40500.0,
             atr=100.0, reason="t", signal_time=T, order_type=STOP_ENTRY, expires_utc=at(120))
    c.update(over)
    return Candidate(**c)


def test_buy_stop_triggers_on_ask_high_and_fills_at_trigger_plus_slip(cfg):
    bars = day({at(45): (40000.0, 40097.0, 39995.0, 40050.0)})        # ask high 40100
    r = run(stop_long(), bars, cfg=cfg)
    assert r.entry_mode == "stop" and r.status == "filled"
    assert r.fill_price == pytest.approx(40100 + 2) and not r.fill_at_open
    bars = day({at(45): (40000.0, 40096.99, 39995.0, 40050.0)})       # ask high 40099.99
    assert run(stop_long(), bars, cfg=cfg).status == "cancelled"


def test_buy_stop_gap_fills_at_ask_open_plus_slip(cfg):
    bars = day({at(45): (40120.0, 40130.0, 40110.0, 40125.0)})
    r = run(stop_long(), bars, cfg=cfg)
    assert r.fill_price == pytest.approx(40120 + 3 + 2) and r.fill_at_open


def test_stop_entry_market_inside_buffer_and_skip_beyond_chase(cfg):
    # trigger 40100, stops_buffer = 100 points x 0.01 = 1.0, chase = 40140
    r = run(stop_long(), day({at(30): (40096.0, 40097.0, 40095.0, 40096.0)}), cfg=cfg)
    assert r.entry_mode == "market"                                    # ask 40099 >= 40099
    assert r.fill_price == pytest.approx(40099 + 2)
    r = run(stop_long(), day({at(30): (40095.0, 40096.0, 40094.0, 40000.0)}), cfg=cfg)
    assert r.entry_mode == "stop"                                      # ask 40098 < 40099
    r = run(stop_long(), day({at(30): (40138.0, 40139.0, 40137.0, 40138.0)}), cfg=cfg)
    assert r.status == "skipped" and "CHASE" in r.reason              # ask 40141 > 40140


def test_sell_stop_triggers_on_bid_low(cfg):
    c = stop_long(side="SELL", entry=39900.0, stop=40100.0, take_profit=39500.0)
    r = run(c, day({at(45): (40000.0, 40005.0, 39900.0, 39950.0)}), cfg=cfg)
    assert r.entry_mode == "stop" and r.fill_price == pytest.approx(39900 - 2)
    r = run(c, day({at(45): (39880.0, 39890.0, 39870.0, 39880.0)}), cfg=cfg)
    assert r.fill_price == pytest.approx(39880 - 2) and r.fill_at_open  # gap


@pytest.mark.parametrize("bid_open, mode, price", [
    (39900.5, "market", 39900.5 - 2),    # bid in (trigger, trigger + buffer]: market
    (39901.0, "market", 39901.0 - 2),    # bid = trigger + buffer: not strictly above
    (39899.5, "market", 39899.5 - 2),    # bid in (trigger - buffer, trigger]: market
    (39901.5, "stop", None),             # bid > trigger + buffer: sell stop order
])
def test_sell_stop_protocol_is_mirrored_with_the_bid(cfg, bid_open, mode, price):
    # COMMON 7: sell stop only if bid > trigger + stops_buffer (buffer = 100 x 0.01 = 1.0)
    c = stop_long(side="SELL", entry=39900.0, stop=40100.0, take_profit=39500.0)
    bars = day({at(30): (bid_open, bid_open + 0.5, bid_open - 0.5, bid_open)})
    r = run(c, bars, cfg=cfg)
    assert r.entry_mode == mode
    if mode == "market":
        assert r.status == "filled" and r.fill_time == at(30) and r.fill_at_open
        assert r.fill_price == pytest.approx(price)
    else:
        assert r.status == "cancelled"                    # bars stay above the trigger


def test_stop_entry_needs_the_stops_level(cfg):
    with pytest.raises(ValueError, match="stops"):
        run(stop_long(), day(), cfg=cfg, spec=replace(SPEC, trade_stops_level=None))


def test_unfilled_stop_is_cancelled_at_expiry(cfg):
    r = run(stop_long(), day(), cfg=cfg)
    assert r.status == "cancelled" and "EXPIRY" in r.reason
    assert pd.isna(r.fill_time) and pd.isna(r.pnl_usd)


# --- 8.5 exits ---------------------------------------------------------------------------
def test_long_sl_intrabar_and_gap(cfg):
    r = run(long_band(), day({at(60): (40000.0, 40005.0, 39600.0, 39700.0)}), cfg=cfg)
    assert (r.exit_type, r.exit_price, r.exit_time) == ("SL", pytest.approx(39598.0), at(75))
    r = run(long_band(), day({at(60): (39500.0, 39550.0, 39450.0, 39500.0)}), cfg=cfg)
    assert r.exit_type == "SL" and r.exit_price == pytest.approx(39498.0)


def test_long_tp_needs_one_cent_and_fills_exactly_even_on_a_gap(cfg):
    r = run(long_band(), day({at(60): (40000.0, 40300.0, 39995.0, 40000.0)}), cfg=cfg)
    assert r.exit_type == "time_exit"                                  # high = TP: not enough
    r = run(long_band(), day({at(60): (40000.0, 40300.005, 39995.0, 40000.0)}), cfg=cfg)
    assert r.exit_type == "time_exit"                                  # under 0.01 beyond
    r = run(long_band(), day({at(60): (40000.0, 40300.02, 39995.0, 40000.0)}), cfg=cfg)
    assert (r.exit_type, r.exit_price) == ("TP", 40300.0)
    r = run(long_band(), day({at(60): (40400.0, 40450.0, 40390.0, 40400.0)}), cfg=cfg)
    assert (r.exit_type, r.exit_price) == ("TP", 40300.0)              # gap: TP, not the open


def test_short_sl_on_the_ask_and_gap(cfg):
    r = run(short_band(), day({at(60): (40000.0, 40396.0, 39995.0, 40000.0)}), cfg=cfg)
    assert r.exit_type == "time_exit"                                  # ask high 40399
    r = run(short_band(), day({at(60): (40000.0, 40397.0, 39995.0, 40000.0)}), cfg=cfg)
    assert (r.exit_type, r.exit_price) == ("SL", pytest.approx(40402.0))
    r = run(short_band(), day({at(60): (40500.0, 40510.0, 40490.0, 40500.0)}), cfg=cfg)
    assert r.exit_price == pytest.approx(40500 + 3 + 2)


def test_short_tp_on_the_ask(cfg):
    r = run(short_band(), day({at(60): (40000.0, 40005.0, 39697.0, 39800.0)}), cfg=cfg)
    assert r.exit_type == "time_exit"                                  # ask low 39700
    r = run(short_band(), day({at(60): (40000.0, 40005.0, 39696.98, 39800.0)}), cfg=cfg)
    assert (r.exit_type, r.exit_price) == ("TP", 39700.0)


def test_sl_first_when_a_bar_reaches_both(cfg):
    r = run(long_band(), day({at(60): (40000.0, 40500.0, 39500.0, 40000.0)}), cfg=cfg)
    assert (r.exit_type, r.exit_price) == ("SL", pytest.approx(39598.0))


def test_fill_bar_tp_ignored_after_an_intrabar_stop_fill(cfg):
    bars = day({at(45): (40000.0, 40600.0, 39995.0, 40050.0)})
    r = run(stop_long(), bars, cfg=cfg)
    assert r.fill_time == at(60) and r.exit_type == "time_exit"


def test_fill_bar_sl_counts_after_an_intrabar_stop_fill(cfg):
    bars = day({at(45): (40000.0, 40101.0, 39890.0, 39950.0)})
    r = run(stop_long(), bars, cfg=cfg)
    assert (r.exit_type, r.exit_price) == ("SL", pytest.approx(39900 - 2))
    assert r.exit_time == at(60)


def test_fill_bar_tp_counts_when_filled_at_the_open(cfg):
    bars = day({at(45): (40120.0, 40600.0, 40110.0, 40500.0)})        # stop gap fill at open
    r = run(stop_long(), bars, cfg=cfg)
    assert r.fill_at_open and (r.exit_type, r.exit_price) == ("TP", 40500.0)
    r = run(long_band(), day({at(30): (40000.0, 40400.0, 39990.0, 40300.0)}), cfg=cfg)
    assert r.fill_time == at(30) and r.exit_type == "TP" and r.exit_time == at(45)


def test_limit_fill_bar_ignores_tp(cfg):
    c = limit_long(take_profit=40030.0)
    bars = day({at(30): (40010.0, 40040.0, 39990.0, 40000.0),        # fills and reaches TP
                at(45): (40010.0, 40031.0, 40005.0, 40010.0)}, base=40010.0)
    r = run(c, bars, cfg=cfg)
    assert r.fill_time == at(45) and r.exit_type == "TP"
    assert r.exit_time == at(60)                                       # the next bar's TP


def test_time_exit_missing_clock_bar_uses_last_close_before(cfg):
    bars = day({EXIT - M15: (40000.0, 40005.0, 39995.0, 40001.0)}, drop=[EXIT])
    r = run(long_band(), bars, cfg=cfg)
    assert r.exit_type == "time_exit" and r.exit_time == EXIT
    assert r.exit_price == pytest.approx(40001 - 2)
    r = run(short_band(), bars, cfg=cfg)
    assert r.exit_price == pytest.approx(40001 + 3 + 2)


def test_missed_exit_holds_overnight_to_tp(cfg):
    nxt = pd.Timestamp("2024-07-11 14:00", tz="UTC")
    bars = day({nxt: (40000.0, 40400.0, 39990.0, 40300.0)}, end=nxt + pd.Timedelta(hours=1))
    sc = SCENARIOS["MISSED-EXIT"]
    r = run(long_band(), bars, sc, cfg=cfg)
    assert r.exit_type == "TP" and r.exit_time == nxt + M15
    assert run(long_band(), bars, SCENARIOS["STRESS-d2"], cfg=cfg).exit_type == "time_exit"


def test_exit_3x_widens_sl_and_time_exit_fills_only(cfg):
    s2, s3 = SCENARIOS["STRESS-d2"], SCENARIOS["EXIT-3X"]
    r2, r3 = run(short_band(), day(), s2, cfg=cfg), run(short_band(), day(), s3, cfg=cfg)
    assert r2.fill_price == r3.fill_price == pytest.approx(40000 - 4)  # entry: c = 2
    assert r2.exit_price == pytest.approx(40000 + 6 + 4)
    assert r3.exit_price == pytest.approx(40000 + 9 + 4)               # time exit: c = 3
    gap = day({at(60): (40500.0, 40510.0, 40490.0, 40500.0)})
    assert run(short_band(), gap, s3, cfg=cfg).exit_price == pytest.approx(40500 + 9 + 4)
    # the SL trigger keeps c = 2: ask high 40393 + 6 = 40399 < SL
    near = day({at(60): (40000.0, 40393.0, 39995.0, 40000.0)})
    assert run(short_band(), near, s3, cfg=cfg).exit_type == "time_exit"
    # the short SL GAP test also keeps c = 2: ask open 40393 + 6 = 40399 < SL is no gap, so
    # the intrabar SL fills at SL + slip even though 40393 + 9 >= SL
    no_gap = day({at(60): (40393.0, 40398.0, 40390.0, 40393.0)})
    r = run(short_band(), no_gap, s3, cfg=cfg)
    assert (r.exit_type, r.exit_price) == ("SL", pytest.approx(40400 + 4))
    gap = day({at(60): (40395.0, 40398.0, 40390.0, 40395.0)})          # ask open 40401 >= SL
    assert run(short_band(), gap, s3, cfg=cfg).exit_price == pytest.approx(40395 + 9 + 4)
    # a long time exit sells at the bid: unchanged
    assert run(long_band(), day(), s3, cfg=cfg).exit_price == \
        run(long_band(), day(), s2, cfg=cfg).exit_price


# --- daily break, segment break, end of data --------------------------------------------
NO_TIME_EXIT = Scenario(name="no-time-exit", missed_exit=True)
BREAK_BAR = pd.Timestamp("2024-07-10 21:00", tz="UTC")       # 00:00 server: inside the break
AFTER_BREAK = pd.Timestamp("2024-07-10 22:00", tz="UTC")     # 01:00 server


@pytest.mark.parametrize("open_after, expected", [(40000.0, 39600 - 2), (39400.0, 39400 - 2)])
def test_sl_touched_in_the_daily_break_resolves_at_the_next_open(cfg, open_after, expected):
    bars = day({BREAK_BAR: (40000.0, 40005.0, 39500.0, 40000.0),
                AFTER_BREAK: (open_after, open_after + 5, open_after - 5, open_after)})
    r = run(long_band(), bars, NO_TIME_EXIT, cfg=cfg)
    assert r.exit_type == "SL" and r.exit_time == AFTER_BREAK
    assert r.exit_price == pytest.approx(expected)


def test_tp_touched_in_the_daily_break_fills_at_tp_after_it(cfg):
    bars = day({BREAK_BAR: (40000.0, 40400.0, 39995.0, 40000.0)})
    r = run(long_band(), bars, NO_TIME_EXIT, cfg=cfg)
    assert (r.exit_type, r.exit_price, r.exit_time) == ("TP", 40300.0, AFTER_BREAK)


def test_placement_in_the_daily_break_is_skipped(cfg):
    t = pd.Timestamp("2024-07-10 20:15", tz="UTC")            # P = 20:45 UTC = 23:45 server
    c = long_band(signal_time=t, time_exit_utc=pd.Timestamp("2024-07-11 19:45", tz="UTC"))
    r = run(c, day(), NO_TIME_EXIT, cfg=cfg)
    assert r.status == "skipped" and "daily break" in r.reason


def test_no_entry_fill_in_the_daily_break(cfg):
    t = pd.Timestamp("2024-07-10 20:00", tz="UTC")            # P = 20:30 UTC, before the break
    c = stop_long(signal_time=t, expires_utc=pd.Timestamp("2024-07-10 22:30", tz="UTC"),
                  time_exit_utc=pd.Timestamp("2024-07-11 19:45", tz="UTC"))
    bars = day({BREAK_BAR: (40000.0, 40200.0, 39995.0, 40000.0)})
    r = run(c, bars, NO_TIME_EXIT, cfg=cfg)
    assert r.status == "cancelled"


def test_segment_break_with_open_position_is_a_data_gap_exit(cfg):
    first = day({at(120): (40000.0, 40005.0, 39995.0, 40010.0)}, end=at(120))
    later = flat_bars(T + pd.Timedelta(days=9), T + pd.Timedelta(days=9, hours=1))
    r = run(long_band(), concat_bars(first, later), NO_TIME_EXIT, cfg=cfg)
    assert r.exit_type == "data_gap_exit" and r.exit_time == at(135)
    assert r.exit_price == pytest.approx(40010 - 2)
    r = run(short_band(), concat_bars(first, later), NO_TIME_EXIT, cfg=cfg)
    assert r.exit_price == pytest.approx(40010 + 3 + 2)


def test_time_exit_due_inside_a_segment_gap_is_a_data_gap_exit_priced_at_c(cfg):
    # same bars as end_of_data below plus a later segment: both close at the last pre-gap
    # close at c (EXIT-3X widens only real time exits)
    first = day(end=at(120))
    later = flat_bars(T + pd.Timedelta(days=9), T + pd.Timedelta(days=9, hours=1))
    r = run(long_band(), concat_bars(first, later), cfg=cfg)
    assert r.exit_type == "data_gap_exit" and r.exit_time == at(135)
    s3 = SCENARIOS["EXIT-3X"]
    gap = run(short_band(), concat_bars(first, later), s3, cfg=cfg)
    end = run(short_band(), first, s3, cfg=cfg)
    assert (gap.exit_type, end.exit_type) == ("data_gap_exit", "end_of_data")
    assert gap.exit_price == end.exit_price == pytest.approx(40000 + 6 + 4)


def test_end_of_data_closes_at_the_last_close(cfg):
    r = run(long_band(), day(end=at(120)), cfg=cfg)
    assert r.exit_type == "end_of_data" and r.exit_time == at(135)
    assert r.exit_price == pytest.approx(40000 - 2)


# --- documented readings (SPEC NOTEs) -----------------------------------------------------
def test_fill_bar_sl_after_an_intrabar_fill_uses_the_gap_clause_on_that_bars_open(cfg):
    # SPEC NOTE in _exit_price: bar opens 39850 (below SL 39900), rallies through the trigger:
    # stop fill 40100 + 2, then the SL on the same bar is priced at the bar open - slip
    bars = day({at(45): (39850.0, 40150.0, 39840.0, 40000.0)})
    r = run(stop_long(), bars, cfg=cfg)
    assert r.entry_mode == "stop" and not r.fill_at_open
    assert r.fill_price == pytest.approx(40102.0)
    assert (r.exit_type, r.exit_price, r.exit_time) == ("SL", pytest.approx(39848.0), at(60))


def test_time_exit_uses_the_last_close_before_the_clock_even_if_a_later_bar_exists(cfg):
    # SPEC NOTE in _exit: no bar opens at 19:45 but 20:00 exists -> close of the 19:30 bar
    bars = day({EXIT - M15: (40000.0, 40005.0, 39995.0, 40003.0),
                EXIT + M15: (40050.0, 40055.0, 40045.0, 40050.0)}, drop=[EXIT])
    r = run(long_band(), bars, cfg=cfg)
    assert r.exit_type == "time_exit" and r.exit_time == EXIT
    assert r.exit_price == pytest.approx(40003 - 2)


def test_deferred_sl_fills_even_if_price_recovered_after_the_break(cfg):
    # SPEC NOTE in _exit: the post-break open 40100 is above SL 39600; the SL still executes
    bars = day({BREAK_BAR: (40000.0, 40005.0, 39500.0, 40000.0),
                AFTER_BREAK: (40100.0, 40105.0, 40095.0, 40100.0)})
    r = run(long_band(), bars, NO_TIME_EXIT, cfg=cfg)
    assert (r.exit_type, r.exit_price, r.exit_time) == ("SL", pytest.approx(39598.0),
                                                        AFTER_BREAK)


# --- COMMON 8.1 / 8.4 spread floor and the 0.5 x floor slippage branch --------------------
# spread = 0 (or below the floor) makes eff_spread = c x floor_{i,h} of the bar's UTC hour;
# ATR 10 makes 0.02 x ATR = 0.2 < 0.5 x floor, so slip = s x 0.5 x floor.
def _floor_case(t, *, side="BUY", atr=10.0, key="US30"):
    sl, tp = (39600.0, 40300.0) if side == "BUY" else (40400.0, 39700.0)
    return Candidate(key=key, side=side, entry=40000.0, stop=sl, take_profit=tp, atr=atr,
                     reason="t", signal_time=t, order_type=MARKET_BAND,
                     time_exit_utc=t + pd.Timedelta(hours=1))


def _floor_bars(t, spread=0, overrides=None):
    return flat_bars(t - M15, t + pd.Timedelta(hours=2), spread=spread, overrides=overrides)


@pytest.mark.parametrize("p_hour, floor", [(18, 1.0), (22, 1.5)])     # US30 floors by UTC hour
@pytest.mark.parametrize("name, c, s", [("BASE", 1, 1), ("STRESS-d2", 2, 2)])
def test_market_fill_uses_the_utc_hour_floor_and_half_floor_slippage(cfg, p_hour, floor,
                                                                    name, c, s):
    p = pd.Timestamp(f"2024-07-10 {p_hour:02d}:00", tz="UTC")       # 22:00 UTC = 01:00 server
    t = p - 2 * M15
    r = run(_floor_case(t), _floor_bars(t), SCENARIOS[name], cfg=cfg)
    assert r.status == "filled" and r.fill_time == p
    assert r.fill_price == pytest.approx(40000 + c * floor + s * 0.5 * floor)
    r = run(_floor_case(t, side="SELL"), _floor_bars(t), SCENARIOS[name], cfg=cfg)
    assert r.fill_price == pytest.approx(40000 - s * 0.5 * floor)
    # time exit (1 h later, same floor hour for 18h; 23h floor 1.5 = 22h floor)
    assert r.exit_type == "time_exit"
    assert r.exit_price == pytest.approx(40000 + c * floor + s * 0.5 * floor)


def test_spread_above_the_floor_wins_the_max(cfg):
    t = pd.Timestamp("2024-07-10 17:30", tz="UTC")                   # P 18:00, floor 1.0
    r = run(_floor_case(t), _floor_bars(t, spread=250), cfg=cfg)     # 2.5 points > 1.0
    assert r.fill_price == pytest.approx(40000 + 2.5 + 0.5)          # slip still 0.5 x floor
    r = run(_floor_case(t), _floor_bars(t, spread=50), cfg=cfg)      # 0.5 points < 1.0
    assert r.fill_price == pytest.approx(40000 + 1.0 + 0.5)


def test_atr_branch_wins_when_larger_than_half_the_floor(cfg):
    t = pd.Timestamp("2024-07-10 17:30", tz="UTC")
    r = run(_floor_case(t, atr=100.0), _floor_bars(t), cfg=cfg)      # 0.02 x 100 = 2 > 0.5
    assert r.fill_price == pytest.approx(40000 + 1.0 + 2.0)


@pytest.mark.parametrize("name, c, s", [("BASE", 1, 1), ("STRESS-d2", 2, 2)])
def test_short_sl_and_tp_triggers_use_the_floored_spread(cfg, name, c, s):
    t = pd.Timestamp("2024-07-10 17:30", tz="UTC")                   # bars 18:xx: floor 1.0
    nxt = t + 3 * M15                                                # 18:15, after the fill
    sl_hi = 40400.0 - c * 1.0                                        # ask high = SL exactly
    r = run(_floor_case(t, side="SELL"),
            _floor_bars(t, overrides={nxt: (40000.0, sl_hi, 39995.0, 40000.0)}),
            SCENARIOS[name], cfg=cfg)
    assert (r.exit_type, r.exit_price) == ("SL", pytest.approx(40400 + s * 0.5))
    r = run(_floor_case(t, side="SELL"),
            _floor_bars(t, overrides={nxt: (40000.0, sl_hi - 0.01, 39995.0, 40000.0)}),
            SCENARIOS[name], cfg=cfg)
    assert r.exit_type == "time_exit"
    tp_lo = 39700.0 - 0.01 - c * 1.0                                 # ask low = TP - 0.01
    r = run(_floor_case(t, side="SELL"),
            _floor_bars(t, overrides={nxt: (40000.0, 40005.0, tp_lo, 40000.0)}),
            SCENARIOS[name], cfg=cfg)
    assert (r.exit_type, r.exit_price) == ("TP", 39700.0)
    r = run(_floor_case(t, side="SELL"),
            _floor_bars(t, overrides={nxt: (40000.0, 40005.0, tp_lo + 0.005, 40000.0)}),
            SCENARIOS[name], cfg=cfg)
    assert r.exit_type == "time_exit"


def test_ger40_hour_21_floor_amendment_a1(cfg):
    # GER40 21:00 UTC floor = 6.0 (A1). Winter (Wed 2024-01-10): 21:00 UTC = 23:00 server,
    # before the break, so the market order fills there. Prices are in points; 1.12 USD per
    # point (sizing) does not move them.
    spec = SymbolSpec("GER40", 2, 0.01, 0.01, 0.01, 0.1, 250.0, 0.1, "EUR",
                      swap_long=0.0, swap_short=0.0, swap_mode=1, swap_rollover3days=3,
                      trade_stops_level=100)
    p = pd.Timestamp("2024-01-10 21:00", tz="UTC")
    t = p - 2 * M15
    c = replace(_floor_case(t, key="GER40"), time_exit_utc=p + M15)
    res = run_backtest([("SES-IBX-01", c)], {"GER40": _floor_bars(t)}, cfg, BASE,
                       {"GER40": spec})
    r = res.trades.iloc[0]
    assert r.status == "filled" and r.fill_time == p
    assert r.fill_price == pytest.approx(40000 + 6.0 + 3.0)
    assert r.exit_type == "time_exit" and r.exit_time == p + M15
    assert r.exit_price == pytest.approx(40000 - 3.0)


def test_missing_segment_break_days_fails_instead_of_defaulting(cfg):
    # frozen constant (COMMON 1.4): no hard-coded fallback in prepare_bars
    from copy import deepcopy

    from backtest.market import prepare_bars
    bad = deepcopy(cfg)
    del bad["data_quality"]["segment_break_days"]
    with pytest.raises(KeyError):
        prepare_bars("US30", day(), bad)
