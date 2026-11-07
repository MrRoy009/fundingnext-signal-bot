"""Reference occupancy (COMMON 7.1): edge cases and the no-look-ahead property."""
import pandas as pd
import pytest

from signals.occupancy import OccupancyTracker, occupancy_end
from strategies.base import MARKET_BAND, MARKET_OR_LIMIT, STOP_ENTRY, Candidate

# US summer: T 14:00 UTC, manual time exit 15:45 New York = 19:45 UTC
T = pd.Timestamp("2026-07-15 14:00", tz="UTC")
EXIT = pd.Timestamp("2026-07-15 19:45", tz="UTC")
M15 = pd.Timedelta(minutes=15)


def bars(ranges, start=T):
    """Bid bars opening at start, start + 15 min, ... with (high, low) per bar."""
    return pd.DataFrame({"time_utc": [start + i * M15 for i in range(len(ranges))],
                         "high": [h for h, _ in ranges], "low": [lo for _, lo in ranges]})


def cand(**over):
    base = dict(key="US30", side="BUY", entry=100.0, stop=90.0, take_profit=120.0, atr=5.0,
                reason="t", signal_time=T, order_type=MARKET_BAND, time_exit_utc=EXIT)
    base.update(over)
    return Candidate(**base)


QUIET = (105.0, 95.0)


def test_sl_touch_ends_at_close_of_touching_bar():
    end, cause = occupancy_end(cand(), bars([QUIET, QUIET, (101.0, 90.0)]))
    assert (end, cause) == (T + 3 * M15, "SL")


def test_tp_touch():
    end, cause = occupancy_end(cand(), bars([QUIET, (120.0, 99.0)]))
    assert (end, cause) == (T + 2 * M15, "TP")


def test_sl_and_tp_same_bar_is_sl_at_that_bar_close():
    end, cause = occupancy_end(cand(), bars([QUIET, (125.0, 85.0), QUIET]))
    assert (end, cause) == (T + 2 * M15, "SL")


def test_duplicate_index_labels_are_handled_by_position():
    # A backtester concatenating segments without ignore_index: labels 0, 1, 0, 1.
    df = pd.concat([bars([QUIET, QUIET]), bars([QUIET, (101.0, 90.0)], start=T + 2 * M15)])
    assert list(df.index) == [0, 1, 0, 1]
    assert occupancy_end(cand(), df) == (T + 4 * M15, "SL")
    lim = cand(order_type=MARKET_OR_LIMIT, entry=100.0, expires_utc=T + 2 * M15)
    # the limit (100) is touched (low 95), so no expiry end; nothing else hit -> time exit
    assert occupancy_end(lim, pd.concat([bars([QUIET]), bars([QUIET], start=T + M15)])) \
        == (EXIT, "time_exit")


def test_bar_closing_at_t_is_not_after_t():
    # the signal bar (open T - 15 min) touches SL; it is not "after T"
    b = pd.concat([bars([(101.0, 80.0)], start=T - M15), bars([QUIET])])
    end, cause = occupancy_end(cand(), b)
    assert (end, cause) == (EXIT, "time_exit")


def test_time_exit_first():
    n = int((EXIT - T) / M15)                      # bars before the exit: all quiet
    end, cause = occupancy_end(cand(), bars([QUIET] * n + [(101.0, 80.0)]))
    assert (end, cause) == (EXIT, "time_exit")


def test_touch_on_bar_closing_at_time_exit_wins_tie():
    n = int((EXIT - T) / M15)
    end, cause = occupancy_end(cand(), bars([QUIET] * (n - 1) + [(101.0, 80.0)]))
    assert (end, cause) == (EXIT, "SL")


def test_sell_side_mirrored():
    c = cand(side="SELL", entry=100.0, stop=110.0, take_profit=80.0)
    assert occupancy_end(c, bars([QUIET, (110.0, 99.0)])) == (T + 2 * M15, "SL")
    assert occupancy_end(c, bars([QUIET, (101.0, 80.0)])) == (T + 2 * M15, "TP")


def test_limit_expiry_untouched():
    exp = T + pd.Timedelta(minutes=45)
    c = cand(order_type=MARKET_OR_LIMIT, expires_utc=exp)
    # buy limit at 100 never reached by the bid low before expiry; reached right after
    b = bars([(106.0, 101.0)] * 3 + [(104.0, 99.0)])
    assert occupancy_end(c, b) == (exp, "expiry")


def test_limit_touched_runs_to_bracket_or_time_exit():
    exp = T + pd.Timedelta(minutes=45)
    c = cand(order_type=MARKET_OR_LIMIT, expires_utc=exp)
    assert occupancy_end(c, bars([(106.0, 100.0)] + [QUIET] * 4)) == (EXIT, "time_exit")
    assert occupancy_end(c, bars([(106.0, 100.0), (121.0, 101.0)])) == (T + 2 * M15, "TP")


def test_stop_entry_expiry_untouched_and_touched():
    exp = T + pd.Timedelta(hours=1)
    c = cand(order_type=STOP_ENTRY, expires_utc=exp)
    # buy stop: triggered when the bid high reaches 100
    assert occupancy_end(c, bars([(99.5, 95.0)] * 4)) == (exp, "expiry")
    assert occupancy_end(c, bars([(99.5, 95.0), (100.0, 96.0)])) == (EXIT, "time_exit")
    s = cand(order_type=STOP_ENTRY, side="SELL", entry=100.0, stop=110.0, take_profit=80.0,
             expires_utc=exp)
    assert occupancy_end(s, bars([(105.0, 100.5)] * 4)) == (exp, "expiry")
    assert occupancy_end(s, bars([(105.0, 100.0)])) == (EXIT, "time_exit")


def test_market_band_has_no_expiry_end():
    c = cand(expires_utc=T + pd.Timedelta(minutes=30))     # ignored: not a resting order
    assert occupancy_end(c, bars([QUIET] * 4)) == (EXIT, "time_exit")


def test_open_without_time_exit_or_touch():
    assert occupancy_end(cand(time_exit_utc=None), bars([QUIET])) == (None, "open")
    with pytest.raises(ValueError):
        occupancy_end(cand(signal_time=None), bars([QUIET]))


def test_tracker_occupied_by_strategy_and_suppressed():
    tr = OccupancyTracker()
    tr.add("SES-IBX-01", cand(), bars([QUIET, (101.0, 85.0)]), suppressed=True)
    end = T + 2 * M15
    assert tr.occupied("US30", "SES-IBX-01", T)                # from T, suppressed included
    assert tr.occupied("US30", None, end - pd.Timedelta(seconds=1))
    assert not tr.occupied("US30", "SES-IBX-01", end)          # free at the close
    assert not tr.occupied("US30", "RS-MOM-01", T)             # other strategy
    assert not tr.occupied("NAS100", None, T)
    assert not tr.occupied("US30", None, T - pd.Timedelta(seconds=1))


def test_tracker_update_bars_and_time_exit_list():
    tr = OccupancyTracker()
    tr.add("A", cand(), bars([QUIET]))
    assert tr.spans[0].cause == "time_exit" and tr.open_at_time_exit()
    tr.update_bars("US30", bars([QUIET, (101.0, 89.0)]))
    assert tr.spans[0].cause == "SL" and not tr.open_at_time_exit()


def test_no_look_ahead_in_occupied_answer():
    """occupied(t) from bars closed by t equals occupied(t) from the full series."""
    full = bars([QUIET] * 6 + [(101.0, 89.0)] + [QUIET] * 4)
    c = cand(order_type=MARKET_OR_LIMIT, expires_utc=T + pd.Timedelta(hours=1))
    whole = OccupancyTracker()
    whole.add("A", c, full)
    for k in range(len(full) + 1):
        t = T + k * M15
        seen = OccupancyTracker()
        seen.add("A", c, full[full["time_utc"] + M15 <= t])
        assert seen.occupied("US30", "A", t) == whole.occupied("US30", "A", t), t
