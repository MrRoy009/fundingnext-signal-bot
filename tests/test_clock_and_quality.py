import numpy as np
import pandas as pd
import pytest

from core.timeutil import server_offset, server_to_utc
from data.clock import fit_server_clock
from data.costs import SpreadFloorMissing, compute_spread_floor, load_spread_floor, spread_floor
from data.download import defects_section, spec_row
from data.mt5_adapter import MT5Client
from data.quality import format_report, holiday_bar_mask, quality_report
from data.store import BarStore
from tests.fakes import FakeMT5


def synthetic_us_index(zone: str, shift: float, weeks: int = 104) -> pd.DataFrame:
    """M15 bars whose tick volume triples at 09:30 New York, stamped in a broker clock."""
    utc = pd.date_range("2024-01-01", periods=weeks * 7 * 96, freq="15min", tz="UTC")
    utc = utc[utc.day_of_week < 5]
    ny = utc.tz_convert("America/New_York")
    after_open = (ny.hour * 60 + ny.minute >= 570) & (ny.hour < 16)
    rng = np.random.default_rng(0)
    vol = np.where(after_open, 300, 100) * rng.uniform(0.9, 1.1, len(utc))
    offsets = np.array([server_offset(t, zone, shift).total_seconds() for t in utc])
    server_time = ((utc - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1) + offsets).astype("int64")
    return pd.DataFrame({"server_time": server_time, "tick_volume": vol})


def test_clock_fit_tells_athens_from_ny_close():
    df = synthetic_us_index("Europe/Athens", 0)
    fit = fit_server_clock(df["server_time"], df["tick_volume"], ("America/New_York", 7.0))
    assert fit.best == ("Europe/Athens", 0.0)
    assert fit.scores[("Europe/Athens", 0.0)] > fit.scores[("America/New_York", 7.0)]


def test_clock_fit_tells_ny_close_from_athens():
    df = synthetic_us_index("America/New_York", 7)
    fit = fit_server_clock(df["server_time"], df["tick_volume"], ("Europe/Athens", 0.0))
    assert fit.best == ("America/New_York", 7.0)


def test_athens_conversion_in_dst_mismatch_week():
    # 16 Mar 2026: US on summer time, EU not. Athens server 15:30 = 13:30 UTC (US open).
    srv = int(pd.Timestamp("2026-03-16 15:30").timestamp())
    assert server_to_utc([srv], "Europe/Athens", 0)[0] == pd.Timestamp("2026-03-16 13:30", tz="UTC")


def test_rebase_recomputes_utc(cfg, tmp_path):
    cfg["data_dir"] = "store"
    cfg["holdout"] = {"start": None}
    store = BarStore(cfg, root=tmp_path)
    srv = np.array([int(pd.Timestamp("2026-03-16 15:30").timestamp())])
    wrong = server_to_utc(srv, "America/New_York", 7)
    store.merge("US30", "M15", pd.DataFrame({"time_utc": wrong, "server_time": srv}))
    assert store.rebase_utc("US30", "M15", "Europe/Athens", 0)
    assert store.load("US30", "M15")["time_utc"][0] == pd.Timestamp("2026-03-16 13:30", tz="UTC")
    assert not store.rebase_utc("US30", "M15", "Europe/Athens", 0)


def test_spikes_vs_large_moves():
    n = 3000
    t = pd.date_range("2026-01-05", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(1)
    close = 40000 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
    close[2500] *= 1.03                 # bad tick: jumps and comes straight back
    close[2700:] *= 1.03                # real move that sticks
    volume = np.full(n, 1000)
    volume[2500] = 50                   # the bad tick prints on thin volume
    close[2900] *= 1.03                 # real whipsaw on heavy volume: not a bad tick
    volume[2900] = 5000
    df = pd.DataFrame({"time_utc": t, "open": close, "high": close * 1.0001,
                       "low": close * 0.9999, "close": close, "spread": 10,
                       "tick_volume": volume})
    r = quality_report(df, 15)
    assert r["spikes"] == 1
    assert r["large_moves"] == 5


def test_daily_break_found_on_server_clock():
    # Break at 23:45-01:00 server time every weekday, server = Athens time.
    srv = pd.date_range("2026-01-05", "2026-04-30", freq="15min")
    srv = srv[(srv.day_of_week < 5) & ~((srv.hour == 0) | ((srv.hour == 23) & (srv.minute == 45)))]
    srv_epoch = ((srv - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype("int64")
    df = pd.DataFrame({"time_utc": server_to_utc(srv_epoch, "Europe/Athens", 0),
                       "server_time": srv_epoch, "open": 1.0, "high": 1.0, "low": 1.0,
                       "close": 1.0, "spread": 1})
    r = quality_report(df, 15)
    assert r["daily_break_server"].strftime("%H:%M") == "23:30"
    assert r["intraday_gaps"] == 0


# --- configured defects in the quality report (ER-09) --------------------------------------
def m15_frame(start: str, periods: int, spread=10) -> pd.DataFrame:
    t = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    return pd.DataFrame({"time_utc": t, "open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "spread": spread})


def test_holiday_bars_counted_on_exchange_local_date(cfg):
    # 4 July 2024 in New York is 04:00 UTC 4 July to 04:00 UTC 5 July (EDT).
    df = m15_frame("2024-07-03 00:00", 3 * 96)
    mask = holiday_bar_mask(df["time_utc"], cfg, "US30")
    assert mask.sum() == 96
    assert df["time_utc"][mask].min() == pd.Timestamp("2024-07-04 04:00", tz="UTC")
    assert quality_report(df, 15, key="US30", cfg=cfg)["holiday_bars"] == 96
    # GER40 follows XETRA, which is open on 4 July.
    assert quality_report(df, 15, key="GER40", cfg=cfg)["holiday_bars"] == 0


def test_xetra_holiday_bars_for_ger40(cfg):
    # XETRA is closed 24-26 Dec 2024. 26 Dec in Berlin (CET) ends at 23:00 UTC, so of the
    # UTC days 25-27 Dec the holiday bars run from 25 Dec 00:00 to 26 Dec 22:45 UTC.
    df = m15_frame("2024-12-25 00:00", 3 * 96)
    assert quality_report(df, 15, key="GER40", cfg=cfg)["holiday_bars"] == 2 * 96 - 4


def test_segments_and_tv_bad_window_from_config(cfg):
    a = m15_frame("2024-10-03 20:00", 32)                   # straddles the tv_bad start
    b = m15_frame("2024-10-14 08:00", 4)                    # after an 11-day hole
    df = pd.concat([a, b]).reset_index(drop=True)
    r = quality_report(df, 15, key="US30", cfg=cfg)
    assert r["long_breaks"] == ["2024-10-04 to 2024-10-14"]
    assert r["segments"] == ["2024-10-03 20:00 to 2024-10-04 03:45 (32 bars)",
                             "2024-10-14 08:00 to 2024-10-14 08:45 (4 bars)"]
    assert r["tv_bad_bars"] == 16 + 4                       # from 2024-10-04 00:00 UTC
    assert r["tv_bad_windows"] == ["2024-10-04T00:00:00Z to 2025-01-08T00:00:00Z"]
    # The break threshold comes from config, not a constant.
    cfg["data_quality"]["segment_break_days"] = 14
    r = quality_report(df, 15, key="US30", cfg=cfg)
    assert r["long_breaks"] == [] and len(r["segments"]) == 1
    # Not a weekend although it spans one: it takes the NYSE sessions of 4 and 7-11 Oct.
    assert r["unexplained_gaps"] == ["2024-10-04 03:45 to 2024-10-14 08:00"]


def frames(*parts) -> pd.DataFrame:
    return pd.concat([m15_frame(start, n) for start, n in parts]).reset_index(drop=True)


def test_holiday_closure_is_explained_by_the_calendar(cfg):
    # 24 Dec 2025 is an NYSE early close (13:00 NY = 18:00 UTC), 25 Dec a full holiday.
    df = frames(("2025-12-24 15:00", 13), ("2025-12-25 23:00", 4))
    r = quality_report(df, 15, key="US30", cfg=cfg)
    assert r["holiday_closures"] == ["2025-12-24 18:00 to 2025-12-25 23:00"]
    assert r["unexplained_gaps"] == [] and r["weekend_closures"] == 0
    assert r["intraday_gaps"] == 0 and r["calendar"] == "NYSE"
    # Without a calendar nothing can explain it, and the configured checks are unknown.
    r = quality_report(df, 15)
    assert r["unexplained_gaps"] == ["2025-12-24 18:00 to 2025-12-25 23:00"]
    assert r["holiday_closures"] is None and r["calendar"] is None
    assert r["holiday_bars"] is None and r["tv_bad_bars"] is None
    # GER40 Christmas 2025 (XETRA closed 24-26 Dec), the longest ordinary closure.
    ger = frames(("2025-12-23 20:00", 4), ("2025-12-29 00:15", 4))
    assert quality_report(ger, 15, key="GER40", cfg=cfg)["holiday_closures"] == [
        "2025-12-23 20:45 to 2025-12-29 00:15"]


def test_plain_weekend_is_counted_not_listed(cfg):
    df = frames(("2026-01-09 21:00", 4), ("2026-01-11 23:00", 4))
    for r in (quality_report(df, 15, key="US30", cfg=cfg), quality_report(df, 15)):
        assert r["weekend_closures"] == 1 and r["unexplained_gaps"] == []


def test_missing_friday_next_to_weekend_is_unexplained(cfg):
    # The GER40 case from stored data: Friday 18 July 2025 entirely missing. It spans a
    # Saturday, which the earlier classifier took as a weekend.
    df = frames(("2025-07-17 22:00", 6), ("2025-07-20 21:00", 4))
    r = quality_report(df, 15, key="GER40", cfg=cfg)
    assert r["unexplained_gaps"] == ["2025-07-17 23:15 to 2025-07-20 21:00"]
    assert r["weekend_closures"] == 0 and r["holiday_closures"] == []
    us = frames(("2026-01-08 20:00", 8), ("2026-01-11 23:00", 4))   # US30: Fri 9 Jan missing
    assert quality_report(us, 15, key="US30", cfg=cfg)["unexplained_gaps"] == [
        "2026-01-08 21:45 to 2026-01-11 23:00"]


def test_missing_monday_morning_is_unexplained(cfg):
    # GER40 2024-10-14: the Sunday-night open bar, then nothing until Monday 16:15 UTC.
    df = frames(("2024-10-11 19:00", 8), ("2024-10-13 23:00", 1), ("2024-10-14 16:15", 4))
    r = quality_report(df, 15, key="GER40", cfg=cfg)
    assert r["weekend_closures"] == 1
    assert r["unexplained_gaps"] == ["2024-10-13 23:00 to 2024-10-14 16:15"]


def test_long_overnight_hole_outside_the_session_is_unexplained(cfg):
    # US30 Wed 20:00 to Thu 13:00 UTC: no NYSE session time lost, but 16.75h of a
    # non-holiday weekday is more than any broker tail.
    df = frames(("2026-01-14 19:00", 5), ("2026-01-15 13:00", 4))
    assert quality_report(df, 15, key="US30", cfg=cfg)["unexplained_gaps"] == [
        "2026-01-14 20:00 to 2026-01-15 13:00"]


def test_format_report_has_new_columns(cfg):
    df = m15_frame("2024-07-03 00:00", 3 * 96)
    out = format_report([("US30", "M15", quality_report(df, 15, key="US30", cfg=cfg))])
    header, _, row = out.splitlines()
    assert "Holiday bars" in header and "tv_bad bars" in header and "Segments" in header
    assert "Unexplained gaps >12h" in header and "Holiday closures >12h" in header
    assert header.count("|") == out.splitlines()[1].count("|") == row.count("|")
    assert "| 96 | 0 |" in row


def test_download_report_lists_defects_and_swap(cfg):
    lines = "\n".join(defects_section(cfg))
    assert "2025-07-16T07:30:00Z to 2025-09-09T11:30:00Z" in lines
    assert "2024-10-04T00:00:00Z to 2025-01-08T00:00:00Z" in lines
    assert "more than 7 days apart" in lines
    spec = MT5Client(cfg, FakeMT5()).spec("USTEC")
    assert spec_row("NAS100", "USTEC", spec, "USD").endswith("| -5.25 | 1.5 | 1 | 5 | 10 |")


# --- spread floor (COMMON 8.2) -------------------------------------------------------------
def test_compute_spread_floor_p90_of_positive_spreads_by_utc_hour():
    df = m15_frame("2022-12-31 23:00", 12)                   # hour 23 of 2022 is outside
    df.loc[:3, "spread"] = 10_000
    spreads = [0, 100, 200, 300]                             # hour 0: zero spread ignored
    df.loc[4:7, "spread"] = spreads
    df.loc[8:11, "spread"] = [10, 20, 30, 40]                # hour 1
    floor = compute_spread_floor(df)
    assert set(floor) == {0, 1}
    assert floor[0] == round(float(np.percentile([1.0, 2.0, 3.0], 90)), 2) == 2.8
    assert floor[1] == 0.37                                  # p90 of 0.1..0.4, linear
    late = m15_frame("2026-04-30 23:45", 2, spread=500)      # the 2026-05-01 bar is out
    late.loc[1, "spread"] = 9_999
    assert compute_spread_floor(late) == {23: 5.0}


def test_spread_floor_table_in_config_is_complete(cfg):
    table = load_spread_floor(cfg)
    assert set(table) == {"US30", "NAS100", "US500", "GER40"}
    for hours in table.values():
        assert sorted(hours) == list(range(24))
        assert all(isinstance(v, float) and v > 0 for v in hours.values())
    assert spread_floor(cfg, "GER40", 14) == table["GER40"][14]


def test_spread_floor_missing_fails_clearly(cfg):
    cfg["costs"]["spread_floor"] = None
    with pytest.raises(SpreadFloorMissing, match="costs.spread_floor is not set"):
        load_spread_floor(cfg)
    cfg["costs"]["spread_floor"] = {"US30": {h: 1.0 for h in range(23)}}
    with pytest.raises(SpreadFloorMissing, match=r"missing UTC hours \[23\]"):
        load_spread_floor(cfg, ["US30"])
    with pytest.raises(SpreadFloorMissing, match="no table for GER40"):
        spread_floor(cfg, "GER40", 9)


@pytest.mark.parametrize("bad", [None, float("nan"), -1.0, 0, "wide", float("inf")])
def test_spread_floor_rejects_unusable_values(cfg, bad):
    hours = {h: 1.0 for h in range(24)}
    hours[7] = bad
    cfg["costs"]["spread_floor"] = {"US30": hours}
    match = r"missing UTC hours \[7\]" if bad is None else "US30 hour 7"
    with pytest.raises(SpreadFloorMissing, match=match):
        load_spread_floor(cfg, ["US30"])
