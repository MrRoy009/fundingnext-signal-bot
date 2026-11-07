from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from core.config import Credentials
from core.timeutil import server_offset, server_to_utc, utc_to_server_naive
from data.mt5_adapter import MT5Client, MT5Error
from data.quality import quality_report
from data.store import BarStore, HoldoutLocked
from features.indicators import atr
from tests.fakes import FakeMT5, make_rates

NO_CREDS = Credentials(None, None, None, r"C:\MT5-Data\terminal64.exe", None)


def test_connect_requires_dedicated_terminal(cfg):
    fake = FakeMT5()
    with pytest.raises(MT5Error, match="MT5_PATH"):
        MT5Client(cfg, fake).connect(Credentials(None, None, None, None, None))
    assert "initialize" not in fake.calls


def epoch(s):
    return int(pd.Timestamp(s).timestamp())


# --- server time --------------------------------------------------------------------------
@pytest.mark.parametrize("server, utc", [
    ("2026-01-15 10:00", "2026-01-15 08:00"),   # NY winter: server = UTC+2
    ("2026-07-15 10:00", "2026-07-15 07:00"),   # NY summer: server = UTC+3
    ("2026-03-09 10:00", "2026-03-09 07:00"),   # Monday after US DST start (8 Mar 2026)
    ("2026-03-06 10:00", "2026-03-06 08:00"),   # Friday before it
    ("2026-01-15 00:00", "2026-01-14 22:00"),   # server midnight crosses the UTC date
])
def test_ny_close_server_time(server, utc):
    got = server_to_utc([epoch(server)], "America/New_York", 7)[0]
    assert got == pd.Timestamp(utc, tz="UTC")


def test_fixed_offset_and_round_trip():
    got = server_to_utc([epoch("2026-07-01 12:00")], "UTC", 2)[0]
    assert got == pd.Timestamp("2026-07-01 10:00", tz="UTC")
    utc = pd.Timestamp("2026-07-01 10:00", tz="UTC")
    assert server_offset(utc, "America/New_York", 7).total_seconds() == 3 * 3600
    assert utc_to_server_naive(utc, "America/New_York", 7) == datetime(2026, 7, 1, 13, tzinfo=timezone.utc)


# --- adapter ------------------------------------------------------------------------------
def test_connect_accepts_demo(cfg):
    assert MT5Client(cfg, FakeMT5()).connect(NO_CREDS)["login"] == 123


@pytest.mark.parametrize("mode", [1, 2])
def test_connect_refuses_contest_and_real(cfg, mode):
    fake = FakeMT5(trade_mode=mode)
    with pytest.raises(MT5Error, match="not a DEMO"):
        MT5Client(cfg, fake).connect(NO_CREDS)
    assert fake.calls[-1] == "shutdown"


def test_connect_refuses_competition_login(cfg):
    creds = Credentials(None, None, None, NO_CREDS.path, competition_login=123)
    with pytest.raises(MT5Error, match="competition account"):
        MT5Client(cfg, FakeMT5()).connect(creds)


def test_symbol_resolution(cfg):
    cfg["instruments"] = cfg["instruments"][:2]   # US30, NAS100
    client = MT5Client(cfg, FakeMT5(symbols=("us30.cash", "USTEC")))
    assert client.resolve_symbols() == {"US30": "us30.cash", "NAS100": "USTEC"}


def test_missing_symbol_fails_clearly(cfg):
    client = MT5Client(cfg, FakeMT5(symbols=("USTEC",)))
    with pytest.raises(MT5Error, match="US30"):
        client.resolve_symbols()


def test_closed_bars_are_converted_to_utc(cfg):
    fake = FakeMT5()
    fake.rates[("USTEC", fake.TIMEFRAME_M15)] = make_rates("2026-01-15 10:00", 3)
    client = MT5Client(cfg, fake)
    client.connect(NO_CREDS)
    df = client.closed_bars("USTEC", "M15", 3)
    assert df["time_utc"].iloc[0] == pd.Timestamp("2026-01-15 08:00", tz="UTC")
    assert df["server_time"].iloc[0] == epoch("2026-01-15 10:00")


def test_account_switch_stops_reads(cfg):
    fake = FakeMT5()
    client = MT5Client(cfg, fake)
    client.connect(NO_CREDS)
    fake.account_login = 99999999        # owner logs the terminal in to another account
    with pytest.raises(MT5Error, match="account changed"):
        client.closed_bars("USTEC", "M15", 3)


# --- store --------------------------------------------------------------------------------
def test_holdout_is_locked(cfg, tmp_path):
    cfg["data_dir"] = "store"
    cfg["holdout"] = {"start": "2026-01-02"}
    store = BarStore(cfg, root=tmp_path)
    times = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    store.merge("US30", "H1", pd.DataFrame({"time_utc": times, "close": np.arange(48.0)}))

    visible = store.load("US30", "H1")
    assert visible["time_utc"].max() < pd.Timestamp("2026-01-02", tz="UTC")
    with pytest.raises(HoldoutLocked):
        store.load("US30", "H1", final_holdout=True)
    assert len(store.load("US30", "H1", final_holdout=True, purpose="final test")) == 48
    assert "final test" in store.access_log.read_text()


def test_merge_dedupes(cfg, tmp_path):
    cfg["data_dir"] = "store"
    store = BarStore(cfg, root=tmp_path)
    t = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    store.merge("X", "H1", pd.DataFrame({"time_utc": t, "close": 1.0}))
    df = store.merge("X", "H1", pd.DataFrame({"time_utc": t[3:], "close": 2.0}))
    assert len(df) == 5 and df["close"].iloc[-1] == 2.0


# --- quality ------------------------------------------------------------------------------
def test_quality_report_finds_problems():
    t = pd.date_range("2026-01-05 08:00", periods=100, freq="15min", tz="UTC")
    df = pd.DataFrame({"time_utc": t, "open": 100.0, "high": 101.0, "low": 99.0,
                       "close": 100.0, "spread": 10})
    df = df.drop(index=[50, 51]).reset_index(drop=True)          # one 45-min gap
    df.loc[10, "high"] = 98.0                                      # bad OHLC
    df.loc[20, "spread"] = 0
    df = pd.concat([df, df.iloc[[5]]]).sort_values("time_utc").reset_index(drop=True)
    hole = df.iloc[-5:].assign(time_utc=df["time_utc"].iloc[-5:] + pd.Timedelta(days=55))
    df = pd.concat([df, hole]).reset_index(drop=True)            # a 55-day hole
    r = quality_report(df, 15)
    assert r["long_breaks"] == ["2026-01-06 to 2026-03-02"]
    assert r["duplicates"] == 1
    assert r["bad_ohlc"] == 1
    assert r["zero_spread_bars"] == 1
    assert r["intraday_gaps"] == 1


# --- look-ahead ---------------------------------------------------------------------------
def test_atr_has_no_lookahead():
    rates = pd.DataFrame(make_rates("2026-01-05", 300))
    base = atr(rates)
    perturbed = rates.copy()
    perturbed.loc[200:, ["high", "close"]] *= 1.5
    assert base.iloc[:200].equals(atr(perturbed).iloc[:200])


# --- closed bars by time (ER-05) and symbol spec (ER-10) -----------------------------------
def client_with_m15(cfg, server_start: str, n: int):
    fake = FakeMT5()
    fake.rates[("USTEC", fake.TIMEFRAME_M15)] = make_rates(server_start, n)
    client = MT5Client(cfg, fake)
    client.connect(NO_CREDS)
    return client, fake


def test_closed_bars_keeps_just_closed_position_zero(cfg):
    # Live race: no tick of the new bar yet, so position 0 is the bar that just closed.
    # Skipping position 0 by position would drop it; selection by time keeps it.
    client, fake = client_with_m15(cfg, "2026-01-15 10:00", 8)   # last opens 09:45 UTC
    now = pd.Timestamp("2026-01-15 10:00", tz="UTC")              # exactly its close
    df = client.closed_bars("USTEC", "M15", 5, now=now)
    assert df["time_utc"].iloc[-1] == pd.Timestamp("2026-01-15 09:45", tz="UTC")
    assert len(df) == 5
    assert "copy_rates_from_pos:0:7" in fake.calls


def test_closed_bars_drops_forming_bar(cfg):
    client, _ = client_with_m15(cfg, "2026-01-15 10:00", 8)
    now = pd.Timestamp("2026-01-15 09:59:59", tz="UTC")           # 09:45 bar still forming
    df = client.closed_bars("USTEC", "M15", 5, now=now)
    assert df["time_utc"].iloc[-1] == pd.Timestamp("2026-01-15 09:30", tz="UTC")
    assert len(df) == 5
    assert (df["time_utc"] + pd.Timedelta(minutes=15) <= now).all()


def test_closed_bars_h1_uses_hour_close(cfg):
    fake = FakeMT5()
    fake.rates[("USTEC", fake.TIMEFRAME_H1)] = make_rates("2026-01-15 10:00", 4, 60)
    client = MT5Client(cfg, fake)
    client.connect(NO_CREDS)
    # Last H1 bar opens 11:00 UTC and closes 12:00; at 11:45 it is still forming.
    df = client.closed_bars("USTEC", "H1", 10, now=pd.Timestamp("2026-01-15 11:45", tz="UTC"))
    assert df["time_utc"].iloc[-1] == pd.Timestamp("2026-01-15 10:00", tz="UTC")
    assert len(df) == 3


def test_closed_bars_full_count_when_local_clock_lags(cfg, caplog):
    # Local clock 1 min behind the broker: the 09:45 bar is forming at the broker AND the
    # 09:30 bar has not closed by the local clock. Still 5 closed bars, no silent short frame.
    client, fake = client_with_m15(cfg, "2026-01-15 09:00", 12)  # last opens 09:45 UTC
    now = pd.Timestamp("2026-01-15 09:44", tz="UTC")
    df = client.closed_bars("USTEC", "M15", 5, now=now)
    assert len(df) == 5
    assert df["time_utc"].iloc[-1] == pd.Timestamp("2026-01-15 09:15", tz="UTC")
    assert "local clock may lag" in caplog.text
    # Lagging by more than one bar: the fetch is widened once to exactly enough bars.
    df = client.closed_bars("USTEC", "M15", 5, now=pd.Timestamp("2026-01-15 09:14", tz="UTC"))
    assert len(df) == 5 and df["time_utc"].iloc[-1] == pd.Timestamp("2026-01-15 08:45", tz="UTC")
    assert "copy_rates_from_pos:0:9" in fake.calls


def test_closed_bars_short_history_is_not_padded(cfg):
    client, _ = client_with_m15(cfg, "2026-01-15 10:00", 4)
    df = client.closed_bars("USTEC", "M15", 5, now=pd.Timestamp("2026-01-15 08:50", tz="UTC"))
    assert len(df) == 3                                           # 4 stored, 1 forming


def test_closed_bars_rejects_naive_now_and_converts_aware(cfg):
    client, _ = client_with_m15(cfg, "2026-01-15 10:00", 8)
    with pytest.raises(ValueError, match="tz-aware"):
        client.closed_bars("USTEC", "M15", 5, now=pd.Timestamp("2026-01-15 10:00"))
    athens = pd.Timestamp("2026-01-15 12:00", tz="Europe/Athens")   # = 10:00 UTC
    df = client.closed_bars("USTEC", "M15", 5, now=athens)
    assert df["time_utc"].iloc[-1] == pd.Timestamp("2026-01-15 09:45", tz="UTC")


def test_closed_bars_empty_history(cfg):
    fake = FakeMT5()
    fake.rates[("USTEC", fake.TIMEFRAME_M15)] = make_rates("2026-01-15 10:00", 0)
    client = MT5Client(cfg, fake)
    client.connect(NO_CREDS)
    assert client.closed_bars("USTEC", "M15", 5).empty


def test_spec_reads_swap_and_stops_level(cfg):
    client, _ = client_with_m15(cfg, "2026-01-15 10:00", 1)
    s = client.spec("USTEC")
    assert (s.swap_long, s.swap_short, s.swap_mode, s.swap_rollover3days,
            s.trade_stops_level) == (-5.25, 1.5, 1, 5, 10)
    assert s.volume_step == 0.01 and s.currency_profit == "USD"
