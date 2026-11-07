"""End-to-end cycle against the fake MT5 module."""
import pandas as pd

import signals.run as run
from core.config import Credentials
from core.timeutil import server_offset, utc_now
from data.mt5_adapter import MT5Client
from tests.fakes import FakeMT5, make_rates


def client_with_bars(cfg, last_bar_age_min: int):
    cfg["instruments"] = cfg["instruments"][1:2]          # NAS100 -> USTEC
    fake = FakeMT5()
    now = utc_now()
    last_open = now.floor("15min") - pd.Timedelta(minutes=15 + last_bar_age_min)
    server_last = (last_open + server_offset(now, "America/New_York", 7)).tz_localize(None)
    start = server_last - pd.Timedelta(minutes=15 * 99)
    fake.rates[("USTEC", fake.TIMEFRAME_M15)] = make_rates(str(start), 100, 15, 20000)
    fake.rates[("USTEC", fake.TIMEFRAME_H1)] = make_rates(str(start), 100, 60, 20000)
    client = MT5Client(cfg, fake)
    client.connect(Credentials(None, None, None, r"C:\MT5-Data\terminal64.exe", None))
    return client, client.resolve_symbols()


def test_no_strategy_means_no_signal(cfg, monkeypatch, capsys):
    client, symbols = client_with_bars(cfg, 0)
    run.cycle(client, cfg, symbols, format_check=False)
    assert "NO SIGNAL | no QA-approved strategy yet" in capsys.readouterr().out


def test_stale_data_means_no_signal(cfg, capsys):
    client, symbols = client_with_bars(cfg, 120)
    run.cycle(client, cfg, symbols, format_check=True)
    out = capsys.readouterr().out
    assert "NO SIGNAL" in out and "data stale" in out and "FORMAT TEST" not in out


def test_format_check_ticket_is_always_suppressed(cfg, monkeypatch, capsys):
    logged = []
    monkeypatch.setattr(run, "log_ticket", logged.append)
    monkeypatch.setattr(run, "notify", lambda t: logged.append("notified"))
    monkeypatch.setattr(run, "load_state",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("no state.yaml")))
    client, symbols = client_with_bars(cfg, 0)
    run.cycle(client, cfg, symbols, format_check=True)
    out = capsys.readouterr().out
    assert "SUPPRESSED - DO NOT TRADE: format test only, not a signal" in out
    assert "| USTEC | BUY |" in out
    assert len(logged) == 1 and logged[0].suppressed
