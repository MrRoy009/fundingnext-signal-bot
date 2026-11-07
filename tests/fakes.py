"""A stand-in for the MetaTrader5 module so tests run on any OS."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

RATE_DTYPE = [("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"),
              ("close", "<f8"), ("tick_volume", "<u8"), ("spread", "<i4"),
              ("real_volume", "<u8")]


def make_rates(server_start: str, n: int, minutes: int = 15, price: float = 100.0) -> np.ndarray:
    times = pd.date_range(server_start, periods=n, freq=f"{minutes}min")
    out = np.zeros(n, dtype=RATE_DTYPE)
    out["time"] = (times - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    close = price + np.cumsum(np.sin(np.arange(n)))
    out["open"] = close - 0.5
    out["close"] = close
    out["high"] = close + 1
    out["low"] = close - 1.5
    out["tick_volume"] = 100
    out["spread"] = 20
    return out


class FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_CONTEST = 1
    ACCOUNT_TRADE_MODE_REAL = 2
    TIMEFRAME_M15 = 15
    TIMEFRAME_H1 = 16385

    def __init__(self, trade_mode: int = 0, login: int = 123, symbols=("US30.cash", "USTEC")):
        self.trade_mode = trade_mode
        self.account_login = login
        self.symbol_names = list(symbols)
        self.rates: dict = {}
        self.calls: list[str] = []

    def initialize(self, **kwargs):
        self.calls.append("initialize")
        return True

    def shutdown(self):
        self.calls.append("shutdown")

    def last_error(self):
        return (0, "ok")

    def account_info(self):
        return SimpleNamespace(login=self.account_login, trade_mode=self.trade_mode,
                               server="Demo-Server", company="Fake Broker", currency="USD")

    def symbols_get(self):
        return [SimpleNamespace(name=n) for n in self.symbol_names]

    def symbol_select(self, name, enable):
        return name in self.symbol_names

    def symbol_info(self, name):
        if name not in self.symbol_names:
            return None
        return SimpleNamespace(name=name, digits=1, point=0.1, trade_tick_size=0.1,
                               trade_tick_value=0.1, volume_min=0.01, volume_max=100,
                               volume_step=0.01, currency_profit="USD",
                               swap_long=-5.25, swap_short=1.5, swap_mode=1,
                               swap_rollover3days=5, trade_stops_level=10)

    def symbol_info_tick(self, name):
        return SimpleNamespace(time=0, bid=100.0, ask=100.2)

    def copy_rates_from_pos(self, symbol, tf, start, count):
        """Like MT5: position 0 is the newest stored bar (forming or just closed), and the
        result is oldest first."""
        self.calls.append(f"copy_rates_from_pos:{start}:{count}")
        rates = self.rates.get((symbol, tf))
        if rates is None:
            return None
        end = len(rates) - start
        return rates[max(0, end - count):max(0, end)]

    def copy_rates_range(self, symbol, tf, start, end):
        return self.rates.get((symbol, tf))
