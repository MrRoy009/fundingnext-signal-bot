"""The only module that talks to MetaTrader 5. Read-only by construction.

The MetaTrader5 module is wrapped in ReadOnlyMT5, which exposes an allowlist of
data-reading functions and raises on anything else. tests/test_readonly_guard.py also
fails the build if any order-placing function name appears in the codebase.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from core.config import Credentials, tf_minutes
from core.timeutil import server_offset, server_to_utc, utc_now, utc_to_server_naive

log = logging.getLogger(__name__)

ALLOWED_FUNCTIONS = frozenset({
    "initialize", "login", "shutdown", "version", "last_error",
    "account_info", "terminal_info",
    "symbols_get", "symbol_info", "symbol_select", "symbol_info_tick",
    "copy_rates_from_pos", "copy_rates_range",
    "positions_get",
})

BAR_COLUMNS = ["time_utc", "server_time", "open", "high", "low", "close",
               "tick_volume", "spread", "real_volume"]


class MT5Error(RuntimeError):
    pass


class ReadOnlyMT5:
    """Proxy around the MetaTrader5 module: constants and allowlisted reads only."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name: str):
        if name.isupper():
            return getattr(self._module, name)
        if name not in ALLOWED_FUNCTIONS:
            raise PermissionError(f"MT5 function '{name}' is not allowed: this bot is read-only")
        return getattr(self._module, name)


@dataclass(frozen=True)
class SymbolSpec:
    name: str
    digits: int
    point: float
    trade_tick_size: float
    trade_tick_value: float
    volume_min: float
    volume_max: float
    volume_step: float
    currency_profit: str
    # Read-only swap and stops-level fields (COMMON 8.6, 7; ER-10). Swap is charged in the
    # backtest's missed-exit scenario; STOP_ENTRY tickets need the stops level.
    # SPEC NOTE: ER-10 gives no defaults. They default to None ("not read from the broker")
    # so specs built by hand, e.g. in tests, keep working; spec() always fills them.
    swap_long: float | None = None
    swap_short: float | None = None
    swap_mode: int | None = None
    swap_rollover3days: int | None = None
    trade_stops_level: int | None = None


def _load_mt5_module():
    try:
        import MetaTrader5
    except ImportError as exc:
        raise MT5Error("MetaTrader5 package not installed (it runs on Windows only)") from exc
    return MetaTrader5


class MT5Client:
    def __init__(self, cfg: dict, module=None):
        self.cfg = cfg
        self.mt5 = ReadOnlyMT5(module if module is not None else _load_mt5_module())
        tz = cfg["server_timezone"]
        self.zone, self.shift = tz["zone"], float(tz["shift_hours"])
        self.symbols: dict[str, str] = {}
        self.login: int | None = None

    # --- connection -------------------------------------------------------------------
    def connect(self, creds: Credentials) -> dict:
        # A dedicated terminal is required: without a path, initialize() attaches to the
        # last-used terminal, which may be the one logged in to the competition account.
        if not creds.path:
            raise MT5Error("MT5_PATH is not set in .env. Point it at a separate MT5 terminal "
                           "used only for the demo data account (see README).")
        kwargs = {"timeout": 60_000, "path": creds.path, "portable": creds.portable}
        if creds.login:
            kwargs.update(login=creds.login, password=creds.password or "",
                          server=creds.server or "")
        if not self.mt5.initialize(**kwargs):
            raise MT5Error(f"MT5 initialize failed: {self.mt5.last_error()}")
        info = self.mt5.account_info()
        if info is None:
            self.shutdown()
            raise MT5Error(f"No account logged in to the MT5 terminal: {self.mt5.last_error()}")
        if creds.competition_login and info.login == creds.competition_login:
            self.shutdown()
            raise MT5Error("Connected account is the competition account. Refusing to run.")
        if info.trade_mode != self.mt5.ACCOUNT_TRADE_MODE_DEMO:
            self.shutdown()
            raise MT5Error(f"Account {info.login} is not a DEMO account "
                           f"(trade_mode={info.trade_mode}). Refusing to run.")
        log.info("Connected to demo account %s on %s (%s)", info.login, info.server, info.company)
        self.login = info.login
        return {"login": info.login, "server": info.server, "company": info.company,
                "currency": info.currency}

    def verify_account(self) -> None:
        """Re-check before reading data: someone may have switched the terminal's account."""
        info = self.mt5.account_info()
        if info is None or info.login != self.login \
                or info.trade_mode != self.mt5.ACCOUNT_TRADE_MODE_DEMO:
            self.shutdown()
            raise MT5Error("The terminal's account changed since connecting. Stopped.")

    def shutdown(self) -> None:
        self.mt5.shutdown()

    # --- symbols ----------------------------------------------------------------------
    def resolve_symbols(self) -> dict[str, str]:
        """Map each configured instrument key to the broker's symbol name."""
        available = {s.name for s in (self.mt5.symbols_get() or ())}
        lower = {n.lower(): n for n in available}
        missing = []
        for inst in self.cfg["instruments"]:
            found = None
            for alias in inst["aliases"]:
                if alias in available:
                    found = alias
                    break
                if alias.lower() in lower:
                    found = lower[alias.lower()]
                    break
            if found is None:
                missing.append(inst["key"])
                continue
            if not self.mt5.symbol_select(found, True) or self.mt5.symbol_info(found) is None:
                missing.append(inst["key"])
                continue
            self.symbols[inst["key"]] = found
        if missing:
            raise MT5Error(f"Symbols not found on this broker: {missing}. Add the broker's names "
                           f"to 'aliases' in config.yaml.")
        log.info("Symbol mapping: %s", self.symbols)
        return dict(self.symbols)

    def spec(self, symbol: str) -> SymbolSpec:
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise MT5Error(f"symbol_info({symbol}) failed: {self.mt5.last_error()}")
        return SymbolSpec(info.name, info.digits, info.point, info.trade_tick_size,
                          info.trade_tick_value, info.volume_min, info.volume_max,
                          info.volume_step, info.currency_profit,
                          swap_long=info.swap_long, swap_short=info.swap_short,
                          swap_mode=info.swap_mode, swap_rollover3days=info.swap_rollover3days,
                          trade_stops_level=info.trade_stops_level)

    def tick(self, symbol: str):
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            raise MT5Error(f"symbol_info_tick({symbol}) failed: {self.mt5.last_error()}")
        return tick

    # --- server clock -----------------------------------------------------------------
    def check_server_clock(self, symbol: str) -> str:
        """Compare the configured server-time rule with the latest tick. Raises on mismatch
        while the market is trading; returns a note when it can't be checked."""
        now = utc_now()
        expected = server_offset(now, self.zone, self.shift)
        tick = self.tick(symbol)
        if not tick.time:
            msg = (f"Server clock not checked live: no {symbol} tick received yet. "
                   f"Assuming UTC{expected.total_seconds() / 3600:+.1f}h")
            log.warning(msg)
            return msg
        tick_server = pd.Timestamp(tick.time, unit="s")
        tick_age = now.tz_localize(None) + expected - tick_server
        if abs(tick_age) <= timedelta(minutes=2):
            msg = f"Server clock OK: UTC{expected.total_seconds() / 3600:+.1f}h"
            log.info(msg)
            return msg
        if timedelta(0) < tick_age:
            msg = (f"Server clock not verified: last {symbol} tick is {tick_age} old "
                   f"(market likely closed). Assuming UTC{expected.total_seconds() / 3600:+.1f}h")
            log.warning(msg)
            return msg
        raise MT5Error(f"Server clock mismatch: latest tick is {-tick_age} in the future under "
                       f"the configured rule. Fix server_timezone in config.yaml.")

    # --- bars -------------------------------------------------------------------------
    def _timeframe(self, tf: str):
        return getattr(self.mt5, f"TIMEFRAME_{tf}")

    def _to_frame(self, rates) -> pd.DataFrame:
        if rates is None or len(rates) == 0:
            return pd.DataFrame(columns=BAR_COLUMNS)
        df = pd.DataFrame(rates)
        df["server_time"] = df["time"].astype("int64")
        df["time_utc"] = server_to_utc(df["server_time"].to_numpy(), self.zone, self.shift)
        return df[BAR_COLUMNS].reset_index(drop=True)

    def closed_bars(self, symbol: str, tf: str, count: int,
                    now: pd.Timestamp | None = None) -> pd.DataFrame:
        """Most recent `count` CLOSED bars, selected by time (COMMON 9, ER-05).

        Position 0 is not always the forming bar: until the first tick of a new bar arrives,
        it is the bar that just closed. Skipping it by position would hand the cycle a stale
        bar the backtest never sees, so fetch from position 0 and keep bars whose close
        (open + tf) is at or before `now`.
        """
        if now is None:
            now = utc_now()
        elif pd.Timestamp(now).tzinfo is None:
            raise ValueError(f"closed_bars needs a tz-aware `now`, got naive {now}")
        now = pd.Timestamp(now).tz_convert("UTC")
        self.verify_account()
        step = pd.Timedelta(minutes=tf_minutes(tf))
        # SPEC NOTE: ER-05 says "fetch from position 0" without a count. count + 2 bars are
        # fetched: position 0 may be forming, and one more bar may close after `now` when the
        # local clock lags the broker. If more bars than that are still open at `now`, the
        # fetch is repeated once with exactly enough extra bars (open bars are always the
        # newest, so their number is known). The result is the last `count` closed bars.
        want = count + 2
        for _ in range(2):
            rates = self.mt5.copy_rates_from_pos(symbol, self._timeframe(tf), 0, want)
            if rates is None:
                raise MT5Error(f"copy_rates_from_pos({symbol}, {tf}) failed: "
                               f"{self.mt5.last_error()}")
            df = self._to_frame(rates)
            if df.empty:
                return df
            closed = df["time_utc"] + step <= now
            n_closed, full = int(closed.sum()), len(df) == want
            if n_closed >= count or not full:
                break
            want = count + (len(df) - n_closed)
        if len(df) - n_closed > 1:
            log.warning("%s %s: %d bars from MT5 close after now=%s; the local clock may lag "
                        "the broker", symbol, tf, len(df) - n_closed, now)
        if n_closed < count and full:
            log.warning("%s %s: only %d of %d closed bars returned although MT5 returned a "
                        "full fetch", symbol, tf, n_closed, count)
        return df[closed].tail(count).reset_index(drop=True)

    def bars_range(self, symbol: str, tf: str, start_utc: pd.Timestamp,
                   end_utc: pd.Timestamp, chunk_days: int = 30) -> pd.DataFrame:
        """Closed bars between two UTC instants, fetched in chunks."""
        frames, cursor = [], start_utc
        while cursor < end_utc:
            self.verify_account()
            stop = min(cursor + pd.Timedelta(days=chunk_days), end_utc)
            rates = self.mt5.copy_rates_range(
                symbol, self._timeframe(tf),
                utc_to_server_naive(cursor, self.zone, self.shift),
                utc_to_server_naive(stop, self.zone, self.shift))
            if rates is None:
                log.warning("copy_rates_range(%s, %s, %s..%s) returned nothing: %s",
                            symbol, tf, cursor, stop, self.mt5.last_error())
            elif len(rates):
                frames.append(self._to_frame(rates))
            cursor = stop
        if not frames:
            return pd.DataFrame(columns=BAR_COLUMNS)
        df = pd.concat(frames).drop_duplicates("time_utc").sort_values("time_utc")
        closed = df["time_utc"] + pd.Timedelta(minutes=tf_minutes(tf)) <= utc_now()
        return df[closed].reset_index(drop=True)
