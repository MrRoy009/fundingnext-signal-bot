"""Backtest outputs (COMMON 8.9, 10.2): the trades table, the MTM equity series, the daily
end-of-day equity series and the $ / R breakdowns every report shows.

trades: one row per candidate, status filled / skipped / suppressed / blocked / cancelled, with
the reason, placement, fill, exit, exit type, lots, planned risk $, commission, swap, $P&L and
realised R (= $P&L / planned risk $). Only filled rows carry P&L.
daily: equity at the end of each server day (00:00 Europe/Athens boundary, owner decision O8):
the last M15 mark closed by the boundary. The 10.2 daily return series is its first difference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.market import ServerClock
from backtest.scenarios import Scenario
from strategies.base import Candidate

TRADE_COLUMNS = [
    "entry_id", "strategy_id", "key", "side", "order_type", "signal_time", "status", "reason",
    "entry", "stop", "take_profit", "chase_limit", "skip_level", "sizing_entry", "expires_utc",
    "time_exit_utc", "atr", "rank_score", "regime", "sizing_balance", "lots",
    "planned_risk_usd", "placement_time", "entry_mode", "fill_time", "fill_price",
    "fill_at_open", "exit_time", "exit_price", "exit_type", "exit_note", "commission_usd",
    "swap_usd", "rollovers", "gross_pnl_usd", "pnl_usd", "r_multiple",
]


@dataclass(frozen=True)
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame       # time_utc (bar close), balance, equity (worst-intrabar MTM)
    daily: pd.DataFrame        # date (server), balance, equity at end of day
    scenario: Scenario
    contract: str
    mode: str

    @property
    def filled(self) -> pd.DataFrame:
        return self.trades[self.trades["status"] == "filled"]


def trade_row(entry_id: int, strategy_id: str, c: Candidate) -> dict:
    row = dict.fromkeys(TRADE_COLUMNS)
    row.update(entry_id=entry_id, strategy_id=strategy_id, key=c.key, side=c.side,
               order_type=c.order_type, signal_time=c.signal_time, status="", reason="",
               entry=c.entry, stop=c.stop, take_profit=c.take_profit,
               chase_limit=c.chase_limit, skip_level=c.skip_level,
               sizing_entry=c.sizing_entry, expires_utc=c.expires_utc,
               time_exit_utc=c.time_exit_utc, atr=c.atr, rank_score=c.rank_score,
               regime=c.regime)
    return row


def trades_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    for col in ("lots", "planned_risk_usd", "fill_price", "exit_price", "commission_usd",
                "swap_usd", "gross_pnl_usd", "pnl_usd", "r_multiple", "sizing_balance"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def daily_eod(grid_ns: np.ndarray, balance: np.ndarray, equity: np.ndarray,
              clock: ServerClock) -> pd.DataFrame:
    """Last mark of each server day. A mark at time m belongs to the day containing m - 1 ns,
    so the bar closing exactly at 00:00 server time ends the previous day."""
    if len(grid_ns) == 0:
        return pd.DataFrame({"date": pd.Series([], dtype="object"), "balance": [],
                             "equity": []})
    days = clock.dates(np.asarray(grid_ns, dtype="int64") - 1)
    df = pd.DataFrame({"day": days, "balance": balance, "equity": equity})
    out = df.groupby("day", sort=True).last().reset_index()
    out["date"] = [pd.Timestamp(d).date() for d in out["day"]]
    return out[["date", "balance", "equity"]]


def summarize(trades: pd.DataFrame, by: str | list[str] | None = None) -> pd.DataFrame:
    """$ and R of filled trades (COMMON 8.9), overall or grouped by e.g. 'key', 'side',
    'exit_type'. Columns: n, pnl_usd (sum), mean_pnl_usd, mean_r, planned_risk_usd (sum)."""
    f = trades[trades["status"] == "filled"]
    agg = dict(n=("pnl_usd", "size"), pnl_usd=("pnl_usd", "sum"),
               mean_pnl_usd=("pnl_usd", "mean"), mean_r=("r_multiple", "mean"),
               planned_risk_usd=("planned_risk_usd", "sum"))
    if by is None:
        return f.assign(all="all").groupby("all").agg(**agg)
    return f.groupby(by, sort=True).agg(**agg)


def report_tables(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The breakdowns COMMON 8.9 requires: overall, per index, per side, per exit type."""
    return {"all": summarize(trades), "index": summarize(trades, "key"),
            "side": summarize(trades, "side"), "exit_type": summarize(trades, "exit_type")}
