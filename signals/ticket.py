"""Trade ticket format and CSV log.

Protocol tickets (COMMON 7) print the owner's placement rule as absolute price levels: ORDER
TYPE, ENTRY or TRIGGER, CHASE LIMIT and/or SKIP LEVEL, EXPIRY, VALID UNTIL and the manual time
exit. The owner only compares the live quote with printed levels; nothing is re-anchored.
"""
from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from core.config import ROOT
from strategies.base import MARKET_BAND, MARKET_OR_LIMIT, STOP_ENTRY, Candidate

SIGNALS_CSV = ROOT / "signals.csv"


@dataclass
class Ticket:
    time_utc: pd.Timestamp       # T, the close of the signal bar
    key: str
    symbol: str
    side: str                    # BUY / SELL
    entry: float                 # E, or the trigger of a STOP_ENTRY ticket
    stop: float
    take_profit: float
    lots: float | None
    risk_usd: float              # at sizing_entry: the worst fill the protocol permits
    stop_points: float           # |sizing_entry - stop| in symbol points
    digits: int
    regime: str
    regime_prob: float
    rank_score: float
    strategy_id: str
    valid_until: pd.Timestamp    # last time to place the order
    price_tolerance: float       # legacy symmetric rule only (order_type None)
    reason: str
    stats: dict = field(default_factory=dict)   # oos_win_rate, expectancy_r, max_dd_pct, n_trades
    suppressed: bool = False
    suppress_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    order_type: str | None = None
    chase_limit: float | None = None
    skip_level: float | None = None
    expires_utc: pd.Timestamp | None = None
    time_exit_utc: pd.Timestamp | None = None
    time_exit_local: str = ""    # e.g. "15:45 New York"
    sizing_entry: float | None = None
    atr: float | None = None
    point: float | None = None
    stops_buffer: float | None = None   # trade_stops_level * point (STOP_ENTRY)
    trigger: float | None = None        # STOP_ENTRY trigger (== entry); None otherwise (ER-01)

    @property
    def rr(self) -> float:
        risk = abs(self.entry - self.stop)
        return abs(self.take_profit - self.entry) / risk if risk else 0.0

    @property
    def sgn(self) -> float:
        return 1.0 if self.side == "BUY" else -1.0

    def _p(self, price: float) -> str:
        return f"{price:.{self.digits}f}"

    def _toward(self, price: float | None, ref: float) -> float | None:
        """A protocol level rounded to the symbol's digits toward the entry, so the printed
        level never permits more than the spec fraction of D.
        SPEC NOTE: COMMON 7 gives no rounding for chase / skip levels; rounding toward E at the
        symbol's digits is the literal-safe choice (sizing uses the unrounded, wider level, so
        printed RISK $ stays a worst case)."""
        if price is None:
            return None
        scale = 10 ** self.digits
        x = price * scale
        x = math.floor(x + 1e-9) if price > ref else math.ceil(x - 1e-9)
        return x / scale

    def _dist(self, a: float, b: float) -> str:
        """Distance in price and in symbol points, e.g. '20.0 = 200 points'."""
        d = abs(a - b)
        pts = f" = {d / self.point:,.0f} points" if self.point else ""
        return f"{self._p(d)}{pts}"

    def _protocol_lines(self) -> list[str]:
        side, quote = self.side, ("ask" if self.side == "BUY" else "bid")
        worse, better = (">", "<") if side == "BUY" else ("<", ">")
        e, tp = self._p(self.entry), self._p(self.take_profit)
        chase_v = self._toward(self.chase_limit, self.entry)
        skip_v = self._toward(self.skip_level, self.entry)
        chase = self._p(chase_v) if chase_v is not None else ""
        skip = self._p(skip_v) if skip_v is not None else ""
        at_tp = ">=" if side == "BUY" else "<="
        expiry = f"{self.expires_utc:%H:%M} UTC" if self.expires_utc is not None else ""
        lines = [f"ORDER TYPE: {self.order_type}"]
        if self.order_type == MARKET_BAND:
            lines += [
                f"ENTRY {e} | CHASE LIMIT {chase} ({self._dist(chase_v, self.entry)} "
                f"worse than entry) | SKIP LEVEL {skip} "
                f"({self._dist(skip_v, self.entry)} better than entry)",
                "EXPIRY: none (market order only)",
                f"PLACE: market {side} now if {quote} is between {skip} and {chase} and not at "
                f"or beyond TP {tp}. Skip if {quote} {worse} {chase} (CHASE LIMIT), "
                f"{quote} {better} {skip} (SKIP LEVEL) or {quote} {at_tp} {tp}.",
            ]
        elif self.order_type == MARKET_OR_LIMIT:
            lines += [
                f"ENTRY {e} | SKIP LEVEL {skip} ({self._dist(skip_v, self.entry)} "
                f"better than entry)",
                f"EXPIRY {expiry} (resting {side} LIMIT order)",
                f"PLACE: skip if {quote} {better} {skip} (SKIP LEVEL) or {quote} {at_tp} {tp}. "
                f"If {quote} is at {e} or better: market {side} now. If {quote} is worse than "
                f"{e}: {side} LIMIT at {e} expiring {expiry}. Never chase.",
            ]
        elif self.order_type == STOP_ENTRY:
            buf = self.stops_buffer
            if buf is not None:
                edge = self._p(self.entry - self.sgn * buf)
                buf_txt = f" | STOPS BUFFER {self._p(buf)}"
                rel, sign = ("<", "-") if side == "BUY" else (">", "+")
                cond = f"{quote} {rel} {edge} (trigger {sign} stops buffer)"
            else:
                buf_txt = " | STOPS BUFFER unknown (check the symbol's stops level)"
                cond = (f"{quote} is further than the broker stops level "
                        f"{'below' if side == 'BUY' else 'above'} {e}")
            lines += [
                f"TRIGGER {e} | CHASE LIMIT {chase} ({self._dist(chase_v, self.entry)} "
                f"beyond trigger){buf_txt}",
                f"EXPIRY {expiry} (resting {side} STOP order)",
                f"PLACE: if {cond}: {side} STOP at {e} expiring {expiry}; "
                f"elif {quote} {'<=' if side == 'BUY' else '>='} {chase} (CHASE LIMIT): market "
                f"{side} now; else skip.",
            ]
        return lines

    def format(self) -> str:
        s = self.stats
        stats = (f"win {s['oos_win_rate']:.1%} | expectancy {s['expectancy_r']:+.2f}R | "
                 f"max DD {s['max_dd_pct']:.1f}% | n={s['n_trades']}") if s else "none"
        lots = f"{self.lots:.2f}" if self.lots is not None else "-"
        protocol = self.order_type is not None
        if protocol:
            valid = (f"VALID UNTIL {self.valid_until:%H:%M} UTC (place the order by then"
                     + (f"; the resting order expires {self.expires_utc:%H:%M} UTC)"
                        if self.expires_utc is not None else ")"))
        else:
            valid = (f"VALID UNTIL {self.valid_until:%H:%M} UTC "
                     f"or ±{self._p(self.price_tolerance)} from entry")
        lines = [
            f"{self.time_utc:%Y-%m-%d %H:%M} UTC | {self.symbol} | {self.side} | "
            f"{self._p(self.entry)} | SL {self._p(self.stop)} | TP {self._p(self.take_profit)} | "
            f"{lots} lots | risk ${self.risk_usd:,.2f} | R:R {self.rr:.2f}",
            f"REGIME {self.regime} (p={self.regime_prob:.2f}) | RANK {self.rank_score:+.2f} | "
            f"STRATEGY {self.strategy_id} | {valid}",
        ]
        if protocol:
            lines += self._protocol_lines()
        lines.append(f"STATS (out-of-sample): {stats}")
        sizing = self.sizing_entry if self.sizing_entry is not None else self.entry
        if protocol and sizing != self.entry:
            lines.append(f"STOP DISTANCE {self._p(abs(sizing - self.stop))} "
                         f"({self.stop_points:,.0f} points) from the worst permitted entry "
                         f"{self._p(sizing)}: RISK $ is the worst case - recheck lot size on "
                         f"the competition platform")
        else:
            lines.append(f"STOP DISTANCE {self._p(abs(sizing - self.stop))} "
                         f"({self.stop_points:,.0f} points) - recheck lot size on the "
                         f"competition platform")
        lines.append(f"REASON: {self.reason}")
        if not protocol:
            lines.append(f"Skip if: price has moved more than {self._p(self.price_tolerance)} "
                         f"beyond entry before you can place it.")
        if self.time_exit_utc is not None:
            lines.append(f"MANUAL TIME EXIT: close at {self.time_exit_utc:%H:%M} UTC "
                         f"({self.time_exit_local}) if still open")
        lines += [f"NOTE: {n}" for n in self.notes]
        if self.suppressed:
            lines.insert(0, "SUPPRESSED - DO NOT TRADE: " + "; ".join(self.suppress_reasons))
        return "\n".join(lines)

    def csv_row(self) -> dict:
        row = asdict(self)
        row.pop("stats")
        row.update({f"stat_{k}": v for k, v in self.stats.items()})
        row["rr"] = round(self.rr, 2)
        row["suppress_reasons"] = "; ".join(self.suppress_reasons)
        row["notes"] = "; ".join(self.notes)
        return row


CSV_FIELDS = ["time_utc", "key", "symbol", "side", "entry", "stop", "take_profit", "lots",
              "risk_usd", "rr", "stop_points", "regime", "regime_prob", "rank_score",
              "strategy_id", "valid_until", "price_tolerance", "reason",
              "stat_oos_win_rate", "stat_expectancy_r", "stat_max_dd_pct", "stat_n_trades",
              "suppressed", "suppress_reasons", "notes",
              # ER-01 protocol fields (appended so the old columns keep their positions)
              "order_type", "chase_limit", "skip_level", "expires_utc", "time_exit_utc",
              "time_exit_local", "sizing_entry", "atr", "stops_buffer", "trigger"]


def _header(path: Path) -> list[str] | None:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return next(csv.reader(f), None)
    except FileNotFoundError:
        return None


def signals_csv_path(path: Path = SIGNALS_CSV) -> Path:
    """The file new rows go to.

    An existing signals.csv with another header (written before ER-01) is never rewritten:
    rows go to signals_v2.csv (then _v3, ...) next to it, so old history stays intact and every
    file has one consistent header. Readers use read_signal_rows(), which reads all of them.
    """
    candidate, n = path, 1
    while True:
        header = _header(candidate)
        if header is None or header == CSV_FIELDS:
            return candidate
        n += 1
        candidate = path.with_name(f"{path.stem}_v{n}{path.suffix}")


def log_ticket(ticket: Ticket, path: Path = SIGNALS_CSV) -> None:
    path = signals_csv_path(path)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(ticket.csv_row())


def read_signal_rows(path: Path = SIGNALS_CSV) -> list[dict]:
    """Rows of signals.csv and its versioned successors, oldest file first."""
    def version(p: Path) -> int:
        tail = p.stem.rsplit("_v", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    newer = [p for p in path.parent.glob(f"{path.stem}_v*{path.suffix}") if version(p) > 1]
    files = [path] + sorted(newer, key=version)
    rows: list[dict] = []
    for p in files:
        if p.exists():
            with open(p, newline="", encoding="utf-8") as f:
                rows += list(csv.DictReader(f))
    return rows


def _ts(value: str | None) -> pd.Timestamp | None:
    if not value or value in ("None", "NaT", "nan"):
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _f(value: str | None) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    return float(value)


def candidate_from_row(row: dict) -> tuple[str, Candidate, bool] | None:
    """(strategy_id, candidate, suppressed) rebuilt from a signals.csv row, for occupancy
    replay and exit reminders. None for pre-ER-01 rows, which carry no protocol fields."""
    if not row.get("order_type") or row.get("order_type") == "None":
        return None
    c = Candidate(
        key=row["key"], side=row["side"], entry=float(row["entry"]), stop=float(row["stop"]),
        take_profit=float(row["take_profit"]), atr=_f(row.get("atr")) or 0.0,
        reason=row.get("reason", ""), regime=row.get("regime", "n/a"),
        regime_prob=_f(row.get("regime_prob")) or 0.0,
        rank_score=_f(row.get("rank_score")) or 0.0,
        signal_time=_ts(row["time_utc"]), order_type=row["order_type"],
        chase_limit=_f(row.get("chase_limit")), skip_level=_f(row.get("skip_level")),
        expires_utc=_ts(row.get("expires_utc")), time_exit_utc=_ts(row.get("time_exit_utc")),
        sizing_entry=_f(row.get("sizing_entry")))
    return row["strategy_id"], c, row.get("suppressed") == "True"
