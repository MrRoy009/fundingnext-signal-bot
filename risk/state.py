"""Owner-maintained account state (state.yaml) for the competition account."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from core.config import ROOT


@dataclass(frozen=True)
class OpenPosition:
    symbol: str
    side: str
    lots: float
    risk_usd: float


@dataclass(frozen=True)
class PendingOrder:
    """A resting limit or stop order on the competition account (ER-04).

    Its worst-case risk counts in every check until it expires: it can fill at any moment
    without the owner looking. expires_utc None means "no expiry set" and counts until the
    owner removes it from state.yaml.
    """
    symbol: str
    side: str
    lots: float
    risk_usd: float
    expires_utc: pd.Timestamp | None = None

    def active(self, now: pd.Timestamp) -> bool:
        return self.expires_utc is None or now < self.expires_utc


@dataclass(frozen=True)
class AccountState:
    updated_at: pd.Timestamp
    balance: float
    equity: float
    start_of_day_balance: float
    start_of_day_equity: float
    trades_today: int
    trading_days_completed: int
    open_positions: tuple[OpenPosition, ...] = field(default_factory=tuple)
    pending_orders: tuple[PendingOrder, ...] = field(default_factory=tuple)

    @property
    def open_risk_usd(self) -> float:
        return sum(p.risk_usd for p in self.open_positions)

    def active_pending(self, now: pd.Timestamp) -> tuple[PendingOrder, ...]:
        return tuple(o for o in self.pending_orders if o.active(now))

    def pending_risk_usd(self, now: pd.Timestamp) -> float:
        return sum(o.risk_usd for o in self.active_pending(now))


def _utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def parse_state(raw: dict) -> AccountState:
    required = ["updated_at", "balance", "equity", "start_of_day_balance",
                "start_of_day_equity", "trades_today", "trading_days_completed"]
    missing = [k for k in required if raw.get(k) is None]
    if missing:
        raise ValueError(f"state.yaml is missing: {missing}")
    positions = tuple(
        OpenPosition(p["symbol"], p["side"], float(p["lots"]), abs(float(p["risk_usd"])))
        for p in raw.get("open_positions") or ())
    pending = tuple(
        PendingOrder(o["symbol"], o["side"], float(o["lots"]), abs(float(o["risk_usd"])),
                     _utc(o["expires_utc"]) if o.get("expires_utc") else None)
        for o in raw.get("pending_orders") or ())
    return AccountState(_utc(raw["updated_at"]), float(raw["balance"]), float(raw["equity"]),
                        float(raw["start_of_day_balance"]), float(raw["start_of_day_equity"]),
                        int(raw["trades_today"]), int(raw["trading_days_completed"]),
                        positions, pending)


def load_state(path: Path | str | None = None) -> AccountState:
    path = Path(path) if path else ROOT / "state.yaml"
    if not path.exists():
        raise FileNotFoundError("state.yaml not found. Copy state.example.yaml to state.yaml "
                                "and fill it in from your competition account.")
    with open(path, encoding="utf-8") as f:
        return parse_state(yaml.safe_load(f) or {})
