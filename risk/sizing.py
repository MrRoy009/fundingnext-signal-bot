"""Position sizing from symbol specs."""
from __future__ import annotations

import math
from dataclasses import dataclass

from data.mt5_adapter import SymbolSpec


@dataclass(frozen=True)
class SizeResult:
    lots: float | None
    risk_usd: float          # actual risk at the rounded lot size
    stop_distance: float     # in price units
    stop_points: float       # in symbol points
    note: str = ""


def size_position(balance: float, risk_pct: float, entry: float, stop: float,
                  spec: SymbolSpec, lot_cap: float) -> SizeResult:
    distance = abs(entry - stop)
    points = distance / spec.point
    if distance <= 0:
        return SizeResult(None, 0.0, distance, points, "stop distance is zero")
    risk_amount = balance * risk_pct / 100
    loss_per_lot = distance / spec.trade_tick_size * spec.trade_tick_value
    raw = risk_amount / loss_per_lot
    cap = min(lot_cap, spec.volume_max)
    lots = min(raw, cap)
    steps = math.floor(lots / spec.volume_step + 1e-9)
    decimals = max(0, -int(math.floor(math.log10(spec.volume_step))))
    lots = round(steps * spec.volume_step, decimals)
    if lots < spec.volume_min:
        return SizeResult(None, 0.0, distance, points,
                          f"size {raw:.4f} lots is below the minimum {spec.volume_min}")
    note = f"capped at {cap} lots" if raw > cap else ""
    return SizeResult(lots, round(lots * loss_per_lot, 2), distance, points, note)
