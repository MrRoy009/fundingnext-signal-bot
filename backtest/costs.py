"""Slippage, commission and swap for the backtester (COMMON 8.4, 8.6).

The spread itself lives with the bars (backtest.market: max(spread_pts, floor) per bar) and the
scenario multiplier c is applied by the fill model, because the multiplier also moves triggers.
Every constant here comes from config (`costs`, `rules`) or from the read-only symbol spec.
"""
from __future__ import annotations

import math
from datetime import date

from data.mt5_adapter import SymbolSpec

from backtest.scenarios import COMMISSION_STRESS, Scenario

# MT5 SYMBOL_SWAP_MODE_* values (symbol_info.swap_mode).
SWAP_DISABLED = 0
SWAP_POINTS = 1
SWAP_CURRENCY_SYMBOL = 2
SWAP_CURRENCY_MARGIN = 3
SWAP_CURRENCY_DEPOSIT = 4
SWAP_INTEREST_CURRENT = 5
SWAP_INTEREST_OPEN = 6
INTEREST_DAYS = 360          # MT5 interest swap convention (annual % / 360 per rollover)


def slippage(scenario: Scenario, cfg: dict, floor_h: float, atr: float) -> float:
    """slip = s * max(0.5 * floor_{i,h}, 0.02 * ATR14_M15 at the signal bar), in points.
    SPEC NOTE: h is the UTC hour of the bar on which the fill happens (COMMON 8.4 names the
    floor of 'h' without saying which bar; the fill bar is the only bar the fill sees)."""
    if not math.isfinite(atr) or atr < 0:
        raise ValueError(f"candidate atr {atr!r} must be a finite ATR14_M15 (slippage term)")
    c = cfg["costs"]
    return scenario.slip_mult * max(c["slippage_floor_frac"] * floor_h,
                                    c["slippage_atr_frac"] * atr)


def commission_per_lot_side(cfg: dict, key: str, scenario: Scenario) -> float:
    """USD per lot per side. Base = config rules.commission_per_lot_side (the owner-confirmed
    competition commission, 0 until confirmed). Stress = max(2 x base, config
    costs.commission_stress_per_lot_side[key]) (COMMON 8.6; GER40's EUR 1.00 x 1.12 is stored
    already converted)."""
    base = float(cfg["rules"].get("commission_per_lot_side") or 0.0)
    if scenario.commission != COMMISSION_STRESS:
        return base
    stress = cfg["costs"]["commission_stress_per_lot_side"].get(key)
    if stress is None:
        raise ValueError(f"config costs.commission_stress_per_lot_side has no entry for {key}")
    return max(2.0 * base, float(stress))


def mt5_weekday(d: date) -> int:
    """MT5 day numbering (0 = Sunday ... 6 = Saturday) of a calendar date."""
    return (d.weekday() + 1) % 7


def margin_ccy_usd(cfg: dict, key: str) -> float | None:
    """USD per unit of instrument `key`'s margin currency: config costs.margin_ccy_usd[key]
    (COMMON A4.1; US30 / NAS100 1.0, GER40 the frozen EUR->USD 1.12). None when the config has
    no entry; swap_value_per_lot then refuses swap mode 3 instead of guessing."""
    v = ((cfg.get("costs") or {}).get("margin_ccy_usd") or {}).get(key)
    return None if v is None else float(v)


def swap_value_per_lot(spec: SymbolSpec, side: str, usd_per_point: float,
                       price: float, margin_ccy_usd: float | None = None) -> float:
    """Signed swap per lot for ONE rollover in USD (negative = charge), per symbol_info
    swap_mode (ER-10). `price` is the reference price for interest modes. `margin_ccy_usd` is
    USD per unit of the symbol's margin currency (config costs.margin_ccy_usd[key], see
    margin_ccy_usd()); only mode 3 reads it, and it is passed in by the caller, never derived
    from the broker symbol name.

    SPEC NOTE: modes implemented: DISABLED (0), POINTS (1: swap x point price units x USD/pt),
    CURRENCY_MARGIN (3: money per lot in the margin currency x margin_ccy_usd, COMMON A4.1),
    CURRENCY_DEPOSIT (4: USD per lot), INTEREST_CURRENT / INTEREST_OPEN (5, 6: annual % of the
    price over 360 days, MT5's convention). Mode 2 (symbol base currency) needs an FX rate the
    backtest does not have and raises instead of guessing; so does mode 3 without a factor and
    a missing swap field when swap is needed.

    SPEC NOTE (open, for the Quant under O4): modes 3 and 4 are money PER LOT, so neither
    scales with usd_per_point. The demo symbols (US30, NAS100, GER40) are all mode 3, so a
    swap result at the competition contract carries the demo per-lot swap unchanged; A4.1 does
    not rule a scaling, and reports must say the swap is demo per-lot money.
    """
    raw = spec.swap_long if side == "BUY" else spec.swap_short
    mode = spec.swap_mode
    if mode is None or raw is None:
        raise ValueError(f"{spec.name}: swap_mode / swap_long / swap_short not read from the "
                         f"broker (ER-10); a position held through rollover needs them")
    if mode == SWAP_DISABLED:
        return 0.0
    if mode == SWAP_POINTS:
        return float(raw) * spec.point * usd_per_point
    if mode == SWAP_CURRENCY_MARGIN:
        if margin_ccy_usd is None:
            raise ValueError(f"{spec.name}: swap_mode 3 (margin currency) needs config "
                             f"costs.margin_ccy_usd for this instrument (COMMON A4.1)")
        f = float(margin_ccy_usd)
        if not math.isfinite(f) or f <= 0:
            raise ValueError(f"{spec.name}: costs.margin_ccy_usd {margin_ccy_usd!r} must be a "
                             f"positive finite USD factor (COMMON A4.1)")
        return float(raw) * f
    if mode == SWAP_CURRENCY_DEPOSIT:
        return float(raw)
    if mode in (SWAP_INTEREST_CURRENT, SWAP_INTEREST_OPEN):
        return float(raw) / 100.0 * price * usd_per_point / INTEREST_DAYS
    raise ValueError(f"{spec.name}: swap_mode {mode} is not supported by the backtest "
                     f"(needs a currency conversion); see backtest.costs")


def swap_charge(spec: SymbolSpec, side: str, lots: float, usd_per_point: float,
                price: float, rollover_day: date, scenario: Scenario,
                margin_ccy_usd: float | None = None) -> float:
    """Swap COST (>= 0, USD) for one rollover at the end of server day `rollover_day`.

    Positive swap is credited as 0; negative swap is multiplied by the scenario's swap_mult
    (stress doubles it) and tripled when rollover_day is symbol_info swap_rollover3days
    (COMMON 8.6). margin_ccy_usd: see swap_value_per_lot (swap mode 3 only).
    """
    per_lot = swap_value_per_lot(spec, side, usd_per_point, price, margin_ccy_usd)
    if per_lot >= 0:
        return 0.0
    triple = spec.swap_rollover3days is not None and \
        mt5_weekday(rollover_day) == int(spec.swap_rollover3days)
    return -per_lot * lots * scenario.swap_mult * (3.0 if triple else 1.0)
