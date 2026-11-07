"""Turn strategy candidates into sized, risk-checked tickets."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from data.mt5_adapter import SymbolSpec
from risk.rules import BLOCKED_OPEN_POSITION, RecentTicket, check_ticket
from risk.sizing import size_position
from risk.state import AccountState
from signals.ticket import Ticket
from strategies.base import (ORDER_TYPES, PENDING_TYPES, STOP_ENTRY, Candidate,
                             time_exit_label, time_exit_utc, with_protocol_levels)


def check_fresh(last_bar_open: pd.Timestamp, tf_min: int, now: pd.Timestamp,
                multiple: float = 2) -> tuple[bool, str]:
    """Stale if the last closed bar closed more than `multiple` timeframes ago."""
    age = now - (last_bar_open + pd.Timedelta(minutes=tf_min))
    limit = pd.Timedelta(minutes=tf_min * multiple)
    if age > limit:
        weekend = " (weekend: market closed)" if now.day_of_week >= 5 else ""
        return False, f"data stale: last bar closed {age} ago, limit {limit}{weekend}"
    return True, ""


def prepare_candidate(c: Candidate, cfg: dict, now: pd.Timestamp) -> Candidate:
    """Fill what a protocol ticket needs and the strategy left blank: chase / skip / sizing
    entry from cfg `protocols`, T, and the manual time exit from cfg `time_exits`.
    SPEC NOTE: a candidate without signal_time takes `now` as T (legacy / test callers);
    every strategy is expected to set it. Idempotent."""
    c = with_protocol_levels(c, cfg)
    if c.signal_time is None:
        c = replace(c, signal_time=now)
    if c.order_type is not None and c.time_exit_utc is None:
        c = replace(c, time_exit_utc=time_exit_utc(cfg, c.key, c.signal_time))
    return c


def sanitize_candidate(c: Candidate, cfg: dict, now: pd.Timestamp) -> tuple[Candidate, str]:
    """prepare_candidate(), but a strategy-set protocol level that differs from the cfg level
    does not abort the cycle: the candidate gets the computed levels and a suppression reason.
    It still occupies its index (COMMON 7.1: every generated candidate occupies) and is logged,
    but is never tradeable. Unknown order types or sides still raise ValueError: no level can
    be computed for them."""
    try:
        return prepare_candidate(c, cfg, now), ""
    except ValueError as exc:
        if c.order_type not in ORDER_TYPES or c.side not in ("BUY", "SELL"):
            raise
        fixed = replace(c, chase_limit=None, skip_level=None, sizing_entry=None)
        return prepare_candidate(fixed, cfg, now), f"invalid protocol levels: {exc}"


def valid_until(c: Candidate, cfg: dict, tf_min: int) -> pd.Timestamp:
    """VALID UNTIL = min(T + valid_for_bars x tf, EXPIRY) for placing the order (ER-02)."""
    v = c.signal_time + pd.Timedelta(minutes=tf_min * cfg["execution"]["valid_for_bars"])
    return min(v, c.expires_utc) if c.expires_utc is not None else v


def build_ticket(c: Candidate, *, strategy_id: str, stats: dict, symbol: str,
                 asset_class: str, spec: SymbolSpec, state: AccountState | None,
                 state_error: str, cfg: dict, now: pd.Timestamp, tf_min: int,
                 recent_tickets: tuple[RecentTicket, ...] | list[RecentTicket] = (),
                 occupied_by: str | None = None) -> Ticket:
    """`now` is wall-clock time; the ticket is stamped with the candidate's T.
    `occupied_by` describes another candidate that occupies this index (COMMON 7.1); the
    ticket is then blocked (ER-04)."""
    execution = cfg["execution"]
    c, invalid_levels = sanitize_candidate(c, cfg, now)
    balance = state.balance if state else cfg["rules"]["initial_balance"]
    lot_cap = cfg["rules"]["max_lots"][asset_class]
    size = size_position(balance, cfg["risk"]["risk_per_trade_pct"], c.sizing_entry, c.stop,
                         spec, lot_cap)
    stops_level = getattr(spec, "trade_stops_level", None)

    ticket = Ticket(
        time_utc=c.signal_time, key=c.key, symbol=symbol, side=c.side, entry=c.entry,
        stop=c.stop, take_profit=c.take_profit, lots=size.lots, risk_usd=size.risk_usd,
        stop_points=size.stop_points, digits=spec.digits, regime=c.regime,
        regime_prob=c.regime_prob, rank_score=c.rank_score, strategy_id=strategy_id,
        valid_until=valid_until(c, cfg, tf_min),
        price_tolerance=round(c.atr * execution["skip_if_atr_mult"], spec.digits),
        reason=c.reason, stats=dict(stats), order_type=c.order_type,
        chase_limit=c.chase_limit, skip_level=c.skip_level, expires_utc=c.expires_utc,
        time_exit_utc=c.time_exit_utc,
        time_exit_local=time_exit_label(cfg, c.key) if c.time_exit_utc is not None else "",
        sizing_entry=c.sizing_entry, atr=c.atr, point=spec.point, trigger=c.trigger,
        stops_buffer=(stops_level * spec.point
                      if c.order_type == STOP_ENTRY and stops_level is not None else None))

    def suppress(reason: str) -> None:
        ticket.suppressed = True
        ticket.suppress_reasons.append(reason)

    tradeable = {i["key"]: i.get("tradeable", True) for i in cfg["instruments"]}
    if not tradeable.get(c.key, False):
        suppress(f"{c.key} is not tradeable (config instruments)")
    if invalid_levels:
        suppress(invalid_levels)
    if c.order_type in PENDING_TYPES and c.expires_utc is None:
        suppress(f"{c.order_type} ticket has no EXPIRY")
    if (c.side == "BUY") != (c.stop < c.entry) or (c.side == "BUY") != (c.take_profit > c.entry):
        suppress("stop/target on the wrong side of entry")
    if now >= ticket.valid_until:
        suppress(f"no longer valid: VALID UNTIL {ticket.valid_until:%H:%M} UTC has passed")
    if occupied_by:
        suppress(f"{BLOCKED_OPEN_POSITION}: {occupied_by}")
    if not stats:
        suppress("strategy has no validated statistics")
    if size.lots is None:
        suppress(size.note)
    elif size.note:
        ticket.notes.append(size.note)
    if state is None:
        suppress(state_error)
    elif size.lots is not None:
        result = check_ticket(key=c.key, asset_class=asset_class, lots=size.lots,
                              risk_usd=size.risk_usd, state=state, cfg=cfg, now=now,
                              signal_time=c.signal_time, symbol=symbol,
                              recent_tickets=recent_tickets)
        if not result.ok:
            ticket.suppressed = True
            ticket.suppress_reasons.extend(result.reasons)
        ticket.notes.extend(result.notes)
    return ticket
