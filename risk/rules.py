"""Competition and personal risk rules, checked before every ticket.

Every loss check assumes the worst case: every open position, every pending order that has not
expired, every ticket issued since state.yaml was last updated, and the new ticket hit their
stops, plus a cost buffer for spread and commission on the new ticket.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time

import pandas as pd

from risk.state import AccountState

BLOCKED_OPEN_POSITION = "blocked_open_position"


@dataclass(frozen=True)
class RecentTicket:
    """An unsuppressed ticket the bot issued (from signals.csv). Tickets newer than
    state.updated_at may already be on the competition account but not in state.yaml yet,
    so they count as open positions, trades today and open risk (ER-04)."""
    key: str
    side: str
    lots: float
    risk_usd: float
    time_utc: pd.Timestamp


@dataclass
class RiskResult:
    ok: bool = True
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.ok = False
        self.reasons.append(reason)


def _in_sessions(t_utc: pd.Timestamp, sessions: list[list[str]] | None) -> bool:
    """start <= T < end on the UTC clock (ER-12); a window with start > end wraps midnight."""
    if not sessions:
        return True
    t = t_utc.tz_convert("UTC").time() if t_utc.tzinfo is not None else t_utc.time()
    for start, end in sessions:
        s, e = time.fromisoformat(start), time.fromisoformat(end)
        if (s <= t < e) if s <= e else (t >= s or t < e):
            return True
    return False


def _names_for(cfg: dict, key: str, symbol: str | None) -> set[str]:
    """Every name the owner might use for an index in state.yaml: key, broker symbol, aliases."""
    names = {key, symbol or key}
    for inst in cfg.get("instruments") or ():
        if inst["key"] == key:
            names.update(inst.get("aliases") or ())
    return {n.lower() for n in names}


def check_ticket(*, key: str, asset_class: str, lots: float, risk_usd: float,
                 state: AccountState, cfg: dict, now: pd.Timestamp,
                 signal_time: pd.Timestamp | None = None, symbol: str | None = None,
                 recent_tickets: tuple[RecentTicket, ...] | list[RecentTicket] = ()
                 ) -> RiskResult:
    """`now` is the wall clock (state age, competition end, restricted windows, pending-order
    expiry); `signal_time` is T, on which the session check runs (ER-12).
    SPEC NOTE: if signal_time is not passed, the session check falls back to `now` (callers
    that predate ER-12); the engine always passes T."""
    rules, risk = cfg["rules"], cfg["risk"]
    res = RiskResult()
    t_session = signal_time if signal_time is not None else now
    recent = [r for r in recent_tickets if r.time_utc > state.updated_at]
    pending = state.active_pending(now)

    if not cfg.get("owner_confirmed"):
        res.fail("owner settings not confirmed (set owner_confirmed: true in config.yaml)")

    age = now - state.updated_at
    if age > pd.Timedelta(hours=risk["state_max_age_hours"]):
        res.fail(f"state.yaml is {age} old; update it from your competition account")

    names = _names_for(cfg, key, symbol)
    for p in state.open_positions:
        if p.symbol.lower() in names:
            res.fail(f"{BLOCKED_OPEN_POSITION}: state.yaml lists an open {p.side} position on {key}")
    for o in pending:
        if o.symbol.lower() in names:
            until = f" until {o.expires_utc:%H:%M} UTC" if o.expires_utc is not None else ""
            res.fail(f"{BLOCKED_OPEN_POSITION}: state.yaml lists a pending {o.side} order "
                     f"on {key}{until}")

    # SPEC NOTE: ER-04 asks for pending orders' worst-case RISK to count until expiry; it does
    # not list them under max_open_positions or trades_today, so they count in risk only (a
    # pending order that fills becomes an open position in the owner's next state.yaml).
    n_open = len(state.open_positions) + len(recent)
    if n_open >= rules["max_open_positions"]:
        extra = f", incl. {len(recent)} ticket(s) issued since state.yaml was updated" if recent else ""
        res.fail(f"already {n_open} open positions{extra} (max {rules['max_open_positions']})")
    lot_cap = rules["max_lots"][asset_class]
    if lots > lot_cap:
        res.fail(f"{lots} lots exceeds the {asset_class} limit of {lot_cap}")
    trades = state.trades_today + len(recent)
    if trades >= rules["max_trades_per_day"]:
        res.fail(f"{trades} trades today (max {rules['max_trades_per_day']})")

    buffered = risk_usd * (1 + risk.get("cost_buffer_pct_of_risk", 0) / 100)
    other_risk = (state.open_risk_usd + sum(o.risk_usd for o in pending)
                  + sum(r.risk_usd for r in recent))
    worst_equity = min(state.balance, state.equity) - other_risk - buffered

    base = {"initial_balance": rules["initial_balance"],
            "start_of_day_balance": state.start_of_day_balance,
            "start_of_day_equity": state.start_of_day_equity}[rules["daily_loss_base"]]
    reference = {"start_of_day_balance": state.start_of_day_balance,
                 "start_of_day_equity": state.start_of_day_equity}[rules["daily_loss_reference"]]
    daily_floor = reference - base * rules["daily_loss_pct"] / 100
    if worst_equity <= daily_floor:
        res.fail(f"worst case equity ${worst_equity:,.0f} would breach the daily loss floor "
                 f"${daily_floor:,.0f}")

    personal_floor = state.start_of_day_balance * (1 - risk["personal_daily_stop_pct"] / 100)
    if worst_equity <= personal_floor:
        res.fail(f"worst case equity ${worst_equity:,.0f} would breach your personal daily "
                 f"stop ${personal_floor:,.0f}")

    total_floor = rules["initial_balance"] * (1 - rules["max_loss_pct"] / 100)
    if worst_equity <= total_floor:
        res.fail(f"worst case equity ${worst_equity:,.0f} would breach the max loss floor "
                 f"${total_floor:,.0f}")

    end = rules.get("competition_end_utc")
    if end:
        end = pd.Timestamp(end).tz_convert("UTC")
        cutoff = end - pd.Timedelta(minutes=rules["no_new_trades_before_end_minutes"])
        if now >= cutoff:
            res.fail(f"within {rules['no_new_trades_before_end_minutes']} min of competition end")
        res.notes.append(f"close all positions by "
                         f"{end - pd.Timedelta(minutes=rules['close_before_end_minutes']):%Y-%m-%d %H:%M} UTC")

    for window in rules.get("restricted_windows_utc") or ():
        if pd.Timestamp(window["start"]) <= now < pd.Timestamp(window["end"]):
            res.fail(f"inside restricted window: {window.get('reason', 'restricted')}")

    if not _in_sessions(t_session, (cfg.get("sessions_utc") or {}).get(key)):
        res.fail(f"signal time {t_session:%H:%M} UTC is outside allowed trading hours for {key}")

    if pending:
        res.notes.append(f"{len(pending)} pending order(s) counted at worst case "
                         f"${sum(o.risk_usd for o in pending):,.0f} until expiry")
    if recent:
        res.notes.append(f"{len(recent)} ticket(s) issued since state.yaml was updated counted "
                         f"as open (${sum(r.risk_usd for r in recent):,.0f} risk)")

    res.notes.append(f"trading days {state.trading_days_completed} of {rules['min_trading_days']}")
    res.notes.append(f"daily loss budget left ${max(0.0, worst_equity - daily_floor):,.0f} "
                     f"after this ticket")
    return res
