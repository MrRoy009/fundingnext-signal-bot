"""COMMON 8 backtester (ER-13): candidates + bid bars + scenario -> trades and MTM equity.

Inputs
- candidates: (strategy_id, Candidate) pairs, each with signal_time T and a COMMON 7 protocol.
  They are prepared exactly as the live engine prepares them (signals.engine.sanitize_candidate:
  protocol levels from config, manual time exit from config).
- bars: per-index M15 bid bars (BarStore.load output: holdout and excluded windows dropped).
- specs: per-index read-only SymbolSpec (lot step / min / max, point, swap fields,
  trade_stops_level). Sizing uses the frozen demo tick value from config (COMMON 8.7).
- scenario: backtest.scenarios.Scenario (COMMON 10.3).
- contract: USD per point per lot by index (backtest.scenarios.contract_table).

How it runs
Events are processed in the order the account LEARNS of them (backtest.fills.seen_ns), across
indices. At one instant t: fills, then exits / skips / cancels, then new candidates. Events
decided by a bar that closed at t (intrabar fills, SL / TP, close-price exits, cancels at
EXPIRY) are seen at t, so a candidate at T = t sees them (e.g. a slot freed by an SL is free).
Events decided by the open of the bar opening at t (market fills at P, skips at P, time exits
at the clock-bar open, gap fills, exits deferred past the daily break) are seen just after t,
so a candidate at T = t never depends on a price that has not closed by T (COMMON 2). A
candidate at T is
  1. blocked if the scenario's own position on its index is open or pending (COMMON 8.8);
  2. otherwise sized on the simulated balance (risk.sizing.size_position on sizing_entry) and
     checked by risk.rules.check_ticket on the simulated AccountState at T; a failed check is
     logged as suppressed and not traded (COMMON 8.7);
  3. otherwise placed and filled per COMMON 8.3-8.5 (backtest.fills).
The account at T only sees events known by T, so no decision uses a bar not closed by T.
No randomness anywhere (COMMON 8.10): the same inputs give the same outputs.
"""
from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from backtest.account import Account, BookEntry, build_position
from backtest.costs import commission_per_lot_side, margin_ccy_usd
from backtest.fills import FILLED, Path, simulate
from backtest.market import IndexBars, ServerClock, prepare_bars
from backtest.results import BacktestResult, daily_eod, trade_row, trades_frame
from backtest.scenarios import Scenario, contract_table
from core.config import tf_minutes
from data.mt5_adapter import SymbolSpec
from risk.rules import BLOCKED_OPEN_POSITION, check_ticket
from risk.sizing import size_position
from signals.engine import sanitize_candidate, valid_until
from strategies.base import PENDING_TYPES, Candidate

SINGLE, COMBINED = "single", "combined"
# COMMON 8.8: combined-book same-T priority.
COMBINED_PRIORITY = ("SES-IBX-01", "RS-MOM-01", "MR-TWAP-01")

STATUS_FILLED, STATUS_SKIPPED, STATUS_CANCELLED = "filled", "skipped", "cancelled"
STATUS_SUPPRESSED, STATUS_BLOCKED = "suppressed", "blocked"

_FILL, _EXIT, _ARRIVE = 0, 1, 2          # event phases at the same seen instant


@dataclass(frozen=True)
class StrategyCandidate:
    strategy_id: str
    candidate: Candidate


def _pairs(candidates: Iterable) -> list[StrategyCandidate]:
    out = []
    for item in candidates:
        if isinstance(item, StrategyCandidate):
            out.append(item)
        else:
            sid, c = item
            out.append(StrategyCandidate(str(sid), c))
    return out


def sizing_spec(spec: SymbolSpec, usd_per_point_demo: float) -> SymbolSpec:
    """The demo spec with its tick value frozen to config costs.usd_per_point_demo.
    SPEC NOTE: COMMON 8.7 sizes on the 'demo spec' with 'USD per point per lot: US30 and NAS100
    1.00, GER40 1.12 (frozen demo tick value)'. Lot step / min / max come from the spec; the
    loss per lot per point is the frozen config value (tick size 1 point, tick value USD/pt),
    whatever the broker's live tick value is. Sizing never uses the competition contract; the
    competition run applies its USD/pt to P&L, planned risk and the risk checks."""
    return replace(spec, trade_tick_size=1.0, trade_tick_value=float(usd_per_point_demo))


def _check_cfg(cfg: dict) -> dict:
    """cfg for risk.rules.check_ticket inside the backtest.
    SPEC NOTE: owner_confirmed gates live tickets until the owner reviews config.yaml; it is not
    a risk rule, so the simulated account evaluates the rules as if confirmed. Every other rule
    (sessions on T, restricted windows, loss floors, open positions, lot cap) applies."""
    out = dict(cfg)
    out["owner_confirmed"] = True
    return out


def run_backtest(candidates: Iterable[StrategyCandidate | tuple[str, Candidate]],
                 bars: Mapping[str, pd.DataFrame], cfg: dict, scenario: Scenario,
                 specs: Mapping[str, SymbolSpec], *,
                 contract: Mapping[str, float] | None = None, contract_name: str = "demo",
                 mode: str = SINGLE) -> BacktestResult:
    """Run one scenario at one contract. See the module docstring.

    mode 'single': every candidate must come from one strategy; at most one open or pending
    position per index. mode 'combined': candidates of several strategies share the book, one
    open or pending position per index across all of them; same-T priority SES-IBX-01,
    RS-MOM-01, MR-TWAP-01 (COMMON 8.8).
    """
    if mode not in (SINGLE, COMBINED):
        raise ValueError(f"mode must be '{SINGLE}' or '{COMBINED}', not {mode!r}")
    items = _pairs(candidates)
    sids = sorted({it.strategy_id for it in items})
    if mode == SINGLE and len(sids) > 1:
        raise ValueError(f"single-strategy run got candidates of {sids}; use mode='combined'")
    if mode == COMBINED:
        unknown = [s for s in sids if s not in COMBINED_PRIORITY]
        if unknown:
            raise ValueError(f"combined-book priority (COMMON 8.8) has no rank for {unknown}")
    if contract is None:
        contract = contract_table(cfg, "demo")
    demo = contract_table(cfg, "demo")

    clock = ServerClock.from_cfg(cfg)
    tf_min = tf_minutes(cfg["timeframes"]["entry"])
    ib: dict[str, IndexBars] = {k: prepare_bars(k, df, cfg, clock) for k, df in bars.items()}
    inst_order = {i["key"]: n for n, i in enumerate(cfg["instruments"])}
    asset_class = {i["key"]: i.get("asset_class", "index") for i in cfg["instruments"]}
    tradeable = {i["key"]: i.get("tradeable", True) for i in cfg["instruments"]}
    check_cfg = _check_cfg(cfg)
    risk_pct = cfg["risk"]["risk_per_trade_pct"]

    # Prepare every candidate exactly like the live engine (levels and time exit from cfg).
    prepared = []
    for n, it in enumerate(items):
        c = it.candidate
        if c.signal_time is None:
            raise ValueError(f"{it.strategy_id} {c.key}: candidate has no signal_time T")
        if c.order_type is None:
            raise ValueError(f"{it.strategy_id} {c.key}: legacy candidate without order_type; "
                             f"the backtester models the COMMON 7 protocols only")
        if c.key not in ib:
            raise ValueError(f"no bars for {c.key}")
        c2, invalid = sanitize_candidate(c, cfg, c.signal_time)
        prio = COMBINED_PRIORITY.index(it.strategy_id) if mode == COMBINED else 0
        order = (int(c2.signal_time.value), prio, inst_order.get(c2.key, len(inst_order)), n)
        prepared.append((order, it.strategy_id, c2, invalid))
    prepared.sort(key=lambda r: r[0])

    acct = Account(cfg["rules"]["initial_balance"], clock)
    heap: list = []
    seq = 0
    for order, sid, c, invalid in prepared:
        heapq.heappush(heap, (order[0], _ARRIVE, seq, ("arrive", (sid, c, invalid))))
        seq += 1
    rows: list[dict] = []
    positions = {}

    while heap:
        t, _, _, (kind, payload) = heapq.heappop(heap)
        if kind == "fill":
            acct.on_fill(positions[payload])
            continue
        if kind == "exit":
            acct.on_exit(positions[payload])
            acct.active.pop(payload, None)
            continue
        if kind == "resolve":
            acct.active.pop(payload, None)
            continue

        sid, c, invalid = payload
        entry_id = len(rows)
        row = trade_row(entry_id, sid, c)
        rows.append(row)
        key = c.key

        # 1. COMMON 8.8 concurrency
        blocker = next((e for e in acct.active.values() if e.key == key
                        and (mode == COMBINED or e.strategy_id == sid)), None)
        if blocker is not None:
            what = "open" if blocker.is_open(t) else "pending"
            row.update(status=STATUS_BLOCKED,
                       reason=f"{BLOCKED_OPEN_POSITION}: {blocker.strategy_id} {blocker.side} "
                              f"{what} on {key}")
            continue

        # 2. ticket checks, sizing and risk rules (COMMON 7, 8.7; signals.engine.build_ticket)
        reasons = []
        if not tradeable.get(key, False):
            reasons.append(f"{key} is not tradeable (config instruments)")
        if invalid:
            reasons.append(invalid)
        if c.order_type in PENDING_TYPES and c.expires_utc is None:
            reasons.append(f"{c.order_type} ticket has no EXPIRY")
        if (c.side == "BUY") != (c.stop < c.entry) or \
                (c.side == "BUY") != (c.take_profit > c.entry):
            reasons.append("stop/target on the wrong side of entry")
        if key not in contract or key not in demo or key not in specs:
            if reasons:                    # e.g. US500: untradeable, has no contract entry
                row.update(status=STATUS_SUPPRESSED, reason="; ".join(reasons))
                continue
            raise ValueError(f"no contract USD/pt or SymbolSpec for {key}")
        spec = specs[key]
        usd = float(contract[key])
        balance = acct.balance_at(t)
        lot_cap = cfg["rules"]["max_lots"][asset_class.get(key, "index")]
        size = size_position(balance, risk_pct, c.sizing_entry, c.stop,
                             sizing_spec(spec, demo[key]), lot_cap)
        row["sizing_balance"] = balance
        if size.lots is None:
            reasons.append(size.note)
        else:
            lots = size.lots
            planned = lots * abs(c.sizing_entry - c.stop) * usd
            row.update(lots=lots, planned_risk_usd=planned)
            state = acct.state(t, updated_at=c.signal_time)
            res = check_ticket(key=key, asset_class=asset_class.get(key, "index"), lots=lots,
                               risk_usd=planned, state=state, cfg=check_cfg,
                               now=c.signal_time, signal_time=c.signal_time, symbol=key)
            reasons.extend(res.reasons)
        if reasons:
            row.update(status=STATUS_SUPPRESSED, reason="; ".join(reasons))
            continue

        # 3. placement, fill and exit (COMMON 8.3-8.5)
        b = ib[key]
        vu = valid_until(replace(c, expires_utc=None), cfg, tf_min)
        stops_buffer = (spec.trade_stops_level * spec.point
                        if spec.trade_stops_level is not None else None)
        path = simulate(c, b, scenario, cfg, valid_until_ns=int(vu.value),
                        stops_buffer=stops_buffer)
        row.update(_path_fields(path))
        if path.status != FILLED:
            row["status"] = path.status
            acct.active[entry_id] = BookEntry(entry_id, sid, key, c.side, lots, planned, t,
                                              None, path.resolve_seen_ns)
            heapq.heappush(heap, (path.resolve_seen_ns, _EXIT, seq, ("resolve", entry_id)))
            seq += 1
            continue
        row["status"] = STATUS_FILLED
        pos = build_position(entry_id, key, c.side, lots, path, b, scenario, usd, planned,
                             commission_per_lot_side(cfg, key, scenario), spec, clock,
                             margin_ccy_usd=margin_ccy_usd(cfg, key))
        positions[entry_id] = pos
        acct.add_position(pos)
        acct.active[entry_id] = BookEntry(entry_id, sid, key, c.side, lots, planned, t,
                                          path.fill_seen_ns, path.exit_seen_ns)
        heapq.heappush(heap, (path.fill_seen_ns, _FILL, seq, ("fill", entry_id)))
        heapq.heappush(heap, (path.exit_seen_ns, _EXIT, seq + 1, ("exit", entry_id)))
        seq += 2

    for entry_id, pos in positions.items():
        rows[entry_id].update(
            commission_usd=pos.commission_usd, swap_usd=pos.swap_usd,
            rollovers=len(pos.swaps), gross_pnl_usd=pos.gross_pnl_usd, pnl_usd=pos.pnl_usd,
            r_multiple=pos.pnl_usd / pos.planned_risk_usd if pos.planned_risk_usd else np.nan)

    grid = np.unique(np.concatenate([b.close_ns for b in ib.values()])) if ib else \
        np.array([], dtype="int64")
    bal, eq = acct.series(grid)
    equity = pd.DataFrame({"time_utc": pd.to_datetime(grid, utc=True),
                           "balance": bal, "equity": eq})
    daily = daily_eod(grid, bal, eq, clock)
    return BacktestResult(trades=trades_frame(rows), equity=equity, daily=daily,
                          scenario=scenario, contract=contract_name, mode=mode)


def _path_fields(p: Path) -> dict:
    ts = lambda v: pd.Timestamp(v, tz="UTC") if v is not None else pd.NaT  # noqa: E731
    return dict(reason=p.reason, exit_note=p.exit_reason, placement_time=ts(p.placement_ns),
                entry_mode=p.entry_mode, fill_time=ts(p.fill_ns), fill_price=p.fill_price,
                fill_at_open=p.fill_at_open, exit_time=ts(p.exit_ns), exit_price=p.exit_price,
                exit_type=p.exit_type or None)


def run_both_contracts(candidates: Sequence, bars: Mapping[str, pd.DataFrame], cfg: dict,
                       scenario: Scenario, specs: Mapping[str, SymbolSpec], *,
                       mode: str = SINGLE) -> dict[str, BacktestResult | None]:
    """COMMON 8.7: results at the demo contract and at the competition contract (None while
    config rules.competition_usd_per_point is null)."""
    out: dict[str, BacktestResult | None] = {}
    for name in ("demo", "competition"):
        table = contract_table(cfg, name)
        out[name] = None if table is None else run_backtest(
            candidates, bars, cfg, scenario, specs, contract=table, contract_name=name,
            mode=mode)
    return out
