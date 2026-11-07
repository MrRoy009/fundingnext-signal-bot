"""Signal loop. Wakes just after each entry-bar close and prints tickets.

    python -m signals.run                 # run continuously
    python -m signals.run --once          # one cycle, then exit
    python -m signals.run --format-check  # one cycle with a clearly marked TEST ticket per
                                          # index built from live demo data (never a signal)

Each cycle runs at T = the close of the latest M15 bar. Freshness is per index: an index whose
M15 bar closing at T has not arrived by T + 60 s is absent for that cycle, and so is every index
of its exchange group (COMMON 4.2.2). NO SIGNAL for everything is printed only when the
connection fails or every index is absent.
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from core.config import ROOT, load_config, load_credentials, tf_minutes
from core.logging_setup import setup_logging
from core.timeutil import utc_now
from data.mt5_adapter import MT5Client, MT5Error
from features.indicators import atr
from risk.rules import RecentTicket
from risk.state import AccountState, load_state
from signals.engine import build_ticket, check_fresh, sanitize_candidate
from signals.occupancy import M15, OccupancyTracker, Span
from signals.ticket import SIGNALS_CSV, Ticket, candidate_from_row, log_ticket, read_signal_rows
from strategies import registry
from strategies.base import MARKET_BAND, PENDING_TYPES, Candidate, StrategyContext, time_exit_label

log = logging.getLogger(__name__)
BARS_NEEDED = 500
POLL_GRACE = pd.Timedelta(seconds=60)    # wait for the bar at T until T + 60 s (COMMON 9)
POLL_INTERVAL_S = 2.0
FORMAT_TEST_ID = "FORMAT-TEST"
# A reminder is issued once, in the first cycle at or after the time exit; the window only
# covers a cycle that ran late or a restart, never a reminder hours after the fact.
REMINDER_WINDOW = pd.Timedelta(hours=1)
# Rows of signals.csv looked at for occupancy, reminders and tickets not yet in state.yaml.
# Occupancy never outlives the time exit on T's local date, so one day is enough.
LOOKBACK = pd.Timedelta(hours=24)
REMINDERS_CSV = ROOT / "logs" / "exit_reminders.csv"
REGIME_CSV = ROOT / "logs" / "regime_history.csv"


# --- output ---------------------------------------------------------------------------------
def notify_message(title: str, message: str) -> None:
    try:
        from plyer import notification
        notification.notify(title=title, message=message, timeout=30)
    except Exception as exc:  # notifications are best effort
        log.warning("Desktop notification failed: %s", exc)


def notify(ticket: Ticket) -> None:
    notify_message(f"{ticket.side} {ticket.symbol} ({ticket.strategy_id})",
                   f"{ticket.order_type or 'MARKET'} entry {ticket.entry} SL {ticket.stop} "
                   f"TP {ticket.take_profit} Lots {ticket.lots}")


def no_signal(reason: str) -> None:
    msg = f"{utc_now():%Y-%m-%d %H:%M} UTC | NO SIGNAL | {reason}"
    print(msg)
    log.info(msg)


def _append_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- bars -----------------------------------------------------------------------------------
def _last_open(df: pd.DataFrame) -> pd.Timestamp | None:
    return None if df.empty else df["time_utc"].iloc[-1]


def poll_bars(client: MT5Client, symbols: dict[str, str], tf: str, t: pd.Timestamp,
              deadline: pd.Timestamp) -> tuple[dict[str, pd.DataFrame], set[str]]:
    """Closed bars per index, polling until `deadline` for the bar that closes at t.

    Returns (bars, present): bars for every index that returned any, and the set of indices
    whose bar closing at t arrived. Bars are selected by time (close <= t, the decision time)
    so neither a forming bar nor a bar that closed while polling can slip in, whatever the
    adapter returns (COMMON 2, 9).
    """
    tf_td = pd.Timedelta(minutes=tf_minutes(tf))
    bars: dict[str, pd.DataFrame] = {}
    present: set[str] = set()
    while True:
        for key, symbol in symbols.items():
            if key in present:
                continue
            df = client.closed_bars(symbol, tf, BARS_NEEDED)
            df = df[df["time_utc"] + tf_td <= t].reset_index(drop=True)
            bars[key] = df
            if _last_open(df) == t - tf_td:
                present.add(key)
        if len(present) == len(symbols) or utc_now() >= deadline:
            return bars, present
        time.sleep(POLL_INTERVAL_S)


def absent_cause(df: pd.DataFrame | None, t: pd.Timestamp, tf_min: int, now: pd.Timestamp,
                 multiple: float) -> str:
    if df is None or df.empty:
        return "no bars"
    ok, msg = check_fresh(df["time_utc"].iloc[-1], tf_min, now, multiple)
    if not ok:
        return msg
    last_close = df["time_utc"].iloc[-1] + pd.Timedelta(minutes=tf_min)
    return (f"bar closing at {t:%H:%M} UTC missing after {POLL_GRACE.seconds} s "
            f"(last bar closed {last_close:%H:%M} UTC)")


def exchange_groups(cfg: dict, keys) -> dict[str, list[str]]:
    """Indices grouped by exchange calendar, in config order.
    SPEC NOTE: COMMON 4.2.2 applies to the strategy window set G(T). In every current spec
    G(T) is {US30, NAS100} or {GER40} (COMMON 4.2.1), i.e. a subset of one exchange group,
    so an absent index removes its whole exchange group from the cycle. Strategies still apply
    4.2.2 within G(T) themselves."""
    groups: dict[str, list[str]] = {}
    for key in keys:
        groups.setdefault(cfg["exchanges"][key]["calendar"], []).append(key)
    return groups


# --- signals.csv replay ---------------------------------------------------------------------
def recent_rows(now: pd.Timestamp, path: Path | None = None) -> list[dict]:
    """Strategy rows (format tests excluded) whose T is within LOOKBACK of now."""
    out = []
    for row in read_signal_rows(path or SIGNALS_CSV):
        if row.get("strategy_id") == FORMAT_TEST_ID:
            continue
        try:
            t = pd.Timestamp(row["time_utc"])
        except (KeyError, ValueError):
            continue
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        if now - LOOKBACK <= t <= now:
            out.append(row)
    return out


def recent_tickets(rows: list[dict]) -> list[RecentTicket]:
    """Unsuppressed tickets with a size; check_ticket keeps those newer than state.updated_at."""
    out = []
    for row in rows:
        if row.get("suppressed") == "True" or row.get("lots") in ("", "None", None):
            continue
        t = pd.Timestamp(row["time_utc"])
        out.append(RecentTicket(row["key"], row["side"], float(row["lots"]),
                                float(row["risk_usd"]),
                                t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")))
    return out


def _bars_for(bars: dict[str, pd.DataFrame], key: str) -> pd.DataFrame:
    df = bars.get(key)
    return df if df is not None else pd.DataFrame(columns=["time_utc", "high", "low"])


def occupancy_from_rows(rows: list[dict], bars: dict[str, pd.DataFrame]) -> OccupancyTracker:
    """Rebuild reference occupancy from logged candidates, suppressed ones included (7.1)."""
    tracker = OccupancyTracker()
    for row in rows:
        parsed = candidate_from_row(row)
        if parsed is None:
            continue
        strategy_id, c, suppressed = parsed
        tracker.add(strategy_id, c, _bars_for(bars, c.key), suppressed)
    return tracker


def due_exit_reminders(rows: list[dict], bars: dict[str, pd.DataFrame], now: pd.Timestamp,
                       done: set[str]) -> list[tuple[str, Span]]:
    """(symbol, span) for every reference-open candidate whose time exit is due (ER-03).

    SPEC NOTE: COMMON 7 says "every reference-open candidate", so suppressed candidates are
    included too, marked as such (the owner may have traded one anyway)."""
    out = []
    for row in rows:
        parsed = candidate_from_row(row)
        if parsed is None:
            continue
        strategy_id, c, suppressed = parsed
        if c.time_exit_utc is None or not c.time_exit_utc <= now < c.time_exit_utc + REMINDER_WINDOW:
            continue
        if reminder_id(strategy_id, c) in done:
            continue
        span = OccupancyTracker().add(strategy_id, c, _bars_for(bars, c.key), suppressed)
        if span.cause == "time_exit":
            out.append((row.get("symbol", c.key), span))
    return out


def reminder_id(strategy_id: str, c: Candidate) -> str:
    return f"{strategy_id}|{c.key}|{c.side}|{c.signal_time.isoformat()}"


REMINDER_FIELDS = ["reminded_utc", "id", "time_exit_utc", "key", "symbol", "side", "entry",
                   "stop", "take_profit", "strategy_id", "signal_time", "suppressed"]


def issue_exit_reminders(rows: list[dict], bars: dict[str, pd.DataFrame], cfg: dict,
                         now: pd.Timestamp, path: Path | None = None) -> list[str]:
    path = path or REMINDERS_CSV
    done = {r["id"] for r in _read_csv(path)}
    messages, logged = [], []
    for symbol, span in due_exit_reminders(rows, bars, now, done):
        c = span.candidate
        flag = " (ticket was SUPPRESSED)" if span.suppressed else ""
        msg = (f"{now:%Y-%m-%d %H:%M} UTC | EXIT REMINDER | close {symbol} {c.side} now if still "
               f"open{flag} | entry {c.entry} SL {c.stop} TP {c.take_profit} | {span.strategy_id} "
               f"signal {c.signal_time:%H:%M} UTC | time exit {c.time_exit_utc:%H:%M} UTC "
               f"({time_exit_label(cfg, c.key)})")
        print(msg)
        log.info(msg)
        notify_message(f"EXIT REMINDER {symbol} {c.side}",
                       f"Close if still open: entry {c.entry} SL {c.stop} TP {c.take_profit} "
                       f"({span.strategy_id})")
        messages.append(msg)
        logged.append({"reminded_utc": now, "id": reminder_id(span.strategy_id, c),
                       "time_exit_utc": c.time_exit_utc, "key": c.key, "symbol": symbol,
                       "side": c.side, "entry": c.entry, "stop": c.stop,
                       "take_profit": c.take_profit, "strategy_id": span.strategy_id,
                       "signal_time": c.signal_time, "suppressed": span.suppressed})
    _append_csv(path, REMINDER_FIELDS, logged)
    return messages


# --- context and regime history --------------------------------------------------------------
def build_context(cfg: dict, t: pd.Timestamp) -> StrategyContext:
    """Per-cycle context. features / regime stay None until the FeatureService and
    RegimeService (ER-06, ER-07) are wired in; strategies must then return no candidate."""
    return StrategyContext(features=None, regime=None, now=t, cfg=cfg)


REGIME_FIELDS = ["time_utc", "key", "v_t", "label", "status", "gate"]


def _field(obj, name: str):
    if isinstance(obj, Mapping):
        return obj.get(name, "n/a")
    return getattr(obj, name, "n/a")


def record_regime(context: StrategyContext, t: pd.Timestamp, path: Path | None = None) -> None:
    """Regime history for the daily summary (ER-17).
    Placeholder interface: context.regime is read as {key: snapshot}, a snapshot being a dict
    or object with v_t, label, status and gate; missing fields print as n/a."""
    if not isinstance(context.regime, Mapping):
        return
    rows = [{"time_utc": t, "key": key, **{f: _field(snap, f) for f in REGIME_FIELDS[2:]}}
            for key, snap in context.regime.items()]
    _append_csv(path or REGIME_CSV, REGIME_FIELDS, rows)


# --- cycle ----------------------------------------------------------------------------------
def format_test_candidate(key: str, df: pd.DataFrame, t: pd.Timestamp) -> Candidate:
    """Engine plumbing test only: MARKET_BAND at the last close, 1 ATR stop, 2 ATR target."""
    last, a = float(df["close"].iloc[-1]), float(atr(df).iloc[-1])
    return Candidate(key=key, side="BUY", entry=last, stop=last - a, take_profit=last + 2 * a,
                     atr=a, reason="FORMAT TEST from live demo data - not a trading signal",
                     signal_time=t, order_type=MARKET_BAND)


def _load_state() -> tuple[AccountState | None, str]:
    try:
        return load_state(), ""
    except (FileNotFoundError, ValueError) as exc:
        return None, str(exc)


def cycle(client: MT5Client, cfg: dict, symbols: dict[str, str],
          format_check: bool) -> dict[str, pd.DataFrame]:
    """One decision cycle at T. Returns the closed M15 bars it saw (close <= T), which
    the daily summary uses to report each candidate's reference occupancy."""
    now = utc_now()
    tf = cfg["timeframes"]["entry"]
    regime_tf = cfg["timeframes"]["regime"]
    tf_min = tf_minutes(tf)
    t = now.floor(f"{tf_min}min")                      # decision time T
    deadline = t + POLL_GRACE
    tradeable = {i["key"] for i in cfg["instruments"] if i.get("tradeable", True)}
    live = {k: s for k, s in symbols.items() if k in tradeable}   # US500: never ticketed

    bars, present = poll_bars(client, live, tf, t, deadline)
    rows = recent_rows(now)
    issue_exit_reminders(rows, bars, cfg, now)

    absent = {k: absent_cause(bars.get(k), t, tf_min, utc_now(), cfg["staleness_multiple"])
              for k in live if k not in present}
    if not present:
        no_signal("; ".join(f"{k}: {msg}" for k, msg in absent.items()) or "no tradeable index")
        return bars
    usable: list[str] = []
    for group in exchange_groups(cfg, live).values():
        missing = [k for k in group if k in absent]
        if missing:
            no_signal(f"{', '.join(group)}: incomplete group: "
                      + "; ".join(f"{k}: {absent[k]}" for k in missing))
        else:
            usable += group

    regime_bars: dict[str, pd.DataFrame] = {}
    if usable:
        # At full hours the H1 bar closing at T is polled too; if it is still missing the
        # regime model reports "regime H1 bar missing" for that index (COMMON 6.6).
        need_h1 = t.minute == 0
        regime_bars, h1_present = poll_bars(client, {k: live[k] for k in usable}, regime_tf,
                                            t if need_h1 else t.floor("60min"), deadline)
        for k in usable:
            if need_h1 and k not in h1_present:
                log.warning("%s: H1 bar closing at %s UTC not arrived by T + 60 s", k,
                            f"{t:%H:%M}")
    entry_bars = {k: bars[k] for k in usable}
    context = build_context(cfg, t)
    record_regime(context, t)

    jobs: list[tuple[str, dict, Candidate]] = []
    for strat in registry.APPROVED:
        jobs += [(strat.id, strat.stats, c)
                 for c in strat.generate(entry_bars, regime_bars, t, context)]
    if format_check:
        jobs += [(FORMAT_TEST_ID, {}, format_test_candidate(k, df, t))
                 for k, df in entry_bars.items()]
    kept = []
    for job in jobs:
        key = job[2].key
        if key not in entry_bars:
            log.warning("Dropped %s candidate on %s: index not usable this cycle (%s)", job[0],
                        key, "not tradeable" if key not in tradeable else "absent group")
        else:
            kept.append(job)
    if not kept:
        if usable:
            no_signal("no QA-approved strategy yet" if not registry.APPROVED
                      else "no setup on this bar")
        return bars

    state, state_error = _load_state()
    tracker = occupancy_from_rows(rows, bars)
    issued = recent_tickets(rows)
    classes = {i["key"]: i["asset_class"] for i in cfg["instruments"]}
    for strategy_id, stats, c in kept:
        try:
            c, invalid_levels = sanitize_candidate(c, cfg, t)
        except ValueError as exc:
            # Unknown order type or side: no level can be computed. One bad candidate must
            # not cost the other strategies their tickets.
            log.error("Rejected %s candidate on %s: %s", strategy_id, c.key, exc)
            no_signal(f"{c.key}: {strategy_id} candidate rejected: {exc}")
            continue
        test = strategy_id == FORMAT_TEST_ID
        occupied_by = None
        if not test:
            # A re-run of the same cycle finds this very candidate in signals.csv already;
            # it must not block itself.
            own = reminder_id(strategy_id, c)
            spans = [s for s in tracker.occupying(c.key, c.signal_time)
                     if reminder_id(s.strategy_id, s.candidate) != own]
            if spans:
                s = spans[0]
                occupied_by = (f"{c.key} occupied by {s.strategy_id} {s.candidate.side} from "
                               f"{s.start:%H:%M} UTC (reference occupancy)")
        ticket = build_ticket(c, strategy_id=strategy_id, stats=stats, symbol=symbols[c.key],
                              asset_class=classes[c.key], spec=client.spec(symbols[c.key]),
                              state=state, state_error=state_error, cfg=cfg, now=utc_now(),
                              tf_min=tf_min, recent_tickets=issued, occupied_by=occupied_by)
        if invalid_levels and invalid_levels not in ticket.suppress_reasons:
            ticket.suppressed = True
            ticket.suppress_reasons.append(invalid_levels)
        if test:
            ticket.suppressed = True
            ticket.suppress_reasons.insert(0, "format test only, not a signal")
        else:
            tracker.add(strategy_id, c, bars[c.key], ticket.suppressed)
            if not ticket.suppressed:
                issued.append(RecentTicket(c.key, c.side, ticket.lots, ticket.risk_usd, ticket.time_utc))
        print("\n" + ticket.format() + "\n")
        log_ticket(ticket)
        if not ticket.suppressed:
            notify(ticket)
    return bars


# --- daily summary --------------------------------------------------------------------------
def _occupancy_text(c: Candidate, sid: str, suppressed: bool,
                    bars: Mapping[str, pd.DataFrame] | None) -> str:
    """How the candidate's reference occupancy (COMMON 7.1) stands with the bars seen."""
    df = None if bars is None else bars.get(c.key)
    if df is None or df.empty:
        return "reference occupancy: unknown (no bars seen)"
    span = OccupancyTracker().add(sid, c, df, suppressed)
    seen_to = df["time_utc"].max() + M15
    if span.cause in ("SL", "TP"):
        return f"reference occupancy ended {span.end:%H:%M} UTC by {span.cause}"
    if span.cause == "expiry" and span.end <= seen_to:
        return f"reference occupancy ended {span.end:%H:%M} UTC at EXPIRY (entry not touched)"
    if span.cause == "time_exit" and span.end <= seen_to:
        return "reference-open at its time exit"
    return f"still reference-open with bars to {seen_to:%H:%M} UTC"


def write_daily_summary(day: pd.Timestamp, out_dir: Path | None = None,
                        bars: Mapping[str, pd.DataFrame] | None = None) -> Path:
    """Summarise one UTC day: tickets, every protocol candidate with its manual time exit and
    reference occupancy, EXIT REMINDERs, pending orders and regime history (ER-17).

    Suppressed candidates are listed too (marked): they occupy their index and get EXIT
    REMINDERs like any other (COMMON 7.1), so the summary and the reminder log agree.
    `bars` are the closed M15 bars the last cycle saw; without them the occupancy end is
    reported as unknown rather than guessed."""
    d = f"{day:%Y-%m-%d}"
    issued = suppressed = 0
    reasons: dict[str, int] = {}
    candidates, pending = [], []
    for row in read_signal_rows(SIGNALS_CSV):
        if not row["time_utc"].startswith(d) or row.get("strategy_id") == FORMAT_TEST_ID:
            continue
        sup = row["suppressed"] == "True"
        if sup:
            suppressed += 1
            for r in filter(None, row["suppress_reasons"].split("; ")):
                reasons[r] = reasons.get(r, 0) + 1
        else:
            issued += 1
        parsed = candidate_from_row(row)
        if parsed is None:
            continue
        sid, c, _ = parsed
        what = (f"{row['symbol']} {c.side} {sid} (signal {c.signal_time:%H:%M} UTC)"
                + (" [SUPPRESSED]" if sup else ""))
        exit_txt = (f"close at {c.time_exit_utc:%H:%M} UTC ({row.get('time_exit_local', '')}) "
                    "if still open" if c.time_exit_utc is not None else "no time exit")
        candidates.append(f"  - {what}: {exit_txt}; {_occupancy_text(c, sid, sup, bars)}")
        if not sup and c.order_type in PENDING_TYPES and c.expires_utc is not None:
            pending.append(f"  - {what}: {c.order_type} at {c.entry}, expires "
                           f"{c.expires_utc:%H:%M} UTC")
    lines = [f"# Daily summary {d} (UTC)", "",
             f"- Tickets issued: {issued}", f"- Tickets suppressed: {suppressed}"]
    lines += [f"  - {r}: {n}" for r, n in sorted(reasons.items(), key=lambda x: -x[1])]
    lines.append(f"- Protocol candidates and manual time exits (suppressed included): "
                 f"{len(candidates)}")
    lines += candidates
    reminders = [r for r in _read_csv(REMINDERS_CSV) if r["reminded_utc"].startswith(d)]
    lines.append(f"- EXIT REMINDERs issued: {len(reminders)}")
    lines += [f"  - {r['symbol']} {r['side']} {r['strategy_id']}: time exit "
              f"{pd.Timestamp(r['time_exit_utc']):%H:%M} UTC"
              + (" (ticket was suppressed)" if r["suppressed"] == "True" else "")
              for r in reminders]
    lines.append(f"- Limit / stop tickets issued: {len(pending)}")
    lines += pending
    try:
        state = load_state()
        lines.append(f"- Pending orders in state.yaml: {len(state.pending_orders)}")
        lines += [f"  - {o.symbol} {o.side} {o.lots} lots, risk ${o.risk_usd:,.0f}, expires "
                  + (f"{o.expires_utc:%Y-%m-%d %H:%M} UTC" if o.expires_utc is not None else "n/a")
                  for o in state.pending_orders]
        lines.append(f"- Trading days completed: {state.trading_days_completed}")
    except (FileNotFoundError, ValueError):
        pass
    regime = [r for r in _read_csv(REGIME_CSV) if r["time_utc"].startswith(d)]
    lines.append(f"- Regime history: {len(regime)} snapshot(s)")
    lines += [f"  - {pd.Timestamp(r['time_utc']):%H:%M} {r['key']}: V_T {r['v_t']} "
              f"{r['label']} status {r['status']} gate {r['gate']}" for r in regime]
    out_dir = out_dir or ROOT / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"daily_summary_{d}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote %s", out)
    return out


def seconds_to_next_close(tf_min: int, now: pd.Timestamp, delay_s: int = 5) -> float:
    nxt = now.floor(f"{tf_min}min") + pd.Timedelta(minutes=tf_min, seconds=delay_s)
    return max(1.0, (nxt - now).total_seconds())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--format-check", action="store_true")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config()
    if not cfg.get("owner_confirmed"):
        log.warning("owner_confirmed is false in config.yaml: every ticket will be suppressed")
    creds = load_credentials()
    client = MT5Client(cfg)
    symbols: dict[str, str] = {}
    last_bars: dict[str, pd.DataFrame] = {}   # closed bars of the last cycle
    last_day = utc_now().normalize()
    tf_min = tf_minutes(cfg["timeframes"]["entry"])
    try:
        while True:
            try:
                if not symbols:
                    client.connect(creds)
                    symbols = client.resolve_symbols()
                    client.check_server_clock(next(iter(symbols.values())))
                last_bars = cycle(client, cfg, symbols, args.format_check)
            except MT5Error as exc:
                no_signal(f"ERROR {exc}")
                symbols = {}
                client.shutdown()
                # Without bars SL / TP touches cannot be checked, so every candidate due for
                # its time exit is reminded ("close if still open" is safe either way).
                now = utc_now()
                issue_exit_reminders(recent_rows(now), {}, cfg, now)
            if args.once or args.format_check:
                break
            today = utc_now().normalize()
            if today > last_day:
                write_daily_summary(last_day, bars=last_bars)
                last_day = today
            time.sleep(seconds_to_next_close(tf_min, utc_now()))
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
