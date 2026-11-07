from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import signals.run as run
from core.config import Credentials
from core.timeutil import server_offset, utc_now
from data.mt5_adapter import MT5Client, SymbolSpec
from risk.rules import BLOCKED_OPEN_POSITION, RecentTicket, check_ticket
from risk.sizing import size_position
from risk.state import load_state, parse_state
from signals.engine import build_ticket, check_fresh
from signals.ticket import log_ticket
from strategies.base import MARKET_BAND, Candidate, StrategyContext
from tests.fakes import FakeMT5, make_rates

NOW = pd.Timestamp("2026-11-02 15:00", tz="UTC")
# US30-like: 1 lot = $1 per 1.0 move (tick 0.1 worth $0.10)
SPEC = SymbolSpec("US30.cash", 1, 0.1, 0.1, 0.1, 0.01, 100, 0.01, "USD")
STATS = {"oos_win_rate": 0.46, "expectancy_r": 0.31, "max_dd_pct": 7.9, "n_trades": 212}


def state(**over):
    raw = {"updated_at": "2026-11-02T14:00:00Z", "balance": 100000, "equity": 100000,
           "start_of_day_balance": 100000, "start_of_day_equity": 100000,
           "trades_today": 0, "trading_days_completed": 2, "open_positions": []}
    raw.update(over)
    return parse_state(raw)


# --- sizing -------------------------------------------------------------------------------
def test_sizing_basic():
    # $500 risk, 100.0 stop => $100 per lot => 5 lots, capped at 3
    r = size_position(100000, 0.5, 42000, 41900, SPEC, lot_cap=3)
    assert r.lots == 3 and r.risk_usd == 300 and "capped" in r.note
    assert r.stop_points == pytest.approx(1000)


def test_sizing_rounds_down():
    r = size_position(100000, 0.5, 42000, 41700, SPEC, lot_cap=3)   # 1.6667 lots
    assert r.lots == 1.66


def test_sizing_below_min_is_suppressed():
    r = size_position(1000, 0.01, 42000, 40000, SPEC, lot_cap=3)
    assert r.lots is None and "below the minimum" in r.note


# --- rules --------------------------------------------------------------------------------
def check(cfg, st, lots=1.0, risk=500.0, key="US30"):
    return check_ticket(key=key, asset_class="index", lots=lots, risk_usd=risk,
                        state=st, cfg=cfg, now=NOW)


def test_clean_ticket_passes(cfg):
    res = check(cfg, state())
    assert res.ok, res.reasons


def test_daily_loss_floor(cfg):
    # down $4,700 today; a $500 risk (+10% buffer) would breach the $95,000 floor
    res = check(cfg, state(balance=95300, equity=95300))
    assert not res.ok and any("daily loss floor" in r for r in res.reasons)


def test_personal_stop(cfg):
    res = check(cfg, state(balance=97200, equity=97200))
    assert any("personal daily stop" in r for r in res.reasons)


def test_open_risk_counts(cfg):
    positions = [{"symbol": "NAS100", "side": "BUY", "lots": 1, "risk_usd": 2700}]
    res = check(cfg, state(open_positions=positions))
    assert any("personal daily stop" in r for r in res.reasons)


def test_total_loss_floor(cfg):
    st = state(balance=90300, equity=90300, start_of_day_balance=90300, start_of_day_equity=90300)
    res = check(cfg, st)
    assert any("max loss floor" in r for r in res.reasons)


def test_position_trade_and_lot_limits(cfg):
    positions = [{"symbol": "X", "side": "BUY", "lots": 0.01, "risk_usd": 1}] * 5
    res = check(cfg, state(open_positions=positions, trades_today=50), lots=3.5)
    joined = " ".join(res.reasons)
    assert "open positions" in joined and "trades today" in joined and "exceeds" in joined


def test_stale_state_and_unconfirmed_owner(cfg):
    cfg["owner_confirmed"] = False
    res = check(cfg, state(updated_at="2026-11-01T00:00:00Z"))
    joined = " ".join(res.reasons)
    assert "owner settings" in joined and "state.yaml" in joined


def test_competition_end_and_sessions(cfg):
    cfg["rules"]["competition_end_utc"] = "2026-11-02T15:30:00Z"
    cfg["sessions_utc"] = {"US30": [["13:30", "14:30"]]}
    joined = " ".join(check(cfg, state()).reasons)
    assert "competition end" in joined and "trading hours" in joined


def test_restricted_window(cfg):
    cfg["rules"]["restricted_windows_utc"] = [
        {"start": "2026-11-02T14:45:00Z", "end": "2026-11-02T15:15:00Z", "reason": "NFP"}]
    assert any("NFP" in r for r in check(cfg, state()).reasons)


# --- freshness ----------------------------------------------------------------------------
def test_freshness():
    assert check_fresh(NOW - pd.Timedelta(minutes=15), 15, NOW)[0]
    ok, msg = check_fresh(NOW - pd.Timedelta(minutes=60), 15, NOW)
    assert not ok and "stale" in msg


# --- tickets ------------------------------------------------------------------------------
def candidate(**over):
    base = dict(key="US30", side="BUY", entry=42000.0, stop=41900.0, take_profit=42200.0,
                atr=60.0, reason="test", regime="bull-quiet", regime_prob=0.82, rank_score=1.35)
    base.update(over)
    return Candidate(**base)


def make(cfg, c=None, stats=STATS, st="default"):
    return build_ticket(c or candidate(), strategy_id="RS-MOM-01", stats=stats,
                        symbol="US30.cash", asset_class="index", spec=SPEC,
                        state=state() if st == "default" else st, state_error="no state.yaml",
                        cfg=cfg, now=NOW, tf_min=15)


def test_ticket_format(cfg):
    t = make(cfg)
    assert not t.suppressed, t.suppress_reasons
    text = t.format()
    assert text.splitlines()[0] == (
        "2026-11-02 15:00 UTC | US30.cash | BUY | 42000.0 | SL 41900.0 | TP 42200.0 | "
        "3.00 lots | risk $300.00 | R:R 2.00")
    assert "REGIME bull-quiet (p=0.82) | RANK +1.35 | STRATEGY RS-MOM-01 | VALID UNTIL 15:45 UTC" in text
    assert "win 46.0% | expectancy +0.31R | max DD 7.9% | n=212" in text
    assert "(1,000 points)" in text and "Skip if: price has moved more than 15.0" in text


def test_ticket_suppressed_without_stats_or_state(cfg):
    t = make(cfg, stats={}, st=None)
    assert t.suppressed
    assert "strategy has no validated statistics" in t.suppress_reasons
    assert "no state.yaml" in t.suppress_reasons
    assert t.format().startswith("SUPPRESSED - DO NOT TRADE")


def test_ticket_wrong_side_stop(cfg):
    t = make(cfg, candidate(side="SELL"))
    assert "stop/target on the wrong side of entry" in t.suppress_reasons


# --- pending orders, per-index block, recent tickets, session on T (ER-04, ER-12) ----------
def pending(symbol="GER40", risk=2700.0, expires="2026-11-02T16:00:00Z"):
    return {"symbol": symbol, "side": "BUY", "lots": 1.0, "risk_usd": risk, "expires_utc": expires}


def blocked(res) -> bool:
    return any(r.startswith(BLOCKED_OPEN_POSITION) for r in res.reasons)


def test_pending_order_risk_counts_until_expiry(cfg):
    live = check(cfg, state(pending_orders=[pending()]))           # expires 16:00 > NOW
    assert any("personal daily stop" in r for r in live.reasons)
    assert any("pending order(s) counted" in n for n in live.notes)
    expired = check(cfg, state(pending_orders=[pending(expires="2026-11-02T14:30:00Z")]))
    assert expired.ok, expired.reasons
    no_expiry = check(cfg, state(pending_orders=[pending(expires=None)]))
    assert not no_expiry.ok                                        # counts until removed


def test_per_index_block_open_position_and_pending(cfg):
    pos = [{"symbol": "US30.cash", "side": "SELL", "lots": 0.1, "risk_usd": 10}]
    assert blocked(check(cfg, state(open_positions=pos)))
    assert check(cfg, state(open_positions=pos), key="NAS100").ok   # other index unaffected
    st = state(pending_orders=[pending(symbol="DE40", risk=10)])   # broker alias of GER40
    assert blocked(check(cfg, st, key="GER40"))
    assert check(cfg, st, key="US30").ok
    gone = state(pending_orders=[pending(symbol="DE40", risk=10, expires="2026-11-02T14:00:00Z")])
    assert check(cfg, gone, key="GER40").ok                        # expired: no block


def test_session_check_runs_on_signal_time(cfg):
    cfg["sessions_utc"] = {"US30": [["13:30", "20:00"]]}
    late_wall = pd.Timestamp("2026-11-02 20:05", tz="UTC")

    def at(t):
        return check_ticket(key="US30", asset_class="index", lots=1.0, risk_usd=500.0,
                            state=state(updated_at="2026-11-02T19:00:00Z"), cfg=cfg,
                            now=late_wall, signal_time=pd.Timestamp(t, tz="UTC"))
    assert at("2026-11-02 19:45").ok                               # T inside, wall clock not
    assert at("2026-11-02 13:30").ok                               # start inclusive
    assert any("trading hours" in r for r in at("2026-11-02 20:00").reasons)   # end exclusive
    assert any("trading hours" in r for r in at("2026-11-02 13:15").reasons)


def test_recent_tickets_count_as_open(cfg):
    new = RecentTicket("NAS100", "BUY", 1.0, 2700.0, pd.Timestamp("2026-11-02 14:30", tz="UTC"))
    old = RecentTicket("NAS100", "BUY", 1.0, 2700.0, pd.Timestamp("2026-11-02 13:30", tz="UTC"))
    four = [{"symbol": "X", "side": "BUY", "lots": 0.01, "risk_usd": 1}] * 4

    def run_check(st, recent):
        return check_ticket(key="US30", asset_class="index", lots=1.0, risk_usd=500.0, state=st,
                            cfg=cfg, now=NOW, recent_tickets=recent)
    assert run_check(state(), [old]).ok                            # already in state.yaml
    res = run_check(state(open_positions=four, trades_today=49), [new])
    joined = " ".join(res.reasons)
    assert "already 5 open positions" in joined and "50 trades today" in joined
    assert "personal daily stop" in joined                         # its risk counts too


def test_state_example_parses_with_pending_orders():
    st = load_state(Path(__file__).resolve().parent.parent / "state.example.yaml")
    assert st.pending_orders == () and st.open_positions == ()
    st2 = state(pending_orders=[pending()])
    assert st2.pending_orders[0].expires_utc == pd.Timestamp("2026-11-02 16:00", tz="UTC")
    assert st2.pending_risk_usd(NOW) == 2700.0
    assert st2.pending_risk_usd(pd.Timestamp("2026-11-02 16:00", tz="UTC")) == 0.0


def test_engine_blocks_occupied_index(cfg):
    t = build_ticket(candidate(), strategy_id="RS-MOM-01", stats=STATS, symbol="US30.cash",
                     asset_class="index", spec=SPEC, state=state(), state_error="", cfg=cfg,
                     now=NOW, tf_min=15, occupied_by="US30 occupied by SES-IBX-01")
    assert "blocked_open_position: US30 occupied by SES-IBX-01" in t.suppress_reasons


# --- run cycle: per-index freshness and groups (ER-18), reminders, summary (ER-03, ER-17) ---
SYMS = ("US30.cash", "USTEC", "US500.cash", "DE40")


class FourIndexMT5(FakeMT5):
    def __init__(self):
        super().__init__(symbols=SYMS)


@pytest.fixture
def cycle_env(cfg, tmp_path, monkeypatch):
    """A fixed decision time, no polling wait, and every output file under tmp_path."""
    now = utc_now().floor("15min") + pd.Timedelta(seconds=5)
    monkeypatch.setattr(run, "utc_now", lambda: now)
    monkeypatch.setattr(run, "POLL_GRACE", pd.Timedelta(0))
    monkeypatch.setattr(run, "SIGNALS_CSV", tmp_path / "signals.csv")
    monkeypatch.setattr(run, "REMINDERS_CSV", tmp_path / "exit_reminders.csv")
    monkeypatch.setattr(run, "REGIME_CSV", tmp_path / "regime_history.csv")
    monkeypatch.setattr(run, "notify_message", lambda *a: None)
    logged = []
    monkeypatch.setattr(run, "log_ticket", logged.append)
    monkeypatch.setattr(run, "load_state",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("no state.yaml")))
    return now, logged


def four_index_client(cfg, now, stale=()):
    fake = FourIndexMT5()
    zone, shift = cfg["server_timezone"]["zone"], cfg["server_timezone"]["shift_hours"]
    t = now.floor("15min")
    for key, sym in zip(("US30", "NAS100", "US500", "GER40"), SYMS):
        last_open = t - pd.Timedelta(minutes=15 + (120 if key in stale else 0))
        server_last = (last_open + server_offset(last_open, zone, shift)).tz_localize(None)
        start = server_last - pd.Timedelta(minutes=15 * 99)
        fake.rates[(sym, fake.TIMEFRAME_M15)] = make_rates(str(start), 100, 15, 20000)
        fake.rates[(sym, fake.TIMEFRAME_H1)] = make_rates(str(start), 30, 60, 20000)
    client = MT5Client(cfg, fake)
    client.connect(Credentials(None, None, None, r"C:\MT5-Data\terminal64.exe", None))
    return client, client.resolve_symbols()


def test_one_stale_index_silences_only_its_group(cfg, cycle_env, capsys):
    now, logged = cycle_env
    client, symbols = four_index_client(cfg, now, stale=("GER40",))
    run.cycle(client, cfg, symbols, format_check=True)
    out = capsys.readouterr().out
    assert "| US30.cash | BUY |" in out and "| USTEC | BUY |" in out
    assert "NO SIGNAL | GER40: incomplete group: GER40: data stale" in out
    assert "| DE40 |" not in out and "US500" not in out           # US500 never ticketed
    assert sorted(t.key for t in logged) == ["NAS100", "US30"]


def test_stale_us_member_removes_whole_us_group(cfg, cycle_env, capsys):
    now, logged = cycle_env
    client, symbols = four_index_client(cfg, now, stale=("NAS100",))
    run.cycle(client, cfg, symbols, format_check=True)
    out = capsys.readouterr().out
    assert "NO SIGNAL | US30, NAS100: incomplete group: NAS100: data stale" in out
    assert "| DE40 | BUY |" in out and "| US30.cash |" not in out
    assert [t.key for t in logged] == ["GER40"]


def test_all_stale_is_global_no_signal(cfg, cycle_env, capsys):
    now, logged = cycle_env
    client, symbols = four_index_client(cfg, now, stale=("US30", "NAS100", "US500", "GER40"))
    run.cycle(client, cfg, symbols, format_check=True)
    out = capsys.readouterr().out
    assert out.count("NO SIGNAL") == 1 and "US30: data stale" in out and "GER40: data stale" in out
    assert "US500" not in out and not logged


def test_format_check_candidate_is_market_band(cfg, cycle_env, capsys):
    now, logged = cycle_env
    client, symbols = four_index_client(cfg, now)
    run.cycle(client, cfg, symbols, format_check=True)
    out = capsys.readouterr().out
    assert "ORDER TYPE: MARKET_BAND" in out and "MANUAL TIME EXIT: close at" in out
    assert len(logged) == 3
    assert all(t.suppressed and t.order_type == MARKET_BAND for t in logged)


def test_no_strategy_with_fresh_data(cfg, cycle_env, capsys):
    now, _ = cycle_env
    client, symbols = four_index_client(cfg, now)
    run.cycle(client, cfg, symbols, format_check=False)
    assert "NO SIGNAL | no QA-approved strategy yet" in capsys.readouterr().out


def _log_candidate(cfg, path, c, sid="SES-IBX-01"):
    t = build_ticket(c, strategy_id=sid, stats=STATS, symbol="US30.cash", asset_class="index",
                     spec=SPEC, state=state(), state_error="", cfg=cfg,
                     now=c.signal_time, tf_min=15)
    log_ticket(t, path)


def _m15(start, ranges):
    return pd.DataFrame({"time_utc": [start + i * pd.Timedelta(minutes=15)
                                      for i in range(len(ranges))],
                         "high": [h for h, _ in ranges], "low": [lo for _, lo in ranges]})


def test_exit_reminder_issued_once_for_reference_open_candidate(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "notify_message", lambda *a: None)
    sig, rem = tmp_path / "signals.csv", tmp_path / "reminders.csv"
    t = pd.Timestamp("2026-11-02 15:00", tz="UTC")              # US winter: exit 20:45 UTC
    _log_candidate(cfg, sig, candidate(signal_time=t, order_type=MARKET_BAND))
    _log_candidate(cfg, sig, candidate(side="SELL", entry=42000.0, stop=42100.0,
                                       take_profit=41800.0, signal_time=t,
                                       order_type=MARKET_BAND), sid="RS-MOM-01")
    now = pd.Timestamp("2026-11-02 20:45:05", tz="UTC")
    rows = run.recent_rows(now, sig)
    # quiet bars for the BUY; the SELL's stop (42100) is touched at 16:00 -> not open
    quiet = [(42050.0, 41950.0)] * 23
    bars = {"US30": _m15(t, quiet[:4] + [(42150.0, 41990.0)] + quiet[5:])}
    msgs = run.issue_exit_reminders(rows, bars, cfg, now, rem)
    assert len(msgs) == 1 and "EXIT REMINDER | close US30.cash BUY" in msgs[0]
    assert "time exit 20:45 UTC (15:45 New York)" in msgs[0] and "SES-IBX-01" in msgs[0]
    assert run.issue_exit_reminders(rows, bars, cfg, now, rem) == []      # once only
    early = pd.Timestamp("2026-11-02 20:30:05", tz="UTC")
    assert run.issue_exit_reminders(run.recent_rows(early, sig), bars, cfg, early,
                                    tmp_path / "r2.csv") == []


def test_occupancy_rebuilt_from_rows_blocks_index(cfg, tmp_path):
    sig = tmp_path / "signals.csv"
    t = pd.Timestamp("2026-11-02 15:00", tz="UTC")
    later = pd.Timestamp("2026-11-02 16:00", tz="UTC")
    _log_candidate(cfg, sig, candidate(signal_time=t, order_type=MARKET_BAND))
    tracker = run.occupancy_from_rows(run.recent_rows(later, sig),
                                      {"US30": _m15(t, [(42050.0, 41950.0)] * 4)})
    assert tracker.occupied("US30", None, later)
    assert not tracker.occupied("NAS100", None, later)


def test_daily_summary_lists_regime_reminders_and_pending(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "SIGNALS_CSV", tmp_path / "signals.csv")
    monkeypatch.setattr(run, "REMINDERS_CSV", tmp_path / "reminders.csv")
    monkeypatch.setattr(run, "REGIME_CSV", tmp_path / "regime.csv")
    monkeypatch.setattr(run, "load_state", lambda: state(pending_orders=[pending()]))
    t = pd.Timestamp("2026-11-02 15:00", tz="UTC")
    _log_candidate(cfg, tmp_path / "signals.csv",
                   candidate(signal_time=t, order_type=MARKET_BAND))
    ctx = StrategyContext(regime={"US30": {"v_t": 0.12, "label": "ACTIVE/ACTIVE_FLAT",
                                           "status": "OK", "gate": "ACTIVE"}}, now=t, cfg=cfg)
    run.record_regime(ctx, t)
    text = run.write_daily_summary(t, out_dir=tmp_path).read_text(encoding="utf-8")
    assert "Protocol candidates and manual time exits (suppressed included): 1" in text
    assert "close at 20:45 UTC (15:45 New York)" in text and "unknown (no bars seen)" in text
    assert "EXIT REMINDERs issued: 0" in text
    assert "Pending orders in state.yaml: 1" in text and "GER40 BUY" in text
    assert "Regime history: 1 snapshot(s)" in text and "US30: V_T 0.12 ACTIVE/ACTIVE_FLAT" in text


def test_daily_summary_lists_suppressed_candidates_and_occupancy(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "SIGNALS_CSV", tmp_path / "signals.csv")
    monkeypatch.setattr(run, "REMINDERS_CSV", tmp_path / "reminders.csv")
    monkeypatch.setattr(run, "REGIME_CSV", tmp_path / "regime.csv")
    monkeypatch.setattr(run, "load_state", lambda: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(run, "notify_message", lambda *a: None)
    sig = tmp_path / "signals.csv"
    t = pd.Timestamp("2026-11-02 15:00", tz="UTC")              # exit 20:45 UTC
    _log_candidate(cfg, sig, candidate(signal_time=t, order_type=MARKET_BAND))
    sup = build_ticket(candidate(side="SELL", entry=42000.0, stop=42100.0, take_profit=41800.0,
                                 signal_time=t, order_type=MARKET_BAND),
                       strategy_id="RS-MOM-01", stats={}, symbol="US30.cash",
                       asset_class="index", spec=SPEC, state=state(), state_error="", cfg=cfg,
                       now=t, tf_min=15)
    assert sup.suppressed                                       # no stats
    log_ticket(sup, sig)
    quiet = [(42050.0, 41950.0)] * 24                            # 15:00 .. 21:00
    bars = {"US30": _m15(t, quiet[:4] + [(42150.0, 41990.0)] + quiet[5:])}
    now = pd.Timestamp("2026-11-02 20:45:05", tz="UTC")
    msgs = run.issue_exit_reminders(run.recent_rows(now, sig), bars, cfg, now)
    text = run.write_daily_summary(t, out_dir=tmp_path, bars=bars).read_text(encoding="utf-8")
    assert "Tickets issued: 1" in text and "Tickets suppressed: 1" in text
    assert "(suppressed included): 2" in text
    buy = next(x for x in text.splitlines() if "US30.cash BUY SES-IBX-01" in x)
    sell = next(x for x in text.splitlines() if "US30.cash SELL RS-MOM-01" in x)
    assert "[SUPPRESSED]" not in buy and "reference-open at its time exit" in buy
    assert "[SUPPRESSED]" in sell and "ended 16:15 UTC by SL" in sell
    assert len(msgs) == 1 and "EXIT REMINDERs issued: 1" in text      # agrees with the list


# --- polling for the bar at T (COMMON 9, ER-05, ER-18) ---------------------------------------
T_FIX = pd.Timestamp("2026-10-19 15:00", tz="UTC")     # past Monday, full hour: H1 polled too


class StubClient:
    """closed_bars(symbol, tf, count) only, returning frames scripted per call."""

    def __init__(self, frames: dict[str, list[pd.DataFrame]]):
        self.frames, self.calls = frames, {k: 0 for k in frames}

    def closed_bars(self, symbol, tf, count):
        self.calls[symbol] += 1
        seq = self.frames[symbol]
        return seq[min(self.calls[symbol], len(seq)) - 1].copy()


def _opens(last_open, n=5, minutes=15):
    step = pd.Timedelta(minutes=minutes)
    return pd.DataFrame({"time_utc": [last_open - (n - 1 - i) * step for i in range(n)],
                         "close": [1.0] * n})


@pytest.fixture
def fake_clock(monkeypatch):
    """run.utc_now from T + 5 s, advanced only by run's sleeps (no real waiting)."""
    clock = {"now": T_FIX + pd.Timedelta(seconds=5), "sleeps": 0}

    def sleep(s):
        clock["sleeps"] += 1
        clock["now"] += pd.Timedelta(seconds=s)

    monkeypatch.setattr(run, "utc_now", lambda: clock["now"])
    monkeypatch.setattr(run, "time", SimpleNamespace(sleep=sleep))
    return clock


def test_poll_bars_retries_until_late_bar_arrives(fake_clock):
    m15 = pd.Timedelta(minutes=15)
    client = StubClient({"US30.cash": [_opens(T_FIX - 2 * m15), _opens(T_FIX - m15)]})
    bars, present = run.poll_bars(client, {"US30": "US30.cash"}, "M15", T_FIX,
                                  T_FIX + run.POLL_GRACE)
    assert present == {"US30"} and client.calls["US30.cash"] == 2 and fake_clock["sleeps"] == 1
    assert bars["US30"]["time_utc"].iloc[-1] == T_FIX - m15


def test_poll_bars_cuts_forming_bar_and_bars_after_t(fake_clock):
    m15 = pd.Timedelta(minutes=15)
    # the adapter hands back the forming bar (open == T) and one more: both close after T
    client = StubClient({"US30.cash": [_opens(T_FIX + m15, n=6)]})
    bars, present = run.poll_bars(client, {"US30": "US30.cash"}, "M15", T_FIX,
                                  T_FIX + run.POLL_GRACE)
    assert present == {"US30"} and fake_clock["sleeps"] == 0
    assert (bars["US30"]["time_utc"] + m15 <= T_FIX).all()
    assert bars["US30"]["time_utc"].iloc[-1] == T_FIX - m15


def test_poll_bars_h1_at_full_hour(fake_clock):
    h1 = pd.Timedelta(hours=1)
    client = StubClient({"US30.cash": [_opens(T_FIX - 2 * h1, minutes=60),
                                       _opens(T_FIX, minutes=60)]})   # 2nd: incl. forming H1
    bars, present = run.poll_bars(client, {"US30": "US30.cash"}, "H1", T_FIX,
                                  T_FIX + run.POLL_GRACE)
    assert present == {"US30"} and client.calls["US30.cash"] == 2
    assert bars["US30"]["time_utc"].iloc[-1] == T_FIX - h1


def test_poll_bars_gives_up_at_deadline(fake_clock):
    m15 = pd.Timedelta(minutes=15)
    client = StubClient({"US30.cash": [_opens(T_FIX - 2 * m15)]})
    bars, present = run.poll_bars(client, {"US30": "US30.cash"}, "M15", T_FIX,
                                  T_FIX + run.POLL_GRACE)
    assert present == set() and fake_clock["now"] >= T_FIX + run.POLL_GRACE
    assert bars["US30"]["time_utc"].iloc[-1] == T_FIX - 2 * m15


class LateBarMT5(FourIndexMT5):
    """Withholds the newest M15 / H1 bar of some symbols (it never arrives)."""

    def __init__(self, late_m15=(), late_h1=()):
        super().__init__()
        self.late = {(s, self.TIMEFRAME_M15) for s in late_m15}
        self.late |= {(s, self.TIMEFRAME_H1) for s in late_h1}

    def copy_rates_from_pos(self, symbol, tf, start, count):
        rates = super().copy_rates_from_pos(symbol, tf, start, count)
        return rates[:-1] if (symbol, tf) in self.late and rates is not None else rates


def fixed_t_client(cfg, fake, extra_m15=0):
    """Bars up to the one closing at T_FIX, plus `extra_m15` bars after T (a live adapter
    returns those once they have closed on the broker clock)."""
    zone, shift = cfg["server_timezone"]["zone"], cfg["server_timezone"]["shift_hours"]
    last_open = T_FIX - pd.Timedelta(minutes=15)
    server_last = (last_open + server_offset(last_open, zone, shift)).tz_localize(None)
    start = server_last - pd.Timedelta(minutes=15 * 99)
    h1_start = start.floor("60min")
    for sym in SYMS:
        fake.rates[(sym, fake.TIMEFRAME_M15)] = make_rates(str(start), 100 + extra_m15, 15,
                                                           20000)
        fake.rates[(sym, fake.TIMEFRAME_H1)] = make_rates(str(h1_start), 25, 60, 20000)
    client = MT5Client(cfg, fake)
    client.connect(Credentials(None, None, None, r"C:\MT5-Data\terminal64.exe", None))
    return client, client.resolve_symbols()


@pytest.fixture
def fixed_env(cycle_env, fake_clock, monkeypatch):
    """cycle_env's output redirection, but T fixed at T_FIX and the real 60 s grace."""
    monkeypatch.setattr(run, "POLL_GRACE", pd.Timedelta(seconds=60))
    return cycle_env[1], fake_clock


def test_bar_missing_after_60s_drops_only_its_group(cfg, fixed_env, capsys):
    logged, clock = fixed_env
    client, symbols = fixed_t_client(cfg, LateBarMT5(late_m15=("DE40",)))
    run.cycle(client, cfg, symbols, format_check=True)
    out = capsys.readouterr().out
    assert ("NO SIGNAL | GER40: incomplete group: GER40: bar closing at 15:00 UTC missing "
            "after 60 s (last bar closed 14:45 UTC)") in out
    assert sorted(t.key for t in logged) == ["NAS100", "US30"]
    assert clock["sleeps"] >= 25                                 # polled until T + 60 s


def test_cycle_cuts_bars_after_t_before_strategies(cfg, fixed_env):
    logged, _ = fixed_env
    client, symbols = fixed_t_client(cfg, LateBarMT5(), extra_m15=2)
    bars = run.cycle(client, cfg, symbols, format_check=True)
    m15 = pd.Timedelta(minutes=15)
    assert all(df["time_utc"].iloc[-1] == T_FIX - m15 for df in bars.values())
    expected = float(make_rates("2026-01-01", 100, 15, 20000)["close"][-1])   # bar closing at T
    assert len(logged) == 3 and all(t.entry == pytest.approx(expected) for t in logged)


def test_missing_h1_at_full_hour_is_logged(cfg, fixed_env, caplog):
    client, symbols = fixed_t_client(cfg, LateBarMT5(late_h1=("US30.cash",)))
    with caplog.at_level("WARNING", logger="signals.run"):
        run.cycle(client, cfg, symbols, format_check=True)
    assert "US30: H1 bar closing at 15:00 UTC not arrived by T + 60 s" in caplog.text
    assert "NAS100: H1" not in caplog.text
