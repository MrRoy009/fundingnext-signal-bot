"""Execution protocols on candidates and tickets (COMMON 7, ER-01, ER-02).

Levels are pure arithmetic on E, SL and the frozen fractions in config `protocols`; nothing here
looks at prices after T or at any outcome.
"""
import pandas as pd
import pytest

from data.mt5_adapter import SymbolSpec
from risk.state import parse_state
from signals.engine import build_ticket, prepare_candidate, valid_until
from signals.ticket import (CSV_FIELDS, candidate_from_row, log_ticket, read_signal_rows,
                            signals_csv_path)
from strategies.base import (MARKET_BAND, MARKET_OR_LIMIT, STOP_ENTRY, Candidate,
                             StrategyContext, protocol_levels, time_exit_label, time_exit_utc,
                             with_protocol_levels)

T = pd.Timestamp("2026-11-02 15:00", tz="UTC")
# US30-like: 1 lot = $1 per 1.0 move (tick 0.1 worth $0.10), stops level 10 points
SPEC = SymbolSpec("US30.cash", 1, 0.1, 0.1, 0.1, 0.01, 100, 0.01, "USD", trade_stops_level=10)
STATS = {"oos_win_rate": 0.46, "expectancy_r": 0.31, "max_dd_pct": 7.9, "n_trades": 212}


def state():
    return parse_state({"updated_at": "2026-11-02T14:00:00Z", "balance": 100000,
                        "equity": 100000, "start_of_day_balance": 100000,
                        "start_of_day_equity": 100000, "trades_today": 0,
                        "trading_days_completed": 2, "open_positions": []})


def cand(**over):
    base = dict(key="US30", side="BUY", entry=42000.0, stop=41700.0, take_profit=42600.0,
                atr=60.0, reason="test", regime="ACTIVE/ACTIVE_FLAT", regime_prob=0.8,
                rank_score=1.35, signal_time=T, order_type=MARKET_BAND)
    base.update(over)
    return Candidate(**base)


def ticket(cfg, c, now=T + pd.Timedelta(seconds=5)):
    return build_ticket(c, strategy_id="RS-MOM-01", stats=STATS, symbol="US30.cash",
                        asset_class="index", spec=SPEC, state=state(), state_error="",
                        cfg=cfg, now=now, tf_min=15)


# --- levels -------------------------------------------------------------------------------
@pytest.mark.parametrize("order_type, side, entry, stop, chase, skip, sizing", [
    # D = 10 in every row; chase 0.20 D on the worse side, skip 0.50 D on the stop side
    (MARKET_BAND, "BUY", 100.0, 90.0, 102.0, 95.0, 102.0),
    (MARKET_BAND, "SELL", 100.0, 110.0, 98.0, 105.0, 98.0),
    (MARKET_OR_LIMIT, "BUY", 100.0, 90.0, None, 95.0, 100.0),
    (MARKET_OR_LIMIT, "SELL", 100.0, 110.0, None, 105.0, 100.0),
    (STOP_ENTRY, "BUY", 100.0, 90.0, 102.0, None, 102.0),
    (STOP_ENTRY, "SELL", 100.0, 110.0, 98.0, None, 98.0),
])
def test_protocol_levels(cfg, order_type, side, entry, stop, chase, skip, sizing):
    lv = protocol_levels(order_type, side, entry, stop, cfg)
    assert lv.chase_limit == (pytest.approx(chase) if chase is not None else None)
    assert lv.skip_level == (pytest.approx(skip) if skip is not None else None)
    assert lv.sizing_entry == pytest.approx(sizing)


def test_levels_read_fractions_from_config(cfg):
    cfg["protocols"] = {"chase_frac": 0.1, "skip_frac": 0.3}
    lv = protocol_levels(MARKET_BAND, "BUY", 100.0, 90.0, cfg)
    assert (lv.chase_limit, lv.skip_level) == (pytest.approx(101.0), pytest.approx(97.0))


def test_unknown_protocol_or_side_raises(cfg):
    with pytest.raises(ValueError):
        protocol_levels("MARKET", "BUY", 100.0, 90.0, cfg)
    with pytest.raises(ValueError):
        protocol_levels(MARKET_BAND, "LONG", 100.0, 90.0, cfg)


@pytest.mark.parametrize("over", [
    {"chase_limit": 103.0},                      # wider than 0.20 D: RISK $ would not cover it
    {"chase_limit": 101.0},                      # narrower: still not the frozen fraction
    {"skip_level": 94.0},
    {"sizing_entry": 101.0},                     # sizing must be the CHASE LIMIT
    {"order_type": MARKET_OR_LIMIT, "chase_limit": 102.0},   # this protocol never chases
    {"order_type": STOP_ENTRY, "skip_level": 95.0},          # nor has a skip level
])
def test_with_protocol_levels_rejects_levels_that_differ_from_cfg(cfg, over):
    base = dict(entry=100.0, stop=90.0, take_profit=120.0)
    with pytest.raises(ValueError, match="differs from the protocol level"):
        with_protocol_levels(cand(**{**base, **over}), cfg)


def test_with_protocol_levels_derives_from_cfg_and_sizes_legacy_on_entry(cfg):
    restated = cand(entry=100.0, stop=90.0, take_profit=120.0, chase_limit=100.0 + 0.2 * 10,
                    skip_level=95.0, sizing_entry=102.0)       # same levels: accepted
    c = with_protocol_levels(restated, cfg)
    assert (c.chase_limit, c.skip_level, c.sizing_entry) == (pytest.approx(102.0),
                                                             pytest.approx(95.0),
                                                             pytest.approx(102.0))
    assert c.sizing_entry == c.chase_limit                     # COMMON 7 worst case
    assert with_protocol_levels(c, cfg) == c                   # idempotent
    legacy = with_protocol_levels(cand(order_type=None), cfg)
    assert legacy.sizing_entry == legacy.entry and legacy.chase_limit is None


def test_ticket_with_overridden_level_is_suppressed_and_uses_cfg_levels(cfg):
    t = ticket(cfg, cand(chase_limit=42100.0))                 # cfg: 42000 + 0.2 * 300 = 42060
    assert t.suppressed and any("invalid protocol levels" in r for r in t.suppress_reasons)
    assert t.chase_limit == pytest.approx(42060.0) and t.sizing_entry == t.chase_limit
    assert "CHASE LIMIT 42060.0" in t.format() and "CHASE LIMIT 42100" not in t.format()


def test_unknown_order_type_still_raises(cfg):
    with pytest.raises(ValueError):
        ticket(cfg, cand(order_type="MARKET"))


def test_trigger_and_d(cfg):
    stop_c = cand(order_type=STOP_ENTRY, entry=100.0, stop=90.0, take_profit=120.0)
    assert stop_c.trigger == 100.0 and stop_c.d == 10.0
    assert cand().trigger is None


def test_strategy_context_is_minimal(cfg):
    ctx = StrategyContext(features={"x": 1}, regime=None, now=T, cfg=cfg)
    assert ctx.now == T and ctx.features == {"x": 1} and ctx.cfg is cfg


# --- validity (ER-02) ----------------------------------------------------------------------
def test_valid_until_market_is_t_plus_valid_for_bars(cfg):
    c = prepare_candidate(cand(), cfg, T)
    assert valid_until(c, cfg, 15) == T + pd.Timedelta(minutes=45)
    cfg["execution"]["valid_for_bars"] = 2
    assert valid_until(c, cfg, 15) == T + pd.Timedelta(minutes=30)


def test_valid_until_is_min_of_window_and_expiry(cfg):
    short = cand(order_type=MARKET_OR_LIMIT, expires_utc=T + pd.Timedelta(minutes=30))
    long = cand(order_type=STOP_ENTRY, expires_utc=T + pd.Timedelta(hours=2))
    assert valid_until(short, cfg, 15) == T + pd.Timedelta(minutes=30)
    assert valid_until(long, cfg, 15) == T + pd.Timedelta(minutes=45)


def test_ticket_after_valid_until_is_suppressed(cfg):
    t = ticket(cfg, cand(), now=T + pd.Timedelta(minutes=45))
    assert t.suppressed and any("no longer valid" in r for r in t.suppress_reasons)


# --- time exit (both DST states, COMMON 2) -------------------------------------------------
@pytest.mark.parametrize("key, t, expected", [
    ("US30", "2026-07-15 14:00", "2026-07-15 19:45"),     # US summer (EDT)
    ("US30", "2026-12-15 15:00", "2026-12-15 20:45"),     # US winter (EST)
    ("NAS100", "2026-03-10 14:00", "2026-03-10 19:45"),   # US already on DST, EU not yet
    ("GER40", "2026-07-15 08:00", "2026-07-15 15:15"),    # EU summer (CEST)
    ("GER40", "2026-12-15 09:00", "2026-12-15 16:15"),    # EU winter (CET)
    ("GER40", "2026-03-10 09:00", "2026-03-10 16:15"),    # mismatch week: EU still winter
])
def test_time_exit_utc_follows_local_dst(cfg, key, t, expected):
    assert time_exit_utc(cfg, key, pd.Timestamp(t, tz="UTC")) == pd.Timestamp(expected, tz="UTC")


def test_time_exit_label(cfg):
    assert time_exit_label(cfg, "US30") == "15:45 New York"
    assert time_exit_label(cfg, "GER40") == "17:15 Berlin"
    assert time_exit_utc(cfg, "US500", T) is None          # no time exit configured


@pytest.mark.parametrize("t, line", [
    ("2026-07-15 14:00", "MANUAL TIME EXIT: close at 19:45 UTC (15:45 New York) if still open"),
    ("2026-11-02 15:00", "MANUAL TIME EXIT: close at 20:45 UTC (15:45 New York) if still open"),
])
def test_ticket_time_exit_line_both_dst_states(cfg, t, line):
    ts = pd.Timestamp(t, tz="UTC")
    text = ticket(cfg, cand(signal_time=ts), now=ts).format()
    assert line in text.splitlines()


def test_ger40_time_exit_line(cfg):
    ts = pd.Timestamp("2026-07-15 08:00", tz="UTC")
    c = cand(key="GER40", signal_time=ts)
    t = build_ticket(c, strategy_id="X", stats=STATS, symbol="DE40", asset_class="index",
                     spec=SPEC, state=state(), state_error="", cfg=cfg, now=ts, tf_min=15)
    assert "MANUAL TIME EXIT: close at 15:15 UTC (17:15 Berlin) if still open" in t.format()


# --- tickets ------------------------------------------------------------------------------
def test_market_band_ticket(cfg):
    # D = 300: CHASE LIMIT 42060, SKIP LEVEL 41850; lots sized on 42060 (360 to SL)
    t = ticket(cfg, cand())
    assert not t.suppressed, t.suppress_reasons
    assert t.sizing_entry == pytest.approx(42060.0)
    assert t.lots == 1.38 and t.risk_usd == pytest.approx(1.38 * 360)   # worst case
    text = t.format()
    lines = text.splitlines()
    assert lines[0] == ("2026-11-02 15:00 UTC | US30.cash | BUY | 42000.0 | SL 41700.0 | "
                        "TP 42600.0 | 1.38 lots | risk $496.80 | R:R 2.00")
    assert "VALID UNTIL 15:45 UTC (place the order by then)" in lines[1]
    assert "ORDER TYPE: MARKET_BAND" in lines
    assert ("ENTRY 42000.0 | CHASE LIMIT 42060.0 (60.0 = 600 points worse than entry) | "
            "SKIP LEVEL 41850.0 (150.0 = 1,500 points better than entry)") in lines
    assert "EXPIRY: none (market order only)" in lines
    assert any(ln.startswith("PLACE: market BUY now if ask is between 41850.0 and 42060.0")
               for ln in lines)
    assert "Skip if: price has moved" not in text        # symmetric line replaced
    assert "from the worst permitted entry 42060.0: RISK $ is the worst case" in text


def test_sell_market_band_levels_mirror(cfg):
    t = ticket(cfg, cand(side="SELL", entry=42000.0, stop=42300.0, take_profit=41400.0))
    text = t.format()
    assert "CHASE LIMIT 41940.0" in text and "SKIP LEVEL 42150.0" in text
    assert "Skip if bid < 41940.0 (CHASE LIMIT), bid > 42150.0 (SKIP LEVEL) or bid <= 41400.0" in text


def test_market_or_limit_ticket(cfg):
    exp = T + pd.Timedelta(minutes=30)
    t = ticket(cfg, cand(side="SELL", entry=42000.0, stop=42300.0, take_profit=41400.0,
                         order_type=MARKET_OR_LIMIT, expires_utc=exp))
    assert not t.suppressed, t.suppress_reasons
    assert t.sizing_entry == 42000.0 and t.lots == 1.66            # sized on E (300 to SL)
    assert t.chase_limit is None and t.skip_level == pytest.approx(42150.0)
    text = t.format()
    assert "ORDER TYPE: MARKET_OR_LIMIT" in text
    assert "ENTRY 42000.0 | SKIP LEVEL 42150.0 (150.0 = 1,500 points better than entry)" in text
    assert "EXPIRY 15:30 UTC (resting SELL LIMIT order)" in text
    assert "SELL LIMIT at 42000.0 expiring 15:30 UTC. Never chase." in text
    assert "VALID UNTIL 15:30 UTC (place the order by then; the resting order expires 15:30 UTC)" in text
    assert "CHASE LIMIT" not in text


def test_stop_entry_ticket(cfg):
    exp = T + pd.Timedelta(hours=2)
    t = ticket(cfg, cand(order_type=STOP_ENTRY, expires_utc=exp))
    text = t.format()
    assert t.sizing_entry == pytest.approx(42060.0)
    assert "ORDER TYPE: STOP_ENTRY" in text
    assert "TRIGGER 42000.0 | CHASE LIMIT 42060.0 (60.0 = 600 points beyond trigger) | STOPS BUFFER 1.0" in text
    assert "if ask < 41999.0 (trigger - stops buffer): BUY STOP at 42000.0 expiring 17:00 UTC" in text
    assert "elif ask <= 42060.0 (CHASE LIMIT): market BUY now; else skip." in text
    assert "VALID UNTIL 15:45 UTC (place the order by then; the resting order expires 17:00 UTC)" in text
    assert "SKIP LEVEL" not in text


def test_pending_protocol_without_expiry_is_suppressed(cfg):
    t = ticket(cfg, cand(order_type=STOP_ENTRY))
    assert t.suppressed and "STOP_ENTRY ticket has no EXPIRY" in t.suppress_reasons


def test_printed_levels_round_toward_entry(cfg):
    # D = 100.1: chase 42020.02 -> 42020.0, skip 41949.95 -> 41950.0 (both toward E);
    # sizing stays on the unrounded chase, so RISK $ is never understated.
    t = ticket(cfg, cand(stop=41899.9, take_profit=42400.0))
    text = t.format()
    assert "CHASE LIMIT 42020.0" in text and "SKIP LEVEL 41950.0" in text
    assert t.sizing_entry == pytest.approx(42020.02)


def test_not_tradeable_index_is_suppressed(cfg):
    c = cand(key="US500")
    t = build_ticket(c, strategy_id="X", stats=STATS, symbol="US500.cash", asset_class="index",
                     spec=SPEC, state=state(), state_error="", cfg=cfg, now=T, tf_min=15)
    assert t.suppressed and "US500 is not tradeable (config instruments)" in t.suppress_reasons


def test_legacy_ticket_keeps_symmetric_line(cfg):
    t = ticket(cfg, cand(order_type=None))
    text = t.format()
    assert "Skip if: price has moved more than 15.0 beyond entry" in text
    assert "ORDER TYPE" not in text and "MANUAL TIME EXIT" not in text


# --- signals.csv ---------------------------------------------------------------------------
def test_signals_csv_new_fields_and_roundtrip(cfg, tmp_path):
    path = tmp_path / "signals.csv"
    exp = T + pd.Timedelta(minutes=30)
    t = ticket(cfg, cand(order_type=MARKET_OR_LIMIT, expires_utc=exp))
    log_ticket(t, path)
    rows = read_signal_rows(path)
    assert list(rows[0]) == CSV_FIELDS
    sid, c, suppressed = candidate_from_row(rows[0])
    assert sid == "RS-MOM-01" and not suppressed
    assert c.order_type == MARKET_OR_LIMIT and c.signal_time == T and c.expires_utc == exp
    assert c.time_exit_utc == pd.Timestamp("2026-11-02 20:45", tz="UTC")
    assert c.skip_level == pytest.approx(41850.0) and c.sizing_entry == 42000.0
    assert rows[0]["trigger"] == ""                         # not a stop ticket


def test_signals_csv_trigger_column_for_stop_entry(cfg, tmp_path):
    path = tmp_path / "signals.csv"
    t = ticket(cfg, cand(order_type=STOP_ENTRY, expires_utc=T + pd.Timedelta(minutes=30)))
    assert t.trigger == 42000.0
    log_ticket(t, path)
    row = read_signal_rows(path)[0]
    assert "trigger" in CSV_FIELDS and float(row["trigger"]) == 42000.0
    assert candidate_from_row(row)[1].trigger == 42000.0


def test_csv_without_trigger_column_goes_to_next_version(cfg, tmp_path):
    path = tmp_path / "signals.csv"
    path.write_text(",".join(CSV_FIELDS[:-1]) + "\n", encoding="utf-8")   # pre-trigger header
    log_ticket(ticket(cfg, cand()), path)
    assert (tmp_path / "signals_v2.csv").exists()
    assert path.read_text(encoding="utf-8") == ",".join(CSV_FIELDS[:-1]) + "\n"


def test_old_signals_csv_is_never_rewritten(cfg, tmp_path):
    path = tmp_path / "signals.csv"
    old = "time_utc,key,symbol,side,entry\n2026-10-01 14:00:00+00:00,US30,US30.cash,BUY,1\n"
    path.write_text(old, encoding="utf-8")
    assert signals_csv_path(path) == tmp_path / "signals_v2.csv"
    log_ticket(ticket(cfg, cand()), path)
    assert path.read_text(encoding="utf-8") == old
    rows = read_signal_rows(path)
    assert len(rows) == 2 and candidate_from_row(rows[0]) is None   # old row: no protocol
    assert candidate_from_row(rows[1])[1].order_type == MARKET_BAND
