"""COMMON 8.6-8.10 on SYNTHETIC bars only: commission, swap (triple day, weekend, positive
swap, stress), sizing on the simulated balance with the lot cap, risk-rule suppression on the
simulated AccountState, concurrency (single and combined book), compounding, MTM equity and the
daily end-of-day series, $ / R accounting, contracts, determinism and no look-ahead.

Same setup as tests/test_backtest_fills.py: US30 / NAS100 on Wednesday 2024-07-10, T = 14:00
UTC, time exit 19:45 UTC, server midnight 21:00 UTC, spread 3.0 points, slip 2.0 (BASE).
"""
from dataclasses import replace

import pandas as pd
import pytest

import backtest.costs as costs
import backtest.engine as engine
from backtest.engine import StrategyCandidate, run_backtest, run_both_contracts
from backtest.results import TRADE_COLUMNS, report_tables, summarize
from backtest.scenarios import SCENARIOS, Scenario, contract_table
from backtest.synthetic import M15, concat_bars, flat_bars
from data.mt5_adapter import SymbolSpec
from strategies.base import MARKET_BAND, MARKET_OR_LIMIT, Candidate

T = pd.Timestamp("2024-07-10 14:00", tz="UTC")
NEXT = pd.Timestamp("2024-07-11 14:00", tz="UTC")
END = pd.Timestamp("2024-07-11 23:00", tz="UTC")


def spec(name="US30", **over):
    s = SymbolSpec(name, 2, 0.01, 0.01, 0.01, 0.1, 250.0, 0.1, "USD", swap_long=-500.0,
                   swap_short=100.0, swap_mode=1, swap_rollover3days=3, trade_stops_level=100)
    return replace(s, **over)


SPECS = {k: spec(k) for k in ("US30", "NAS100", "GER40", "US500")}
BASE = SCENARIOS["BASE"]
HOLD = Scenario(name="hold", missed_exit=True)       # no time exit, base costs


def at(minutes: int, t0: pd.Timestamp = T) -> pd.Timestamp:
    return t0 + pd.Timedelta(minutes=minutes)


def bars(overrides=None, start=T - M15, end=END, **kw):
    return flat_bars(start, end, overrides=overrides, **kw)


def cand(key="US30", side="BUY", t=T, entry=40000.0, d=400.0, tp=300.0, **over):
    sgn = 1.0 if side == "BUY" else -1.0
    c = dict(key=key, side=side, entry=entry, stop=entry - sgn * d,
             take_profit=entry + sgn * tp, atr=100.0, reason="t", signal_time=t,
             order_type=MARKET_BAND)
    c.update(over)
    return Candidate(**c)


def run(cands, bar_map, scenario=BASE, cfg=None, mode="single", **kw):
    pairs = [c if isinstance(c, tuple) else ("RS-MOM-01", c) for c in cands]
    return run_backtest(pairs, bar_map, cfg, scenario, SPECS, mode=mode, **kw)


def one(c, b=None, scenario=BASE, cfg=None, **kw):
    res = run([c], {c.key: b if b is not None else bars()}, scenario, cfg, **kw)
    return res.trades.iloc[0], res


# --- 8.6 commission ----------------------------------------------------------------------
def test_commission_base_and_stress(cfg):
    cfg["rules"]["commission_per_lot_side"] = 3.0
    r, _ = one(cand(), cfg=cfg)
    assert r.lots == 1.0 and r.commission_usd == pytest.approx(2 * 3.0)
    assert r.pnl_usd == pytest.approx((39998 - 40005) - 6.0)
    r, _ = one(cand(), scenario=SCENARIOS["STRESS-d2"], cfg=cfg)
    assert r.commission_usd == pytest.approx(2 * max(2 * 3.0, 1.0))


def test_stress_commission_floor_from_config(cfg):
    cfg["rules"]["commission_per_lot_side"] = 0
    r, _ = one(cand(), scenario=SCENARIOS["STRESS-d2"], cfg=cfg)
    assert r.commission_usd == pytest.approx(2 * 1.0 * r.lots)
    r, _ = one(cand(), cfg=cfg)
    assert r.commission_usd == 0.0


# --- 8.6 swap ----------------------------------------------------------------------------
def overnight_tp():
    return bars({NEXT: (40000.0, 40400.0, 39990.0, 40300.0)})


def test_swap_only_when_held_through_server_midnight(cfg):
    r, _ = one(cand(), cfg=cfg)                        # intraday: closed at the time exit
    assert r.rollovers == 0 and r.swap_usd == 0.0
    r, _ = one(cand(), overnight_tp(), HOLD, cfg=cfg)
    assert r.exit_type == "TP" and r.rollovers == 1
    # -500 points x 0.01 x $1/pt x 1 lot, Wednesday = swap_rollover3days 3 -> tripled
    assert r.swap_usd == pytest.approx(15.0)
    assert r.pnl_usd == pytest.approx((40300 - 40005) - 15.0)


def test_swap_triple_day_and_stress(cfg):
    specs = dict(SPECS, US30=spec(swap_rollover3days=4))
    res = run_backtest([("RS-MOM-01", cand())], {"US30": overnight_tp()}, cfg, HOLD, specs)
    assert res.trades.iloc[0].swap_usd == pytest.approx(5.0)
    r, _ = one(cand(), overnight_tp(), replace(HOLD, swap_mult=2.0), cfg=cfg)
    assert r.swap_usd == pytest.approx(30.0)                            # stress doubles


def test_positive_swap_is_credited_as_zero(cfg):
    b = bars({NEXT: (40000.0, 40010.0, 39600.0, 39700.0)})              # short TP next day
    r, _ = one(cand(side="SELL"), b, HOLD, cfg=cfg)
    assert r.exit_type == "TP" and r.rollovers == 1 and r.swap_usd == 0.0


def test_weekend_is_one_rollover_on_the_triple_day(cfg):
    fri = pd.Timestamp("2024-07-12 14:00", tz="UTC")
    mon = pd.Timestamp("2024-07-15 14:00", tz="UTC")
    b = concat_bars(flat_bars(fri - M15, pd.Timestamp("2024-07-12 20:30", tz="UTC")),
                    flat_bars(pd.Timestamp("2024-07-14 22:00", tz="UTC"), mon + M15,
                              overrides={mon: (40000.0, 40400.0, 39990.0, 40300.0)}))
    specs = dict(SPECS, US30=spec(swap_rollover3days=5))                # Friday
    res = run_backtest([("RS-MOM-01", cand(t=fri))], {"US30": b}, cfg, HOLD, specs)
    r = res.trades.iloc[0]
    assert r.exit_type == "TP" and r.rollovers == 1 and r.swap_usd == pytest.approx(15.0)


def interest_hold():
    # the bar closing at the 21:00 UTC rollover (opens 20:45) closes at 39800: mode 5 price
    return bars({at(405): (40000.0, 40005.0, 39795.0, 39800.0),
                 NEXT: (40000.0, 40400.0, 39990.0, 40300.0)})


@pytest.mark.parametrize("mode, raw, price", [
    (0, -500.0, None),             # disabled
    (4, -5.0, None),               # USD per lot
    (5, -3.6, 39800.0),            # annual %, current price = bid close at the rollover
    (6, -3.6, 40005.0),            # annual %, open price = fill price
])
def test_swap_modes_triple_day_and_stress(cfg, mode, raw, price):
    def one_run(triple_day, mult):
        specs = dict(SPECS, US30=spec(swap_mode=mode, swap_long=raw,
                                      swap_rollover3days=triple_day))
        res = run_backtest([("RS-MOM-01", cand())], {"US30": interest_hold()}, cfg,
                           replace(HOLD, swap_mult=mult), specs)
        r = res.trades.iloc[0]
        assert r.exit_type == "TP" and r.rollovers == 1 and r.fill_price == 40005.0
        return r.swap_usd

    if mode == 0:
        expected = 0.0
    elif mode == 4:
        expected = 5.0                                               # $5 x 1 lot
    else:
        expected = 3.6 / 100 * price * 1.0 / 360                     # 360-day year, $1/pt
    assert one_run(4, 1.0) == pytest.approx(expected)                # Thursday: single
    assert one_run(3, 1.0) == pytest.approx(3 * expected)            # Wednesday: triple
    assert one_run(3, 2.0) == pytest.approx(6 * expected)            # stress doubles


def test_unsupported_swap_mode_raises(cfg):
    specs = dict(SPECS, US30=spec(swap_mode=2))
    with pytest.raises(ValueError, match="swap_mode"):
        run_backtest([("RS-MOM-01", cand())], {"US30": overnight_tp()}, cfg, HOLD, specs)


# --- swap mode 3, money in the margin currency (COMMON A4.1) ------------------------------
@pytest.mark.parametrize("key, factor", [("US30", 1.0), ("GER40", 1.12)])
def test_swap_mode_3_margin_currency_triple_day_and_stress(cfg, key, factor):
    assert cfg["costs"]["margin_ccy_usd"][key] == factor                # frozen config value

    def one_run(triple_day, mult):
        specs = dict(SPECS, **{key: spec(key, swap_mode=3, swap_long=-4.0,
                                         swap_rollover3days=triple_day)})
        res = run_backtest([("RS-MOM-01", cand(key))], {key: overnight_tp()}, cfg,
                           replace(HOLD, swap_mult=mult), specs)
        r = res.trades.iloc[0]
        assert r.exit_type == "TP" and r.rollovers == 1
        return r.swap_usd, r.lots, r

    single, lots, r = one_run(4, 1.0)                                    # Thursday: single
    assert single == pytest.approx(4.0 * factor * lots)                  # raw x factor x lots
    assert r.pnl_usd == pytest.approx(r.gross_pnl_usd - r.commission_usd - single)
    assert one_run(3, 1.0)[0] == pytest.approx(3 * single)               # Wednesday: triple
    assert one_run(3, 2.0)[0] == pytest.approx(6 * single)               # stress doubles


def test_swap_mode_3_positive_swap_is_credited_as_zero(cfg):
    b = bars({NEXT: (40000.0, 40010.0, 39600.0, 39700.0)})              # short TP next day
    specs = dict(SPECS, US30=spec(swap_mode=3, swap_short=2.5))
    res = run_backtest([("RS-MOM-01", cand(side="SELL"))], {"US30": b}, cfg, HOLD, specs)
    r = res.trades.iloc[0]
    assert r.exit_type == "TP" and r.rollovers == 1 and r.swap_usd == 0.0


def test_swap_mode_3_without_a_config_factor_raises(cfg):
    del cfg["costs"]["margin_ccy_usd"]["GER40"]
    specs = dict(SPECS, GER40=spec("GER40", swap_mode=3, swap_long=-4.0))
    with pytest.raises(ValueError, match="margin_ccy_usd"):
        run_backtest([("RS-MOM-01", cand("GER40"))], {"GER40": overnight_tp()}, cfg, HOLD,
                     specs)
    # an intraday position never reaches the swap and needs no factor
    res = run_backtest([("RS-MOM-01", cand("GER40"))], {"GER40": overnight_tp()}, cfg, BASE,
                       specs)
    assert res.trades.iloc[0].rollovers == 0


def test_swap_value_per_lot_mode_3_unit():
    s3 = spec(swap_mode=3, swap_long=-4.0, swap_short=1.5)
    assert costs.swap_value_per_lot(s3, "BUY", 1.0, 40000.0, 1.12) == pytest.approx(-4.48)
    assert costs.swap_value_per_lot(s3, "SELL", 1.0, 40000.0, 1.12) == pytest.approx(1.68)
    # the factor is used as given: neither USD/pt nor the price enter mode 3
    assert costs.swap_value_per_lot(s3, "BUY", 7.0, 1.0, 1.0) == pytest.approx(-4.0)
    for bad in (None, 0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="margin_ccy_usd"):
            costs.swap_value_per_lot(s3, "BUY", 1.0, 40000.0, bad)
    with pytest.raises(ValueError, match="swap_mode 2"):
        costs.swap_value_per_lot(spec(swap_mode=2), "BUY", 1.0, 40000.0, 1.0)
    wed = pd.Timestamp("2024-07-10").date()                              # MT5 day 3
    hold2 = replace(HOLD, swap_mult=2.0)
    assert costs.swap_charge(s3, "BUY", 0.5, 1.0, 40000.0, wed, hold2, 1.12) == \
        pytest.approx(4.0 * 1.12 * 0.5 * 2.0 * 3.0)
    assert costs.swap_charge(s3, "SELL", 0.5, 1.0, 40000.0, wed, hold2, 1.12) == 0.0


def test_margin_ccy_usd_lookup(cfg):
    assert costs.margin_ccy_usd(cfg, "US30") == 1.0
    assert costs.margin_ccy_usd(cfg, "NAS100") == 1.0
    assert costs.margin_ccy_usd(cfg, "GER40") == pytest.approx(1.12)
    assert costs.margin_ccy_usd(cfg, "US500") is None
    assert costs.margin_ccy_usd({"costs": {}}, "US30") is None


# --- 8.7 sizing --------------------------------------------------------------------------
def test_sizing_on_sizing_entry_with_floor_and_cap(cfg):
    r, _ = one(cand(), cfg=cfg)
    # MARKET_BAND sizing entry = CHASE LIMIT 40080: 500 / 480 = 1.04 -> 1.0 lot
    assert r.sizing_entry == 40080.0 and r.lots == 1.0
    assert r.planned_risk_usd == pytest.approx(480.0)
    r, _ = one(cand(d=50.0, tp=100.0), cfg=cfg)
    assert r.lots == 3.0                                                # cap from config
    assert r.planned_risk_usd == pytest.approx(3 * 60.0)


def test_ger40_uses_the_frozen_demo_usd_per_point(cfg):
    tg = pd.Timestamp("2024-07-10 08:00", tz="UTC")                    # 10:00 Berlin
    b = flat_bars(tg - M15, tg + pd.Timedelta(hours=8))
    res = run_backtest([("RS-MOM-01", cand("GER40", t=tg))], {"GER40": b}, cfg, BASE, SPECS)
    r = res.trades.iloc[0]
    assert r.lots == 0.9                                                # 500 / (480 x 1.12)
    assert r.planned_risk_usd == pytest.approx(0.9 * 480 * 1.12)
    assert r.time_exit_utc == pd.Timestamp("2024-07-10 15:15", tz="UTC")
    assert r.gross_pnl_usd == pytest.approx(0.9 * (39998 - 40005) * 1.12)


def test_competition_contract_scales_dollars_not_lots(cfg):
    comp = {"US30": 2.0, "NAS100": 2.0, "GER40": 2.0}
    r, res = one(cand(), cfg=cfg, contract=comp, contract_name="competition")
    assert res.contract == "competition" and r.lots == 1.0
    assert r.planned_risk_usd == pytest.approx(960.0)
    assert r.gross_pnl_usd == pytest.approx(2 * (39998 - 40005))


def test_contract_tables_and_both_contracts(cfg):
    assert contract_table(cfg, "demo") == {"US30": 1.0, "NAS100": 1.0, "GER40": 1.12}
    assert contract_table(cfg, "competition") is None
    out = run_both_contracts([("RS-MOM-01", cand())], {"US30": bars()}, cfg, BASE, SPECS)
    assert out["competition"] is None and out["demo"].contract == "demo"
    cfg["rules"]["competition_usd_per_point"] = {"US30": 0.5, "NAS100": 0.5, "GER40": 0.5}
    out = run_both_contracts([("RS-MOM-01", cand())], {"US30": bars()}, cfg, BASE, SPECS)
    assert out["competition"].trades.iloc[0].planned_risk_usd == pytest.approx(240.0)


# --- 8.7 risk checks and suppression -------------------------------------------------------
def test_size_below_minimum_is_suppressed(cfg):
    r, _ = one(cand(d=6000.0), cfg=cfg)
    assert r.status == "suppressed" and "below the minimum" in r.reason
    assert pd.isna(r.pnl_usd)


def test_personal_daily_stop_suppresses(cfg):
    cfg["risk"]["personal_daily_stop_pct"] = 0.3                        # $300 < 480 x 1.1
    r, _ = one(cand(), cfg=cfg)
    assert r.status == "suppressed" and "personal daily stop" in r.reason


def test_restricted_window_and_session_on_t_suppress(cfg):
    cfg["rules"]["restricted_windows_utc"] = [
        {"start": "2024-07-10T13:30:00Z", "end": "2024-07-10T14:30:00Z", "reason": "CPI"}]
    r, _ = one(cand(), cfg=cfg)
    assert r.status == "suppressed" and "CPI" in r.reason
    cfg["rules"]["restricted_windows_utc"] = []
    cfg["sessions_utc"] = {"US30": [["14:30", "20:00"]]}
    r, _ = one(cand(), cfg=cfg)
    assert r.status == "suppressed" and "outside allowed trading hours" in r.reason
    cfg["sessions_utc"] = {"US30": [["14:00", "20:00"]]}                # start <= T
    assert one(cand(), cfg=cfg)[0].status == "filled"


def test_untradeable_index_and_invalid_levels_are_suppressed(cfg):
    r, _ = one(cand("US500"), cfg=cfg)
    assert r.status == "suppressed" and "not tradeable" in r.reason
    r, _ = one(cand(chase_limit=40090.0), cfg=cfg)
    assert r.status == "suppressed" and "invalid protocol levels" in r.reason


def test_max_open_positions_counts_simulated_open_positions(cfg):
    cfg["rules"]["max_open_positions"] = 1
    res = run([cand("US30"), cand("NAS100", t=at(60))],
              {"US30": bars(), "NAS100": bars()}, cfg=cfg)
    a, b = res.trades.iloc[0], res.trades.iloc[1]
    assert a.status == "filled"
    assert b.status == "suppressed" and "open positions" in b.reason


def test_owner_confirmed_is_not_a_backtest_rule(cfg):
    cfg["owner_confirmed"] = False
    assert one(cand(), cfg=cfg)[0].status == "filled"


def test_account_state_seen_by_the_risk_rules(cfg, monkeypatch):
    seen = []
    real = engine.check_ticket

    def spy(**kw):
        seen.append(kw)
        return real(**kw)

    monkeypatch.setattr(engine, "check_ticket", spy)
    us = bars({at(60): (40000.0, 40005.0, 39600.0, 39700.0)})          # day 1: SL
    nas = bars()
    res = run([cand("US30"), cand("US30", t=NEXT), cand("NAS100", t=at(60, NEXT))],
              {"US30": us, "NAS100": nas}, cfg=cfg)
    loss = 1.0 * (39598 - 40005)
    assert res.trades.iloc[0].pnl_usd == pytest.approx(loss)
    s1 = seen[1]["state"]                                               # US30 next day
    assert s1.balance == pytest.approx(100000 + loss)
    assert s1.start_of_day_balance == pytest.approx(100000 + loss)
    assert s1.trades_today == 0 and s1.trading_days_completed == 1
    assert not s1.open_positions and not s1.pending_orders
    s2 = seen[2]["state"]                                               # NAS100, US30 open
    assert s2.trades_today == 1
    assert [(p.symbol, p.lots, p.risk_usd) for p in s2.open_positions] == \
        [("US30", 1.0, pytest.approx(480.0))]
    # worst-intrabar MTM of the open long at T + 60: low 39995 vs fill 40005
    assert s2.equity == pytest.approx(100000 + loss - 10.0)
    assert seen[2]["now"] == at(60, NEXT) and seen[2]["signal_time"] == at(60, NEXT)


def test_pending_ticket_counts_as_pending_order_risk(cfg, monkeypatch):
    seen = []
    real = engine.check_ticket
    monkeypatch.setattr(engine, "check_ticket", lambda **kw: seen.append(kw) or real(**kw))
    run([cand("US30"), cand("NAS100", t=at(15))], {"US30": bars(), "NAS100": bars()}, cfg=cfg)
    st = seen[1]["state"]                                   # US30 placed at T + 30: pending
    assert not st.open_positions
    assert [(o.symbol, o.risk_usd) for o in st.pending_orders] == [("US30", 480.0)]


# --- 8.8 concurrency ------------------------------------------------------------------------
def test_blocked_while_open_and_freed_at_the_exit_instant(cfg):
    us = bars({at(60): (40000.0, 40005.0, 39600.0, 39700.0)})          # SL, exit at T + 75
    res = run([cand(), cand(t=at(60)), cand(t=at(75))], {"US30": us}, cfg=cfg)
    st = list(res.trades.status)
    assert st == ["filled", "blocked", "filled"]
    assert res.trades.iloc[1].reason.startswith("blocked_open_position")


def test_pending_limit_blocks_until_cancelled(cfg):
    lim = cand(order_type=MARKET_OR_LIMIT, expires_utc=at(60))
    b = bars(base=40010.0)                                              # never fills E
    res = run([lim, cand(t=at(45), entry=40010.0), cand(t=at(60), entry=40010.0)],
              {"US30": b}, cfg=cfg)
    assert list(res.trades.status) == ["cancelled", "blocked", "filled"]


def test_skipped_ticket_blocks_until_just_after_placement(cfg):
    # The skip is decided by the open of the bar opening at P = T + 30, which has not closed by
    # T + 30: a candidate at T + 30 still sees the ticket pending (COMMON 2); one at T + 45 not.
    b = bars({at(30): (40200.0, 40205.0, 40195.0, 40200.0)})          # beyond chase: skip
    res = run([cand(), cand(t=at(15), entry=40200.0), cand(t=at(30), entry=40200.0),
               cand(t=at(45), entry=40200.0)], {"US30": b}, cfg=cfg)
    assert list(res.trades.status) == ["skipped", "blocked", "blocked", "filled"]


def test_exit_at_the_open_of_the_bar_at_t_still_blocks_t(cfg):
    # the time exit at the 19:45 bar open is not known at T = 19:45; at 20:00 the slot is free
    ex = pd.Timestamp("2024-07-10 19:45", tz="UTC")
    res = run([cand(), cand(t=ex), cand(t=ex + M15)], {"US30": bars()}, cfg=cfg)
    t = res.trades
    assert t.iloc[0].exit_type == "time_exit" and t.iloc[0].exit_time == ex
    assert list(t.status[:2]) == ["filled", "blocked"]
    assert "open on US30" in t.iloc[1].reason
    assert t.iloc[2].status != "blocked"


def test_other_indices_are_independent(cfg):
    res = run([cand("US30"), cand("NAS100")], {"US30": bars(), "NAS100": bars()}, cfg=cfg)
    assert list(res.trades.status) == ["filled", "filled"]


def test_combined_book_same_t_priority(cfg):
    c = cand()
    res = run([("RS-MOM-01", c), ("SES-IBX-01", c), ("MR-TWAP-01", c)], {"US30": bars()},
              cfg=cfg, mode="combined")
    t = res.trades
    assert list(t.strategy_id) == ["SES-IBX-01", "RS-MOM-01", "MR-TWAP-01"]
    assert list(t.status) == ["filled", "blocked", "blocked"]
    assert "SES-IBX-01" in t.iloc[1].reason


def test_single_mode_rejects_mixed_strategies_and_combined_unknown_ids(cfg):
    with pytest.raises(ValueError, match="combined"):
        run([("RS-MOM-01", cand()), ("SES-IBX-01", cand())], {"US30": bars()}, cfg=cfg)
    with pytest.raises(ValueError, match="priority"):
        run([("XYZ", cand())], {"US30": bars()}, cfg=cfg, mode="combined")


def test_legacy_candidate_is_rejected(cfg):
    with pytest.raises(ValueError, match="order_type"):
        one(cand(order_type=None), cfg=cfg)


# --- 8.7 compounding, MTM and the daily series --------------------------------------------
def test_balance_compounds_into_the_next_sizing(cfg):
    us = bars({at(60): (40000.0, 40600.0, 39995.0, 40500.0)})          # TP for trade 1
    c1 = cand(d=200.0, tp=500.0)
    c2 = cand(t=at(120), d=200.0, tp=500.0)
    res = run([c1, c2], {"US30": us}, cfg=cfg)
    a, b = res.trades.iloc[0], res.trades.iloc[1]
    assert a.lots == 2.0 and a.pnl_usd == pytest.approx(2 * (40500 - 40005))
    assert b.sizing_balance == pytest.approx(100990.0)
    assert b.lots == 2.1                                                # 504.95 / 240
    assert res.equity.equity.iloc[-1] == pytest.approx(100000 + a.pnl_usd + b.pnl_usd)


def test_mtm_at_worst_intrabar_price(cfg):
    _, res = one(cand(), cfg=cfg)
    eq = res.equity.set_index("time_utc")
    assert eq.loc[at(30), "equity"] == 100000.0                         # before the fill bar
    assert eq.loc[at(45), "equity"] == pytest.approx(100000 + (39995 - 40005))
    _, res = one(cand(side="SELL"), cfg=cfg)
    eq = res.equity.set_index("time_utc")
    assert eq.loc[at(45), "equity"] == pytest.approx(100000 + (39998 - (40005 + 3)))


def test_daily_end_of_day_series_uses_server_midnight(cfg):
    r, res = one(cand(), overnight_tp(), HOLD, cfg=cfg)
    d = res.daily.set_index("date")
    day1, day2 = pd.Timestamp("2024-07-10").date(), pd.Timestamp("2024-07-11").date()
    # 2024-07-10 ends at 21:00 UTC (00:00 Athens): open long marked at low 39995, no swap yet
    assert d.loc[day1, "equity"] == pytest.approx(100000 + (39995 - 40005))
    assert d.loc[day1, "balance"] == 100000.0
    assert d.loc[day2, "equity"] == pytest.approx(100000 + r.pnl_usd)
    eq = res.equity.set_index("time_utc")
    after = pd.Timestamp("2024-07-10 21:15", tz="UTC")
    assert eq.loc[after, "equity"] == pytest.approx(100000 + (39995 - 40005) - 15.0)


# --- 8.9 accounting -------------------------------------------------------------------------
def test_pnl_and_r_accounting(cfg):
    cfg["rules"]["commission_per_lot_side"] = 1.5
    us = bars({at(60): (40000.0, 40005.0, 39600.0, 39700.0)})
    res = run([cand(), cand("NAS100")], {"US30": us, "NAS100": bars()}, cfg=cfg)
    for _, r in res.filled.iterrows():
        gross = r.lots * (r.exit_price - r.fill_price) * (1 if r.side == "BUY" else -1)
        assert r.pnl_usd == pytest.approx(gross - r.commission_usd - r.swap_usd)
        assert r.r_multiple == pytest.approx(r.pnl_usd / r.planned_risk_usd)
    tabs = report_tables(res.trades)
    assert set(tabs) == {"all", "index", "side", "exit_type"}
    assert set(tabs["exit_type"].index) == {"SL", "time_exit"}
    assert tabs["all"]["pnl_usd"].iloc[0] == pytest.approx(res.filled.pnl_usd.sum())
    assert summarize(res.trades, "key").loc["US30", "n"] == 1
    assert list(res.trades.columns) == TRADE_COLUMNS


# --- 8.10 determinism and no look-ahead -----------------------------------------------------
def test_deterministic_and_independent_of_input_order(cfg):
    us = bars({at(60): (40000.0, 40600.0, 39995.0, 40500.0)})
    cands = [StrategyCandidate("RS-MOM-01", cand(d=200.0, tp=500.0)),
             StrategyCandidate("RS-MOM-01", cand("NAS100", t=at(30))),
             StrategyCandidate("RS-MOM-01", cand(t=at(120), d=200.0, tp=500.0))]
    m = {"US30": us, "NAS100": bars()}
    a = run_backtest(cands, m, cfg, SCENARIOS["STRESS-d2"], SPECS)
    b = run_backtest(cands, m, cfg, SCENARIOS["STRESS-d2"], SPECS)
    c = run_backtest(list(reversed(cands)), m, cfg, SCENARIOS["STRESS-d2"], SPECS)
    for x in (b, c):
        pd.testing.assert_frame_equal(a.trades, x.trades)
        pd.testing.assert_frame_equal(a.equity, x.equity)
        pd.testing.assert_frame_equal(a.daily, x.daily)


def _spy_states(monkeypatch):
    seen = []
    real = engine.check_ticket
    monkeypatch.setattr(engine, "check_ticket", lambda **kw: seen.append(kw) or real(**kw))
    return seen


def test_market_fill_or_skip_at_the_open_of_t_is_not_seen_at_t(cfg, monkeypatch):
    # US30 ticket placed at P = T + 30: filled or skipped depending ONLY on the bar opening at
    # T + 30. A NAS100 candidate at T + 30 must see the same account either way.
    cfg["rules"]["max_open_positions"] = 1
    seen = _spy_states(monkeypatch)
    out, us_status = [], []
    for p_bar in ((40000.0, 40005.0, 39995.0, 40000.0),               # fill
                  (40200.0, 40205.0, 40195.0, 40200.0)):              # beyond chase: skip
        seen.clear()
        res = run([cand("US30"), cand("NAS100", t=at(30))],
                  {"US30": bars({at(30): p_bar}), "NAS100": bars()}, cfg=cfg)
        nas = res.trades.iloc[1]
        us_status.append(res.trades.iloc[0].status)
        out.append((seen[1]["state"], nas.sizing_balance, nas.status, nas.lots))
    assert us_status == ["filled", "skipped"]
    assert out[0] == out[1]
    st = out[0][0]
    assert not st.open_positions and [o.symbol for o in st.pending_orders] == ["US30"]
    assert st.trades_today == 0 and out[0][2] == "filled"


def test_exit_at_the_open_of_t_on_another_index_is_not_seen_at_t(cfg, monkeypatch):
    # US30 time-exits at the open of the 19:45 bar; its price is the only difference. A NAS100
    # candidate at T = 19:45 must see the same balance, equity and positions either way, and
    # the MTM mark at 19:45 must not use the 19:45 bar either.
    ex = pd.Timestamp("2024-07-10 19:45", tz="UTC")
    seen = _spy_states(monkeypatch)
    out = []
    for o in (40000.0, 40250.0):
        seen.clear()
        us = bars({ex: (o, o + 5.0, o - 5.0, o)})
        res = run([cand("US30"), cand("NAS100", t=ex)], {"US30": us, "NAS100": bars()},
                  cfg=cfg)
        assert res.trades.iloc[0].exit_time == ex
        assert res.trades.iloc[0].exit_price == pytest.approx(o - 2)
        eq = res.equity.set_index("time_utc")
        out.append((seen[1]["state"], res.trades.iloc[1].sizing_balance,
                    res.trades.iloc[1].status, eq.loc[ex, "balance"], eq.loc[ex, "equity"]))
    assert out[0] == out[1]
    st = out[0][0]
    assert st.balance == 100000.0 and [p.symbol for p in st.open_positions] == ["US30"]
    assert st.equity == pytest.approx(100000 + (39995 - 40005))      # mark of the 19:30 bar
    assert out[0][3] == 100000.0 and out[0][4] == pytest.approx(100000 + (39995 - 40005))


def test_decision_at_t_ignores_bars_after_t(cfg, monkeypatch):
    seen = []
    real = engine.check_ticket
    monkeypatch.setattr(engine, "check_ticket", lambda **kw: seen.append(kw) or real(**kw))
    t2 = at(60)
    base = bars()
    later = bars({t2 + M15: (40000.0, 40005.0, 39700.0, 39800.0)})     # deep dip after T2
    out = []
    for us in (base, later):
        seen.clear()
        res = run([cand(), cand("NAS100", t=t2)], {"US30": us, "NAS100": bars()}, cfg=cfg)
        out.append((seen[1]["state"], res.trades.iloc[1].lots))
    assert out[0] == out[1]
