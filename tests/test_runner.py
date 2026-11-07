"""backtest.folds and backtest.runner on SYNTHETIC data only: the COMMON 10.2 windows, fold of a
trade = fold of its signal time T, the daily-series calendar, the run-B embargo, selection and
ties, trial dedup and N_raw through the runner, the 10.4 trade-count checks, the market cache
and the MR-TWAP-01 refusal. No real bar is read; every log goes to tmp_path."""
from __future__ import annotations

import dataclasses
import gzip
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest import folds as F
from backtest import runner as R
from backtest import trials as tl
from backtest.engine import StrategyCandidate
from backtest.folds import Fold, WalkForwardFold
from backtest.scenarios import SCENARIOS, Scenario
from backtest.synthetic import flat_bars
from core.config import load_config
from data.mt5_adapter import SymbolSpec
from strategies.base import MARKET_BAND, Candidate
from strategies.rs_mom_01 import PARAMS as RS_PARAMS
from tests.strategy_testkit import SyntheticMarket, make_market, proxy_regime_tables

ROOT = Path(__file__).resolve().parent.parent
UTC = "UTC"
HOLD = Scenario(name="hold", missed_exit=True)        # base costs, no time exit


def ts(s) -> pd.Timestamp:
    return pd.Timestamp(s, tz=UTC)


def spec(name: str) -> SymbolSpec:
    return SymbolSpec(name, 2, 0.01, 0.01, 0.01, 0.1, 250.0, 0.1, "USD", swap_long=-5.0,
                      swap_short=-5.0, swap_mode=1, swap_rollover3days=3, trade_stops_level=100)


SPECS = {k: spec(k) for k in ("US30", "NAS100", "GER40")}


@pytest.fixture(scope="module")
def mcfg():
    c = load_config(ROOT / "config.yaml")
    c["owner_confirmed"] = True
    c["sessions_utc"] = {}
    c["rules"]["competition_end_utc"] = None
    return c


def proxy_factory(calls=None):
    def build(cfg, m15, h1):
        if calls is not None:
            calls.append(1)
        mk = SyntheticMarket(dict(m15), dict(h1), ts("2024-01-08"), ts("2024-04-13"))
        return proxy_regime_tables(mk, cfg)
    return build


@pytest.fixture(scope="module")
def synth():
    return make_market()


@pytest.fixture(scope="module")
def market(mcfg, synth):
    return R.market_from_frames(mcfg, synth.m15, synth.h1, SPECS,
                                regime_factory=proxy_factory(), regime_tag="proxy")


# Short windows inside the synthetic market (2024-01-08..2024-04-12; features from ~02-05).
W = (Fold(1, date(2024, 2, 5), date(2024, 2, 29)), Fold(2, date(2024, 3, 1), date(2024, 3, 31)),
     Fold(3, date(2024, 4, 1), date(2024, 4, 12)))
# Two short windows for the protocol test (29 grid cells are scanned there).
WS = (Fold(1, date(2024, 3, 4), date(2024, 3, 15)), Fold(2, date(2024, 3, 18), date(2024, 3, 28)))
SEC = Fold(0, date(2024, 3, 4), date(2024, 3, 28))
WF = (WalkForwardFold(1, Fold(1, date(2024, 2, 5), date(2024, 3, 8)),
                      Fold(1, date(2024, 3, 11), date(2024, 4, 12))),)


def flat_market(mcfg, start, end, overrides=None) -> R.MarketData:
    """MarketData with flat US30 bars only (for hand-made candidates; no features)."""
    b = flat_bars(ts(start), ts(end), overrides=overrides)
    return R.MarketData(mcfg, ("US30",), {"US30": b}, {"US30": b.iloc[0:0]}, None, {}, SPECS,
                        "flat", "test")


def cand(T, entry=40000.0, d=400.0, tp=300.0, side="BUY", key="US30"):
    sgn = 1.0 if side == "BUY" else -1.0
    return StrategyCandidate("RS-MOM-01", Candidate(
        key=key, side=side, entry=entry, stop=entry - sgn * d, take_profit=entry + sgn * tp,
        atr=100.0, reason="t", signal_time=ts(T), order_type=MARKET_BAND))


# ---------------------------------------------------------------------------------------------
# Windows (COMMON 10.2)
# ---------------------------------------------------------------------------------------------

def test_test_windows_exactly_as_common_10_2():
    want = [("2024-02-01", "2024-04-30"), ("2024-05-01", "2024-07-31"),
            ("2024-08-01", "2024-10-31"), ("2024-11-01", "2025-01-31"),
            ("2025-02-01", "2025-04-30"), ("2025-05-01", "2025-07-31"),
            ("2025-08-01", "2025-10-31"), ("2025-11-01", "2026-01-31"),
            ("2026-02-01", "2026-04-30")]
    assert [(f.first.isoformat(), f.last.isoformat()) for f in F.TEST_FOLDS] == want
    assert [f.n for f in F.TEST_FOLDS] == list(range(1, 10))
    assert F.TEST_FOLDS[0].start == ts("2024-02-01") and F.TEST_FOLDS[0].end == ts("2024-05-01")
    assert F.is_contiguous(F.TEST_FOLDS)
    assert F.window_span(F.TEST_FOLDS) == (ts("2024-02-01"), ts("2026-05-01"))
    assert F.window_label(F.TEST_FOLDS) == "2024-02-01..2026-04-30 (folds 1-9)"
    assert (F.SECONDARY_SAMPLE.first, F.SECONDARY_SAMPLE.last) == (date(2023, 3, 1),
                                                                   date(2026, 4, 30))


def test_train_windows_are_the_12_months_before_each_test_window():
    assert len(F.WALK_FORWARD) == 9
    for wf in F.WALK_FORWARD:
        assert wf.test == F.TEST_FOLDS[wf.n - 1] and wf.train.n == wf.n
        assert wf.train.end == wf.test.start                 # no gap, no overlap
        assert pd.Timestamp(wf.train.first) == pd.Timestamp(wf.test.first) - pd.DateOffset(
            months=12)
    assert F.WALK_FORWARD[0].train.label == "2023-02-01..2024-01-31"
    assert F.WALK_FORWARD[3].train.label == "2023-11-01..2024-10-31"
    assert F.WALK_FORWARD[8].train.label == "2025-02-01..2026-01-31"


def test_assign_folds_boundaries():
    t = [ts("2024-01-31 23:59:59"), ts("2024-02-01"), ts("2024-04-30 23:59:59.999"),
         ts("2024-05-01"), ts("2026-04-30 23:45"), ts("2026-05-01"), pd.NaT]
    assert F.assign_folds(t).tolist() == [-1, 1, 1, 2, 9, -1, -1]
    assert F.assign_folds([ts("2023-03-01")], F.SECONDARY_SAMPLE).tolist() == [0]
    assert F.in_windows(t[:6]).tolist() == [False, True, True, True, True, False]
    with pytest.raises(ValueError, match="overlap"):
        F.assign_folds(t, [F.TEST_FOLDS[0], F.SECONDARY_SAMPLE])


def test_trading_calendar_is_union_of_exchange_trading_days(mcfg):
    cal = F.trading_calendar(mcfg, Fold(1, date(2024, 3, 25), date(2024, 4, 5)))
    # Good Friday 2024-03-29: NYSE and Xetra closed; Easter Monday 2024-04-01: Xetra closed,
    # NYSE open -> in the union. Weekends never.
    assert [d.date().isoformat() for d in cal] == [
        "2024-03-25", "2024-03-26", "2024-03-27", "2024-03-28", "2024-04-01", "2024-04-02",
        "2024-04-03", "2024-04-04", "2024-04-05"]
    # MLK day 2024-01-15: NYSE holiday, Xetra open -> in; with US keys only -> out.
    one = Fold(1, date(2024, 1, 15), date(2024, 1, 15))
    assert len(F.trading_calendar(mcfg, one)) == 1
    assert len(F.trading_calendar(mcfg, one, ["US30", "NAS100"])) == 0


def test_trading_calendar_keeps_early_close_days(mcfg):
    """QA minor: the exchanges trade on early-close dates, so they are trading days of the
    daily series (COMMON 10.2), although not 'valid trading dates' for tickets (1.6)."""
    cal = F.trading_calendar(mcfg, Fold(8, date(2025, 12, 22), date(2025, 12, 31)))
    assert [d.date().isoformat() for d in cal] == [
        "2025-12-22", "2025-12-23", "2025-12-24", "2025-12-26", "2025-12-29", "2025-12-30",
        "2025-12-31"]
    us = F.trading_calendar(mcfg, Fold(8, date(2025, 11, 27), date(2025, 11, 28)),
                            ["US30", "NAS100"])
    assert [d.date().isoformat() for d in us] == ["2025-11-28"]      # Thanksgiving out


# ---------------------------------------------------------------------------------------------
# Trades, daily series, embargo
# ---------------------------------------------------------------------------------------------

def test_trade_belongs_to_the_fold_of_its_signal_time(mcfg):
    # T on the last day of fold 1, held (no time exit) to a TP on the first day of fold 2.
    mk = flat_market(mcfg, "2024-04-29 00:00", "2024-05-03 20:00",
                     {ts("2024-05-01 14:00"): (40000.0, 40400.0, 39990.0, 40300.0)})
    run = R.run_candidates(mk, [cand("2024-04-30 18:00")], HOLD, F.TEST_FOLDS[:2])
    (row,) = run.filled.itertuples()
    assert row.status == "filled" and row.exit_type == "TP"
    assert F.assign_folds([row.exit_time]).tolist() == [2]
    assert row.fold == 1
    m = run.metrics()
    assert m["folds"]["1"]["n"] == 1 and m["folds"]["2"]["n"] == 0
    assert m["n_trades"] == 1 and m["total_usd"] == pytest.approx(row.pnl_usd)


def test_daily_series_calendar_zero_days_and_total(mcfg):
    win = Fold(1, date(2024, 3, 25), date(2024, 4, 5))
    mk = flat_market(mcfg, "2024-03-22 00:00", "2024-04-05 22:00",
                     {ts("2024-04-02 16:00"): (40000.0, 40400.0, 39990.0, 40300.0)})
    run = R.run_candidates(mk, [cand("2024-04-02 14:00")], SCENARIOS["BASE"], win)
    d = run.daily
    assert list(d["date"]) == list(F.trading_calendar(mcfg, win))
    assert len(d) == 9 and list(d.columns) == ["date", "equity", "pnl_usd", "ret"]
    trade = run.filled["pnl_usd"].sum()
    assert trade > 0
    by_day = dict(zip(d["date"].dt.date.astype(str), d["pnl_usd"]))
    assert by_day["2024-04-02"] == pytest.approx(trade)
    assert all(v == 0 for k, v in by_day.items() if k != "2024-04-02")
    assert d["pnl_usd"].sum() == pytest.approx(trade)
    assert run.metrics()["daily_total_usd"] == pytest.approx(trade)


def test_daily_series_rolls_non_calendar_days_forward():
    eod = pd.DataFrame({"date": [date(2024, 3, 27), date(2024, 3, 28), date(2024, 3, 29),
                                 date(2024, 4, 1)],
                        "balance": 0.0, "equity": [100010.0, 100000.0, 100050.0, 100050.0]})
    cal = pd.DatetimeIndex(["2024-03-28", "2024-04-01", "2024-04-02"])
    d = R.daily_series(eod, cal, 100000.0)
    # first day vs the end of 03-27; Good Friday's change lands on 04-01; 04-02 carries forward
    assert d["pnl_usd"].tolist() == [-10.0, 50.0, 0.0]
    assert d["equity"].tolist() == [100000.0, 100050.0, 100050.0]
    assert d["ret"].iloc[0] == pytest.approx(-10.0 / 100010.0)
    empty = R.daily_series(eod.iloc[0:0], cal, 100000.0)
    assert empty["pnl_usd"].tolist() == [0.0, 0.0, 0.0]


def test_embargo_drops_train_trades_exiting_after_the_window(mcfg):
    win = Fold(1, date(2024, 7, 1), date(2024, 7, 10))          # ends 2024-07-11 00:00 UTC
    ov = {ts("2024-07-08 16:00"): (40000.0, 40400.0, 39990.0, 40300.0),   # A: TP same day
          ts("2024-07-10 18:00"): (40000.0, 40200.0, 39990.0, 40000.0),   # C: TP (B's not)
          ts("2024-07-11 14:00"): (40000.0, 40400.0, 39990.0, 40300.0)}   # B: TP after the end
    mk = flat_market(mcfg, "2024-06-28 00:00", "2024-07-12 20:00", ov)
    A = cand("2024-07-08 14:00")
    B = cand("2024-07-10 14:00")
    C = cand("2024-07-10 16:00", tp=100.0)
    plain = R.run_candidates(mk, [A, B, C], HOLD, win)
    st = dict(zip(plain.trades["signal_time"].map(str), plain.trades["status"]))
    assert list(st.values()) == ["filled", "filled", "blocked"]        # C blocked by B
    emb = R.run_candidates(mk, [A, B, C], HOLD, win, embargo=True)
    kept = sorted(map(str, emb.filled["signal_time"]))
    assert kept == [str(ts("2024-07-08 14:00")), str(ts("2024-07-10 16:00"))]   # A and C
    assert list(map(str, emb.embargoed["signal_time"])) == [str(ts("2024-07-10 14:00"))]
    assert (pd.to_datetime(emb.filled["exit_time"], utc=True) <= win.end).all()
    assert emb.metrics()["n_embargoed"] == 1
    # the train daily series has no MTM of the dropped trade
    assert emb.daily["pnl_usd"].sum() == pytest.approx(emb.filled["pnl_usd"].sum())


# ---------------------------------------------------------------------------------------------
# Run B selection and ties
# ---------------------------------------------------------------------------------------------

def test_grid_distance():
    g = RS_PARAMS
    assert R.grid_distance(dict(g.defaults), g) == 0
    assert R.grid_distance({"theta": 0.8, "k": 1.0, "R": 1.5}, g) == 2
    assert R.grid_distance({"theta": 1.0, "k": 1.125, "R": 2.0}, g) == 3


def test_select_params_highest_sharpe_then_closest_to_default():
    g = RS_PARAMS
    pts = g.one_at_a_time()
    lab = [tl.params_text(p) for p in pts]

    def pick(sh):
        return lab[R.select_params(pts, sh, g)]
    base = [0.01] * len(pts)
    s = list(base)
    s[lab.index("theta=1.2,k=1.0,R=1.5")] = 0.3
    assert pick(s) == "theta=1.2,k=1.0,R=1.5"
    s[lab.index("theta=0.9,k=1.0,R=1.5")] = 0.3                     # tie: distance 1 < 2
    assert pick(s) == "theta=0.9,k=1.0,R=1.5"
    s[lab.index("theta=1.1,k=1.0,R=1.5")] = 0.3                     # distance 1 = 1: order
    assert pick(s) == "theta=0.9,k=1.0,R=1.5"
    s = list(base)
    s[lab.index("theta=1.0,k=1.0,R=1.25")] = 0.3
    s[lab.index("theta=0.8,k=1.0,R=1.5")] = 0.3 * (1 + 1e-14)      # within the tie tolerance
    assert pick(s) == "theta=1.0,k=1.0,R=1.25"
    assert pick(base) == "theta=1.0,k=1.0,R=1.5"                     # all equal: the default
    s = [np.nan] * len(pts)
    assert pick(s) == "theta=1.0,k=1.0,R=1.5"                        # all NaN: the default
    s[lab.index("theta=1.0,k=0.75,R=1.5")] = -0.5                    # finite beats NaN
    assert pick(s) == "theta=1.0,k=0.75,R=1.5"


def test_run_b_selects_on_train_and_applies_to_test(market):
    rb = R.run_b(market, "RS-MOM-01", folds=WF, log=False)
    (sel,) = rb.selections
    t = sel.table
    assert len(t) == 13 and t["chosen"].sum() == 1
    assert list(t["label"]) == [tl.params_text(p) for p in RS_PARAMS.one_at_a_time()]
    k = R.select_params(list(t["params"]), list(t["sharpe"]), RS_PARAMS)
    assert t.loc[t["chosen"], "order"].item() == k and sel.chosen == t["params"].iloc[k]
    # each train Sharpe is the embargoed train-window run of that grid point
    p = t["params"].iloc[k]
    tr = R.run_config("RS-MOM-01", p, "STRESS-d2", (WF[0].train,), market=market, embargo=True)
    assert tr.sharpe == pytest.approx(t["sharpe"].iloc[k], nan_ok=True)
    assert (pd.to_datetime(tr.filled["exit_time"], utc=True) <= WF[0].train.end).all()
    # out of sample = the test window scanned with the chosen parameters
    oos = rb.runs["STRESS-d2"]
    want = R.run_config("RS-MOM-01", p, "STRESS-d2", (WF[0].test,), market=market)
    assert oos.trades["signal_time"].tolist() == want.trades["signal_time"].tolist()
    assert oos.metrics()["total_usd"] == pytest.approx(want.metrics()["total_usd"])
    assert set(oos.trades["fold"]) == {1}


# ---------------------------------------------------------------------------------------------
# Runner + trial log (synthetic market, tmp log)
# ---------------------------------------------------------------------------------------------

def test_run_config_end_to_end(market, mcfg):
    for sid in ("RS-MOM-01", "SES-IBX-01"):
        run = R.run_config(sid, None, "STRESS-d2", W, market=market)
        assert len(run.filled) > 0 and len(run.selection_lines) > 0
        assert list(run.daily["date"]) == list(F.trading_calendar(mcfg, W))
        assert set(run.trades["fold"]) <= {1, 2, 3}
        assert (run.trades["fold"] == F.assign_folds(run.trades["signal_time"], W)).all()
        m = run.metrics()
        json.dumps(m)
        assert m["n_trades"] == len(run.filled) and m["contract"] == "demo"
        assert sum(v["n"] for v in m["folds"].values()) == m["n_trades"]


def test_protocols_log_one_row_per_configuration(market, tmp_path):
    p, runs = tmp_path / "log.csv", tmp_path / "runs"
    kw = dict(log_path=p, runs_dir=runs)
    rb = R.run_b(market, "RS-MOM-01", folds=WF, canonical_windows=WS, **kw)
    log = tl.read_log(p)
    assert list(log["stage"]) == ["run_b_train"] * 13 + ["run_b"]
    assert tl.n_raw(p) == 13                         # the run-B composite is not counted
    assert rb.trial.trial_id == "T0014"
    # run A of the default merges into the run_b_train row of the default: no new trial
    ra = R.run_a(market, "RS-MOM-01", windows=WS, secondary_window=SEC, **kw)
    assert ra.trial.trial_id == "T0001" and not ra.trial.created
    assert tl.n_raw(p) == 13 and len(tl.read_log(p)) == 14
    notes = tl.read_log(p)["notes_json"].iloc[0]
    assert notes["stages"] == ["run_b_train", "run_a"]
    assert set(notes["scenarios"]) == set(SCENARIOS)
    assert set(notes["extra"]["secondary"]) == set(SCENARIOS)
    assert notes["extra"]["contracts"] == ["demo"] and "null" in notes["extra"]["contract_note"]
    assert notes["extra"]["trade_counts"]["min_total"] == 150
    d = tl.trial_dir("T0001", runs)
    assert (d / "daily.parquet").exists() and (d / "selection.log.gz").exists()
    for s in SCENARIOS:
        assert (d / "scenarios" / s / "trades.parquet").exists()
        assert (d / "secondary" / s / "daily.parquet").exists()
    lines = gzip.decompress((d / "selection.log.gz").read_bytes()).decode().splitlines()
    assert lines and all(x.startswith("RS-MOM-01\t") for x in lines)
    assert tl.load_trades("T0001", runs)["fold"].isin([1, 2, -1]).all()
    # sensitivity: 29 cells, 13 already logged -> 16 new rows, N_raw = 29 (README budget)
    se = R.run_sensitivity(market, "RS-MOM-01", windows=WS, **kw)
    assert len(se.table) == 29 and tl.n_raw(p) == 29 and len(tl.read_log(p)) == 30
    assert set(se.table["status"]) <= {"positive", "not positive", "insufficient"}
    assert (se.table["status"] == "insufficient").all()       # synthetic: < 100 trades a cell
    assert se.criterion5["passed"] is False and se.criterion5["n_cells"] == 29
    assert se.heatmap().shape == (5, 5)
    # the trial matrix: one column per counted configuration, one calendar
    m = tl.trial_matrix(p, runs)
    assert m.shape == (len(F.trading_calendar(market.cfg, WS)), 29)
    # running again changes nothing (dedup, deterministic)
    before = p.read_bytes()
    R.run_sensitivity(market, "RS-MOM-01", windows=WS, **kw)
    assert p.read_bytes() == before
    # combined book: logged, never counted
    pr = R.run_portfolio(market, ["RS-MOM-01", "SES-IBX-01"], windows=WS, **kw)
    assert pr.strategy_ids == ("SES-IBX-01", "RS-MOM-01")
    assert tl.read_log(p)["stage"].iloc[-1] == "portfolio" and tl.n_raw(p) == 29
    t = pr.runs["STRESS-d2"].filled
    assert set(t["strategy_id"]) <= {"SES-IBX-01", "RS-MOM-01"}
    assert set(pr.competition) == {"SES-IBX-01", "RS-MOM-01"}
    for _, g in t.groupby("key"):                      # one position per index (COMMON 8.8)
        g = g.sort_values("fill_time")
        assert (pd.to_datetime(g["fill_time"].iloc[1:], utc=True).to_numpy() >=
                pd.to_datetime(g["exit_time"].iloc[:-1], utc=True).to_numpy()).all()


def test_rerun_after_bug_fix_through_the_runner(market, tmp_path):
    p, runs = tmp_path / "log.csv", tmp_path / "runs"
    R.evaluate_configuration(market, "RS-MOM-01", None, stage="sensitivity", windows=W,
                             log_path=p, runs_dir=runs)
    ev = R.evaluate_configuration(market, "RS-MOM-01", None, stage="run_a", windows=W,
                                  scenarios=("STRESS-d2", "BASE"), log_path=p, runs_dir=runs,
                                  rerun_note="test fix")
    assert ev.trial.created and ev.trial.trial_id == "T0002"
    log = tl.read_log(p)
    assert len(log) == 2 and log["notes_json"].iloc[1]["rerun_of"] == "T0001"
    assert tl.n_raw(p) == 1
    assert (tl.trial_dir("T0002", runs) / "scenarios" / "BASE" / "daily.parquet").exists()


def _clone(market, *, cfg=None, specs=None, source="test") -> R.MarketData:
    """The same bars, features, regime tables and scan cache under another cfg / specs /
    source (as market_from_frames would return for the same data)."""
    return R.MarketData(cfg or market.cfg, market.keys, market.m15, market.h1, market.features,
                        market.tables, specs or SPECS, market.data_hash, source, market.scans)


def test_run_b_rerun_note_reruns_every_grid_point(market, tmp_path):
    """QA major: a bug-fix or spec-snapshot rerun of run B alone logs a rerun row for every
    one-at-a-time grid point (earlier rows kept, N_raw unchanged), so the trial matrix reads
    the post-fix daily series."""
    p, runs = tmp_path / "log.csv", tmp_path / "runs"
    kw = dict(folds=WF, canonical_windows=WS, log_path=p, runs_dir=runs)
    R.run_b(market, "RS-MOM-01", **kw)
    assert len(tl.read_log(p)) == 14 and tl.n_raw(p) == 13
    # (a) a new SymbolSpec snapshot changes config_hash: without a note it is a conflict ...
    specs2 = dict(SPECS)
    specs2["US30"] = dataclasses.replace(SPECS["US30"], trade_stops_level=150)
    mk2 = _clone(market, specs=specs2)
    assert mk2.config_hash != market.config_hash
    with pytest.raises(tl.TrialConflict):
        R.run_b(mk2, "RS-MOM-01", **kw)
    assert len(tl.read_log(p)) == 14
    # ... with one, 13 rerun rows + the composite
    R.run_b(mk2, "RS-MOM-01", rerun_note="spec snapshot refreshed", **kw)
    log = tl.read_log(p)
    assert len(log) == 28 and tl.n_raw(p) == 13
    new = log.iloc[14:]
    assert list(new["stage"]) == ["run_b_train"] * 13 + ["run_b"]
    for (_, old), (_, nw) in zip(log.iloc[:14].iterrows(), new.iterrows()):
        assert nw["notes_json"]["rerun_of"] == old["trial_id"]
        assert nw["notes_json"]["note"] == "rerun after bug fix: spec snapshot refreshed"
        assert nw["notes_json"]["config_hash"] == mk2.config_hash
    # the trial matrix reads the rerun rows' files
    assert list(tl.trial_matrix(p, runs).columns) == list(new["trial_id"].iloc[:13])
    # running again in the same session adds nothing
    R.run_b(mk2, "RS-MOM-01", rerun_note="spec snapshot refreshed", **kw)
    assert len(tl.read_log(p)) == 28
    # (b) a code fix with the same hashes: again 13 + 1 rerun rows
    R.run_b(mk2, "RS-MOM-01", rerun_note="fixed X", **kw)
    log = tl.read_log(p)
    assert len(log) == 42 and tl.n_raw(p) == 13
    assert all(n["note"] == "rerun after bug fix: fixed X" for n in log["notes_json"].iloc[28:])


def test_first_log_under_a_rerun_note_is_not_a_phantom_rerun(market, tmp_path):
    """QA minor: a configuration first logged in a rerun session gets one plain row (no
    'note', no 'rerun_of'); later logs in the same session merge into it."""
    p, runs = tmp_path / "log.csv", tmp_path / "runs"
    kw = dict(windows=W, log_path=p, runs_dir=runs, rerun_note="fix Y")
    a = R.evaluate_configuration(market, "RS-MOM-01", None, stage="run_a", **kw)
    b = R.evaluate_configuration(market, "RS-MOM-01", None, stage="sensitivity", **kw)
    log = tl.read_log(p)
    assert len(log) == 1 and a.trial.created and not b.trial.created
    n = log["notes_json"].iloc[0]
    assert "note" not in n and "rerun_of" not in n
    assert n["rerun_session"] == "rerun after bug fix: fix Y"
    assert n["stages"] == ["run_a", "sensitivity"]
    assert R.ensure_logged(market, "RS-MOM-01", RS_PARAMS.defaults, stage="run_b_train",
                           windows=W, log_path=p, runs_dir=runs, rerun_note="fix Y") == "T0001"
    assert len(tl.read_log(p)) == 1
    # a later, different rerun session does append a real rerun row
    R.evaluate_configuration(market, "RS-MOM-01", None, stage="run_a", windows=W, log_path=p,
                             runs_dir=runs, rerun_note="fix Z")
    log = tl.read_log(p)
    assert len(log) == 2 and log["notes_json"].iloc[1]["rerun_of"] == "T0001"
    assert tl.n_raw(p) == 1


def test_scan_cache_is_keyed_on_the_full_config(market, mcfg):
    """QA minor: SES-IBX-01 reads rules.restricted_windows_utc (outside the shared settings);
    two configs that differ only there must not share scans."""
    base = R.scan_config(market, "SES-IBX-01", None, W)
    assert len(base.candidates) > 0
    cfg2 = json.loads(json.dumps(mcfg))
    cfg2["rules"]["restricted_windows_utc"] = [
        {"start": "2024-01-01T00:00:00Z", "end": "2024-12-31T00:00:00Z", "reason": "test"}]
    assert tl.shared_settings_hash(cfg2) == market.shared_hash      # identity unchanged
    mk2 = _clone(market, cfg=cfg2)
    assert mk2.scans is market.scans and mk2.cfg_digest != market.cfg_digest
    blocked = R.scan_config(mk2, "SES-IBX-01", None, W)
    assert len(blocked.candidates) == 0
    assert R.scan_config(market, "SES-IBX-01", None, W) is base     # still cached per config


def test_real_data_market_runs_only_inside_a_logging_protocol(market, tmp_path):
    """QA minor: on a load_market MarketData every outcome goes through the trial log."""
    real = _clone(market, source=R.REAL_DATA_SOURCE)
    assert real.is_real and not market.is_real
    refused = [
        lambda: R.run_config("RS-MOM-01", None, "STRESS-d2", W, market=real),
        lambda: R.run_candidates(real, [], "STRESS-d2", W),
        lambda: R.evaluate_configuration(real, "RS-MOM-01", None, stage="run_a", windows=W,
                                         log=False),
        lambda: R.run_a(real, "RS-MOM-01", windows=W, secondary=False, log=False),
        lambda: R.run_sensitivity(real, "RS-MOM-01", windows=W, log=False),
        lambda: R.run_b(real, "RS-MOM-01", folds=WF, log=False),
        lambda: R.run_portfolio(real, ["RS-MOM-01"], windows=W, log=False),
    ]
    for call in refused:
        with pytest.raises(PermissionError, match="real pre-holdout"):
            call()
    R.scan_config(real, "RS-MOM-01", None, W)                       # candidates only: allowed
    p, runs = tmp_path / "log.csv", tmp_path / "runs"
    ev = R.evaluate_configuration(real, "RS-MOM-01", None, stage="run_a", windows=W,
                                  log_path=p, runs_dir=runs)
    assert ev.trial.trial_id == "T0001"
    assert tl.read_log(p)["notes_json"].iloc[0]["extra"]["data_source"] == R.REAL_DATA_SOURCE
    with pytest.raises(PermissionError):                            # the context was closed
        R.run_config("RS-MOM-01", None, "STRESS-d2", W, market=real)
    R.run_config("RS-MOM-01", None, "STRESS-d2", W, market=market)  # synthetic: unrestricted


def test_competition_contract_only_when_configured(market, mcfg):
    assert R.contracts(mcfg) == ("demo",)
    cfg2 = json.loads(json.dumps(mcfg))
    cfg2["rules"]["competition_usd_per_point"] = {"US30": 2.0, "NAS100": 2.0, "GER40": 2.24}
    assert R.contracts(cfg2) == ("demo", "competition")
    mk2 = R.MarketData(cfg2, market.keys, market.m15, market.h1, market.features, market.tables,
                       SPECS, market.data_hash, "test", market.scans)
    ev = R.evaluate_configuration(mk2, "RS-MOM-01", None, stage="run_a", windows=W, log=False)
    assert set(ev.runs) == {"STRESS-d2", "STRESS-d2@competition"}
    with pytest.raises(ValueError, match="competition contract"):
        R.run_config("RS-MOM-01", None, "STRESS-d2", W, market=market, contract="competition")


def test_market_built_once_per_data_hash(mcfg):
    R.clear_market_cache()
    calls: list = []
    mk = make_market("2024-01-08", "2024-02-10", drops={}, holes={})
    a = R.market_from_frames(mcfg, mk.m15, mk.h1, SPECS, regime_factory=proxy_factory(calls),
                             regime_tag="proxy-count")
    b = R.market_from_frames(mcfg, mk.m15, mk.h1, SPECS, regime_factory=proxy_factory(calls),
                             regime_tag="proxy-count")
    assert len(calls) == 1 and a.features is b.features and a.scans is b.scans
    m15 = dict(mk.m15)
    m15["US30"] = m15["US30"].iloc[:-1]
    c = R.market_from_frames(mcfg, m15, mk.h1, SPECS, regime_factory=proxy_factory(calls),
                             regime_tag="proxy-count")
    assert len(calls) == 2 and c.data_hash != a.data_hash
    late = dict(mk.m15)
    extra = late["US30"].iloc[-1:].copy()
    extra["time_utc"] = ts("2026-05-01 00:00")
    late["US30"] = pd.concat([late["US30"], extra], ignore_index=True)
    with pytest.raises(PermissionError, match="holdout"):
        R.market_from_frames(mcfg, late, mk.h1, SPECS, regime_factory=proxy_factory(calls))


@pytest.mark.parametrize("call", [
    lambda mk: R.run_config("MR-TWAP-01", None, "STRESS-d2", W, market=mk),
    lambda mk: R.scan_config(mk, "MR-TWAP-01", None, W),
    lambda mk: R.run_portfolio(mk, ["RS-MOM-01", "MR-TWAP-01"], windows=W, log=False),
    lambda mk: R.run_a(mk, "MR-TWAP-01", windows=W, log=False),
])
def test_mr_twap_01_is_never_run(market, call):
    with pytest.raises(ValueError, match="A3"):
        call(market)


def test_off_grid_parameters_refused(market):
    with pytest.raises(ValueError, match="grid"):
        R.run_config("RS-MOM-01", {"theta": 0.95}, "STRESS-d2", W, market=market)


# ---------------------------------------------------------------------------------------------
# Sensitivity plan and criterion 5
# ---------------------------------------------------------------------------------------------

def test_sensitivity_plan_rs_mom_01():
    plan = R.sensitivity_plan(RS_PARAMS)
    assert len(plan) == 29 and plan["is_default"].sum() == 1
    assert plan["in_oat"].sum() == 13 and plan["in_heatmap"].sum() == 25
    nb = plan[plan["neighbour_of"] != ""]
    assert sorted(nb["label"]) == sorted([
        "theta=0.9,k=1.0,R=1.5", "theta=1.1,k=1.0,R=1.5", "theta=1.0,k=0.875,R=1.5",
        "theta=1.0,k=1.125,R=1.5", "theta=1.0,k=1.0,R=1.25", "theta=1.0,k=1.0,R=1.75"])


def test_criterion5():
    plan = R.sensitivity_plan(RS_PARAMS)
    plan["status"] = "positive"
    assert R.criterion5(plan)["passed"]
    t = plan.copy()
    t.loc[t["label"] == "theta=0.9,k=1.0,R=1.5", "status"] = "insufficient"
    c = R.criterion5(t)
    assert not c["neighbours_positive"] and not c["passed"] and c["n_insufficient"] == 1
    t = plan.copy()
    far = t[(t["neighbour_of"] == "") & ~t["is_default"]].index[:6]
    t.loc[far, "status"] = "not positive"                     # 23/29 = 0.79 < 0.8
    assert R.criterion5(t)["neighbours_positive"] and not R.criterion5(t)["passed"]
    t.loc[far[:5], "status"] = "positive"                     # 28/29
    assert R.criterion5(t)["passed"]


# ---------------------------------------------------------------------------------------------
# COMMON 10.4 trade counts
# ---------------------------------------------------------------------------------------------

def _trades(per_fold: dict[int, int]) -> pd.DataFrame:
    rows = []
    for f, n in per_fold.items():
        t0 = F.TEST_FOLDS[f - 1].start + pd.Timedelta(days=3, hours=15)
        for i in range(n):
            rows.append({"signal_time": t0 + pd.Timedelta(hours=i), "status": "filled",
                         "pnl_usd": 1.0, "r_multiple": 0.1, "key": "US30", "side": "BUY"})
    return pd.DataFrame(rows)


def test_trade_count_check_pro_rated():
    shares = {n: 1.0 for n in range(1, 10)}
    shares.update({4: 0.25, 6: 0.5, 7: 0.0})
    full = {n: 20 for n in range(1, 10)}
    full.update({4: 1, 6: 2, 7: 0})                           # 4 x share: 1, 2, 0 -> all ok
    c = F.trade_count_check(_trades(full), shares)
    assert c.n_total == 123 and not c.total_ok              # < 150
    assert c.n_folds_ok == 9 and c.folds_ok and not c.passed
    assert [f.required for f in c.folds][3] == 1.0
    more = dict(full)
    more[1] = 47                                              # 150 in total
    assert F.trade_count_check(_trades(more), shares).passed
    weak = dict(more)
    weak.update({2: 3, 3: 3, 4: 0})                           # three folds below the minimum
    c = F.trade_count_check(_trades(weak), shares)
    assert c.n_folds_ok == 6 and not c.folds_ok
    df = pd.DataFrame({"fold": list(shares), "data_share": list(shares.values())})
    assert F.trade_count_check(_trades(more), df).passed
    with pytest.raises(ValueError, match="no data share"):
        F.trade_count_check(_trades(more), {1: 1.0})


def test_bucket_stats_merges_small_buckets():
    t = pd.DataFrame({"status": "filled", "key": ["US30"] * 40 + ["NAS100"] * 10 + ["GER40"] * 5,
                      "pnl_usd": 1.0, "r_multiple": 0.5})
    b = F.bucket_stats(t, "key")
    assert list(b["bucket"]) == ["US30", "OTHER"] and list(b["n"]) == [40, 15]
    assert b["merged"].iloc[1] == "GER40,NAS100"


def test_sensitivity_cell_status():
    assert F.sensitivity_cell_status(99, 50.0) == "insufficient"
    assert F.sensitivity_cell_status(100, 0.01) == "positive"
    assert F.sensitivity_cell_status(100, 0.0) == "not positive"
    assert F.sensitivity_cell_status(150, float("nan")) == "not positive"


def test_competition_feasibility(mcfg):
    cal = pd.DatetimeIndex(pd.bdate_range("2024-07-01", periods=25))
    # trades (fills) on 5 of every 20 days at first, then none: windows starting later fail
    days = list(cal[[0, 4, 8, 12, 16]]) + list(cal[[20]])
    fills = [d + pd.Timedelta(hours=14) for d in days]          # 14:00 UTC = same Athens date
    t = pd.DataFrame({"status": "filled", "fill_time": pd.DatetimeIndex(fills, tz=UTC)})
    c = F.competition_feasibility(t, cal, mcfg)
    assert c.n_windows == 6
    assert c.windows["trade_days"].tolist() == [5, 5, 5, 5, 5, 4]
    assert c.n_ok == 5 and c.share == pytest.approx(5 / 6) and c.passed
    # two fills on one day count once
    t2 = pd.concat([t, t.iloc[:1]], ignore_index=True)
    assert F.competition_feasibility(t2, cal, mcfg).windows["trade_days"].iloc[0] == 5
    short = F.competition_feasibility(t, cal[:10], mcfg)
    assert short.n_windows == 0 and not short.passed


def test_competition_feasibility_reports_both_hole_readings(mcfg):
    """QA minor (escalated): windows over the data hole count as failed in `share` (literal);
    `share_with_data` / `passed_with_data` use only windows with data. Both are reported."""
    cal = pd.DatetimeIndex(pd.bdate_range("2024-07-01", periods=60))
    data = cal[:30]                                       # the last 30 days are a data hole
    fills = pd.DatetimeIndex(list(data), tz=UTC) + pd.Timedelta(hours=14)
    t = pd.DataFrame({"status": "filled", "fill_time": fills})
    c = F.competition_feasibility(t, cal, mcfg, data_days=data)
    assert c.n_windows == 41
    assert c.share < F.COMP_MIN_SHARE and not c.passed   # windows inside the hole fail
    assert c.share_with_data == pytest.approx(26 / 30) and c.passed_with_data is True
    d = c.to_dict()
    assert d["share"] == pytest.approx(26 / 41) and d["share_with_data"] == c.share_with_data
    assert d["passed"] is False and d["passed_with_data"] is True
    assert "awaiting Quant ruling" in d["ruling"]
    assert F.competition_feasibility(t, cal, mcfg).passed_with_data is None
