"""COMMON 6 regime model: forward filter, emissions, schedule, look-ahead, refit window, model
hash, staleness, live rule, failure fallback and diagnostics. Synthetic bars only (no MT5)."""
from __future__ import annotations

import copy
import itertools
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from core.config import load_config
from regime import hmm_fit as hmm_mod
from regime import service as service_mod
from regime.diagnostics import gate_block_share, refit_diagnostics, summary
from regime.filter import forward_filter, log_emissions, stationary_distribution, warmup_steps
from regime.hmm_fit import (FitRecord, Prep, fit_regime, fit_window, n_params, settings_hash,
                            training_set, window_filter, window_sha256, window_warmup)
from regime.inputs import FULL, MARGINAL, NONE, build_slots, compute_inputs, core_slots, ns
from regime.service import (NO_M15, OK, REFIT_FIELDS, UNAVAILABLE, RegimeService, _vt_columns,
                            first_saturday_noon, latest_refit_at_or_before, refit_schedule, v_t)

ROOT = Path(__file__).resolve().parent.parent
KEY = "US30"
T_MID = pd.Timestamp("2024-05-15 15:00", tz="UTC")       # core close (10:00-11:00 New York bar)


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


# ---------------------------------------------------------------------------------------------
# Synthetic data and a small, fast config (fewer seeds/K, lower row minimum; logic unchanged)
# ---------------------------------------------------------------------------------------------

def small_cfg(**data_quality) -> dict:
    c = load_config(ROOT / "config.yaml")
    c["owner_confirmed"] = True
    c["sessions_utc"] = {}
    c["rules"]["competition_end_utc"] = None
    c["regime"].update(k_set=[2, 3], seeds=[0, 1], stability_seeds=[10], min_train_rows=250)
    c["data_quality"].update(data_quality)
    return c


def make_m15(start: str = "2024-01-02", end: str = "2024-08-01", seed: int = 7) -> pd.DataFrame:
    """Weekday M15 bars around the clock with volatility that switches by week and is higher in
    the US cash session, so the HMM has something to separate."""
    t = pd.date_range(start, end, freq="15min", tz="UTC", inclusive="left")
    t = t[t.day_of_week < 5]
    rng = np.random.default_rng(seed)
    week = np.asarray((t - t[0]) // pd.Timedelta(days=7), dtype=int)
    sig = rng.choice([0.5, 1.0, 2.5], size=week.max() + 1)[week]
    sig = sig * np.where((t.hour >= 13) & (t.hour <= 20), 1.8, 1.0)
    n = len(t)
    close = 30000 + np.cumsum(rng.normal(0, 4 * sig))
    open_ = close + rng.normal(0, 2 * sig)
    high = np.maximum(open_, close) + rng.uniform(0.5, 4, n) * sig
    low = np.minimum(open_, close) - rng.uniform(0.5, 4, n) * sig
    tv = (300 * sig * rng.uniform(0.5, 1.5, n)).astype("uint64")
    return pd.DataFrame({"time_utc": t, "open": open_, "high": high, "low": low, "close": close,
                         "tick_volume": tv, "spread": np.full(n, 20, dtype="int32"),
                         "real_volume": np.zeros(n, dtype="uint64")})


def to_h1(m15: pd.DataFrame) -> pd.DataFrame:
    g = m15.set_index("time_utc").resample("1h")
    h = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "tick_volume": g["tick_volume"].sum(),
                      "spread": g["spread"].min(), "real_volume": g["real_volume"].sum()})
    h = h.dropna(subset=["open"]).reset_index()
    h["tick_volume"] = h["tick_volume"].astype("uint64")
    return h


def service(c: dict, **kw) -> RegimeService:
    return RegimeService(c, model_dir=kw.pop("model_dir", None),
                         refit_log=kw.pop("refit_log", None), **kw)


@pytest.fixture(scope="module")
def bars():
    m15 = make_m15()
    return to_h1(m15), m15


@pytest.fixture(scope="module")
def rcfg():
    return small_cfg()


@pytest.fixture(scope="module")
def base(bars, rcfg):
    h1, m15 = bars
    svc = service(rcfg)
    return svc, svc.run_batch(KEY, h1, m15)


def decision_times(start, end) -> pd.DatetimeIndex:
    def utc(x):
        x = pd.Timestamp(x)
        return x.tz_localize("UTC") if x.tz is None else x
    return pd.date_range(utc(start), utc(end), freq="15min")


def same_outputs(a: pd.DataFrame, b: pd.DataFrame) -> None:
    for c in ("status", "cause", "label", "tag", "model_id"):
        assert (a[c].to_numpy() == b[c].to_numpy()).all(), c
    for c in ("v_t", "regime_prob"):
        np.testing.assert_array_equal(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float))


# ---------------------------------------------------------------------------------------------
# Forward filter (COMMON 6.6)
# ---------------------------------------------------------------------------------------------

def brute_force_filter(log_b: np.ndarray, start: np.ndarray, a: np.ndarray) -> np.ndarray:
    """p(s_t | y_0..t) by enumerating every state path; NaN rows contribute no emission."""
    n, k = log_b.shape
    b = np.where(np.isnan(log_b), 1.0, np.exp(log_b))
    out = np.zeros((n, k))
    for t in range(n):
        for path in itertools.product(range(k), repeat=t + 1):
            w = start[path[0]] * b[0, path[0]]
            for u in range(1, t + 1):
                w *= a[path[u - 1], path[u]] * b[u, path[u]]
            out[t, path[-1]] += w
        out[t] /= out[t].sum()
    return out


def test_forward_filter_matches_brute_force():
    rng = np.random.default_rng(3)
    k, n = 3, 7
    start = np.array([0.5, 0.3, 0.2])
    a = rng.uniform(0.1, 1.0, (k, k))
    a /= a.sum(axis=1, keepdims=True)
    log_b = rng.normal(-2.0, 1.5, (n, k))
    log_b[3] = np.nan                       # prediction-only step (missing core bar)
    alpha, bad = forward_filter(log_b, start, a)
    assert bad == []
    np.testing.assert_allclose(alpha, brute_force_filter(log_b, start, a), rtol=1e-10)
    # Prediction-only step is exactly alpha_{t-1} A.
    np.testing.assert_allclose(alpha[3], alpha[2] @ a, rtol=1e-12)


def test_segment_reset_uses_stationary_distribution():
    a = np.array([[0.9, 0.1], [0.3, 0.7]])
    start = np.array([0.99, 0.01])
    pi = stationary_distribution(a, start)
    np.testing.assert_allclose(pi @ a, pi, atol=1e-12)
    np.testing.assert_allclose(pi, [0.75, 0.25], atol=1e-12)
    log_b = np.array([[0.0, -5.0], [0.0, -5.0], [np.nan, np.nan], [-1.0, -2.0]])
    reset = np.array([False, False, True, False])
    alpha, _ = forward_filter(log_b, start, a, reset)
    np.testing.assert_allclose(alpha[2], pi, atol=1e-12)              # reset, no emission
    expect = pi @ a * np.exp([-1.0, -2.0])
    np.testing.assert_allclose(alpha[3], expect / expect.sum(), atol=1e-12)
    # Reset on an observation: stationary prior, then that emission.
    alpha2, _ = forward_filter(log_b[[0, 1, 3]], start, a, np.array([False, True, False]))
    e = pi * np.exp([0.0, -5.0])
    np.testing.assert_allclose(alpha2[1], e / e.sum(), atol=1e-12)


def test_ambiguous_stationary_falls_back_to_startprob():
    start = np.array([0.2, 0.8])
    np.testing.assert_allclose(stationary_distribution(np.eye(2), start), start)


def test_non_finite_step_is_prediction_only(caplog):
    a = np.array([[0.8, 0.2], [0.4, 0.6]])
    log_b = np.array([[0.0, 0.0], [-np.inf, -np.inf], [0.0, -1.0]])
    alpha, bad = forward_filter(log_b, np.array([0.5, 0.5]), a, label="t")
    assert bad == [1]
    np.testing.assert_allclose(alpha[1], alpha[0] @ a)
    assert "non-finite" in caplog.text


def test_emissions_full_marginal_none():
    rng = np.random.default_rng(1)
    means = rng.normal(0, 1, (2, 3))
    covars = np.stack([np.eye(3) + 0.3, np.diag([1.0, 2.0, 0.5])])
    z = rng.normal(0, 1, (3, 3))
    z[1, 2] = np.nan
    lb = log_emissions(z, np.array([FULL, MARGINAL, NONE]), means, covars)
    for j in range(2):
        assert lb[0, j] == pytest.approx(multivariate_normal.logpdf(z[0], means[j], covars[j]))
        assert lb[1, j] == pytest.approx(multivariate_normal.logpdf(
            z[1, :2], means[j][:2], covars[j][:2, :2]))
    assert np.isnan(lb[2]).all()


def test_tv_bad_bars_use_marginal_and_stay_out_of_training(bars):
    h1, m15 = bars
    c = small_cfg(tick_volume_bad=[{"start": "2024-03-04T00:00:00Z",
                                    "end": "2024-03-09T00:00:00Z", "reason": "test"}])
    slots = build_slots(h1, KEY, c, m15)
    inside = (slots["close_utc"] > ts("2024-03-04 01:00")) & (slots["close_utc"] < ts("2024-03-08"))
    s = slots[inside & slots["usable"]]
    assert len(s) and s["x3"].isna().all() and (s["kind"] == MARGINAL).all()
    tr = training_set(slots, ts("2024-04-06 12:00"), c)
    assert not slots.iloc[tr.rows]["tv_bad"].any()
    assert (slots.iloc[tr.rows]["kind"] == FULL).all()


def test_regime_code_never_uses_hmmlearn_posteriors():
    """Outputs come from the hand-written filter only (COMMON 6.6)."""
    pat = re.compile(r"\.(predict_proba|predict|decode|score_samples)\s*\(")
    for p in (ROOT / "regime").glob("*.py"):
        assert not pat.search(p.read_text(encoding="utf-8")), p.name


# ---------------------------------------------------------------------------------------------
# Schedule, observations, BIC
# ---------------------------------------------------------------------------------------------

def test_refit_schedule():
    assert first_saturday_noon(2024, 6) == ts("2024-06-01 12:00")
    assert first_saturday_noon(2024, 7) == ts("2024-07-06 12:00")
    assert first_saturday_noon(2025, 11) == ts("2025-11-01 12:00")
    assert latest_refit_at_or_before(ts("2024-06-01 11:59")) == ts("2024-05-04 12:00")
    assert latest_refit_at_or_before(ts("2024-06-01 12:00")) == ts("2024-06-01 12:00")
    assert latest_refit_at_or_before(ts("2024-01-03")) == ts("2023-12-02 12:00")
    assert refit_schedule(ts("2024-01-01"), ts("2024-03-02 12:00")) == [
        ts("2024-01-06 12:00"), ts("2024-02-03 12:00"), ts("2024-03-02 12:00")]


def test_bic_parameter_counts():
    assert [n_params(k) for k in (2, 3, 4)] == [21, 35, 51]


def test_core_slots_and_inputs(bars, rcfg):
    h1, m15 = bars
    slots = build_slots(h1, KEY, rcfg, m15)
    local = slots["close_utc"].dt.tz_convert("America/New_York") - pd.Timedelta(hours=1)
    assert local.dt.hour.between(9, 15).all()
    # Independence Day 2024 (NYSE holiday, bars printed): prediction-only, never usable.
    hol = slots[slots["local_date"] == pd.Timestamp("2024-07-04").date()]
    assert len(hol) == 7 and not hol["usable"].any()
    # A normal day has 7 usable core observations with finite inputs once warmed up.
    day = slots[slots["local_date"] == pd.Timestamp("2024-05-15").date()]
    assert len(day) == 7 and day["usable"].all() and (day["kind"] == FULL).all()


# ---------------------------------------------------------------------------------------------
# Refit window, hash, persistence
# ---------------------------------------------------------------------------------------------

def test_refit_uses_only_data_before_refit_time(bars, rcfg):
    h1, m15 = bars
    r = ts("2024-05-04 12:00")
    full = build_slots(h1, KEY, rcfg, m15)
    tr = training_set(full, r, rcfg)
    closes = full.iloc[tr.rows]["close_utc"]
    assert (closes < r).all() and (closes >= r - pd.Timedelta(days=365)).all()
    # Bars after R (perturbed) cannot change the fit.
    h1p, m15p = h1.copy(), m15.copy()
    for f in (h1p, m15p):
        after = f["time_utc"] >= r
        f.loc[after, ["high", "close"]] *= 1.01
        f.loc[after, "tick_volume"] = f.loc[after, "tick_volume"] * 3
    trunc = build_slots(h1[h1["time_utc"] < r], KEY, rcfg, m15[m15["time_utc"] < r])
    a = fit_regime(KEY, full, r, rcfg)
    b = fit_regime(KEY, build_slots(h1p, KEY, rcfg, m15p), r, rcfg)
    c = fit_regime(KEY, trunc, r, rcfg)
    assert a.status == "OK"
    assert a.train_sha256 == b.train_sha256 == c.train_sha256
    assert a.model.model_hash == b.model.model_hash == c.model.model_hash


def test_model_hash_reproducible_and_persisted(bars, rcfg, tmp_path, monkeypatch):
    h1, m15 = bars
    slots = build_slots(h1, KEY, rcfg, m15)
    r = ts("2024-05-04 12:00")
    a = fit_regime(KEY, slots, r, rcfg)
    hashes = {fit_regime(KEY, slots, r, rcfg).model.model_hash for _ in range(3)}
    hashes.add(fit_regime(KEY, slots, r, rcfg, n_jobs=2).model.model_hash)   # joblib workers
    assert hashes == {a.model.model_hash} and len(a.model.model_hash) == 64
    log = tmp_path / "refits.csv"
    s1 = service(rcfg, model_dir=tmp_path / "models", refit_log=log)
    rec = s1.fit(KEY, slots, r)
    assert rec.model.model_hash == a.model.model_hash
    assert list((tmp_path / "models" / KEY).glob("*.pkl"))
    # A new service loads the persisted model instead of refitting.
    monkeypatch.setattr(service_mod, "fit_regime",
                        lambda *a, **k: pytest.fail("refit despite a matching persisted model"))
    s2 = service(rcfg, model_dir=tmp_path / "models", refit_log=log)
    rec2 = s2.fit(KEY, slots, r)
    assert rec2.model.model_hash == a.model.model_hash
    assert rec2.train_sha256 == a.train_sha256 and rec2.fit_time == r
    logged = pd.read_csv(log)
    assert len(logged) == 1                                   # idempotent refit log
    row = logged.iloc[0]
    assert row["model_hash"] == a.model.model_hash and row["k"] == a.model.k
    assert '"restart_lnl"' in row["per_k"] and '"bic"' in row["per_k"]
    assert '"mu_vol"' in row["states"] and row["transmat"].startswith("[[")


def test_relabelling_order_and_tags(bars, rcfg):
    h1, m15 = bars
    m = fit_regime(KEY, build_slots(h1, KEY, rcfg, m15), ts("2024-05-04 12:00"), rcfg).model
    assert np.all(np.diff(m.mu_vol) >= 0)                       # ascending mu_vol
    for s in range(m.k):
        vol, d = m.labels[s].split("_")
        assert vol == ("ACTIVE" if m.mu_vol[s] >= 0 else "CALM")
        assert d in ("UP", "DOWN", "FLAT")
        assert m.se[s] == pytest.approx(m.sd_ret[s] / np.sqrt(m.occ[s]))
    assert m.occ.sum() == pytest.approx(m.n_rows)


# ---------------------------------------------------------------------------------------------
# Batch table, look-ahead, live path
# ---------------------------------------------------------------------------------------------

def test_batch_table_outputs(base):
    svc, table = base
    recs = {r.fit_time: r.status for r in table.records}
    assert recs[ts("2024-02-03 12:00")] == "INSUFFICIENT"
    assert recs[ts("2024-05-04 12:00")] == "OK"
    early = table.at(ts("2024-02-20 15:00"))
    assert early.status == UNAVAILABLE and "fewer than 250" in early.cause
    snap = table.at(T_MID)
    assert snap.ok and np.isfinite(snap.v_t)
    m = next(iv.model for iv in table.intervals if iv.start == ts("2024-05-04 12:00"))
    assert snap.v_t == pytest.approx(float(np.dot(snap.alpha, m.mu_vol)))
    assert snap.tag == ("ACTIVE" if snap.v_t >= 0 else "CALM")
    assert snap.passes_active == (snap.v_t >= 0) and snap.passes_calm == (snap.v_t < 0)
    assert snap.regime == f"{snap.tag}/{snap.label}"
    assert snap.label == m.labels[int(np.argmax(snap.alpha))]
    match = [i for i, t in enumerate(m.vol_tags) if t == snap.tag]
    assert snap.regime_prob == pytest.approx(sum(snap.alpha[i] for i in match))
    assert snap.obs_close == T_MID
    # Between observations the latest one is used; 14:59 still sees the 14:00 close.
    assert table.at(T_MID - pd.Timedelta(minutes=1)).obs_close == T_MID - pd.Timedelta(hours=1)


def test_lookahead_perturbing_bars_after_t(bars, rcfg, base):
    h1, m15 = bars
    _, table = base
    rng = np.random.default_rng(11)
    h1p, m15p = h1.copy(), m15.copy()
    for f in (h1p, m15p):
        after = f["time_utc"] >= T_MID                    # bars closing after T
        k = int(after.sum())
        f.loc[after, ["open", "high", "low", "close"]] = (
            f.loc[after, ["open", "high", "low", "close"]].to_numpy()
            * rng.uniform(0.97, 1.03, (k, 1)))
        f.loc[after, "high"] += 5.0
        f.loc[after, "tick_volume"] = rng.integers(1, 10_000, k).astype("uint64")
    h1p = h1p.drop(h1p.index[h1p["time_utc"] >= T_MID][5:40])     # and delete some bars
    pert = service(rcfg).run_batch(KEY, h1p, m15p)
    trunc = service(rcfg).run_batch(KEY, h1[h1["time_utc"] < T_MID],
                                    m15[m15["time_utc"] < T_MID])
    times = decision_times("2024-03-25", T_MID)
    ref = table.at_many(times)
    same_outputs(ref, pert.at_many(times))
    same_outputs(ref, trunc.at_many(times))
    assert (ref["status"] == OK).sum() > 1000
    # Sanity: the perturbation does change outputs after T.
    later = decision_times(T_MID + pd.Timedelta(hours=1), "2024-07-20")
    assert not np.array_equal(table.at_many(later)["v_t"].to_numpy(dtype=float),
                              pert.at_many(later)["v_t"].to_numpy(dtype=float), equal_nan=True)


def test_live_equals_batch_and_missing_bar_rule(bars, rcfg, base):
    h1, m15 = bars
    _, table = base
    svc = service(rcfg)
    for t in (T_MID, ts("2024-06-12 18:00"), ts("2024-06-12 18:15")):
        live = svc.live(KEY, h1, m15, t)
        ref = table.at(t, live_rule=True)
        assert live.status == ref.status == OK
        assert live.v_t == ref.v_t and live.label == ref.label and live.model_id == ref.model_id
    # The core H1 bar closing at T has not arrived: UNAVAILABLE, never the previous value.
    gone = h1[h1["time_utc"] != T_MID - pd.Timedelta(hours=1)]
    snap = svc.live(KEY, gone, m15, T_MID)
    assert snap.status == UNAVAILABLE and snap.cause == "regime H1 bar missing"
    assert not snap.passes_active and not snap.passes_calm
    assert snap.reason == "regime unavailable: regime H1 bar missing"
    # Backtest decisions apply the live rule by default (COMMON 9); only an explicit
    # live_rule=False (diagnostics) falls back to the latest observation.
    t2 = service(rcfg).run_batch(KEY, gone, m15)
    assert t2.at(T_MID).cause == "regime H1 bar missing"
    assert t2.at_many([T_MID])["cause"].iloc[0] == "regime H1 bar missing"
    assert t2.at(T_MID, live_rule=False).status == OK


def test_staleness(base):
    _, table = base
    last = table.frame.loc[table.frame["status"] == OK, "obs_close"].max()
    assert table.at(last + pd.Timedelta(days=4, hours=23)).status == OK
    s = table.at(last + pd.Timedelta(days=5, minutes=15))
    assert s.status == UNAVAILABLE and s.cause.startswith("stale")


def test_refit_failure_keeps_previous_model_up_to_62_days(bars, rcfg, monkeypatch):
    h1, m15 = bars
    real = service_mod.fit_regime
    cutoff = ts("2024-06-01 12:00")

    def failing(key, slots, fit_time, cfg, seeds=None, n_jobs=1):
        if pd.Timestamp(fit_time) < cutoff:
            return real(key, slots, fit_time, cfg, seeds, n_jobs)
        return FitRecord(key, pd.Timestamp(fit_time), "FAILED", "refit failed: test", 1000, 1,
                         "x", tuple(seeds or ()))

    monkeypatch.setattr(service_mod, "fit_regime", failing)
    table = service(rcfg).run_batch(KEY, h1, m15)
    may = next(iv.model for iv in table.intervals if iv.start == ts("2024-05-04 12:00"))
    retries = [r for r in table.records if r.fit_time > cutoff]
    assert len(retries) > 50                                # retried at each later core close
    s = table.at(ts("2024-06-20 15:00"))
    assert s.ok and s.model_id == may.model_id            # previous model, 47 days old
    late = table.at(may.fit_time + pd.Timedelta(days=62, hours=1))
    assert late.status == UNAVAILABLE and "older than 62 days" in late.cause


def test_diagnostics(base):
    svc, table = base
    d = refit_diagnostics(svc, table, stability_seeds=[10])
    assert len(d) == sum(r.status == "OK" for r in table.records)
    for c in ("share_active", "share_v_near_zero", "agree_sign_x2", "seed_stability"):
        v = d[c].dropna()
        assert len(v) and ((v >= 0) & (v <= 1)).all(), c
    assert d["label_churn"].iloc[1:].between(0, 1).all()
    s = summary(table)
    assert s["first_valid_model"] == next(r.fit_time for r in table.records if r.status == "OK")
    share = gate_block_share(table, decision_times("2024-05-06", "2024-05-31"), "ACTIVE")
    assert 0.0 <= share <= 1.0


# ---------------------------------------------------------------------------------------------
# M15 is required (COMMON 1.6): no silent H1 proxy, live history depth
# ---------------------------------------------------------------------------------------------

def test_m15_required_no_silent_h1_proxy(bars, rcfg):
    h1, m15 = bars
    with pytest.raises(ValueError, match="M15"):
        build_slots(h1, KEY, rcfg, None)
    with pytest.raises(ValueError, match="M15"):
        service(rcfg).run_batch(KEY, h1, None)
    # The proxy is reachable only through the explicit test flag.
    assert len(build_slots(h1, KEY, rcfg, None, h1_proxy=True))
    svc = service(rcfg)
    for snap in (svc.live(KEY, h1, None, T_MID),
                 svc.live_all({KEY: h1}, {}, T_MID)[KEY]):
        assert snap.status == UNAVAILABLE and snap.cause == NO_M15
        assert snap.reason == f"regime unavailable: {NO_M15}"


def test_history_needed_and_short_live_m15(bars, rcfg):
    h1, m15 = bars
    svc = service(rcfg)
    # T - 62 d = 2024-03-14 -> oldest refit that can serve T is 2024-03-02 12:00; its window
    # starts 2023-03-03; 20 NYSE trading dates before that (Presidents Day skipped) start on
    # 2023-02-02, midnight New York.
    assert svc.history_needed(KEY, T_MID) == ts("2023-02-02 05:00")
    # M15 covering only the last 60 days while H1 goes back further: truncated store.
    short = m15[m15["time_utc"] >= T_MID - pd.Timedelta(days=60)]
    snap = svc.live(KEY, h1, short, T_MID)
    assert snap.status == UNAVAILABLE and snap.cause == NO_M15
    # M15 and H1 both starting at the data start (shorter than needed) is not truncation.
    assert svc.live(KEY, h1, m15, T_MID).ok


# ---------------------------------------------------------------------------------------------
# Rounding: outputs independent of how many rows are processed (live == batch, ER-15)
# ---------------------------------------------------------------------------------------------

def test_vt_and_emissions_do_not_depend_on_row_count():
    rng = np.random.default_rng(5)
    n = 6000
    m = SimpleNamespace(k=4, mu_vol=rng.normal(0, 0.3, 4),
                        vol_tags=["CALM", "CALM", "ACTIVE", "ACTIVE"],
                        labels=["CALM_FLAT", "CALM_UP", "ACTIVE_DOWN", "ACTIVE_FLAT"])
    alpha = rng.dirichlet(np.full(4, 0.3), size=n)
    full = _vt_columns(alpha, m, 4)
    np.testing.assert_allclose(full["v_t"], alpha @ m.mu_vol, rtol=0, atol=1e-15)
    z = rng.normal(0, 1, (n, 3))
    kind = np.where(rng.uniform(size=n) < 0.1, MARGINAL, FULL)
    kind[rng.uniform(size=n) < 0.05] = NONE
    means = rng.normal(0, 1, (4, 3))
    a = rng.normal(0, 1, (4, 3, 3))
    covars = a @ a.transpose(0, 2, 1) + 0.5 * np.eye(3)
    lb = log_emissions(z, kind, means, covars)
    for _ in range(100):
        e = int(rng.integers(1, n + 1))
        s = int(rng.integers(0, e))
        part = _vt_columns(alpha[s:e], m, 4)
        for c in ("v_t", "regime_prob"):
            assert np.array_equal(part[c], full[c][s:e]), c
        assert np.array_equal(v_t(alpha[s:e], m), full["v_t"][s:e])
        assert np.array_equal(log_emissions(z[s:e], kind[s:e], means, covars), lb[s:e],
                              equal_nan=True)


# ---------------------------------------------------------------------------------------------
# Segment restart after a > 7-day gap (COMMON 1.4, 6.1, 6.3, 6.6; ER-15)
# ---------------------------------------------------------------------------------------------

GAP = (ts("2024-03-11"), ts("2024-03-21"))


@pytest.fixture(scope="module")
def gap_bars(bars):
    _, m15 = bars
    m = m15[(m15["time_utc"] < GAP[0]) | (m15["time_utc"] >= GAP[1])].reset_index(drop=True)
    return to_h1(m), m


def test_segment_restart_inputs_filter_and_sequences(gap_bars, rcfg):
    h1, m15 = gap_bars
    n_d = int(rcfg["regime"]["daily_cycle_bars"][KEY])
    df = compute_inputs(h1, KEY, rcfg, m15)
    assert df["segment"].nunique() == 2
    i0 = int(np.flatnonzero(df["time_utc"] >= GAP[1])[0])
    assert df["segment"].iloc[i0] != df["segment"].iloc[i0 - 1]
    first = df.iloc[i0]
    assert first["tr"] == first["high"] - first["low"]          # no C_prev across the gap
    # x2 needs 20 * n_d bars inside the new segment: NaN before, finite at the 20*n_d-th bar.
    assert df["x2"].iloc[i0:i0 + 20 * n_d - 1].isna().all()
    assert np.isfinite(df["x2"].iloc[i0 + 20 * n_d - 1])
    # x1, x3 need >= 10 same-hour values on earlier valid dates of the new segment.
    new = df.iloc[i0:]
    vdates = sorted(d for d in new.loc[new["valid_date"], "local_date"].unique())
    first10 = new["local_date"].isin(vdates[:10])
    assert new.loc[first10, ["x1", "x3"]].isna().all().all()
    day11 = new[(new["local_date"] == vdates[10]) & new["local_hour"].between(9, 15)]
    assert day11[["x1", "x3"]].notna().all().all()

    # Filter: alpha resets to the stationary distribution at the first slot of the new segment.
    slots = core_slots(df, KEY, rcfg)
    j = int(np.flatnonzero(slots["segment"].to_numpy() != slots["segment"].iloc[0])[0])
    assert slots["open_utc"].iloc[j] >= GAP[1] and slots["open_utc"].iloc[j - 1] < GAP[0]
    assert slots["kind"].iloc[j] == NONE                        # x2 still warming up
    tm = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    sp = np.array([0.6, 0.3, 0.1])
    means = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.5]])
    covars = np.stack([np.eye(3)] * 3)
    prep = Prep(np.full(3, -np.inf), np.full(3, np.inf), np.zeros(3), np.ones(3))
    alpha, _ = window_filter(slots, 0, len(slots), sp, tm, means, covars, prep, "gap test")
    pi = stationary_distribution(tm, sp)
    np.testing.assert_allclose(alpha[j], pi, rtol=0, atol=1e-12)
    assert not np.allclose(alpha[j - 1], pi, atol=1e-6)
    # Training rows on both sides of the gap form two hmmlearn sequences.
    tr = training_set(slots, ts("2024-05-04 12:00"), rcfg)
    assert len(tr.lengths) == 2 and tr.lengths.sum() == len(tr.x)
    split = slots.iloc[tr.rows]["close_utc"].iloc[int(tr.lengths[0])]
    assert split > GAP[1]


# ---------------------------------------------------------------------------------------------
# Regime warm-up after a segment reset (COMMON 12B, amendment A2)
# ---------------------------------------------------------------------------------------------

# A > 7-day gap while a model is serving: the May model serves the reset, the June model (fitted
# with x2 still NaN in the new segment) serves the rest of the warm-up stretch.
GAP_W = (ts("2024-05-20"), ts("2024-05-30"))
R_JUN = ts("2024-06-01 12:00")


def test_warmup_steps_unit():
    f, t = False, True
    #            0  1  2  3  4  5  6  7  8
    reset = [f, f, f, t, f, f, t, f, f]
    emit = [t, f, t, f, f, t, t, f, f]
    #    reset at 3 without emission -> 3, 4 warm-up; 5 emits -> OK; reset at 6 with an
    #    emission -> not warm-up (the reset is applied, then that step's emission).
    np.testing.assert_array_equal(warmup_steps(reset, emit), [f, f, f, t, t, f, f, f, f])
    # Step 0 is the filter start (startprob), never a reset: no emission is not warm-up.
    np.testing.assert_array_equal(warmup_steps([f, f], [f, f]), [f, f])
    np.testing.assert_array_equal(warmup_steps([f, t, f, t], [f, f, f, f]), [f, t, t, t])
    assert warmup_steps([], []).shape == (0,)


@pytest.fixture(scope="module")
def warm_gap(bars, rcfg):
    _, m15 = bars
    m = m15[(m15["time_utc"] < GAP_W[0]) | (m15["time_utc"] >= GAP_W[1])].reset_index(drop=True)
    h1 = to_h1(m)
    svc = service(rcfg)
    table = svc.run_batch(KEY, h1, m)
    s = table.slots
    j = int(np.flatnonzero(ns(s["close_utc"]) >= GAP_W[1].value)[0])      # reset slot
    e = j + int(np.flatnonzero(s["kind"].to_numpy()[j:] != NONE)[0])      # first emission
    return SimpleNamespace(h1=h1, m15=m, svc=svc, table=table, j=j, e=e,
                           reset=s["close_utc"].iloc[j], first=s["close_utc"].iloc[e])


def test_warmup_unavailable_until_first_emitting_observation(warm_gap):
    g, table = warm_gap, warm_gap.table
    s = table.slots
    assert s["segment"].iloc[g.j] != s["segment"].iloc[g.j - 1]
    assert g.e > g.j and g.reset < R_JUN < g.first          # the warm-up spans a refit
    # The stretch is the NaN-x2 stretch: every observation in it lacks x2; the first emitting
    # one has x1 and x2 finite.
    warm_obs = s.iloc[g.j:g.e]
    assert warm_obs.loc[warm_obs["usable"], "x2"].isna().all()
    assert np.isfinite(s["x1"].iloc[g.e]) and np.isfinite(s["x2"].iloc[g.e])
    # Every decision time from the reset observation up to the first emitting one: UNAVAILABLE
    # ("regime warm-up"), never the stationary-prior V_T, under both models.
    times = decision_times(g.reset, g.first - pd.Timedelta(minutes=15))
    out = table.at_many(times)
    assert len(out) > 1000
    assert (out["status"] == UNAVAILABLE).all() and (out["cause"] == "regime warm-up").all()
    assert out["v_t"].isna().all() and (out["label"] == "").all()
    assert set(out["model_id"]) == {iv.model.model_id for iv in table.intervals
                                    if iv.model is not None and iv.start in (R_MAY, R_JUN)}
    for t in (g.reset, R_JUN, g.first - pd.Timedelta(minutes=15)):
        snap = table.at(t)
        assert snap.status == UNAVAILABLE and snap.cause == "regime warm-up"
        assert not snap.passes_active and not snap.passes_calm
        assert snap.reason == "regime unavailable: regime warm-up" and snap.regime == "UNAVAILABLE"
    # The filter still ran: the reset row's alpha is the stationary distribution of the model.
    may = next(iv.model for iv in table.intervals if iv.start == R_MAY)
    row = table.frame[table.frame["obs_close"] == g.reset].iloc[0]
    np.testing.assert_allclose([row[f"p{i}"] for i in range(may.k)],
                               stationary_distribution(may.transmat, may.startprob), atol=1e-12)
    # From the first emitting observation on, 6.6 applies unchanged.
    first = table.at(g.first)
    assert first.ok and np.isfinite(first.v_t) and first.obs_close == g.first
    after = table.at_many(decision_times(g.first, g.first + pd.Timedelta(days=10)))
    assert (after["cause"] != "regime warm-up").all()
    assert (after["status"] == OK).mean() > 0.95
    # Before the reset, the gap itself: the pre-gap observation goes stale, not warm-up.
    assert table.at(GAP_W[1]).cause.startswith("stale")
    assert (table.frame["cause"] == "regime warm-up").sum() == len(
        table.frame[(table.frame["obs_close"] >= g.reset) & (table.frame["obs_close"] < g.first)])


def test_warmup_live_equals_batch(warm_gap):
    g = warm_gap
    svc = service(small_cfg())
    for t in (g.reset, g.reset + pd.Timedelta(hours=2, minutes=15), R_JUN,
              g.first - pd.Timedelta(minutes=15), g.first, g.first + pd.Timedelta(hours=1)):
        live = svc.live(KEY, g.h1, g.m15, t)
        ref = g.table.at(t)
        assert (live.status, live.cause, live.model_id) == (ref.status, ref.cause, ref.model_id)
        assert live.v_t == ref.v_t or (np.isnan(live.v_t) and np.isnan(ref.v_t))
    assert svc.live(KEY, g.h1, g.m15, g.reset).cause == "regime warm-up"
    assert svc.live(KEY, g.h1, g.m15, g.first).ok


def test_warmup_ends_at_first_marginal_emission(warm_gap):
    """A2: the warm-up ends at the first observation with an emission, full OR marginal. Flag
    the tick volume of the day of the first x2-finite slot as bad: that slot becomes MARGINAL
    (x1, x2 finite, x3 NaN) and must still end the warm-up and be OK."""
    g = warm_gap
    day = g.first.normalize()
    c = small_cfg(tick_volume_bad=[{"start": day.isoformat(),
                                    "end": (day + pd.Timedelta(days=1)).isoformat(),
                                    "reason": "test"}])
    slots = build_slots(g.h1, KEY, c, g.m15)
    close = slots["close_utc"]
    j = int(np.flatnonzero(close == g.reset)[0])
    e = int(np.flatnonzero(close == g.first)[0])
    assert slots["kind"].iloc[e] == MARGINAL and np.isnan(slots["x3"].iloc[e])
    assert np.isfinite(slots["x1"].iloc[e]) and np.isfinite(slots["x2"].iloc[e])
    assert (slots["kind"].iloc[j:e] == NONE).all()
    warm = window_warmup(slots, 0, len(slots))
    assert warm[j:e].all() and not warm[e] and not warm[e:].any()
    # The marginal kind is what ends it: the same slot treated as NONE would still be warm-up.
    as_none = slots.copy()
    as_none.loc[e, "kind"] = NONE
    assert window_warmup(as_none, 0, len(as_none))[e]
    # Batch row and live snapshot at that observation: OK under the June model (whose training
    # window ends before the flagged day, so it is the same model as in warm_gap).
    svc = service(c)
    table = svc.run_batch(KEY, g.h1, g.m15)
    row = table.at(g.first)
    assert row.ok and row.cause != "regime warm-up" and np.isfinite(row.v_t)
    assert row.obs_close == g.first and row.model_id == g.table.at(g.first).model_id
    assert table.at(g.first - pd.Timedelta(minutes=15)).cause == "regime warm-up"
    live = svc.live(KEY, g.h1, g.m15, g.first)
    assert (live.status, live.cause, live.v_t) == (row.status, row.cause, row.v_t)


def test_warmup_reset_slot_not_a_usable_observation(warm_gap):
    """A2 when the reset slot itself is not a core observation (the real 2025 post-hole case:
    the first post-gap date has no cash-open M15 bar, so it is not a valid date, COMMON 1.6).
    The warm-up starts at that slot all the same (resets come from every slot, not only usable
    ones); until the first usable observation the latest output row is still the pre-gap one,
    so T in between is stale; every usable observation up to the first emission is warm-up;
    live equals batch at those times."""
    g = warm_gap
    day = GAP_W[1]                                       # Thu 2024-05-30, first post-gap date
    cash_open = day + pd.Timedelta(hours=13, minutes=30)  # 09:30 New York (EDT)
    m = g.m15[g.m15["time_utc"] != cash_open].reset_index(drop=True)
    assert len(m) == len(g.m15) - 1
    c = small_cfg()
    svc = service(c)
    table = svc.run_batch(KEY, g.h1, m)                  # H1 complete; only M15 lacks the bar
    s = table.slots
    close = s["close_utc"]
    j = int(np.flatnonzero(ns(close) >= day.value)[0])               # reset slot
    assert s["segment"].iloc[j] != s["segment"].iloc[j - 1]
    assert close.iloc[j].normalize() == day
    assert not s["valid_date"].iloc[j] and not s["usable"].iloc[j]
    u = j + int(np.flatnonzero(s["usable"].to_numpy()[j:])[0])       # first usable observation
    e = j + int(np.flatnonzero(s["kind"].to_numpy()[j:] != NONE)[0])  # first emission
    assert close.iloc[u].normalize() > day and u < e
    warm = window_warmup(s, 0, len(s))
    assert warm[j] and warm[j:e].all() and not warm[e:].any()          # starts at the reset slot
    # Gap end up to the first usable observation: the pre-gap observation, stale.
    t_u, t_e = close.iloc[u], close.iloc[e]
    pre = table.at_many(decision_times(day, t_u - pd.Timedelta(minutes=15)))
    assert len(pre) > 50 and (pre["status"] == UNAVAILABLE).all()
    assert pre["cause"].str.startswith("stale").all()
    assert (pre["obs_close"] < GAP_W[0]).all()
    # First usable observation up to the first emitting one: warm-up, never the stationary prior.
    wu = table.at_many(decision_times(t_u, t_e - pd.Timedelta(minutes=15)))
    assert len(wu) > 1000 and (wu["cause"] == "regime warm-up").all()
    assert wu["v_t"].isna().all() and (wu["label"] == "").all()
    f = table.frame
    in_stretch = (f["obs_close"] >= close.iloc[j]) & (f["obs_close"] < t_e)
    assert (f.loc[in_stretch, "cause"] == "regime warm-up").all()
    assert int((f["cause"] == "regime warm-up").sum()) == int(in_stretch.sum())
    # One row per usable observation in [reset slot, first emission); the June refit row
    # repeats the last pre-refit observation, hence the unique count.
    assert f.loc[in_stretch, "obs_close"].nunique() == int(s["usable"].to_numpy()[j:e].sum())
    assert table.at(t_e).ok
    # Live equals batch across the stretch.
    for t in (day + pd.Timedelta(hours=14), close.iloc[j], t_u - pd.Timedelta(minutes=15), t_u,
              R_JUN, t_e - pd.Timedelta(minutes=15), t_e):
        live = svc.live(KEY, g.h1, m, t)
        ref = table.at(t)
        assert (live.status, live.cause, live.model_id) == (ref.status, ref.cause, ref.model_id)
        assert live.v_t == ref.v_t or (np.isnan(live.v_t) and np.isnan(ref.v_t))
    assert svc.live(KEY, g.h1, m, close.iloc[j]).cause.startswith("stale")
    assert svc.live(KEY, g.h1, m, t_u).cause == "regime warm-up"


def test_warmup_diagnostics_and_no_warmup_without_reset(warm_gap, base):
    g = warm_gap
    _, plain = base
    assert not (plain.frame["cause"] == "regime warm-up").any()       # no reset, no warm-up
    assert summary(plain)["n_warmup_rows"] == 0
    s = summary(g.table)
    n_warm = int((g.table.frame["cause"] == "regime warm-up").sum())
    assert s["n_warmup_rows"] == n_warm > 0
    assert s["share_warmup_rows"] == pytest.approx(n_warm / s["n_model_rows"])
    d = refit_diagnostics(g.svc, g.table, stability_seeds=())
    jun = d[d["fit_time_utc"] == R_JUN].iloc[0]
    assert jun["n_warmup_obs"] > 0
    # Shares are over observations with a regime output only (warm-up excluded).
    assert np.isfinite(jun["agree_sign_x2"]) and jun["n_core_obs"] > 0


# ---------------------------------------------------------------------------------------------
# Degeneracy screens, rejected K, FAILED, BIC tie-break (COMMON 6.4)
# ---------------------------------------------------------------------------------------------

R_MAY = ts("2024-05-04 12:00")


@pytest.fixture(scope="module")
def restarts(bars, rcfg):
    """Real restart results at R_MAY, computed once; scenarios modify copies of them."""
    h1, m15 = bars
    slots = build_slots(h1, KEY, rcfg, m15)
    tr = training_set(slots, R_MAY, rcfg)
    real: dict = {}
    orig = hmm_mod.fit_restart

    def record(z, lengths, k, seed, rc):
        real[(k, seed)] = orig(z, lengths, k, seed, rc)
        return real[(k, seed)]

    mp = pytest.MonkeyPatch()
    mp.setattr(hmm_mod, "fit_restart", record)
    try:
        base_rec = fit_regime(KEY, slots, R_MAY, rcfg)
    finally:
        mp.undo()
    return slots, len(tr.x), real, base_rec


def _fit_with(monkeypatch, restarts, rcfg, change):
    slots, _, real, _ = restarts

    def fake(z, lengths, k, seed, rc):
        return change(copy.deepcopy(real[(k, seed)]))

    monkeypatch.setattr(hmm_mod, "fit_restart", fake)
    return fit_regime(KEY, slots, R_MAY, rcfg)


def _for_k(k, fn):
    return lambda r: fn(r) if r["k"] == k else r


def test_unmodified_restarts_pass_every_screen(restarts):
    _, _, _, rec = restarts
    assert rec.status == "OK"
    assert all(rec.per_k[k]["status"] == "OK" for k in (2, 3))


@pytest.mark.parametrize("name, change, reason", [
    ("not PD", lambda r: r["covars"].__setitem__(0, np.diag([1.0, 1.0, -1.0])) or r,
     "not positive definite"),
    ("condition", lambda r: r["covars"].__setitem__(0, np.diag([1.0, 1.0, 1e-7])) or r,
     "condition number"),
    ("row sum", lambda r: r["transmat"].__setitem__(0, r["transmat"][0] * (1 + 1e-5)) or r,
     "row sum"),
    ("occupancy", lambda r: r["means"].__setitem__(0, np.full(3, 50.0)) or r,
     "state occupancy"),
    ("not converged", lambda r: r.update(converged=False) or r, "no converged restart"),
])
def test_degeneracy_screen_rejects_k(monkeypatch, restarts, rcfg, name, change, reason):
    rec = _fit_with(monkeypatch, restarts, rcfg, _for_k(3, change))
    assert rec.per_k[3]["status"] == "REJECTED" and reason in rec.per_k[3]["reason"], name
    assert rec.per_k[2]["status"] == "OK"
    assert rec.status == "OK" and rec.model.k == 2


def test_every_k_rejected_is_failed(monkeypatch, restarts, rcfg):
    def bad(r):
        r["covars"][0] = np.diag([1.0, 1.0, -1.0])
        return r
    rec = _fit_with(monkeypatch, restarts, rcfg, bad)
    assert rec.status == "FAILED" and rec.model is None
    assert rec.cause == "refit failed: every K rejected or every restart failed"
    assert all(rec.per_k[k]["status"] == "REJECTED" for k in (2, 3))


@pytest.mark.parametrize("delta, k_sel", [(0.0, 2), (5e-7, 2), (1e-3, 3)])
def test_bic_tie_goes_to_smaller_k(monkeypatch, restarts, rcfg, delta, k_sel):
    """BIC(3) = BIC(2) - delta: within 1e-6 the smaller K wins."""
    _, n, real, _ = restarts
    lnl2 = max(r["lnl"] for (k, _), r in real.items() if k == 2 and r["converged"])
    bic2 = -2.0 * lnl2 + n_params(2) * np.log(n)
    lnl3 = (n_params(3) * np.log(n) - (bic2 - delta)) / 2.0

    def tie(r):
        r["lnl"] = lnl3
        return r
    rec = _fit_with(monkeypatch, restarts, rcfg, _for_k(3, tie))
    assert rec.per_k[2]["bic"] - rec.per_k[3]["bic"] == pytest.approx(delta, abs=1e-8)
    assert rec.status == "OK" and rec.model.k == k_sel


# ---------------------------------------------------------------------------------------------
# GER40 core slots and the DST mismatch weeks (COMMON 2, 6.1; ER-15)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("start, end, switches", [
    # US clocks move 2024-03-10, Europe 2024-03-31; back: Europe 2024-10-27, US 2024-11-03.
    ("2024-02-26", "2024-04-13", {"US30": "2024-03-10", "GER40": "2024-03-31"}),
    ("2024-10-14", "2024-11-16", {"US30": "2024-11-03", "GER40": "2024-10-27"}),
])
def test_core_slots_across_dst_mismatch_weeks(rcfg, start, end, switches):
    m15 = make_m15(start, end, seed=2)
    h1 = to_h1(m15)
    spec = {"US30": ("America/New_York", 9, 15), "GER40": ("Europe/Berlin", 9, 16)}
    for key, (tz, h_first, h_last) in spec.items():
        slots = build_slots(h1, key, rcfg, m15)
        local_open = slots["open_utc"].dt.tz_convert(tz)
        assert local_open.dt.hour.between(h_first, h_last).all()
        assert (local_open.dt.minute == 0).all()
        assert (slots["close_utc"] - slots["open_utc"] == pd.Timedelta(hours=1)).all()
        use = slots[slots["usable"]]
        per_day = use.groupby("local_date").size()
        assert (per_day == h_last - h_first + 1).all(), key
        # The first core bar's UTC hour moves on the index's own switch date, not the other's.
        first_utc = use.groupby("local_date")["open_utc"].min().dt.hour
        sw = pd.Timestamp(switches[key]).date()
        assert first_utc[first_utc.index < sw].nunique() == 1
        assert first_utc[first_utc.index > sw].nunique() == 1
        assert first_utc[first_utc.index < sw].iloc[0] != first_utc[first_utc.index > sw].iloc[0]
    # Good Friday 2024-03-29 is a holiday for both exchanges: bars printed, none usable.
    if start.startswith("2024-02"):
        for key in spec:
            s = build_slots(h1, key, rcfg, m15)
            gf = s[s["local_date"] == pd.Timestamp("2024-03-29").date()]
            assert len(gf) and not gf["usable"].any()


# ---------------------------------------------------------------------------------------------
# Persisted-model cache keyed on the whole filter window; refit log header; config checks
# ---------------------------------------------------------------------------------------------

def _fake_fit(calls):
    def fake(key, slots, fit_time, cfg, seeds=None, n_jobs=1):
        calls.append(fit_time)
        tr = training_set(slots, fit_time, cfg)
        seeds = tuple(int(s) for s in (cfg["regime"]["seeds"] if seeds is None else seeds))
        return FitRecord(key, pd.Timestamp(fit_time), "FAILED", "refit failed: fake",
                         len(tr.x), len(tr.lengths), tr.sha256, seeds,
                         settings_hash=settings_hash(cfg, seeds),
                         window_sha256=window_sha256(slots, tr.start, fit_time))
    return fake


def test_cache_refits_when_only_non_training_slots_change(bars, rcfg, tmp_path, monkeypatch):
    h1, m15 = bars
    slots = build_slots(h1, KEY, rcfg, m15)
    tr = training_set(slots, R_MAY, rcfg)
    lo, hi = fit_window(slots, tr.start, R_MAY)
    other = [j for j in range(lo, hi) if slots["kind"].iloc[j] != FULL]
    assert other
    changed = slots.copy()
    changed.loc[other[-1], "x1"] = 0.123                    # not a training row
    assert training_set(changed, R_MAY, rcfg).sha256 == tr.sha256
    assert window_sha256(changed, tr.start, R_MAY) != window_sha256(slots, tr.start, R_MAY)
    calls: list = []
    monkeypatch.setattr(service_mod, "fit_regime", _fake_fit(calls))
    s1 = service(rcfg, model_dir=tmp_path / "models")
    s1.fit(KEY, slots, R_MAY)
    service(rcfg, model_dir=tmp_path / "models").fit(KEY, slots, R_MAY)
    assert len(calls) == 1                                  # identical window: reloaded
    service(rcfg, model_dir=tmp_path / "models").fit(KEY, changed, R_MAY)
    assert len(calls) == 2                                  # window changed: refitted
    s1.fit(KEY, changed, R_MAY)                             # in-memory cache checks it too
    assert len(calls) == 2


def test_refit_log_gains_window_hash_column(bars, rcfg, tmp_path, monkeypatch):
    h1, m15 = bars
    slots = build_slots(h1, KEY, rcfg, m15)
    log = tmp_path / "refits.csv"
    old = [c for c in REFIT_FIELDS if c != "window_sha256"]
    pd.DataFrame([{c: "old" for c in old}]).to_csv(log, index=False)
    monkeypatch.setattr(service_mod, "fit_regime", _fake_fit([]))
    service(rcfg, refit_log=log).fit(KEY, slots, R_MAY)
    out = pd.read_csv(log, dtype=str, keep_default_na=False)
    assert list(out.columns) == REFIT_FIELDS and len(out) == 2
    assert out["key"].iloc[0] == "old" and out["window_sha256"].iloc[0] == ""
    assert len(out["window_sha256"].iloc[1]) == 64


def test_config_refit_rule_and_state_columns(bars, rcfg, base):
    bad = copy.deepcopy(rcfg)
    bad["regime"]["refit"] = "weekly"
    with pytest.raises(ValueError, match="first_saturday_12utc"):
        RegimeService(bad, model_dir=None, refit_log=None)
    _, table = base                                       # small config: k_set [2, 3]
    assert table.n_states == 3
    assert [c for c in table.frame.columns if re.fullmatch(r"p\d", c)] == ["p0", "p1", "p2"]
    snap = table.at(T_MID)
    assert len(snap.alpha) == snap.k and snap.obs_kind == FULL
