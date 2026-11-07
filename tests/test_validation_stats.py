"""validation.stats on SYNTHETIC data only, each statistic checked against an independent
reference (plain loops, statistics.NormalDist hand computation, brute-force CSCV)."""
from __future__ import annotations

import itertools
import math
from statistics import NormalDist

import numpy as np
import pytest

from validation import stats as vs


# --- helpers ---------------------------------------------------------------------------------
def _dd_loop(x) -> float:
    peak, cum, worst = 0.0, 0.0, 0.0
    for v in x:
        cum += v
        peak = max(peak, cum)
        worst = max(worst, peak - cum)
    return worst


def _streak_loop(x) -> int:
    best = run = 0
    for v in x:
        run = run + 1 if v < 0 else 0
        best = max(best, run)
    return best


def test_max_drawdown_and_streak_match_loops():
    rng = np.random.default_rng(7)
    m = rng.normal(0, 1, (50, 40))
    m[3] = -1.0                                    # only losses: dd = -total
    m[4] = 0.0
    dd = vs.max_drawdown(m, axis=1)
    st = vs.worst_losing_streak(m, axis=1)
    for i in range(len(m)):
        assert dd[i] == pytest.approx(_dd_loop(m[i]), abs=1e-12)
        assert st[i] == _streak_loop(m[i])
    assert dd[3] == pytest.approx(40.0) and st[3] == 40
    assert dd[4] == 0.0 and st[4] == 0
    assert vs.worst_losing_streak([-1, 0, -1, -1, 2, -3]) == 2      # zero ends a streak
    assert vs.max_drawdown([5.0, -2.0, -4.0, 3.0]) == pytest.approx(6.0)


def test_sharpe_ddof1_and_flat_series():
    x = np.array([1.0, 2.0, 4.0, -1.0])
    assert vs.sharpe(x) == pytest.approx(x.mean() / x.std(ddof=1))
    assert vs.sharpe(np.zeros(10)) == 0.0
    cols = vs.sharpe(np.column_stack([x, 2 * x, np.ones(4)]), axis=0)
    assert cols[0] == pytest.approx(cols[1]) and cols[2] == 0.0


# --- stationary bootstrap ----------------------------------------------------------------------
def _sb_reference(n, n_boot, mean_block, seed):
    """Plain-loop Politis-Romano with the same documented draw order."""
    rng = np.random.default_rng(seed)
    u = rng.random((n_boot, n))
    starts = rng.integers(0, n, size=(n_boot, n))
    out = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        for t in range(n):
            if t == 0 or u[b, t] < 1.0 / mean_block:
                out[b, t] = starts[b, t]
            else:
                out[b, t] = (out[b, t - 1] + 1) % n
    return out


def test_stationary_indices_match_loop_reference():
    idx = vs.stationary_bootstrap_indices(37, 200, 5.0, np.random.default_rng(99))
    assert np.array_equal(idx, _sb_reference(37, 200, 5.0, 99))


def test_stationary_block_length_mean_is_five():
    n = 2000
    rng = np.random.default_rng(3)
    idx = vs.stationary_bootstrap_indices(n, 500, 5.0, rng)
    breaks = idx[:, 1:] != (idx[:, :-1] + 1) % n          # a new block (bar a 1/n coincidence)
    n_blocks = breaks.sum() + idx.shape[0]
    mean_len = idx.size / n_blocks
    assert mean_len == pytest.approx(5.0, rel=0.02)


def test_stationary_bootstrap_reproducible_and_unbiased():
    rng = np.random.default_rng(11)
    x = rng.normal(50.0, 400.0, 300)
    a = vs.stationary_bootstrap(x)
    b = vs.stationary_bootstrap(x)
    c = vs.stationary_bootstrap(x, seed=1)
    assert a.seed == 12345 and a.n_resamples == 10_000 and a.mean_block == 5.0
    assert np.array_equal(a.totals, b.totals) and np.array_equal(a.drawdowns, b.drawdowns)
    assert a == b and not np.array_equal(a.totals, c.totals)
    # The circular stationary bootstrap is unbiased for the total.
    se = a.totals.std() / math.sqrt(a.n_resamples)
    assert a.totals.mean() == pytest.approx(x.sum(), abs=5 * se)
    assert a.observed_total == pytest.approx(x.sum())
    assert a.observed_max_dd == pytest.approx(_dd_loop(x))
    assert a.total_pnl.p5 < a.total_pnl.p50 < a.total_pnl.p95
    assert 0 <= a.max_drawdown.p5 <= a.max_drawdown.p50 <= a.max_drawdown.p95
    q = np.percentile(a.totals, [5, 50, 95])
    assert (a.total_pnl.p5, a.total_pnl.p50, a.total_pnl.p95) == tuple(q)
    idx = vs.stationary_bootstrap_indices(len(x), 10_000, 5.0, np.random.default_rng(12345))
    for i in (0, 17, 9999):                            # totals / drawdowns agree with loops
        assert a.totals[i] == pytest.approx(sum(x[idx[i]]))
        assert a.drawdowns[i] == pytest.approx(_dd_loop(x[idx[i]]))


def test_stationary_bootstrap_rejects_nan():
    with pytest.raises(ValueError):
        vs.stationary_bootstrap([1.0, np.nan, 2.0])


# --- regime-conditional bootstrap ----------------------------------------------------------------
def test_regime_bootstrap_keeps_bucket_counts_and_order():
    regimes = ["ACTIVE", "CALM", "ACTIVE", "ACTIVE", "CALM", "ACTIVE"]
    pnl = np.where(np.array(regimes) == "ACTIVE", 2.0, -1.0)
    r = vs.regime_bootstrap(pnl, regimes, n_resamples=500)
    assert r.buckets == {"ACTIVE": 4, "CALM": 2}
    # Constant within bucket + kept sequence => every resample equals the observed path.
    assert np.all(r.totals == pnl.sum())
    assert np.allclose(r.drawdowns, _dd_loop(pnl))


def test_regime_bootstrap_draws_only_from_own_bucket():
    rng = np.random.default_rng(5)
    regimes = rng.choice(["ACTIVE", "CALM"], 60)
    pnl = np.where(regimes == "ACTIVE", rng.uniform(1, 3, 60), rng.uniform(-2, -1, 60))
    r = vs.regime_bootstrap(pnl, regimes)
    na, nc = (regimes == "ACTIVE").sum(), (regimes == "CALM").sum()
    a, c = pnl[regimes == "ACTIVE"], pnl[regimes == "CALM"]
    assert np.all(r.totals >= na * a.min() + nc * c.min() - 1e-9)
    assert np.all(r.totals <= na * a.max() + nc * c.max() + 1e-9)
    again = vs.regime_bootstrap(pnl, regimes)
    assert np.array_equal(r.totals, again.totals)
    # Reference: same documented draw order (sorted buckets, one integers() call each).
    g = np.random.default_rng(12345)
    sample = np.empty((10_000, 60))
    for b in ("ACTIVE", "CALM"):
        pos = np.flatnonzero(regimes == b)
        sample[:, pos] = pnl[pos][g.integers(0, len(pos), size=(10_000, len(pos)))]
    assert np.allclose(r.totals, sample.sum(axis=1))


def test_regime_bootstrap_other_labels_and_length_check():
    r = vs.regime_bootstrap([1.0, -1.0, 2.0], ["ACTIVE", "n/a", "ACTIVE"], n_resamples=10)
    assert r.buckets == {"ACTIVE": 2, "n/a": 1}
    with pytest.raises(ValueError):
        vs.regime_bootstrap([1.0, 2.0], ["ACTIVE"])


# --- shuffle test -------------------------------------------------------------------------------
def test_shuffle_matches_loop_reference():
    rng = np.random.default_rng(21)
    x = rng.normal(10, 100, 80)
    s = vs.shuffle_test(x)
    assert s.n_permutations == 10_000 and s.seed == 12345
    g = np.random.default_rng(12345)
    perms = g.permuted(np.tile(x, (10_000, 1)), axis=1)
    for i in (0, 1, 500, 9999):
        assert sorted(perms[i]) == sorted(x)
        assert s.streaks[i] == _streak_loop(perms[i])
        assert s.drawdowns[i] == pytest.approx(_dd_loop(perms[i]))
    assert s.observed_worst_streak == _streak_loop(x)
    assert s.p95_worst_streak == pytest.approx(np.percentile(s.streaks, 95))
    assert s.p95_max_drawdown == pytest.approx(np.percentile(s.drawdowns, 95))
    assert np.array_equal(s.drawdowns, vs.shuffle_test(x).drawdowns)


# --- Deflated Sharpe -----------------------------------------------------------------------------
X_HAND = [0.012, -0.004, 0.008, 0.015, -0.011, 0.003, 0.020, -0.007, 0.001, 0.009, -0.002, 0.006]


def _hand_moments(x):
    n = len(x)
    mu = sum(x) / n
    sd1 = math.sqrt(sum((v - mu) ** 2 for v in x) / (n - 1))
    m2 = sum((v - mu) ** 2 for v in x) / n
    m3 = sum((v - mu) ** 3 for v in x) / n
    m4 = sum((v - mu) ** 4 for v in x) / n
    return mu / sd1, m3 / m2 ** 1.5, m4 / m2 ** 2


def test_dsr_n1_is_psr_against_zero_by_hand():
    sr, sk, ku = _hand_moments(X_HAND)
    z = sr * math.sqrt(len(X_HAND) - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    r = vs.deflated_sharpe(X_HAND, 1, float("nan"), "N_raw")
    assert r.sr_star == 0.0 and r.t_days == 12
    assert r.sr_hat == pytest.approx(sr, rel=1e-12)
    assert r.skew == pytest.approx(sk, rel=1e-12)
    assert r.kurtosis == pytest.approx(ku, rel=1e-12)
    assert r.dsr == pytest.approx(NormalDist().cdf(z), rel=1e-12)


def test_dsr_with_trials_by_hand():
    n_trials, v = 58, 0.0123
    g = 0.5772156649015329
    nd = NormalDist()
    sr_star = math.sqrt(v) * ((1 - g) * nd.inv_cdf(1 - 1 / n_trials)
                              + g * nd.inv_cdf(1 - 1 / (n_trials * math.e)))
    sr, sk, ku = _hand_moments(X_HAND)
    z = (sr - sr_star) * math.sqrt(11) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    r = vs.deflated_sharpe(X_HAND, n_trials, v)
    assert r.sr_star == pytest.approx(sr_star, rel=1e-10)
    assert r.dsr == pytest.approx(nd.cdf(z), rel=1e-10)
    assert vs.expected_max_sharpe(n_trials, 0.0) == 0.0
    assert vs.expected_max_sharpe(1000, 1.0) == pytest.approx(3.2546, abs=1e-3)
    assert r.passes == (r.dsr >= 0.95)


def test_dsr_more_trials_lowers_dsr():
    rng = np.random.default_rng(2)
    x = rng.normal(0.1, 1.0, 500)
    ds = [vs.deflated_sharpe(x, n, 0.002).dsr for n in (1, 2, 10, 100, 1000)]
    assert all(a > b for a, b in zip(ds, ds[1:]))


def _clustered_matrix(seed=4, t=400, groups=(4, 3, 5)):
    rng = np.random.default_rng(seed)
    cols, truth = [], []
    for gi, size in enumerate(groups):
        f = rng.normal(0.05 * gi, 1.0, t)
        for _ in range(size):
            cols.append(f + 0.2 * rng.normal(0, 1, t))
            truth.append(gi)
    return np.column_stack(cols), np.array(truth)


def test_effective_trials_recovers_groups_and_medoids():
    m, truth = _clustered_matrix()
    e = vs.effective_trials(m)
    assert e.n_raw == 12 and e.n_eff == 3
    for gi in range(3):                                   # clusters == true groups
        assert len(set(e.labels[truth == gi])) == 1
    # Medoid reference: min summed (1 - rho) inside the cluster.
    rho = np.corrcoef(m, rowvar=False)
    for c, med in zip(np.unique(e.labels), e.medoids):
        mem = np.flatnonzero(e.labels == c)
        costs = [(1 - rho[i, mem]).sum() for i in mem]
        assert med == mem[int(np.argmin(costs))]
    srs = m.mean(0) / m.std(0, ddof=1)
    assert np.allclose(e.sharpes, srs)
    assert e.v_raw == pytest.approx(np.var(srs, ddof=1))
    assert e.v_eff == pytest.approx(np.var(srs[e.medoids], ddof=1))


def test_effective_trials_independent_flat_and_single():
    rng = np.random.default_rng(8)
    ind = rng.normal(0, 1, (300, 6))
    assert vs.effective_trials(ind).n_eff == 6
    flat = np.column_stack([ind[:, 0], ind[:, 0] * 2 + 0.01, np.zeros(300)])
    e = vs.effective_trials(flat)
    assert e.n_eff == 2 and e.labels[0] == e.labels[1] != e.labels[2]
    one = vs.effective_trials(ind[:, :1])
    assert one.n_eff == 1 and math.isnan(one.v_raw) and math.isnan(one.v_eff)


def test_dsr_report_raw_and_eff():
    m, _ = _clustered_matrix()
    rep = vs.dsr_report(m[:, 0], m)
    assert rep.raw.label == "N_raw" and rep.raw.n_trials == 12
    assert rep.eff.label == "N_eff" and rep.eff.n_trials == 3
    assert rep.raw.sr_variance == pytest.approx(rep.trials.v_raw)
    assert rep.eff.sr_variance == pytest.approx(rep.trials.v_eff)
    assert rep.passes == (rep.raw.dsr >= 0.95 and rep.eff.dsr >= 0.95)
    single = vs.dsr_report(m[:, 0], m[:, :1])
    psr = vs.deflated_sharpe(m[:, 0], 1, 0.0)
    assert single.raw.dsr == single.eff.dsr == pytest.approx(psr.dsr)
    with pytest.raises(ValueError):
        vs.dsr_report(m[:, 0], m, n_raw=5)
    assert vs.dsr_report(m[:, 0], m, n_raw=20).raw.n_trials == 20


# --- PBO by CSCV ----------------------------------------------------------------------------------
def _pbo_brute(m, s=16):
    t, n = m.shape
    x = m[t % s:]
    blocks = np.array_split(np.arange(len(x)), s)
    logits = []
    for combo in itertools.combinations(range(s), s // 2):
        is_rows = np.concatenate([blocks[i] for i in combo])
        oos_rows = np.concatenate([blocks[i] for i in range(s) if i not in combo])
        is_sr = x[is_rows].mean(0) / x[is_rows].std(0, ddof=1)
        oos_sr = x[oos_rows].mean(0) / x[oos_rows].std(0, ddof=1)
        best = int(np.argmax(is_sr))
        rank = 1 + int((oos_sr < oos_sr[best]).sum())
        w = rank / (n + 1)
        logits.append(math.log(w / (1 - w)))
    logits = np.array(logits)
    return float(np.mean(logits <= 0)), logits


def test_pbo_matches_brute_force_cscv():
    rng = np.random.default_rng(31)
    m = rng.normal(0, 1, (16 * 5 + 3, 6)) + np.linspace(0, 0.2, 6)
    r = vs.pbo_cscv(m)
    ref_pbo, ref_logits = _pbo_brute(m)
    assert r.n_splits == 12_870 and r.dropped_days == 3 and r.t_used == 80 and r.block_len == 5
    assert np.allclose(r.logits, ref_logits)
    assert r.pbo == pytest.approx(ref_pbo)


def _pbo_brute_ties(m, s=16):
    """Independent CSCV reference with SPEC NOTE 7 tie rules: zero-std Sharpe = 0, IS ties ->
    lowest column, OOS ties -> scipy average rank (values rounded to 10 decimals)."""
    from scipy.stats import rankdata

    def sr(a):
        sd = a.std(0, ddof=1)
        return np.where(sd > 0, a.mean(0) / np.where(sd > 0, sd, 1.0), 0.0)

    t, n = m.shape
    x = m[t % s:]
    blocks = np.array_split(np.arange(len(x)), s)
    logits, sel, ranks = [], [], []
    for combo in itertools.combinations(range(s), s // 2):
        is_rows = np.concatenate([blocks[i] for i in combo])
        oos_rows = np.concatenate([blocks[i] for i in range(s) if i not in combo])
        is_sr = np.round(sr(x[is_rows]), 10)
        best = int(np.flatnonzero(is_sr == is_sr.max())[0])
        rank = float(rankdata(np.round(sr(x[oos_rows]), 10), method="average")[best])
        w = rank / (n + 1)
        logits.append(math.log(w / (1 - w)))
        sel.append(best)
        ranks.append(rank)
    logits = np.array(logits)
    return float(np.mean(logits <= 0)), logits, np.array(sel), np.array(ranks)


def test_pbo_ties_match_independent_reference():
    """Duplicated best column (IS and OOS ties) plus a flat column (Sharpe 0)."""
    rng = np.random.default_rng(32)
    base = rng.normal(0, 1, (16 * 4 + 2, 5)) + np.linspace(0, 0.3, 5)
    m = np.column_stack([base, base[:, 4], np.zeros(len(base))])
    r = vs.pbo_cscv(m)
    ref_pbo, ref_logits, ref_sel, ref_rank = _pbo_brute_ties(m)
    assert np.any(ref_sel == 4) and not np.any(ref_sel == 5)   # IS tie -> lowest column
    assert np.any(ref_rank % 1 == 0.5)                          # OOS ties averaged
    assert np.array_equal(r.selected, ref_sel)
    assert np.allclose(r.logits, ref_logits, rtol=0, atol=1e-12)
    assert r.pbo == pytest.approx(ref_pbo)


def test_pbo_noise_near_half_and_dominant_near_zero():
    rng = np.random.default_rng(12)
    noise = rng.normal(0, 1, (1600, 20))
    r = vs.pbo_cscv(noise)
    assert 0.3 < r.pbo < 0.7
    dom = noise.copy()
    dom[:, 7] += 0.5
    d = vs.pbo_cscv(dom)
    assert d.pbo < 0.02 and np.all(d.selected == 7)
    assert d.status == "PASS"


def test_pbo_drops_earliest_remainder_rows():
    rng = np.random.default_rng(13)
    m = rng.normal(0, 1, (16 * 10 + 7, 5))
    a = vs.pbo_cscv(m)
    m2 = m.copy()
    m2[:7] = 1e6                                   # earliest remainder rows are ignored
    b = vs.pbo_cscv(m2)
    assert a.dropped_days == 7 and np.array_equal(a.logits, b.logits)


def test_pbo_n1_and_ties():
    rng = np.random.default_rng(14)
    one = vs.pbo_cscv(rng.normal(0, 1, (160, 1)))
    assert math.isnan(one.pbo) and one.status == "UNDEFINED" and "N = 1" in one.note
    col = rng.normal(0, 1, 160)
    same = vs.pbo_cscv(np.column_stack([col, col, col]))
    # All columns tie: IS pick = column 0, OOS average rank (N + 1) / 2 -> logit 0 -> counted.
    assert np.all(same.selected == 0) and np.allclose(same.logits, 0.0) and same.pbo == 1.0
    assert same.status == "FAIL"
    flat = vs.pbo_cscv(np.zeros((160, 4)))
    assert flat.pbo == 1.0
    with pytest.raises(ValueError):
        vs.pbo_cscv(rng.normal(0, 1, (10, 3)))     # fewer rows than blocks
