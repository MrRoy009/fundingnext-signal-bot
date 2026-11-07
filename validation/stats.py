"""GATE 3 statistics (COMMON 10.5 items 6, 8, 10 and 10.7): pure functions on arrays.

Nothing here knows about strategies, bars or the trial log. The runner builds the inputs from
pre-registered configurations (daily series on the 10.2 calendar, filled-trade $ P&L in trade
order, the trial matrix T_days x N) and passes them in; every function is deterministic for a
given seed and never reads data itself.

Contents
- stationary_bootstrap (Politis-Romano 1994) of the daily $ series: total $ P&L and max drawdown
  quantiles. regime_bootstrap: trades resampled within their regime bucket, counts kept.
- shuffle_test: permutations of trade order; worst losing streak and max drawdown quantiles.
- sharpe / deflated_sharpe / effective_trials / dsr_report: Bailey and Lopez de Prado (2014) as
  COMMON 10.7 writes it, with N_raw (V across all trials) and N_eff (V across cluster medoids).
- pbo_cscv: Bailey, Borwein, Lopez de Prado, Zhu (2017), S = 16 blocks, all C(16, 8) splits.

Conventions (WHY)
- Seeds: every resampling function builds its own numpy Generator from `seed` (12345 per COMMON
  10.5), so a result never depends on which other statistic was computed first.
- Max drawdown is in $, positive, measured on the cumulative P&L path that starts at 0 (the
  starting balance is the first peak), so a series that only loses has drawdown = -total.
- Sharpe = mean / std with ddof = 1 (sample std), non-annualised. A series with zero std has
  Sharpe 0 (a trial with no trades is flat, not infinitely good or bad).

SPEC NOTEs (most literal reading; listed in the GATE 3 tooling report)
- SR std and V use ddof = 1 (COMMON 10.7 says "mean / std" and "variance" without a ddof).
- g is numpy's Euler-Mascheroni constant 0.5772156649...; COMMON 10.7 prints it as 0.5772.
- Skew and kurtosis are the population (biased) moments, kurtosis NON-excess, as in the
  Bailey-Lopez de Prado PSR formula.
- N = 1 (raw or eff): SR* = 0, so DSR = PSR against 0 (COMMON 10.7). V is then not needed.
- N_eff: a pair whose correlation is undefined (a zero-variance series) gets correlation 0, so
  flat trials never merge with anything; fcluster merges at distance <= 1 - 0.7.
- Medoid = cluster member with the smallest summed distance (1 - rho) to the other members;
  ties go to the lowest column index (trial-log order).
- PBO: in-sample ties go to the lowest column index; out-of-sample ranks of ties are averaged;
  ties are detected with a relative tolerance of 1e-12 so BLAS rounding does not break them.
  N = 1 has no selection, so PBO is undefined (NaN) and reported as such.
- Regime bootstrap: the bucket SEQUENCE of the observed trades is kept and every trade slot is
  refilled from its own bucket (with replacement), so bucket counts are exact and drawdown is
  measured in the original regime order. Labels other than ACTIVE / CALM form their own buckets.
- Worst losing streak counts consecutive trades with $ P&L < 0; a zero P&L trade ends a streak.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm

# COMMON 10.5 / 10.7 frozen values.
SEED = 12345
N_RESAMPLES = 10_000
MEAN_BLOCK = 5.0
N_PERMUTATIONS = 10_000
QUANTILES = (5.0, 50.0, 95.0)
CLUSTER_CORR = 0.7
PBO_BLOCKS = 16
PBO_FLAG = 0.2          # 0.2 <= PBO < 0.5 is flagged (COMMON 10.5.8)
PBO_FAIL = 0.5
SHARPE_DDOF = 1
EULER_GAMMA = float(np.euler_gamma)
_TIE_RTOL = 1e-12


# --- small helpers -------------------------------------------------------------------------
@dataclass(frozen=True)
class Quantiles:
    p5: float
    p50: float
    p95: float

    @classmethod
    def of(cls, x: np.ndarray) -> "Quantiles":
        q = np.percentile(np.asarray(x, dtype="float64"), QUANTILES)
        return cls(float(q[0]), float(q[1]), float(q[2]))


def _as_1d(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype="float64").ravel()
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or inf")
    return a


def max_drawdown(pnl: np.ndarray, axis: int = -1) -> np.ndarray | float:
    """Peak-to-trough $ drawdown of the cumulative path of `pnl` increments along `axis`,
    starting at 0 (positive number; 0 when the path never falls below a prior peak)."""
    x = np.asarray(pnl, dtype="float64")
    x = np.moveaxis(x, axis, -1)
    cum = np.cumsum(x, axis=-1)
    peak = np.maximum.accumulate(np.maximum(cum, 0.0), axis=-1)
    dd = (peak - cum).max(axis=-1, initial=0.0)
    return float(dd) if np.ndim(dd) == 0 else dd


def worst_losing_streak(pnl: np.ndarray, axis: int = -1) -> np.ndarray | int:
    """Longest run of consecutive trades with $ P&L < 0 along `axis`."""
    lose = np.moveaxis(np.asarray(pnl, dtype="float64"), axis, -1) < 0
    c = np.cumsum(lose, axis=-1)
    reset = np.where(~lose, c, 0)
    np.maximum.accumulate(reset, axis=-1, out=reset)
    run = (c - reset).max(axis=-1, initial=0)
    return int(run) if np.ndim(run) == 0 else run


def sharpe(x: np.ndarray, axis: int = 0) -> np.ndarray | float:
    """Non-annualised mean / std (ddof = 1) along `axis`; 0 where std is 0 or undefined."""
    a = np.asarray(x, dtype="float64")
    if a.shape[axis] < 2:
        raise ValueError("Sharpe needs at least 2 observations")
    m = a.mean(axis=axis)
    s = a.std(axis=axis, ddof=SHARPE_DDOF)
    out = np.divide(m, s, out=np.zeros_like(np.asarray(m, dtype="float64")),
                    where=np.asarray(s) > 0)
    return float(out) if np.ndim(out) == 0 else out


# --- 10.5.6 bootstrap -------------------------------------------------------------------------
@dataclass(frozen=True)
class BootstrapResult:
    """Distribution of total $ P&L and max drawdown over resamples (COMMON 10.5.6)."""
    method: str                      # 'stationary_block' | 'regime_conditional'
    n_resamples: int
    seed: int
    n_obs: int                       # days (stationary) or trades (regime)
    mean_block: float | None         # stationary only
    buckets: dict[str, int] | None   # regime only: label -> count (kept in every resample)
    observed_total: float
    observed_max_dd: float
    total_pnl: Quantiles
    max_drawdown: Quantiles
    totals: np.ndarray = field(repr=False, compare=False)
    drawdowns: np.ndarray = field(repr=False, compare=False)


def stationary_bootstrap_indices(n: int, n_resamples: int, mean_block: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano (1994) stationary bootstrap index matrix (n_resamples x n).

    Draw order (fixed, so a seed reproduces the matrix): U = rng.random((n_resamples, n)), then
    starts = rng.integers(0, n, (n_resamples, n)). Position t opens a new block when t == 0 or
    U[t] < 1 / mean_block, at index starts[t]; otherwise it continues the block with the next
    index, wrapping circularly. Block lengths are geometric with mean `mean_block`."""
    if n < 1:
        raise ValueError("bootstrap needs at least one observation")
    if mean_block < 1:
        raise ValueError("mean block length must be >= 1")
    new = rng.random((n_resamples, n)) < 1.0 / mean_block
    new[:, 0] = True
    starts = rng.integers(0, n, size=(n_resamples, n))
    pos = np.arange(n)
    last = np.where(new, pos, 0)
    np.maximum.accumulate(last, axis=1, out=last)
    first = np.take_along_axis(starts, last, axis=1)
    return (first + (pos - last)) % n


def stationary_bootstrap(daily_pnl, n_resamples: int = N_RESAMPLES, mean_block: float = MEAN_BLOCK,
                         seed: int = SEED) -> BootstrapResult:
    """COMMON 10.5.6: stationary block bootstrap of the daily $ series (end-of-day MTM equity
    change on the 10.2 calendar). Reports 5 / 50 / 95% of total $ P&L and of max drawdown."""
    x = _as_1d(daily_pnl, "daily_pnl")
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(len(x), n_resamples, mean_block, rng)
    sample = x[idx]
    totals = sample.sum(axis=1)
    dds = max_drawdown(sample, axis=1)
    return BootstrapResult("stationary_block", n_resamples, seed, len(x), float(mean_block), None,
                           float(x.sum()), float(max_drawdown(x)), Quantiles.of(totals),
                           Quantiles.of(dds), totals, dds)


def regime_bootstrap(trade_pnl, regimes, n_resamples: int = N_RESAMPLES,
                     seed: int = SEED) -> BootstrapResult:
    """COMMON 10.5.6 regime-conditional bootstrap: `trade_pnl` and `regimes` in trade order;
    each trade slot is redrawn (with replacement) from the trades of its own regime bucket, so
    every resample keeps the observed bucket counts and bucket sequence. Buckets are processed
    in sorted label order, one rng.integers((n_resamples, count)) draw each."""
    x = _as_1d(trade_pnl, "trade_pnl")
    labels = np.asarray(regimes, dtype=object).ravel()
    if len(labels) != len(x):
        raise ValueError("trade_pnl and regimes differ in length")
    if len(x) == 0:
        raise ValueError("bootstrap needs at least one trade")
    labels = np.array([str(v) for v in labels], dtype=object)
    rng = np.random.default_rng(seed)
    sample = np.empty((n_resamples, len(x)))
    buckets: dict[str, int] = {}
    for b in sorted(set(labels)):
        pos = np.flatnonzero(labels == b)
        buckets[b] = int(len(pos))
        draw = rng.integers(0, len(pos), size=(n_resamples, len(pos)))
        sample[:, pos] = x[pos][draw]
    totals = sample.sum(axis=1)
    dds = max_drawdown(sample, axis=1)
    return BootstrapResult("regime_conditional", n_resamples, seed, len(x), None, buckets,
                           float(x.sum()), float(max_drawdown(x)), Quantiles.of(totals),
                           Quantiles.of(dds), totals, dds)


# --- 10.5.10 shuffle test ---------------------------------------------------------------------
@dataclass(frozen=True)
class ShuffleResult:
    """COMMON 10.5.10 (report only): sequence risk of the observed trades."""
    n_permutations: int
    seed: int
    n_trades: int
    observed_worst_streak: int
    observed_max_dd: float
    worst_streak: Quantiles
    max_drawdown: Quantiles
    streaks: np.ndarray = field(repr=False, compare=False)
    drawdowns: np.ndarray = field(repr=False, compare=False)

    @property
    def p95_worst_streak(self) -> float:
        return self.worst_streak.p95

    @property
    def p95_max_drawdown(self) -> float:
        return self.max_drawdown.p95


def shuffle_test(trade_pnl, n_permutations: int = N_PERMUTATIONS,
                 seed: int = SEED) -> ShuffleResult:
    """Permute the trade order `n_permutations` times (rng.permuted on a tiled matrix, axis 1);
    total P&L is invariant, so only the worst losing streak and the max drawdown are reported."""
    x = _as_1d(trade_pnl, "trade_pnl")
    rng = np.random.default_rng(seed)
    perms = rng.permuted(np.tile(x, (n_permutations, 1)), axis=1)
    streaks = worst_losing_streak(perms, axis=1)
    dds = max_drawdown(perms, axis=1)
    return ShuffleResult(n_permutations, seed, len(x), int(worst_losing_streak(x)),
                         float(max_drawdown(x)), Quantiles.of(streaks), Quantiles.of(dds),
                         np.asarray(streaks), np.asarray(dds))


# --- 10.7 Deflated Sharpe -----------------------------------------------------------------------
@dataclass(frozen=True)
class DSRResult:
    label: str              # 'N_raw' | 'N_eff' | free text
    n_trials: int
    sr_variance: float      # V (NaN when N = 1: not used)
    sr_hat: float           # non-annualised daily Sharpe of the selected strategy
    sr_star: float          # expected max Sharpe under the null (0 when N = 1)
    t_days: int
    skew: float
    kurtosis: float         # NON-excess
    dsr: float

    @property
    def passes(self) -> bool:
        """COMMON 10.5.8: DSR >= 0.95."""
        return bool(self.dsr >= 0.95)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """SR* = sqrt(V) * ((1 - g) * PhiInv(1 - 1/N) + g * PhiInv(1 - 1/(N e))); 0 for N = 1."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    if not np.isfinite(sr_variance) or sr_variance < 0:
        raise ValueError(f"SR variance must be finite and >= 0, got {sr_variance}")
    g = EULER_GAMMA
    z = (1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * math.e))
    return float(math.sqrt(sr_variance) * z)


def deflated_sharpe(daily, n_trials: int, sr_variance: float, label: str = "") -> DSRResult:
    """COMMON 10.7: DSR = Phi((SR_hat - SR*) * sqrt(T - 1) /
    sqrt(1 - skew * SR_hat + (kurt - 1) / 4 * SR_hat^2)) on the selected strategy's daily series.
    N = 1 gives the PSR against 0."""
    x = _as_1d(daily, "daily")
    t = len(x)
    if t < 2:
        raise ValueError("DSR needs at least 2 daily observations")
    sr = float(sharpe(x))
    c = x - x.mean()
    m2 = float(np.mean(c ** 2))
    if m2 > 0:
        skew = float(np.mean(c ** 3) / m2 ** 1.5)
        kurt = float(np.mean(c ** 4) / m2 ** 2)
    else:                               # flat series: SR_hat = 0, moments irrelevant
        skew, kurt = 0.0, 3.0
    sr_star = expected_max_sharpe(n_trials, sr_variance)
    denom_sq = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    # kurt >= 1 + skew^2 for any distribution, so denom_sq >= 0; guard rounding only.
    denom = math.sqrt(max(denom_sq, 1e-300))
    dsr = float(norm.cdf((sr - sr_star) * math.sqrt(t - 1) / denom))
    v = float(sr_variance) if n_trials > 1 else float("nan")
    return DSRResult(label, int(n_trials), v, sr, sr_star, t, skew, kurt, dsr)


@dataclass(frozen=True)
class EffectiveTrials:
    """N_eff by average-linkage clustering of trial daily series at correlation 0.7."""
    n_raw: int
    n_eff: int
    labels: np.ndarray        # cluster id (1..n_eff) per trial column
    medoids: np.ndarray       # column index of each cluster's medoid, sorted by cluster id
    sharpes: np.ndarray       # per trial column
    v_raw: float              # var(SR) across all trials (ddof 1; NaN if N_raw = 1)
    v_eff: float              # var(SR) across medoids (ddof 1; NaN if N_eff = 1)


def _corr_distance(matrix: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.corrcoef(matrix, rowvar=False)
    rho = np.atleast_2d(rho)
    rho = np.where(np.isfinite(rho), rho, 0.0)
    rho = np.clip((rho + rho.T) / 2, -1.0, 1.0)
    np.fill_diagonal(rho, 1.0)
    d = 1.0 - rho
    np.fill_diagonal(d, 0.0)
    return d


def effective_trials(trial_matrix, corr_threshold: float = CLUSTER_CORR) -> EffectiveTrials:
    """`trial_matrix` is T_days x N (one column per trial, common 10.2 calendar, zero on days
    without a position). Distance = 1 - rho, average linkage, flat clusters cut at 1 - 0.7."""
    m = np.asarray(trial_matrix, dtype="float64")
    if m.ndim != 2 or m.shape[1] < 1 or m.shape[0] < 2:
        raise ValueError("trial_matrix must be T_days x N with T_days >= 2, N >= 1")
    if not np.all(np.isfinite(m)):
        raise ValueError("trial_matrix contains NaN or inf")
    n = m.shape[1]
    srs = np.atleast_1d(sharpe(m, axis=0))
    if n == 1:
        return EffectiveTrials(1, 1, np.array([1]), np.array([0]), srs, float("nan"),
                               float("nan"))
    d = _corr_distance(m)
    z = linkage(squareform(d, checks=False), method="average")
    labels = fcluster(z, t=1.0 - corr_threshold, criterion="distance")
    medoids = []
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        cost = d[np.ix_(members, members)].sum(axis=1)
        medoids.append(int(members[int(np.argmin(cost))]))
    medoids_a = np.asarray(medoids)
    n_eff = len(medoids_a)
    v_raw = float(np.var(srs, ddof=1))
    v_eff = float(np.var(srs[medoids_a], ddof=1)) if n_eff > 1 else float("nan")
    return EffectiveTrials(n, n_eff, labels, medoids_a, srs, v_raw, v_eff)


@dataclass(frozen=True)
class DSRReport:
    raw: DSRResult
    eff: DSRResult
    trials: EffectiveTrials

    @property
    def passes(self) -> bool:
        """COMMON 10.5.8: DSR >= 0.95 under both N_raw and N_eff."""
        return self.raw.passes and self.eff.passes


def dsr_report(selected_daily, trial_matrix, n_raw: int | None = None) -> DSRReport:
    """DSR of the selected strategy under N_raw (V across all trials) and N_eff (V across
    cluster medoids). `n_raw` defaults to the number of trial columns.
    SPEC NOTE: a larger `n_raw` (trials that ran but left no daily series, e.g. a crash) may be
    passed; V is still computed from the available columns."""
    trials = effective_trials(trial_matrix)
    n_r = trials.n_raw if n_raw is None else int(n_raw)
    if n_r < trials.n_raw:
        raise ValueError(f"n_raw {n_r} is below the {trials.n_raw} trial columns")
    v_raw = trials.v_raw
    if n_r > 1 and not np.isfinite(v_raw):
        raise ValueError("N_raw > 1 needs at least two trial series to estimate V")
    if n_r == 1:
        v_raw = 0.0
    raw = deflated_sharpe(selected_daily, n_r, v_raw, "N_raw")
    eff = deflated_sharpe(selected_daily, trials.n_eff,
                          trials.v_eff if trials.n_eff > 1 else 0.0, "N_eff")
    return DSRReport(raw, eff, trials)


# --- 10.7 PBO by CSCV -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PBOResult:
    pbo: float                     # NaN when N = 1 (no selection to overfit)
    n_trials: int
    n_blocks: int
    n_splits: int
    t_used: int                    # rows after dropping the earliest remainder
    dropped_days: int
    block_len: int
    logits: np.ndarray = field(repr=False, compare=False)
    selected: np.ndarray = field(repr=False, compare=False)   # IS-best column per split
    note: str = ""

    @property
    def status(self) -> str:
        """COMMON 10.5.8: PBO < 0.5 passes; 0.2 <= PBO < 0.5 is flagged."""
        if not np.isfinite(self.pbo):
            return "UNDEFINED"
        if self.pbo >= PBO_FAIL:
            return "FAIL"
        return "FLAG" if self.pbo >= PBO_FLAG else "PASS"


def _close(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b) <= _TIE_RTOL * np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))


def _split_sharpe(s1: np.ndarray, d: np.ndarray, q: np.ndarray, n: int) -> np.ndarray:
    """Sharpe from sums: s1 = sum x, d = sum (x - K), q = sum (x - K)^2 (shift K per column
    keeps the variance numerically stable)."""
    mean = s1 / n
    var = np.maximum((q - d * d / n) / (n - SHARPE_DDOF), 0.0)
    sd = np.sqrt(var)
    return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)


def pbo_cscv(trial_matrix, n_blocks: int = PBO_BLOCKS) -> PBOResult:
    """COMMON 10.7 PBO by CSCV over the T_days x N trial matrix: drop the earliest T mod S
    rows, cut S contiguous blocks, and for every choice of S/2 blocks as in-sample select the
    column with the best IS Sharpe; w = its OOS rank / (N + 1) (rank 1 = worst), logit =
    ln(w / (1 - w)); PBO = share of logits <= 0."""
    m = np.asarray(trial_matrix, dtype="float64")
    if m.ndim != 2:
        raise ValueError("trial_matrix must be 2-D (T_days x N)")
    if not np.all(np.isfinite(m)):
        raise ValueError("trial_matrix contains NaN or inf")
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even number >= 2")
    t, n = m.shape
    drop = t % n_blocks
    block_len = t // n_blocks
    n_splits = math.comb(n_blocks, n_blocks // 2)
    if n == 1:
        return PBOResult(float("nan"), 1, n_blocks, n_splits, t - drop, drop, block_len,
                         np.array([]), np.array([], dtype=int),
                         "N = 1: no selection, PBO undefined")
    if block_len < 1 or (block_len * n_blocks // 2) < 2:
        raise ValueError(f"{t} rows are too few for {n_blocks} blocks")
    x = m[drop:]
    k = x.mean(axis=0)
    xb = x.reshape(n_blocks, block_len, n)
    s1 = xb.sum(axis=1)
    d = (xb - k).sum(axis=1)
    q = ((xb - k) ** 2).sum(axis=1)
    combos = np.array(list(itertools.combinations(range(n_blocks), n_blocks // 2)))
    ind = np.zeros((len(combos), n_blocks))
    ind[np.arange(len(combos))[:, None], combos] = 1.0
    n_half = block_len * n_blocks // 2
    is_sr = _split_sharpe(ind @ s1, ind @ d, ind @ q, n_half)
    oos_ind = 1.0 - ind
    oos_sr = _split_sharpe(oos_ind @ s1, oos_ind @ d, oos_ind @ q, n_half)
    # In-sample selection; ties -> lowest column index.
    best_val = is_sr.max(axis=1, keepdims=True)
    tied = _close(is_sr, best_val)
    best = np.argmax(tied, axis=1)
    rows = np.arange(len(combos))
    oos_best = oos_sr[rows, best][:, None]
    eq = _close(oos_sr, oos_best)
    less = (oos_sr < oos_best) & ~eq
    rank = less.sum(axis=1) + (eq.sum(axis=1) + 1) / 2.0     # average rank, 1 = worst
    w = rank / (n + 1)
    logits = np.log(w / (1 - w))
    pbo = float(np.mean(logits <= 0))
    return PBOResult(pbo, n, n_blocks, len(combos), len(x), drop, block_len, logits, best)
