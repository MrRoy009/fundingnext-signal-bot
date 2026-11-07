"""Hand-written log-space forward filter (COMMON 6.6).

The filter never calls hmmlearn's predict_proba, decode, score_samples or Viterbi: those use
future observations (smoothing) or a different start rule. alpha_t depends only on
observations up to t, which is what makes the regime output causal.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import multivariate_normal

from regime.inputs import FULL, MARGINAL

log = logging.getLogger(__name__)


def log_emissions(z: np.ndarray, kind: np.ndarray, means: np.ndarray,
                  covars: np.ndarray) -> np.ndarray:
    """(T, K) log emission densities; rows without an emission are NaN.

    kind FULL: 3-D density of (x1, x2, x3). kind MARGINAL (x3 NaN, e.g. tv_bad): the 2-D
    marginal on (x1, x2), i.e. means[:2] and covars[:2, :2]. Anything else: no emission."""
    z = np.asarray(z, dtype=float)
    kind = np.asarray(kind)
    n_states = means.shape[0]
    out = np.full((len(z), n_states), np.nan)
    full, marg = kind == FULL, kind == MARGINAL
    for j in range(n_states):
        if full.any():
            out[full, j] = np.atleast_1d(multivariate_normal.logpdf(z[full], means[j], covars[j]))
        if marg.any():
            out[marg, j] = np.atleast_1d(multivariate_normal.logpdf(
                z[marg][:, :2], means[j][:2], covars[j][:2, :2]))
    return out


def stationary_distribution(transmat: np.ndarray, startprob: np.ndarray) -> np.ndarray:
    """Left eigenvector of A for eigenvalue 1, normalised to sum 1. If eigenvalue 1 is not
    simple, or its eigenvector is not a valid distribution, startprob is used (COMMON 6.6)."""
    vals, vecs = np.linalg.eig(np.asarray(transmat, dtype=float).T)
    ones = np.flatnonzero(np.abs(vals - 1.0) < 1e-8)
    if len(ones) != 1:
        return np.asarray(startprob, dtype=float)
    v = np.real(vecs[:, ones[0]])
    v = v / v.sum() if v.sum() != 0 else v
    if not np.all(np.isfinite(v)) or np.any(v < -1e-12):
        return np.asarray(startprob, dtype=float)
    v = np.clip(v, 0.0, None)
    return v / v.sum()


def warmup_steps(reset: np.ndarray, has_emission: np.ndarray) -> np.ndarray:
    """(T,) True at the steps that are in regime warm-up (COMMON 12B, amendment A2).

    After a segment reset the filter sits at the stationary distribution and only moves by
    prediction steps until an emission arrives (x2 needs 20 * n_d bars of the new segment), so
    its output is the prior, not an estimate. A step is in warm-up if a reset happened at or
    before it and no step from that reset up to and including it had an emission.

    SPEC NOTE: "until the first core observation after the reset that has an emission" is read
    as inclusive of the reset step itself: the reset is applied and then that step's emission
    (COMMON 6.6), so a reset step that has an emission is not in warm-up. The filter's own
    start (step 0, from startprob) is not a reset and never starts a warm-up.

    Vectorised with running maxima: index of the latest reset vs the latest emission."""
    reset = np.asarray(reset, dtype=bool)
    has_emission = np.asarray(has_emission, dtype=bool)
    idx = np.arange(len(reset))
    last_reset = np.maximum.accumulate(np.where(reset, idx, -1)) if len(idx) else idx
    last_emit = np.maximum.accumulate(np.where(has_emission, idx, -1)) if len(idx) else idx
    return (last_reset >= 0) & (last_emit < last_reset)


def _lse(x: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """log-sum-exp that returns -inf (not NaN) when every term is -inf."""
    m = np.max(x, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        s = np.log(np.sum(np.exp(x - m_safe), axis=axis, keepdims=True)) + m_safe
    return np.squeeze(s, axis=axis) if axis is not None else float(s.reshape(()))


def forward_filter(log_b: np.ndarray, startprob: np.ndarray, transmat: np.ndarray,
                   reset: np.ndarray | None = None, label: str = "") -> tuple[np.ndarray, list[int]]:
    """Normalised filtered state probabilities alpha_t for t = 0..T-1.

    log_b:  (T, K) log emissions; an all-NaN row is a prediction-only step.
    reset:  (T,) True at a segment start: alpha is reset to the stationary distribution of A
            and the step's emission is then applied. Row 0 always starts from startprob.
    Returns (alpha (T, K), indices of steps whose log-likelihood was non-finite; those were
    treated as prediction-only and logged as WARNING)."""
    log_b = np.asarray(log_b, dtype=float)
    n, k = log_b.shape
    with np.errstate(divide="ignore"):
        log_a = np.log(np.asarray(transmat, dtype=float))
        log_start = np.log(np.asarray(startprob, dtype=float))
        log_stat = np.log(stationary_distribution(transmat, startprob))
    reset = np.zeros(n, dtype=bool) if reset is None else np.asarray(reset, dtype=bool)
    has_b = ~np.all(np.isnan(log_b), axis=1)

    out = np.empty((n, k))
    bad: list[int] = []
    prev = None
    for t in range(n):
        if t == 0:
            pred = log_start
        elif reset[t]:
            pred = log_stat
        else:
            pred = _lse(prev[:, None] + log_a, axis=0)       # log (alpha_{t-1} A)
        pred = pred - _lse(pred)
        cur = pred
        if has_b[t]:
            s = log_b[t] + pred
            norm = _lse(s)
            if np.isfinite(norm) and np.all(~np.isnan(s)):
                cur = s - norm
            else:
                bad.append(t)
                log.warning("Regime filter %s: non-finite step log-likelihood at step %d; "
                            "treated as prediction-only", label, t)
        out[t] = np.exp(cur)
        prev = cur
    return out, bad
