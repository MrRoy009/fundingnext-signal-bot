"""Regime model fitting (COMMON 6.2-6.5): training set, preprocessing, GaussianHMM restarts,
degeneracy screens, BIC selection, canonical state order and relabelling.

Everything here is fitted on the training window only, [R - 365 days, R), and stored with the
model so the same clip bounds, means and stds are applied to every later observation until the
next refit.
"""
from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from regime.filter import forward_filter, log_emissions, warmup_steps
from regime.inputs import FULL, INPUTS, MARGINAL, ns

log = logging.getLogger(__name__)

FORMAT_VERSION = 1          # bump when the stored model layout or fitting code changes
D = len(INPUTS)             # d = 3 in the BIC parameter count


# ---------------------------------------------------------------------------------------------
# Preprocessing (COMMON 6.2)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Prep:
    lo: np.ndarray       # winsor bounds per input (training 0.5th / 99.5th percentiles)
    hi: np.ndarray
    mean: np.ndarray     # mean and std (ddof=0) of the clipped training values
    std: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Clip then z-score; NaN stays NaN (it marks a missing input for the filter)."""
        x = np.asarray(x, dtype=float)
        return (np.clip(x, self.lo, self.hi) - self.mean) / self.std


def fit_prep(x: np.ndarray, pct: tuple[float, float]) -> Prep:
    lo = np.quantile(x, pct[0] / 100.0, axis=0, method="linear")
    hi = np.quantile(x, pct[1] / 100.0, axis=0, method="linear")
    c = np.clip(x, lo, hi)
    return Prep(lo, hi, c.mean(axis=0), c.std(axis=0, ddof=0))


# ---------------------------------------------------------------------------------------------
# Training set (COMMON 6.3)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingSet:
    fit_time: pd.Timestamp
    start: pd.Timestamp            # fit_time - train_days
    rows: np.ndarray               # positions in the slots table
    x: np.ndarray                  # raw (x1, x2, x3), shape (n, 3)
    lengths: np.ndarray            # hmmlearn sequence lengths
    sha256: str


def training_set(slots: pd.DataFrame, fit_time: pd.Timestamp, cfg: dict) -> TrainingSet:
    """Core observations with close in [R - train_days, R) and x1, x2, x3 all finite (so tv_bad
    rows are out); split into sequences where consecutive rows are > sequence_break_days apart."""
    rc = cfg["regime"]
    fit_time = pd.Timestamp(fit_time)
    start = fit_time - pd.Timedelta(days=float(rc["train_days"]))
    close = slots["close_utc"]
    sel = (close >= start) & (close < fit_time) & (slots["kind"] == FULL)
    rows = np.flatnonzero(sel.to_numpy())
    x = slots.loc[sel, INPUTS].to_numpy(dtype=float)
    t = ns(close[sel])
    if len(t):
        brk = np.flatnonzero(np.diff(t) > pd.Timedelta(days=float(rc["sequence_break_days"])).value) + 1
        edges = np.concatenate([[0], brk, [len(t)]])
        lengths = np.diff(edges).astype("int64")
    else:
        lengths = np.zeros(0, dtype="int64")
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(t, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(lengths, dtype="<i8").tobytes())
    return TrainingSet(fit_time, start, rows, x, lengths, h.hexdigest())


# ---------------------------------------------------------------------------------------------
# Fitted model
# ---------------------------------------------------------------------------------------------

@dataclass
class RegimeModel:
    """One fitted, relabelled regime model; states are in ascending (mu_vol, mu_ret) order."""
    key: str
    fit_time: pd.Timestamp
    train_start: pd.Timestamp
    k: int
    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray            # z-score units
    covars: np.ndarray           # (K, 3, 3), z-score units
    prep: Prep
    mu_vol: np.ndarray           # per state, clipped (unstandardised) units (COMMON 6.5)
    mu_ret: np.ndarray
    sd_ret: np.ndarray
    occ: np.ndarray              # sum of filtered probabilities over training rows
    se: np.ndarray
    vol_tags: list[str]
    dir_tags: list[str]
    labels: list[str]
    n_rows: int
    n_sequences: int
    train_sha256: str
    model_hash: str = ""
    seed: int = -1               # restart chosen for the selected K

    @property
    def model_id(self) -> str:
        return f"{self.key}@{self.fit_time:%Y-%m-%dT%H:%M}Z#{self.model_hash[:12]}"

    def emissions(self, x: np.ndarray, kind: np.ndarray) -> np.ndarray:
        return log_emissions(self.prep.apply(x), kind, self.means, self.covars)


def model_hash(m: RegimeModel) -> str:
    """SHA-256 of everything that determines the model's outputs. A refit on identical data
    with identical settings reproduces it (ER-06)."""
    h = hashlib.sha256()
    h.update(f"{FORMAT_VERSION}|{m.key}|{m.k}|{','.join(m.labels)}".encode())
    for a in (m.startprob, m.transmat, m.means, m.covars, m.prep.lo, m.prep.hi, m.prep.mean,
              m.prep.std, m.mu_vol, m.mu_ret, m.sd_ret, m.occ):
        h.update(np.ascontiguousarray(a, dtype="<f8").tobytes())
    return h.hexdigest()


@dataclass
class FitRecord:
    """Outcome of one refit attempt at fit_time (also what refits.csv logs)."""
    key: str
    fit_time: pd.Timestamp
    status: str                      # OK | FAILED | INSUFFICIENT
    cause: str
    n_rows: int
    n_sequences: int
    train_sha256: str
    seeds: tuple[int, ...]
    per_k: dict[int, dict] = field(default_factory=dict)
    model: RegimeModel | None = None
    settings_hash: str = ""
    window_sha256: str = ""          # every filter-window slot the fit read (see window_sha256)


# ---------------------------------------------------------------------------------------------
# Fitting (COMMON 6.4)
# ---------------------------------------------------------------------------------------------

def settings_hash(cfg: dict, seeds) -> str:
    """Hash of every setting a fit depends on; a cached model is reused only on a match."""
    import hmmlearn
    import sklearn
    blob = json.dumps({"regime": cfg["regime"], "features": cfg["features"],
                       "data_quality": cfg["data_quality"], "seeds": list(seeds),
                       "hmmlearn": hmmlearn.__version__, "sklearn": sklearn.__version__,
                       "format": FORMAT_VERSION}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def single_thread():
    """BLAS/OpenMP limited to one thread. With several threads the k-means initialisation and
    the EM sums are reduced in a run-dependent order, so two fits on identical data differ in
    the last bits and the model hash would not reproduce (ER-06). threadpoolctl ships with the
    pinned scikit-learn (a required dependency of it), so nothing new is installed. Restarts
    still run in parallel across processes (joblib)."""
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=1)


def fit_restart(z: np.ndarray, lengths: np.ndarray, k: int, seed: int, rc: dict) -> dict:
    """One GaussianHMM restart. Never raises: a failure is returned as `error`."""
    from hmmlearn.hmm import GaussianHMM
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)   # "not converging" chatter
    out = {"k": k, "seed": seed, "lnl": float("nan"), "converged": False, "error": ""}
    try:
        with warnings.catch_warnings(), single_thread():
            warnings.simplefilter("ignore")
            m = GaussianHMM(n_components=k, covariance_type=rc["covariance_type"],
                            n_iter=int(rc["n_iter"]), tol=float(rc["tol"]),
                            min_covar=float(rc["min_covar"]), init_params="stmc",
                            params="stmc", random_state=seed)
            m.fit(z, lengths)
            lnl = float(m.score(z, lengths))
        out.update(lnl=lnl, converged=bool(m.monitor_.converged), n_iter=int(m.monitor_.iter),
                   startprob=np.array(m.startprob_, dtype=float),
                   transmat=np.array(m.transmat_, dtype=float),
                   means=np.array(m.means_, dtype=float), covars=np.array(m.covars_, dtype=float))
    except Exception as exc:  # noqa: BLE001 - any failed restart is just excluded (COMMON 6.4)
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def n_params(k: int, d: int = D) -> int:
    return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2


def _screen_params(r: dict, rc: dict) -> str:
    """'' if the covariances and transition matrix pass COMMON 6.4, else the reason."""
    for s, cov in enumerate(r["covars"]):
        try:
            np.linalg.cholesky(cov)
            ev = np.linalg.eigvalsh(cov)
        except np.linalg.LinAlgError:
            return f"state {s} covariance not positive definite"
        if not np.all(np.isfinite(ev)) or ev.min() <= 0:
            return f"state {s} covariance not positive definite"
        if ev.max() / ev.min() > float(rc["max_condition_number"]):
            return f"state {s} covariance condition number {ev.max() / ev.min():.3g}"
    dev = np.abs(r["transmat"].sum(axis=1) - 1.0)
    if not np.all(np.isfinite(dev)) or dev.max() > 1e-6:
        return "transmat row sum differs from 1 by more than 1e-6"
    return ""


def filter_start(slots: pd.DataFrame, train_start: pd.Timestamp) -> int:
    """Slot position where a model's filter starts (COMMON 6.6 "Start").

    SPEC NOTE: "the first core observation of the new model's training window" is read as the
    first usable core slot (core hour, valid date, H > L; COMMON 6.1) with close >= R - 365 d.
    Prediction-only slots before it (missing bars, invalid dates) are skipped, so alpha_0 is
    startprob times that observation's emission. If its inputs are NaN it has no emission and
    alpha_0 = startprob. Falls back to the first slot of the window if none is usable."""
    close = ns(slots["close_utc"])
    lo = int(np.searchsorted(close, pd.Timestamp(train_start).value, side="left"))
    use = np.flatnonzero(slots["usable"].to_numpy(dtype=bool)[lo:])
    return lo + int(use[0]) if len(use) else lo


def fit_window(slots: pd.DataFrame, train_start: pd.Timestamp,
               fit_time: pd.Timestamp) -> tuple[int, int]:
    """Slot range [lo, hi) a fit at fit_time filters over for the occupancy screen and the
    relabelling: from filter_start to the last slot closing before fit_time."""
    close = ns(slots["close_utc"])
    lo = filter_start(slots, train_start)
    hi = int(np.searchsorted(close, pd.Timestamp(fit_time).value, side="left"))
    return lo, hi


def window_sha256(slots: pd.DataFrame, train_start: pd.Timestamp,
                  fit_time: pd.Timestamp) -> str:
    """SHA-256 of every slot in the fit's filter window, not only the training rows.

    The training-rows hash misses slots that change the occupancy screen and the relabelling
    but are not training rows: MARGINAL (tv_bad) rows, absent core slots, bars on invalid dates,
    H = L bars and segment resets. A calendar fix or a backfill that touches only those must
    not reload a stale persisted model."""
    lo, hi = fit_window(slots, train_start, fit_time)
    part = slots.iloc[lo:max(lo, hi)]
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(ns(part["close_utc"]), dtype="<i8").tobytes())
    seg = part["segment"].to_numpy(dtype="int64")
    h.update(np.ascontiguousarray(seg, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(part["usable"].to_numpy(dtype=bool)).tobytes())
    h.update("|".join(part["kind"].astype(str)).encode())
    h.update(np.ascontiguousarray(part[INPUTS].to_numpy(dtype=float), dtype="<f8").tobytes())
    return h.hexdigest()


def window_resets(slots: pd.DataFrame, lo: int, hi: int) -> np.ndarray:
    """(hi - lo,) True at the slots of slots[lo:hi] where a segment reset applies (COMMON 6.6):
    the first slot of every segment after the window's first. The window's first slot is the
    filter start (startprob), not a reset."""
    seg = slots["segment"].to_numpy()[lo:hi]
    return np.r_[False, seg[1:] != seg[:-1]] if len(seg) else np.zeros(0, dtype=bool)


def window_warmup(slots: pd.DataFrame, lo: int, hi: int) -> np.ndarray:
    """(hi - lo,) True at the slots of slots[lo:hi] that are in regime warm-up (COMMON 12B):
    after a reset and before the first slot with an emission (kind full or marginal, i.e.
    x1 and x2 finite). Uses the same resets as window_filter, so the batch table and the live
    path, which both go through window_filter, see identical warm-up stretches."""
    kind = slots["kind"].to_numpy()[lo:hi]
    return warmup_steps(window_resets(slots, lo, hi), (kind == FULL) | (kind == MARGINAL))


def window_filter(slots: pd.DataFrame, lo: int, hi: int, startprob, transmat, means, covars,
                  prep: Prep, label: str = "") -> tuple[np.ndarray, list[int]]:
    """Run the COMMON 6.6 filter over slots[lo:hi] (lo = first slot of a training window)."""
    part = slots.iloc[lo:hi]
    z = prep.apply(part[INPUTS].to_numpy(dtype=float))
    with single_thread():          # bit-identical filter outputs between runs and processes
        lb = log_emissions(z, part["kind"].to_numpy(), means, covars)
    return forward_filter(lb, startprob, transmat, window_resets(slots, lo, hi), label)


def _canonical(r: dict, prep: Prep) -> tuple[dict, np.ndarray]:
    """Reorder states by ascending (mu_vol, mu_ret)."""
    mu_vol = r["means"][:, 1] * prep.std[1] + prep.mean[1]
    mu_ret = r["means"][:, 0] * prep.std[0] + prep.mean[0]
    order = np.lexsort((mu_ret, mu_vol))
    p = dict(r)
    p["startprob"] = r["startprob"][order]
    p["transmat"] = r["transmat"][np.ix_(order, order)]
    p["means"] = r["means"][order]
    p["covars"] = r["covars"][order]
    return p, order


def fit_regime(key: str, slots: pd.DataFrame, fit_time: pd.Timestamp, cfg: dict,
               seeds=None, n_jobs: int = 1) -> FitRecord:
    """Fit the regime model at refit time `fit_time` from slots with close < fit_time."""
    rc = cfg["regime"]
    seeds = tuple(int(s) for s in (rc["seeds"] if seeds is None else seeds))
    ts = training_set(slots, fit_time, cfg)
    rec = FitRecord(key, pd.Timestamp(fit_time), "FAILED", "", len(ts.x), len(ts.lengths),
                    ts.sha256, seeds, settings_hash=settings_hash(cfg, seeds),
                    window_sha256=window_sha256(slots, ts.start, fit_time))
    if len(ts.x) < int(rc["min_train_rows"]):
        rec.status, rec.cause = "INSUFFICIENT", (
            f"fewer than {int(rc['min_train_rows']):,} training rows ({len(ts.x)})")
        return rec

    prep = fit_prep(ts.x, tuple(rc["winsor_pct"]))
    z = prep.apply(ts.x)
    jobs = [(int(k), s) for k in rc["k_set"] for s in seeds]
    if n_jobs != 1 and len(jobs) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(delayed(fit_restart)(z, ts.lengths, k, s, rc)
                                          for k, s in jobs)
    else:
        results = [fit_restart(z, ts.lengths, k, s, rc) for k, s in jobs]

    # Filter window: first slot of the training window up to (excluding) fit_time.
    lo, hi = fit_window(slots, ts.start, fit_time)
    train_pos = ts.rows - lo

    best_by_k: dict[int, dict] = {}
    for k in (int(k) for k in rc["k_set"]):
        rs = [r for r in results if r["k"] == k]
        info = {"restart_lnl": {r["seed"]: (None if not np.isfinite(r["lnl"]) else r["lnl"])
                                for r in rs},
                "restart_converged": {r["seed"]: r["converged"] for r in rs},
                "restart_error": {r["seed"]: r["error"] for r in rs if r["error"]},
                "p": n_params(k), "lnl": None, "bic": None, "status": "", "reason": "",
                "seed": None}
        ok = [r for r in rs if not r["error"] and r["converged"] and np.isfinite(r["lnl"])]
        if not ok:
            info.update(status="REJECTED", reason="no converged restart")
            rec.per_k[k] = info
            continue
        best = max(ok, key=lambda r: r["lnl"])        # ties keep the lowest seed
        info.update(lnl=best["lnl"], seed=best["seed"],
                    bic=-2.0 * best["lnl"] + n_params(k) * np.log(len(ts.x)))
        reason = _screen_params(best, rc)
        if not reason:
            alpha, _ = window_filter(slots, lo, hi, best["startprob"], best["transmat"],
                                     best["means"], best["covars"], prep, f"{key} K={k} screen")
            occ = alpha[train_pos].sum(axis=0) / len(ts.x)
            info["occupancy"] = [float(o) for o in occ]
            if occ.min() < float(rc["min_state_occupancy"]):
                reason = f"state occupancy {occ.min():.4f} < {rc['min_state_occupancy']}"
        info.update(status="REJECTED" if reason else "OK", reason=reason)
        rec.per_k[k] = info
        if not reason:
            best_by_k[k] = best

    if not best_by_k:
        rec.cause = "refit failed: every K rejected or every restart failed"
        return rec
    ks = sorted(best_by_k)
    bics = {k: rec.per_k[k]["bic"] for k in ks}
    min_bic = min(bics.values())
    k_sel = min(k for k in ks if bics[k] - min_bic <= 1e-6)     # ties go to the smaller K

    params, _ = _canonical(best_by_k[k_sel], prep)
    alpha, _ = window_filter(slots, lo, hi, params["startprob"], params["transmat"],
                             params["means"], params["covars"], prep, f"{key} relabel")
    rec.model = relabel(key, ts, prep, params, alpha[train_pos], cfg)
    rec.model.seed = int(best_by_k[k_sel]["seed"])
    rec.model.model_hash = model_hash(rec.model)
    rec.status, rec.cause = "OK", ""
    return rec


def relabel(key: str, ts: TrainingSet, prep: Prep, params: dict, alpha_train: np.ndarray,
            cfg: dict) -> RegimeModel:
    """COMMON 6.5: per-state descriptive stats in clipped units and the state labels."""
    z_dir = float(cfg["regime"]["direction_z"])
    means, covars = params["means"], params["covars"]
    mu_vol = means[:, 1] * prep.std[1] + prep.mean[1]
    mu_ret = means[:, 0] * prep.std[0] + prep.mean[0]
    sd_ret = np.sqrt(covars[:, 0, 0]) * prep.std[0]
    occ = alpha_train.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = sd_ret / np.sqrt(occ)
    vol = ["ACTIVE" if v >= 0 else "CALM" for v in mu_vol]
    dirs = ["UP" if r > z_dir * s else "DOWN" if r < -z_dir * s else "FLAT"
            for r, s in zip(mu_ret, se)]
    return RegimeModel(
        key=key, fit_time=ts.fit_time, train_start=ts.start, k=len(mu_vol),
        startprob=params["startprob"], transmat=params["transmat"], means=means, covars=covars,
        prep=prep, mu_vol=mu_vol, mu_ret=mu_ret, sd_ret=sd_ret, occ=occ, se=se,
        vol_tags=vol, dir_tags=dirs, labels=[f"{v}_{d}" for v, d in zip(vol, dirs)],
        n_rows=len(ts.x), n_sequences=len(ts.lengths), train_sha256=ts.sha256)
