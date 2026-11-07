"""RegimeService (COMMON 6, ER-06, ER-07): refit schedule, model persistence, refit log, the
batch (backtest) table and the live snapshot.

Batch path: `run_batch(key, h1, m15)` processes a full H1 history once and returns a
`RegimeTable`, a time-indexed table of filter outputs. Each row is valid from its
`valid_from` time (a core observation's close, or a refit time); a decision at T uses the latest
row with valid_from <= T, then applies the staleness, model-age and missing-bar rules (the
missing-bar rule is on by default, exactly as live).

SPEC NOTE: COMMON 6.6 "alpha after the latest core observation with close <= T" is read
literally: rows are emitted only at usable core observations (COMMON 6.1). Prediction-only
steps for missing core bars or core bars on invalid dates still propagate alpha inside the
filter (they change the next observation's alpha) but never become an output row themselves.
Staleness is measured from that same latest core observation.

Live path: `live(key, h1, m15, T)` truncates the bars to close <= T and runs the same code over
the refits that can serve T, so live and backtest outputs are identical by construction. M15 is
required on both paths: valid trading dates (COMMON 1.6) need the cash-open M15 bar, so M15
decides which core slots are observations. The live store must therefore hold M15 (and H1) back
to `history_needed(key, T)`: about 62 + 31 + 365 days plus 20 valid dates, i.e. roughly 490
calendar days. ER-06's "60 trading days of M15" is not enough for the regime; with less, live
returns UNAVAILABLE ("insufficient M15 history for valid trading dates") instead of a different
model. Fits
are cached on disk (regime/models/), keyed by refit time, settings hash and training-data
SHA-256, so the live loop refits once per refit time and reloads afterwards.

Schedule and failure rules (COMMON 6.3, 6.7):
- R_m = first Saturday of each month 12:00 UTC; the model fitted at R_m serves [R_m, R_{m+1}).
- Fewer than min_train_rows training rows: no model, UNAVAILABLE until the next R.
- Refit failure (every K rejected / every restart failed): the previous model keeps serving while
  it is <= max_model_age_days old at T; a new fit is tried at each later core H1 close (with that
  close as the refit time) until one succeeds or the next R arrives.

Warm-up (COMMON 12B, amendment A2): after a segment reset (6.6) x2 is NaN for 20 * n_d bars, so
the filter only predicts from the stationary distribution. Every output row from the reset up to
the first core observation with an emission (x1 and x2 finite, full or marginal) is UNAVAILABLE
("regime warm-up"); from that observation on 6.6 applies unchanged. The flag is set in `_rows`,
which both the batch table and the live path go through, so they agree by construction.
"""
from __future__ import annotations

import csv
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from core.calendar import exchange_for, is_calendar_trading_date, local_to_utc
from core.config import ROOT, tf_minutes
from regime.hmm_fit import FitRecord, RegimeModel, filter_start, fit_regime, settings_hash, \
    training_set, window_filter, window_sha256, window_warmup
from regime.inputs import build_slots, ns

log = logging.getLogger(__name__)

MODEL_DIR = ROOT / "regime" / "models"
REFIT_LOG = ROOT / "regime" / "refits.csv"
REFIT_FIELDS = ["logged_utc", "key", "fit_time_utc", "status", "cause", "n_rows", "n_sequences",
                "train_sha256", "window_sha256", "model_hash", "k", "seed", "per_k", "states",
                "transmat"]
OK, UNAVAILABLE = "OK", "UNAVAILABLE"
REFIT_RULE = "first_saturday_12utc"     # the only schedule implemented (cfg regime.refit)
NO_M15 = "insufficient M15 history for valid trading dates"
WARMUP = "regime warm-up"               # COMMON 12B (A2): after a reset, before the first emission


def n_state_columns(cfg: dict) -> int:
    """Number of p0..p{n-1} alpha columns: the largest K the config allows."""
    return max(int(k) for k in cfg["regime"]["k_set"])


# ---------------------------------------------------------------------------------------------
# Refit schedule (COMMON 6.3)
# ---------------------------------------------------------------------------------------------

def first_saturday_noon(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1, hour=12, tz="UTC")
    return d + pd.Timedelta(days=(5 - d.weekday()) % 7)


def refit_schedule(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Every first-Saturday-12:00-UTC refit time R with start <= R <= end."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    y, m = start.year, start.month
    while True:
        r = first_saturday_noon(y, m)
        if r > end:
            return out
        if r >= start:
            out.append(r)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def latest_refit_at_or_before(t: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(t)
    r = first_saturday_noon(t.year, t.month)
    if r <= t:
        return r
    y, m = (t.year - 1, 12) if t.month == 1 else (t.year, t.month - 1)
    return first_saturday_noon(y, m)


# ---------------------------------------------------------------------------------------------
# Outputs (COMMON 6.7)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeSnapshot:
    """Regime output for one index at decision time T."""
    key: str
    time_utc: pd.Timestamp
    status: str                      # OK | UNAVAILABLE
    cause: str = ""
    v_t: float = float("nan")        # sum_s alpha_T(s) * mu_vol_s
    tag: str = ""                    # ACTIVE | CALM (sign of V_T)
    label: str = ""                  # label of the argmax state, e.g. CALM_FLAT
    regime_prob: float = float("nan")
    k: int = 0
    alpha: tuple = ()
    model_id: str = ""
    obs_close: pd.Timestamp | None = None
    threshold: float = 0.0           # gate threshold from cfg regime.gate_threshold
    obs_kind: str = ""               # emission of the latest observation: full | marg | none

    @property
    def ok(self) -> bool:
        return self.status == OK

    @property
    def regime(self) -> str:
        """Candidate.regime: '<ACTIVE|CALM>/<argmax label>' (COMMON 6.7)."""
        return f"{self.tag}/{self.label}" if self.ok else "UNAVAILABLE"

    @property
    def gate(self) -> str:
        """Gate state for the regime history log (signals.run.record_regime)."""
        return self.tag if self.ok else UNAVAILABLE

    @property
    def passes_active(self) -> bool:
        return self.ok and bool(self.v_t >= self.threshold)

    @property
    def passes_calm(self) -> bool:
        return self.ok and bool(self.v_t < self.threshold)

    @property
    def reason(self) -> str:
        """Log text for an ineligible index (COMMON 6.7)."""
        return "" if self.ok else f"regime unavailable: {self.cause}"


def passes_gate(snap: RegimeSnapshot, gate: str) -> bool:
    """gate 'ACTIVE' or 'CALM' (the strategy's regime gate, COMMON 6.7)."""
    if gate == "ACTIVE":
        return snap.passes_active
    if gate == "CALM":
        return snap.passes_calm
    raise ValueError(f"Unknown regime gate {gate!r}")


# ---------------------------------------------------------------------------------------------
# Batch table
# ---------------------------------------------------------------------------------------------

@dataclass
class Interval:
    """One stretch of the timeline: a model serving [start, end), or UNAVAILABLE with a cause."""
    start: pd.Timestamp
    end: pd.Timestamp | None
    model: RegimeModel | None
    cause: str = ""
    fallback: bool = False


def v_t(alpha: np.ndarray, m: RegimeModel) -> np.ndarray:
    """V_T = sum_s alpha(s) * mu_vol_s per row, summed state by state in a fixed order.

    Elementwise on purpose: a BLAS matrix-vector product may round differently depending on how
    many rows it is given, so a truncated (live) replay could differ from the full batch in the
    last bit, and a V_T next to 0 could flip the gate (COMMON 9, ER-15)."""
    v = np.zeros(len(alpha))
    for s in range(m.k):
        v = v + alpha[:, s] * m.mu_vol[s]
    return v


def _p_active(alpha: np.ndarray, m: RegimeModel) -> np.ndarray:
    """Sum of alpha over ACTIVE-tagged states, fixed order (same reason as v_t)."""
    p = np.zeros(len(alpha))
    for s in range(m.k):
        if m.vol_tags[s] == "ACTIVE":
            p = p + alpha[:, s]
    return p


def _vt_columns(alpha: np.ndarray, m: RegimeModel, n_states: int) -> dict:
    v = v_t(alpha, m)
    arg = alpha.argmax(axis=1)
    tag = np.where(v >= 0, "ACTIVE", "CALM")
    p_active = _p_active(alpha, m)
    out = {"v_t": v, "tag": tag, "label": np.array(m.labels, dtype=object)[arg],
           "regime_prob": np.where(tag == "ACTIVE", p_active, 1.0 - p_active)}
    for s in range(n_states):
        out[f"p{s}"] = alpha[:, s] if s < alpha.shape[1] else np.full(len(alpha), np.nan)
    return out


@dataclass
class RegimeTable:
    key: str
    frame: pd.DataFrame                 # sorted by valid_from
    slots: pd.DataFrame
    intervals: list[Interval]
    records: list[FitRecord]
    threshold: float
    stale: pd.Timedelta
    max_age: pd.Timedelta
    n_states: int = 4                   # p0..p{n-1} columns (max of cfg regime.k_set)
    _vf: np.ndarray = field(init=False, repr=False)
    _miss: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._vf = ns(self.frame["valid_from"])
        # Close times of core slots on valid dates whose H1 bar is absent (live rule, 6.6).
        self._miss = ns(self.slots.loc[self.slots["bar_missing"], "close_utc"])

    def at_many(self, times, live_rule: bool = True) -> pd.DataFrame:
        """Regime output at each decision time T (vectorised `at`). Columns: time_utc, status,
        cause, v_t, tag, label, regime_prob, k, model_id, obs_close, kind, p0..p{n-1}.

        live_rule (default True) applies the live rule of COMMON 6.6: if the core H1 bar
        closing exactly at T is absent, UNAVAILABLE ("regime H1 bar missing"). Decisions, live
        and backtest, must keep the default so both treat a missing bar identically (COMMON 9);
        False is for diagnostics only.

        SPEC NOTE: staleness is "T - close of the latest core observation > stale_days". The
        clause "and no reset applies" cannot change the result: a segment reset is applied
        together with the next core observation (6.6), which then becomes the latest one; from
        that observation until the first one with an emission the row itself is already
        UNAVAILABLE ("regime warm-up", COMMON 12B), which no later rule here overrides."""
        ts = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit("ns")
        if not len(self.frame):
            return pd.DataFrame({
                "time_utc": ts, "status": UNAVAILABLE, "cause": "no regime model yet",
                "model_id": "", "fit_time": pd.NaT, "obs_close": pd.NaT, "k": 0,
                "fallback": False, "kind": "", "v_t": np.nan, "tag": "", "label": "",
                "regime_prob": np.nan, **{f"p{s}": np.nan for s in range(self.n_states)}})
        idx = np.searchsorted(self._vf, ts.asi8, side="right") - 1
        f = self.frame
        rows = f.iloc[np.maximum(idx, 0)].reset_index(drop=True)
        out = rows.drop(columns=["valid_from"]).copy()
        out.insert(0, "time_utc", ts)
        status = out["status"].to_numpy(dtype=object).copy()
        cause = out["cause"].to_numpy(dtype=object).copy()
        t = ts.asi8
        fit = ns(out["fit_time"])
        obs = ns(out["obs_close"])
        fallback = out["fallback"].to_numpy(dtype=bool)
        is_ok = status == OK
        nat = np.iinfo("int64").min
        # Model age only binds for a previous model kept after a refit failure (6.7).
        too_old = is_ok & fallback & (fit != nat) & (t - fit > self.max_age.value)
        missing = is_ok & bool(live_rule) & np.isin(t, self._miss)
        stale = is_ok & (obs != nat) & (t - obs > self.stale.value)
        for mask, why in ((stale, "stale: last core H1 observation more than "
                                  f"{self.stale.days} days before T"),
                          (missing, "regime H1 bar missing"),
                          (too_old, "refit failed and previous model is older than "
                                    f"{self.max_age.days} days")):
            status[mask], cause[mask] = UNAVAILABLE, why
        none = idx < 0
        status[none], cause[none] = UNAVAILABLE, "no regime model yet"
        out["status"], out["cause"] = status, cause
        bad = out["status"] != OK
        out.loc[bad, ["v_t", "regime_prob"]] = np.nan
        out.loc[bad, ["tag", "label"]] = ""
        return out

    def at(self, t, live_rule: bool = True) -> RegimeSnapshot:
        """Regime output at decision time T; see at_many for live_rule."""
        r = self.at_many([t], live_rule).iloc[0]
        ok = r["status"] == OK
        k = int(r["k"]) if ok else 0
        obs = r["obs_close"]
        return RegimeSnapshot(
            key=self.key, time_utc=r["time_utc"], status=r["status"], cause=r["cause"] or "",
            v_t=float(r["v_t"]) if ok else float("nan"), tag=r["tag"] if ok else "",
            label=r["label"] if ok else "",
            regime_prob=float(r["regime_prob"]) if ok else float("nan"),
            k=k, alpha=tuple(float(r[f"p{s}"]) for s in range(k)),
            model_id=r["model_id"] or "", obs_close=None if pd.isna(obs) else obs,
            threshold=self.threshold, obs_kind=(r["kind"] or "") if ok else "")


# ---------------------------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------------------------

class RegimeService:
    """Per-index regime models. One instance can serve the backtest and the live loop."""

    def __init__(self, cfg: dict, model_dir: Path | None = MODEL_DIR,
                 refit_log: Path | None = REFIT_LOG, n_jobs: int = 1):
        self.cfg = cfg
        self.rc = cfg["regime"]
        if self.rc.get("refit") != REFIT_RULE:
            raise ValueError(f"regime.refit must be {REFIT_RULE!r} (COMMON 6.3); "
                             f"got {self.rc.get('refit')!r}")
        self.n_states = n_state_columns(cfg)
        self.model_dir = Path(model_dir) if model_dir else None
        self.refit_log = Path(refit_log) if refit_log else None
        self.n_jobs = n_jobs
        self.tables: dict[str, RegimeTable] = {}
        self._cache: dict[tuple, FitRecord] = {}
        self._logged: set[tuple] | None = None
        self._warned_h1: set[str] = set()

    @property
    def keys(self) -> list[str]:
        return list(self.rc["core_hours_local"])

    # --- fitting with cache, persistence and refit log -----------------------------------
    def _model_path(self, key: str, fit_time: pd.Timestamp, shash: str) -> Path | None:
        if self.model_dir is None:
            return None
        return self.model_dir / key / f"{key}_{fit_time:%Y%m%dT%H%M%S}_{shash[:10]}.pkl"

    def fit(self, key: str, slots: pd.DataFrame, fit_time: pd.Timestamp, seeds=None,
            log_refit: bool = True) -> FitRecord:
        """Refit attempt at fit_time; reuses a persisted fit only if the settings, the
        training-data SHA-256 and the filter-window SHA-256 all match exactly."""
        seeds = tuple(int(s) for s in (self.rc["seeds"] if seeds is None else seeds))
        shash = settings_hash(self.cfg, seeds)
        ts = training_set(slots, fit_time, self.cfg)
        sha, wsha = ts.sha256, window_sha256(slots, ts.start, fit_time)
        ck = (key, pd.Timestamp(fit_time), shash)
        rec = self._cache.get(ck)
        if rec is None or rec.train_sha256 != sha or rec.window_sha256 != wsha:
            rec = self._load(key, fit_time, shash, sha, wsha)
        if rec is None:
            rec = fit_regime(key, slots, fit_time, self.cfg, seeds, self.n_jobs)
            if rec.status == "OK":
                log.info("Regime refit %s at %s: K=%d, %d rows, hash %s", key, fit_time,
                         rec.model.k, rec.n_rows, rec.model.model_hash[:12])
            elif rec.status == "FAILED":
                log.warning("Regime refit %s at %s failed: %s", key, fit_time, rec.cause)
            self._save(rec)
        self._cache[ck] = rec
        if log_refit:
            self._log(rec)
        return rec

    def _load(self, key, fit_time, shash, sha, wsha) -> FitRecord | None:
        p = self._model_path(key, pd.Timestamp(fit_time), shash)
        if p is None or not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                rec = pickle.load(f)
        except Exception as exc:  # noqa: BLE001 - a corrupt cache file is refitted
            log.warning("Ignoring unreadable regime model %s: %s", p, exc)
            return None
        # Records saved before the window hash existed read "" here and are refitted.
        if (rec.train_sha256 != sha or rec.settings_hash != shash
                or getattr(rec, "window_sha256", "") != wsha):
            return None
        return rec

    def _save(self, rec: FitRecord) -> None:
        if rec.status == "INSUFFICIENT":
            return                      # nothing fitted; cheap to recompute
        p = self._model_path(rec.key, rec.fit_time, rec.settings_hash)
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)

    def _log(self, rec: FitRecord) -> None:
        """One refits.csv row per attempt (COMMON 6.4, 6.5); idempotent across reruns."""
        if self.refit_log is None:
            return
        if self._logged is None:
            self._logged = set()
            if self.refit_log.exists():
                with open(self.refit_log, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    old = list(reader)
                    header = reader.fieldnames or []
                if header != REFIT_FIELDS:
                    self._migrate_log(old)
                self._logged = {(r["key"], r["fit_time_utc"], r["train_sha256"],
                                 r.get("window_sha256") or "") for r in old}
        ident = (rec.key, rec.fit_time.isoformat(), rec.train_sha256, rec.window_sha256)
        if ident in self._logged:
            return
        m = rec.model
        states = [] if m is None else [
            {"state": s, "label": m.labels[s], "mu_vol": m.mu_vol[s], "mu_ret": m.mu_ret[s],
             "sd_ret": m.sd_ret[s], "occ": m.occ[s], "se": m.se[s]} for s in range(m.k)]
        row = {
            "logged_utc": pd.Timestamp.now(tz="UTC").isoformat(), "key": rec.key,
            "fit_time_utc": rec.fit_time.isoformat(), "status": rec.status, "cause": rec.cause,
            "n_rows": rec.n_rows, "n_sequences": rec.n_sequences,
            "train_sha256": rec.train_sha256, "window_sha256": rec.window_sha256,
            "model_hash": m.model_hash if m else "",
            "k": m.k if m else "", "seed": m.seed if m else "",
            "per_k": json.dumps(rec.per_k, default=_json_default),
            "states": json.dumps(states, default=_json_default),
            "transmat": json.dumps(m.transmat.tolist() if m else []),
        }
        new = not self.refit_log.exists()
        self.refit_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.refit_log, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=REFIT_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        self._logged.add(ident)

    def _migrate_log(self, rows: list[dict]) -> None:
        """Rewrite refits.csv under the current header (older rows get "" in new columns), so
        appended rows stay aligned with the header."""
        tmp = self.refit_log.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=REFIT_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") or "" for c in REFIT_FIELDS})
        tmp.replace(self.refit_log)

    # --- timeline -------------------------------------------------------------------------
    def _timeline(self, key: str, slots: pd.DataFrame, as_of: pd.Timestamp,
                  first_refit: pd.Timestamp | None) -> tuple[list[Interval], list[FitRecord]]:
        close = slots["close_utc"]
        if close.empty:
            return [], []
        start = close.iloc[0] if first_refit is None else max(first_refit, close.iloc[0])
        sched = refit_schedule(start, as_of)
        usable_close = ns(close[slots["usable"]])
        events: list[Interval] = []
        records: list[FitRecord] = []
        last_ok: RegimeModel | None = None
        for i, r in enumerate(sched):
            nxt = sched[i + 1] if i + 1 < len(sched) else None
            rec = self.fit(key, slots, r)
            records.append(rec)
            if rec.status == "OK":
                events.append(Interval(r, None, rec.model))
                last_ok = rec.model
                continue
            if rec.status == "INSUFFICIENT":
                events.append(Interval(r, None, None, rec.cause))
                continue
            # Refit failure: previous model (age checked at T), retry at later core closes.
            events.append(Interval(r, None, last_ok, "" if last_ok else
                                   f"{rec.cause}; no previous model", fallback=True))
            lo = np.searchsorted(usable_close, r.value, side="right")
            hi = (np.searchsorted(usable_close, nxt.value, side="left") if nxt is not None
                  else np.searchsorted(usable_close, as_of.value, side="right"))
            for c in usable_close[lo:hi]:
                c = pd.Timestamp(int(c), tz="UTC")
                retry = self.fit(key, slots, c)
                records.append(retry)
                if retry.status == "OK":
                    events.append(Interval(c, None, retry.model))
                    last_ok = retry.model
                    break
        for a, b in zip(events, events[1:]):
            a.end = b.start
        return events, records

    def _rows(self, slots: pd.DataFrame, iv: Interval, as_of: pd.Timestamp) -> pd.DataFrame:
        close = ns(slots["close_utc"])
        if iv.model is None:
            return pd.DataFrame({"valid_from": [iv.start], "status": [UNAVAILABLE],
                                 "cause": [iv.cause], "model_id": [""], "fit_time": [pd.NaT],
                                 "obs_close": [pd.NaT], "slot": [-1], "kind": [""],
                                 "x2": [np.nan], "k": [0], "fallback": [iv.fallback]})
        m = iv.model
        end = iv.end if iv.end is not None else as_of + pd.Timedelta(microseconds=1)
        # Re-run from the training-window start through every slot before the refit, then
        # observation by observation until the next refit (COMMON 6.6).
        lo = filter_start(slots, m.train_start)
        s_pos = int(np.searchsorted(close, iv.start.value, side="left"))
        hi = int(np.searchsorted(close, end.value, side="left"))
        if hi <= lo:
            return pd.DataFrame()
        alpha, _ = window_filter(slots, lo, hi, m.startprob, m.transmat, m.means, m.covars,
                                 m.prep, m.model_id)
        obs = np.flatnonzero(slots["usable"].to_numpy(dtype=bool)[lo:hi]) + lo
        before, after = obs[obs < s_pos], obs[obs >= s_pos]
        pos = np.concatenate([before[-1:], after])   # last pre-refit observation, then the interval
        if not len(pos):
            return pd.DataFrame()
        a = alpha[pos - lo]
        part = slots.iloc[pos]
        vf = close[pos].copy()
        if len(before):
            vf[0] = iv.start.value                       # its alpha becomes valid at the refit
        # COMMON 12B (A2): after a segment reset the output is the stationary prior until the
        # first emitting observation, so those rows are UNAVAILABLE ("regime warm-up"). The
        # alpha columns stay for inspection; V_T, tag, label and regime_prob are blanked.
        warm = window_warmup(slots, lo, hi)[pos - lo]
        cols = _vt_columns(a, m, self.n_states)
        if warm.any():
            cols["v_t"] = np.where(warm, np.nan, cols["v_t"])
            cols["regime_prob"] = np.where(warm, np.nan, cols["regime_prob"])
            cols["tag"] = np.where(warm, "", cols["tag"]).astype(object)
            cols["label"] = np.where(warm, "", cols["label"]).astype(object)
        return pd.DataFrame({
            "valid_from": pd.to_datetime(vf, utc=True),
            "status": np.where(warm, UNAVAILABLE, OK).astype(object),
            "cause": np.where(warm, WARMUP, "").astype(object),
            "model_id": m.model_id, "fit_time": m.fit_time,
            "obs_close": pd.to_datetime(close[pos], utc=True), "slot": pos,
            "kind": part["kind"].to_numpy(), "x2": part["x2"].to_numpy(), "k": m.k,
            "fallback": iv.fallback, **cols})

    # --- public paths ---------------------------------------------------------------------
    def run_batch(self, key: str, h1: pd.DataFrame, m15: pd.DataFrame, *,
                  as_of: pd.Timestamp | None = None,
                  first_refit: pd.Timestamp | None = None, keep: bool = True) -> RegimeTable:
        """Process a full H1 history once (backtest path). M15 is required (valid trading
        dates, COMMON 1.6). `as_of` defaults to the last bar close; `first_refit` skips earlier
        scheduled refits (used by the live path); `keep` stores the table for `snapshot`."""
        if m15 is None:
            raise ValueError(f"{key}: the regime needs M15 bars for valid trading dates "
                             "(COMMON 1.6)")
        h1_min = tf_minutes(self.cfg["timeframes"]["regime"])
        if as_of is None:
            ends = [pd.to_datetime(h1["time_utc"], utc=True).max() + pd.Timedelta(minutes=h1_min)]
            if len(m15):
                e_min = tf_minutes(self.cfg["timeframes"]["entry"])
                ends.append(pd.to_datetime(m15["time_utc"], utc=True).max()
                            + pd.Timedelta(minutes=e_min))
            as_of = max(ends)
        as_of = pd.Timestamp(as_of).as_unit("ns")
        slots = build_slots(h1, key, self.cfg, m15, as_of=as_of)
        intervals, records = self._timeline(key, slots, as_of, first_refit)
        parts = [self._rows(slots, iv, as_of) for iv in intervals]
        parts = [p for p in parts if len(p)]
        frame = (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["valid_from", "status", "cause", "model_id", "fit_time", "obs_close",
                     "slot", "kind", "x2", "k", "fallback", "v_t", "tag", "label",
                     "regime_prob"] + [f"p{s}" for s in range(self.n_states)]))
        for c in ("valid_from", "fit_time", "obs_close"):
            frame[c] = pd.to_datetime(frame[c], utc=True).astype("datetime64[ns, UTC]")
        frame = frame.sort_values("valid_from", kind="stable").reset_index(drop=True)
        table = RegimeTable(key, frame, slots, intervals, records,
                            float(self.rc["gate_threshold"]),
                            pd.Timedelta(days=float(self.rc["stale_days"])),
                            pd.Timedelta(days=float(self.rc["max_model_age_days"])),
                            self.n_states)
        if keep:
            self.tables[key] = table
        return table

    def snapshot(self, key: str, t, live_rule: bool = True) -> RegimeSnapshot:
        """Output at T from the batch table built earlier (backtest decisions). Keep the
        default live_rule=True for decisions (see RegimeTable.at_many)."""
        if key not in self.tables:
            return RegimeSnapshot(key, pd.Timestamp(t), UNAVAILABLE, "no regime table for index",
                                  threshold=float(self.rc["gate_threshold"]))
        return self.tables[key].at(t, live_rule)

    def history_needed(self, key: str, t) -> pd.Timestamp:
        """Earliest bar time (UTC) the live regime at T reads: the training-window start of
        the oldest refit that can still serve T (the latest refit at or before
        T - max_model_age_days), minus tod_lookback_dates valid dates for the x1/x3 lookbacks.

        SPEC NOTE: the 20 lookback dates are counted on the exchange calendar (weekday, no
        holiday, no early close) because M15, which decides the rest of the valid-date rule, is
        what is being checked. x2's 20*n_d-bar window covers about the same 20 dates. Dates
        before the frozen calendar's first year count weekdays only (history check only)."""
        t = pd.Timestamp(t)
        first = latest_refit_at_or_before(
            t - pd.Timedelta(days=float(self.rc["max_model_age_days"])))
        start = first - pd.Timedelta(days=float(self.rc["train_days"]))
        ex = exchange_for(self.cfg, key)
        d = start.tz_convert(ex.tz).date()
        found, n = 0, int(self.cfg["features"]["tod_lookback_dates"])
        while found < n:
            d -= timedelta(days=1)
            found += _trading_date(ex.calendar, d)
        return local_to_utc(d, time(0), ex.tz)

    def _short_history(self, key: str, h1: pd.DataFrame, m15: pd.DataFrame, t) -> str:
        """'' if the live M15 reaches back far enough, else a description.

        SPEC NOTE (QA, COMMON 1.6 / 9): M15 must start on or before the first exchange-calendar
        trading date on or after max(history_needed, first H1 bar). The H1 start stands in for
        the start of the available data: the store updates both timeframes together, so M15
        shallower than H1 inside the needed window means truncated M15 (e.g. ER-06's 60-day
        minimum), which would silently change valid dates and the model. A store whose H1 is
        itself shorter than needed (the broker's history start) is logged once as a WARNING."""
        ex = exchange_for(self.cfg, key)
        need = self.history_needed(key, t)
        h1_first = pd.to_datetime(h1["time_utc"], utc=True).min()
        m15_first = pd.to_datetime(m15["time_utc"], utc=True).min()
        if h1_first > need and key not in self._warned_h1:
            self._warned_h1.add(key)
            log.warning("Regime %s: H1 starts %s, after the %s the regime needs at %s; the "
                        "result equals a full-history replay only if that is the data start",
                        key, h1_first, need, t)
        d = max(need, h1_first).tz_convert(ex.tz).date()
        for _ in range(14):                         # next exchange trading date on or after d
            if _trading_date(ex.calendar, d):
                break
            d += timedelta(days=1)
        m15_date = m15_first.tz_convert(ex.tz).date()
        if m15_date > d:
            return f"M15 starts {m15_date}, needs {d} or earlier"
        return ""

    def live(self, key: str, h1: pd.DataFrame, m15: pd.DataFrame | None,
             t: pd.Timestamp) -> RegimeSnapshot:
        """Live output at decision time T: bars with close <= T only; refits that can serve T
        (fitted within max_model_age_days) are fitted or loaded; then the missing-bar rule.
        Without M15 back to history_needed(key, T): UNAVAILABLE (NO_M15)."""
        t = pd.Timestamp(t).as_unit("ns")
        thr = float(self.rc["gate_threshold"])
        if h1 is None or not len(h1):
            return RegimeSnapshot(key, t, UNAVAILABLE, "no H1 bars", threshold=thr)
        h1_min = tf_minutes(self.cfg["timeframes"]["regime"])
        e_min = tf_minutes(self.cfg["timeframes"]["entry"])
        h = h1[pd.to_datetime(h1["time_utc"], utc=True) + pd.Timedelta(minutes=h1_min) <= t]
        m = (None if m15 is None else
             m15[pd.to_datetime(m15["time_utc"], utc=True) + pd.Timedelta(minutes=e_min) <= t])
        if not len(h):
            return RegimeSnapshot(key, t, UNAVAILABLE, "no H1 bars", threshold=thr)
        why = "no M15 bars" if m is None or not len(m) else self._short_history(key, h, m, t)
        if why:
            log.warning("Regime %s at %s: %s (%s)", key, t, NO_M15, why)
            return RegimeSnapshot(key, t, UNAVAILABLE, NO_M15, threshold=thr)
        first = latest_refit_at_or_before(
            t - pd.Timedelta(days=float(self.rc["max_model_age_days"])))
        table = self.run_batch(key, h, m, as_of=t, first_refit=first, keep=False)
        return table.at(t, live_rule=True)

    def live_all(self, h1_bars: dict[str, pd.DataFrame], m15_bars: dict[str, pd.DataFrame],
                 t: pd.Timestamp) -> dict[str, RegimeSnapshot]:
        """{key: snapshot} for every regime index present in h1_bars (StrategyContext.regime).
        An index without M15 bars is UNAVAILABLE (NO_M15), never an H1-only guess."""
        return {k: self.live(k, h1_bars[k], m15_bars.get(k), t)
                for k in self.keys if k in h1_bars}


def _trading_date(calendar: str, d) -> bool:
    """Exchange-calendar trading date; outside the frozen calendar's years, weekdays only
    (used for the history-depth check, never for the model's valid dates)."""
    try:
        return is_calendar_trading_date(calendar, d)
    except ValueError:
        return d.weekday() < 5


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
