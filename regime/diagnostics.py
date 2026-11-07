"""Pre-registered regime diagnostics (COMMON 6.8), descriptive only.

Everything here describes the regime model and its gate on core observations. Nothing reads a
price after the observation or computes a return, profit or win rate of any rule; the numbers
are reported before any P&L is examined and nothing in the regime spec changes in response.

Per index and per refit (one row per scheduled or retry refit that produced a model):
- K and the model's state labels;
- label distribution: share of the core observations it served by argmax label, and the share
  ACTIVE (V_T >= threshold);
- label churn vs the previous model;
- share of core observations with V_T in [-0.05, 0.05];
- agreement between the gate decision and sign(x2) alone;
- seed stability: the same refit with seeds 10..19, share of the served core observations where
  the gate decision agrees with the main model.
The "share of in-window cycles where the strategy's gate blocks" needs the strategy's windows;
`gate_block_share` computes it from the decision times a strategy supplies.

SPEC NOTE (COMMON 12B): core observations in regime warm-up (after a segment reset, before the
first emission) have no regime output, so they are left out of every share above and counted
separately (`n_warmup_obs`); `summary` reports the warm-up share of all output rows.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from regime.hmm_fit import RegimeModel, filter_start, window_filter, window_warmup
from regime.inputs import ns
from regime.service import OK, WARMUP, RegimeService, RegimeTable, v_t

NEAR_ZERO = 0.05      # COMMON 6.8: share of core observations with V_T in [-0.05, 0.05]


def _served_positions(table: RegimeTable, m: RegimeModel, start: pd.Timestamp,
                      end: pd.Timestamp | None) -> np.ndarray:
    """Slot positions of the core observations a model served: close in [start, end)."""
    slots = table.slots
    close = ns(slots["close_utc"])
    use = slots["usable"].to_numpy(dtype=bool)
    lo = int(np.searchsorted(close, start.value, side="left"))
    hi = len(close) if end is None else int(np.searchsorted(close, end.value, side="left"))
    return np.flatnonzero(use[lo:hi]) + lo


def warmup_at(table: RegimeTable, m: RegimeModel, pos: np.ndarray) -> np.ndarray:
    """Warm-up flag (COMMON 12B) of model m's filter at slot positions `pos`, from the same
    window the service runs (training-window start onward)."""
    if not len(pos):
        return np.zeros(0, dtype=bool)
    lo = filter_start(table.slots, m.train_start)
    return window_warmup(table.slots, lo, int(pos.max()) + 1)[pos - lo]


def model_alpha(table: RegimeTable, m: RegimeModel, pos: np.ndarray) -> np.ndarray:
    """Filtered alpha of model m at slot positions `pos`, with the filter run from m's
    training-window start exactly as the service runs it (COMMON 6.6)."""
    if not len(pos):
        return np.zeros((0, m.k))
    lo = filter_start(table.slots, m.train_start)
    hi = int(pos.max()) + 1
    alpha, _ = window_filter(table.slots, lo, hi, m.startprob, m.transmat, m.means, m.covars,
                             m.prep, f"{m.model_id} diagnostics")
    return alpha[pos - lo]


def refit_diagnostics(service: RegimeService, table: RegimeTable,
                      stability_seeds=None) -> pd.DataFrame:
    """One row per refit that produced a model (COMMON 6.8). `stability_seeds` defaults to
    cfg regime.stability_seeds; pass () to skip the seed-stability refits (they cost as much
    as the main fits)."""
    rc = service.rc
    thr = float(rc["gate_threshold"])
    seeds = tuple(rc["stability_seeds"] if stability_seeds is None else stability_seeds)
    slots = table.slots
    x2 = slots["x2"].to_numpy(dtype=float)
    rows = []
    prev: RegimeModel | None = None
    for iv in table.intervals:
        m = iv.model
        if m is None or iv.fallback or m.fit_time != iv.start:
            continue                                    # no model, or a kept previous model
        pos_all = _served_positions(table, m, iv.start, iv.end)
        warm = warmup_at(table, m, pos_all)
        pos = pos_all[~warm]                            # observations with a regime output
        a = model_alpha(table, m, pos)
        v = v_t(a, m)
        labels = np.array(m.labels, dtype=object)[a.argmax(axis=1)] if len(a) else np.array([])
        gate = v >= thr
        n = len(pos)
        fin = np.isfinite(x2[pos])
        row = {
            "key": table.key, "fit_time_utc": m.fit_time, "serves_until_utc": iv.end,
            "k": m.k, "model_hash": m.model_hash[:12], "labels": json.dumps(m.labels),
            "n_core_obs": n, "n_warmup_obs": int(warm.sum()),
            "label_dist": json.dumps({lb: round(float(np.mean(labels == lb)), 4)
                                      for lb in sorted(set(labels))}) if n else "{}",
            "share_active": float(np.mean(gate)) if n else np.nan,
            "share_v_near_zero": float(np.mean(np.abs(v) <= NEAR_ZERO)) if n else np.nan,
            # SPEC NOTE: sign(x2) >= 0 counts as ACTIVE, mirroring the frozen gate convention
            # V_T >= 0 (x2's neutral point is 0); observations with NaN x2 are left out.
            "agree_sign_x2": (float(np.mean(gate[fin] == (x2[pos][fin] >= 0.0)))
                              if fin.any() else np.nan),
        }
        # SPEC NOTE: "label churn between consecutive refits" is measured on the same core
        # observations: the previous model's filter is continued over the new model's served
        # observations, and churn = share whose argmax label string differs. The label sets of
        # both models are logged as well.
        if prev is not None and n:
            ap = model_alpha(table, prev, pos)
            prev_labels = np.array(prev.labels, dtype=object)[ap.argmax(axis=1)]
            row["label_churn"] = float(np.mean(prev_labels != labels))
            row["prev_labels"] = json.dumps(prev.labels)
        else:
            row["label_churn"], row["prev_labels"] = np.nan, ""
        row["seed_stability"] = np.nan
        row["stability_k"] = np.nan
        if seeds and n:
            rec = service.fit(table.key, slots, m.fit_time, seeds=seeds, log_refit=False)
            if rec.status == "OK":
                alt = rec.model
                va = v_t(model_alpha(table, alt, pos), alt)
                row["seed_stability"] = float(np.mean((va >= thr) == gate))
                row["stability_k"] = alt.k
        rows.append(row)
        prev = m
    return pd.DataFrame(rows)


def gate_block_share(table: RegimeTable, times, gate: str, live_rule: bool = True) -> float:
    """Share of a strategy's in-window decision times where its regime gate blocks (status not
    OK, or the gate fails). COMMON 6.8 calls a gate non-functional below 5% or above 95%."""
    if gate not in ("ACTIVE", "CALM"):
        raise ValueError(f"Unknown regime gate {gate!r}")
    out = table.at_many(times, live_rule=live_rule)
    if not len(out):
        return float("nan")
    ok = out["status"].to_numpy() == OK
    v = out["v_t"].to_numpy(dtype=float)
    side = (v >= table.threshold) if gate == "ACTIVE" else (v < table.threshold)
    return float(1.0 - (ok & side).mean())


def summary(table: RegimeTable) -> dict:
    """First valid model, chosen K per refit and refit outcomes for one index (descriptive)."""
    recs = table.records
    ok = [r for r in recs if r.status == "OK"]
    f = table.frame
    served = f[(f["status"] == OK) & f["kind"].ne("")]
    with_model = f[f["model_id"].ne("")]                # rows of a model's core observations
    n_warm = int((with_model["cause"] == WARMUP).sum())
    return {
        "key": table.key,
        "first_valid_model": ok[0].fit_time if ok else None,
        "k_by_refit": {r.fit_time.isoformat(): (r.model.k if r.model else r.status) for r in recs},
        "n_refits": len(recs),
        "n_ok": len(ok),
        "n_insufficient": sum(r.status == "INSUFFICIENT" for r in recs),
        "n_failed": sum(r.status == "FAILED" for r in recs),
        "share_active_rows": (float((served["v_t"] >= table.threshold).mean())
                              if len(served) else float("nan")),
        "n_model_rows": len(with_model),
        "n_warmup_rows": n_warm,
        "share_warmup_rows": n_warm / len(with_model) if len(with_model) else float("nan"),
    }
