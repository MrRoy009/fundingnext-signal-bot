"""Regime-model observations (COMMON 6.1): core H1 slots and the inputs x1, x2, x3.

The three inputs are computed on the full H1 series of one index (all hours, causal, within
segment) and then sampled at core H1 bars. Every value at a bar closing at T uses only H1 bars
with close <= T and, for the "previous 20 valid trading dates" sets, only earlier local dates.

Valid trading dates (COMMON 1.6) need the cash-open M15 bar, so the M15 frame is an input too.
The cash-open M15 bar closes before the first core H1 bar of its date closes (09:45 vs 10:00
New York, 09:15 vs 10:00 Berlin), so using it to classify today's core slots is causal.

All constants come from config.yaml (`regime`, `features`, `data_quality`, `exchanges`).
"""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from core.calendar import exchange_for, is_calendar_trading_date, local_to_utc, to_local
from core.config import tf_minutes
from data.segments import segment_ids, tv_bad

INPUTS = ["x1", "x2", "x3"]

# Emission kinds of a core slot in the forward filter (COMMON 6.6).
FULL, MARGINAL, NONE = "full", "marg", "none"


def prepare(bars: pd.DataFrame) -> pd.DataFrame:
    """Sorted, unique, ns-resolution copy. Stored parquet times are microsecond resolution;
    searchsorted on mixed resolutions compares raw integers, so everything is ns."""
    df = bars.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True).astype("datetime64[ns, UTC]")
    df = df.sort_values("time_utc", kind="stable").reset_index(drop=True)
    if df["time_utc"].duplicated().any():
        raise ValueError("Duplicate bar open times; the store should have removed them")
    return df


def ns(times) -> np.ndarray:
    """int64 nanoseconds since the epoch (UTC) for tz-aware times. A tz-aware Series'
    to_numpy() is an object array, which searchsorted cannot compare with datetime64."""
    return pd.DatetimeIndex(times).asi8


def _date_ord(local: pd.Series) -> np.ndarray:
    """Day number of each local date (sortable integer)."""
    return local.dt.tz_localize(None).dt.normalize().to_numpy().astype("datetime64[D]").astype("int64")


# ---------------------------------------------------------------------------------------------
# Valid trading dates (COMMON 1.6)
# ---------------------------------------------------------------------------------------------

def valid_dates(key: str, cfg: dict, m15: pd.DataFrame | None,
                h1: pd.DataFrame | None = None, *, h1_proxy: bool = False) -> pd.Series:
    """Bool per local date: weekday, not a full holiday, not an early close, and the cash-open
    M15 bar exists (COMMON 1.6).

    M15 is required. Without it a different set of dates would be valid and the regime model
    would silently change (QA: the H1 proxy marked 1,015 valid US30 dates against 874 from M15).
    SPEC NOTE: `h1_proxy=True` (unit tests only, never the services) replaces the cash-open M15
    check by "an H1 bar whose local open hour is the cash-open hour exists"."""
    if m15 is None and not h1_proxy:
        raise ValueError("valid trading dates need the cash-open M15 bar (COMMON 1.6); "
                         "pass the M15 frame (h1_proxy=True is for unit tests only)")
    ex = exchange_for(cfg, key)
    if m15 is not None and len(m15):
        f = prepare(m15)
        local = to_local(f["time_utc"], ex.tz)
        is_open = (local.dt.hour == ex.cash_open.hour) & (local.dt.minute == ex.cash_open.minute)
    elif m15 is None and h1 is not None and len(h1):
        f = prepare(h1)
        local = to_local(f["time_utc"], ex.tz)
        is_open = local.dt.hour == ex.cash_open.hour
    else:
        return pd.Series(dtype=bool)
    cash_open = is_open.groupby(local.dt.date).any()
    cal = np.array([is_calendar_trading_date(ex.calendar, d) for d in cash_open.index], dtype=bool)
    return (cash_open & cal).rename("valid")


# ---------------------------------------------------------------------------------------------
# Inputs x1, x2, x3 (COMMON 6.1)
# ---------------------------------------------------------------------------------------------

def _same_hour_median(df: pd.DataFrame, values: np.ndarray, ref_date: pd.Series,
                      n_dates: int, min_obs: int) -> np.ndarray:
    """Per bar: median of `values` over the bars with the same local open hour on the previous
    `n_dates` reference dates of the same segment (current date excluded), NaN if fewer than
    `min_obs` such bars.

    SPEC NOTE: COMMON 6.1 reads "on the previous 20 valid trading dates (>= 10 required)"
    (unlike TODV it does not add "that have such a bar"), so the date window is the last 20
    reference dates and the >= 10 count is over the bars found inside it."""
    out = np.full(len(df), np.nan)
    is_ref = df["local_date"].map(ref_date).eq(True).to_numpy()
    seg = df["segment"].to_numpy()
    dord = df["date_ord"].to_numpy()
    hour = df["local_hour"].to_numpy()
    for s in np.unique(seg):
        in_seg = np.flatnonzero(seg == s)
        ref_days = np.unique(dord[in_seg][is_ref[in_seg]])          # sorted reference dates
        if len(ref_days) == 0:
            continue
        for h in np.unique(hour[in_seg]):
            q = in_seg[hour[in_seg] == h]                          # time order = date order
            r = q[is_ref[q] & np.isfinite(values[q])]
            rd, rv = dord[r], values[r]
            k = np.searchsorted(ref_days, dord[q], side="left")     # ref dates before the bar's date
            lo = ref_days[np.maximum(k - n_dates, 0)]
            a = np.searchsorted(rd, lo, side="left")
            b = np.searchsorted(rd, dord[q], side="left")
            for qi, ai, bi, ki in zip(q, a, b, k):
                if ki > 0 and bi - ai >= min_obs:
                    out[qi] = np.median(rv[ai:bi])
    return out


def compute_inputs(h1: pd.DataFrame, key: str, cfg: dict, m15: pd.DataFrame | None,
                   *, h1_proxy: bool = False) -> pd.DataFrame:
    """Per H1 bar: time_utc, close_utc, local_date, local_hour, date_ord, segment, tv_bad,
    valid_date, tr, x1, x2, x3, plus OHLC and tick_volume (float)."""
    df = prepare(h1)
    ex = exchange_for(cfg, key)
    rc, fc = cfg["regime"], cfg["features"]
    bar_min = tf_minutes(cfg["timeframes"]["regime"])
    n_dates, min_obs = int(fc["tod_lookback_dates"]), int(fc["tod_min_obs"])
    n_d = int(rc["daily_cycle_bars"][key])

    local = to_local(df["time_utc"], ex.tz)
    df["close_utc"] = df["time_utc"] + pd.Timedelta(minutes=bar_min)
    df["local_date"] = local.dt.date
    df["local_hour"] = local.dt.hour.astype("int64")
    df["date_ord"] = _date_ord(local)
    df["segment"] = segment_ids(df["time_utc"], cfg["data_quality"]["segment_break_days"])
    df["tv_bad"] = tv_bad(df["time_utc"], cfg).to_numpy()
    df["tick_volume"] = df["tick_volume"].astype("float64")      # uint64 in the store

    vd = valid_dates(key, cfg, m15, df, h1_proxy=h1_proxy)
    df["valid_date"] = df["local_date"].map(vd).eq(True).to_numpy()

    # TR with C_prev from the same segment; the first bar of a segment uses H - L (COMMON 3).
    same_seg = df["segment"].eq(df["segment"].shift(1))
    prev_close = df["close"].shift(1).where(same_seg)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["tr"] = tr

    # x1: return over the same-hour median TR of the previous valid dates.
    s_t = _same_hour_median(df, tr.to_numpy(dtype=float), vd, n_dates, min_obs)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["x1"] = (df["close"] - prev_close).to_numpy() / s_t

    # x2: log ratio of the short (one daily cycle) to the long (20 cycles) mean TR, by
    # position, inside the segment; NaN until the long window is full.
    g = tr.groupby(df["segment"])
    short = g.transform(lambda s: s.rolling(n_d, min_periods=n_d).mean())
    long_ = g.transform(lambda s: s.rolling(20 * n_d, min_periods=20 * n_d).mean())
    with np.errstate(divide="ignore", invalid="ignore"):
        df["x2"] = np.log(short / long_)

    # x3: relative tick volume vs the same hour on the previous valid, non-flagged dates.
    # SPEC NOTE: a "flagged trading date" (COMMON 1.5) is a local date with at least one
    # tv_bad H1 bar; the corrupt window starts and ends at 00:00 UTC, so the local dates that
    # straddle a boundary are flagged too.
    flagged = df.groupby("local_date")["tv_bad"].any()
    ref_tv = vd.reindex(flagged.index).eq(True) & ~flagged
    tv = df["tick_volume"].to_numpy(dtype=float)
    tv_ref_vals = np.where(df["tv_bad"].to_numpy(), np.nan, tv)
    med = _same_hour_median(df, tv_ref_vals, ref_tv, n_dates, min_obs)
    with np.errstate(divide="ignore", invalid="ignore"):
        x3 = np.log((tv + 1.0) / (med + 1.0))
    df["x3"] = np.where(df["tv_bad"].to_numpy(), np.nan, x3)

    for c in INPUTS:     # +-inf (e.g. a zero median TR) is not a usable value
        df[c] = df[c].where(np.isfinite(df[c]))
    df.attrs["valid_dates"] = vd
    return df


# ---------------------------------------------------------------------------------------------
# Core slots (COMMON 6.1, 6.6)
# ---------------------------------------------------------------------------------------------

def core_slots(inputs: pd.DataFrame, key: str, cfg: dict,
               as_of: pd.Timestamp | None = None, vd: pd.Series | None = None) -> pd.DataFrame:
    """The filter's time grid: one row per core H1 slot, sorted by close time.

    SPEC NOTE: the grid is (a) every H1 bar whose exchange-local open hour is a core hour, on
    any date, plus (b) every core hour of a valid trading date that has no H1 bar. A slot is
    `usable` (a core observation) only if its bar exists, its date is valid and H > L; every
    other slot is a prediction-only step. Core hours of dates without any bars that are not
    valid trading dates (weekends, closed holidays, the data hole) are not steps at all.
    Absent slots closing after `as_of` (default: last H1 bar close) are not created.

    Columns: open_utc, close_utc, local_date, local_hour, segment, present, valid_date,
    usable, bar_missing (valid date, no bar: the live rule of 6.6), tv_bad, x1, x2, x3, kind.
    """
    ex = exchange_for(cfg, key)
    h_first, h_last = (int(h) for h in cfg["regime"]["core_hours_local"][key])
    bar_min = tf_minutes(cfg["timeframes"]["regime"])
    if as_of is None:
        as_of = inputs["close_utc"].max() if len(inputs) else None

    core = inputs[inputs["local_hour"].between(h_first, h_last)]
    present = pd.DataFrame({
        "open_utc": core["time_utc"], "close_utc": core["close_utc"],
        "local_date": core["local_date"], "local_hour": core["local_hour"],
        "segment": core["segment"], "present": True, "valid_date": core["valid_date"],
        "hl_ok": core["high"] > core["low"], "tv_bad": core["tv_bad"],
        "x1": core["x1"], "x2": core["x2"], "x3": core["x3"],
    })

    # Expected core slots on valid trading dates that have no bar.
    if vd is None:
        vd = inputs.attrs.get("valid_dates")
    if vd is None:
        vd = inputs.groupby("local_date")["valid_date"].first()
    first_open = inputs["time_utc"].min() if len(inputs) else None
    have = set(zip(present["local_date"], present["local_hour"]))
    rows = []
    for d in vd.index[vd.to_numpy(dtype=bool)]:
        for h in range(h_first, h_last + 1):
            if (d, h) in have:
                continue
            o = local_to_utc(d, time(h), ex.tz).as_unit("ns")
            c = o + pd.Timedelta(minutes=bar_min)
            if (as_of is not None and c > as_of) or (first_open is not None and o < first_open):
                continue
            rows.append((o, c, d, h))
    if rows:
        o, c, d, h = map(list, zip(*rows))
        pos = np.searchsorted(ns(inputs["time_utc"]), ns(pd.DatetimeIndex(o)), side="right") - 1
        seg_vals = inputs["segment"].to_numpy()
        seg = np.where(pos >= 0, seg_vals[np.maximum(pos, 0)], seg_vals[0] if len(seg_vals) else 0)
        absent = pd.DataFrame({
            "open_utc": pd.DatetimeIndex(o), "close_utc": pd.DatetimeIndex(c),
            "local_date": d, "local_hour": h, "segment": seg, "present": False,
            "valid_date": True, "hl_ok": False, "tv_bad": False,
            "x1": np.nan, "x2": np.nan, "x3": np.nan,
        })
        slots = pd.concat([present, absent], ignore_index=True)
    else:
        slots = present.reset_index(drop=True)

    slots = slots.sort_values("close_utc", kind="stable").reset_index(drop=True)
    slots["present"] = slots["present"].astype(bool)
    slots["valid_date"] = slots["valid_date"].astype(bool)
    slots["usable"] = slots["present"] & slots["valid_date"] & slots["hl_ok"].astype(bool)
    slots["bar_missing"] = slots["valid_date"] & ~slots["present"]
    f1, f2, f3 = (np.isfinite(slots[c].to_numpy(dtype=float)) for c in INPUTS)
    use = slots["usable"].to_numpy()
    slots["kind"] = np.where(use & f1 & f2 & f3, FULL,
                             np.where(use & f1 & f2, MARGINAL, NONE))
    return slots.drop(columns="hl_ok")


def build_slots(h1: pd.DataFrame, key: str, cfg: dict, m15: pd.DataFrame | None,
                as_of: pd.Timestamp | None = None, *, h1_proxy: bool = False) -> pd.DataFrame:
    """compute_inputs + core_slots in one call."""
    return core_slots(compute_inputs(h1, key, cfg, m15, h1_proxy=h1_proxy), key, cfg,
                      as_of=as_of)
