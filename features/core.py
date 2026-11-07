"""COMMON feature library (COMMON sections 2, 3 and 4.1): pure, causal, per index.

Every function here takes bar frames as stored by `data.store.BarStore` (`time_utc` = bar OPEN,
tz-aware UTC, sorted, unique) and returns values that, at a bar closing at T, use only bars with
close time <= T. Lookbacks never cross a segment start (COMMON 1.4) and "previous N dates" sets
use valid trading dates only (COMMON 1.6). All constants come from config.yaml (`features`,
`data_quality`, `exchanges`, `timeframes`); nothing is tuned here.

Public API (strategies and the backtester normally use `features.service` instead):

    compute_m15_features(m15, key, cfg) -> (m15_features: DataFrame, dates: DataFrame)
        Per M15 bar: time_utc, close_utc, local_time, local_date, local_min, segment, tv_bad,
        is_cash_open_bar, in_session, valid_date, tr, atr14, todv, rs_4h, twap, spread_pts,
        plus the input OHLC columns. Per local date (`dates`, indexed by local_date): segment,
        calendar_ok, cash_open_present, valid, n_session_bars, session_range, datr_ok, datr.
    compute_h1_features(h1, key, cfg) -> DataFrame
        Per H1 bar: time_utc, close_utc, local_time, local_date, local_min, segment, tv_bad,
        tr, atr14, rs_day, plus the input OHLC columns.
    ib_table(m15_features, key, cfg, ib_minutes) -> DataFrame indexed by local_date
        Columns ibh, ibl, w, defined_utc (close of the last IB bar), complete.
    h1_alignment(T, h1_close, m15_open, m15_minutes, m15_segment=None) -> (h1_pos, cause)
        COMMON 2 H1 rule, vectorised over decision times T. h1_pos = -1 or cause != "" means
        H1-derived inputs are UNAVAILABLE at that T. Pass m15_segment to also refuse an H1 bar
        from before a data hole (COMMON 1.4).
    true_range, atr_by_segment, todv, rs_4h, rs_day, twap, datr, spread_pts: the building blocks.

Column conventions: `local_date` (datetime.date) and `local_min` (minutes after local midnight)
come from the bar's OPEN time in the exchange's zone (zoneinfo via pandas, no fixed offsets).

KNOWN DATA ISSUE (escalated to Quant/owner, not handled here because COMMON 1.3 has no rule for
it): inside the tick-volume corrupt window 2024-10-04..2025-01-08 the demo PRICES are frozen
too (about half or more of in-session M15 bars have H - L <= 0.5; ~25 flat sessions per index).
TR, ATR14, TODV, DATR, IB and RS_4h are meaningless there, and the 20-date TODV / 14-date DATR
lookbacks carry the damage for several weeks after the window. This module only follows the
spec; once COMMON 1.3 gets a price-corruption entry and config a matching window, it must be
consumed here (for example as a segment break).
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from core.calendar import Exchange, exchange_for, is_calendar_trading_date, to_local
from core.config import tf_minutes
from data.segments import segment_ids, tv_bad
from features.indicators import atr

# SPEC NOTE: COMMON 3 defines spread_pts_t = spread_t * 0.01 literally; config.yaml has no
# point-size constant and this module may not edit config, so the spec's factor is used as is.
SPREAD_POINT = 0.01

OHLC = ["open", "high", "low", "close"]


# ---------------------------------------------------------------------------------------------
# Inputs and time conventions (COMMON 2)
# ---------------------------------------------------------------------------------------------

def _prepare(bars: pd.DataFrame) -> pd.DataFrame:
    """Sorted, unique, ns-resolution copy. Stored parquet times are microsecond resolution;
    mixing resolutions in searchsorted silently compares raw integers, so everything is ns."""
    df = bars.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True).astype("datetime64[ns, UTC]")
    df = df.sort_values("time_utc", kind="stable").reset_index(drop=True)
    if df["time_utc"].duplicated().any():
        raise ValueError("Duplicate bar open times; the store should have removed them")
    return df


def _minutes(t) -> int:
    return t.hour * 60 + t.minute


def bar_times(time_utc: pd.Series, bar_minutes: int, tz: str) -> pd.DataFrame:
    """Close time, exchange-local open time, local date and local minute-of-day per bar."""
    local = to_local(time_utc, tz)
    out = pd.DataFrame({
        "close_utc": time_utc + pd.Timedelta(minutes=bar_minutes),
        "local_time": local,
        "local_date": local.dt.date,
        "local_min": (local.dt.hour * 60 + local.dt.minute).astype("int64"),
    }, index=time_utc.index)
    # Day number of the local date, for fast sorted lookups (searchsorted needs numbers).
    out["date_ord"] = (local.dt.tz_localize(None).dt.normalize()
                       .to_numpy().astype("datetime64[D]").astype("int64"))
    return out


# ---------------------------------------------------------------------------------------------
# True range and ATR (COMMON 3)
# ---------------------------------------------------------------------------------------------

def true_range(df: pd.DataFrame, seg: pd.Series) -> pd.Series:
    """TR with C_prev from the same segment; the first bar of a segment uses H - L."""
    prev_close = df["close"].shift(1).where(seg.eq(seg.shift(1)))
    hl = df["high"] - df["low"]
    tr = pd.concat([hl, (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
                   axis=1).max(axis=1)          # NaN C_prev is skipped -> H - L
    return tr.rename("tr")


def atr_by_segment(df: pd.DataFrame, seg: pd.Series, period: int) -> pd.Series:
    """`features.indicators.atr` restarted at every segment start, so no EWM state or
    previous close leaks across a data hole; NaN for the first period-1 bars of a segment."""
    if df.empty:
        return pd.Series(np.nan, index=df.index, name="atr14")
    parts = [atr(df.loc[idx, ["high", "low", "close"]], period)
             for _, idx in seg.groupby(seg, sort=False).groups.items()]
    return pd.concat(parts).reindex(df.index).rename("atr14")


# ---------------------------------------------------------------------------------------------
# Prior-window helper shared by TODV and DATR
# ---------------------------------------------------------------------------------------------

def _prior_window(ref_keys: np.ndarray, ref_vals: np.ndarray, query_keys: np.ndarray,
                  window: int, min_obs: int, how: str) -> np.ndarray:
    """For each query key q: aggregate of the last `window` reference values whose key is
    strictly < q (so the current date is excluded), NaN if fewer than `min_obs` exist.
    ref_keys must be sorted ascending."""
    out = np.full(len(query_keys), np.nan)
    if len(ref_keys) == 0:
        return out
    roll = pd.Series(ref_vals, dtype="float64").rolling(window, min_periods=min_obs)
    agg = (roll.median() if how == "median" else roll.mean()).to_numpy()
    k = np.searchsorted(ref_keys, query_keys, side="left")     # number of refs with key < q
    has = k > 0
    out[has] = agg[k[has] - 1]
    return out


# ---------------------------------------------------------------------------------------------
# Valid trading dates and DATR (COMMON 1.6, 3)
# ---------------------------------------------------------------------------------------------

def _cash_minutes(ex: Exchange) -> tuple[int, int]:
    return _minutes(ex.cash_open), _minutes(ex.cash_close)


def trading_dates(f: pd.DataFrame, ex: Exchange) -> pd.DataFrame:
    """One row per local date that has M15 bars: its segment, calendar rule, cash-open bar,
    validity, number of in-session bars and the in-session range (max H - min L).

    A local date never holds bars of two segments: a segment break is a gap of more than
    7 days, so the bars on either side of it lie on different dates."""
    g = f.groupby("local_date", sort=True)
    dates = pd.DataFrame({
        "date_ord": g["date_ord"].first(),
        "segment": g["segment"].first(),
        "cash_open_present": g["is_cash_open_bar"].any(),
    })
    dates["calendar_ok"] = [is_calendar_trading_date(ex.calendar, d) for d in dates.index]
    dates["valid"] = dates["calendar_ok"] & dates["cash_open_present"]
    sess = f[f["in_session"]].groupby("local_date")
    dates["n_session_bars"] = sess.size().reindex(dates.index, fill_value=0).astype("int64")
    dates["session_range"] = (sess["high"].max() - sess["low"].min()).reindex(dates.index)
    return dates


def datr(dates: pd.DataFrame, expected_bars: int | None, n_dates: int,
         min_coverage: float) -> pd.DataFrame:
    """DATR_d: mean in-session range over the `n_dates` most recent PRIOR valid trading dates
    of the same segment with >= min_coverage x expected session bars. Today excluded; NaN until
    the full lookback exists in the segment."""
    out = dates.copy()
    if expected_bars is None:
        # SPEC NOTE: config features.expected_session_bars has no entry for this index (US500,
        # which is not ranked or traded), so its DATR is undefined and left NaN.
        out["datr_ok"] = False
        out["datr"] = np.nan
        return out
    out["datr_ok"] = out["valid"] & (out["n_session_bars"] >= min_coverage * expected_bars)
    vals = np.full(len(out), np.nan)
    for _, pos in out.groupby("segment", sort=False).indices.items():
        part = out.iloc[pos]
        ref = part[part["datr_ok"]]
        vals[pos] = _prior_window(ref["date_ord"].to_numpy(), ref["session_range"].to_numpy(),
                                  part["date_ord"].to_numpy(), n_dates, n_dates, "mean")
    out["datr"] = vals
    return out


# ---------------------------------------------------------------------------------------------
# TODV, RS_4h, RS_day, TWAP, spread (COMMON 3, 4.1)
# ---------------------------------------------------------------------------------------------

def todv(f: pd.DataFrame, ref_valid: pd.Series, n_dates: int, min_obs: int) -> pd.Series:
    """Median TR at the same exchange-local HH:MM over the previous `n_dates` valid trading
    dates of the same segment that have such a bar (>= min_obs, current date excluded).

    `ref_valid` is the date-level validity of each row's local date; only prior dates are
    read, and those are complete at any T on a later date."""
    vals = np.full(len(f), np.nan)
    work = f[["segment", "local_min", "date_ord", "tr"]].assign(ref=ref_valid.to_numpy())
    for _, pos in work.groupby(["segment", "local_min"], sort=False).indices.items():
        part = work.iloc[pos]                       # time-sorted, so date_ord is ascending
        ref = part[part["ref"]]
        vals[pos] = _prior_window(ref["date_ord"].to_numpy(), ref["tr"].to_numpy(),
                                  part["date_ord"].to_numpy(), n_dates, min_obs, "median")
    return pd.Series(vals, index=f.index, name="todv")


def rs_4h(f: pd.DataFrame, lookback: int, bar_minutes: int) -> pd.Series:
    """(C_t - C_{t-L}) / sqrt(sum_{j=t-L+1..t} TODV_j^2), NaN unless bars t-L..t have
    consecutive open times (so no missing bar, daily break or weekend), one segment and all
    TODV_j finite."""
    t, seg = f["time_utc"], f["segment"]
    consecutive = (t - t.shift(lookback)) == pd.Timedelta(minutes=lookback * bar_minutes)
    one_segment = seg.eq(seg.shift(lookback))
    # A rolling sum with min_periods=L is NaN as soon as any TODV_j in the window is NaN.
    denom = np.sqrt((f["todv"] ** 2).rolling(lookback, min_periods=lookback).sum())
    rs = (f["close"] - f["close"].shift(lookback)) / denom
    # SPEC NOTE: a zero denominator (all 16 TODV_j = 0, never seen in real data) gives an
    # infinite score; it is set to NaN like any other invalid score.
    ok = consecutive & one_segment & np.isfinite(rs)
    return rs.where(ok).rename("rs_4h")


def rs_day(h: pd.DataFrame, lookback: int) -> pd.Series:
    """(C_j - C_{j-24}) / ATR14_H1_j over the previous `lookback` H1 bars by position, NaN
    unless j-24..j share one segment and ATR14_H1_j > 0."""
    seg = h["segment"]
    ok = seg.eq(seg.shift(lookback)) & (h["atr14"] > 0)
    rs = (h["close"] - h["close"].shift(lookback)) / h["atr14"]
    return rs.where(ok).rename("rs_day")


def twap(f: pd.DataFrame, open_min: int) -> pd.Series:
    """Equal-weight mean of (H+L+C)/3 from the date's cash-open bar to bar t inclusive.
    Missing bars are skipped; NaN if the cash-open bar is missing.

    SPEC NOTE: bars of the local date before the cash open have no session mean and are NaN;
    bars after the cash close keep accumulating, because the spec gives no end (strategies
    only read TWAP inside their windows)."""
    out = pd.Series(np.nan, index=f.index, name="twap")
    has_open = f.groupby("local_date")["is_cash_open_bar"].transform("any")
    rows = f[(f["local_min"] >= open_min) & has_open]
    typical = (rows["high"] + rows["low"] + rows["close"]) / 3.0
    key = [rows["segment"], rows["local_date"]]
    out.loc[rows.index] = typical.groupby(key).cumsum() / typical.groupby(key).cumcount().add(1)
    return out


def spread_pts(spread: pd.Series) -> pd.Series:
    """Bar spread in index points. Cost use only, never a signal input."""
    return (spread.astype("float64") * SPREAD_POINT).rename("spread_pts")


# ---------------------------------------------------------------------------------------------
# Initial balance (COMMON 3)
# ---------------------------------------------------------------------------------------------

def ib_table(f: pd.DataFrame, key: str, cfg: dict, ib_minutes: int) -> pd.DataFrame:
    """IB per local date for one `ib_minutes` grid value: the m/15 M15 bars with local open
    times cash_open, cash_open + 15, ..., all required. IBH = max H, IBL = min L, W = IBH - IBL,
    defined_utc = close of the last IB bar (the IB must not be read before it)."""
    bar_min = tf_minutes(cfg["timeframes"]["entry"])
    if ib_minutes <= 0 or ib_minutes % bar_min:
        raise ValueError(f"ib_minutes must be a positive multiple of {bar_min}, got {ib_minutes}")
    n = ib_minutes // bar_min
    open_min, _ = _cash_minutes(exchange_for(cfg, key))
    wanted = open_min + bar_min * np.arange(n)
    rows = f[f["local_min"].isin(wanted)]
    g = rows.groupby("local_date")
    out = pd.DataFrame({
        "n_bars": g.size(),
        "ibh": g["high"].max(),
        "ibl": g["low"].min(),
        "defined_utc": g["close_utc"].max(),
    }).reindex(sorted(f["local_date"].unique()))
    out.index.name = "local_date"
    out["complete"] = out["n_bars"].fillna(0).astype("int64").eq(n)
    for col in ("ibh", "ibl"):
        out[col] = out[col].where(out["complete"])
    out["defined_utc"] = out["defined_utc"].where(out["complete"])
    out["w"] = out["ibh"] - out["ibl"]
    return out[["ibh", "ibl", "w", "defined_utc", "complete"]]


# ---------------------------------------------------------------------------------------------
# H1 alignment at decision time T (COMMON 2)
# ---------------------------------------------------------------------------------------------

def h1_alignment(T, h1_close: pd.Series | np.ndarray, m15_open: pd.Series | np.ndarray,
                 m15_minutes: int = 15, m15_segment: pd.Series | np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised COMMON 2 rule for decision times T (scalar or array of UTC timestamps).

    j = latest H1 bar with close <= T. UNAVAILABLE if there is none, or if close(j) <= T - 60
    min while an M15 bar exists with open in [close(j), floor_hour(T)): that M15 bar proves the
    H1 bar after j should exist, so j is never silently used in its place.

    With `m15_segment` (segment id per M15 bar, aligned with m15_open), j is also UNAVAILABLE
    when it closed before the start of the M15 segment that T is in (COMMON 1.4).

    Returns (h1_pos, cause): positional index into the H1 frame (-1 if none) and a cause
    string per T ("" when available)."""
    if isinstance(T, (str, datetime)):           # pd.Timestamp is a datetime
        T = [T]
    Ts = pd.DatetimeIndex(pd.to_datetime(T, utc=True)).as_unit("ns")
    hc = pd.DatetimeIndex(pd.to_datetime(h1_close, utc=True)).as_unit("ns").asi8
    mo = pd.DatetimeIndex(pd.to_datetime(m15_open, utc=True)).as_unit("ns").asi8
    t = Ts.asi8
    j = np.searchsorted(hc, t, side="right") - 1
    cause = np.full(len(t), "", dtype=object)
    cause[j < 0] = "no H1 bar closed by T"
    cj = np.where(j >= 0, hc[np.maximum(j, 0)], np.iinfo("int64").min)
    # Latest M15 bar closed by T (open <= T - bar length); -1 if none.
    k = np.searchsorted(mo, t - pd.Timedelta(minutes=m15_minutes).value, side="right") - 1

    if m15_segment is not None and len(mo):
        # SPEC NOTE: the literal COMMON 2 rule finds no M15 bar in [close(j), floor_hour(T)) at
        # the first decisions after a data hole, so it would hand over the last H1 bar before
        # the hole, weeks old. COMMON 1.4 forbids using bars of an earlier segment, so j is
        # UNAVAILABLE when it closed before the first M15 open of T's segment. The first M15
        # segment is exempt: no hole precedes it, and H1 history may simply start earlier.
        seg = np.asarray(m15_segment)
        first_pos = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1]   # segment start rows
        start_of = np.repeat(mo[first_pos], np.diff(np.r_[first_pos, len(seg)]))
        seg_start = np.where(k >= 0, start_of[np.maximum(k, 0)], np.iinfo("int64").min)
        earlier = (j >= 0) & (k >= 0) & (seg_start > mo[0]) & (cj < seg_start)
        for i in np.flatnonzero(earlier):
            cause[i] = (f"H1 bar from earlier segment: last H1 bar closed "
                        f"{pd.Timestamp(cj[i], tz='UTC')}, before the data hole")

    old = (j >= 0) & (cj <= t - pd.Timedelta(minutes=60).value)
    # Upper bound floor_hour(T), and never an M15 bar that has not closed by T.
    hi_t = np.minimum(Ts.floor("h").asi8, t - pd.Timedelta(minutes=m15_minutes).value + 1)
    n_between = np.searchsorted(mo, hi_t, side="left") - np.searchsorted(mo, cj, side="left")
    missing = old & (n_between > 0) & (cause == "")
    for i in np.flatnonzero(missing):
        cause[i] = (f"H1 bar missing: last H1 bar closed {pd.Timestamp(cj[i], tz='UTC')} "
                    f"but M15 bars exist after it")
    j = np.where(cause == "", j, -1)
    return j, cause


# ---------------------------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------------------------

def compute_m15_features(m15: pd.DataFrame, key: str, cfg: dict
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All COMMON M15 features for one index, plus the per-local-date table with DATR."""
    fcfg = cfg["features"]
    ex = exchange_for(cfg, key)
    bar_min = tf_minutes(cfg["timeframes"]["entry"])
    open_min, close_min = _cash_minutes(ex)

    f = _prepare(m15)
    f = pd.concat([f, bar_times(f["time_utc"], bar_min, ex.tz)], axis=1)
    f["segment"] = segment_ids(f["time_utc"], cfg["data_quality"]["segment_break_days"])
    f["tv_bad"] = tv_bad(f["time_utc"], cfg)
    f["is_cash_open_bar"] = f["local_min"].eq(open_min)
    f["in_session"] = (f["local_min"] >= open_min) & (f["local_min"] < close_min)

    dates = trading_dates(f, ex)
    dates = datr(dates, (fcfg.get("expected_session_bars") or {}).get(key),
                 fcfg["datr_dates"], fcfg["datr_min_coverage"])
    date_valid = f["local_date"].map(dates["valid"]).astype(bool)
    # SPEC NOTE: per-row `valid_date` is causal: True only from the cash-open bar onward on a
    # valid date, because before that bar closes nobody can know it will exist. The date-level
    # flag (dates.valid) is used for prior-date lookbacks, where it is fully known.
    f["valid_date"] = date_valid & (f["local_min"] >= open_min)

    f["tr"] = true_range(f, f["segment"])
    f["atr14"] = atr_by_segment(f, f["segment"], fcfg["atr_period"])
    f["todv"] = todv(f, date_valid, fcfg["tod_lookback_dates"], fcfg["tod_min_obs"])
    f["rs_4h"] = rs_4h(f, fcfg["rs4h_lookback_bars"], bar_min)
    f["twap"] = twap(f, open_min)
    if "spread" in f:
        f["spread_pts"] = spread_pts(f["spread"])
    return f, dates


def compute_h1_features(h1: pd.DataFrame, key: str, cfg: dict) -> pd.DataFrame:
    """All COMMON H1 features for one index (ATR14_H1, RS_day), segment-restarted."""
    fcfg = cfg["features"]
    ex = exchange_for(cfg, key)
    bar_min = tf_minutes(cfg["timeframes"]["regime"])
    h = _prepare(h1)
    h = pd.concat([h, bar_times(h["time_utc"], bar_min, ex.tz)], axis=1)
    h["segment"] = segment_ids(h["time_utc"], cfg["data_quality"]["segment_break_days"])
    h["tv_bad"] = tv_bad(h["time_utc"], cfg)
    h["tr"] = true_range(h, h["segment"])
    h["atr14"] = atr_by_segment(h, h["segment"], fcfg["atr_period"])
    h["rs_day"] = rs_day(h, fcfg["rsday_lookback_bars"])
    return h


def is_finite(x) -> bool:
    """True for a finite float; convenience for strategies' data-completeness checks."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


__all__ = [
    "SPREAD_POINT", "bar_times", "true_range", "atr_by_segment", "trading_dates", "datr",
    "todv", "rs_4h", "rs_day", "twap", "spread_pts", "ib_table", "h1_alignment",
    "compute_m15_features", "compute_h1_features", "is_finite",
]
