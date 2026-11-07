"""Data-quality checks. Problems are reported, never silently fixed.

With `key` and `cfg` the report also covers the frozen defects of COMMON 1.3-1.6 (ER-09):
segments, the configured corrupt tick-volume window and bars printed on exchange holidays.
"""
from __future__ import annotations

from datetime import time, timedelta

import numpy as np
import pandas as pd

from core.calendar import (Exchange, early_close, exchange_for, is_full_holiday, local_to_utc,
                           to_local)
from data.segments import segment_ids, tv_bad

DEFAULT_BREAK_DAYS = 7
CLOSURE_MIN = pd.Timedelta(hours=12)      # gaps up to this are "other gaps" (intraday)


def holiday_bar_mask(time_utc: pd.Series, cfg: dict, key: str) -> pd.Series:
    """True for bars whose exchange-local date is a full holiday of the instrument's
    exchange (NYSE for the US indices, XETRA for GER40). The demo prints normal-looking
    bars on those dates, so lookbacks must not mistake them for trading days.

    SPEC NOTE: ER-09(d) says "flag US holiday bars" without saying which clock decides the
    date. A bar counts when the exchange-local date of its OPEN time is a full holiday (the
    same local date COMMON 1.6 uses for valid trading dates). Early-close dates are not full
    holidays and are not counted; COMMON 1.6 already excludes them from valid trading dates.
    """
    ex = exchange_for(cfg, key)
    local_dates = pd.Series(to_local(time_utc, ex.tz).dt.date, index=time_utc.index)
    holiday = {d: is_full_holiday(ex.calendar, d) for d in local_dates.unique()}
    return local_dates.map(holiday).astype(bool)


def _overlap(a: pd.Timestamp, b: pd.Timestamp, lo: pd.Timestamp,
             hi: pd.Timestamp) -> pd.Timedelta:
    return max(pd.Timedelta(0), min(b, hi) - max(a, lo))


def classify_closure(start: pd.Timestamp, end: pd.Timestamp, ex: Exchange | None) -> str:
    """'weekend', 'holiday' or 'unexplained' for bars missing over [start, end) UTC.

    GATE 1 called every gap over 12h a weekend, and a first fix called any gap spanning a
    Saturday a weekend, which still hid a missing Friday next to a weekend. A closure is now
    expected only if the calendar explains it, judged on exchange-local dates:
    - it takes no time from the cash session (shortened on early-close dates) of any local
      weekday that is not a full holiday, and
    - it takes at most 12h in total from such weekdays (the broker's evening and overnight
      tails around a weekend or holiday; a longer loss is a hole even outside the session).
    It is a 'holiday' closure if it touches a full-holiday weekday, else a 'weekend'.
    Without an exchange (no cfg or key) dates are UTC, no holidays are known and only the
    12h rule applies, so every mid-week closure is reported as unexplained.

    SPEC NOTE: ER-09(b) asks to fix the classifier but gives no rule. QA asked for "every
    weekday it fully covers is a weekend or full holiday"; read literally that passes a
    missing Monday morning (it fully covers no weekday), so the stricter session/12h rule
    above is used. Both QA test cases (missing Friday, missing Monday morning) fail it.
    """
    tz = ex.tz if ex is not None else "UTC"
    d, last = start.tz_convert(tz).date(), (end - pd.Timedelta(1, "ns")).tz_convert(tz).date()
    weekday_lost, holiday = pd.Timedelta(0), False
    while d <= last:
        if d.weekday() < 5:
            if ex is not None and is_full_holiday(ex.calendar, d):
                holiday = True
            else:
                weekday_lost += _overlap(start, end, local_to_utc(d, time(0), tz),
                                         local_to_utc(d + timedelta(days=1), time(0), tz))
                if ex is not None:
                    close = early_close(ex.calendar, d) or ex.cash_close
                    if _overlap(start, end, local_to_utc(d, ex.cash_open, tz),
                                local_to_utc(d, close, tz)) > pd.Timedelta(0):
                        return "unexplained"
        d += timedelta(days=1)
    if weekday_lost > CLOSURE_MIN:
        return "unexplained"
    return "holiday" if holiday else "weekend"


def quality_report(df: pd.DataFrame, tf_min: int, spike_pct: float = 1.0, *,
                   key: str | None = None, cfg: dict | None = None) -> dict:
    t = df["time_utc"]
    dq = (cfg or {}).get("data_quality") or {}
    break_days = float(dq.get("segment_break_days", DEFAULT_BREAK_DAYS))
    step = pd.Timedelta(minutes=tf_min)
    diffs = t.diff()

    # Gaps inside the trading week. Breaks longer than 12h (weekends) are expected, and so
    # is the broker's daily break: a gap that starts at the same server-clock time on at
    # least half of the trading days. (It moves in UTC when daylight saving changes.)
    clock = pd.to_datetime(df["server_time"], unit="s") if "server_time" in df else t
    gap_start = (clock - diffs).dt.time
    gap_mask = (diffs > step) & (diffs <= CLOSURE_MIN)
    counts = gap_start[gap_mask].value_counts()
    daily_break = None
    if len(counts) and counts.iloc[0] >= max(3, 0.5 * t.dt.normalize().nunique()):
        daily_break = counts.index[0]
        gap_mask &= gap_start != daily_break

    # Gaps over 12h used to be classed as weekends wholesale, which hid a 55-day hole
    # (GATE 1). Now: over segment_break_days is a segment break (COMMON 1.4); between 12h
    # and that, each closure is checked against the exchange calendar (classify_closure)
    # and listed with its dates (last bar before, first bar after) unless a plain weekend.
    # SPEC NOTE: the 12h boundary is the GATE 1 one, kept; a gap of exactly
    # segment_break_days is a closure, not a break (COMMON 1.4 says "> 7 days").
    long = diffs > pd.Timedelta(days=break_days)
    long_breaks = [f"{(t[i] - diffs[i]):%Y-%m-%d} to {t[i]:%Y-%m-%d}" for i in df.index[long]]
    ex = None
    if cfg is not None and key is not None and key in (cfg.get("exchanges") or {}):
        ex = exchange_for(cfg, key)
    closures = {"weekend": [], "holiday": [], "unexplained": []}
    for i in df.index[(diffs > CLOSURE_MIN) & ~long]:
        prev = t[i] - diffs[i]
        kind = classify_closure(prev + step, t[i], ex)
        closures[kind].append(f"{prev:%Y-%m-%d %H:%M} to {t[i]:%Y-%m-%d %H:%M}")

    seg = segment_ids(t, break_days) if len(df) else pd.Series(dtype="int64")
    segments = [f"{g.min():%Y-%m-%d %H:%M} to {g.max():%Y-%m-%d %H:%M} ({len(g)} bars)"
                for _, g in t.groupby(seg)]

    # Configured defects need the config; without it they are reported as None (unknown).
    tv_bad_bars = int(tv_bad(t, cfg).sum()) if cfg is not None else None
    tv_windows = [f"{w['start']} to {w['end']}" for w in dq.get("tick_volume_bad") or ()]
    holiday_bars = int(holiday_bar_mask(t, cfg, key).sum()) if ex is not None else None

    bad_ohlc = (df["high"] < df[["open", "close"]].max(axis=1)) | \
               (df["low"] > df[["open", "close"]].min(axis=1)) | (df["low"] <= 0)

    # Possible bad ticks: a bar-to-bar move over spike_pct that the next bar mostly reverses,
    # on below-normal tick volume. Real whipsaws (news, crashes) come with heavy volume.
    # Returns across a session break are excluded.
    r = np.log(df["close"]).diff().where(diffs == step)
    big = r.abs() > spike_pct / 100
    spikes = big & (r.shift(-1) * r < 0) & (r.shift(-1).abs() > 0.7 * r.abs())
    if "tick_volume" in df:
        typical = df["tick_volume"].rolling(500, min_periods=50).median()
        spikes &= df["tick_volume"] < typical

    return {
        "bars": int(len(df)),
        "first": t.min(),
        "last": t.max(),
        "daily_break_server": daily_break,
        "duplicates": int(t.duplicated().sum()),
        "out_of_order": int((diffs < pd.Timedelta(0)).sum()),
        "intraday_gaps": int(gap_mask.sum()),
        "largest_intraday_gap": diffs[gap_mask].max() if gap_mask.any() else pd.Timedelta(0),
        "long_breaks": long_breaks,
        "calendar": ex.calendar if ex is not None else None,
        "weekend_closures": len(closures["weekend"]),
        "holiday_closures": closures["holiday"] if ex is not None else None,
        "unexplained_gaps": closures["unexplained"],
        "segments": segments,
        "holiday_bars": holiday_bars,
        "tv_bad_bars": tv_bad_bars,
        "tv_bad_windows": tv_windows,
        "bad_ohlc": int(bad_ohlc.sum()),
        "zero_spread_bars": int((df["spread"] <= 0).sum()),
        "large_moves": int(big.sum()),
        "spikes": int(spikes.sum()),
        "median_spread_points": float(df["spread"].median()) if len(df) else float("nan"),
    }


def _count(value) -> str:
    return "-" if value is None else str(value)


def _dates(value) -> str:
    return "-" if value is None else ("; ".join(value) or "none")


def format_report(rows: list[tuple[str, str, dict]]) -> str:
    lines = ["| Instrument | TF | Bars | First (UTC) | Last (UTC) | Daily break (server) | "
             "Dupes | Other gaps | Max gap | Weekend closures | Holiday closures >12h | "
             "Unexplained gaps >12h | Segment breaks | Segments | Holiday bars | tv_bad bars | "
             "Bad OHLC | Zero spread | Large moves | Spikes | Median spread (pts) |",
             "|" + "---|" * 21]
    for key, tf, r in rows:
        brk = r["daily_break_server"].strftime("%H:%M") if r["daily_break_server"] else "-"
        lines.append(
            f"| {key} | {tf} | {r['bars']} | {r['first']:%Y-%m-%d %H:%M} | "
            f"{r['last']:%Y-%m-%d %H:%M} | {brk} | {r['duplicates']} | {r['intraday_gaps']} | "
            f"{r['largest_intraday_gap']} | {r['weekend_closures']} | "
            f"{_dates(r['holiday_closures'])} | {_dates(r['unexplained_gaps'])} | "
            f"{'; '.join(r['long_breaks']) or 'none'} | {'; '.join(r['segments']) or 'none'} | "
            f"{_count(r['holiday_bars'])} | {_count(r['tv_bad_bars'])} | {r['bad_ohlc']} | "
            f"{r['zero_spread_bars']} | {r['large_moves']} | {r['spikes']} | "
            f"{r['median_spread_points']:.1f} |")
    return "\n".join(lines)
