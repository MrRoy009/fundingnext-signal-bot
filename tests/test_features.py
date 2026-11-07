"""COMMON 2, 3, 4.1 feature library: synthetic-data checks, DST, holidays, segments, look-ahead."""
from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from core.calendar import is_calendar_trading_date
from features.core import (compute_h1_features, compute_m15_features, h1_alignment, ib_table,
                           spread_pts)
from features.indicators import atr
from features.service import FeatureService, IndexFeatures

UTC = "UTC"


# ---------------------------------------------------------------------------------------------
# Synthetic bars
# ---------------------------------------------------------------------------------------------

def make_m15(start: str, end: str, seed: int = 0, drop=()) -> pd.DataFrame:
    """Weekday (UTC) M15 bars around the clock with a random walk; `drop` removes open times."""
    t = pd.date_range(start, end, freq="15min", tz=UTC, inclusive="left")
    t = t[t.day_of_week < 5]
    t = t[~t.isin(pd.DatetimeIndex(pd.to_datetime(list(drop), utc=True)))] if drop else t
    rng = np.random.default_rng(seed)
    n = len(t)
    close = 1000 + np.cumsum(rng.normal(0, 2, n))
    open_ = close + rng.normal(0, 1, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 3, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 3, n)
    return pd.DataFrame({"time_utc": t, "open": open_, "high": high, "low": low, "close": close,
                         "tick_volume": rng.integers(50, 500, n).astype("uint64"),
                         "spread": rng.integers(0, 300, n).astype("int32")})


def to_h1(m15: pd.DataFrame) -> pd.DataFrame:
    g = m15.set_index("time_utc").resample("1h")
    h = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "tick_volume": g["tick_volume"].sum(),
                      "spread": g["spread"].min()}).dropna(subset=["open"])
    return h.reset_index()


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=UTC)


# ---------------------------------------------------------------------------------------------
# Time conventions and DST (COMMON 2)
# ---------------------------------------------------------------------------------------------

def test_bar_times_local_fields(cfg):
    f, _ = compute_m15_features(make_m15("2024-06-03", "2024-06-05"), "US30", cfg)
    assert (f["close_utc"] - f["time_utc"] == pd.Timedelta(minutes=15)).all()
    r = f[f["time_utc"] == ts("2024-06-03 13:30")].iloc[0]
    assert r["local_date"] == date(2024, 6, 3) and r["local_min"] == 9 * 60 + 30
    assert r["is_cash_open_bar"] and r["in_session"]
    # 20:00 UTC = 16:00 New York: cash close is exclusive.
    assert not f[f["time_utc"] == ts("2024-06-03 20:00")].iloc[0]["in_session"]
    h = compute_h1_features(to_h1(make_m15("2024-06-03", "2024-06-05")), "US30", cfg)
    assert (h["close_utc"] - h["time_utc"] == pd.Timedelta(minutes=60)).all()


@pytest.mark.parametrize("key, start, end, expect", [
    # US DST starts 2024-03-10, EU 2024-03-31: New York open 14:30 UTC before, 13:30 after.
    ("US30", "2024-03-04", "2024-03-16", {date(2024, 3, 8): "14:30", date(2024, 3, 11): "13:30",
                                          date(2024, 3, 15): "13:30"}),
    # EU DST ends 2024-10-27, US 2024-11-03: New York open stays 13:30 UTC in the mismatch week.
    ("NAS100", "2024-10-21", "2024-11-09", {date(2024, 10, 25): "13:30", date(2024, 10, 28): "13:30",
                                            date(2024, 11, 1): "13:30", date(2024, 11, 4): "14:30"}),
    # Berlin open 09:00 local: 08:00 UTC in winter, 07:00 UTC in summer.
    ("GER40", "2024-03-25", "2024-04-06", {date(2024, 3, 28): "08:00", date(2024, 4, 3): "07:00"}),
    ("GER40", "2024-10-21", "2024-11-09", {date(2024, 10, 25): "07:00", date(2024, 10, 28): "08:00",
                                           date(2024, 11, 4): "08:00"}),
])
def test_dst_cash_open_detection(cfg, key, start, end, expect):
    fx = IndexFeatures.build(key, make_m15(start, end), to_h1(make_m15(start, end)), cfg)
    opens = fx.m15[fx.m15["is_cash_open_bar"]].set_index("local_date")["time_utc"]
    for d, hhmm in expect.items():
        assert opens[d] == ts(f"{d} {hhmm}"), (key, d)
        # TWAP starts at the cash-open bar; the bar before it has none.
        row = fx.m15[fx.m15["time_utc"] == opens[d]].iloc[0]
        assert row["twap"] == pytest.approx((row["high"] + row["low"] + row["close"]) / 3)
        assert np.isnan(fx.m15_at(opens[d])["twap"])
        # IB(60) is defined at the close of the 4th bar, i.e. one hour after the open in UTC.
        ib = fx.ib_final(d, 60)
        assert ib.defined_utc == opens[d] + pd.Timedelta(hours=1)
        assert not fx.ib(d, 60, T=opens[d] + pd.Timedelta(minutes=45)).available
        assert fx.ib(d, 60, T=opens[d] + pd.Timedelta(hours=1)).available


@pytest.mark.parametrize("key, start, end", [
    ("US30", "2024-03-06", "2024-03-14"),       # US switch, EU not yet
    ("US30", "2024-10-30", "2024-11-07"),       # US switch back
    ("GER40", "2024-03-27", "2024-04-04"),
    ("GER40", "2024-10-24", "2024-10-31"),
])
def test_dst_h1_used_at_T_closed_by_T(cfg, key, start, end):
    m = make_m15(start, end, seed=3)
    fx = IndexFeatures.build(key, m, to_h1(m), cfg)
    Ts = fx.m15["close_utc"]
    Ts = Ts[Ts >= fx.h1["close_utc"].iloc[0]].reset_index(drop=True)
    pos, cause = fx.h1_positions(Ts)
    assert (pos >= 0).all() and (cause == "").all()
    used_close = fx.h1["close_utc"].to_numpy()[pos]
    assert (used_close <= Ts.to_numpy()).all()
    # Complete data: the H1 bar used is the one closing at floor_hour(T) whenever that hour
    # traded (not on the Monday-open hour after a weekend).
    traded = (Ts.dt.floor("h") - pd.Timedelta(hours=1)).isin(fx.h1["time_utc"]).to_numpy()
    assert traded.mean() > 0.9
    assert (used_close[traded] == Ts.dt.floor("h").to_numpy()[traded]).all()


# ---------------------------------------------------------------------------------------------
# TR, ATR and segments (COMMON 1.4, 3)
# ---------------------------------------------------------------------------------------------

def test_tr_and_atr_restart_after_gap(cfg):
    a = make_m15("2024-06-03", "2024-06-06", seed=1)
    b = make_m15("2024-06-17", "2024-06-20", seed=2)          # 11-day gap: new segment
    b[["open", "high", "low", "close"]] += 500                # huge gap move must not leak
    m = pd.concat([a, b], ignore_index=True)
    f, _ = compute_m15_features(m, "US30", cfg)
    assert f["segment"].tolist() == [0] * len(a) + [1] * len(b)
    first = f.index[f["segment"] == 1][0]
    assert f.at[first, "tr"] == pytest.approx(f.at[first, "high"] - f.at[first, "low"])
    seg1 = f[f["segment"] == 1]
    assert seg1["atr14"].iloc[:13].isna().all() and seg1["atr14"].iloc[13:].notna().all()
    expect = atr(b[["high", "low", "close"]].reset_index(drop=True), 14)
    np.testing.assert_allclose(seg1["atr14"].to_numpy(), expect.to_numpy(), equal_nan=True)
    # TR inside a segment uses the previous close.
    i = first + 5
    hi, lo, cp = f.at[i, "high"], f.at[i, "low"], f.at[i - 1, "close"]
    assert f.at[i, "tr"] == pytest.approx(max(hi - lo, abs(hi - cp), abs(lo - cp)))
    # Same restart on H1.
    h = compute_h1_features(to_h1(m), "US30", cfg)
    hs1 = h[h["segment"] == 1]
    assert hs1["atr14"].iloc[:13].isna().all() and hs1["atr14"].iloc[13:].notna().all()
    assert hs1["rs_day"].iloc[:24].isna().all()


def test_lookbacks_restart_after_gap(cfg):
    a = make_m15("2024-04-01", "2024-05-25", seed=4)
    b = make_m15("2024-06-05", "2024-06-29", seed=5)
    f, dates = compute_m15_features(pd.concat([a, b], ignore_index=True), "US30", cfg)
    s1 = f[f["segment"] == 1]
    # TODV needs 10 prior valid dates in segment 1; the first 10 valid dates of it have none.
    valid1 = sorted(dates.index[(dates["segment"] == 1) & dates["valid"]])
    early = s1[s1["local_date"].isin(valid1[:10])]
    assert early["todv"].isna().all() and early["rs_4h"].isna().all()
    assert s1[s1["local_date"] == valid1[10]]["todv"].notna().any()
    # DATR needs 14 prior qualifying dates inside the segment.
    assert dates.loc[valid1[:14], "datr"].isna().all()
    assert np.isfinite(dates.at[valid1[14], "datr"])


# ---------------------------------------------------------------------------------------------
# Valid trading dates, TODV and DATR against brute-force references (COMMON 1.6, 3)
# ---------------------------------------------------------------------------------------------

def _brute_todv(f: pd.DataFrame, dates: pd.DataFrame, i: int) -> float:
    r = f.loc[i]
    ref = f[(f["segment"] == r["segment"]) & (f["local_min"] == r["local_min"])
            & (f["local_date"] < r["local_date"])]
    ref = ref[ref["local_date"].map(dates["valid"]).astype(bool)].tail(20)
    return float(ref["tr"].median()) if len(ref) >= 10 else np.nan


def test_valid_dates_and_todv_bruteforce(cfg):
    # 2024-06-19 and 2024-07-04 are NYSE holidays, 2024-07-03 an early close; 2024-06-26 has no
    # cash-open bar. Holiday bars get huge ranges so their inclusion would show.
    m = make_m15("2024-05-13", "2024-07-20", seed=6, drop=["2024-06-26 13:30"])
    hol = m["time_utc"].dt.tz_convert("America/New_York").dt.date.isin(
        [date(2024, 6, 19), date(2024, 7, 4), date(2024, 7, 3)])
    m.loc[hol, "high"] += 400
    f, dates = compute_m15_features(m, "US30", cfg)
    for d in (date(2024, 6, 19), date(2024, 7, 4), date(2024, 7, 3), date(2024, 6, 26)):
        assert not dates.at[d, "valid"]
    assert dates.at[date(2024, 7, 5), "valid"] and dates.at[date(2024, 6, 25), "valid"]
    # Per-row valid_date is causal: False before the cash-open bar, True from it on.
    day = f[f["local_date"] == date(2024, 7, 5)]
    assert not day[day["local_min"] < 570]["valid_date"].any()
    assert day[day["local_min"] >= 570]["valid_date"].all()
    for i in range(0, len(f), 37):
        b = _brute_todv(f, dates, i)
        got = f.at[i, "todv"]
        assert (np.isnan(b) and np.isnan(got)) or got == pytest.approx(b), i
    # No TODV after the holidays is inflated by the +400 holiday ranges.
    after = f[(f["local_date"] > date(2024, 7, 5)) & f["todv"].notna()]
    assert (after["todv"] < 50).all()


def test_todv_minimum_ten_dates(cfg):
    m = make_m15("2024-09-03", "2024-10-05", seed=7)
    f, dates = compute_m15_features(m, "GER40", cfg)
    valid = sorted(dates.index[dates["valid"]])
    at = f[f["local_min"] == 10 * 60].set_index("local_date")["todv"]
    assert at.loc[valid[:10]].isna().all()
    first = f[(f["local_date"] < valid[10]) & (f["local_min"] == 600)
              & f["local_date"].isin(valid)]["tr"]
    assert at.loc[valid[10]] == pytest.approx(first.median())


def test_datr_bruteforce(cfg):
    # Thin out one date below 90% coverage (drop 4 of 26 session bars) and keep the holidays.
    thin = [f"2024-06-12 {h}" for h in ("14:00", "15:00", "16:00", "17:00")]
    m = make_m15("2024-04-22", "2024-07-20", seed=8, drop=thin)
    f, dates = compute_m15_features(m, "US30", cfg)
    assert dates.at[date(2024, 6, 12), "n_session_bars"] == 22
    assert dates.at[date(2024, 6, 12), "valid"] and not dates.at[date(2024, 6, 12), "datr_ok"]
    ok = dates[dates["valid"] & (dates["n_session_bars"] >= 0.9 * 26)]
    assert date(2024, 7, 4) not in ok.index and date(2024, 6, 19) not in ok.index
    for d in dates.index:
        prior = ok[ok.index < d].tail(14)
        expect = prior["session_range"].mean() if len(prior) == 14 else np.nan
        got = dates.at[d, "datr"]
        assert (np.isnan(expect) and np.isnan(got)) or got == pytest.approx(expect), d
    sess = f[(f["local_date"] == date(2024, 7, 5)) & f["in_session"]]
    assert dates.at[date(2024, 7, 5), "session_range"] == pytest.approx(
        sess["high"].max() - sess["low"].min())
    assert len(sess) == 26


# ---------------------------------------------------------------------------------------------
# RS_4h, RS_day, TWAP, IB, spread (COMMON 3, 4.1)
# ---------------------------------------------------------------------------------------------

def test_rs_4h_formula_and_validity(cfg):
    missing = "2024-07-17 15:00"
    m = make_m15("2024-06-03", "2024-07-20", seed=9, drop=[missing])
    f, _ = compute_m15_features(m, "US30", cfg)
    i = f.index[f["time_utc"] == ts("2024-07-16 17:00")][0]
    expect = (f.at[i, "close"] - f.at[i - 16, "close"]) / np.sqrt(
        (f.loc[i - 15:i, "todv"] ** 2).sum())
    assert f.at[i, "rs_4h"] == pytest.approx(expect)
    # A missing bar anywhere in t-16..t makes the score NaN for 17 bars.
    j = f.index[f["time_utc"] == ts("2024-07-17 15:15")][0]
    assert f.loc[j:j + 15, "rs_4h"].isna().all()
    assert np.isfinite(f.at[j + 16, "rs_4h"])
    # Across the weekend the open times are not consecutive.
    mon = f.index[f["time_utc"] == ts("2024-07-15 01:00")][0]
    assert np.isnan(f.at[mon, "rs_4h"])
    # A NaN TODV inside the window (first 10 valid dates) makes it NaN.
    assert f[f["local_date"] == date(2024, 6, 4)]["rs_4h"].isna().all()


def test_rs_day(cfg):
    m = make_m15("2024-06-03", "2024-06-14", seed=10)
    h = compute_h1_features(to_h1(m), "NAS100", cfg)
    j = 100
    assert h.at[j, "rs_day"] == pytest.approx(
        (h.at[j, "close"] - h.at[j - 24, "close"]) / h.at[j, "atr14"])
    assert h["rs_day"].iloc[:24].isna().all()


def test_twap(cfg):
    d = "2024-06-05"
    m = make_m15("2024-06-03", "2024-06-08", seed=11, drop=[f"{d} 14:00"])
    f, _ = compute_m15_features(m, "US30", cfg)
    day = f[(f["local_date"] == date(2024, 6, 5))]
    sess = day[day["local_min"] >= 570]
    tp = (sess["high"] + sess["low"] + sess["close"]) / 3
    np.testing.assert_allclose(sess["twap"].to_numpy(), tp.expanding().mean().to_numpy())
    assert day[day["local_min"] < 570]["twap"].isna().all()
    # No cash-open bar: TWAP is NaN all day.
    m2 = make_m15("2024-06-03", "2024-06-08", seed=11, drop=[f"{d} 13:30"])
    f2, _ = compute_m15_features(m2, "US30", cfg)
    assert f2[f2["local_date"] == date(2024, 6, 5)]["twap"].isna().all()


def test_ib(cfg):
    m = make_m15("2024-06-03", "2024-06-08", seed=12, drop=["2024-06-06 14:15"])
    fx = IndexFeatures.build("US30", m, to_h1(m), cfg)
    f = fx.m15
    bars = f[(f["local_date"] == date(2024, 6, 5)) & f["local_min"].between(570, 570 + 45)]
    assert len(bars) == 4
    ib = fx.ib(date(2024, 6, 5), 60, ts("2024-06-05 14:30"))
    assert ib.ibh == bars["high"].max() and ib.ibl == bars["low"].min()
    assert ib.w == pytest.approx(ib.ibh - ib.ibl)
    assert ib.defined_utc == ts("2024-06-05 14:30")
    assert fx.ib_final(date(2024, 6, 5), 30).defined_utc == ts("2024-06-05 14:00")
    # A missing IB bar: NaN, whichever T.
    bad = fx.ib(date(2024, 6, 6), 60, T=ts("2024-06-06 16:00"))
    assert not bad.available and np.isnan(bad.ibh)
    # ib_minutes = 30 on the same date ends before the missing bar.
    assert fx.ib(date(2024, 6, 6), 30, ts("2024-06-06 14:00")).available
    with pytest.raises(ValueError):
        ib_table(f, "US30", cfg, 50)


def test_spread_pts(cfg):
    assert spread_pts(pd.Series([0, 150], dtype="int32")).tolist() == [0.0, 1.5]
    f, _ = compute_m15_features(make_m15("2024-06-03", "2024-06-04"), "US30", cfg)
    np.testing.assert_allclose(f["spread_pts"], f["spread"] * 0.01)


# ---------------------------------------------------------------------------------------------
# H1 alignment (COMMON 2)
# ---------------------------------------------------------------------------------------------

def test_h1_alignment_rule(cfg):
    m = make_m15("2024-06-03", "2024-06-08", seed=13)
    h1 = to_h1(m)
    h1_missing = h1[h1["time_utc"] != ts("2024-06-05 15:00")].reset_index(drop=True)
    fx = IndexFeatures.build("US30", m, h1_missing, cfg)
    # Present: T = 15:00 uses the bar 14:00-15:00; T = 15:15 also uses it.
    for T in ("2024-06-05 15:00", "2024-06-05 15:15", "2024-06-05 15:45"):
        a = fx.h1_at(ts(T))
        assert a.available and a.row["close_utc"] == ts("2024-06-05 15:00")
    # The bar 15:00-16:00 is missing while M15 bars of that hour exist: UNAVAILABLE at 16:00
    # and for the whole next hour, never the 15:00 bar silently.
    for T in ("2024-06-05 16:00", "2024-06-05 16:45"):
        a = fx.h1_at(ts(T))
        assert not a.available and "H1 bar missing" in a.cause
    assert fx.h1_at(ts("2024-06-05 17:00")).row["close_utc"] == ts("2024-06-05 17:00")
    # Market closed (no M15 bars after the last H1 close): the older bar stays usable.
    a = fx.h1_at(ts("2024-06-08 10:00"))        # Saturday
    assert a.available and a.row["close_utc"] == ts("2024-06-08 00:00")
    assert not fx.h1_at(ts("2024-06-01 10:00")).available         # before any H1 bar
    with pytest.raises(ValueError):
        fx.h1_at(pd.Timestamp("2024-06-05 15:00"))                # naive time


def test_h1_alignment_ignores_unclosed_m15(cfg):
    # At T = 16:00 the M15 bar 16:00-16:15 has not closed and is not evidence of anything.
    h1_close = pd.Series(pd.to_datetime(["2024-06-05 15:00"], utc=True))
    m15_open = pd.Series(pd.to_datetime(["2024-06-05 14:45", "2024-06-05 16:00"], utc=True))
    pos, cause = h1_alignment(ts("2024-06-05 16:00"), h1_close, m15_open)
    assert pos[0] == 0 and cause[0] == ""
    pos, cause = h1_alignment(ts("2024-06-05 16:15"), h1_close, m15_open)
    assert pos[0] == 0              # 16:00 bar is not inside [15:00, floor_hour = 16:00)
    m15_open = pd.Series(pd.to_datetime(["2024-06-05 15:00"], utc=True))
    pos, cause = h1_alignment([ts("2024-06-05 16:00")], h1_close, m15_open)
    assert pos[0] == -1 and cause[0]


def test_h1_alignment_refuses_bar_from_before_hole(cfg):
    # 11-day hole; segment 1 starts mid-hour (11:30) like the real 2025-09-09 restart.
    a = make_m15("2024-06-03", "2024-06-06", seed=17)
    b = make_m15("2024-06-17 11:30", "2024-06-20", seed=18)
    m = pd.concat([a, b], ignore_index=True)
    fx = IndexFeatures.build("US30", m, to_h1(m), cfg)
    last_pre = fx.h1[fx.h1["segment"] == 0]["close_utc"].max()
    # First post-hole decisions: the literal rule would return the pre-hole bar.
    al = fx.h1_at(ts("2024-06-17 11:45"))
    assert not al.available and "earlier segment" in al.cause
    # At 12:00 the 11:00-12:00 H1 bar of segment 1 has closed and is used.
    al = fx.h1_at(ts("2024-06-17 12:00"))
    assert al.available and al.row["segment"] == 1
    pos, cause = fx.h1_positions([ts("2024-06-17 11:45")])
    assert pos[0] == -1 and str(last_pre.date()) in cause[0]
    # Without segment ids the literal rule is unchanged (backwards compatible).
    pos, _ = h1_alignment(ts("2024-06-17 11:45"), fx.h1["close_utc"], fx.m15["time_utc"])
    assert fx.h1["close_utc"].iloc[pos[0]] == last_pre
    # The first segment is exempt: H1 history may start before M15 history.
    h_early = to_h1(make_m15("2024-05-27", "2024-06-06", seed=17))
    fx2 = IndexFeatures.build("US30", make_m15("2024-06-03 10:30", "2024-06-06", seed=17),
                              h_early, cfg)
    assert fx2.h1_at(ts("2024-06-03 10:45")).available


def test_decision_time_accessors_require_T(cfg):
    m = make_m15("2024-06-03", "2024-06-08", seed=19)
    fx = IndexFeatures.build("US30", m, to_h1(m), cfg)
    d = date(2024, 6, 5)
    with pytest.raises(TypeError):
        fx.is_valid_date(d)
    with pytest.raises(TypeError):
        fx.ib(d, 60)
    with pytest.raises(ValueError):
        fx.ib(d, 60, pd.Timestamp("2024-06-05 16:00"))          # naive time
    # The causal and after-the-fact forms agree once the date is known.
    assert fx.is_valid_date(d, ts("2024-06-05 20:00")) == fx.is_valid_date_eod(d)
    x, y = fx.ib(d, 60, ts("2024-06-05 20:00")), fx.ib_final(d, 60)
    assert (x.ibh, x.ibl, x.defined_utc) == (y.ibh, y.ibl, y.defined_utc)


# ---------------------------------------------------------------------------------------------
# Look-ahead (COMMON 2; ER-15)
# ---------------------------------------------------------------------------------------------

FEATURE_COLS = ["close_utc", "local_date", "local_min", "segment", "tv_bad", "is_cash_open_bar",
                "in_session", "valid_date", "tr", "atr14", "todv", "rs_4h", "twap", "spread_pts"]
H1_COLS = ["close_utc", "segment", "tr", "atr14", "rs_day"]


def _future_variants(m15, h1, T, seed):
    rng = np.random.default_rng(seed)
    pm, ph = m15.copy(), h1.copy()
    for df, minutes in ((pm, 15), (ph, 60)):
        fut = df["time_utc"] + pd.Timedelta(minutes=minutes) > T
        k = int(fut.sum())
        shift = rng.normal(0, 50, k)
        for c in ("open", "high", "low", "close"):
            df.loc[fut, c] = df.loc[fut, c] + shift + rng.normal(0, 5, k)
        df.loc[fut, "high"] = df.loc[fut, ["open", "high", "low", "close"]].max(axis=1)
        df.loc[fut, "low"] = df.loc[fut, ["open", "high", "low", "close"]].min(axis=1)
        df.loc[fut, "spread"] = rng.integers(0, 999, k).astype(df["spread"].dtype)
    trunc_m = m15[m15["time_utc"] + pd.Timedelta(minutes=15) <= T]
    trunc_h = h1[h1["time_utc"] + pd.Timedelta(minutes=60) <= T]
    return (pm, ph), (trunc_m, trunc_h)


@pytest.mark.parametrize("key, T", [
    ("US30", "2024-07-10 15:00"),      # full hour, US session
    ("US30", "2024-07-10 14:15"),      # inside the IB
    ("NAS100", "2024-07-15 13:45"),    # Monday, right after the cash open
    ("GER40", "2024-07-11 08:30"),
])
def test_no_look_ahead(cfg, key, T):
    T = ts(T)
    m15 = make_m15("2024-05-20", "2024-07-20", seed=14,
                   drop=["2024-07-09 15:30", "2024-07-10 18:00", "2024-07-10 18:15"])
    h1 = to_h1(m15)
    h1 = h1[h1["time_utc"] != ts("2024-07-10 18:00")].reset_index(drop=True)
    base = IndexFeatures.build(key, m15, h1, cfg)
    T_dates = base.m15.loc[base.m15["close_utc"] <= T, "local_date"].unique()
    Ts = base.m15.loc[base.m15["close_utc"] <= T, "close_utc"]
    for i, (vm, vh) in enumerate(_future_variants(m15, h1, T, seed=15)):
        alt = IndexFeatures.build(key, vm, vh, cfg)
        a = base.m15[base.m15["close_utc"] <= T][FEATURE_COLS].reset_index(drop=True)
        b = alt.m15[alt.m15["close_utc"] <= T][FEATURE_COLS].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        pd.testing.assert_frame_equal(
            base.h1[base.h1["close_utc"] <= T][H1_COLS].reset_index(drop=True),
            alt.h1[alt.h1["close_utc"] <= T][H1_COLS].reset_index(drop=True), check_exact=True)
        pa, ca = base.h1_positions(Ts)
        pb, cb = alt.h1_positions(Ts)
        assert (pa == pb).all() and (ca == cb).all(), i
        for d in T_dates:
            assert base.datr(d) == alt.datr(d) or (np.isnan(base.datr(d)) and np.isnan(alt.datr(d)))
            for mins in (30, 60, 90):
                x, y = base.ib(d, mins, T=T), alt.ib(d, mins, T=T)
                assert (x.available, x.cause) == (y.available, y.cause), (d, mins)
                if x.available:
                    assert (x.ibh, x.ibl, x.w, x.defined_utc) == (y.ibh, y.ibl, y.w, y.defined_utc)
        for d in T_dates:
            assert base.is_valid_date(d, T=T) == alt.is_valid_date(d, T=T), d
        prior = [d for d in T_dates if d < base.local_date(T)]
        assert base.dates.loc[prior, "valid"].equals(alt.dates.loc[prior, "valid"])
        assert base.local_date(T) == alt.local_date(T)


# ---------------------------------------------------------------------------------------------
# FeatureService
# ---------------------------------------------------------------------------------------------

def test_feature_service_cache_and_accessors(cfg):
    m = make_m15("2024-06-03", "2024-07-20", seed=16)
    h = to_h1(m)
    svc = FeatureService(cfg)
    fx = svc.build("US30", m, h)
    assert svc.build("US30", m.copy(), h.copy()) is fx and "US30" in svc and svc.get("US30") is fx
    m2 = m.copy()
    m2.loc[len(m2) - 1, "close"] += 1
    assert svc.build("US30", m2, h) is not fx
    T = ts("2024-07-16 15:00")
    row = fx.m15_at(T)
    assert row["time_utc"] == ts("2024-07-16 14:45") and fx.m15_pos(T) >= 0
    assert fx.m15_at(ts("2024-07-13 15:00")) is None                # Saturday: no bar
    assert fx.local_date(T) == date(2024, 7, 16)
    assert fx.local_date(ts("2024-07-17 04:15")) == date(2024, 7, 17)   # 00:00 New York bar
    assert fx.is_valid_date_eod(date(2024, 7, 16)) and not fx.is_valid_date_eod(date(2024, 7, 4))
    assert not fx.is_valid_date_eod(date(2024, 7, 13))
    # 2024-07-04 is a holiday: invalid even once its 09:30 bar has closed.
    assert not fx.is_valid_date(date(2024, 7, 4), ts("2024-07-04 16:00"))
    # Causal form: unknown (False) until the 09:30 bar has closed at 13:45 UTC.
    assert not fx.is_valid_date(date(2024, 7, 16), T=ts("2024-07-16 13:30"))
    assert fx.is_valid_date(date(2024, 7, 16), T=ts("2024-07-16 13:45"))
    assert np.isfinite(fx.datr(date(2024, 7, 16))) and np.isnan(fx.datr(date(2030, 1, 1)))
    assert fx.ib_table(60) is fx.ib_table(60)
    assert fx.h1_at(T).row["rs_day"] == fx.h1.set_index("close_utc").at[T, "rs_day"]
    assert is_calendar_trading_date("NYSE", date(2024, 7, 16))
    assert fx.exchange.cash_open == time(9, 30) and fx.tz == "America/New_York"
