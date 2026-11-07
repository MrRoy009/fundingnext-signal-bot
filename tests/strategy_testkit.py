"""Shared test utilities for strategies (synthetic data only; never real bars).

What it gives a strategy test:
  make_market(...)                    synthetic multi-index M15 + H1 bid bars across New York and
                                      Berlin DST changes (2024-03-10, 2024-03-31: the March
                                      mismatch weeks), exchange holidays, single missing bars, a
                                      missing cash-open bar and a > 7-day hole (segment break)
  SyntheticMarket.truncated(T)        bars with close <= T (what live sees at T)
  SyntheticMarket.perturbed_after(T)  bars with close > T changed and partly dropped
  ProxyRegimeTable / proxy_regime_tables(market, cfg)
                                      a causal stand-in for regime.service.RegimeTable (same
                                      at_many / threshold interface) computed from H1 bars, so a
                                      test needs no HMM fit; it changes when H1 bars change and
                                      applies the COMMON 6.6 live rule (a core H1 bar closing
                                      exactly at T absent -> "regime H1 bar missing")
  build_services(cfg, market)         (FeatureService, {key: regime table})
  assert_no_lookahead(...)            (a) perturbing bars after T leaves every candidate at or
                                      before T unchanged
  assert_generate_matches_scan(...)   (b) live generate() at sampled T on truncated history equals
                                      scan() over the full history
  sample_times(...)                   decision times worth checking (candidate times first)

Usage in a strategy test:

    from tests.strategy_testkit import make_market, assert_no_lookahead, ...
    market = make_market()
    strat = MyStrategy(cfg)
    full = assert_no_lookahead(strat, cfg, market, cuts=[...], start=market.scan_start)
    assert_generate_matches_scan(strat, cfg, market, sample_times(strat, cfg, full, ...),
                                 start=market.scan_start)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from datetime import time
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from core.calendar import exchange_for, is_calendar_trading_date, local_to_utc
from features.service import FeatureService
from strategies.base import Candidate, StrategyContext
from strategies.context import window_schedule

UTC = "UTC"
KEYS = ("US30", "NAS100", "GER40")
BASE_PRICE = {"US30": 39000.0, "NAS100": 18000.0, "GER40": 17500.0, "US500": 5200.0}
M15 = pd.Timedelta(minutes=15)
H1 = pd.Timedelta(hours=1)

# Default defects (UTC bar OPEN times), each chosen to hit a strategy window:
DEFAULT_DROPS = {
    # 10:30-10:45 New York bar (EST): incomplete US group at 10:45 and RS_4h NaN after it.
    "NAS100": ["2024-02-22 15:30"],
    # US30 cash-open bar (09:30 New York, EST): not a valid trading date for US30 that day.
    "US30": ["2024-03-05 14:30"],
}
# GER40 hole of more than 7 days: a segment break, so every lookback restarts after it.
DEFAULT_HOLES = {"GER40": [("2024-02-26", "2024-03-05")]}


def ts(s) -> pd.Timestamp:
    t = pd.Timestamp(s)
    return (t.tz_localize(UTC) if t.tzinfo is None else t.tz_convert(UTC)).as_unit("ns")


def to_h1(m15: pd.DataFrame) -> pd.DataFrame:
    """H1 bars aggregated from M15 (open-time labels); hours without M15 bars are absent."""
    if not len(m15):
        return m15.iloc[0:0].copy()
    g = m15.set_index("time_utc").resample("1h")
    h = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "tick_volume": g["tick_volume"].sum(),
                      "spread": g["spread"].min()}).dropna(subset=["open"])
    h["tick_volume"] = h["tick_volume"].astype("uint64")
    h["spread"] = h["spread"].astype("int32")
    return h.reset_index()


def make_m15(key: str, start: str, end: str, seed: int = 0, drop: Iterable = (),
             holes: Iterable = ()) -> pd.DataFrame:
    """Weekday M15 bid bars around the clock with a daily break 21:00-22:00 UTC. Volatility is
    higher in the hours after each cash open and varies by day, and each day has its own drift,
    so normalised scores cross typical thresholds in both directions."""
    t = pd.date_range(start, end, freq="15min", tz=UTC, inclusive="left").as_unit("ns")
    t = t[(t.day_of_week < 5) & (t.hour != 21)]
    for a, b in holes:
        t = t[(t < ts(a)) | (t >= ts(b))]
    drop = pd.DatetimeIndex([ts(d) for d in drop])
    if len(drop):
        t = t[~t.isin(drop)]
    rng = np.random.default_rng(seed)
    n = len(t)
    base = BASE_PRICE.get(key, 10000.0)
    hour = t.hour.to_numpy()
    busy = (hour >= 13) & (hour <= 16) if key != "GER40" else (hour >= 7) & (hour <= 10)
    day = (t.normalize().asi8 // (86400 * 10**9)).astype(np.int64)
    _, day_idx = np.unique(day, return_inverse=True)
    day_vol = rng.uniform(0.6, 1.6, day_idx.max() + 1)[day_idx]
    day_drift = rng.normal(0.0, 0.35, day_idx.max() + 1)[day_idx]
    sigma = 0.0006 * np.where(busy, 1.8, 1.0) * day_vol
    r = sigma * (rng.standard_normal(n) + day_drift)
    close = base * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.00005, n))
    wick = np.abs(rng.normal(0, 0.5, (2, n))) * sigma * base
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    return pd.DataFrame({"time_utc": t, "open": open_, "high": high, "low": low, "close": close,
                         "tick_volume": rng.integers(100, 2000, n).astype("uint64"),
                         "spread": rng.integers(50, 250, n).astype("int32")})


@dataclass
class SyntheticMarket:
    m15: dict[str, pd.DataFrame]
    h1: dict[str, pd.DataFrame]
    scan_start: pd.Timestamp          # first date with full feature lookbacks (about 4 weeks in)
    end: pd.Timestamp                 # end of the generated history
    meta: dict = field(default_factory=dict)

    @property
    def keys(self) -> list[str]:
        return list(self.m15)

    def truncated(self, T) -> SyntheticMarket:
        """What live sees at T: bars with close time <= T."""
        T = ts(T)
        m = {k: v[v["time_utc"] + M15 <= T].reset_index(drop=True) for k, v in self.m15.items()}
        h = {k: v[v["time_utc"] + H1 <= T].reset_index(drop=True) for k, v in self.h1.items()}
        return SyntheticMarket(m, h, self.scan_start, T, dict(self.meta))

    def perturbed_after(self, T, seed: int = 1, drop_frac: float = 0.15) -> SyntheticMarket:
        """Every M15 bar with close > T (open >= T) gets a random level shift and range change,
        and a share of them is dropped; H1 bars with close > T are rebuilt from the perturbed
        M15. Bars with close <= T are untouched."""
        T = ts(T)
        rng = np.random.default_rng(seed)
        m, h = {}, {}
        for k, df in self.m15.items():
            after = df["time_utc"] >= T
            out = df.copy()
            n = int(after.sum())
            if n:
                base = BASE_PRICE.get(k, 10000.0)
                shift = np.cumsum(rng.normal(0, 0.004 * base, n))
                scale = rng.uniform(0.3, 3.0, n)
                rows = out.loc[after]
                mid = (rows["high"] + rows["low"]) / 2 + shift
                half = (rows["high"] - rows["low"]) / 2 * scale
                out.loc[after, "high"] = mid + half
                out.loc[after, "low"] = mid - half
                out.loc[after, "open"] = mid + half * rng.uniform(-1, 1, n)
                out.loc[after, "close"] = mid + half * rng.uniform(-1, 1, n)
                keep = ~after.to_numpy() | (rng.uniform(0, 1, len(out)) >= drop_frac)
                out = out[keep].reset_index(drop=True)
            m[k] = out
            old = self.h1[k]
            new = to_h1(out)
            h[k] = pd.concat([old[old["time_utc"] + H1 <= T], new[new["time_utc"] + H1 > T]],
                             ignore_index=True)
        return SyntheticMarket(m, h, self.scan_start, self.end, dict(self.meta))


def make_market(start: str = "2024-01-08", end: str = "2024-04-13", keys: Sequence[str] = KEYS,
                seed: int = 7, drops: Mapping[str, Iterable] | None = None,
                holes: Mapping[str, Iterable] | None = None,
                h1_drops: Mapping[str, Iterable] | None = None) -> SyntheticMarket:
    """Synthetic market for `keys`. Defaults span the US (2024-03-10) and EU (2024-03-31) DST
    starts, NYSE holidays 2024-01-15, 2024-02-19 and Good Friday 2024-03-29 (also Xetra), Easter
    Monday 2024-04-01 (Xetra), plus DEFAULT_DROPS / DEFAULT_HOLES. Pass {} to disable defects.
    h1_drops: H1 bars (UTC open times) removed while their M15 bars stay, as when the H1 feed
    lags the M15 feed; this exercises the COMMON 6.6 live rule and the COMMON 2 H1 alignment."""
    drops = DEFAULT_DROPS if drops is None else drops
    holes = DEFAULT_HOLES if holes is None else holes
    h1_drops = h1_drops or {}
    m15, h1 = {}, {}
    for n, k in enumerate(keys):
        m15[k] = make_m15(k, start, end, seed + 101 * n, drops.get(k, ()), holes.get(k, ()))
        h = to_h1(m15[k])
        gone = pd.DatetimeIndex([ts(d) for d in h1_drops.get(k, ())])
        h1[k] = h[~h["time_utc"].isin(gone)].reset_index(drop=True) if len(gone) else h
    return SyntheticMarket(m15, h1, ts(start) + pd.Timedelta(days=28), ts(end),
                           {"drops": dict(drops), "holes": dict(holes),
                            "h1_drops": dict(h1_drops)})


# ---------------------------------------------------------------------------------------------
# Causal regime stand-in
# ---------------------------------------------------------------------------------------------

def missing_core_h1_closes(cfg: dict, key: str, m15: pd.DataFrame, h1: pd.DataFrame
                           ) -> np.ndarray:
    """Close times (UTC ns) of core H1 slots (cfg regime.core_hours_local) on valid trading dates
    (calendar trading date with the cash-open M15 bar, COMMON 1.6) that have no H1 bar: the
    slots the COMMON 6.6 live rule makes UNAVAILABLE, built like regime.inputs.build_slots
    (no slot before the first bar or closing after the last bar close). Causal: a slot closing
    at T depends only on bars closed by T."""
    if not len(m15):
        return np.zeros(0, np.int64)
    ex = exchange_for(cfg, key)
    h_first, h_last = (int(h) for h in cfg["regime"]["core_hours_local"][key])
    t = pd.DatetimeIndex(pd.to_datetime(m15["time_utc"], utc=True)).as_unit("ns")
    local = t.tz_convert(ex.tz)
    opens = {d for d, c in zip(local.date, local.time) if c == ex.cash_open}
    have = set(pd.DatetimeIndex(pd.to_datetime(h1["time_utc"], utc=True)).as_unit("ns").asi8) \
        if len(h1) else set()
    as_of = (t.max() + M15).value
    if len(h1):
        as_of = max(as_of, (pd.Timestamp(h1["time_utc"].max()) + H1).value)
    first = t.min().value
    out = []
    for d in sorted(opens):
        if not is_calendar_trading_date(ex.calendar, d):
            continue
        for hr in range(h_first, h_last + 1):
            o = local_to_utc(d, time(hr), ex.tz).as_unit("ns").value
            c = o + H1.value
            if o >= first and c <= as_of and o not in have:
                out.append(c)
    return np.array(sorted(out), dtype=np.int64)


@dataclass
class ProxyRegimeTable:
    """RegimeTable stand-in: V_T = ln(mean H1 range of the last 4 bars / last 40 bars) at the
    latest H1 bar closed by T. UNAVAILABLE during warm-up, when that bar is more than
    `stale_days` old, inside `unavailable` windows, and (live_rule, the default as in
    RegimeTable.at_many) when the core H1 bar closing exactly at T is absent ("regime H1 bar
    missing", COMMON 6.6); the last needs `cfg` and `m15` to know the core slots of valid dates.
    Pure function of bars closed by T."""
    key: str
    h1: pd.DataFrame
    threshold: float = 0.0
    stale_days: float = 5.0
    unavailable: tuple = ()
    cfg: dict | None = field(default=None, repr=False)
    m15: pd.DataFrame | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        h = self.h1
        close = pd.to_datetime(h["time_utc"], utc=True) + H1 if len(h) else pd.Series(
            [], dtype="datetime64[ns, UTC]")
        self._close = pd.DatetimeIndex(close).as_unit("ns").asi8
        rng = (h["high"] - h["low"]).astype(float) if len(h) else pd.Series([], dtype=float)
        self._v = np.log(rng.rolling(4).mean() / rng.rolling(40).mean()).to_numpy()
        self._miss = (missing_core_h1_closes(self.cfg, self.key, self.m15, h)
                      if self.cfg is not None and self.m15 is not None
                      else np.zeros(0, np.int64))

    def at_many(self, times, live_rule: bool = True) -> pd.DataFrame:
        t = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit("ns")
        j = np.searchsorted(self._close, t.asi8, side="right") - 1
        v = np.where(j >= 0, self._v[np.maximum(j, 0)] if len(self._v) else np.nan, np.nan)
        last = np.where(j >= 0, self._close[np.maximum(j, 0)] if len(self._close) else 0, 0)
        status = np.full(len(t), "OK", dtype=object)
        cause = np.full(len(t), "", dtype=object)
        stale = (j >= 0) & (t.asi8 - last > pd.Timedelta(days=self.stale_days).value)
        warm = ~np.isfinite(v)
        off = np.zeros(len(t), bool)
        for a, b in self.unavailable:
            off |= (t.asi8 >= ts(a).value) & (t.asi8 < ts(b).value)
        missing = bool(live_rule) & np.isin(t.asi8, self._miss)
        for mask, why in ((warm, "proxy warm-up"), (stale, "stale"), (off, "test outage"),
                          (missing, "regime H1 bar missing")):
            status[mask], cause[mask] = "UNAVAILABLE", why
        ok = status == "OK"
        tag = np.where(ok, np.where(v >= self.threshold, "ACTIVE", "CALM"), "")
        label = np.where(ok, np.char.add(tag.astype(str), "_FLAT"), "")
        prob = np.where(ok, 1 / (1 + np.exp(-5 * np.abs(np.nan_to_num(v)))), np.nan)
        return pd.DataFrame({"time_utc": t, "status": status, "cause": cause,
                             "v_t": np.where(ok, v, np.nan), "tag": tag.astype(object),
                             "label": label.astype(object), "regime_prob": prob})


def proxy_regime_tables(market: SyntheticMarket, cfg: dict) -> dict[str, ProxyRegimeTable]:
    thr = float(cfg["regime"]["gate_threshold"])
    return {k: ProxyRegimeTable(k, market.h1[k], thr, cfg=cfg, m15=market.m15[k])
            for k in market.keys}


RegimeFactory = Callable[[SyntheticMarket, dict], Mapping[str, object]]


def build_services(cfg: dict, market: SyntheticMarket,
                   regime: RegimeFactory = proxy_regime_tables
                   ) -> tuple[FeatureService, Mapping[str, object]]:
    svc = FeatureService(cfg)
    for k in market.keys:
        svc.build(k, market.m15[k], market.h1[k])
    return svc, regime(market, cfg)


# ---------------------------------------------------------------------------------------------
# Candidate comparison
# ---------------------------------------------------------------------------------------------

def candidate_tuple(c: Candidate) -> tuple:
    """All Candidate fields, with NaN == NaN and timestamps normalised to UTC ns."""
    out = []
    for f in fields(Candidate):
        v = getattr(c, f.name)
        if isinstance(v, float) and math.isnan(v):
            v = "NaN"
        elif isinstance(v, pd.Timestamp):
            v = ts(v)
        out.append((f.name, v))
    return tuple(out)


def assert_candidates_equal(a: Sequence[Candidate], b: Sequence[Candidate], what: str = "") -> None:
    ta, tb = [candidate_tuple(c) for c in a], [candidate_tuple(c) for c in b]
    if ta != tb:
        lines = [f"candidates differ {what}: {len(ta)} vs {len(tb)}"]
        for x, y in zip(ta, tb):
            if x != y:
                diff = [(n, v, w) for (n, v), (_, w) in zip(x, y) if v != w]
                lines.append(f"  first difference: {diff}")
                break
        raise AssertionError("\n".join(lines))


# ---------------------------------------------------------------------------------------------
# (a) look-ahead and (b) live == batch
# ---------------------------------------------------------------------------------------------

def assert_no_lookahead(strategy, cfg: dict, market: SyntheticMarket, cuts: Iterable,
                        start=None, end=None, seeds: Sequence[int] = (1, 2),
                        regime: RegimeFactory = proxy_regime_tables) -> list[Candidate]:
    """For each cut time T and seed: scan the market with every bar after T perturbed and
    check that the candidates with signal_time <= T are identical to the unperturbed scan.
    Returns the unperturbed candidates."""
    start = market.scan_start if start is None else ts(start)
    end = market.end if end is None else ts(end)
    full = strategy.scan(*build_services(cfg, market, regime), start, end)
    for cut in cuts:
        cut = ts(cut)
        before = [c for c in full if c.signal_time <= cut]
        for seed in seeds:
            pert = market.perturbed_after(cut, seed)
            got = strategy.scan(*build_services(cfg, pert, regime), start, end)
            assert_candidates_equal([c for c in got if c.signal_time <= cut], before,
                                    f"after perturbing bars after {cut} (seed {seed})")
    return full


def assert_generate_matches_scan(strategy, cfg: dict, market: SyntheticMarket,
                                 times: Iterable, start=None,
                                 regime: RegimeFactory = proxy_regime_tables,
                                 full: Sequence[Candidate] | None = None) -> int:
    """For each T: generate() on history truncated at T (services rebuilt from the truncated
    bars) equals scan() over the full history at T. Any T >= `start` is fair: scan() itself
    replays the session of `start` from local midnight, as generate() does. Returns the number
    of candidates compared."""
    start = market.scan_start if start is None else ts(start)
    times = sorted(ts(t) for t in times)
    if full is None:
        full = strategy.scan(*build_services(cfg, market, regime), start, market.end)
    n = 0
    for T in times:
        if T < start:
            raise ValueError(f"{T} is before the scan start {start}")
        trunc = market.truncated(T)
        svc, tabs = build_services(cfg, trunc, regime)
        ctx = StrategyContext(features=svc, regime=None, now=T, cfg=cfg, regime_tables=tabs)
        live = strategy.generate(trunc.m15, trunc.h1, T, ctx)
        expected = [c for c in full if ts(c.signal_time) == T]
        assert_candidates_equal(live, expected, f"generate() vs scan() at {T}")
        n += len(expected)
    return n


def sample_times(strategy, cfg: dict, candidates: Sequence[Candidate], start, end,
                 n_candidates: int = 4, n_other: int = 3, seed: int = 0) -> list[pd.Timestamp]:
    """Decision times to check: some with candidates (spread over the list, so lockout replay
    after an earlier candidate is covered) and some random in-window times without."""
    rng = np.random.default_rng(seed)
    with_c = sorted({ts(c.signal_time) for c in candidates})
    pick = [with_c[i] for i in np.linspace(0, len(with_c) - 1, min(n_candidates, len(with_c)))
            .round().astype(int)] if with_c else []
    sched = window_schedule(cfg, strategy.windows(strategy.params), ts(start), ts(end))
    others = sorted(set(pd.DatetimeIndex(sched["T"])) - set(with_c))
    if others:
        pick += [others[i] for i in rng.choice(len(others), min(n_other, len(others)),
                                               replace=False)]
    return sorted(set(pick))
