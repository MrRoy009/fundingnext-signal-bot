"""FeatureService: COMMON features per index, built once and read at decision time T (ER-06).

Strategies and the backtester read features only through `IndexFeatures` accessors keyed by the
decision time T (the close of M15 bar t, tz-aware UTC). Every accessor returns only values that
are knowable at T: the M15 row closing at T, the H1 row chosen by the COMMON 2 alignment rule,
the IB once its last bar has closed, and DATR from prior dates.

Public API

    svc = FeatureService(cfg)
    fx = svc.build(key, m15, h1)            # IndexFeatures; cached until the frames change
    fx = svc.get(key)                       # last build for key (KeyError if none)
    svc = FeatureService.from_store(cfg, keys, store=None)   # research data via BarStore.load

    fx.m15                                   # DataFrame, see features.core.compute_m15_features
    fx.h1                                    # DataFrame, see features.core.compute_h1_features
    fx.dates                                 # per local date: valid, datr, n_session_bars, ...
    fx.m15_at(T) -> pd.Series | None         # M15 row whose close == T (None if the bar is absent)
    fx.m15_pos(T) -> int                     # positional index of that row, -1 if absent
    fx.h1_at(T) -> H1Alignment               # COMMON 2 H1 rule: .row or .cause (UNAVAILABLE);
                                             # also UNAVAILABLE for an H1 bar from before a hole
    fx.h1_positions(Ts) -> (pos, cause)      # vectorised h1_at for the backtester
    fx.ib(local_date, ib_minutes, T) -> IB   # NaN levels + cause unless complete and closed by T
    fx.ib_final(local_date, ib_minutes) -> IB     # whole-date IB, reporting only
    fx.ib_table(ib_minutes) -> DataFrame     # all dates, cached per ib_minutes (whole-date)
    fx.datr(local_date) -> float             # NaN if undefined (prior dates only, so causal)
    fx.is_valid_date(local_date, T) -> bool  # COMMON 1.6 (calendar + cash-open bar) as known at
                                             # T: False until the cash-open bar has closed by T
    fx.is_valid_date_eod(local_date) -> bool # whole-date validity, reporting / prior dates only
    fx.local_date(T) -> date                 # local date of the M15 bar closing at T (open time)
    fx.tz, fx.exchange                       # exchange-local zone and session

M15 row columns used by strategies: rs_4h, todv, atr14, twap, close/high/low/open, valid_date
(causal), in_session, segment, spread_pts, tv_bad. H1 row columns: atr14, rs_day, close, segment.

Look-ahead: `fx.m15` / `fx.h1` rows closing <= T, `h1_at(T)`, `ib(d, m, T)`, `datr(d)` and
`is_valid_date(d, T)` are causal; T is a required argument wherever the answer depends on it.
The `*_final` / `*_eod` forms, `ib_table`, and the `fx.dates` columns valid, cash_open_present,
n_session_bars and session_range describe whole dates and must not be read for the CURRENT
local date at T.

Known data issue: prices are frozen inside the 2024-10-04..2025-01-08 tick-volume window, so
features there (and in lookbacks shortly after) are unreliable; see features.core and the
escalation to Quant/owner.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from core.calendar import Exchange, exchange_for, local_to_utc
from core.config import tf_minutes
from features.core import compute_h1_features, compute_m15_features, h1_alignment, ib_table

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class H1Alignment:
    """Result of the COMMON 2 H1 rule at T. `row` is None exactly when `cause` is set."""
    T: pd.Timestamp
    row: pd.Series | None
    cause: str | None

    @property
    def available(self) -> bool:
        return self.row is not None


@dataclass(frozen=True)
class IB:
    """Initial balance of one local date for one ib_minutes value. Levels are NaN and `cause`
    is set when the IB is incomplete or not yet defined at the T asked for."""
    local_date: date
    ib_minutes: int
    ibh: float
    ibl: float
    w: float
    defined_utc: pd.Timestamp | None     # close of the last IB bar
    cause: str | None = None

    @property
    def available(self) -> bool:
        return self.cause is None


def _ts(T) -> pd.Timestamp:
    t = pd.Timestamp(T)
    if t.tzinfo is None:
        raise ValueError("Decision times must be tz-aware UTC timestamps")
    return t.tz_convert("UTC").as_unit("ns")


def _fingerprint(df: pd.DataFrame) -> str:
    """Content hash, so a cached build is reused only for identical frames."""
    return pd.util.hash_pandas_object(df, index=False).sum().__repr__() + f":{len(df)}"


@dataclass
class IndexFeatures:
    key: str
    m15: pd.DataFrame
    h1: pd.DataFrame
    dates: pd.DataFrame
    exchange: Exchange
    cfg: dict = field(repr=False)
    _m15_close: pd.Index = field(init=False, repr=False)
    _ib_cache: dict = field(init=False, repr=False, default_factory=dict)
    _bar_min: int = field(init=False, repr=False, default=15)

    def __post_init__(self) -> None:
        self._m15_close = pd.Index(self.m15["close_utc"])
        self._bar_min = tf_minutes(self.cfg["timeframes"]["entry"])

    @classmethod
    def build(cls, key: str, m15: pd.DataFrame, h1: pd.DataFrame, cfg: dict) -> IndexFeatures:
        f, dates = compute_m15_features(m15, key, cfg)
        h = compute_h1_features(h1, key, cfg)
        return cls(key, f, h, dates, exchange_for(cfg, key), cfg)

    @property
    def tz(self) -> str:
        return self.exchange.tz

    # --- M15 ---------------------------------------------------------------------------------
    def m15_pos(self, T) -> int:
        return int(self._m15_close.get_indexer([_ts(T)])[0])

    def m15_at(self, T) -> pd.Series | None:
        """Row of the M15 bar closing exactly at T; None if that bar is absent (COMMON 4.2.2)."""
        pos = self.m15_pos(T)
        return None if pos < 0 else self.m15.iloc[pos]

    def local_date(self, T) -> date:
        """Exchange-local date of the M15 bar closing at T, taken from its open time."""
        return (_ts(T) - pd.Timedelta(minutes=self._bar_min)).tz_convert(self.tz).date()

    # --- H1 ----------------------------------------------------------------------------------
    def h1_positions(self, Ts) -> tuple[np.ndarray, np.ndarray]:
        return h1_alignment(Ts, self.h1["close_utc"], self.m15["time_utc"], self._bar_min,
                            self.m15["segment"])

    def h1_at(self, T) -> H1Alignment:
        T = _ts(T)
        pos, cause = self.h1_positions(T)
        if pos[0] < 0:
            return H1Alignment(T, None, str(cause[0]))
        return H1Alignment(T, self.h1.iloc[int(pos[0])], None)

    # --- dates -------------------------------------------------------------------------------
    def is_valid_date(self, d: date, T) -> bool:
        """COMMON 1.6 valid trading date as known at T: False while the cash-open bar of d has
        not closed by T, since its presence is not yet known. T is required so that the
        decision-time path can never read whole-day information by default."""
        open_close = local_to_utc(d, self.exchange.cash_open, self.tz) + pd.Timedelta(
            minutes=self._bar_min)
        if _ts(T) < open_close:
            return False
        return self.is_valid_date_eod(d)

    def is_valid_date_eod(self, d: date) -> bool:
        """After-the-fact validity of d (whole date known). Reporting and lookbacks over
        PRIOR dates only; never call it for the current date at decision time."""
        return bool(d in self.dates.index and self.dates.at[d, "valid"])

    def datr(self, d: date) -> float:
        return float(self.dates.at[d, "datr"]) if d in self.dates.index else float("nan")

    # --- IB ----------------------------------------------------------------------------------
    def ib_table(self, ib_minutes: int) -> pd.DataFrame:
        if ib_minutes not in self._ib_cache:
            self._ib_cache[ib_minutes] = ib_table(self.m15, self.key, self.cfg, ib_minutes)
        return self._ib_cache[ib_minutes]

    def ib(self, d: date, ib_minutes: int, T) -> IB:
        """IB of local date d as known at T: available only once its last bar closed by T.
        T is required; `ib_final` is the after-the-fact form for reporting."""
        T = _ts(T)
        # Clock check first, so the answer before the IB end never depends on whether the
        # later IB bars exist (that would leak data from after T).
        end = (datetime.combine(d, self.exchange.cash_open) + timedelta(minutes=ib_minutes)).time()
        end_utc = local_to_utc(d, end, self.tz)
        if T < end_utc:
            nan = float("nan")
            return IB(d, ib_minutes, nan, nan, nan, None, f"IB not defined before {end_utc}")
        return self.ib_final(d, ib_minutes)

    def ib_final(self, d: date, ib_minutes: int) -> IB:
        """IB of d with the whole date known. Reporting only, never at decision time."""
        tab = self.ib_table(ib_minutes)
        nan = float("nan")
        if d not in tab.index or not tab.at[d, "complete"]:
            return IB(d, ib_minutes, nan, nan, nan, None, "IB incomplete: missing IB bar")
        r = tab.loc[d]
        return IB(d, ib_minutes, float(r["ibh"]), float(r["ibl"]), float(r["w"]),
                  pd.Timestamp(r["defined_utc"]))


class FeatureService:
    """Builds and caches one IndexFeatures per index. Rebuilds only when its frames change."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._cache: dict[str, tuple[str, IndexFeatures]] = {}

    def build(self, key: str, m15: pd.DataFrame, h1: pd.DataFrame) -> IndexFeatures:
        fp = _fingerprint(m15) + "|" + _fingerprint(h1)
        hit = self._cache.get(key)
        if hit and hit[0] == fp:
            return hit[1]
        fx = IndexFeatures.build(key, m15, h1, self.cfg)
        self._cache[key] = (fp, fx)
        log.debug("Features built for %s: %d M15, %d H1 bars", key, len(fx.m15), len(fx.h1))
        return fx

    def get(self, key: str) -> IndexFeatures:
        return self._cache[key][1]

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    @classmethod
    def from_store(cls, cfg: dict, keys, store=None) -> FeatureService:
        """Research path: pre-holdout bars from BarStore.load (never the final holdout)."""
        from data.store import BarStore
        store = store or BarStore(cfg)
        svc = cls(cfg)
        entry, regime = cfg["timeframes"]["entry"], cfg["timeframes"]["regime"]
        for key in keys:
            svc.build(key, store.load(key, entry), store.load(key, regime))
        return svc
