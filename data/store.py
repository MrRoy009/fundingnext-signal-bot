"""Parquet bar storage with a locked final-holdout period."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from core.config import ROOT
from core.timeutil import server_to_utc, utc_now

log = logging.getLogger(__name__)


class HoldoutLocked(PermissionError):
    pass


class BarStore:
    def __init__(self, cfg: dict, root: Path | None = None):
        self.excluded = (cfg.get("data_quality") or {}).get("excluded_windows") or ()
        self.dir = (root or ROOT) / cfg.get("data_dir", "data/store")
        self.dir.mkdir(parents=True, exist_ok=True)
        start = (cfg.get("holdout") or {}).get("start")
        self.holdout_start = pd.Timestamp(start, tz="UTC") if start else None
        self.access_log = self.dir / "holdout_access.log"

    def path(self, key: str, tf: str) -> Path:
        return self.dir / f"{key}_{tf}.parquet"

    def last_time(self, key: str, tf: str) -> pd.Timestamp | None:
        p = self.path(key, tf)
        if not p.exists():
            return None
        df = pd.read_parquet(p, columns=["time_utc"])
        return df["time_utc"].max() if len(df) else None

    def first_time(self, key: str, tf: str) -> pd.Timestamp | None:
        p = self.path(key, tf)
        if not p.exists():
            return None
        df = pd.read_parquet(p, columns=["time_utc"])
        return df["time_utc"].min() if len(df) else None

    def rebase_utc(self, key: str, tf: str, zone: str, shift_hours: float) -> bool:
        """Recompute time_utc from the stored server_time with the current clock rule.
        Returns True if anything changed."""
        p = self.path(key, tf)
        if not p.exists():
            return False
        df = pd.read_parquet(p)
        fresh = server_to_utc(df["server_time"].to_numpy(), zone, shift_hours)
        if (pd.DatetimeIndex(df["time_utc"]) == fresh).all():
            return False
        df["time_utc"] = fresh
        df.to_parquet(p, index=False)
        log.warning("Re-based %s %s timestamps to %s %+gh", key, tf, zone, shift_hours)
        return True

    def merge(self, key: str, tf: str, new: pd.DataFrame) -> pd.DataFrame:
        p = self.path(key, tf)
        df = pd.concat([pd.read_parquet(p), new]) if p.exists() else new
        df = df.drop_duplicates("time_utc", keep="last").sort_values("time_utc")
        df = df.reset_index(drop=True)
        df.to_parquet(p, index=False)
        return df

    def drop_excluded(self, key: str, df: pd.DataFrame) -> pd.DataFrame:
        """Drop bars inside config data_quality.excluded_windows for this instrument
        (frozen or corrupt prices). The gap left behind starts a new segment downstream."""
        keep = pd.Series(True, index=df.index)
        for w in self.excluded:
            if key in w.get("instruments", ()):
                keep &= ~((df["time_utc"] >= pd.Timestamp(w["start"]))
                          & (df["time_utc"] < pd.Timestamp(w["end"])))
        if not keep.all():
            log.info("Dropped %d %s bars in excluded windows", int((~keep).sum()), key)
        return df[keep].reset_index(drop=True)

    def load(self, key: str, tf: str, *, final_holdout: bool = False,
             purpose: str = "", raw: bool = False) -> pd.DataFrame:
        """Load bars. Bars in configured excluded windows are dropped unless raw=True. Bars on
        or after holdout.start are dropped unless final_holdout=True, and every holdout access
        is logged with a data hash."""
        df = pd.read_parquet(self.path(key, tf))
        if not raw:
            df = self.drop_excluded(key, df)
        if self.holdout_start is None:
            return df
        if not final_holdout:
            return df[df["time_utc"] < self.holdout_start].reset_index(drop=True)
        if not purpose:
            raise HoldoutLocked("Opening the final holdout requires a stated purpose")
        digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values).hexdigest()
        with open(self.access_log, "a", encoding="utf-8") as f:
            f.write(f"{utc_now().isoformat()}\t{key}\t{tf}\t{digest[:16]}\t{purpose}\n")
        log.warning("FINAL HOLDOUT opened for %s %s: %s", key, tf, purpose)
        return df
