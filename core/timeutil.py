"""Broker server time <-> UTC.

MT5 returns bar times as the broker's wall-clock time packed into a Unix timestamp, as if
that wall clock were UTC. The server clock is modelled as an IANA zone plus a fixed
shift, e.g. New York + 7h ("NY close" brokers). The zone's UTC offset is taken at noon of
each local date: DST switches happen early on Sunday morning, when index CFDs are closed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def server_to_utc(server_epoch, zone: str, shift_hours: float) -> pd.DatetimeIndex:
    """Convert MT5 server-time epochs (seconds) to a tz-aware UTC DatetimeIndex."""
    wall = pd.to_datetime(np.asarray(server_epoch, dtype="int64"), unit="s")
    zone_local = wall - pd.Timedelta(hours=shift_hours)
    dates = zone_local.normalize()
    offsets = {
        d: (d + pd.Timedelta(hours=12)).tz_localize(zone).utcoffset()
        for d in pd.unique(dates)
    }
    offset = pd.to_timedelta([offsets[d] for d in dates])
    return pd.DatetimeIndex(zone_local - offset).tz_localize("UTC")


def server_offset(utc_ts: datetime, zone: str, shift_hours: float) -> timedelta:
    """Server clock minus UTC at a given UTC instant."""
    ts = pd.Timestamp(utc_ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.tz_convert(zone).utcoffset() + timedelta(hours=shift_hours)


def utc_to_server_naive(utc_ts: datetime, zone: str, shift_hours: float) -> datetime:
    """UTC instant -> server wall clock, returned as a UTC-tagged datetime (MT5's convention
    for copy_rates_range arguments)."""
    ts = pd.Timestamp(utc_ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    wall = (ts + server_offset(ts, zone, shift_hours)).tz_localize(None)
    return wall.to_pydatetime().replace(tzinfo=timezone.utc)


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")
