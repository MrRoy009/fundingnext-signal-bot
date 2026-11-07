"""Load config.yaml and .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

TIMEFRAME_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}


def tf_minutes(tf: str) -> int:
    try:
        return TIMEFRAME_MINUTES[tf]
    except KeyError:
        raise ValueError(f"Unsupported timeframe {tf!r}; use one of {sorted(TIMEFRAME_MINUTES)}")


def load_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("instruments", "timeframes", "server_timezone", "risk", "rules"):
        if key not in cfg:
            raise ValueError(f"config.yaml is missing '{key}'")
    tf_minutes(cfg["timeframes"]["entry"])
    tf_minutes(cfg["timeframes"]["regime"])
    return cfg


@dataclass(frozen=True)
class Credentials:
    login: int | None
    password: str | None
    server: str | None
    path: str | None
    competition_login: int | None
    portable: bool = False


def _int_or_none(value: str | None) -> int | None:
    value = (value or "").strip()
    return int(value) if value else None


def load_credentials() -> Credentials:
    load_dotenv(ROOT / ".env")
    return Credentials(
        login=_int_or_none(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD") or None,
        server=os.getenv("MT5_SERVER") or None,
        path=os.getenv("MT5_PATH") or None,
        competition_login=_int_or_none(os.getenv("COMPETITION_LOGIN")),
        portable=(os.getenv("MT5_PORTABLE") or "").strip().lower() in ("1", "true", "yes"),
    )
