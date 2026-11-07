"""GATE 1: connect to the demo account, map symbols, download history, report data quality.

    python -m data.download [--years N]

Also checks the server_timezone rule in config.yaml against the history itself and exits
with an error if another rule fits the data better.
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from core.config import ROOT, load_config, load_credentials, tf_minutes
from core.logging_setup import setup_logging
from core.timeutil import utc_now
from data.clock import fit_server_clock
from data.mt5_adapter import MT5Client
from data.quality import format_report, quality_report
from data.store import BarStore

log = logging.getLogger(__name__)
US_INDEX_KEYS = ("US30", "NAS100", "US500")


def download(client: MT5Client, store: BarStore, key: str, symbol: str, tf: str,
             years: float, now: pd.Timestamp) -> pd.DataFrame | None:
    store.rebase_utc(key, tf, client.zone, client.shift)
    target_start = now - pd.Timedelta(days=365 * years)
    first, last = store.first_time(key, tf), store.last_time(key, tf)
    ranges = []
    if last is None:
        ranges.append((target_start, now))
    else:
        if first > target_start + pd.Timedelta(days=7):
            ranges.append((target_start, first))        # backfill if the limit was raised
        ranges.append((last - pd.Timedelta(minutes=tf_minutes(tf)), now))
    frames = []
    for start, end in ranges:
        log.info("Downloading %s (%s) %s from %s to %s", key, symbol, tf, start, end)
        part = client.bars_range(symbol, tf, start, end)
        if len(part):
            frames.append(part)
    if frames:
        return store.merge(key, tf, pd.concat(frames))
    if last is None:
        log.error("No history returned for %s %s", key, tf)
        return None
    return pd.read_parquet(store.path(key, tf))


def clock_section(cfg: dict, frames: dict) -> tuple[bool, list[str]]:
    configured = (cfg["server_timezone"]["zone"], float(cfg["server_timezone"]["shift_hours"]))
    ref = next((k for k in US_INDEX_KEYS if k in frames), None)
    if ref is None:
        return True, ["No US index configured, so the server clock rule could not be "
                      "checked from history."]
    df = frames[ref]
    fit = fit_server_clock(df["server_time"], df["tick_volume"], configured)
    ok = fit.scores[configured] >= fit.scores[fit.best] - 0.02
    lines = [f"Checked against the US cash open in {ref} over {fit.weeks} weeks:", ""]
    lines += [f"- {fit.describe(rule)}: matches {score:.0%} of weeks"
              + (" (configured)" if rule == configured else "")
              for rule, score in sorted(fit.scores.items(), key=lambda x: -x[1])[:4]]
    lines.append("")
    if ok:
        lines.append(f"Configured rule {fit.describe(configured)} is consistent with the data.")
    else:
        lines.append(f"**MISMATCH:** the data fits {fit.describe(fit.best)} better than the "
                     f"configured {fit.describe(configured)}. Set server_timezone to "
                     f"zone: {fit.best[0]}, shift_hours: {fit.best[1]:g} in config.yaml and "
                     f"run the download again (stored timestamps are recomputed).")
    return ok, lines


def defects_section(cfg: dict) -> list[str]:
    """The frozen known-defect list from config (COMMON 1.3), so every report restates it."""
    dq = cfg.get("data_quality") or {}
    lines = [f"Segment break: consecutive bars more than {dq.get('segment_break_days', 7)} days "
             "apart start a new segment; lookbacks restart there (COMMON 1.4).", ""]
    for w in dq.get("known_holes") or ():
        lines.append(f"- Data hole {w['start']} to {w['end']} "
                     f"({', '.join(w.get('instruments') or ['all'])}): {w.get('reason', '')}")
    for w in dq.get("tick_volume_bad") or ():
        lines.append(f"- Tick volume corrupt {w['start']} to {w['end']} (all instruments, "
                     f"tv_bad flag): {w.get('reason', '')}")
    lines.append("- Bars printed on exchange full holidays (NYSE for US indices, XETRA for "
                 "GER40): counted in the Holiday bars column, skipped by lookbacks.")
    return lines


def spec_row(key: str, symbol: str, s, currency: str) -> str:
    return (f"| {key} | {symbol} | {s.digits} | {s.point} | {s.trade_tick_size} | "
            f"{s.trade_tick_value} {currency} | {s.volume_min} | {s.volume_step} | "
            f"{s.volume_max} | {s.swap_long} | {s.swap_short} | {s.swap_mode} | "
            f"{s.swap_rollover3days} | {s.trade_stops_level} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=float, default=None)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config()
    years = args.years or cfg.get("history_years", 5)
    client = MT5Client(cfg)
    account = client.connect(load_credentials())
    try:
        symbols = client.resolve_symbols()
        live_clock = client.check_server_clock(next(iter(symbols.values())))
        store = BarStore(cfg)
        now = utc_now()
        rows, spec_lines, entry_frames, short_history = [], [], {}, []
        for key, symbol in symbols.items():
            spec_lines.append(spec_row(key, symbol, client.spec(symbol), account["currency"]))
            for tf in dict.fromkeys(cfg["timeframes"].values()):
                # Quality checks only count bars and look for errors; they don't study
                # returns, so they cover the holdout period too.
                df = download(client, store, key, symbol, tf, years, now)
                if df is None:
                    continue
                rows.append((key, tf, quality_report(df, tf_minutes(tf), key=key, cfg=cfg)))
                if tf == cfg["timeframes"]["entry"]:
                    entry_frames[key] = df
                if df["time_utc"].min() > now - pd.Timedelta(days=365 * years - 7):
                    short_history.append(f"{key} {tf} (from {df['time_utc'].min():%Y-%m-%d})")
    finally:
        client.shutdown()

    clock_ok, clock_lines = clock_section(cfg, entry_frames)
    notes = []
    if short_history:
        notes.append(f"History shorter than {years:g} years for: {', '.join(short_history)}. "
                     "MT5 caps downloads at its 'Max. bars in chart' setting (Tools > Options > "
                     "Charts). Raise it to Unlimited in the data terminal and rerun to backfill.")
    report = "\n".join([
        f"# Data quality report ({now:%Y-%m-%d %H:%M} UTC)", "",
        f"Demo account (login not shown) at {account['company']} ({account['server']}).", "",
        "## Server clock", "", f"Live check: {live_clock}.", "", *clock_lines, "",
        "## Symbol specs (demo broker; the competition platform may differ)", "",
        "| Instrument | Broker symbol | Digits | Point | Tick size | Tick value | Min lot | "
        "Lot step | Max lot | Swap long | Swap short | Swap mode | Triple swap day | "
        "Stops level (points) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        *spec_lines, "",
        "Swap values are the demo's current ones; the backtest applies them to all history "
        "(COMMON 8.6). Triple swap day: 0 = Sunday ... 6 = Saturday.", "",
        "## Known data defects (config data_quality, frozen)", "",
        *defects_section(cfg), "",
        "## Bars", "",
        format_report(rows), "",
        "- Other gaps: missing bars inside the trading day (up to 12h), excluding the daily "
        "break. Most are exchange holidays and early closes.",
        "- Closures >12h that are not segment breaks are checked against the exchange "
        "calendar (NYSE or XETRA, exchange-local dates); each is shown as last bar before to "
        "first bar after (UTC bar open times). Weekend closures (count): lose no cash "
        "session and at most 12h of non-holiday weekdays. Holiday closures: the same, but "
        "touching a full holiday. **Unexplained gaps: everything else, i.e. missing data "
        "(cash-session time lost on a non-holiday weekday, or over 12h of such a weekday).**",
        "- Segment breaks / Segments: gaps longer than data_quality.segment_break_days and the "
        "resulting segments (COMMON 1.4).",
        "- Holiday bars: bars whose exchange-local date is a full exchange holiday (COMMON 1.6). "
        "tv_bad bars: bars inside the configured corrupt tick-volume window (COMMON 1.5).",
        "- Large moves: bar-to-bar moves over 1%. Spikes (possible bad ticks): large moves "
        "that the next bar mostly reverses, on below-normal tick volume.",
        "- Spread is the bar's spread in points as reported by the demo broker.",
        *[f"- {n}" for n in notes],
    ])
    out = ROOT / "docs" / "data_quality.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved to {out}")
    if not clock_ok:
        sys.exit("Server clock rule does not match the data. See the report.")


if __name__ == "__main__":
    main()
