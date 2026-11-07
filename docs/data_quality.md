# Data quality report (2026-11-02 10:36 UTC)

Demo account (login redacted) at MetaQuotes Ltd. (MetaQuotes-Demo).

## Server clock

Live check: Server clock OK: UTC+2.0h.

Checked against the US cash open in US30 over 225 weeks:

- Europe/Athens +0h: matches 93% of weeks (configured)
- America/New_York +7h: matches 86% of weeks
- UTC +3h: matches 58% of weeks
- UTC +2h: matches 36% of weeks

Configured rule Europe/Athens +0h is consistent with the data.

## Symbol specs (demo broker; the competition platform may differ)

| Instrument | Broker symbol | Digits | Point | Tick size | Tick value | Min lot | Lot step | Max lot |
|---|---|---|---|---|---|---|---|---|
| US30 | US30 | 2 | 0.01 | 0.01 | 0.01 USD | 0.1 | 0.1 | 250.0 |
| NAS100 | USTEC | 2 | 0.01 | 0.01 | 0.01 USD | 0.1 | 0.1 | 250.0 |
| US500 | US500 | 2 | 0.01 | 0.01 | 0.01 USD | 0.1 | 0.1 | 250.0 |
| GER40 | DE40 | 2 | 0.01 | 0.01 | 0.011207799999999999 USD | 0.1 | 0.1 | 250.0 |

## Bars

| Instrument | TF | Bars | First (UTC) | Last (UTC) | Daily break (server) | Dupes | Other gaps | Max gap | Breaks >7d | Bad OHLC | Zero spread | Large moves | Spikes | Median spread (pts) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| US30 | M15 | 100604 | 2022-05-24 22:00 | 2026-11-02 10:15 | 23:45 | 0 | 140 | 0 days 10:30:00 | 2025-07-16 to 2025-09-09 | 0 | 686 | 44 | 0 | 101.0 |
| US30 | H1 | 28444 | 2021-11-03 11:00 | 2026-11-02 09:00 | 23:00 | 0 | 99 | 0 days 10:00:00 | 2025-07-16 to 2025-09-09 | 0 | 537 | 120 | 0 | 100.0 |
| NAS100 | M15 | 97742 | 2022-07-06 09:30 | 2026-11-02 10:15 | 23:45 | 0 | 156 | 0 days 09:00:00 | 2025-07-16 to 2025-09-09 | 0 | 514 | 126 | 1 | 100.0 |
| NAS100 | H1 | 24485 | 2022-07-06 09:00 | 2026-11-02 09:00 | 23:00 | 0 | 92 | 0 days 09:00:00 | 2025-07-16 to 2025-09-09 | 0 | 421 | 260 | 0 | 100.0 |
| US500 | M15 | 100798 | 2022-05-22 22:00 | 2026-11-02 10:15 | 23:45 | 0 | 191 | 0 days 09:00:00 | 2025-07-16 to 2025-09-09 | 0 | 50 | 65 | 0 | 50.0 |
| US500 | H1 | 28487 | 2021-11-03 11:00 | 2026-11-02 09:00 | 23:00 | 0 | 97 | 0 days 09:00:00 | 2025-07-16 to 2025-09-09 | 0 | 43 | 172 | 0 | 50.0 |
| GER40 | M15 | 100254 | 2022-03-02 23:00 | 2026-11-02 10:15 | 22:45 | 0 | 132 | 0 days 09:00:00 | 2025-07-22 to 2025-09-09 | 0 | 84 | 54 | 0 | 80.0 |
| GER40 | H1 | 27030 | 2021-11-03 11:00 | 2026-11-02 09:00 | 22:00 | 0 | 140 | 0 days 12:00:00 | 2025-07-22 to 2025-09-09 | 0 | 71 | 144 | 0 | 70.0 |

- Other gaps: missing bars inside the trading week, excluding weekends (>12h) and the daily break. Most are exchange holidays and early closes.
- Large moves: bar-to-bar moves over 1%. Spikes (possible bad ticks): large moves that the next bar mostly reverses, on below-normal tick volume.
- Spread is the bar's spread in points as reported by the demo broker.
- History shorter than 5 years for: US30 M15 (from 2022-05-24), NAS100 M15 (from 2022-07-06), NAS100 H1 (from 2022-07-06), US500 M15 (from 2022-05-22), GER40 M15 (from 2022-03-02). MT5 caps downloads at its 'Max. bars in chart' setting (Tools > Options > Charts). Raise it to Unlimited in the data terminal and rerun to backfill.