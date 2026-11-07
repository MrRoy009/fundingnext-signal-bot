# Index signal bot (MT5, read-only)

Reads price data from a **MetaTrader 5 demo account**, ranks your preferred index CFDs and
prints BUY/SELL tickets for you to place **by hand** on your competition account. The bot
never places, changes or closes an order: the MT5 module is wrapped in a read-only proxy,
and a CI test fails the build if any order function appears in the code.

The full build specification is in [PROMPT.md](PROMPT.md).

## Status

| Gate | What | State |
|---|---|---|
| 1 | Config, symbol mapping, data download, data-quality report | Built; needs your demo login to run |
| 2 | Quant strategy specs (`docs/specs/`) | Not started |
| 3 | QA validation reports | Not started |
| 4 | Holdout test, go/no-go | Not started |

Until a strategy passes GATE 4 and is added to `strategies/registry.py`, the bot prints
`NO SIGNAL: no QA-approved strategy yet`. That is intended.

## Setup (Windows only)

The `MetaTrader5` Python package only works on Windows, next to an installed MT5 terminal.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
copy state.example.yaml state.yaml
```

### Connect the demo account

The bot needs **its own MT5 terminal**, separate from the one you trade the competition
from. Without one, the MetaTrader5 package attaches to whichever terminal was used last,
which could be your competition terminal.

1. Make a second terminal. Either copy only the program files (`terminal64.exe`,
   `MetaEditor64.exe`, `metatester64.exe`, `Terminal.ico`, `Sounds`, not `Config` or
   `Bases`) from your MT5 folder into e.g. `D:\MT5-Data` and start it with
   `terminal64.exe /portable`, then set `MT5_PORTABLE=true` in `.env`; or run the MT5
   installer and pick a different folder under **Settings**.
2. Open that terminal, open a free demo account (File > Open an Account) and log in.
   Don't log in to your competition account in this terminal.
3. In `.env`, set `MT5_PATH=C:\MT5-Data\terminal64.exe`. Optionally fill `MT5_LOGIN`,
   `MT5_PASSWORD` and `MT5_SERVER` for the demo (`.env` is git-ignored).

`COMPETITION_LOGIN` in `.env` holds your competition account number. The bot refuses to run
if it finds itself connected to that account, or to any account that isn't a demo.

### Map your symbols

Brokers name indices differently (`US30`, `US30.cash`, `DJ30`...). Each instrument in
`config.yaml` has a list of `aliases`; the first one the broker has is used. If a symbol
is missing, the bot stops and names it. Add your broker's name to the list.

### Check the server clock

MT5 stamps bars in the broker's server time. `server_timezone` in `config.yaml` tells the
bot how to convert to UTC. Most brokers are "New York + 7h" (the default). At startup the
bot compares this with the latest tick and stops if they disagree while the market is open.

## Running

```powershell
.\.venv\Scripts\python.exe -m data.download          # GATE 1: download history + quality report
.\.venv\Scripts\python.exe -m signals.run --format-check   # test ticket from live data (never a signal)
.\.venv\Scripts\python.exe -m signals.run --once     # one signal cycle
.\.venv\Scripts\python.exe -m signals.run            # run continuously
.\.venv\Scripts\python.exe -m pytest -q              # tests
```

`data.download` writes Parquet files to `data/store/` and a report to
`docs/data_quality.md`.

## Before tickets can go out

1. Review every value marked `OWNER` in `config.yaml`, then set `owner_confirmed: true`.
2. Keep `state.yaml` up to date from your competition account (balance, equity,
   start-of-day values, trades today, open positions with their risk to stop, and
   `pending_orders`: resting limit / stop orders with lots, worst-case risk and
   `expires_utc`). If it is older than `state_max_age_hours`, every ticket is suppressed.
   Tickets the bot issued after `updated_at` are counted on top of the file (open positions,
   trades today, open risk) until you update it. An open position or pending order on an
   index blocks every new ticket on that index (`blocked_open_position`).

## Reading a ticket

```
2026-11-02 15:00 UTC | US30.cash | BUY | 42000.0 | SL 41700.0 | TP 42600.0 | 1.38 lots | risk $496.80 | R:R 2.00
REGIME ACTIVE/ACTIVE_FLAT (p=0.82) | RANK +1.35 | STRATEGY RS-MOM-01 | VALID UNTIL 15:45 UTC (place the order by then)
ORDER TYPE: MARKET_BAND
ENTRY 42000.0 | CHASE LIMIT 42060.0 (60.0 = 600 points worse than entry) | SKIP LEVEL 41850.0 (150.0 = 1,500 points better than entry)
EXPIRY: none (market order only)
PLACE: market BUY now if ask is between 41850.0 and 42060.0 and not at or beyond TP 42600.0. Skip if ask > 42060.0 (CHASE LIMIT), ask < 41850.0 (SKIP LEVEL) or ask >= 42600.0.
STATS (out-of-sample): win 46.0% | expectancy +0.31R | max DD 7.9% | n=212
STOP DISTANCE 360.0 (3,600 points) from the worst permitted entry 42060.0: RISK $ is the worst case - recheck lot size on the competition platform
REASON: ...
MANUAL TIME EXIT: close at 20:45 UTC (15:45 New York) if still open
```

Every strategy uses one of three entry protocols (COMMON 7 in `docs/specs/`). All levels are
absolute prices: compare your live quote (ask to buy, bid to sell) with them; never move SL or
TP to your fill.

- **MARKET_BAND**: market order now if the quote is between SKIP LEVEL and CHASE LIMIT and
  not at or beyond TP; otherwise skip.
- **MARKET_OR_LIMIT**: skip beyond SKIP LEVEL or at / beyond TP; market order if the quote is
  at ENTRY or better; otherwise a LIMIT order at ENTRY with the printed EXPIRY. Never chase.
- **STOP_ENTRY**: a STOP order at TRIGGER with the printed EXPIRY while the quote is further
  than the broker's stops buffer from it; market order if the quote is past the trigger but
  not past CHASE LIMIT; otherwise skip.
- **VALID UNTIL** is the last time to place the order: T + `valid_for_bars` x 15 min, or the
  EXPIRY if earlier. A resting limit / stop order lives until its EXPIRY; add it to
  `pending_orders` in `state.yaml`.
- **Lots and RISK $** are sized on the worst fill the protocol permits (CHASE LIMIT, or ENTRY
  for MARKET_OR_LIMIT), so a permitted fill never risks more than the printed amount. The
  levels always come from `protocols` in `config.yaml`; a strategy that sets a different one
  gets a suppressed ticket ("invalid protocol levels").
- **MANUAL TIME EXIT**: close the position at that time if it is still open (15:45 New York
  for US30 / NAS100, 17:15 Berlin for GER40; the UTC time follows each market's DST). At that
  time the bot also prints and notifies an `EXIT REMINDER` for every candidate whose reference
  SL / TP was not touched by then.

A ticket that breaks any rule starts with `SUPPRESSED - DO NOT TRADE` and the reasons.
Every ticket, suppressed or not, is logged to `signals.csv` (if an older `signals.csv` with
different columns exists, it is left untouched and new rows go to `signals_v2.csv`, then
`_v3`, ...). EXIT REMINDERs go to `logs/exit_reminders.csv`; daily summaries go to `logs/`:
tickets, every protocol candidate (suppressed ones marked) with its manual time exit and how
its reference occupancy ended (SL, TP, EXPIRY, or still open at the time exit), reminders,
pending orders and regime history. US500 is downloaded for data only and never gets a ticket.

If one index's bar at T has not arrived by T + 60 s, that index's exchange group (US30 +
NAS100, or GER40) prints NO SIGNAL for the cycle and the other group carries on. NO SIGNAL for
everything is printed only when the connection fails or every index is stale.

## Risk checks

Before each ticket the bot assumes every open position, every pending order that has not
expired, every ticket issued since `state.yaml` was updated, and the new trade hit their stops
(plus a cost buffer) and checks: the competition daily loss limit, the 10% max loss, your
personal daily stop, max open positions, max lots, max trades per day, the end-of-competition
cutoff, restricted news windows, the per-index block (open position, pending order, or
another candidate still occupying the index), and allowed trading hours (`sessions_utc`,
checked on the signal time T with start <= T < end).

## Limitations

- Demo prices, spreads and contract sizes can differ from the competition platform. Always
  recheck lot size there using the stop distance in points.
- VIX, breadth and sector data are not available from MT5; the bot works without them.
- MT5's "Max bars in chart" setting can limit how much history is returned.
- Backtested and validated statistics are not a promise of future results.
