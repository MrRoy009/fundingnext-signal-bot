# Build prompt: MT5 index signal bot (read-only)

Paste everything below the line into Claude Code (or another coding agent) opened in an
empty folder. Fill in the OWNER SETTINGS block first; the agent must stop and ask if any
value is still blank.

---

# OWNER SETTINGS (fill before starting; agent must ask if any is blank)
- Preferred indices (broker names, comma-separated): [e.g. US30, NAS100, US500, GER40]
- Entry timeframe: [e.g. M15]        Regime timeframe: [e.g. H1]
- History to download: [e.g. 5 years, or the maximum the demo broker provides]
- Final holdout period (start → end): [e.g. last 6 months]
- Risk per trade: [e.g. 0.5]% of current balance
- Personal daily stop: [e.g. 3]% of start-of-day balance
- Manual execution delay to simulate: [e.g. 1-3] bars
- Skip-if tolerance: [e.g. 0.25 x ATR, or N points] beyond entry
- Trading hours / sessions allowed (UTC): [e.g. 13:30-20:00 for US indices]
- Competition end date/time (UTC): [....]
- Competition rule set (verify against the official rules page; these are defaults):
  - Initial balance $100,000; daily loss limit 5%; max total loss 10% of initial balance
  - Daily loss measured from: [start-of-day balance / equity, per the rules]
  - Max 5 open positions; max 3 lots per index trade; max 5 lots per forex trade;
    max 50 trades/day
  - Trades required on at least 5 separate days
  - All positions closed 15 min before competition end
  - News-trading or weekend-holding restrictions: [none / describe]

# ROLE
You work as three roles in a fixed order. Each role writes its output to the repo before
the next role starts. Name the active role at the top of each message.
- Quant: designs features, the regime model and strategies, and picks every parameter.
  Writes specs to /docs/specs/<strategy_id>.md before any code for that strategy exists.
- Engineer: implements the specs exactly. May not change logic, add indicators or tune
  parameters. If a spec is ambiguous or unworkable, stop and send it back to Quant.
  Owns the data pipeline, backtester, signal engine, logging and tests.
- QA/Risk: verifies independently. Assumes every strategy is overfit until the evidence
  says otherwise. Can veto any strategy or code change, and must write down why.
The human owner approves requirements, merges and anything that goes live. Stop for owner
approval at every GATE below. Use git with small commits; CI runs on every push.

# OBJECTIVE
Build a Python bot that reads price data from a MetaTrader 5 DEMO account, ranks the
preferred indices, and prints BUY/SELL trade tickets. The owner places trades by hand on a
separate competition account. The bot never places, changes or closes an order.

# HARD CONSTRAINTS (non-negotiable)
1. READ-ONLY. The codebase must never call mt5.order_send, mt5.order_check, or any other
   function that places, changes or closes orders or positions. Add a CI test that scans
   every .py file (AST scan, plus a plain text search to catch getattr or string tricks)
   and fails the build if any of these names appear. Read-only calls such as positions_get
   and order_calc_profit are allowed.
2. Connect only to the separate MT5 demo account. Credentials come from a local .env file
   (in .gitignore, never hard-coded, never logged). Ship .env.example with blank values.
   At startup, check account_info().trade_mode is DEMO; if not, exit with an error.
3. No part of the bot may touch the competition platform, its website or its account.
4. If the data is stale (last closed bar older than 2x the timeframe, outside market
   closures) or the connection fails, print NO SIGNAL with the error. Never guess.
5. Never claim or imply guaranteed profit. Every ticket carries its validated statistics.
6. Signals come from CLOSED bars only. The forming bar is never used for a decision.

# DATA LAYER
- Use the official MetaTrader5 Python package. It runs on Windows only; say so in the
  README. All MT5 access goes through one adapter module, so tests can use a fake adapter
  and CI can run on any OS.
- Calls used: initialize(), login(), account_info(), symbol_info(), symbol_select(),
  symbol_info_tick(), copy_rates_from_pos(), copy_rates_range().
- Broker symbol names vary (US30 / US30.cash / DJ30, NAS100 / USTEC, US500 / SPX500).
  Map them in config.yaml and check each one with symbol_info() at startup. Fail clearly
  on a missing symbol.
- MT5 bar times are broker server time stored as if they were UTC. Detect the server
  offset (including daylight-saving changes), convert every timestamp to true UTC, and
  log the offset used.
- Store bars as Parquet, one file per symbol and timeframe, with incremental updates.
- Available from MT5: OHLC, tick volume and spread. VIX, market breadth and sector
  strength are not. They are optional external inputs, off by default; the bot must work
  without them.
- Data quality checks: gaps, duplicate bars, zero or negative spreads, outliers. Log them;
  never fill gaps silently.
- Demo prices and spreads can differ from the competition feed. Record this in the README
  and in every validation report.

# INDEX SELECTION
On each closed entry bar, rank the configured indices by a relative-strength score that
Quant specifies (e.g. N-period return divided by ATR). Only the strongest index that
passes the regime filter can get a long signal; only the weakest passing index can get a
short signal.

# REGIME MODEL (HMM)
- Gaussian HMM (hmmlearn) on returns, ATR-normalised volatility and tick volume, plus
  external inputs if enabled. Quant chooses the number of states by BIC on training data
  only.
- NO LOOK-AHEAD: signals use filtered (forward-algorithm) probabilities computed only from
  data up to the current closed bar. Never use smoothed probabilities or full-sample
  Viterbi paths. Write the forward filter yourself if the library doesn't expose it.
- Refit on a rolling past window at a fixed schedule. After each refit, relabel states by
  mean return and volatility so labels stay consistent. Log each refit.
- Output: current regime label and its probability.

# STRATEGY RESEARCH
Quant tests candidates such as momentum, breakout, relative strength, mean reversion and
regime-conditional versions. Each spec states: entry rules, exit rules, stop rules,
sizing rules, allowed sessions, and the economic reason it should work. Keep the number of
free parameters small. Only QA-approved strategies go live.

# MANDATORY VALIDATION (QA runs all of these; any failure means rejection)
1. Walk-forward tests with out-of-sample windows. The final holdout period from OWNER
   SETTINGS is locked: the data loader refuses holdout dates unless run with
   --final-holdout, which logs the run with a timestamp and data hash. It is used once,
   after every other step passes.
2. Parameter sensitivity: nearby values (e.g. 18-22 around 20) must stay profitable.
   Reject knife-edge parameters. Include a heatmap.
3. Cost stress: 2x historical spread and commission, plus the manual execution delay from
   OWNER SETTINGS, plus slippage. The edge must survive.
4. Block bootstrap (not i.i.d.) and regime-conditional bootstrap of trades: report
   distributions and confidence intervals for profit and max drawdown.
5. Shuffle test for sequence risk: worst losing streak and worst drawdown.
6. Competition-rule simulation: replay the strategy against the rule set and report how
   often the daily and total loss limits would have been breached.
7. Trial log: record every variant ever tested, failures included. Report the deflated
   Sharpe ratio or the probability of backtest overfitting (CSCV) based on the total
   number of trials.
8. CI tests for look-ahead bias (changing future bars must not change past signals),
   timestamp alignment, server-time conversion, and cost assumptions.

# RISK CHECKS (run before every ticket; rule set lives in config.yaml)
- The owner keeps a local state file (state.yaml) with current balance, start-of-day
  balance, today's realised P&L, open positions and trading days completed. If the file is
  older than the configured limit (e.g. 12 hours), warn and suppress tickets.
- Check every ticket against the competition rule set and the personal daily stop,
  assuming the stop-loss is hit. Include the risk of open positions.
- Block new tickets in the final [N] minutes before competition end and inside any
  restricted news or weekend windows.
- If a ticket would break any rule, suppress it and state the rule.
- Report the trading-days requirement (e.g. "3 of 5 days done").

# POSITION SIZING
- Risk amount = risk % x current balance from state.yaml.
- Lots = risk amount / (stop distance in ticks x trade_tick_value), using symbol_info
  (trade_tick_size, trade_tick_value, volume_min, volume_max, volume_step).
- Round down to volume_step. Cap at the rule-set lot limit and volume_max. If the result
  is below volume_min, suppress the ticket rather than round up.
- Demo contract specs may differ from the competition platform. Print the stop distance
  in points and the risk in dollars so the owner can recheck the size there.

# SIGNAL TICKET (exact format)
TIME (UTC) | SYMBOL | BUY/SELL | ENTRY | STOP-LOSS | TAKE-PROFIT | LOTS | RISK $ | R:R
REGIME + probability | RANK SCORE | STRATEGY ID | VALID UNTIL (time and price tolerance)
STATS: out-of-sample win rate, expectancy (R), max drawdown, number of trades
REASON: one line
Skip if: price has moved more than [tolerance] past entry before you can place it.

# OUTPUTS
- Console output and a Windows desktop notification for each ticket.
- signals.csv: every ticket, including suppressed ones with the reason.
- Daily summary: tickets issued and suppressed, regime history, rule budget left.
- Rotating log file for errors and data-quality warnings.
- A scheduler that wakes just after each entry-bar close. Run it with one command:
  python -m signals.run

# DELIVERABLES
1. Repo: /data, /features, /regime, /strategies, /backtest, /risk, /signals, /tests,
   /docs, config.yaml, .env.example, state.example.yaml, requirements.txt (pinned), README.
2. A validation report for each strategy (Markdown plus charts), signed off by QA.
3. A working signal engine with the read-only CI guard passing.
4. README covering setup on Windows, demo-account connection, symbol mapping, the state
   file, how to read a ticket, and known limitations.

# GATES (stop and wait for owner approval at each)
- GATE 1: config, symbol mapping and data download working; data-quality report shown.
- GATE 2: Quant's strategy specs written, before any backtest runs.
- GATE 3: QA validation reports, before the holdout is opened.
- GATE 4: holdout result and final go/no-go, before the live signal loop is enabled.

# ACCEPTANCE CRITERIA
- All CI tests pass, including the no-order-functions guard and the look-ahead tests.
- At least one strategy passes every validation step, or the report says plainly that
  none did. Never lower the standards to get a strategy through.
- A dry run against live demo data produces correctly formatted tickets, and a test with
  a stale or disconnected feed prints NO SIGNAL.
