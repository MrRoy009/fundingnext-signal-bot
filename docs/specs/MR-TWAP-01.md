# MR-TWAP-01: Calm-regime session-mean stretch fade (formerly MR-VWAP-01)

## Status
SPECIFIED - not tested. Written at GATE 2 before any backtest. Revised after QA review: the anchor is
now the equal-weight session mean (TWAP) of typical price, because demo tick volume is corrupt from
2024-10-04 to 2025-01-07 and differs from the competition feed. Everything not stated here follows
`COMMON.md`. The MARKET_OR_LIMIT protocol needs owner approval (O1); if refused, the COMMON fallback
protocol replaces it (not run in addition).

## Rationale
1. **Liquidity provision.** When an impatient order-flow imbalance pushes price away from where the
   session has been trading, the patient side earns a reversal premium as the imbalance is absorbed
   (Grossman and Miller 1988, JF; Jegadeesh 1990 and Lehmann 1990 on short-horizon reversal; Nagel
   2012, RFS, reversal as a return to liquidity provision).
2. **Session mean as fair-value anchor.** The equal-weight mean of the session's typical prices is a
   simple, feed-independent estimate of where today's two-sided trade has centred. The earlier claim that
   VWAP algorithms "pull" price to VWAP is withdrawn: those algorithms follow the volume curve.
3. **Why only the CALM regime (a hypothesis, not a measured mechanism).** Intraday index moves tend to
   continue when option dealers are short gamma (stressed, high-volatility regimes) and to be damped when
   they are long gamma (calm regimes): Baltussen, Da, Lammers and Martens (2021, JFE); Ni, Pearson,
   Poteshman and White (2021, RFS). An H1 HMM cannot observe dealer gamma; the CALM gate is a
   trend-day-avoidance bet that also keeps stops small in points. Counterpoint: Nagel (2012) finds reversal
   premia larger when volatility is high. QA reports results by regime as a descriptive split only.
4. **Relative-strength alignment.** Only the index that is strongest over the last trading day may be
   bought on a dip, and only the weakest sold on a rally ("short-term reversal within intermediate-horizon
   relative strength"). GER40 is ranked alone in its window, so for GER40 this means: buy dips only when
   its own one-day move is up, sell rallies only when it is down.
5. **Prior strength: weak to moderate.** Short-horizon reversal is well documented; its survival through
   CFD costs, manual delay and the lot cap is not.

## Instruments, timeframes, sessions
- US30, NAS100 (USTEC), GER40 (DE40). US500 excluded.
- Signal on M15 (bid OHLC); ranking (RS_day) and regime on H1.
- Signal window on T, both ends inclusive, valid trading dates only:
  - US30, NAS100: 10:30-14:30 New York = 14:30-18:30 UTC (US summer), 15:30-19:30 UTC (US winter).
  - GER40: 10:00-16:00 Berlin = 08:00-14:00 UTC (EU summer), 09:00-15:00 UTC (EU winter).
  - Inside config `sessions_utc` in both DST states.
- Start = cash open + 60 min (M15 ATR lags in the first hour and the opening auction inflates the stretch).
  End = time exit - 75 min, so every limit order expires at least 30 minutes before the time exit
  (QA: the old limit could outlive the exit).
- Manual time exit 15:45 New York / 17:15 Berlin.

## Index selection
COMMON 4.2 with rank score **RS_day** (24 H1 bars / ATR14_H1, H1 alignment rule). Window set {US30, NAS100}
in the US window and {GER40} in its window. E(T) = members passing the CALM gate. LONG-SELECTED = argmax
RS_day over E(T), SHORT-SELECTED = argmin. With two US indices passing there is no sign condition (the
stronger may be bought even if both fell; this was decided from setup counts before any outcome was seen).
With one passing index the sign rule applies (BUY only if RS_day > 0, SELL only if RS_day < 0).

## Regime condition
CALM gate (COMMON 6.7): index passes iff regime status OK and V_T < 0. If the model has no CALM-tagged
state, the strategy is inactive for that index and the feasibility diagnostics say so.

## Entry
At each cycle T (close of M15 bar t), per index i in E(T):
1. `TWAP_t` (COMMON 3) from the cash-open bar (09:30 New York / 09:00 Berlin) to bar t inclusive.
   `ATR_t` = ATR14_M15 at t. Stretch `Z_t = (C_t - TWAP_t) / ATR_t`.
2. Long trigger: i = LONG-SELECTED and `Z_t <= -K`. Short trigger: i = SHORT-SELECTED and `Z_t >= +K`.
   If both fire on different indices at the same T, both are candidates.
3. One candidate per index per session (local date), whichever side, consumed even if suppressed,
   skipped or unfilled. No new candidate on i while an earlier MR-TWAP-01 candidate occupies i
   (reference occupancy, COMMON 7.1).
4. E = C_t. Protocol **MARKET_OR_LIMIT** (COMMON 7): SKIP LEVEL = E - 0.50 D (long) / E + 0.50 D (short);
   skip if the quote is at or beyond TP; market if at E or better; otherwise LIMIT at E.
   EXPIRY = min(T + 60 min, time exit - 30 min). No order is placed at or after EXPIRY.
5. Candidate fields: order_type MARKET_OR_LIMIT, entry = E, skip level, expiry, atr = ATR_t, rank_score =
   RS_day, regime fields per COMMON 6.7.
   Reason line: `MR-TWAP-01 <LONG|SHORT> <index>: Z <-x.xx> from session mean <TWAP>, <strongest|weakest|
   sole index, RS_day +x.xx>, regime CALM. Limit at entry if price is worse; expires hh:mm UTC.
   MANUAL TIME EXIT hh:mm UTC (15:45 New York) if still open.`

## Stop
Fixed at signal time, absolute price, never re-anchored to the fill. Long SL = E - M * ATR_t; short
SL = E + M * ATR_t. Rounded to 0.01 away from entry. D = M * ATR_t.

## Exit
- **Bracket at entry:** TP = TWAP_t frozen at T (reward >= K * ATR_t, so planned R:R >= K / M = 1.33 at
  the defaults), rounded toward entry.
- **MANUAL TIME EXIT (flagged, owner action):** close at 15:45 New York / 17:15 Berlin if still open;
  EXIT REMINDER from the engine (ER-03). Backtest per COMMON 8.5. MISSED-EXIT must pass; it holds GER40
  overnight through 2-6 point night spreads, which COMMON's hourly p90 floor models.
- Unfilled limit orders are cancelled at EXPIRY.

## Sizing
Engine standard on `sizing_entry` = E (the protocol never pays more than E). Cap 3 lots; the cap binds on
94-100% of setups. Median in-session ATR14_M15 (US30 56, NAS100 35, GER40 25 points) gives D at M = 1.5 of
about 85 / 52 / 38 points and capped risk of about $253 / $157 / $128 (0.13-0.25% of balance). A market fill
better than E lowers real risk. Results in $ and R per index.

## Parameters
| Name | Default | Sensitivity grid | Reason |
|---|---|---|---|
| K (stretch threshold, ATR units) | 2.0 | 1.6, 1.8, 2.0, 2.2, 2.4 | Two units of local volatility is the conventional "stretched" threshold (the Bollinger 2-sigma analogue); \|Z\| >= 2 is about the 5-10% tail of in-session bars. It also gives a reward of at least 2 ATR, so stressed spread stays a small share of reward. |
| M (stop, ATR units) | 1.5 | 1.0, 1.25, 1.5, 1.75, 2.0 | About 2-3 median M15 bar ranges plus room for the 1-3 bar delay (median 3-bar move 0.78 ATR); keeps planned R:R >= 1.33 at K = 2. |

Grid: the 5 x 5 heatmap of K x M contains the one-at-a-time grid = **25 configurations**.
The old P_MIN parameter is removed: QA showed filtered probabilities are nearly saturated, so it was a
hidden binary; the gate is now the K-invariant V_T < 0 with no threshold to tune.

Frozen: TWAP of typical price with equal weights; window start (open + 60 min) and end (time exit - 75 min);
one candidate per index per session; EXPIRY rule; protocol constants (0.50 D skip); RS_day lookback 24 H1;
ATR period 14; time-exit clocks.

## Expected trade frequency
Descriptive setup counts only (COMMON 12; proxy `sign(x2)` instead of the HMM), 2023-03-01 to 2026-04-30
(about 158 weeks):
- K = 2.0: 395 setups (2.5 per week) on 311 distinct dates; 268 inside the 9 test windows (2.45 per week);
  GER40 141, US30 132, NAS100 122. The proxy marks 55% of window bars CALM.
- K = 1.6: 563; K = 2.4: 266.
Some limit orders will not fill (not estimated). Expect roughly 1.5-2 filled trades per week and about
160-220 filled trades in the test windows: close to the 150-trade minimum. If the real HMM's CALM share is
much lower than the proxy, the feasibility gate (COMMON 10.1, >= 200 candidates) drops the strategy before
any backtest.

## Manual execution fit
Good. A 2-ATR stretch reverts over several bars, not inside 15 minutes. MARKET_OR_LIMIT turns the delay
into a price improvement or a missed trade, never a chase: if price keeps stretching, the owner gets E or
better at market (until half the stop is used); if it has already reverted, the limit rests at E and costs
only the opportunity. SL and TP are absolute. Only one manual action after entry: the flagged time exit.

## Costs and lot cap
Stops of about 85 / 52 / 38 points and rewards of at least 113 / 70 / 51 points against p90 in-window
spreads of about 1-2 points: spread is 1-3% of D, and limit entries pay no entry slippage. Under STRESS the
round trip is roughly 4-8% of D. The 3-lot cap keeps realised risk at about $130-250 per trade; five
simultaneous full stops would be about $1,000 (1% of balance). Dollar results are small at the demo contract.

## Look-ahead safeguards
- TWAP, Z and ATR use bars up to and including t; the session anchor comes from tz-converted open times.
- RS_day and the regime come from H1 bars closed by T (H1 presence rule; filtered probabilities only).
- TP and SL are frozen at T; limit fills use bars from P onward only.
- One-per-session and occupancy use only earlier candidates and bars after T up to now.
- CI: perturbing any bar after T leaves the candidate at T unchanged, including TWAP, Z, RS_day and V_T.

## Failure modes
- Trend days that begin inside a CALM regime: the hourly HMM lags, and the first stretch of a breakout day
  is faded and stopped. One per index per session caps this at one loss per index per day.
- News stretches do not revert (CPI, FOMC, NFP, ECB): owner fills `restricted_windows_utc` (O5).
- Adverse selection of limit fills: limits fill on trades that keep going against the entry. QA reports fill
  rate and results split by market vs limit fills for each d (descriptive split, not a selection rule).
- Low frequency limits statistical power and the 5-trading-day requirement; it is a complement to the
  ACTIVE-regime strategies, which trade when it is silent.
- Gate non-functional or K selection without a CALM state.
- Missed time exit leaves GER40 exposed to wide night spreads.
- Lot-cap economics: positive R can still mean small dollars.

## QA issues resolved
| QA issue | Resolution |
|---|---|
| Frequency rests on an over-optimistic calm proxy (blocker) | Deseasonalised HMM (COMMON 6); feasibility gate with the real HMM, >= 200 candidates in the test windows, before any backtest |
| Regime model unspecified; clock not regime (COMMON blockers) | COMMON 6 |
| 8-week data gap (COMMON blocker) | COMMON segment rule; ATR, TWAP, RS_day restart after the hole |
| Tick-volume VWAP unreliable (major) | Equal-weight TWAP of typical price; rationale restated |
| Limit order outlives the time exit (major) | EXPIRY = min(T + 60, time exit - 30); last T = time exit - 75 |
| Lot-cap economics; US500 masks $ (major) | $ and R per index; US500 excluded (O6) |
| P_MIN saturated, seed dependence (COMMON major) | P_MIN removed; K-invariant gate; fixed seeds |
| Protocol overrides owner settings (COMMON major) | Owner approval O1; fallback replaces, not adds |
| Time exit outside owner hours (COMMON major) | Owner decision O3; MISSED-EXIT pass/fail; EXIT REMINDER |
| Rationale conflicted (minor) | Gamma link stated as a hypothesis; regime split reported descriptively |
| R:R assumes fill at E (minor) | Fill rates and market vs limit splits per d reported; R measured against planned risk |
| Delay edge cases (COMMON minor) | Skip if the quote is at or beyond TP; skip at half the stop |
