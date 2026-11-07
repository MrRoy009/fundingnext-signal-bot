# RS-MOM-01: Intraday relative-strength momentum ignition (M15, ACTIVE regime)

## Status
SPECIFIED - not tested. Written at GATE 2 before any backtest. Revised after QA review.
Everything not stated here follows `COMMON.md`.

## Rationale
- **Main premise: intraday time-series momentum in equity index futures.** The return earlier in a
  session predicts the return later in the same session: Gao, Han, Li and Zhou (2018, JFE, "Market
  intraday momentum"); Li, Sakkas and Urquhart (2022, J. Financial Markets) across 16 developed index
  futures; Baltussen, Da, Lammers and Martens (2021, JFE), who link it to dealer gamma hedging and to
  institutions that split large orders through the day (Chordia, Roll and Subrahmanyam 2005; Hong and
  Stein 1999 on gradual information diffusion).
- **Why only the ACTIVE regime.** These papers find the effect stronger on high-volatility days, and
  short-gamma hedging flows that push in the direction of the move are larger then. The gate trades
  only when the HMM's filtered expected volatility ratio is at or above its neutral point (COMMON 6.7).
- **Relative strength is the vehicle selector, not the edge.** Within the US group the score picks
  US30 or NAS100 (M15 return correlation about 0.79): mostly a tech versus industrials/value tilt on a
  shared move (Moskowitz and Grinblatt 1999 on industry momentum). GER40 is ranked alone in its window,
  so for GER40 the rule is plain time-series momentum. No cross-sectional claim is made for GER40.
- **Why the score is time-of-day normalised.** M15 ATR lags at the cash open (QA: at 10:00 New York it
  is about 80% of its noon level), which inflated the old score and tightened the old stop exactly when
  ranges are largest. RS_4h divides by the typical displacement for those exact times of day, and the
  stop uses H1 ATR, which does not jump at the open.
- **Prior strength:** moderate for the time-series effect in index futures; weak for its survival
  through CFD costs, a 15-45 minute manual delay and the 3-lot cap. The published effect concentrates in
  the last half hour, which this strategy does not trade (it exits at 15:45 New York); it relies on the
  part of the drift that accrues during the session.

## Instruments, timeframes, sessions
- US30, NAS100 (USTEC), GER40 (DE40). US500 excluded (COMMON 1.1).
- Signal and score on M15; stop scale (ATR14_H1) and regime on H1.
- Signal window on T (signal bar close), both ends inclusive, valid trading dates only:
  - US30, NAS100: 10:30-13:30 New York = 14:30-17:30 UTC (US summer), 15:30-18:30 UTC (US winter).
  - GER40: 10:00-15:00 Berlin = 08:00-13:00 UTC (EU summer), 09:00-14:00 UTC (EU winter).
  - Both lie inside config `sessions_utc` (US 13:30-20:00, GER40 07:00-15:30) in both DST states.
- First cash hour excluded (opening-auction noise; the 16-bar lookback would be mostly pre-open).
  The window ends 2 h 15 min before the time exit so the bracket usually resolves first.
- Manual time exit 15:45 New York (US) / 17:15 Berlin (GER40).

## Index selection
COMMON 4.2 with rank score **RS_4h** (L = 16). The window set is {US30, NAS100} in the US window and
{GER40} in the GER40 window (they never overlap). E(T) = window-set members that pass the ACTIVE gate.
LONG-SELECTED = argmax RS_4h over E(T), SHORT-SELECTED = argmin. With one passing index the sign rule
applies (automatically satisfied, since the trigger needs |RS_4h| >= theta > 0).

## Regime condition
ACTIVE gate (COMMON 6.7): index passes iff regime status OK and V_T >= 0. Both sides use the same gate.
B3 (gate off) is reported as a benchmark; the gate is not changed in response.

## Entry
At each cycle T (close of M15 bar t), for the strategy's E(T):
1. Long fires on index i iff: i = LONG-SELECTED; `RS_4h(i,t) >= theta`; and `RS_4h(i,t-1) < theta`,
   where t-1 is i's previous M15 bar (15 min earlier) and `RS_4h(i,t-1)` is computed with the same
   formula from bars closed by T - 15 min. Freshness is a property of the index's own score: rank and
   gate are filters at t only. If `RS_4h(i,t-1)` is NaN, no fire.
2. Short fires on index i iff: i = SHORT-SELECTED; `RS_4h(i,t) <= -theta`; `RS_4h(i,t-1) > -theta`.
3. Lockouts (deterministic, reference occupancy COMMON 7.1):
   - no new candidate on index i while an earlier RS-MOM-01 candidate on i occupies it;
   - at most 2 candidates per index per side per local date (consumed even if suppressed or skipped).
4. Entry reference E = C_t (bid close of bar t). Protocol: **MARKET_BAND** (COMMON 7).
   CHASE LIMIT = E + 0.20 D (long) / E - 0.20 D (short); SKIP LEVEL = E - 0.50 D (long) / E + 0.50 D (short).
5. Candidate fields: entry = E, side, stop, take_profit, atr = ATR14_M15 at t (information only),
   rank_score = RS_4h, regime and regime_prob per COMMON 6.7.
   Reason line: `RS-MOM-01 <LONG|SHORT> <index>: <leader|laggard|sole index> of {<window set>},
   RS_4h <+x.xx> crossed <theta>, regime ACTIVE. MANUAL TIME EXIT hh:mm UTC (15:45 New York) if still open.`

## Stop
Fixed at signal time, never moved, absolute price. ATRH = ATR14_H1 of H1 bar j (COMMON 2, H1 alignment;
if unavailable, no candidate). Long SL = E - k * ATRH; short SL = E + k * ATRH. Rounded to 0.01 away
from entry. D = k * ATRH.

## Exit
- **Bracket at entry:** TP = E + R * k * ATRH (long) / E - R * k * ATRH (short), rounded toward entry.
- **MANUAL TIME EXIT (flagged, owner action):** if neither SL nor TP is hit, close at market at 15:45
  New York (US) / 17:15 Berlin (GER40) on the signal date. Printed on the ticket in UTC and local time;
  the engine prints an EXIT REMINDER (ER-03). Backtest per COMMON 8.5.
- Expected resolution: with in-session H1 sigma of about 0.8 ATRH, a [-1, +1.5] ATRH band has an
  expected first-exit time of roughly 2-3 hours under a random walk, against 2.25-5.25 hours left
  before the time exit. Most trades should end at SL or TP; QA reports the exit-type shares.
- MISSED-EXIT scenario (COMMON 10.3) must pass: the bracket is held overnight with swap.

## Sizing
Engine standard on `sizing_entry` = CHASE LIMIT: risk 0.5% of balance, floor to 0.1 lot, cap 3 lots,
suppress below 0.1 lot. At the demo contract the cap binds on almost every trade: worst-case stop
1.2 D needs 167 points (US30) before 3 lots stop binding. Typical capped risk at k = 1.0, from
in-session ATR14_H1 medians (since 2024 / 2025-26): US30 92/113 points, $275-340; NAS100 64/82 points,
$190-247; GER40 49/63 points, $163-212. All results are reported in $ and R per index.

## Parameters
| Name | Default | Sensitivity grid | Reason |
|---|---|---|---|
| theta (momentum threshold on RS_4h) | 1.0 | 0.8, 0.9, 1.0, 1.1, 1.2 | RS_4h = 1 means the 4-hour move equals one typical displacement for those times of day: a natural "clearly more than noise" unit, same for all indices. |
| k (stop, multiples of ATR14_H1) | 1.0 | 0.75, 0.875, 1.0, 1.125, 1.25 | One hour of typical true range: the smallest stop that clears a full hour of noise for an hours-long thesis; about 2x the median 3-bar manual-delay move (0.78 M15 ATR = about 0.47 ATRH). |
| R (TP as multiple of stop) | 1.5 | 1.0, 1.25, 1.5, 1.75, 2.0 | 1.5 ATRH is about one sigma of the 2-5 hours left in the session, so the target captures a typical trend leg and the bracket usually resolves before the time exit. |

Grid: one-at-a-time (13 configurations) plus a 5 x 5 heatmap of k x R (16 more) = **29 configurations**.

Frozen (not tuned; changing any is a new trial): RS lookback L = 16 (shared; varied only in shared-
component sensitivity), ATR period 14, windows 10:30-13:30 New York and 10:00-15:00 Berlin, freshness
definition, 2 per index per side per day, protocol constants (0.20 D, 0.50 D, 45 min), time-exit clocks.
The old cost filter (25 x RefSpread) is removed: QA showed it was inert.

## Expected trade frequency
Descriptive setup counts only (COMMON 12; proxy `sign(x2)` instead of the HMM; 2-per-day cap applied;
reference-occupancy lockout not applied), 2023-03-01 to 2026-04-30, about 158 weeks:
- theta = 1.0: 624 setups (3.96 per week), 445 inside the 9 test windows (4.1 per week); GER40 340,
  NAS100 160, US30 124; 330 long, 294 short; 325 distinct dates. The proxy marks 45% of window bars ACTIVE.
- theta = 0.8: 754; theta = 1.2: 495.
After the lockout and risk suppression, expect about 2.5-3.5 tickets per week and roughly 300-400
candidates in the test windows. The real HMM gate may differ from the proxy; the feasibility gate
(COMMON 10.1) decides.

## Manual execution fit
The thesis is a multi-hour drift; a 15-45 minute delay is small against it. The stop (1 ATRH, about
1.6-1.7 M15 ATR) is about twice the median 3-bar delay move. MARKET_BAND does not skip continuations
the way the old symmetric 0.25 ATR rule did (that rule skipped about 37-43% of tickets, mostly the ones
that kept going): it accepts up to 0.20 D of chase (about 18 points on US30) and lets the owner take a
better price until half the stop is used. SL and TP are absolute levels. The last signal at 13:30 New
York leaves at least 1.5 hours after a 45-minute delay. Two manual time-exit clocks at most per day.

## Costs and lot cap
Stops of about 92-113 (US30), 64-82 (NAS100) and 49-63 (GER40) points against p90 in-window bar
spreads of about 1-2 points: one spread is 1-2% of D, and STRESS (2x spread, 2x slippage, stress
commission) stays below about 6-8% of R round trip. The 3-lot cap keeps realised risk at about
0.16-0.34% of balance, so the daily and total loss limits are far away at the demo contract; dollar
results are small. If the competition contract is larger (owner decision O4), sizing and the rule replay
change and are reported separately.

## Look-ahead safeguards
- Only bars with close <= T; the forming bar is never read; bars selected by time live (COMMON 9).
- RS_4h(t-1) uses only bars closed by T - 15 min; TODV uses strictly earlier dates.
- ATRH and the regime come from H1 bars closed by T, with the H1 presence rule (COMMON 2, 6.6).
- Regime probabilities are filtered (hand-written forward filter) from a model fitted before its refit time.
- Lockouts use reference occupancy from bars after T up to now only.
- CI tests: perturbing any M15 or H1 bar after T leaves every candidate at or before T unchanged; the
  H1 bar used at T closed at or before T across New York and Berlin DST changes, including the
  March and November mismatch weeks.

## Failure modes
- Reversal days (V-shaped sessions, FOMC at 14:00 New York after the last signal but inside the hold):
  stopped out or time-exit losses. Owner fills `restricted_windows_utc` (O5).
- In broad rallies the US leader is often the higher-beta NAS100, so the strategy becomes a beta bet.
  B1 placebo and B2 random-index benchmarks test this.
- GER40 provides about 55% of setups and is ranked alone: the strategy is mostly GER40 time-series
  momentum. Results must be shown per index.
- HMM gate non-functional (blocks < 5% or > 95% of window cycles) or starved by the warm-up.
- The owner misses the time exit: MISSED-EXIT must pass.
- Demo feed, spreads and contracts differ from the competition platform.
- Volatility rose 25-35% in 2025-26; normalisation handles scale, behaviour at extremes may differ.
- Overlap with SES-IBX-01 on the same index in the US morning; the combined book uses one position per index.

## QA issues resolved
| QA issue | Resolution |
|---|---|
| Ranking across sessions (major) | Rank only within the window set; GER40 ranked alone and stated as such |
| ATR lag at window start (major) | Score normalised by same-time-of-day TODV; stop on ATR14_H1; window starts 10:30 New York / 10:00 Berlin |
| Edge depends on the daily manual time exit (major) | Window ends 13:30 New York / 15:00 Berlin; R chosen so the bracket usually resolves; MISSED-EXIT is pass/fail; EXIT REMINDER requested |
| Trigger fires on rank flips (major) | Fresh = own score crosses theta while leader; previous-bar edge cases defined |
| Directional regime gate is noise (COMMON major) | Replaced by the ACTIVE volatility gate |
| Skip rule adverse selection (COMMON major) | MARKET_BAND protocol; delayed fill is the base case |
| Cost filter inert (minor) | Removed |
| US500 negligible in $ (minor) | US500 excluded |
| Lookback gap tolerance (minor) | All 17 bars must exist with consecutive open times |
| Grids inconsistent (minor) | One grid per parameter, quoted counts match |
| Hidden constants | Listed as frozen above and in COMMON 11 |
| Beta / benchmarks | COMMON 10.6 B1-B3 |
