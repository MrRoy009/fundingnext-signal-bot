# SES-IBX-01: Initial-balance extension with resting stop orders (ACTIVE regime)

## Status
SPECIFIED - not tested. Written at GATE 2 before any backtest. Revised after QA review.
Everything not stated here follows `COMMON.md`.
**Conditional on owner approval** of the STOP_ENTRY protocol and the pending-order engine changes
(O1, ER-01, ER-04) and on confirmation that the competition platform accepts pending stop orders with
an expiry time (O4). If either is refused, this strategy is dropped (it cannot be expressed as a market
ticket without changing its logic).

## Rationale
- **Market structure.** The cash open concentrates the release of overnight information and
  institutional order flow: M15 range and tick volume at the open are about twice the midday level
  (U-shaped intraday activity; Wood, McInish and Ord 1985; Admati and Pfleiderer 1988). The first hour
  of cash trading, the Market Profile "initial balance" (Dalton et al., Mind Over Markets), is where the
  opening auction looks for two-sided trade. A move beyond the IB shows that the opening imbalance has
  not been absorbed; parent orders executed on VWAP/TWAP schedules and dealer short-gamma hedging then
  tend to push the same way (Baltussen, Da, Lammers and Martens 2021, JFE; Gao, Han, Li and Zhou 2018, JFE).
- **Related evidence:** Crabel (1990) on opening-range breakouts; Holmberg, Lonnbark and Lundstrom
  (2013, FRL); Zarattini and Aziz (2023) on ORB in QQQ.
- **Why the ACTIVE regime.** In calm, balanced regimes the market rotates inside the IB rather than
  extending it, so breakouts fail more often. The ACTIVE gate (COMMON 6.7) keeps only sessions where
  short-run volatility is at or above its recent level.
- **Direction from relative strength.** Only the strongest passing index may break up and only the
  weakest may break down, ranked by RS_4h at the IB end (the pre-market plus IB move, in line with the
  overnight-plus-first-half-hour evidence of Gao et al.).
- **Prior strength: weak.** The published evidence uses 5-minute opening ranges held to the close; the
  documented predictability concentrates in the last half hour, which this strategy does not hold
  (time exit 15:45 New York). ORB evidence is mixed and has likely decayed since publication.

## Instruments, timeframes, sessions
- US30, NAS100 (USTEC), GER40 (DE40). US500 excluded.
- IB and orders on M15; regime on H1.
- IB: the first `ib_minutes` of the cash session (09:30 New York; 09:00 Berlin). Ticket time T = close
  of the last IB bar: default 10:30 New York (14:30 UTC US summer, 15:30 UTC US winter) and 10:00 Berlin
  (08:00 / 09:00 UTC). Across the grid (30-90 min), T is 10:00-11:00 New York and 09:30-10:30 Berlin,
  inside config `sessions_utc` in both DST states.
- Order EXPIRY (fixed): 12:00 New York (US), 11:30 Berlin (GER40), where the tick-volume profile enters
  the midday lull. At ib_minutes = 90 this leaves 60 minutes after T.
- Manual time exit: 15:45 New York / 17:15 Berlin.
- Valid trading dates only (no holidays or early closes, COMMON 1.6).

## Index selection
COMMON 4.2 with rank score **RS_4h** at T. Window set at the US T = {US30, NAS100}; at the GER40 T =
{GER40} (alone; for GER40 the direction is the sign of its own RS_4h). E(T) = eligible members that pass
the ACTIVE gate. LONG-SELECTED may receive only a BUY STOP and only if its RS_4h > 0; SHORT-SELECTED may
receive only a SELL STOP and only if its RS_4h < 0. With two eligible US indices of different sign, both
tickets are issued (a pair day); QA reports pair days separately.

## Regime condition
ACTIVE gate (COMMON 6.7): V_T >= 0 at the latest core H1 observation closed by T (at 10:30 New York this
is the 09:00-10:00 bar). B3 (gate off) is a reported benchmark only.

## Entry
At T, for each index in the window set:
1. **Eligibility:** valid trading date; IB complete (all ib_minutes/15 bars present); DATR finite;
   `W <= max_ib_datr * DATR`; regime gate passes; no config restricted window overlaps [T, EXPIRY];
   no earlier SES-IBX-01 candidate on this index today (one per index per day, consumed even if
   suppressed, skipped or unfilled).
2. **Selection** per the section above.
3. **Order levels:** b = 0.05 W. Long trigger = IBH + b; short trigger = IBL - b.
4. **Protocol: STOP_ENTRY** (COMMON 7). At placement the owner compares the live quote with the trigger:
   long: ask < trigger - stops_buffer, place BUY STOP at trigger expiring at EXPIRY; trigger -
   stops_buffer <= ask <= CHASE LIMIT (= trigger + 0.20 D), market BUY; ask above the CHASE LIMIT, skip;
   placement time at or after EXPIRY, skip. Short mirrored with bid.
5. **Candidate fields:** order_type STOP_ENTRY, entry = trigger, chase limit, expiry, atr = ATR14_M15 at T
   (information only), rank_score = RS_4h, regime fields per COMMON 6.7.
   Reason line: `SES-IBX-01 BUY STOP <index> at <trigger> (IB <IBL>-<IBH>, W <W> = <W/DATR> DATR),
   RS_4h <+x.xx> <top|sole> of {<set>}, regime ACTIVE. Order expires hh:mm UTC (12:00 New York).
   MANUAL TIME EXIT hh:mm UTC (15:45 New York) if filled and still open.`
6. **Backtest:** placement at P = T + d bars (COMMON 8.3); pending fill model, stop-entry fills and
   fill-bar rules per COMMON 8.4-8.5 (on the fill bar, SL first; TP on the fill bar ignored).

## Stop
SL = IB midpoint (IBH + IBL) / 2, absolute price fixed at T, unchanged by worse fills. D = |trigger - SL|
= 0.55 W. Market Profile convention: a return to the middle of the IB means the extension failed and the
market is back in balance.

## Exit
- **Bracket set with the order:** TP = IBH + tp_ib_mult * W (long) / IBL - tp_ib_mult * W (short),
  absolute price. Default R:R at the trigger = (1.0 - 0.05) / 0.55 = 1.73.
- **MANUAL TIME EXIT (flagged, owner action):** close any filled position still open at 15:45 New York /
  17:15 Berlin. EXIT REMINDER from the engine (ER-03). Backtest per COMMON 8.5.
- Unfilled orders expire at EXPIRY (set as the MT5 order expiration; if the platform cannot set an
  expiry, the owner must cancel by hand at that time, which is a second flagged manual action, O4).
- Honest expectation: TP needs a day range of about 2 W (median US30 2 x 198 = 396 points against a
  median cash range of 361), so many trades will end at the time exit. QA reports exit-type shares; a flat
  tp_ib_mult response is not robustness evidence (COMMON 10.5.5). MISSED-EXIT must pass.

## Sizing
Engine standard on `sizing_entry` = CHASE LIMIT (trigger + 0.20 D), so a market fallback fill never risks
more than printed. Cap 3 lots. Median D (from the IB-width medians): US30 109, NAS100 69, GER40 45
points. The cap binds on about 80% (US30) to 98% (GER40) of tickets. Typical printed (worst-case) risk:
US30 about $390, NAS100 about $250, GER40 about $180; at a fill at the trigger about $326 / $207 / $151.
Pair days are two tickets, each with its own risk in the engine checks.

## Parameters
| Name | Default | Sensitivity grid | Reason |
|---|---|---|---|
| ib_minutes | 60 | 30, 45, 60, 75, 90 | The Market Profile initial balance is by convention the first trading hour; M15 multiples only. |
| tp_ib_mult | 1.0 | 0.5, 0.75, 1.0, 1.25, 1.5 | A 1x IB range extension (the day range doubling the IB) is the standard Market Profile extension target. |
| max_ib_datr | 0.6 | 0.5, 0.55, 0.6, 0.65, 0.7 | When the IB already spans 60% of a typical day's range, little extension is left and the half-IB stop is wide. Set from range geometry (65% of US days and 81% of GER40 days pass), not outcomes. |

Grid: one-at-a-time (13) plus a 5 x 5 heatmap of ib_minutes x tp_ib_mult (16 more) = **29 configurations**.

Frozen: b = 0.05 W; midpoint stop; DATR over 14 valid dates with >= 90% of bars; EXPIRY 12:00 New York /
11:30 Berlin for every ib_minutes; protocol constants (0.20 D chase, stops_buffer from symbol_info);
one ticket per index per day; time-exit clocks. The old order_life parameter is replaced by the fixed
clock EXPIRY (QA: it interacted with ib_minutes). The old 20-spread minimum-width guard is removed
(QA: inert).

## Expected trade frequency
Descriptive ticket counts only (COMMON 12; proxy `sign(x2)` instead of the HMM; ib 60), 2023-03-01 to
2026-04-30 (about 158 weeks):
- max_ib_datr 0.6: 491 tickets (3.1 per week), 327 inside the 9 test windows (3.0 per week); GER40 221,
  US30 140, NAS100 130. The proxy marks 35% of eligible IB ends ACTIVE.
- max_ib_datr 0.5: 346; 0.7: 592.
The fill rate of the stop orders was deliberately not estimated (it depends on prices after T). If fewer
than about half of the orders trigger, the strategy will miss the 150-trade minimum and fail on statistical
power; it is not to be loosened to fix that.

## Manual execution fit
Good. The ticket is printed at the IB end, before any breakout, and rests as a pending stop order at a
structural price; a 15-45 minute delay only matters when the breakout happens inside the delay, which the
state-only placement rule handles (market fill up to the CHASE LIMIT, else skip). SL, TP and trigger are
absolute structural levels. The order life (60-90 minutes after T at most grid points) is longer than the
maximum delay. One manual time exit per market.

## Costs and lot cap
Median stops of 109 / 69 / 45 points against p90 in-window spreads of about 1-2 points: round-trip spread
1-3% of D at base, under about 8% of D under STRESS with stop-order slippage. The cap holds realised risk to
about 0.15-0.4% of balance at the demo contract; the dollar edge per trade is small. Stop-entry fills pay
slippage (COMMON 8.4); TP fills do not.

## Look-ahead safeguards
- IBH, IBL, W use only bars closed by T; DATR uses prior dates only; RS_4h and the regime as in COMMON.
- Placement and fills use bars from P onward only; the placement rule uses only the quote at P.
- Local times by tz conversion; DST CI tests for New York and Berlin.
- CI: perturbing any bar after T leaves the ticket at T unchanged.

## Failure modes
- False breakouts on balance days: stop fills on the poke, then reversal to the midpoint.
- 10:00 New York releases (ISM, sentiment) fall inside the order life; FOMC/CPI days need
  `restricted_windows_utc` (O5). Backtest has no news calendar (limitation).
- Breakouts inside the delay beyond the chase limit are skipped, removing some of the strongest extensions.
- Pair days (US30 long, NAS100 short) can lose on both legs.
- Platform without order expiry adds a manual cancel.
- Demo IB levels can differ from the competition feed by a few points; the owner uses the ticket's prices.
- Overlap with RS-MOM-01 in the US morning; combined book uses one position per index, IBX first.

## QA issues resolved
| QA issue | Resolution |
|---|---|
| Engine cannot express pending orders (blocker) | Owner approval item O1/O4 and ER-01, ER-04 (order_type, trigger, expiry, pending risk in state.yaml and check_ticket, placement instructions). Strategy dropped if refused. |
| Placement rule needs chart history (major) | State-only STOP_ENTRY rule at placement; tolerance printed in points (0.20 D) |
| Rationale vs holding window (major) | Prior stated as weak; exit-type shares reported; MISSED-EXIT pass/fail; EXIT REMINDER requested |
| HMM measures the clock; lowest-vol state meaning drifts (COMMON major) | Deseasonalised HMM; K-invariant ACTIVE gate |
| Holidays (COMMON major) | Exchange calendar; DATR and eligibility use valid trading dates only |
| Overlap with NBM (COMMON major) | SES-NBM-02 dropped; one position per index across strategies |
| Ranking near-vacuous; pair days (minor) | Stated; pair days reported separately; each leg risk-checked |
| Hidden constants; order_life x ib_minutes (minor) | Frozen list; fixed clock EXPIRY for every grid point |
| Market fallback exceeds printed risk (minor) | Sized on the CHASE LIMIT |
| Restricted windows vs pending orders (minor) | No ticket if a restricted window overlaps [T, EXPIRY] |
| US500 / GER40 alone (COMMON minor) | US500 excluded (O6); GER40 alone stated |
| Missing member changes ranking (COMMON minor) | COMMON 4.2.2 group rule |
