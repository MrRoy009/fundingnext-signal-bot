# Strategy specs (GATE 2)

**GATE 2 APPROVED by the owner on 2026-11-02 with the default answers** (recorded in
`config.yaml` under `gates`):
- O1 entry protocols approved; O2 `valid_for_bars: 3`; O3 manual time exits and EXIT REMINDER
  approved (owner availability at those hours not yet confirmed); O6 US500 dropped from ranking and
  trading; O7 filter then rank; O8 day boundary 00:00 server time; O9 tick volume flagged and
  marginalised.
- Still open: O4 competition contract, commission, swap and pending-order support (backtests report
  the demo contract only until confirmed; live tickets stay suppressed); O5 holding and news rules;
  O10 hole backfill not done; O11 tick-spread calibration not approved (read-only allowlist unchanged).

Status: final GATE 2 spec set, written by Quant after QA review. **Nothing is backtested until the
owner approves GATE 2.** After approval the order is: Engineer implements `COMMON.md` and the
engineering requests, runs the descriptive feasibility gate (COMMON 10.1), and only then runs the
backtests listed below, logging every run in `docs/trial_log.csv`.

Shared rules for all strategies: [COMMON.md](COMMON.md) (features, index selection, HMM regime model,
entry protocols, backtest conventions, walk-forward validation, minimum trade counts, cost and delay
stress, trial accounting). Engineering work needed: [../engineering_requests.md](../engineering_requests.md).

**Feasibility gate (2026-11-03, counts only, `docs/feasibility_gate.md`):** RS-MOM-01 284 and
SES-IBX-01 268 candidates proceed to backtest. **MR-TWAP-01 is dropped** with 190 candidates,
below the pre-registered 200 (COMMON 12C, amendment A3). Trial budget now 58 (ceiling 78).

**GATE 3 validation (2026-11-07): both strategies REJECTED by QA** (`docs/validation/`).
58 pre-registered configurations run and logged (T0001-T0066 incl. run B and benchmarks); holdout
never opened. RS-MOM-01: 106 trades (< 150), STRESS-d3 E(R) -0.01, 4/9 folds positive, below the
placebo median, DSR 0.009/0.199. SES-IBX-01: 90 trades, negative in every scenario, 0/9 folds
positive, all 29 grid cells negative. Global PBO 0.067. No strategy is approved; the live registry
stays empty. Any new idea is a new GATE 2 spec and adds to N.

## Shortlist

| ID | Family | One-line idea | Regime gate | Free params | Grid size (configurations) |
|---|---|---|---|---|---|
| [RS-MOM-01](RS-MOM-01.md) | Momentum / relative strength (intraday) | Buy the in-window leader (sell the laggard) when its time-of-day-normalised 4-hour move freshly crosses a threshold; H1-ATR bracket; time exit 15:45 New York / 17:15 Berlin | ACTIVE (V_T >= 0) | 3 (theta, k, R) | 29 |
| [SES-IBX-01](SES-IBX-01.md) | Session structure (initial-balance breakout) | At the end of the first cash hour, a resting stop order beyond the IB on the strongest (long) / weakest (short) index; midpoint stop, IB-extension target | ACTIVE (V_T >= 0) | 3 (ib_minutes, tp_ib_mult, max_ib_datr) | 29 |
| [MR-TWAP-01](MR-TWAP-01.md) | Regime-conditional mean reversion | Fade a 2-ATR stretch from the session mean in the day-horizon leader (dip) / laggard (rally), limit-or-better entry, target the session mean | CALM (V_T < 0) | 2 (K, M) | 25 |

Grid size = one-at-a-time grid around the default plus one 5 x 5 heatmap of the two most important
parameters (k x R; ib_minutes x tp_ib_mult; K x M), counted once each.

The two ACTIVE strategies and the CALM strategy are complementary by construction: on a given index and
hour, at most one regime family can trade. SES-IBX-01 is conditional on owner approval of pending stop
orders (O1, O4); if refused it is dropped and the budget falls by 29.

## Planned trial budget

**Total configurations to be tested = 29 + 29 + 25 = 83.**

- Contingent, only for strategies that pass run A: shared-component sensitivity, 10 configurations each
  (rank lookback 4 values: L in {12, 14, 18, 20} for RS_4h or {20, 22, 26, 28} for RS_day; HMM training
  window {300, 430} days; Kmax {3, 5}; refit cadence {semi-monthly, bi-monthly}). Ceiling 3 x 10 = 30.
  **Hard ceiling of the trial count: 113.**
- Not trials (pass/fail checks or diagnostics on one configuration, never used to choose between
  configurations): delay d = 1/2/3, cost and slippage stress, MISSED-EXIT, EXIT-3X, benchmarks B1-B3,
  combined-book replay. They are logged with stage `stress`, `benchmark` or `portfolio`.
- If the owner rejects the entry protocols (O1), the COMMON fallback protocol replaces them; it is not
  run in addition. If the owner keeps US500 (O6), each strategy's grid is re-run with US500 as a separate,
  logged trial set and the ceiling rises accordingly.
- Every configuration run counts toward N_raw for the deflated Sharpe ratio and PBO (COMMON 10.7),
  including failed and abandoned ones. Any change to a frozen constant (COMMON 11 and each spec) is a new
  trial and needs QA sign-off.

## Dropped candidates

| ID | Family | Reason dropped |
|---|---|---|
| RS-MOM-02 | Momentum (H1 leader, M15 pullback, multi-day bracket) | Largely the same bet as RS-MOM-01 (long the leader; combined book is one exposure). Needs overnight and weekend holding, which the competition rules have not confirmed. Swap at the 3-lot notional costs an estimated 5-20% of R per hold. Friday manual exits fail on holiday and early-close Fridays. Can be re-proposed as a new spec at a later gate if the owner confirms holding rules. |
| SES-NBM-02 | Session (noise-band intraday momentum) | Overlaps SES-IBX-01 and RS-MOM-01 (same indices, same morning, same direction; 45% of crossings at the first check). Its 2R target is rarely reachable, so the real exit was the manual time exit and tp_r would look flat by construction. It departs from the cited paper in nearly every component (static stop, no flips, CFDs, morning only), so the published evidence does not transfer. The fixed skip rule adversely selected about 40% of signals. |
| MR-BB-02 | Mean reversion (Bollinger re-entry) | Same hypothesis as MR-TWAP-01, so it adds trials without adding diversity. Its 20-bar band spans overnight and pre-open bars, which narrows the bands and turns early re-entries into fades of opening drives (a definition problem). It has 5 hidden constants, two set after looking at setup counts, and a planned R:R of about 1, which makes it the most cost- and fill-sensitive candidate. |

## Design decisions made from descriptive counts (non-outcome; recorded for the trial audit)
No profit, win rate or forward return of any rule has been computed. These choices used counts or
distributions of setup conditions only:
- MR-TWAP-01: no RS sign condition when two US indices pass (a sign condition halved setup counts).
- RS-MOM-01: window starts after the first cash hour and ends 2 h 15 min before the time exit
  (QA's ATR-lag and time-exit findings); the trigger requires the index's own score to cross theta.
- SES-IBX-01: max_ib_datr default 0.6 from IB-width geometry (share of days passing).
- All: US500 removed (lot-cap risk about $35-60); GER40 ranked alone because its window never overlaps the
  US window; tick volume flagged 2024-10-04 to 2025-01-07; segment break at gaps over 7 days.
- Setup counts in each spec were computed with a volatility proxy `sign(x2)` in place of the HMM:
  `C:\Users\lunar\AppData\Local\Temp\claude\D--Finance-Fundingnext-bot\6c140617-18ea-4b43-9cc9-6862d0b2cb3e\scratchpad\gate2_final_setupcount_c.py`.

## Owner decisions needed at GATE 2
Listed in COMMON section 13 (O1-O11). The ones that block backtests: O1 entry protocols, O2
valid_for_bars = 3, O3 time-exit availability and EXIT REMINDER, O4 competition contract, commission,
swap and pending-order support, O5 holding and news rules, O6 US500, O7 selection reading, O8 day boundary.

## Rule
Nothing in this folder is backtested, optimised or run on the holdout until the owner approves GATE 2.
The holdout (from 2026-05-01) stays locked until GATE 3 approval and is then used once.
