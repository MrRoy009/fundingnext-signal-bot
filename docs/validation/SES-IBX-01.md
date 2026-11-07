# Validation report: SES-IBX-01

**Overall verdict: FAIL**

- Demo prices, spreads and contracts may differ from the competition feed: every number in this report comes from the MT5 DEMO feed and the demo contract.
- The final holdout (bars from 2026-05-01 00:00 UTC) is untouched: no data on or after that date was loaded for this report (final_holdout was never passed). It opens once, after the owner approves GATE 3.
- The competition contract (owner decision O4: USD per point per lot, commission, swap) is NOT confirmed; the replay below is at the demo contract only.

## Provenance

- Strategy: SES-IBX-01; parameters: ib_minutes=60, tp_ib_mult=1, max_ib_datr=0.6
- Git SHA: 734cc50f1fe1eb8a3488aff40d8ed974d9bcaa51-dirty
- Data SHA-256: 9ab58a511671c5c1
- config.yaml SHA-256: 9fd0c89a54d21899
- Data: MT5 DEMO, BarStore.load pre-holdout, last bar date 2026-04-30
- Trial log rows: T0034, T0035, T0036, T0037, T0038, T0039, T0040, T0041, T0042, T0043, T0044, T0045, T0046, T0047, T0048, T0049, T0050, T0051, T0052, T0053, T0054, T0055, T0056, T0057, T0058, T0059, T0060, T0061, T0062, T0063, T0064, T0065, T0066
- Competition contract (O4): NOT confirmed (config.yaml rules.competition_usd_per_point)
- Generated (UTC): 2026-11-07 06:12
- Run rows: run A T0034; run B T0063; benchmarks B1-placebo T0064, B2-random-index T0065, B3-gate-off T0066.
- Run git state(s): 734cc50f1fe1eb8a3488aff40d8ed974d9bcaa51-dirty.
- Global step (COMMON 10.7) artifacts: backtest/runs/global/ (global_stats.json, trial_matrix.parquet 579 days x 58 trials, clusters.csv, pbo_logits.csv); trial log sha256 375e778206cffb25.
- Data last M15 bar: US30 2026-04-30 23:45:00+00:00, NAS100 2026-04-30 23:45:00+00:00, GER40 2026-04-30 19:45:00+00:00.
- 10.4 competition feasibility (report only): 170 of 560 rolling windows (30.4%); over windows with data only 34.3%.
- B1 placebo geometry: shifted run-A bracket (SES_PLACEBO_SHIFT, COMMON 12D A4.2).

## Scenarios (run A, concatenated test windows; COMMON 10.3)

| Scenario | Trades | Win rate | Expectancy $ | Expectancy R | Daily Sharpe | Days | Total $ | Max DD $ | Max DD % |
|---|---|---|---|---|---|---|---|---|---|
| BASE | 90 | 31.1% | -59.51 | -0.179R | -0.0803 | 579 | -5,355.50 | 5,828.85 | 5.80% |
| STRESS-d1 | 113 | 35.4% | -46.57 | -0.131R | -0.0650 | 579 | -5,262.13 | 6,230.01 | 6.20% |
| STRESS-d2 | 90 | 31.1% | -70.70 | -0.229R | -0.0939 | 579 | -6,363.09 | 6,789.63 | 6.76% |
| STRESS-d3 | 63 | 31.7% | -82.20 | -0.252R | -0.0969 | 579 | -5,178.83 | 5,552.89 | 5.53% |
| MISSED-EXIT | 90 | 30.0% | -76.79 | -0.242R | -0.0942 | 579 | -6,911.36 | 7,337.90 | 7.31% |
| EXIT-3X | 90 | 31.1% | -70.85 | -0.229R | -0.0941 | 579 | -6,376.63 | 6,803.17 | 6.77% |

Win rate = share of filled trades with $ P&L > 0. Daily Sharpe = mean / std of the end-of-day MTM equity change over 'Days' calendar days, non-annualised. Max DD % = the largest drawdown relative to the running peak (it can occur at a different time than the largest $ drawdown); docs/trial_log.csv's max_dd_pct is measured at the largest $ drawdown (negative sign) and can be smaller in magnitude. BASE, STRESS-d1 and EXIT-3X are report-only; STRESS-d2, STRESS-d3 and MISSED-EXIT are pass/fail. Ticket STATS (win rate, expectancy) come from STRESS-d2.

Secondary sample 2023-03-01..2026-04-30 (STRESS-d2, report only): 165 trades, -42.22 $/trade, -0.154R, Sharpe -0.0716, total -6,966.84, max DD 7,038.75.

## Walk-forward folds (run A, STRESS-d2; COMMON 10.2, 10.4)

| Fold | Window | Data share | Trades | Pro-rated minimum | Minimum met | P&L $ | Mean R | Positive |
|---|---|---|---|---|---|---|---|---|
| 1 | 2024-02-01..2024-04-30 | 100.0% | 23 | 4.00 | yes | -940.23 | -0.280R | no |
| 2 | 2024-05-01..2024-07-31 | 100.0% | 16 | 4.00 | yes | -317.61 | +0.069R | no |
| 3 | 2024-08-01..2024-10-31 | 69.7% | 9 | 2.79 | yes | -1,127.78 | -0.271R | no |
| 4 | 2024-11-01..2025-01-31 | 28.6% | 0 | 1.14 | NO | +0.00 | n/a | no |
| 5 | 2025-02-01..2025-04-30 | 100.0% | 11 | 4.00 | yes | -337.09 | -0.163R | no |
| 6 | 2025-05-01..2025-07-31 | 80.3% | 6 | 3.21 | yes | -665.24 | -0.485R | no |
| 7 | 2025-08-01..2025-10-31 | 59.1% | 0 | 2.36 | NO | +0.00 | n/a | no |
| 8 | 2025-11-01..2026-01-31 | 100.0% | 13 | 4.00 | yes | -1,320.08 | -0.303R | no |
| 9 | 2026-02-01..2026-04-30 | 100.0% | 12 | 4.00 | yes | -1,655.06 | -0.345R | no |

Pro-rated minimum = 4 trades per full fold x data share (amendment A1). A trade belongs to the fold of its signal time T.

## Breakdowns (run A, STRESS-d2; COMMON 8.9, 10.4 bucket rule)

### Per index

| Bucket | Trades | Share | Total $ | Mean $ | Mean R |
|---|---|---|---|---|---|
| GER40 | 51 | 56.7% | -2,914.86 | -57.15 | -0.234R |
| OTHER (NAS100, US30) | 39 | 43.3% | -3,448.23 | -88.42 | -0.221R |

### Per side

| Bucket | Trades | Share | Total $ | Mean $ | Mean R |
|---|---|---|---|---|---|
| BUY | 54 | 60.0% | -2,330.51 | -43.16 | -0.184R |
| SELL | 36 | 40.0% | -4,032.59 | -112.02 | -0.295R |

### Per exit type

| Bucket | Trades | Share | Total $ | Mean $ | Mean R |
|---|---|---|---|---|---|
| SL | 58 | 64.4% | -14,219.69 | -245.17 | -0.922R |
| OTHER (TP, time_exit) | 32 | 35.6% | +7,856.60 | +245.52 | +1.028R |

### Per regime (ACTIVE / CALM at signal time)

| Bucket | Trades | Share | Total $ | Mean $ | Mean R |
|---|---|---|---|---|---|
| ACTIVE | 90 | 100.0% | -6,363.09 | -70.70 | -0.229R |

Buckets with fewer than 30 trades are merged into OTHER; OTHER's statistics are withheld if it has fewer than 30 trades.

### Pair days (SES-IBX-01: US30 and NAS100 tickets of opposite sides on one New York date)

Pair days: 13; with both legs filled: 2. A pair day is two tickets, each with its own risk; both legs can lose.

| Trades | N | Share | Total $ | Mean $ | Mean R |
|---|---|---|---|---|---|
| single ticket | 82 | 91.1% | -4,775.62 | -58.24 | -0.195R |
| OTHER (pair-day leg) | 8 | 8.9% | withheld (n < 30) |  |  |

## Run B (walk-forward selected; COMMON 10.2)

| Fold | Selected (one-at-a-time grid, train window) | Train daily Sharpe | Embargoed train trades | Test trades | Test P&L $ |
|---|---|---|---|---|---|
| 1 | ib_minutes=60, tp_ib_mult=1, max_ib_datr=0.55 | -0.0130 | 0 | 21 | -1,002.64 |
| 2 | ib_minutes=60, tp_ib_mult=1, max_ib_datr=0.55 | -0.0284 | 0 | 13 | +590.65 |
| 3 | ib_minutes=60, tp_ib_mult=1, max_ib_datr=0.55 | 0.0269 | 0 | 8 | -1,024.52 |
| 4 | ib_minutes=60, tp_ib_mult=1, max_ib_datr=0.7 | -0.0363 | 0 | 0 | +0.00 |
| 5 | ib_minutes=60, tp_ib_mult=1, max_ib_datr=0.7 | -0.0171 | 0 | 14 | +574.42 |
| 6 | ib_minutes=60, tp_ib_mult=1, max_ib_datr=0.5 | 0.0130 | 0 | 7 | -1,148.38 |
| 7 | ib_minutes=60, tp_ib_mult=1, max_ib_datr=0.5 | -0.0442 | 0 | 0 | +0.00 |
| 8 | ib_minutes=60, tp_ib_mult=1, max_ib_datr=0.5 | -0.0108 | 0 | 8 | +118.99 |
| 9 | ib_minutes=60, tp_ib_mult=1, max_ib_datr=0.5 | -0.0047 | 0 | 10 | -1,849.11 |

Out-of-sample (STRESS-d2): 81 trades, -46.18 $/trade, -0.163R, total -3,740.58, daily Sharpe -0.0577. Embargo: filled train trades exiting after the train window end are dropped from selection (10.2).

## Parameter sensitivity (COMMON 10.5.5)

Scenario STRESS-d2. Defaults: ib_minutes=60, tp_ib_mult=1, max_ib_datr=0.6. Pre-registered configurations: 29. Cell value = $ expectancy per trade (trade count); a cell with fewer than 100 trades is insufficient and counts as failed; a cell not run is never a pass.

### One at a time

| Parameter | Value | Role | Trades | Expectancy $ | Expectancy R | Cell | Exit-type shares |
|---|---|---|---|---|---|---|---|
| ib_minutes | 30 |  | 142 | -46.37 | -0.276R | FAIL | SL 66% TP 25% TE 8% |
| ib_minutes | 45 | neighbour | 106 | -48.74 | -0.153R | FAIL | SL 63% TP 29% TE 8% |
| ib_minutes | 60 | default | 90 | -70.70 | -0.229R | insufficient | SL 64% TP 26% TE 10% |
| ib_minutes | 75 | neighbour | 57 | -80.91 | -0.218R | insufficient | SL 65% TP 28% TE 7% |
| ib_minutes | 90 |  | 49 | -73.51 | -0.210R | insufficient | SL 63% TP 24% TE 12% |
| tp_ib_mult | 0.5 |  | 90 | -99.60 | -0.344R | insufficient | SL 60% TP 37% TE 3% |
| tp_ib_mult | 0.75 | neighbour | 90 | -86.38 | -0.290R | insufficient | SL 63% TP 30% TE 7% |
| tp_ib_mult | 1 | default | 90 | -70.70 | -0.229R | insufficient | SL 64% TP 26% TE 10% |
| tp_ib_mult | 1.25 | neighbour | 90 | -74.30 | -0.207R | insufficient | SL 67% TP 20% TE 13% |
| tp_ib_mult | 1.5 |  | 90 | -65.81 | -0.168R | insufficient | SL 67% TP 17% TE 17% |
| max_ib_datr | 0.5 |  | 62 | -64.73 | -0.249R | insufficient | SL 66% TP 26% TE 8% |
| max_ib_datr | 0.55 | neighbour | 79 | -60.01 | -0.203R | insufficient | SL 65% TP 28% TE 8% |
| max_ib_datr | 0.6 | default | 90 | -70.70 | -0.229R | insufficient | SL 64% TP 26% TE 10% |
| max_ib_datr | 0.65 | neighbour | 109 | -64.32 | -0.206R | FAIL | SL 61% TP 25% TE 15% |
| max_ib_datr | 0.7 |  | 115 | -48.20 | -0.164R | FAIL | SL 59% TP 27% TE 14% |

### Heatmap ib_minutes x tp_ib_mult

Rows tp_ib_mult, columns ib_minutes; the default cell is in bold. Exit-type shares of each cell are next to the expectancy grid (SL, TP, TE = time exit, GAP = data_gap_exit, END = end_of_data).

| tp_ib_mult \ ib_minutes | ib_minutes=30 | ib_minutes=45 | ib_minutes=60 | ib_minutes=75 | ib_minutes=90 | exits ib_minutes=30 | exits ib_minutes=45 | exits ib_minutes=60 | exits ib_minutes=75 | exits ib_minutes=90 |
|---|---|---|---|---|---|---|---|---|---|---|
| tp_ib_mult=1.5 | -45.22 (n=142) | -27.17 (n=106) | insufficient (n=90) | insufficient (n=57) | insufficient (n=49) | SL 68% TE 16% TP 15% | SL 64% TP 20% TE 16% | SL 67% TP 17% TE 17% | SL 67% TE 18% TP 16% | SL 65% TE 20% TP 14% |
| tp_ib_mult=1.25 | -48.03 (n=142) | -36.45 (n=106) | insufficient (n=90) | insufficient (n=57) | insufficient (n=49) | SL 68% TP 19% TE 13% | SL 64% TP 25% TE 11% | SL 67% TP 20% TE 13% | SL 65% TP 21% TE 14% | SL 63% TP 18% TE 18% |
| tp_ib_mult=1 | -46.37 (n=142) | -48.74 (n=106) | **insufficient (n=90)** | insufficient (n=57) | insufficient (n=49) | SL 66% TP 25% TE 8% | SL 63% TP 29% TE 8% | SL 64% TP 26% TE 10% | SL 65% TP 28% TE 7% | SL 63% TP 24% TE 12% |
| tp_ib_mult=0.75 | -50.21 (n=142) | -58.78 (n=106) | insufficient (n=90) | insufficient (n=57) | insufficient (n=49) | SL 63% TP 32% TE 5% | SL 61% TP 35% TE 4% | SL 63% TP 30% TE 7% | SL 60% TP 35% TE 5% | SL 59% TP 35% TE 6% |
| tp_ib_mult=0.5 | -69.75 (n=137) | -56.29 (n=106) | insufficient (n=90) | insufficient (n=57) | insufficient (n=49) | SL 61% TP 38% TE 1% | SL 53% TP 47% | SL 60% TP 37% TE 3% | SL 56% TP 44% | SL 53% TP 45% TE 2% |

Pre-registered cells: 29; run: 29; positive and sufficient: 0 (0.0% of 29); insufficient: 17; missing: 0. Immediate neighbours of the default: 0 of 6 pass. A flat response of a parameter that rarely binds (for example a TP that is rarely reached) is not robustness evidence: read the exit-type shares.

## Bootstrap (COMMON 10.5.6)

| Method | Total $ p5 | Total $ p50 | Total $ p95 | Max DD $ p5 | Max DD $ p50 | Max DD $ p95 | Settings |
|---|---|---|---|---|---|---|---|
| Stationary block bootstrap of the daily series | -10,826.69 | -6,312.10 | -2,049.91 | 3,437.71 | 7,020.54 | 11,349.67 | STRESS-d2, n_obs 579, 10000 resamples, seed 12345, mean block 5 days |
| Regime-conditional bootstrap of trades | -10,664.29 | -6,413.91 | -1,983.48 | 3,632.49 | 7,202.78 | 11,087.82 | STRESS-d2, n_obs 90, 10000 resamples, seed 12345 |

Regime-conditional bucket counts (kept fixed): ACTIVE 90.

## Shuffle test (COMMON 10.5.10, report only)

10000 permutations of trade order (seed 12345): 95th percentile of the worst losing streak 15.0 trades; of max drawdown 8,206.65 $.

## Benchmarks (COMMON 10.6; never selectable, not counted in N)

Benchmarks run under STRESS-d2.

| Run | Expectancy $ / trade | Total $ | Trades | Role |
|---|---|---|---|---|
| Strategy (run A, STRESS-d2) | -70.70 | -6,363.09 | 90 |  |
| B1 placebo (200 draws, seeds 1..200) |  | mean -4,467.09; p95 -1,717.53 |  | pass criterion 7 |
| B2 random index | mean -31.55 |  | 86 | report |
| B3 gate off | -52.10 | -9,950.45 | 191 | report |

The strategy does not beat B2's mean: the relative-strength selection adds nothing.
B3 (gate off) >= the gated run in $ expectancy: the gate adds nothing. The gate is not changed in response.

## Competition replay (COMMON 10.5.9)

30-calendar-day windows starting every Monday in the test span.

The competition contract (owner decision O4: USD per point per lot, commission, swap) is NOT confirmed; the replay below is at the demo contract only.

| Contract | USD per point per lot | Windows | 10% total-loss breaches | 5% daily-loss breaches | Personal 3% stop breaches | Windows with >= 5 trade days |
|---|---|---|---|---|---|---|
| demo | US30 1, NAS100 1, GER40 1.12 | 100 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 37.0% |

## Deflated Sharpe and PBO (COMMON 10.7)

| Trial count | N | V (variance of SR) | DSR | Threshold |
|---|---|---|---|---|
| N_raw (all configurations ever run) | 58 | 0.005044 | 0.0000 | >= 0.95 |
| N_eff (clusters at correlation 0.7) | 10 | 0.004752 | 0.0000 | >= 0.95 |

Trial log: 58 counted configurations (all strategies) at render time; N_raw used by the DSR: 58.

SR_hat (daily, non-annualised) -0.0939 over 579 days; skew -0.651, kurtosis (non-excess) 15.850.

PBO (CSCV, S = 16 blocks, 12870 splits, 58 trials): 0.067; threshold < 0.5.

## Regime diagnostics (COMMON 6.8)

Source: docs/feasibility_gate.md. Gate ACTIVE: blocks 59.0% of in-window cycles: **functional**.

| Index | Block share | 5-95% check |
|---|---|---|
| GER40 | 55.0% | functional |
| NAS100 | 63.9% | functional |
| US30 | 58.1% | functional |

Folds where every in-window cycle is blocked (regime UNAVAILABLE: no model or warm-up): 4, 7.

## Pass / fail (COMMON 10.4 minimums, 10.5 criteria 1-9)

| Criterion | Requirement | Measured | Threshold | Result |
|---|---|---|---|---|
| 10.4 trades | Filled out-of-sample trades, run A STRESS-d2 | 90 | >= 150 | **FAIL** |
| 10.4 folds | Folds with >= 4 trades per full fold (pro-rated by data share) | 7 of 9 | >= 7 of 9 | **PASS** |
| 1 | Run A expectancy > 0 in $ and R, daily Sharpe > 0, STRESS-d2 and STRESS-d3 | STRESS-d2: -70.70 $/trade, -0.229R, Sharpe -0.0939; STRESS-d3: -82.20 $/trade, -0.252R, Sharpe -0.0969 | > 0 each | **FAIL** |
| 2 | MISSED-EXIT expectancy > 0 in $ | -76.79 $/trade | > 0 | **FAIL** |
| 3 | Run B out-of-sample expectancy > 0 in $ (STRESS-d2) | -46.18 $/trade (81 trades) | > 0 | **FAIL** |
| 4 | Folds with positive $ P&L, run A STRESS-d2 | 0 of 9 | >= 6 of 9 | **FAIL** |
| 5 | Sensitivity: every immediate neighbour of each default has $ expectancy > 0 (>= 100 trades); >= 80% of all grid cells positive | neighbours 0 of 6 pass; cells 0 of 29 positive (0.0%), 17 insufficient | all neighbours; >= 80% of 29 | **FAIL** |
| 6 | Stationary block bootstrap of the run A STRESS-d2 daily series (10,000 resamples, mean block 5, seed 12345): 5th percentile of total $ P&L > 0 | -10,826.69 | > 0 | **FAIL** |
| 7 | Run A total $ P&L > 95th percentile of placebo B1 (STRESS-d2, 200 draws, seeds 1..200) | -6,363.09 vs B1 p95 -1,717.53 | > B1 p95 | **FAIL** |
| 8 | DSR >= 0.95 under N_raw and N_eff; PBO < 0.5 | DSR 0.000 (N_raw 58; trial log 58), 0.000 (N_eff 10); PBO 0.067 | DSR >= 0.95; PBO < 0.5 | **FAIL** |
| 9 | Competition replay at the demo AND the competition contract | demo: 0 total-loss windows, daily-loss 0.00% of 100; competition: contract NOT confirmed (O4) | 0 total-loss windows; daily-loss <= 1% of windows | **NOT EVALUATED** |
| 10 (report) | Shuffle test: p95 worst losing streak and max DD | streak 15.0, DD 8,206.65 | report only | **INFO** |
| 10.4 (report) | Trade days: share of rolling 20-trading-day windows with >= 5 trade days (applies to the combined set) | 30.4% | >= 80% (combined set) | **INFO** |

**Overall verdict: FAIL.** PASS only if every row except the report-only rows passes; NOT EVALUATED never counts as a pass. Standards are never lowered.

## SPEC NOTEs (report generator readings)

1. Pro-rated fold minimum: a fold meets it when trades >= 4 x data share (no rounding); a fold with data share 0 has minimum 0. Criterion 4 is not pro-rated: a fold without trades is not positive.
2. Bucket rule: per-index, per-side, per-exit-type and per-regime buckets with < 30 trades merge into OTHER; if OTHER has < 30 trades its statistics are withheld.
3. Immediate grid neighbours: the adjacent grid values of each parameter with the others at default, and the 4 axis-adjacent cells of the default in the heatmap; all grid cells = the distinct configurations of both grids, each counted once; insufficient (< 100 trades) cells count as failed.
4. B2 mean: mean $ P&L per trade of the B2 runs (pooled over draws), compared with the strategy's $ expectancy in the same scenario; the strategy must exceed it.
5. Criterion 9 needs the competition contract (O4); while it is not confirmed the criterion is NOT EVALUATED (FAIL if the demo replay fails) and the overall verdict cannot be PASS.
6. 10.4 competition feasibility applies to the approved combined set; per strategy it is reported only.
7. Criteria 1, 4, 6, 7 and the 10.4 minimums use run A under STRESS-d2 (criterion 1 also STRESS-d3); run B, sensitivity and benchmarks must also be STRESS-d2.
8. An undefined DSR or PBO (NaN) makes criterion 8 NOT EVALUATED.
9. N_raw must equal the counted configurations in docs/trial_log.csv at render time (all strategies); otherwise criterion 8 is NOT EVALUATED until DSR / PBO are recomputed.
10. Max DD % is the largest drawdown relative to the running peak, independent of the largest $ drawdown; the trial log's max_dd_pct is measured at the largest $ drawdown (negative sign) and can be smaller in magnitude.
11. Run A must be the registered default parameters; any other vector makes every run-A row NOT EVALUATED.
12. Run B: selections for exactly folds 1..9, each a one-at-a-time grid cell with its embargo count; per-fold test trades and P&L must sum to the run-B totals.
13. Criterion 6 uses the stationary bootstrap of the STRESS-d2 daily series (n_obs = its days); criterion 8 needs the DSR's SR_hat and T_days to equal the STRESS-d2 daily Sharpe and days; the regime-conditional buckets must be ACTIVE / CALM.
14. The trades table, the STRESS-d2 trade count and the fold trade counts must agree; otherwise the 10.4 rows and criterion 4 are NOT EVALUATED.
15. Pair days (SES-IBX-01): New York dates with issued US30 and NAS100 candidates of opposite sides (any status); pair-day legs vs other filled trades under the 30-trade rule.
16. Global step: N_raw = 58 counted configurations of all strategies (RS-MOM-01 29, SES-IBX-01 29; benchmarks and run B composites not counted); V and N_eff are computed across all 58 trials' STRESS-d2 daily $ series, so N_eff and PBO are the same numbers in every strategy's report.
17. PBO is computed once by CSCV over the GLOBAL trial matrix (579 days x 58 trials, S = 16, earliest 579 mod 16 = 3 days dropped); it measures the overfitting risk of selecting the best of all 58 configurations, not of this strategy alone.
18. Bootstrap, regime bootstrap and shuffle were recomputed with validation.stats from the saved STRESS-d2 run files and equal the run phase's stats.json.

## Limitations

- Demo prices, spreads and contracts may differ from the competition feed: every number in this report comes from the MT5 DEMO feed and the demo contract.
- The final holdout (bars from 2026-05-01 00:00 UTC) is untouched: no data on or after that date was loaded for this report (final_holdout was never passed). It opens once, after the owner approves GATE 3.
- Swap and tick value are current demo values applied to all history (COMMON 8.6, 8.7).
