# Validation report: RS-MOM-01

**Overall verdict: FAIL**

- Demo prices, spreads and contracts may differ from the competition feed: every number in this report comes from the MT5 DEMO feed and the demo contract.
- The final holdout (bars from 2026-05-01 00:00 UTC) is untouched: no data on or after that date was loaded for this report (final_holdout was never passed). It opens once, after the owner approves GATE 3.
- The competition contract (owner decision O4: USD per point per lot, commission, swap) is NOT confirmed; the replay below is at the demo contract only.

## Provenance

- Strategy: RS-MOM-01; parameters: theta=1, k=1, R=1.5
- Git SHA: 734cc50f1fe1eb8a3488aff40d8ed974d9bcaa51-dirty
- Data SHA-256: 9ab58a511671c5c1
- config.yaml SHA-256: ee55bca19d48a871 (scenario config hashes: {'MISSED-EXIT': '9fd0c89a54d21899'})
- Data: MT5 DEMO, BarStore.load pre-holdout, last bar date 2026-04-30
- Trial log rows: T0001, T0002, T0003, T0004, T0005, T0006, T0007, T0008, T0009, T0010, T0011, T0012, T0013, T0014, T0015, T0016, T0017, T0018, T0019, T0020, T0021, T0022, T0023, T0024, T0025, T0026, T0027, T0028, T0029, T0030, T0031, T0032, T0033
- Competition contract (O4): NOT confirmed (config.yaml rules.competition_usd_per_point)
- Generated (UTC): 2026-11-07 06:12
- Run rows: run A T0001; run B T0030; benchmarks B1-placebo T0031, B2-random-index T0032, B3-gate-off T0033.
- Run git state(s): 734cc50f1fe1eb8a3488aff40d8ed974d9bcaa51-dirty.
- Global step (COMMON 10.7) artifacts: backtest/runs/global/ (global_stats.json, trial_matrix.parquet 579 days x 58 trials, clusters.csv, pbo_logits.csv); trial log sha256 375e778206cffb25.
- Data last M15 bar: US30 2026-04-30 23:45:00+00:00, NAS100 2026-04-30 23:45:00+00:00, GER40 2026-04-30 19:45:00+00:00.
- 10.4 competition feasibility (report only): 175 of 560 rolling windows (31.2%); over windows with data only 35.4%.
- MISSED-EXIT was run after A4.1 (swap mode 3 converted with costs.margin_ccy_usd) as a scenario of T0001 (COMMON 12D A4.3); its config hash differs from T0001's only by that A4 key.

## Scenarios (run A, concatenated test windows; COMMON 10.3)

| Scenario | Trades | Win rate | Expectancy $ | Expectancy R | Daily Sharpe | Days | Total $ | Max DD $ | Max DD % |
|---|---|---|---|---|---|---|---|---|---|
| BASE | 107 | 51.4% | +68.02 | +0.192R | 0.1062 | 579 | +7,278.56 | 1,703.73 | 1.67% |
| STRESS-d1 | 141 | 46.8% | +29.18 | +0.080R | 0.0511 | 579 | +4,114.55 | 1,846.34 | 1.80% |
| STRESS-d2 | 106 | 49.1% | +48.77 | +0.122R | 0.0767 | 579 | +5,169.29 | 1,857.79 | 1.84% |
| STRESS-d3 | 91 | 41.8% | +5.75 | -0.012R | 0.0078 | 579 | +522.89 | 2,374.92 | 2.37% |
| MISSED-EXIT | 106 | 44.3% | +65.31 | +0.153R | 0.0818 | 579 | +6,922.39 | 2,410.33 | 2.40% |
| EXIT-3X | 106 | 49.1% | +48.53 | +0.122R | 0.0763 | 579 | +5,143.98 | 1,865.86 | 1.84% |

Win rate = share of filled trades with $ P&L > 0. Daily Sharpe = mean / std of the end-of-day MTM equity change over 'Days' calendar days, non-annualised. Max DD % = the largest drawdown relative to the running peak (it can occur at a different time than the largest $ drawdown); docs/trial_log.csv's max_dd_pct is measured at the largest $ drawdown (negative sign) and can be smaller in magnitude. BASE, STRESS-d1 and EXIT-3X are report-only; STRESS-d2, STRESS-d3 and MISSED-EXIT are pass/fail. Ticket STATS (win rate, expectancy) come from STRESS-d2.

Secondary sample 2023-03-01..2026-04-30 (STRESS-d2, report only): 182 trades, +24.35 $/trade, +0.026R, Sharpe 0.0481, total +4,432.59, max DD 2,405.16.

## Walk-forward folds (run A, STRESS-d2; COMMON 10.2, 10.4)

| Fold | Window | Data share | Trades | Pro-rated minimum | Minimum met | P&L $ | Mean R | Positive |
|---|---|---|---|---|---|---|---|---|
| 1 | 2024-02-01..2024-04-30 | 100.0% | 21 | 4.00 | yes | -54.47 | +0.067R | no |
| 2 | 2024-05-01..2024-07-31 | 100.0% | 16 | 4.00 | yes | +427.99 | +0.038R | yes |
| 3 | 2024-08-01..2024-10-31 | 69.7% | 12 | 2.79 | yes | -787.83 | -0.259R | no |
| 4 | 2024-11-01..2025-01-31 | 28.6% | 0 | 1.14 | NO | +0.00 | n/a | no |
| 5 | 2025-02-01..2025-04-30 | 100.0% | 24 | 4.00 | yes | +2,762.39 | +0.250R | yes |
| 6 | 2025-05-01..2025-07-31 | 80.3% | 8 | 3.21 | yes | -220.27 | -0.171R | no |
| 7 | 2025-08-01..2025-10-31 | 59.1% | 0 | 2.36 | NO | +0.00 | n/a | no |
| 8 | 2025-11-01..2026-01-31 | 100.0% | 17 | 4.00 | yes | +1,740.20 | +0.380R | yes |
| 9 | 2026-02-01..2026-04-30 | 100.0% | 8 | 4.00 | yes | +1,301.28 | +0.374R | yes |

Pro-rated minimum = 4 trades per full fold x data share (amendment A1). A trade belongs to the fold of its signal time T.

## Breakdowns (run A, STRESS-d2; COMMON 8.9, 10.4 bucket rule)

### Per index

| Bucket | Trades | Share | Total $ | Mean $ | Mean R |
|---|---|---|---|---|---|
| GER40 | 65 | 61.3% | +5,409.85 | +83.23 | +0.207R |
| OTHER (NAS100, US30) | 41 | 38.7% | -240.56 | -5.87 | -0.012R |

### Per side

| Bucket | Trades | Share | Total $ | Mean $ | Mean R |
|---|---|---|---|---|---|
| BUY | 58 | 54.7% | +1,764.65 | +30.42 | +0.028R |
| SELL | 48 | 45.3% | +3,404.64 | +70.93 | +0.237R |

### Per exit type

| Bucket | Trades | Share | Total $ | Mean $ | Mean R |
|---|---|---|---|---|---|
| SL | 47 | 44.3% | -9,617.01 | -204.62 | -0.749R |
| TP | 30 | 28.3% | +11,903.05 | +396.77 | +1.328R |
| OTHER (time_exit) | 29 | 27.4% | withheld (n < 30) |  |  |

### Per regime (ACTIVE / CALM at signal time)

| Bucket | Trades | Share | Total $ | Mean $ | Mean R |
|---|---|---|---|---|---|
| ACTIVE | 106 | 100.0% | +5,169.29 | +48.77 | +0.122R |

Buckets with fewer than 30 trades are merged into OTHER; OTHER's statistics are withheld if it has fewer than 30 trades.

## Run B (walk-forward selected; COMMON 10.2)

| Fold | Selected (one-at-a-time grid, train window) | Train daily Sharpe | Embargoed train trades | Test trades | Test P&L $ |
|---|---|---|---|---|---|
| 1 | theta=1, k=0.75, R=1.5 | 0.0238 | 0 | 17 | +341.48 |
| 2 | theta=1, k=0.75, R=1.5 | 0.0547 | 0 | 12 | +343.84 |
| 3 | theta=1, k=0.75, R=1.5 | 0.0932 | 0 | 11 | -1,112.50 |
| 4 | theta=0.8, k=1, R=1.5 | 0.0316 | 0 | 0 | +0.00 |
| 5 | theta=0.8, k=1, R=1.5 | 0.0238 | 0 | 28 | +1,718.35 |
| 6 | theta=1, k=1, R=1.25 | 0.0915 | 0 | 8 | -293.89 |
| 7 | theta=1, k=1, R=1.25 | 0.0742 | 0 | 0 | +0.00 |
| 8 | theta=1, k=1, R=1.25 | 0.1021 | 0 | 17 | +1,321.70 |
| 9 | theta=1, k=1, R=1 | 0.1302 | 0 | 8 | +1,241.08 |

Out-of-sample (STRESS-d2): 101 trades, +35.25 $/trade, +0.074R, total +3,560.05, daily Sharpe 0.0476. Embargo: filled train trades exiting after the train window end are dropped from selection (10.2).

## Parameter sensitivity (COMMON 10.5.5)

Scenario STRESS-d2. Defaults: theta=1, k=1, R=1.5. Pre-registered configurations: 29. Cell value = $ expectancy per trade (trade count); a cell with fewer than 100 trades is insufficient and counts as failed; a cell not run is never a pass.

### One at a time

| Parameter | Value | Role | Trades | Expectancy $ | Expectancy R | Cell | Exit-type shares |
|---|---|---|---|---|---|---|---|
| theta | 0.8 |  | 156 | +16.59 | +0.006R | PASS | SL 50% TP 25% TE 25% |
| theta | 0.9 | neighbour | 130 | +1.08 | -0.040R | PASS | SL 50% TE 25% TP 25% |
| theta | 1 | default | 106 | +48.77 | +0.122R | PASS | SL 44% TP 28% TE 27% |
| theta | 1.1 | neighbour | 92 | +45.92 | +0.107R | insufficient | SL 43% TE 29% TP 27% |
| theta | 1.2 |  | 94 | +8.43 | +0.034R | insufficient | SL 47% TE 30% TP 23% |
| k | 0.75 |  | 84 | +26.43 | +0.038R | insufficient | SL 54% TP 33% TE 13% |
| k | 0.875 | neighbour | 95 | +52.28 | +0.123R | insufficient | SL 47% TP 34% TE 19% |
| k | 1 | default | 106 | +48.77 | +0.122R | PASS | SL 44% TP 28% TE 27% |
| k | 1.125 | neighbour | 118 | +19.34 | +0.042R | PASS | SL 43% TE 36% TP 21% |
| k | 1.25 |  | 126 | -0.87 | -0.004R | FAIL | SL 43% TE 39% TP 18% |
| R | 1 |  | 109 | +43.98 | +0.096R | PASS | TP 45% SL 41% TE 14% |
| R | 1.25 | neighbour | 106 | +50.39 | +0.119R | PASS | SL 44% TP 40% TE 16% |
| R | 1.5 | default | 106 | +48.77 | +0.122R | PASS | SL 44% TP 28% TE 27% |
| R | 1.75 | neighbour | 105 | +50.12 | +0.122R | PASS | SL 45% TE 33% TP 22% |
| R | 2 |  | 105 | +49.19 | +0.127R | PASS | SL 46% TE 35% TP 19% |

### Heatmap k x R

Rows R, columns k; the default cell is in bold. Exit-type shares of each cell are next to the expectancy grid (SL, TP, TE = time exit, GAP = data_gap_exit, END = end_of_data).

| R \ k | k=0.75 | k=0.875 | k=1 | k=1.125 | k=1.25 | exits k=0.75 | exits k=0.875 | exits k=1 | exits k=1.125 | exits k=1.25 |
|---|---|---|---|---|---|---|---|---|---|---|
| R=2 | insufficient (n=83) | insufficient (n=94) | +49.19 (n=105) | +10.40 (n=116) | +6.61 (n=125) | SL 57% TP 23% TE 20% | SL 49% TE 30% TP 21% | SL 46% TE 35% TP 19% | SL 45% TE 42% TP 13% | TE 45% SL 43% TP 12% |
| R=1.75 | insufficient (n=83) | insufficient (n=95) | +50.12 (n=105) | +13.54 (n=117) | +9.01 (n=125) | SL 57% TP 29% TE 14% | SL 47% TP 26% TE 26% | SL 45% TE 33% TP 22% | SL 44% TE 38% TP 18% | SL 43% TE 39% TP 18% |
| R=1.5 | insufficient (n=84) | insufficient (n=95) | **+48.77 (n=106)** | +19.34 (n=118) | -0.87 (n=126) | SL 54% TP 33% TE 13% | SL 47% TP 34% TE 19% | SL 44% TP 28% TE 27% | SL 43% TE 36% TP 21% | SL 43% TE 39% TP 18% |
| R=1.25 | insufficient (n=85) | insufficient (n=97) | +50.39 (n=106) | +19.84 (n=118) | +11.26 (n=127) | SL 54% TP 38% TE 8% | SL 45% TP 40% TE 14% | SL 44% TP 40% TE 16% | SL 42% TP 31% TE 27% | SL 42% TE 32% TP 26% |
| R=1 | insufficient (n=89) | insufficient (n=99) | +43.98 (n=109) | +26.39 (n=119) | +14.67 (n=127) | SL 52% TP 43% TE 6% | TP 46% SL 43% TE 10% | TP 45% SL 41% TE 14% | TP 40% SL 39% TE 20% | SL 41% TP 37% TE 22% |

Pre-registered cells: 29; run: 29; positive and sufficient: 16 (55.2% of 29); insufficient: 12; missing: 0. Immediate neighbours of the default: 4 of 6 pass. A flat response of a parameter that rarely binds (for example a TP that is rarely reached) is not robustness evidence: read the exit-type shares.

## Bootstrap (COMMON 10.5.6)

| Method | Total $ p5 | Total $ p50 | Total $ p95 | Max DD $ p5 | Max DD $ p50 | Max DD $ p95 | Settings |
|---|---|---|---|---|---|---|---|
| Stationary block bootstrap of the daily series | +457.24 | +5,112.43 | +10,041.10 | 844.71 | 1,502.01 | 2,851.12 | STRESS-d2, n_obs 579, 10000 resamples, seed 12345, mean block 5 days |
| Regime-conditional bootstrap of trades | +482.54 | +5,141.83 | +9,916.89 | 959.38 | 1,668.42 | 3,170.06 | STRESS-d2, n_obs 106, 10000 resamples, seed 12345 |

Regime-conditional bucket counts (kept fixed): ACTIVE 106.

## Shuffle test (COMMON 10.5.10, report only)

10000 permutations of trade order (seed 12345): 95th percentile of the worst losing streak 9.0 trades; of max drawdown 2,638.61 $.

## Benchmarks (COMMON 10.6; never selectable, not counted in N)

Benchmarks run under STRESS-d2.

| Run | Expectancy $ / trade | Total $ | Trades | Role |
|---|---|---|---|---|
| Strategy (run A, STRESS-d2) | +48.77 | +5,169.29 | 106 |  |
| B1 placebo (200 draws, seeds 1..200) |  | mean +5,361.76; p95 +9,200.49 |  | pass criterion 7 |
| B2 random index | mean +19.14 |  | 105 | report |
| B3 gate off | +38.82 | +6,210.54 | 160 | report |

The strategy beats B2's mean $ per trade.
The gated run beats B3 (gate off) in $ expectancy.

## Competition replay (COMMON 10.5.9)

30-calendar-day windows starting every Monday in the test span.

The competition contract (owner decision O4: USD per point per lot, commission, swap) is NOT confirmed; the replay below is at the demo contract only.

| Contract | USD per point per lot | Windows | 10% total-loss breaches | 5% daily-loss breaches | Personal 3% stop breaches | Windows with >= 5 trade days |
|---|---|---|---|---|---|---|
| demo | US30 1, NAS100 1, GER40 1.12 | 100 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 39.0% |

## Deflated Sharpe and PBO (COMMON 10.7)

| Trial count | N | V (variance of SR) | DSR | Threshold |
|---|---|---|---|---|
| N_raw (all configurations ever run) | 58 | 0.005044 | 0.0091 | >= 0.95 |
| N_eff (clusters at correlation 0.7) | 10 | 0.004752 | 0.1992 | >= 0.95 |

Trial log: 58 counted configurations (all strategies) at render time; N_raw used by the DSR: 58.

SR_hat (daily, non-annualised) 0.0767 over 579 days; skew 2.693, kurtosis (non-excess) 20.418.

PBO (CSCV, S = 16 blocks, 12870 splits, 58 trials): 0.067; threshold < 0.5.

## Regime diagnostics (COMMON 6.8)

Source: docs/feasibility_gate.md. Gate ACTIVE: blocks 57.8% of in-window cycles: **functional**.

| Index | Block share | 5-95% check |
|---|---|---|
| GER40 | 55.5% | functional |
| NAS100 | 62.2% | functional |
| US30 | 57.0% | functional |

Folds where every in-window cycle is blocked (regime UNAVAILABLE: no model or warm-up): 4, 7.

## Pass / fail (COMMON 10.4 minimums, 10.5 criteria 1-9)

| Criterion | Requirement | Measured | Threshold | Result |
|---|---|---|---|---|
| 10.4 trades | Filled out-of-sample trades, run A STRESS-d2 | 106 | >= 150 | **FAIL** |
| 10.4 folds | Folds with >= 4 trades per full fold (pro-rated by data share) | 7 of 9 | >= 7 of 9 | **PASS** |
| 1 | Run A expectancy > 0 in $ and R, daily Sharpe > 0, STRESS-d2 and STRESS-d3 | STRESS-d2: +48.77 $/trade, +0.122R, Sharpe 0.0767; STRESS-d3: +5.75 $/trade, -0.012R, Sharpe 0.0078 | > 0 each | **FAIL** |
| 2 | MISSED-EXIT expectancy > 0 in $ | +65.31 $/trade | > 0 | **PASS** |
| 3 | Run B out-of-sample expectancy > 0 in $ (STRESS-d2) | +35.25 $/trade (101 trades) | > 0 | **PASS** |
| 4 | Folds with positive $ P&L, run A STRESS-d2 | 4 of 9 | >= 6 of 9 | **FAIL** |
| 5 | Sensitivity: every immediate neighbour of each default has $ expectancy > 0 (>= 100 trades); >= 80% of all grid cells positive | neighbours 4 of 6 pass; cells 16 of 29 positive (55.2%), 12 insufficient | all neighbours; >= 80% of 29 | **FAIL** |
| 6 | Stationary block bootstrap of the run A STRESS-d2 daily series (10,000 resamples, mean block 5, seed 12345): 5th percentile of total $ P&L > 0 | +457.24 | > 0 | **PASS** |
| 7 | Run A total $ P&L > 95th percentile of placebo B1 (STRESS-d2, 200 draws, seeds 1..200) | +5,169.29 vs B1 p95 +9,200.49 | > B1 p95 | **FAIL** |
| 8 | DSR >= 0.95 under N_raw and N_eff; PBO < 0.5 | DSR 0.009 (N_raw 58; trial log 58), 0.199 (N_eff 10); PBO 0.067 | DSR >= 0.95; PBO < 0.5 | **FAIL** |
| 9 | Competition replay at the demo AND the competition contract | demo: 0 total-loss windows, daily-loss 0.00% of 100; competition: contract NOT confirmed (O4) | 0 total-loss windows; daily-loss <= 1% of windows | **NOT EVALUATED** |
| 10 (report) | Shuffle test: p95 worst losing streak and max DD | streak 9.0, DD 2,638.61 | report only | **INFO** |
| 10.4 (report) | Trade days: share of rolling 20-trading-day windows with >= 5 trade days (applies to the combined set) | 31.2% | >= 80% (combined set) | **INFO** |

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
