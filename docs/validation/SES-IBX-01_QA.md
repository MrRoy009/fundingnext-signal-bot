# SES-IBX-01: QA/Risk sign-off for GATE 3

- Date: 2026-11-07
- Role: QA/Risk (veto)
- Baseline commit: 734cc50 (working tree dirty; validation tooling uncommitted)
- Report reviewed: `docs/validation/SES-IBX-01.md` (generated 2026-11-07 06:12 UTC, overall verdict FAIL)
- **Verdict: REJECT. SES-IBX-01 does not go to the final holdout.**

This file replaces the earlier QA note, which was written before the SES-IBX-01 run phase existed.

## Summary

SES-IBX-01 now has a complete run phase: run A T0034, sensitivity T0035-T0062, run B T0063 and
benchmarks T0064-T0066. QA started from the assumption that the strategy is overfit. The evidence
does more than fail to clear that bar: the default loses money in every pass/fail scenario, in every
fold that has trades, in all 29 grid cells, out of sample in run B, and against its own timing placebo.

QA recomputed every number it relied on from the saved trades and daily files, with its own code where
practical. All of them match the report. **Eight of the nine 10.5 criteria fail, criterion 9 is NOT
EVALUATED, and the 10.4 total-trade minimum fails.** Only the 10.4 fold minimum passes (7 of 9). PBO
passes on its own, but criterion 8 also needs the DSR, and the DSR fails.

The integrity checks raise no concerns. Every run on disk has a log row. No bar on or after 2026-05-01
was used. The holdout access log does not exist. Strategy code and parameters were not changed.

## Independent recomputation

Script: `C:\Users\lunar\AppData\Local\Temp\claude\0e1d0cbf78eabeaa\6c140617-18ea-4b43-9cc9-6862d0b2cb3e\scratchpad\qa_ses_ibx_recompute.py`.
It reads only the saved files under `backtest/runs/SES-IBX-01/trials/` and `backtest/runs/global/`.
It runs no configuration and logs nothing.

### Run A scenarios (T0034, filled trades, 579-day MTM daily series)

| Scenario | Trades | E $ | E R | Daily Sharpe | Total $ | Report |
|---|---|---|---|---|---|---|
| STRESS-d2 | 90 | -70.70 | -0.229 | -0.0939 | -6,363.09 | match |
| STRESS-d3 | 63 | -82.20 | -0.252 | -0.0969 | -5,178.83 | match |
| MISSED-EXIT | 90 | -76.79 | -0.242 | -0.0942 | -6,911.36 | match |
| BASE (report) | 90 | -59.51 | -0.179 | -0.0803 | -5,355.50 | match |
| STRESS-d1 (report) | 113 | -46.57 | -0.131 | -0.0650 | -5,262.13 | match |
| EXIT-3X (report) | 90 | -70.85 | -0.229 | -0.0941 | -6,376.63 | match |

- Each scenario's trade P&L sums exactly to its daily P&L sum.
- The canonical `T0034/daily.parquet` equals the STRESS-d2 scenario daily series.
- In the STRESS-d2 run, 268 candidates split into 90 filled, 39 skipped and 139 cancelled.
- MISSED-EXIT has no time exits (SL 63, TP 27). 7 trades carried rollovers.
- MISSED-EXIT swap is a cost: pnl = gross - commission - swap holds exactly. With swap set to zero,
  the expectancy would still be -75.97 $. Criterion 2 does not depend on the A4 swap conversion.

### Folds (run A, STRESS-d2)

- Fold assignment was recomputed from the signal time: 0 mismatches against the stored `fold` column.
- Positive folds: **0 of 9**. Fold P&L in $:
  - Fold 1: -940.23
  - Fold 2: -317.61
  - Fold 3: -1,127.78
  - Fold 4: 0 (no trades)
  - Fold 5: -337.09
  - Fold 6: -665.24
  - Fold 7: 0 (no trades)
  - Fold 8: -1,320.08
  - Fold 9: -1,655.06
- 10.4 fold minimum (4 trades x data share, A1): met in 7 of 9 folds. Folds 4 and 7 have 0 trades
  against minimums of 1.14 and 2.36. This item **passes**.
- 10.4 total trades: **90 < 150. FAIL.**

### Run B (T0063)

- 81 out-of-sample trades, -46.18 $/trade, total -3,740.58. These match the trial-log row and the
  T0063 STRESS-d2 scenario file.
- The selected max_ib_datr changes from fold to fold (0.55, 0.7, 0.5). Every selection is a
  one-at-a-time grid cell, with its train Sharpe in `driver.log`.

### Sensitivity (T0034-T0062, STRESS-d2)

- All 29 pre-registered cells were run, and their parameter vectors match the spec grid exactly
  (13 one-at-a-time + 16 heatmap).
- Expectancy is negative in **29 of 29** cells, whatever the trade count. Range: -99.60 to -27.17 $.
- 17 cells have fewer than 100 trades.
- 0 of the 6 immediate neighbours pass.
- The least-bad cell (ib_minutes=45, tp_ib_mult=1.5) is still -27.17 $/trade. No part of the grid is
  positive.

### Bootstrap (criterion 6)

- `validation.stats.stationary_bootstrap` on the saved STRESS-d2 daily series reproduces
  `stats.json` exactly: p5 -10,826.69, p50 -6,312.10, p95 -2,049.91.
- QA also ran its own Politis-Romano implementation (seed 12345, mean block 5, 10,000 resamples,
  different random stream). It gave p5 -10,723.86, p50 -6,292.94, p95 -1,972.47.
- Even the 95th percentile is below zero.

### Placebo B1 (criterion 7; A4.2 shifted-bracket geometry)

- 200 draws, seeds 1..200, all unique, max unmatched 0.
- Placebo p95 is -1,717.53 with linear interpolation, the same as `benchmarks.json`. It ranges from
  -1,725.95 to -1,557.68 across quantile methods.
- Run A total is -6,363.09, below the p95 under every method.
- 86% of placebo draws did at least as well as run A. The IB-break timing is worse than random timing
  within the same window.

### DSR and PBO (criterion 8)

**Global trial matrix**
- QA rebuilt the 579 x 58 matrix from each trial's `daily.parquet` on disk. It equals
  `backtest/runs/global/trial_matrix.parquet`.
- Its T0034 column equals the run-A STRESS-d2 daily series.

**Clustering (own scipy code)**
- Average linkage on 1 - correlation, cut at correlation 0.7.
- N_eff = 10. The medoids are the same as in `global_stats.json`. No cluster mixes the two strategies.
- V_raw = 0.005044 (ddof 1), V_eff = 0.004752.

**DSR (own code, formula from COMMON 10.7)**
- Inputs: SR_hat -0.0939, skew -0.651, non-excess kurtosis 15.85, T 579.
- SR* = 0.1657 under N_raw = 58 and 0.1085 under N_eff = 10.
- DSR = 1.2e-10 (N_raw) and 3.9e-7 (N_eff). Both must be >= 0.95. **FAIL.**
- For reference, the PSR against zero is only 0.011.

**PBO (own CSCV code)**
- S = 16, the earliest 3 days dropped, 12,870 splits, rank/(N+1) logit.
- PBO = 0.0668, the same as the global step. That is < 0.5 and below the 0.2 flag.
- It is a global number: the in-sample best is an RS-MOM-01 trial in 96% of splits. It says nothing
  in SES-IBX-01's favour.

## Criterion table

Every value in this table was independently verified by QA, except where a note says otherwise.

| Item | Required | Measured | Result |
|---|---|---|---|
| 10.4 total trades (run A, STRESS-d2) | >= 150 | 90 | **FAIL** |
| 10.4 fold minimum (pro-rated) | >= 7 of 9 folds | 7 of 9 | PASS |
| 10.4 sensitivity cell counts | >= 100 trades per cell | 17 of 29 cells insufficient | **FAIL** (counted under criterion 5) |
| 10.4 competition feasibility | >= 80% of windows (combined set) | 30.4% | INFO (applies to the combined set) |
| 10.5-1 | E $ > 0, E R > 0, Sharpe > 0, STRESS-d2 and STRESS-d3 | d2: -70.70 / -0.229R / -0.0939; d3: -82.20 / -0.252R / -0.0969 | **FAIL** |
| 10.5-2 | MISSED-EXIT E $ > 0 | -76.79 | **FAIL** |
| 10.5-3 | Run B OOS E $ > 0 | -46.18 (81 trades) | **FAIL** |
| 10.5-4 | >= 6 of 9 positive folds | 0 of 9 | **FAIL** |
| 10.5-5 | All neighbours > 0; >= 80% of cells > 0 | 0 of 6 neighbours; 0 of 29 cells | **FAIL** |
| 10.5-6 | Bootstrap total $ p5 > 0 | -10,826.69 | **FAIL** |
| 10.5-7 | Run A total > B1 p95 | -6,363.09 vs -1,717.53 | **FAIL** |
| 10.5-8 | DSR >= 0.95 (N_raw and N_eff); PBO < 0.5 | DSR 1.2e-10 / 3.9e-7; PBO 0.067 | **FAIL** |
| 10.5-9 | Replay at demo AND competition contract | demo 0 / 100 total-loss, 0% daily-loss; competition contract not confirmed (O4) | **NOT EVALUATED** (counts as not passed) |
| 10.5-10 | Shuffle (report only) | p95 streak 15, DD 8,206.65 (from stats.json, not recomputed) | INFO |

## Integrity checks

1. **Every run on disk has a log row.**
   - `docs/trial_log.csv` has 66 unique rows: RS-MOM-01 T0001-T0033 and SES-IBX-01 T0034-T0066.
     There are no MR-TWAP-01 rows.
   - The run directories on disk are the same 66 IDs, with no orphan in either direction.
   - Every SES-IBX-01 row's `run_files` points to its own directory.
   - All 33 SES-IBX-01 rows have config hash 9fd0c89a54d21899 (A4 key present).
   - The only non-T directories are `artifacts`, `staging` and `global`.
   - `driver.log` shows exactly one run A, 29 sensitivity cells (the default merged into T0034 as
     stage `sensitivity`), 9 run-B fold selections and 3 benchmarks.
2. **The aborted 10:16 run was not a hidden trial.**
   - It ran under config hash ee55bca19d48a871 and crashed in `run_a` (the secondary-sample call) on
     USTEC swap_mode 3.
   - It wrote no trial and no log row, and its log contains no outcome numbers.
   - This fits A4's statement that the swap ruling was made before any SES-IBX-01 outcome existed.
   - The rerun was a code/data-handling fix (A4.1), not a change to any configuration.
3. **No bar on or after 2026-05-01 was used.**
   - No trade file under `trials/` has a signal, fill or exit time >= 2026-05-01 UTC. The last exit is
     2026-04-08.
   - No daily file has a date >= 2026-05-01.
   - The last equity grid point, 2026-05-01 00:00 UTC, is the close time of the 2026-04-30 23:45 bar.
     The grid is `close_ns` in `backtest/engine.py`, and equity there equals the 23:45 point.
   - The store holds bars up to 2026-11-02, but `BarStore.load` cuts at `holdout.start` unless
     `final_holdout=True`.
   - The driver and provenance logs say "BarStore.load pre-holdout". No non-test `.py` file passes
     `final_holdout=True`.
4. **No holdout access log.** `data/store/holdout_access.log` does not exist.
5. **Nothing was tuned.** The baseline is 734cc50.
   - `strategies/`, `features/`, `regime/` and `data/` have no diff and no untracked files.
   - `docs/specs/` has one diff: `COMMON.md` +16 lines, which is section 12D (amendment A4) only.
   - `config.yaml` has one diff: +1 line, the A4 key `costs.margin_ccy_usd`.
   - Run A uses the registered defaults (ib_minutes 60, tp_ib_mult 1.0, max_ib_datr 0.6).
6. **Tooling check.** `tests/test_validation_stats.py` and `tests/test_report.py` pass: 112 passed.

## Observations (none changes the verdict)

- **B2 and B3.** B2 random index (-31.55 $/trade) and B3 gate off (-52.10 $/trade) both beat the
  strategy. The report correctly states that the relative-strength selection and the regime gate add
  nothing.
- **Per-index and per-side buckets.** Every bucket loses: GER40 -2,914.86 and OTHER -3,448.23; BUY
  -2,330.51 and SELL -4,032.59. There is no sub-population to rescue, and QA would not accept a
  post-hoc subset anyway.
- **R:R versus win rate.** The default bracket's nominal R:R is 1.73 and the win rate is 31.1%. That
  is below the roughly 37% break-even that R:R implies, before costs.

## Decision

REJECT. SES-IBX-01 fails the 10.4 trade minimum and criteria 1, 2, 3, 4, 5, 6, 7 and 8, and criterion 9
is not evaluated. Under QA rules each one alone blocks approval. The final holdout stays closed for
SES-IBX-01.

The configuration must not be re-tuned or re-run in the hope of a better result. Any successor would
be a new spec with new trials, counted in N_raw.
