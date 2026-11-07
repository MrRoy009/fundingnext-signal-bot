# RS-MOM-01: QA/Risk sign-off for GATE 3 (final)

**Verdict: REJECT. Do not open the holdout for RS-MOM-01.**

- Date: 2026-11-07. Reviewer: QA/Risk (veto). This file replaces the earlier provisional QA note.
- Baseline commit: 734cc50. Reviewed artifacts:
  - `docs/validation/RS-MOM-01.md` (generated 2026-11-07 06:12 UTC, verdict FAIL)
  - `backtest/runs/RS-MOM-01/` (T0001-T0033 plus `artifacts/`)
  - `backtest/runs/global/` (global_stats.json, trial_matrix.parquet, clusters.csv, pbo_logits.csv)
  - `docs/trial_log.csv` (66 rows, sha256 375e778206cffb25...)
- Starting assumption: the strategy is overfit until the evidence shows otherwise. The evidence does not show otherwise.
- QA recomputed every number below from the saved `trades.parquet`, `daily.parquet`, `equity.parquet`, `b1_placebo_draws.csv` and per-trial daily files. The DSR, N_eff clustering, CSCV PBO and stationary bootstrap use QA's own code. Report numbers are only used for comparison. QA ran no backtest, added no configuration, logged nothing and never passed `final_holdout`.
- Recompute script: `C:\Users\lunar\AppData\Local\Temp\claude\0e1d0cbf78eabeaa\6c140617-18ea-4b43-9cc9-6862d0b2cb3e\scratchpad\qa2_rsmom01_gate3_signoff.py`

**Summary.**
- The report and QA agree on every number.
- RS-MOM-01 fails:
  - criteria 1 (STRESS-d3), 4, 5, 7 and 8;
  - the 10.4 minimums for total trades and sensitivity cells.
- Criterion 9 cannot be evaluated because the competition contract (O4) is not confirmed.
- These are outcome failures of the pre-registered configuration. No rerun or code fix can change them.

## COMMON 10.5 pass criteria

| # | Criterion | QA recomputed value | Report | Threshold | Result |
|---|---|---|---|---|---|
| 1 | Run A STRESS-d2 (T0001) | n 106; E$ +48.77; E(R) +0.1225 (mean `r_multiple` = mean pnl / planned risk); daily Sharpe 0.0767 on $ (0.0766 on returns); 579 days; trade total = daily total = +5,169.29 | same | > 0 in $, R, Sharpe | pass |
| 1 | Run A STRESS-d3 | n 91; E$ +5.75; **E(R) -0.0119**; daily Sharpe 0.0078 on $ (0.0083 on returns); total +522.89 | same | > 0 in $, R, Sharpe | **FAIL** (R < 0) |
| 2 | MISSED-EXIT (T0001 scenario, A4.3) | n 106; **E$ +65.31**; E(R) +0.153; Sharpe 0.0818; total +6,922.39. Exits: SL 59, TP 47, no time exits. Longest hold 73.5 h. Swap: 16 rollovers, total $176.30 (QA hand-check below) | same | > 0 | pass |
| 3 | Run B out-of-sample (T0030) | n 101; **E$ +35.25**; E(R) +0.0739; total +3,560.05. QA re-derived every fold's choice from the logged train grid: highest train Sharpe, ties to the closest. All 9 match; 13 candidates per fold; 0 embargoed trades | same | > 0 | pass |
| 4 | Positive folds, run A STRESS-d2 | **4 of 9** positive: 2 (+427.99), 5 (+2,762.39), 8 (+1,740.20), 9 (+1,301.28). Negative: 1 (-54.47), 3 (-787.83), 6 (-220.27). No trades: 4 and 7. QA's own fold assignment from `signal_time` against the 10.2 windows equals the `fold` column | same | >= 6 of 9 | **FAIL** |
| 5 | Every immediate neighbour > 0 with >= 100 trades | **4 of 6**. theta 1.1 (n 92) and k 0.875 (n 95) are insufficient, which counts as a fail. theta 0.9 passes on E$ +1.08 only; its E(R) is -0.040 | same | 6 of 6 | **FAIL** |
| 5 | Grid cells positive and sufficient | **16 of 29 = 55.2%**: 12 insufficient (every k 0.75 and k 0.875 cell, plus theta 1.1 and 1.2), 1 negative (k 1.25 / R 1.5, E$ -0.87) | same | >= 80% | **FAIL** |
| 6 | Stationary bootstrap, p5 of total $ | `validation.stats`: p5 / p50 / p95 = **+457.24** / 5,112.43 / 10,041.10. QA's own Politis-Romano code with seed 12345: +517.50 / 5,101.73 / 10,044.72. Seeds 1-5: p5 from +351.64 to +521.68 | same | p5 > 0 | pass, thin: p5 is 7-10% of the observed total |
| 7 | Placebo B1 (200 draws, seeds 1..200, 0 unmatched) | Run A +5,169.29. Placebo p95 **+9,200.49** (linear; +9,190.94 nearest, +9,381.91 higher). Placebo mean +5,361.76, median +5,298.31. **53.5% of draws make at least as much as run A**: run A sits at the 46.5th percentile | same | run A > p95 | **FAIL** |
| 8 | DSR, both N; PBO | See the next section: **DSR 0.0091 (N_raw 58), 0.1992 (N_eff 10)**; PBO 0.0668 | same | DSR >= 0.95; PBO < 0.5 | **FAIL** (DSR) |
| 9 | Competition replay | Demo contract: 0 total-loss breaches and 0 daily-loss breaches. stats.json lists 113 windows, 100 of them with data; the report table shows the 100. Competition contract: **not run**, because `competition_usd_per_point` is null (O4) | same | both contracts | **NOT EVALUATED**, which cannot pass |
| 10 | Shuffle (report only) | Worst-streak p95 9, max-drawdown p95 $2,638.61 (artifact; the report says they equal validation.stats) | same | report | reported |

## COMMON 10.4 minimums

| Minimum | QA value | Threshold | Result |
|---|---|---|---|
| Filled out-of-sample trades, run A STRESS-d2 | **106** | >= 150 | **FAIL** |
| Folds meeting 4 x data share | 7 of 9. Fold 4: 0 trades vs a minimum of 1.14. Fold 7: 0 vs 2.36. Shares come from the T0001 notes | >= 7 of 9 | pass, exactly at the bar |
| Every sensitivity cell >= 100 trades | 17 of 29; 12 insufficient | every cell | **FAIL** |
| Competition feasibility (>= 5 trade days in >= 80% of rolling 20-day windows) | 31.2% (35.4% over windows with data). The rule applies to the approved combined set; if RS-MOM-01 were the only approved strategy, it would fail | >= 80% | report only; would fail |
| Bucket rule (>= 30 trades) | NAS100 (19) and US30 (22) merged into OTHER; time_exit (29) withheld | rule | applied correctly |

## DSR and PBO inputs (COMMON 10.7)

QA rebuilt the global matrix and re-derived each input from it.

- **Global matrix.** QA rebuilt it from the 58 counted trials' canonical `daily.parquet` files:
  - 579 days (2024-02-01..2026-04-30) by 58 trials, no NaN;
  - RS-MOM-01 T0001-T0029, SES-IBX-01 T0034-T0062;
  - **max |difference| from `trial_matrix.parquet` = 0.0**.
- **N_raw = 58.** These are the counted rows (stage run_a or sensitivity): 29 per strategy, no duplicate parameter vectors. Run B and benchmarks are excluded, as 10.6/10.7 require.
- **N_eff = 10.** QA used scipy average linkage on distance 1 - correlation, cut at 0.3.
  - The medoids are identical to global_stats.json: T0004, T0005, T0002, T0008, T0007, T0036, T0035, T0038, T0037, T0034.
  - No cluster mixes the two strategies.
- **Variance of SR.**
  - V_raw = 0.005044 (sample variance; 0.004957 with ddof 0).
  - V_eff = 0.004752.
- **DSR inputs for T0001.** SR_hat = 0.07669, skew 2.693, kurtosis (non-excess) 20.418, T = 579.
  - SR* = 0.16567 under N_raw and 0.10854 under N_eff.
  - **DSR = 0.00915 (N_raw) and 0.19917 (N_eff).** Both match global_stats.json.
  - PSR against 0 is 0.979.
  - Reaching DSR 0.95 would take a daily SR_hat of 0.2205 (N_raw) or 0.1653 (N_eff). The observed 0.0767 is less than half of either.
- **PBO.** QA ran its own CSCV:
  - S = 16, 3 earliest days dropped, block length 36, all 12,870 splits;
  - **PBO = 0.0668**, equal to the global figure, so it passes (< 0.2, not flagged).
  - Context only, not a criterion: the same CSCV over the 29 RS-MOM-01 trials alone gives 0.318, inside the 0.2-0.5 flag band. The low global PBO mostly reflects that the RS-MOM-01 trials dominate SES-IBX-01, which lost money in every cell. It is not evidence that the RS-MOM-01 default survives selection out of sample.

## MISSED-EXIT checks (A4.1 / A4.3)

- **Run under the A4 key only.** The scenario ran under config hash 9fd0c89a54d21899. Removing `costs.margin_ccy_usd` gives T0001's hash, ee55bca19d48a871, with the same data and shared hashes (`staging/missed_exit/stage_info.json`, `stage.log`).
- **Logged on T0001.** It is recorded in T0001's `merged` notes as a scenario, not a trial, and N_raw is unchanged.
- **Same trades as STRESS-d2.** The 106 entry ids and every fill time, fill price and stop are identical. Lots differ on 1 trade and sizing balance on 105, which is expected from compounding.
- **Swap amounts, hand-checked** as lots x |demo swap| x 2 (stress) x margin factor, tripled on the Friday rollover:
  - NAS100 BUY 3 lots: 0.82 x 3 x 2 = 4.92, and 14.76 tripled.
  - US30 BUY 3 lots: 2.21 x 3 x 2 = 13.26.
  - US30 SELL 3 lots on a Friday: 1.99 x 3 x 2 x 3 = 35.82.
  - GER40 SELL 3 lots: 1.12 x 3 x 2 x 1.12 = 7.5264.
  - GER40 BUY 3 lots: 0.7 x 3 x 2 x 1.12 = 4.704, and 14.112 tripled.
- **Accounting identity holds exactly.** `lots x (exit - fill) x side x USD/pt - commission - swap` matches `pnl_usd` with zero difference in STRESS-d2, STRESS-d3 and MISSED-EXIT.

## Integrity checks

- **Trial log: OK.**
  - 66 rows, T0001-T0066 contiguous, no MR-TWAP-01 rows.
  - Every T directory on disk has a row and every row has a directory. RS-MOM-01 has 33 rows (T0001-T0033); SES-IBX-01 has 33 (T0034-T0066).
  - Per strategy: run_a 1, sensitivity 28, run_b 1, benchmark 3.
  - The log's sha256 equals the value recorded by the global step (375e7782...): nothing was appended after it.
- **Holdout: OK.**
  - `data/store/holdout_access.log` does not exist.
  - QA scanned all 376 parquet files under `backtest/runs/`. No trades file has any timestamp at or after 2026-05-01.
  - The only such timestamps are single `equity.parquet` marks at exactly 2026-05-01 00:00:00 UTC. Equity is stamped at bar close (the first mark is 00:15), so each is the close of the 2026-04-30 23:45 bar, and balance and equity are unchanged at that mark.
  - `market.json` m15_last is 2026-04-30 23:45 (US) and 19:45 (GER40).
- **No tuning: OK.**
  - `git diff 734cc50` is empty for `strategies/`, `features/`, `regime/`, `data/` and `risk/`.
  - `docs/specs/` changes only by the appended amendment 12D (A4).
  - `config.yaml` changes only by the A4 key `costs.margin_ccy_usd: {US30: 1.0, NAS100: 1.0, GER40: 1.12}`.
  - Engine changes (`backtest/costs.py`, `account.py`, `engine.py`) are limited to passing that factor into swap mode 3. Swap affects only MISSED-EXIT for this intraday strategy, and the T0001 note records that STRESS-d2 reproduced exactly under the new code.
  - Run A uses the registered defaults theta 1.0, k 1.0, R 1.5.
  - No MT5 order-function names appear in any `.py` file outside `.venv`.

## Evidence that there is no real edge (beyond the failed criteria)

- **No timing edge.** Random signal times on the same date, index and side make as much as the strategy: run A sits at the 46th percentile of the placebo. The P&L comes from date, side and bracket geometry, not from the RS_4h trigger.
- **Fragile to delay.** One more bar of manual delay (STRESS-d3) cuts E$ from 48.8 to 5.7 and makes E(R) negative. With one bar less (STRESS-d1), E$ falls to 29.2.
- **Concentration.**
  - GER40 makes +$5,410 on 65 trades; US30 and NAS100 lose $241 on 41.
  - Mean R is +0.237 on SELL and +0.028 on BUY.
  - Daily returns have kurtosis 20.4.
- **Weak multiple-testing evidence.** DSR is 0.009 / 0.199. Even PSR against zero (0.979) assumes a single trial, and 58 were run.
- **The criteria that pass are thin.**
  - Bootstrap p5 is under 10% of the total.
  - The fold minimum passes exactly at 7 of 9.
  - The MISSED-EXIT and run B passes rest on the same 100-odd trades that fail the placebo test.

## Minor findings for the owners (no effect on the verdict)

- The report's replay table shows "Windows 100", while stats.json lists 113 windows, 100 of them with data. The report should say the 13 no-data windows are excluded.
- The 10.4 competition-feasibility rule awaits a Quant ruling on whether windows over the data hole count (31.2% literal, 35.4% with data). Both are far below 80%.

## Decision

**REJECT.** Failed:
- 10.4: total trades and sensitivity cells.
- 10.5: criteria 1 (STRESS-d3 E(R)), 4, 5, 7 and 8 (DSR).
- Criterion 9: not evaluated.

These are outcome failures of the pre-registered configuration and cannot be fixed by reruns or tuning. Standards are not lowered. Per A3 and PROMPT, RS-MOM-01 may come back only as a new spec with a new trial budget, and the holdout stays closed.
