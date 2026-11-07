# Regime diagnostics (COMMON 6.8, amendment A2)

Generated 2026-11-03 04:39 UTC by the regime batch path (`RegimeService.run_batch`) on the pre-holdout demo bars (`BarStore.load`, which drops the holdout and the excluded window of amendment A1). Config `config.yaml` as frozen at GATE 2: K in {2, 3, 4}, seeds 0..9, seed-stability seeds 10..19, 365-day window, first-Saturday 12:00 UTC refits. Revised after QA the same day: the warm-up table gains the reset slot and the first emitting observation (where the warm-up ends), and the reproduction note now describes the refit log and model cache correctly; no number in the tables changed.

These numbers describe the regime model itself (states, labels, the gate variable V_T and its agreement with simpler quantities). No price after a decision is read and no return, P&L, win rate or other outcome of any rule is computed. Nothing in the regime spec changes in response to them (COMMON 6.8).

Amendment A2 (COMMON 12B) is in force: after a segment reset, every regime output is UNAVAILABLE ("regime warm-up") until the first core observation with an emission (x1 and x2 finite). Warm-up observations are excluded from every share below and counted separately.

## Summary per index

| Index | H1 bars | Data (UTC) | Refits | OK | Insufficient | Failed | First valid model | K chosen (count) | Warm-up rows / model rows | Warm-up share of 15-min times with a model |
|---|---|---|---|---|---|---|---|---|---|---|
| US30 | 23,896 | 2021-11-03 .. 2026-04-30 | 54 | 37 | 17 | 0 | 2023-01-07 12:00 | K=4: 37 | 127 / 4,640 (2.74%) | 2.50% |
| NAS100 | 19,960 | 2022-07-06 .. 2026-04-30 | 45 | 35 | 10 | 0 | 2023-03-04 12:00 | K=4: 35 | 127 / 4,361 (2.91%) | 2.64% |
| GER40 | 22,871 | 2021-11-03 .. 2026-04-30 | 54 | 42 | 12 | 0 | 2022-10-01 12:00 | K=4: 42 | 305 / 6,283 (4.85%) | 4.08% |

Medians over refits that served at least one core observation with a regime output:

| Index | share ACTIVE (V_T >= 0) | share V_T in [-0.05, 0.05] | agreement gate vs sign(x2) | label churn | seed-stability share |
|---|---|---|---|---|---|
| US30 | 47.9% | 9.1% | 92.9% | 5.0% | 100.0% |
| NAS100 | 44.6% | 4.2% | 95.2% | 6.1% | 100.0% |
| GER40 | 55.6% | 21.2% | 90.1% | 1.3% | 100.0% |

Pooled over all served core observations with a regime output (weighted by observations):

| Index | Obs | share ACTIVE | share V_T in [-0.05, 0.05] | agreement with sign(x2) | label churn | seed-stability share | min seed-stability (refit) |
|---|---|---|---|---|---|---|---|
| US30 | 4,477 | 48.4% | 18.4% | 91.8% | 13.7% | 99.4% | 82.0% (2026-03-07) |
| NAS100 | 4,200 | 46.8% | 6.3% | 93.8% | 16.2% | 98.5% | 82.7% (2024-03-02) |
| GER40 | 5,937 | 51.0% | 20.4% | 91.0% | 9.0% | 99.9% | 97.5% (2022-11-05) |

## Runtime

Windows 11, 12 logical cores, Python 3.12. "Cold" = empty model directory, the 3 x 10 restarts per refit run in 12 joblib processes (`n_jobs=-1`; each restart single-threaded for reproducible hashes). "Cached" = a new service reloading the models persisted by the cold run (a scratch model directory, not `regime/models/`), `n_jobs=1`; its table is compared with the cold one. Diagnostics include the seed-stability refits (seeds 10..19, also cold) and the churn re-filters.

| Index | Load H1+M15 | Batch, cold | Diagnostics, cold | Batch, cached | One live snapshot, cached | Cached table identical | Live == batch at last obs |
|---|---|---|---|---|---|---|---|
| US30 | 1.4 s | 124 s | 128 s | 14.6 s | 3.3 s | True | True |
| NAS100 | 0.1 s | 110 s | 101 s | 10.8 s | 2.2 s | True | True |
| GER40 | 0.0 s | 133 s | 143 s | 12.2 s | 2.3 s | True | True |

## Warm-up stretches (A2)

A warm-up starts at the reset slot (the first core slot of a new segment after a gap of more than 7 days: the excluded window of A1 and the 2025 data hole) and lasts until the first core observation with an emission (x1 and x2 finite), which is the end that matters for COMMON 10.2. The warm-up itself depends only on the data, not on the model; the table rows exist only where a model serves the stretch. All times are H1 close times in UTC; times were recomputed independently from `build_slots` (the reset slot from the segment ids, the first emission from the slot kinds).

| Index | Segment | Reset slot | Reset slot usable | First warm-up obs | Last warm-up row | Warm-up rows | First emitting obs (warm-up ends) | First OK row after |
|---|---|---|---|---|---|---|---|---|
| US30 | 1 | 2025-01-08 15:00 | yes | 2025-01-08 15:00 | 2025-02-04 21:00 | 127 | 2025-02-05 15:00 | 2025-02-05 15:00 |
| US30 | 2 | 2025-09-09 14:00 | yes | 2025-09-09 14:00 | none (no model) | 0 (a) | 2025-10-07 15:00 | 2025-12-06 12:00 refit (a) |
| NAS100 | 1 | 2025-01-08 15:00 | yes | 2025-01-08 15:00 | 2025-02-04 21:00 | 127 | 2025-02-05 15:00 | 2025-02-05 15:00 |
| NAS100 | 2 | 2025-09-09 14:00 | yes | 2025-09-09 14:00 | none (no model) | 0 (a) | 2025-10-07 15:00 | 2025-12-06 12:00 refit (a) |
| GER40 | 1 | 2025-01-08 09:00 | yes | 2025-01-08 09:00 | 2025-02-04 16:00 | 161 | 2025-02-05 09:00 | 2025-02-05 09:00 |
| GER40 | 2 | 2025-09-09 13:00 | no (b) | 2025-09-10 08:00 | 2025-10-03 15:00 (c) | 144 | 2025-10-08 08:00 | 2025-11-01 12:00 refit (c) |

Rows include the refit row when a refit falls inside a stretch (one extra row: segment 1 has 126 US or 160 GER40 warm-up observations).

- (a) US30 and NAS100: the 2025-09-06, 2025-10-04 and 2025-11-01 refits have fewer than 1,000 training rows, so no model serves the post-hole warm-up (2025-09-09 14:00 to 2025-10-07 15:00, 134 core observations). Those times are UNAVAILABLE with the "fewer than 1,000 training rows" cause, which hides the warm-up; the first model after the hole is the 2025-12-06 refit.
- (b) Data resume 2025-09-09 11:30 UTC, after the GER40 cash-open M15 bar (07:00 UTC), so 2025-09-09 is not a valid GER40 date (COMMON 1.6) and its reset slot is not a core observation. The warm-up starts at that slot; until the first usable observation (2025-09-10 08:00) the latest output row is the pre-gap one (2025-07-16 08:00), so those times are stale. The US cash-open bar (13:30 UTC) exists, so the US30 and NAS100 reset slots are usable.
- (c) GER40 segment 2: the warm-up runs to 2025-10-08 08:00, but the warm-up rows end at 2025-10-03 15:00 because the 2025-09-06 model's interval ends at the 2025-10-04 refit, which had too few training rows (no model). From 2025-10-04 12:00 to the 2025-11-01 refit the output is "fewer than 1,000 training rows", so the first OK row after the stretch is the 2025-11-01 refit row (its alpha is that of the 2025-10-31 16:00 observation), not the end of the warm-up.
- For a fold check against COMMON 10.2 ("regime warm-up to about 2025-10-07"): the post-hole warm-up ends at 2025-10-07 15:00 (US30, NAS100) and 2025-10-08 08:00 (GER40), but no regime output is OK on any index before the 2025-11-01 (GER40) or 2025-12-06 (US30, NAS100) refits.

Status of the regime output at every 15-minute time inside the data span (descriptive; "fewer than 1,000 training rows" pools the per-refit causes):

| Cause | US30 | NAS100 | GER40 |
|---|---|---|---|
| OK | 91,296 (58.0%) | 85,923 (64.2%) | 104,029 (66.1%) |
| fewer than 1,000 training rows | 49,728 (31.6%) | 28,896 (21.6%) | 34,272 (21.8%) |
| no regime model yet | 292 (0.2%) | 2,988 (2.2%) | 292 (0.2%) |
| regime H1 bar missing | 3 (0.0%) | 0 (0.0%) | 7 (0.0%) |
| regime warm-up | 2,688 (1.7%) | 2,688 (2.0%) | 5,008 (3.2%) |
| stale: last core H1 observation more than 5 days before T | 13,386 (8.5%) | 13,386 (10.0%) | 13,769 (8.7%) |

## Per refit

Columns: K and the state labels of the model; `obs` = core observations served with a regime output; `warm` = served core observations in warm-up (excluded); label distribution = share of `obs` by argmax label; ACTIVE = share with V_T >= 0; abs V <= 0.05 = share with V_T in [-0.05, 0.05]; sign(x2) = agreement of the gate with x2 >= 0; churn = share of `obs` whose argmax label differs from the previous model's (its filter continued over the same observations); seed = share of `obs` where the seeds-10..19 model gives the same gate (its K in brackets). Refits without a model are listed under each table.

### US30

| Refit (UTC) | K | Labels | obs | warm | Label distribution | ACTIVE | abs V <= 0.05 | sign(x2) | churn | seed (K) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2023-01-07 12:00 | 4 | CALM_FLAT, CALM_UP, ACTIVE_FLAT, ACTIVE_DOWN | 133 | 0 | ACTIVE_DOWN 11%, ACTIVE_FLAT 32%, CALM_FLAT 20%, CALM_UP 37% | 39.8% | 33.1% | 85.7% |  | 100.0% (4) |
| 2023-02-04 12:00 | 4 | CALM_FLAT, CALM_UP, ACTIVE_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 53%, CALM_FLAT 5%, CALM_UP 41% | 47.4% | 37.6% | 77.4% | 22.6% | 100.0% (4) |
| 2023-03-04 12:00 | 4 | CALM_FLAT, CALM_FLAT, CALM_FLAT, ACTIVE_FLAT | 140 | 0 | ACTIVE_FLAT 46%, CALM_FLAT 54% | 49.3% | 9.3% | 96.4% | 22.1% | 100.0% (4) |
| 2023-04-01 12:00 | 4 | CALM_FLAT, CALM_FLAT, CALM_FLAT, ACTIVE_FLAT | 168 | 0 | ACTIVE_FLAT 23%, CALM_FLAT 77% | 28.0% | 8.9% | 98.2% | 0.0% | 99.4% (4) |
| 2023-05-06 12:00 | 4 | CALM_FLAT, CALM_FLAT, CALM_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 23%, CALM_FLAT 77% | 36.8% | 43.6% | 85.0% | 0.0% | 100.0% (4) |
| 2023-06-03 12:00 | 4 | CALM_FLAT, CALM_FLAT, CALM_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 5%, CALM_FLAT 95% | 18.8% | 39.1% | 94.7% | 1.5% | 100.0% (4) |
| 2023-07-01 12:00 | 4 | CALM_FLAT, CALM_FLAT, CALM_UP, ACTIVE_FLAT | 161 | 0 | ACTIVE_FLAT 27%, CALM_FLAT 19%, CALM_UP 54% | 39.1% | 51.6% | 82.0% | 54.0% | 100.0% (4) |
| 2023-08-05 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_UP, ACTIVE_FLAT | 140 | 0 | ACTIVE_FLAT 61%, ACTIVE_UP 10%, CALM_FLAT 24%, CALM_UP 6% | 69.3% | 5.7% | 95.7% | 16.4% | 100.0% (4) |
| 2023-09-02 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_UP, ACTIVE_FLAT | 168 | 0 | ACTIVE_FLAT 48%, ACTIVE_UP 16%, CALM_FLAT 36% | 58.9% | 14.3% | 97.0% | 12.5% | 100.0% (4) |
| 2023-10-07 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_UP, ACTIVE_FLAT | 140 | 0 | ACTIVE_FLAT 26%, ACTIVE_UP 46%, CALM_FLAT 27% | 65.7% | 44.3% | 80.7% | 0.7% | 100.0% (4) |
| 2023-11-04 12:00 | 4 | CALM_FLAT, CALM_FLAT, CALM_FLAT, ACTIVE_FLAT | 126 | 0 | ACTIVE_FLAT 9%, CALM_FLAT 91% | 11.1% | 0.0% | 100.0% | 4.8% | 100.0% (4) |
| 2023-12-02 12:00 | 4 | CALM_FLAT, CALM_FLAT, CALM_FLAT, ACTIVE_FLAT | 161 | 0 | ACTIVE_FLAT 39%, CALM_FLAT 61% | 48.4% | 28.6% | 91.3% | 0.0% | 100.0% (4) |
| 2024-01-06 12:00 | 4 | CALM_FLAT, CALM_FLAT, CALM_UP, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 57%, CALM_FLAT 23%, CALM_UP 20% | 63.2% | 17.3% | 93.2% | 21.1% | 100.0% (4) |
| 2024-02-03 12:00 | 4 | CALM_FLAT, CALM_FLAT, CALM_FLAT, ACTIVE_FLAT | 126 | 0 | ACTIVE_FLAT 52%, CALM_FLAT 48% | 54.0% | 7.9% | 85.7% | 10.3% | 100.0% (4) |
| 2024-03-02 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 168 | 0 | ACTIVE_FLAT 61%, CALM_FLAT 30%, CALM_UP 10% | 58.3% | 18.5% | 91.7% | 29.2% | 100.0% (4) |
| 2024-04-06 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 140 | 0 | ACTIVE_FLAT 79%, CALM_FLAT 15%, CALM_UP 6% | 76.4% | 15.7% | 95.7% | 0.7% | 99.3% (4) |
| 2024-05-04 12:00 | 4 | CALM_FLAT, CALM_UP, ACTIVE_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 24%, CALM_FLAT 72%, CALM_UP 4% | 24.1% | 0.0% | 100.0% | 71.4% | 100.0% (4) |
| 2024-06-01 12:00 | 4 | CALM_FLAT, CALM_UP, ACTIVE_FLAT, ACTIVE_FLAT | 154 | 0 | ACTIVE_FLAT 66%, CALM_FLAT 13%, CALM_UP 21% | 65.6% | 4.5% | 92.9% | 0.6% | 100.0% (4) |
| 2024-07-06 12:00 | 4 | CALM_FLAT, CALM_UP, ACTIVE_FLAT, ACTIVE_FLAT | 140 | 0 | ACTIVE_FLAT 81%, CALM_FLAT 3%, CALM_UP 16% | 81.4% | 6.4% | 92.9% | 5.0% | 100.0% (4) |
| 2024-08-03 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 168 | 0 | ACTIVE_FLAT 36%, CALM_FLAT 64% | 35.1% | 2.4% | 97.0% | 15.5% | 100.0% (4) |
| 2024-09-07 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 52%, CALM_FLAT 36%, CALM_UP 12% | 51.9% | 1.5% | 95.5% | 12.0% | 100.0% (4) |
| 2024-10-05 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 0 | 0 |  |  |  |  |  |  |
| 2024-11-02 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 0 | 0 |  |  |  |  |  |  |
| 2024-12-07 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 0 | 0 |  |  |  |  |  |  |
| 2025-01-04 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 0 | 112 |  |  |  |  |  |  |
| 2025-02-01 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 119 | 14 | ACTIVE_FLAT 38%, CALM_FLAT 62% | 35.3% | 3.4% | 98.3% | 0.8% | 100.0% (4) |
| 2025-03-01 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 175 | 0 | ACTIVE_FLAT 55%, CALM_FLAT 45% | 54.3% | 0.6% | 96.6% | 0.0% | 100.0% (4) |
| 2025-04-05 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 32%, CALM_FLAT 68% | 31.6% | 0.0% | 100.0% | 0.0% | 100.0% (4) |
| 2025-05-03 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_DOWN | 168 | 0 | ACTIVE_DOWN 2%, ACTIVE_FLAT 24%, CALM_FLAT 36%, CALM_UP 38% | 26.2% | 0.0% | 92.3% | 40.5% | 100.0% (4) |
| 2025-06-07 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_DOWN | 119 | 0 | ACTIVE_DOWN 7%, ACTIVE_FLAT 36%, CALM_FLAT 57% | 42.9% | 0.8% | 92.4% | 16.0% | 100.0% (4) |
| 2025-07-05 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 49 | 0 | ACTIVE_FLAT 31%, CALM_FLAT 69% | 30.6% | 0.0% | 87.8% | 4.1% | 100.0% (4) |
| 2025-08-02 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 0 | 0 |  |  |  |  |  |  |
| 2025-12-06 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 119 | 0 | ACTIVE_FLAT 41%, CALM_FLAT 43%, CALM_UP 16% | 34.5% | 34.5% | 91.6% | 5.0% | 100.0% (4) |
| 2026-01-03 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 168 | 0 | ACTIVE_FLAT 93%, CALM_FLAT 7% | 91.1% | 50.0% | 84.5% | 0.0% | 100.0% (4) |
| 2026-02-07 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 130 | 0 | ACTIVE_FLAT 75%, CALM_FLAT 25% | 68.5% | 43.1% | 83.8% | 1.5% | 100.0% (4) |
| 2026-03-07 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 77%, CALM_FLAT 23% | 71.4% | 53.4% | 77.4% | 5.3% | 82.0% (4) |
| 2026-04-04 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 15%, CALM_FLAT 85% | 15.0% | 2.3% | 98.5% | 41.4% | 100.0% (4) |

No model: 2021-11-06 (insufficient, 0 rows); 2021-12-04 (insufficient, 0 rows); 2022-01-01 (insufficient, 0 rows); 2022-02-05 (insufficient, 0 rows); 2022-03-05 (insufficient, 0 rows); 2022-04-02 (insufficient, 0 rows); 2022-05-07 (insufficient, 0 rows); 2022-06-04 (insufficient, 0 rows); 2022-07-02 (insufficient, 112 rows); 2022-08-06 (insufficient, 280 rows); 2022-09-03 (insufficient, 420 rows); 2022-10-01 (insufficient, 553 rows); 2022-11-05 (insufficient, 728 rows); 2022-12-03 (insufficient, 854 rows); 2025-09-06 (insufficient, 903 rows); 2025-10-04 (insufficient, 763 rows); 2025-11-01 (insufficient, 895 rows).

### NAS100

| Refit (UTC) | K | Labels | obs | warm | Label distribution | ACTIVE | abs V <= 0.05 | sign(x2) | churn | seed (K) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2023-03-04 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 140 | 0 | ACTIVE_FLAT 54%, CALM_FLAT 16%, CALM_UP 30% | 52.1% | 6.4% | 94.3% |  | 100.0% (4) |
| 2023-04-01 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 168 | 0 | ACTIVE_FLAT 29%, CALM_FLAT 34%, CALM_UP 38% | 28.6% | 6.0% | 94.6% | 3.6% | 100.0% (4) |
| 2023-05-06 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 49%, CALM_FLAT 36%, CALM_UP 15% | 49.6% | 9.8% | 94.7% | 0.8% | 100.0% (4) |
| 2023-06-03 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 72%, CALM_FLAT 25%, CALM_UP 3% | 69.2% | 7.5% | 97.0% | 0.0% | 100.0% (4) |
| 2023-07-01 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 161 | 0 | ACTIVE_FLAT 52%, CALM_FLAT 39%, CALM_UP 9% | 52.2% | 4.3% | 96.3% | 2.5% | 100.0% (4) |
| 2023-08-05 12:00 | 4 | CALM_UP, CALM_UP, ACTIVE_FLAT, ACTIVE_FLAT | 140 | 0 | ACTIVE_FLAT 47%, CALM_UP 53% | 47.1% | 8.6% | 95.7% | 43.6% | 100.0% (4) |
| 2023-09-02 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 168 | 0 | ACTIVE_FLAT 45%, CALM_FLAT 55% | 46.4% | 3.6% | 97.0% | 54.8% | 100.0% (4) |
| 2023-10-07 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 140 | 0 | ACTIVE_FLAT 41%, CALM_FLAT 59% | 42.1% | 9.3% | 89.3% | 7.9% | 100.0% (4) |
| 2023-11-04 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 126 | 0 | ACTIVE_FLAT 15%, CALM_FLAT 17%, CALM_UP 67% | 15.1% | 0.8% | 96.8% | 67.5% | 100.0% (4) |
| 2023-12-02 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 161 | 0 | ACTIVE_FLAT 58%, CALM_FLAT 11%, CALM_UP 31% | 58.4% | 2.5% | 97.5% | 4.3% | 100.0% (4) |
| 2024-01-06 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 77%, CALM_FLAT 17%, CALM_UP 6% | 77.4% | 1.5% | 97.7% | 2.3% | 100.0% (4) |
| 2024-02-03 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 126 | 0 | ACTIVE_FLAT 29%, CALM_FLAT 52%, CALM_UP 19% | 31.7% | 8.7% | 92.9% | 0.0% | 100.0% (4) |
| 2024-03-02 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 168 | 0 | ACTIVE_FLAT 38%, CALM_FLAT 34%, CALM_UP 28% | 39.3% | 3.0% | 95.8% | 28.6% | 82.7% (4) |
| 2024-04-06 12:00 | 4 | CALM_FLAT, CALM_UP, CALM_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 70%, CALM_FLAT 15%, CALM_UP 15% | 71.4% | 5.3% | 95.5% | 30.1% | 100.0% (4) |
| 2024-05-04 12:00 | 4 | CALM_FLAT, CALM_FLAT, CALM_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 26%, CALM_FLAT 74% | 25.6% | 1.5% | 98.5% | 5.3% | 100.0% (4) |
| 2024-06-01 12:00 | 4 | CALM_FLAT, CALM_FLAT, CALM_FLAT, ACTIVE_FLAT | 154 | 0 | ACTIVE_FLAT 51%, CALM_FLAT 49% | 54.5% | 9.7% | 90.9% | 2.6% | 100.0% (4) |
| 2024-07-06 12:00 | 4 | CALM_FLAT, CALM_UP, CALM_FLAT, ACTIVE_FLAT | 140 | 0 | ACTIVE_FLAT 84%, CALM_FLAT 13%, CALM_UP 3% | 84.3% | 0.0% | 100.0% | 2.9% | 100.0% (4) |
| 2024-08-03 12:00 | 4 | CALM_FLAT, CALM_FLAT, CALM_FLAT, ACTIVE_FLAT | 168 | 0 | ACTIVE_FLAT 30%, CALM_FLAT 70% | 33.9% | 8.3% | 96.4% | 18.5% | 98.8% (4) |
| 2024-09-07 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 25%, CALM_FLAT 75% | 31.6% | 30.1% | 89.5% | 1.5% | 100.0% (4) |
| 2024-10-05 12:00 | 4 | CALM_FLAT, CALM_UP, ACTIVE_FLAT, ACTIVE_FLAT | 0 | 0 |  |  |  |  |  |  |
| 2024-11-02 12:00 | 4 | CALM_FLAT, CALM_UP, ACTIVE_FLAT, ACTIVE_FLAT | 0 | 0 |  |  |  |  |  |  |
| 2024-12-07 12:00 | 4 | CALM_FLAT, CALM_UP, ACTIVE_FLAT, ACTIVE_FLAT | 0 | 0 |  |  |  |  |  |  |
| 2025-01-04 12:00 | 4 | CALM_FLAT, CALM_UP, ACTIVE_FLAT, ACTIVE_FLAT | 0 | 112 |  |  |  |  |  |  |
| 2025-02-01 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_DOWN | 119 | 14 | ACTIVE_DOWN 21%, ACTIVE_FLAT 24%, CALM_FLAT 55% | 42.9% | 22.7% | 85.7% | 29.4% | 95.0% (4) |
| 2025-03-01 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_DOWN | 175 | 0 | ACTIVE_DOWN 39%, ACTIVE_FLAT 20%, CALM_FLAT 41% | 58.3% | 0.6% | 93.7% | 20.0% | 86.9% (4) |
| 2025-04-05 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 28%, CALM_FLAT 72% | 28.6% | 5.3% | 98.5% | 33.1% | 100.0% (4) |
| 2025-05-03 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 168 | 0 | ACTIVE_FLAT 33%, CALM_FLAT 67% | 32.7% | 0.6% | 95.2% | 16.1% | 100.0% (4) |
| 2025-06-07 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 119 | 0 | ACTIVE_FLAT 40%, CALM_FLAT 60% | 37.0% | 2.5% | 92.4% | 2.5% | 100.0% (4) |
| 2025-07-05 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 49 | 0 | ACTIVE_FLAT 8%, CALM_FLAT 92% | 8.2% | 2.0% | 100.0% | 6.1% | 100.0% (4) |
| 2025-08-02 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 0 | 0 |  |  |  |  |  |  |
| 2025-12-06 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 119 | 0 | ACTIVE_FLAT 41%, CALM_FLAT 29%, CALM_UP 30% | 39.5% | 3.4% | 95.0% | 34.5% | 98.3% (4) |
| 2026-01-03 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 168 | 0 | ACTIVE_FLAT 80%, CALM_FLAT 13%, CALM_UP 7% | 79.8% | 2.4% | 95.2% | 1.2% | 99.4% (4) |
| 2026-02-07 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 126 | 0 | ACTIVE_FLAT 54%, CALM_FLAT 40%, CALM_UP 6% | 53.2% | 4.0% | 95.2% | 1.6% | 100.0% (4) |
| 2026-03-07 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 53%, CALM_FLAT 47% | 51.1% | 1.5% | 70.7% | 9.0% | 99.2% (4) |
| 2026-04-04 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 133 | 0 | ACTIVE_FLAT 35%, CALM_FLAT 65% | 28.6% | 22.6% | 79.7% | 36.8% | 100.0% (4) |

No model: 2022-08-06 (insufficient, 21 rows); 2022-09-03 (insufficient, 161 rows); 2022-10-01 (insufficient, 294 rows); 2022-11-05 (insufficient, 469 rows); 2022-12-03 (insufficient, 595 rows); 2023-01-07 (insufficient, 756 rows); 2023-02-04 (insufficient, 889 rows); 2025-09-06 (insufficient, 903 rows); 2025-10-04 (insufficient, 763 rows); 2025-11-01 (insufficient, 895 rows).

### GER40

| Refit (UTC) | K | Labels | obs | warm | Label distribution | ACTIVE | abs V <= 0.05 | sign(x2) | churn | seed (K) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2022-10-01 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 200 | 0 | ACTIVE_FLAT 42%, CALM_FLAT 57% | 37.0% | 29.0% | 87.5% |  | 100.0% (4) |
| 2022-11-05 12:00 | 4 | CALM_FLAT, CALM_FLAT, CALM_FLAT, ACTIVE_FLAT | 160 | 0 | ACTIVE_FLAT 14%, CALM_FLAT 86% | 15.6% | 20.6% | 91.2% | 17.5% | 97.5% (4) |
| 2022-12-03 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 184 | 0 | ACTIVE_FLAT 66%, CALM_FLAT 34% | 63.6% | 29.9% | 89.1% | 35.3% | 100.0% (4) |
| 2023-01-07 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 160 | 0 | ACTIVE_FLAT 52%, CALM_FLAT 48% | 45.6% | 39.4% | 83.1% | 4.4% | 100.0% (4) |
| 2023-02-04 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 160 | 0 | ACTIVE_FLAT 68%, CALM_FLAT 32% | 63.1% | 26.9% | 86.9% | 1.2% | 100.0% (4) |
| 2023-03-04 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 160 | 0 | ACTIVE_FLAT 49%, CALM_FLAT 51% | 47.5% | 4.4% | 97.5% | 1.2% | 100.0% (4) |
| 2023-04-01 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 176 | 0 | ACTIVE_FLAT 32%, CALM_FLAT 68% | 31.8% | 2.8% | 100.0% | 0.0% | 100.0% (4) |
| 2023-05-06 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 160 | 0 | ACTIVE_FLAT 54%, CALM_FLAT 46% | 53.1% | 21.9% | 91.9% | 1.9% | 100.0% (4) |
| 2023-06-03 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 160 | 0 | ACTIVE_FLAT 45%, CALM_FLAT 55% | 42.5% | 34.4% | 88.8% | 0.6% | 100.0% (4) |
| 2023-07-01 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 200 | 0 | ACTIVE_FLAT 67%, CALM_FLAT 33% | 62.5% | 34.5% | 89.5% | 0.5% | 100.0% (4) |
| 2023-08-05 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 160 | 0 | ACTIVE_FLAT 61%, CALM_FLAT 39% | 58.8% | 29.4% | 91.9% | 0.0% | 100.0% (4) |
| 2023-09-02 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 200 | 0 | ACTIVE_FLAT 68%, CALM_FLAT 32% | 67.0% | 28.0% | 93.5% | 0.0% | 100.0% (4) |
| 2023-10-07 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 160 | 0 | ACTIVE_FLAT 61%, CALM_FLAT 39% | 58.1% | 30.6% | 90.6% | 0.0% | 100.0% (4) |
| 2023-11-04 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 160 | 0 | ACTIVE_FLAT 16%, CALM_FLAT 84% | 15.6% | 16.2% | 92.5% | 0.0% | 100.0% (4) |
| 2023-12-02 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 168 | 0 | ACTIVE_FLAT 83%, CALM_FLAT 17% | 79.8% | 36.9% | 88.7% | 0.6% | 100.0% (4) |
| 2024-01-06 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 160 | 0 | ACTIVE_FLAT 68%, CALM_FLAT 32% | 64.4% | 40.6% | 88.1% | 0.6% | 100.0% (4) |
| 2024-02-03 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 160 | 0 | ACTIVE_FLAT 43%, CALM_FLAT 57% | 38.8% | 31.2% | 86.9% | 3.1% | 100.0% (4) |
| 2024-03-02 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 184 | 0 | ACTIVE_FLAT 69%, CALM_FLAT 31% | 65.8% | 25.5% | 86.4% | 1.1% | 100.0% (4) |
| 2024-04-06 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 152 | 0 | ACTIVE_FLAT 80%, CALM_FLAT 20% | 75.7% | 20.4% | 94.1% | 0.0% | 100.0% (4) |
| 2024-05-04 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 160 | 0 | ACTIVE_FLAT 28%, CALM_FLAT 72% | 26.2% | 5.0% | 99.4% | 0.0% | 100.0% (4) |
| 2024-06-01 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 200 | 0 | ACTIVE_FLAT 77%, CALM_FLAT 23% | 74.0% | 30.5% | 87.5% | 0.5% | 99.5% (4) |
| 2024-07-06 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 160 | 0 | ACTIVE_FLAT 92%, CALM_FLAT 8% | 89.4% | 45.6% | 81.2% | 0.0% | 100.0% (4) |
| 2024-08-03 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 200 | 0 | ACTIVE_FLAT 32%, CALM_FLAT 68% | 32.0% | 3.0% | 99.5% | 56.0% | 100.0% (4) |
| 2024-09-07 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 152 | 0 | ACTIVE_FLAT 62%, CALM_FLAT 38% | 59.2% | 17.1% | 85.5% | 0.7% | 100.0% (4) |
| 2024-10-05 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 0 | 0 |  |  |  |  |  |  |
| 2024-11-02 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 0 | 0 |  |  |  |  |  |  |
| 2024-12-07 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 0 | 0 |  |  |  |  |  |  |
| 2025-01-04 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 0 | 144 |  |  |  |  |  |  |
| 2025-02-01 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 144 | 16 | ACTIVE_FLAT 55%, CALM_FLAT 45% | 52.8% | 12.5% | 96.5% | 1.4% | 100.0% (4) |
| 2025-03-01 12:00 | 4 | CALM_UP, CALM_UP, ACTIVE_FLAT, ACTIVE_FLAT | 200 | 0 | ACTIVE_FLAT 60%, CALM_UP 40% | 60.0% | 4.5% | 93.5% | 39.5% | 100.0% (4) |
| 2025-04-05 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_DOWN | 136 | 0 | ACTIVE_DOWN 29%, ACTIVE_FLAT 2%, CALM_FLAT 4%, CALM_UP 65% | 32.4% | 3.7% | 100.0% | 33.1% | 100.0% (4) |
| 2025-05-03 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 192 | 0 | ACTIVE_FLAT 17%, CALM_FLAT 27%, CALM_UP 56% | 17.2% | 1.0% | 88.0% | 15.1% | 100.0% (4) |
| 2025-06-07 12:00 | 4 | CALM_UP, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 144 | 0 | ACTIVE_FLAT 68%, CALM_FLAT 26%, CALM_UP 6% | 63.9% | 50.0% | 85.4% | 54.9% | 100.0% (4) |
| 2025-07-05 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 57 | 0 | ACTIVE_FLAT 21%, CALM_FLAT 79% | 19.3% | 5.3% | 84.2% | 19.3% | 100.0% (4) |
| 2025-08-02 12:00 | 4 | CALM_FLAT, CALM_FLAT, CALM_FLAT, ACTIVE_FLAT | 0 | 0 |  |  |  |  |  |  |
| 2025-09-06 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 0 | 144 |  |  |  |  |  |  |
| 2025-11-01 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 200 | 0 | ACTIVE_FLAT 65%, CALM_FLAT 35% | 63.5% | 25.5% | 87.5% | 4.5% | 100.0% (4) |
| 2025-12-06 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 112 | 0 | ACTIVE_FLAT 33%, CALM_FLAT 67% | 32.1% | 3.6% | 84.8% | 13.4% | 100.0% (4) |
| 2026-01-03 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 200 | 0 | ACTIVE_FLAT 70%, CALM_FLAT 30% | 70.0% | 4.0% | 94.0% | 2.5% | 100.0% (4) |
| 2026-02-07 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 160 | 0 | ACTIVE_FLAT 36%, CALM_FLAT 64% | 35.0% | 1.9% | 96.2% | 2.5% | 100.0% (4) |
| 2026-03-07 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 152 | 0 | ACTIVE_FLAT 68%, CALM_FLAT 32% | 67.1% | 2.6% | 94.1% | 1.3% | 100.0% (4) |
| 2026-04-04 12:00 | 4 | CALM_FLAT, CALM_FLAT, ACTIVE_FLAT, ACTIVE_FLAT | 144 | 0 | ACTIVE_FLAT 15%, CALM_FLAT 85% | 14.6% | 1.4% | 96.5% | 4.2% | 100.0% (4) |

No model: 2021-11-06 (insufficient, 0 rows); 2021-12-04 (insufficient, 0 rows); 2022-01-01 (insufficient, 0 rows); 2022-02-05 (insufficient, 0 rows); 2022-03-05 (insufficient, 0 rows); 2022-04-02 (insufficient, 96 rows); 2022-05-07 (insufficient, 280 rows); 2022-06-04 (insufficient, 440 rows); 2022-07-02 (insufficient, 600 rows); 2022-08-06 (insufficient, 800 rows); 2022-09-03 (insufficient, 960 rows); 2025-10-04 (insufficient, 873 rows).

## Notes

- **First valid model** (COMMON 6.3 asks for the actual dates): US30 2023-01-07, NAS100
  2023-03-04, GER40 2022-10-01, all at or before the expected 2023-03. H1 starts 2021-11-03 for
  US30 and GER40, but valid trading dates need the cash-open M15 bar (COMMON 1.6) and M15 starts
  2022-05-24 (US30), 2022-07-06 (NAS100) and 2022-03-02 (GER40); 1,000 full training rows at
  7 (US) or 8 (GER40) core observations per date then take about 6 to 7 months. K = 4 was chosen
  at every successful refit of every index; no refit failed.

- **A2 counts.** The warm-up rows match the QA finding that triggered amendment A2: US30 127
  and GER40 305 outputs that 6.6 alone would have reported as OK from the stationary prior.
  NAS100 has the same 127 (same US calendar and segment breaks). All of them are now
  UNAVAILABLE ("regime warm-up").
- **Which resets produce warm-up rows.** Both resets start a warm-up on all three indices (see
  the warm-up table for where each one ends). The reset after the A1 excluded window (data
  resumes 2025-01-08) produces warm-up rows on all three. The reset after the 2025 data hole
  produces warm-up rows only on GER40: for US30 and NAS100 the refits of 2025-09-06, 2025-10-04
  and 2025-11-01 have fewer than 1,000 training rows, so those dates are already UNAVAILABLE
  with the "fewer than 1,000 training rows" cause (no model, no warm-up row). On GER40 the
  2025-09-06 model serves part of the warm-up (to 2025-10-03 15:00), the 2025-10-04 refit has
  no model, and the first OK output after the stretch is the 2025-11-01 refit row (its alpha is
  that of the 2025-10-31 16:00 observation).
- **Strategy gate-block shares** ("share of in-window cycles where the strategy's gate blocks",
  COMMON 6.8) need each strategy's decision windows and are not part of this regime-only
  report; `regime.diagnostics.gate_block_share(table, times, gate)` computes them from the
  decision times a strategy supplies (feasibility gate, COMMON 10.1).
- **Reading the per-refit columns.** `sign(x2)` and the gate share the same neutral point 0
  (x2 >= 0 counts as ACTIVE). Churn compares argmax label strings of consecutive models on the
  same observations, so a relabelled but otherwise identical state counts as churn.
- **Reproduction.** Scratch scripts `a2_regime_diag_real.py` (batch, diagnostics, timings) and
  `a2_regime_diag_md.py` (this page) in the session scratchpad; per-refit CSVs
  `a2_regime_diag_<KEY>.csv` alongside.
- **Refit log and model cache (correction, 2026-11-03).** An earlier version of this note said
  these runs do not log refit attempts; that was wrong for the runs that produced the numbers.
  The diagnostics runs (2026-11-03 04:25 to 04:35 UTC) used the default refit log and appended
  127 rows to `regime/refits.csv` (US30 28, NAS100 45, GER40 54); `refit_log=None` was set in
  the script only afterwards. Those 127 rows were then removed from the log (it was truncated
  back to its 179 rows of 2026-11-02). They were not duplicates: `train_sha256` and
  `model_hash` matched the retained rows, but `window_sha256` did not (the retained rows of
  NAS100, GER40 and 28 of US30 predate the window hash and carry none). The removal has been
  undone: the 127 rows were restored verbatim from the backup taken before the truncation, so
  `regime/refits.csv` has 306 rows again. The production model cache `regime/models/` held
  pickles whose window hash no longer matched (only 23 of 114 under the current settings hash),
  so a production run would have refitted them. One production (logged) batch run per index
  with the default model directory and refit log (2026-11-03, after the restore) regenerated
  `regime/models/`: all 114 pickles under the current settings hash now match a logged
  (key, refit time, training hash, window hash) identity, and the run appended no refits.csv
  row because every attempt was already logged. A reload from the regenerated cache gave a
  table identical to the run that wrote it, with the same warm-up and model row counts as
  above. 102 pickles under an older settings hash remain in `regime/models/`; they are never
  loaded (the settings hash is part of the file name and is checked on load).
