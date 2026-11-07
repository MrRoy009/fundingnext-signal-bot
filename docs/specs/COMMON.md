# COMMON spec: shared features, index selection, regime model, execution, backtest and validation

Status: SPECIFIED - not tested. GATE 2 document written by Quant before any backtest.
Applies to every strategy in this folder (RS-MOM-01, SES-IBX-01, MR-TWAP-01). A strategy spec may
add rules but may not change anything here. Anything here that is changed after the first backtest
is a new trial (section 11) and needs QA sign-off.

Every number in this file comes from convention, market structure or the descriptive (non-outcome)
analysis of pre-holdout data listed in section 12. No profit, win rate or forward return of any
rule was computed while writing it.

---

## 1. Universe, data and calendar

### 1.1 Instruments
- Ranked and traded: US30, NAS100 (broker USTEC), GER40 (broker DE40).
- US500 is NOT ranked and NOT traded (owner decision O6). Reasons: at the 3-lot cap its risk is
  about $35-60 per trade (0.04% of balance), it correlates 0.91-0.95 with the other US indices,
  and it ranks top or bottom in only about 10% of bars. If the owner keeps US500, it re-enters as
  a separate, logged trial set for every strategy.
- Entry timeframe M15, regime timeframe H1. MT5 bars are BID prices. `time_utc` is the bar OPEN
  time in true UTC (GATE 1 conversion from Europe/Athens server time).

### 1.2 Data windows
- Research data: `BarStore(load_config()).load(key, tf)` only. It drops bars on or after
  2026-05-01 00:00 UTC. `final_holdout=True` is never passed before GATE 3 approval.
- Usable common span: M15 from 2022-07-06 (NAS100 is the binding start), H1 from 2021-11-03
  (NAS100 H1 from 2022-07-06). Last pre-holdout bar: 2026-04-30 23:45 UTC.

### 1.3 Known data defects (frozen list, stored in config by the Engineer; see engineering request ER-09)
| Defect | Window (UTC) | Instruments | Handling |
|---|---|---|---|
| Data hole: no M15 or H1 bars | 2025-07-16 07:30 to 2025-09-09 11:30 (GER40 from 2025-07-22 21:00) | all | Segment break (1.4) |
| Tick volume corrupt (flat ~8,900 per bar, single bars of 100k+, GER40 = 15) | 2024-10-04 00:00 to 2025-01-08 00:00 | all | `tv_bad` flag (1.5) |
| Demo prints normal-looking bars on US exchange holidays | NYSE holiday dates | US30, NAS100 | Exchange calendar (1.6) |
| Zero-spread bars (0.1-0.7% of bars, more in 2025-26) | scattered | all | Spread floor (8.2) |

The data hole is 55 days. The GATE 1 report missed it because gaps over 12 h were classed as
weekends. The owner may backfill it before GATE 3 (decision O10); any backfill changes the data
hash, which every trial log row records.

### 1.4 Segments
For each index and timeframe, a new segment starts at bar t when `time_utc[t] - time_utc[t-1] > 7 days`.
The longest ordinary closure in the data is 5 days 4 hours (GER40 Christmas 2025), so only the
2025 hole triggers it. Every rolling, EWM or "previous N dates" computation in this spec restarts
at a segment start: it never uses bars from an earlier segment. A feature is NaN until its full
lookback exists inside the current segment.

### 1.5 Tick-volume flag
`tv_bad(t) = True` if bar t opens inside the corrupt window in 1.3. Tick volume of flagged bars
is never used: features that need it are NaN on flagged bars, and flagged dates are skipped when
building any "previous N dates" reference set for tick volume.

### 1.6 Exchange calendar
The Engineer adds a frozen calendar for 2021-2027 (ER-08): NYSE full holidays and early closes
(13:00 New York; e.g. 3 July, day after Thanksgiving, 24 December when weekdays) for US30 and
NAS100; Xetra holidays and early closes (24 and 31 December) for GER40.
- A "valid trading date" for an index is a local exchange date that is a weekday, not a full
  holiday and not an early-close date, and that has the cash-open M15 bar (09:30 New York or
  09:00 Berlin).
- No strategy issues a ticket on a date that is not a valid trading date.
- Every "previous N dates" lookback in this spec uses valid trading dates only.

---

## 2. Time conventions

- **Bar close time.** For a bar with open time `o`: `close = o + 15 min` (M15) or `o + 60 min` (H1).
- **Decision time T.** A cycle runs at T = the close time of M15 bar t (open `T - 15 min`). It may
  use only bars with close time <= T. The forming bar is never used.
- **Exchange-local time.** America/New_York for US30 and NAS100, Europe/Berlin for GER40.
  Conversion uses `zoneinfo` on `time_utc`. No fixed UTC offsets anywhere.
- **Cash session.** US: 09:30-16:00 New York. GER40: 09:00-17:30 Berlin. A bar is "in session"
  if its local open time is >= cash open and < cash close.
- **Signal windows** are defined by each strategy in local time, both ends inclusive, on T.
  Every strategy window also lies inside config `sessions_utc` in both DST states (checked in each
  spec), so the UTC config does not bind with current values. The engine's session check is
  applied to T, not to wall-clock now, with the convention `start <= T < end` (ER-12).
- **H1 alignment at T.** Let j be the latest H1 bar of index i with close <= T. H1-derived inputs
  for i (H1 ATR, RS_day) are UNAVAILABLE at T if close(j) <= T - 60 min while index i has at least
  one M15 bar with open time in [close(j), floor_hour(T)). This catches a live H1 bar that has not
  arrived yet (it is never silently replaced by the previous bar). The regime uses its own rule (6.6).
- **Manual time exit clocks** (all strategies): 15:45 New York for US30/NAS100, 17:15 Berlin for
  GER40, on the signal's local date. UTC equivalents: US 19:45 (US summer) or 20:45 (US winter);
  GER40 15:15 (EU summer) or 16:15 (EU winter).

---

## 3. Features (all causal, per index, within segment)

| Name | Definition |
|---|---|
| `TR_t` | `max(H_t - L_t, abs(H_t - C_prev), abs(L_t - C_prev))`, C_prev = close of the previous bar in the same series and segment. First bar of a segment: `H_t - L_t`. |
| `ATR14_M15`, `ATR14_H1` | Wilder ATR, period 14: EWM of TR with alpha = 1/14, adjust=False, min_periods=14 (`features.indicators.atr(df, 14)`), restarted at each segment start. Price points. |
| `TODV_t` (M15) | Median of TR over the M15 bars of the same index whose exchange-local open time HH:MM equals bar t's, on the previous 20 valid trading dates (same segment) that have such a bar. Requires >= 10 such bars, else NaN. Current date excluded. |
| `RS_4h` (M15) | See 4.1. |
| `RS_day` (H1) | See 4.1. |
| `TWAP_t` | Mean of typical price `(H+L+C)/3` over the M15 bars of the current local date from the cash-open bar to bar t inclusive, equal weights; missing bars are skipped, never filled. NaN if the cash-open bar is missing. |
| `IB` | For a given `ib_minutes` m: the first m/15 M15 bars with local open time from cash open onward. All must exist (no missing open time), else NaN. `IBH = max H`, `IBL = min L`, `W = IBH - IBL`. Defined only from the close of the last IB bar. |
| `DATR_d` | Mean of (max H - min L) over in-session M15 bars, over the 14 most recent prior valid trading dates on which at least 90% of the expected session bars exist (US 26, GER40 34). Today excluded. |
| `spread_pts_t` | `spread_t * 0.01` (bar spread in index points). Cost use only, never a signal input. |

---

## 4. Index selection (rank score)

### 4.1 Rank scores
Each strategy names exactly one rank score; it is fixed and not a tuning choice.

**RS_4h** (used by RS-MOM-01 and SES-IBX-01), at M15 bar t of index i, with L = 16:

    RS_4h(i,t) = (C_t - C_{t-L}) / sqrt( sum_{j=t-L+1..t} TODV_j^2 )

Valid only if bars t-L..t all exist with consecutive 15-minute open times (no missing bar, no
daily break or weekend inside), all TODV_j are finite, and all are in one segment. Otherwise NaN.
The denominator is the typical displacement for those exact 16 local times of day, so the score is
not inflated at the cash open (the problem with M15 ATR, which lags at the open) and is comparable
across indices (points cancel).

**RS_day** (used by MR-TWAP-01), at the H1 bar j defined in section 2 (H1 alignment):

    RS_day(i,T) = (C^H1_j - C^H1_{j-24}) / ATR14_H1_j

Valid only if bars j-24..j are in one segment and ATR14_H1_j > 0. One trading day of H1 bars.

### 4.2 Selection mechanics (each strategy, each cycle T, separately per strategy)
1. **Window set** G(T) = indices whose strategy signal window contains T on a valid trading date.
   In every current spec the US window and the GER40 window never overlap, so G(T) is either
   {US30, NAS100} or {GER40}.
2. **Data completeness.** If any member of G(T) lacks its M15 bar closing at T, or has a NaN rank
   score or NaN strategy input because of missing data, the strategy produces NO candidate for the
   whole group at T (logged "incomplete group: <index>: <cause>"). No partial ranking.
3. **Regime filter.** E(T) = members of G(T) whose regime status is OK and that pass the
   strategy's regime gate (section 6.7). Gates are side-symmetric, so one set serves both sides.
4. **Rank among passing indices** (owner decision O7; default is the literal PROMPT reading,
   "the strongest index that passes the regime filter"): LONG-SELECTED = argmax of the rank score
   over E(T); SHORT-SELECTED = argmin over E(T). Ties within 1e-12 go to config order
   (US30, NAS100, GER40): the earlier index is higher for the long and lower for the short.
5. **Single index.** If |E(T)| = 1, that index may receive only a BUY if its rank score > 0 and
   only a SELL if < 0; neither if = 0.
6. Only LONG-SELECTED may get a BUY candidate and only SHORT-SELECTED a SELL candidate. If the
   selected index fails a strategy entry condition, the next index is never promoted.
7. If E(T) is empty: no candidates; log the reason per index.
8. `Candidate.rank_score` = the chosen index's rank score, rounded to 2 dp. Every cycle logs the
   rank scores of all members of G(T).

Honest scope: with these windows GER40 is always ranked alone, so for GER40 "relative strength"
reduces to the sign of its own score; for the US group it is US30 vs NAS100 (M15 return
correlation about 0.79). No spec may claim a cross-sectional effect for GER40.

---

## 5. Not used
No external inputs (VIX, breadth, sector strength). No strategy uses a directional regime gate:
QA showed the sign of a state's mean H1 return is statistically indistinguishable from zero and can
flip between refits. Direction tags exist only for logging (6.5).

---

## 6. Regime model (Gaussian HMM, per index, H1)

### 6.1 Observations
One independent model per index (US30, NAS100, GER40). Observation set = **core H1 bars**:
H1 bars whose exchange-local open hour is 09..15 New York (US, 7 bars, 09:00-16:00) or 09..16
Berlin (GER40, 8 bars, 09:00-17:00), on valid trading dates, with H > L. Core bars on other dates
or with H = L are "missing" (prediction-only step in the filter, excluded from training).

Three inputs, computed on the full H1 series (all hours) and then sampled at core bars:

| Input | Definition |
|---|---|
| x1 (normalised return) | `(C_t - C_prev) / S_t`, where S_t = median TR of H1 bars with the same exchange-local open hour on the previous 20 valid trading dates (>= 10 required). |
| x2 (ATR-normalised volatility) | `ln( mean(TR over the last n_d H1 bars) / mean(TR over the last 20*n_d H1 bars) )`, by position, ending at t, n_d = 23 for US30/NAS100 and 22 for GER40 (median H1 bars per day). The short window always covers one full daily cycle, so it carries no time-of-day pattern. Neutral point 0. |
| x3 (relative tick volume) | `ln( (TV_t + 1) / (median TV of H1 bars with the same local open hour on the previous 20 valid, non-flagged trading dates + 1) )`, >= 10 required. NaN if `tv_bad`. tick_volume is cast to float first. |

All three need their lookback inside the current segment (1.4).

### 6.2 Preprocessing (fitted on the training window only, stored with the model)
1. Winsorise each input at its training 0.5th and 99.5th percentiles (`np.quantile`, linear).
2. z-score with the training mean and std (ddof=0) of the clipped values.
The same clip bounds, means and stds are applied unchanged to every observation until the next refit.

### 6.3 Training window and refit schedule
- Refit times R_m = the first Saturday of each calendar month, 12:00 UTC (markets closed, so no
  session ever spans two parameter sets).
- Training rows: core observations with close time in [R_m - 365 days, R_m) and x1, x2, x3 all finite.
  Rows with `tv_bad` are therefore excluded.
- Sequences: consecutive training rows more than 7 days apart start a new sequence (hmmlearn
  `lengths`). Overnight and weekend gaps do not split sequences.
- Minimum 1,000 training rows; otherwise no model and status UNAVAILABLE.
- The model fitted at R_m serves every decision with T in [R_m, R_{m+1}). Live: refit on the first
  cycle with now >= R_m, before that cycle's decision; persist the model, its fit time, the training
  data SHA-256 and a model hash (ER-06).
- Expected first valid model: US30 and GER40 well before 2023-03; NAS100 around 2023-03 (H1 from
  2022-07-06 plus warm-up). The Engineer logs the actual dates.

### 6.4 Fitting and state selection
- `hmmlearn.hmm.GaussianHMM` (hmmlearn 0.3.3, scikit-learn 1.9.1, as pinned), for K in {2, 3, 4}:
  `covariance_type='full', n_iter=500, tol=1e-4, min_covar=1e-3, init_params='stmc', params='stmc'`,
  `random_state` in 0..9 (10 restarts per K). Per K keep the restart with the highest training
  `score(X, lengths)` among restarts that converged (`monitor_.converged`) without exception.
- Reject a K if any state's training occupancy (sum of FILTERED probabilities over training rows /
  number of rows) < 0.02, or any covariance matrix is not positive definite or has condition
  number > 1e6, or any `transmat_` row sum differs from 1 by more than 1e-6.
- BIC = -2 lnL + p ln(n_rows), p = (K-1) + K(K-1) + K*d + K*d(d+1)/2 with d = 3
  (p = 21, 35, 51 for K = 2, 3, 4). Choose the minimum BIC among non-rejected K; ties within 1e-6
  go to the smaller K. Training data only.
- Log per refit: K, lnL / p / BIC for every K, lnL for every restart.
- Seed-stability diagnostic (descriptive, not a choice): refit with seeds 10..19 and report the
  share of the next month's core observations where the gate decision (6.7) agrees with the main
  model.

### 6.5 Relabelling (after every refit, from training data only)
For each state s, in unstandardised (clipped) units:
- `mu_vol_s = means_[s,1]*std_2 + mean_2` (mean of x2).
- `mu_ret_s = means_[s,0]*std_1 + mean_1`; `sd_ret_s = sqrt(covars_[s,0,0])*std_1`;
  `occ_s` = sum of filtered probabilities of s over training rows; `SE_s = sd_ret_s / sqrt(occ_s)`.
- Volatility tag: ACTIVE if `mu_vol_s >= 0`, CALM if `mu_vol_s < 0`.
- Direction tag (logging only): UP if `mu_ret_s > 2*SE_s`, DOWN if `< -2*SE_s`, else FLAT.
- State label = `<VOL>_<DIR>`, e.g. CALM_FLAT. States are listed in ascending (mu_vol, mu_ret).
- Log per refit: per-state mu_vol, mu_ret, sd_ret, occ, SE, label, and the transition matrix to
  `regime/refits.csv`. These describe training states; they are not outcome statistics of any rule.

### 6.6 Forward filter (hand-written; never `predict_proba`, `decode` or Viterbi)
In log space with log-sum-exp:
- Emission: if x1, x2, x3 are all finite, `log b_t(j) = multivariate_normal.logpdf(z_t, means_[j], covars_[j])`.
  If only x3 is NaN (e.g. `tv_bad`), use the 2-D marginal on (x1, x2): `means_[j][:2]`, `covars_[j][:2,:2]`.
  If x1 or x2 is NaN, or the core bar is missing: no emission (prediction-only step).
- Start: at the first core observation of the new model's training window,
  `log alpha_0(j) = log startprob_[j] + log b_0(j)`, normalised.
- Step: `log alpha_t(j) = log b_t(j) + logsumexp_i( log alpha_{t-1}(i) + log A_ij )`, normalised.
  Prediction-only step: `alpha_t = alpha_{t-1} A`.
- After each refit the new model's filter is re-run from the start of its training window through
  every core observation with close < R_m, then continues observation by observation; every
  observation enters only at or after its own close time.
- Segment start: reset alpha to the stationary distribution of A (left eigenvector for eigenvalue 1;
  if ambiguous, `startprob_`), then apply the next emission.
- Non-finite step log-likelihood: treat as prediction-only and log a WARNING.
- **Output at decision time T:** alpha after the latest core observation with close <= T. If that
  observation is more than 5 calendar days before T and no reset applies, status UNAVAILABLE (stale).
  Live: if the core H1 bar that closes exactly at T (full-hour T inside core hours) has not arrived,
  status for that cycle is UNAVAILABLE ("regime H1 bar missing"), never the previous value.

### 6.7 Regime outputs and gates
- `V_T = sum_s alpha_T(s) * mu_vol_s`: the filtered expected log volatility ratio. This is
  K-invariant (it means the same thing whatever K is), continuous, and needs no probability threshold.
- **ACTIVE gate:** passes iff status OK and `V_T >= 0`. **CALM gate:** passes iff status OK and `V_T < 0`.
  Threshold 0 is the neutral point of the log ratio and is FROZEN.
- `Candidate.regime` = `ACTIVE` or `CALM` (sign of V_T) + `/` + label of the argmax state.
  `Candidate.regime_prob` = sum of alpha over states whose volatility tag matches the sign of V_T.
- Status UNAVAILABLE (no model, stale, missing H1 bar, fewer than 1,000 rows): index not eligible;
  the engine logs "regime unavailable: <cause>" and never guesses.
- Refit failure (every K rejected or every restart failing): keep the previous model if it was fitted
  <= 62 days before T, else UNAVAILABLE; retry at each later core H1 close, identically live and in
  the backtest.

### 6.8 Pre-registered regime diagnostics (reported before any P&L is examined)
Per index and per refit: K, label distribution, share of in-window cycles where the strategy's gate
blocks, label churn between consecutive refits, share of core observations with V_T in [-0.05, 0.05],
agreement between the gate and sign(x2) alone, and the seed-stability share. If a gate blocks fewer
than 5% or more than 95% of a strategy's in-window cycles, the report must say the gate is
non-functional for that strategy and no regime-conditional claim may be made. Nothing in the
regime spec is changed in response to these numbers.

---

## 7. Execution protocols and tickets

The owner places every ticket by hand d = 1, 2 or 3 M15 bars after T, with SL and TP attached.
Each strategy uses exactly one protocol. All price levels on a ticket are absolute prices; the owner
never re-anchors SL or TP to the fill. These protocols replace the symmetric "skip if +/-0.25 x ATR"
rule for these strategies (owner decision O1). If the owner rejects them, the fallback is the current
symmetric rule at the market (MARKET_BAND with tolerance 0.25 x ATR14_M15 on both sides); it
REPLACES the protocol (it is not run in addition), and SES-IBX-01 is dropped because it cannot be
expressed without a pending stop order.

Let D = |E - SL| for a market or limit ticket (E = entry reference), or |trigger - SL| for a stop ticket.
Q = the owner's live price at placement (ask to buy, bid to sell).

| Protocol | Rule at placement (state-only: the owner compares the live quote with printed levels) |
|---|---|
| MARKET_BAND | Skip if Q is worse than E by more than 0.20 D (CHASE LIMIT printed). Skip if Q is better than E by more than 0.50 D (price has covered half the way to SL; SKIP LEVEL printed). Skip if Q is at or beyond TP. Otherwise market order now. |
| MARKET_OR_LIMIT | Skip if Q is better than E by more than 0.50 D (SKIP LEVEL). Skip if Q is at or beyond TP. If Q is at E or better: market order now. If Q is worse than E: LIMIT order at E, expiring at the printed EXPIRY. Never chase. |
| STOP_ENTRY | Long: if ask < trigger - stops_buffer: BUY STOP at trigger with the printed EXPIRY; elif ask <= trigger + 0.20 D (CHASE LIMIT): market BUY now; else skip. Short mirrored with bid. stops_buffer = `trade_stops_level * point` from symbol_info (demo). |

Common rules:
- **Validity.** Market tickets are valid until T + 45 min (config `execution.valid_for_bars` = 3;
  owner decision O2). Limit and stop orders carry their own EXPIRY (strategy spec); the ticket's
  VALID UNTIL = min(T + 45 min, EXPIRY) for placing the order, and EXPIRY for the resting order.
- **Sizing entry.** Lots are sized on the worst acceptable entry, so actual risk never exceeds the
  printed RISK $: `sizing_entry` = CHASE LIMIT (MARKET_BAND, STOP_ENTRY) or E (MARKET_OR_LIMIT).
  Engine sizing is otherwise unchanged (0.5% of balance, floor to 0.1 lot, cap 3 lots and
  volume_max, suppress below 0.1 lot).
- **Ticket lines** (ER-01): ORDER TYPE, ENTRY or TRIGGER, CHASE LIMIT and/or SKIP LEVEL, EXPIRY,
  and the time-exit line `MANUAL TIME EXIT: close at hh:mm UTC (15:45 New York) if still open`.
- **Manual time exit** (every strategy here): at the clocks in section 2. It is flagged on every
  ticket and in the daily summary, and the engine prints an EXIT REMINDER at that time for every
  reference-open candidate (ER-03, owner decision O3). The "missed time exit" variant must also pass
  (10.5).

### 7.1 Reference occupancy (lockouts, live and backtest identical)
A candidate occupies its index (for that strategy's lockout rules and the live per-index block)
from T until the earliest of: (a) the close of the first M15 bar after T whose bid range touches SL
or TP (bid prices, no spread, no delay); (b) its time exit; (c) for a limit or stop order, its EXPIRY
if the bid range never touched the limit or trigger price by then. Every generated candidate occupies,
including suppressed ones, so lockouts are deterministic. In live use the engine also treats an index
as occupied if state.yaml lists an open position or pending order on it (ER-04).

---

## 8. Backtest conventions

### 8.1 Prices
Bars are bid. `ask = bid + eff_spread`, `eff_spread = c * max(spread_pts_bar, floor_{i,h})`, where
c = 1 (BASE) or 2 (STRESS) and h is the bar's UTC hour. The multiplier is applied to triggers and
exits as well as to cost.

### 8.2 Spread floor
`floor_{i,h}` = 90th percentile of strictly positive M15 bar spreads (points) of index i in UTC hour h,
over pre-holdout bars 2023-01-01 to 2026-04-30. Computed once by the Engineer, frozen in config
(ER-11). It is a cost constant, deliberately conservative because the MT5 bar spread is probably the
bar's minimum spread; it is never a signal input. Optional calibration (owner decision O11): compare
read-only tick ask-bid with bar spread on sample weeks; if tick spreads are materially wider, scale the
floor by the measured ratio.

### 8.3 Delay and placement
Placement time P = open of the first bar with open time >= T + d * 15 min. Scenarios d = 1, 2, 3.
The protocol rule (section 7) is evaluated with Q = ask = bid_open(P) + eff_spread (buy) or bid_open(P)
(sell). Additional skips: P after VALID UNTIL; P past a segment break or beyond the data.

### 8.4 Fills
- Market: buy at `bid_open(P) + eff_spread + slip`; sell at `bid_open(P) - slip`.
- Limit: buy limit at E fills in the first bar in [P, EXPIRY) with `bid_low + eff_spread <= E - 0.01`,
  at E, no slippage. Sell limit fills when `bid_high >= E + 0.01`, at E.
- Stop entry: buy stop triggers in the first bar in [P, EXPIRY) with `bid_high + eff_spread >= trigger`;
  fills at `max(trigger, bid_open + eff_spread) + slip`. Sell stop triggers when `bid_low <= trigger`;
  fills at `min(trigger, bid_open) - slip`.
- Unfilled at EXPIRY: cancelled, logged, no trade.
- Slippage `slip = s * max(0.5 * floor_{i,h}, 0.02 * ATR14_M15 at the signal bar)` points on market
  entries, stop entries, stop-loss exits and time exits; zero on limit entries and TP exits. s = 1
  BASE, 2 STRESS.

### 8.5 Exits
- Long SL: `bid_low <= SL`; fills at SL - slip, or at bid_open - slip if the bar opens beyond SL.
- Long TP: `bid_high >= TP + 0.01`; fills at TP exactly, including a gap through TP.
- Short SL: `bid_high + eff_spread >= SL`; fills at SL + slip, or at bid_open + eff_spread + slip on a gap.
- Short TP: `bid_low + eff_spread <= TP - 0.01`; fills at TP.
- Same bar touches SL and TP: SL first. On the fill bar (after filling at the open or at the
  trigger/limit): if the bar also reaches SL, the trade is stopped (SL first); TP on the fill bar is
  ignored unless the fill was at the bar open.
- Time exit: at the open of the first bar with open time >= the time-exit clock, sell at bid - slip
  (long) or buy at ask + slip (short). If that bar is missing, at the close of the last bar before it.
- No fills during the daily break (23:45-01:00 server time); exits pending then resolve on the first
  bar after it with the gap rule.
- Segment break with a position open: close at the last pre-gap bar close (bid or ask, with slippage),
  tagged `data_gap_exit`. End of pre-holdout data: close at the last bar close, tagged `end_of_data`.

### 8.6 Costs
- Commission: base = owner-confirmed competition commission (0 until confirmed). Stress =
  max(2 x confirmed, $1.00 per lot per side for US30/NAS100, EUR 1.00 per lot per side x 1.12 for GER40).
- Swap: a position held through the 00:00 server-time rollover is charged `swap_long`/`swap_short`
  per `swap_mode`, tripled on `swap_rollover3days` (symbol_info, read-only, ER-10). Positive swap is
  credited as 0. Stress doubles negative swap. Current demo values are applied to all history
  (limitation). Intraday strategies only pay swap in the missed-exit variant.

### 8.7 Sizing and account simulation
- `risk.sizing.size_position(sim_balance, 0.5, sizing_entry, SL, demo spec, cap 3)`.
  USD per point per lot: US30 and NAS100 1.00, GER40 1.12 (frozen demo tick value; limitation).
- Risk checks: `risk.rules.check_ticket` against a simulated AccountState (balance, equity,
  start-of-day balance, open-position risk, trades today, positions), day boundary 00:00
  Europe/Athens (owner decision O8). Suppressed tickets are logged and not traded.
- Equity is marked to market every M15 bar at the worst intrabar price (bid low for longs,
  bid high + eff_spread for shorts). Balance compounds from $100,000.
- Results are reported at the demo contract and at the owner's competition contract
  (`competition_usd_per_point`, owner decision O4).

### 8.8 Concurrency
- Single-strategy runs: at most one open or pending position per index (the strategy's own);
  further rules in each spec.
- Combined-book run (report, not a trial): at most one open or pending position per index across all
  strategies; same-T priority SES-IBX-01, then RS-MOM-01, then MR-TWAP-01.
- A candidate arriving while the scenario's own position on that index is open or pending is logged
  `blocked_open_position`.

### 8.9 Accounting
`$P&L = lots * (exit - fill) * side * USD/pt - commission - swap`. Planned risk $ =
`lots * |sizing_entry - SL| * USD/pt`. Realised R = $P&L / planned risk $. Every report shows $ and R,
per index, per side, and per exit type (SL, TP, time exit, data_gap_exit, end_of_data).

### 8.10 Determinism and logging
No randomness in the backtest. Every run writes one row to `docs/trial_log.csv` (section 11).

---

## 9. Live / backtest alignment (Engineer)
- Closed bars are selected by time, not by position: fetch from position 0 and keep bars with
  `open_time + tf <= now`. Poll until T + 60 s; an index whose M15 bar at T is still missing is
  "absent" for that cycle (and the group rule 4.2.2 applies). Freshness is checked per index
  (ER-05).
- Live history depth: at least 400 trading days of H1 and 60 trading days of M15, from the store
  updated incrementally each cycle (ER-06).
- Strategy state (one-per-session rules, lockouts, daily caps) is rebuilt by deterministic replay of
  the session's earlier bars with the model in force at each bar (section 7.1).
- CI test: `generate()` on truncated history equals the full-history backtest candidate at the same T.

---

## 10. Validation plan

### 10.1 Pre-backtest feasibility gate (descriptive; runs after features and HMM are built, before any P&L code)
For each strategy at its default parameters, over the 9 test windows (10.2): setup count (candidates
generated, no fills, no prices after T), by index, side and fold; the regime diagnostics of 6.8.
A strategy with fewer than 200 candidates in total is DROPPED without a backtest (it cannot reach the
trade minimum). This is a count of setup conditions, not an outcome.

### 10.2 Walk-forward
Rolling by calendar: train 12 months, test 3 months, step 3 months, 9 folds. Test windows:
2024-02-01..04-30, 2024-05-01..07-31, 2024-08-01..10-31, 2024-11-01..2025-01-31, 2025-02-01..04-30,
2025-05-01..07-31 (data ends 07-16), 2025-08-01..10-31 (data from 09-09; regime warm-up to about
2025-10-07), 2025-11-01..2026-01-31, 2026-02-01..04-30. The HMM refit schedule (6.3) is independent
of the folds and uses only data before each refit.
- **Run A (primary):** defaults fixed ex ante, over the concatenated test windows. Secondary sample,
  reported only: 2023-03-01 to 2026-04-30.
- **Run B (walk-forward selected):** in each train window choose the grid point from the strategy's
  one-at-a-time grid with the highest daily-return Sharpe (ties: closest to default); apply it to the
  next test window. Train trades whose exit falls after the train window end are dropped from selection
  (embargo).
- Trades belong to the fold of their signal time T (counts, R statistics). The daily series for
  Sharpe, DSR, PBO and bootstrap = end-of-day MTM equity change, day boundary 00:00 Europe/Athens,
  zero on days with no position; calendar = union of trading days of the three indices inside the
  test windows.

### 10.3 Scenarios
| Scenario | d | spread c | slip s | commission | swap | Role |
|---|---|---|---|---|---|---|
| BASE | 2 | 1 | 1 | base | 1x | report |
| STRESS-d2 | 2 | 2 | 2 | stress | 2x neg | PASS; ticket STATS come from here |
| STRESS-d3 | 3 | 2 | 2 | stress | 2x neg | PASS |
| STRESS-d1 | 1 | 2 | 2 | stress | 2x neg | report |
| MISSED-EXIT | 2 | 2 | 2 | stress | 2x neg | PASS: no time exit; bracket held until SL/TP, overnight and weekend |
| EXIT-3X | 2 | 3 on SL and time-exit fills only | 2 | stress | 2x neg | report |
Scenarios are pass/fail checks on one configuration. They never select between configurations and
are not trials.

### 10.4 Minimum trade counts (run A, STRESS-d2)
- At least 150 filled out-of-sample trades in total (SE of mean R about 0.1R at sd(R) 1.2).
- At least 4 trades per full fold in at least 7 of 9 folds, pro-rated by the share of trading days
  with data in each fold (folds 6 and 7 are cut by the data hole).
- Per-index, per-side or per-regime statistics are reported only for buckets with >= 30 trades;
  smaller buckets merge into OTHER.
- Every sensitivity cell needs >= 100 trades or it is "insufficient", which counts as a failed cell.
- Competition feasibility (for the approved combined set, reported per strategy): at least 5 distinct
  trade days in at least 80% of rolling 20-trading-day windows.

### 10.5 Pass criteria (all required; QA may add stricter ones, never weaker)
1. Run A expectancy > 0 in $ and in R, and daily-return Sharpe > 0, under STRESS-d2 and STRESS-d3.
2. MISSED-EXIT expectancy > 0 in $.
3. Run B out-of-sample expectancy > 0 in $.
4. At least 6 of 9 folds with positive $ P&L in run A (STRESS-d2).
5. Sensitivity: every immediate grid neighbour of each default has positive $ expectancy, and at
   least 80% of all grid cells (one-at-a-time and the 2-D heatmap) are positive. A flat response of a
   parameter that rarely binds (for example a TP that is rarely reached) is not robustness evidence;
   QA reports the exit-type shares next to each heatmap.
6. Stationary block bootstrap (Politis-Romano) of the daily series, mean block length 5 trading days,
   10,000 resamples, seed 12345: 5th percentile of total $ P&L > 0. Report 5/50/95% of total P&L and
   max drawdown. Regime-conditional bootstrap: resample trades within ACTIVE/CALM buckets keeping
   bucket counts; report the same quantiles.
7. Placebo (B1 below): run A total $ P&L > 95th percentile of the placebo distribution.
8. DSR >= 0.95 under both N_raw and N_eff (10.7). PBO < 0.5 (0.2-0.5 flagged).
9. Competition replay (30-calendar-day windows starting every Monday in the test span), at the demo
   contract and the competition contract: zero windows breaching the 10% total loss; 5% daily-loss
   breach in at most 1% of windows. The personal 3% stop breach frequency and the share of windows
   with >= 5 trade days are reported.
10. Shuffle test, 10,000 permutations of trade order (seed 12345): report the 95th percentile of the
    worst losing streak and of max drawdown (report only).

### 10.6 Benchmarks (stage = benchmark; never selectable; logged but not counted in N)
- **B1 placebo (pass criterion 7):** for each run-A trade, draw a random signal time uniformly from
  the same strategy window on the same local date, same index and side, same SL/TP geometry in the
  strategy's own units, same protocol and costs; 200 draws, seeds 1..200.
- **B2 random index (report):** same entry times and side, index replaced by a uniformly random other
  index whose window contains T. If the strategy does not beat B2's mean, the report must say the
  relative-strength selection adds nothing.
- **B3 gate off (report):** same rules without the regime gate. If B3 >= the gated run in $ expectancy,
  the report must say the gate adds nothing. The gate is not changed in response.

### 10.7 Trial accounting, DSR and PBO
- `docs/trial_log.csv` gets one row per configuration run on any pre-holdout data, failures included.
  A configuration = (strategy_id, full parameter vector, shared-component settings, entry protocol).
  Scenarios (10.3) of the same configuration are recorded in the same row's notes, not as new trials.
- N_raw = number of distinct configurations ever run, all strategies, including every grid cell,
  every run-B candidate and every shared-component variant.
- N_eff: hierarchical clustering (average linkage) of the trials' daily return series, cut at
  correlation 0.7; N_eff = number of clusters; V = variance of SR across cluster medoids.
- Deflated Sharpe (Bailey and Lopez de Prado 2014): SR_n = mean / std of the daily series,
  non-annualised. `SR* = sqrt(V) * ((1-g) * PhiInv(1 - 1/N) + g * PhiInv(1 - 1/(N e)))`, g = 0.5772.
  `DSR = Phi((SR_hat - SR*) * sqrt(T_days - 1) / sqrt(1 - skew*SR_hat + (kurt - 1)/4 * SR_hat^2))`,
  skew and non-excess kurtosis from the selected strategy's daily returns. If N = 1, DSR = PSR vs 0.
  Reported with N_raw (V across all trials) and N_eff (V across medoids).
- PBO by CSCV over the global trial matrix (T_days x N_raw): S = 16 contiguous blocks (drop the
  earliest remainder days), all C(16,8) = 12,870 splits, selection = best in-sample Sharpe, logit of
  its out-of-sample relative rank (rank/(N+1)); PBO = share of logits <= 0.
- Shared-component sensitivity (rank lookback, HMM window, Kmax, refit cadence; README) runs only
  for strategies that pass run A, and each variant is a trial.
- Every report states that demo prices, spreads and contracts may differ from the competition feed.
  The holdout opens once, after GATE 3 approval, with `--final-holdout`.

---

## 11. Frozen constants (not tuned; any change is a new trial and needs QA sign-off)

| Constant | Value | Reason |
|---|---|---|
| Segment break | gap > 7 days | Longest ordinary closure is 5 d 4 h; only the 2025 hole qualifies |
| ATR period | 14 (Wilder) | Wilder's convention |
| Same-time-of-day lookback | previous 20 valid trading dates, min 10 | About one trading month; robust median |
| RS_4h lookback L | 16 M15 bars | 4 hours, about half a US cash session; varied only in shared sensitivity |
| RS_day lookback | 24 H1 bars | One trading day |
| Daily-cycle bars n_d | 23 US, 22 GER40 | Median H1 bars per day |
| Winsorisation | 0.5 / 99.5 percentiles | Standard robust clip |
| HMM K set, covariance, restarts | {2,3,4}, full, seeds 0..9 | About 30+ rows per parameter on 12 months; deterministic |
| Training window / refit | 365 days / first Saturday of month 12:00 UTC | One volatility cycle; weekend refit |
| Degeneracy screens | occupancy 0.02, condition 1e6 | Reject empty or singular states |
| Min training rows | 1,000 | About 5 months of US core bars |
| Regime staleness / model age | 5 days / 62 days | Covers long holidays / two missed refits |
| Gate threshold on V_T | 0 | Neutral point of the log volatility ratio |
| Direction tag z | 2.0 | Two-sided 95% convention; logging only |
| Protocol chase / skip | 0.20 D / 0.50 D | Chase limit keeps worst-case risk within 1.2 R; skip when half the stop is used |
| Market ticket validity | 45 min (3 bars) | Maximum manual delay |
| Time exit | 15:45 New York, 17:15 Berlin | 15 min before cash close |
| Spread floor | UTC-hour p90 of positive spreads, 2023-01..2026-04 | Conservative vs bar minimum spread |
| Slippage | s x max(0.5 floor, 0.02 ATR14_M15) | Half a spread or 2% of a bar's range |

---

## 12. Descriptive inputs used (pre-holdout only; no outcomes)
- Data hole and tick-volume corruption dates: section 1.3 (daily median M15 tick volume jumps from
  about 700-2,100 to 8,000-120,000 on 2024-10-04 and returns to normal on 2025-01-07; GER40 shows 15
  per bar through December 2024).
- Volatility, spread, correlation, lot-cap and session facts from the GATE 2 drafts, as cited in each
  strategy spec.
- Setup counts for the revised specs (2023-03-01 to 2026-04-30, about 158 weeks net of the hole), with
  a volatility proxy `sign(x2)` standing in for the not-yet-built HMM gate. These are counts of setup
  conditions only and are quoted in each spec. The proxy marks 45% of RS-MOM-01 window bars ACTIVE,
  35% of IB-end bars ACTIVE and 55% of MR window bars CALM.
- Script: `C:\Users\lunar\AppData\Local\Temp\claude\D--Finance-Fundingnext-bot\6c140617-18ea-4b43-9cc9-6862d0b2cb3e\scratchpad\gate2_final_setupcount_c.py`
  and `gate2_final_desc_a.py`, `gate2_final_desc_b.py` in the same folder.

---

## 12A. Amendment A1 (2026-11-02, before any backtest; data handling only)
Found by QA during the Engineer's build, from data-quality counts only (no outcome of any rule was
computed). No strategy rule, parameter or frozen constant of section 11 changes.
1. **Frozen prices 2024-10-04 00:00 to 2025-01-08 00:00 UTC, all instruments.** Not only tick volume
   (1.3): 52-83% of in-session M15 bars have a range <= 0.5 points against about 0% elsewhere, with
   whole flat sessions. These bars are dropped at load (`config data_quality.excluded_windows`,
   `BarStore.load`). The 96-day gap they leave is a segment break (1.4), so TR, ATR, TODV, DATR, RS
   and the HMM restart after it and no lookback uses the window. Walk-forward fold 4
   (2024-11-01..2025-01-31) and part of fold 3 lose their data; the pro-rated fold minimum (10.4)
   applies, and the regime warm-up after the break follows 6.3 and 6.6.
2. **GER40 hole starts 2025-07-16 07:30 UTC**, like the US indices (1.3 said 2025-07-22 21:00). The
   stray flat evening bars from 2025-07-16 to 2025-07-22 are dropped with the hole.
3. **Spread floor minimum count (8.2):** an hour with fewer than 100 positive-spread bars takes the
   larger floor of its two neighbouring hours. Only GER40 hour 21 (5 bars) is affected: 0.5 -> 6.0.
   This makes costs more conservative, never less.

## 12B. Amendment A2 (2026-11-02, before any backtest; regime warm-up)
Found by QA in the regime build: after a segment reset (6.6) x2 is NaN for 20 x n_d bars, so the
filter sits at the stationary distribution with no emission, yet 6.6 reported status OK and the gate
acted on V_T computed from the prior alone (US30: 127 such outputs, GER40: 305). Amendment: after a
reset, status is **UNAVAILABLE ("regime warm-up")** until the first core observation after the reset
that has an emission (x1 and x2 finite, full or marginal). From then on 6.6 applies unchanged. This
follows the PROMPT rule "never a guess" and changes no constant of section 11.

## 12C. Amendment A3 (2026-11-03, Quant rulings after the feasibility gate; before any backtest)
Rulings on points the Engineers escalated. Each takes the literal reading of the pre-registered text;
none was chosen with knowledge of any outcome (only setup counts exist at this point).
1. **10.1 threshold is not pro-rated.** "Fewer than 200 candidates in total" stands as written; only
   the per-fold minimum of 10.4 is pro-rated. Lowering a pre-registered bar after a strategy misses it
   would be a researcher degree of freedom. **MR-TWAP-01 (190 candidates) is DROPPED without a
   backtest.** It is not run, not counted in N_raw, and may be re-proposed only as a new spec.
2. **4.2.1 is per exchange group.** G(T) is formed separately for the NYSE group {US30, NAS100} and the
   XETRA group {GER40}; at a T inside both windows (DST mismatch weeks) each group is selected on its own.
   This is what 4.2's honest-scope text and every strategy spec already say.
3. **An index whose date is not a valid trading date (cash-open bar missing) is outside G(T)** (1.6,
   4.2.1); the remaining member is ranked alone under 4.2.5. Inside the usable span this affects 26 US
   cycles (NAS100 on 2024-05-03 and 2026-02-26).
4. **NaN strategy inputs from missing data are group-blocking** (4.2.2), including SES-IBX-01's IB and
   DATR; a finite input that fails a filter (e.g. W > max_ib_datr x DATR, including DATR = 0) only
   removes that index from E(T).
5. The 4.2.8 per-cycle selection log is stored with each backtest run under `backtest/runs/`.

Trial budget after A3: RS-MOM-01 29 + SES-IBX-01 29 = **58 configurations** (contingent shared-component
sensitivity: at most 2 x 10 = 20; hard ceiling **78**).

## 12D. Amendment A4 (2026-11-07, Quant rulings during GATE 3 runs)
Made before any SES-IBX-01 outcome existed; RS-MOM-01 had already failed criteria 1, 4, 5, 7 and the
10.4 minimum, which neither ruling can change.
1. **Swap mode 3 (money in the margin currency)** is converted to USD with
   `costs.margin_ccy_usd` (US30 and NAS100: margin currency USD, factor 1.0; GER40: EUR, factor 1.12,
   the same frozen EUR->USD factor already used for GER40 in `usd_per_point_demo` and the stress
   commission). The deposit currency is USD. Swap only enters the MISSED-EXIT scenario for these
   intraday strategies.
2. **SES-IBX-01 B1 placebo geometry = shifted run-A bracket** (`SES_PLACEBO_SHIFT`): the run-A trigger,
   midpoint stop and target of that date are translated by the price move C' - C_T, keeping run A's
   distances to the reference price in W units, so only the timing is randomised and the T' = IB-end
   draw equals run A. Rejected alternative (`SES_PLACEBO_CLOSE`): its trigger sits 0.05 W from price,
   so far more placebo orders are skipped than in run A, which thins the placebo distribution and makes
   criterion 7 easier to pass. The stricter test is chosen.
3. RS-MOM-01's MISSED-EXIT scenario is added to the same configuration T0001 (a scenario, not a trial).

## 13. Owner decisions needed before any backtest
O1 approve the entry protocols in section 7 and the new ticket fields; O2 set `execution.valid_for_bars: 3`;
O3 confirm you can close positions at 15:45 New York and 17:15 Berlin in both DST states, and approve
the EXIT REMINDER; O4 confirm the competition contract per index (USD per point per lot), commission,
swap, and that pending stop and limit orders with expiry are supported; O5 confirm overnight and
weekend holding rules and fill `restricted_windows_utc` (FOMC, CPI, NFP, ECB); O6 drop US500 from
ranking and trading; O7 confirm "filter then rank" (default) or "rank then filter"; O8 confirm the
daily-loss day boundary (default 00:00 server time); O9 accept the tick-volume handling (flag and
marginalise) instead of dropping tick volume from the HMM; O10 optionally backfill the 2025 data hole;
O11 optionally approve read-only `copy_ticks_range` for spread calibration.

---

## 14. QA issues resolved (COMMON)
| QA issue | Resolution |
|---|---|
| Regime model unspecified (blocker) | Section 6: features, preprocessing, K grid, covariance, seeds, window, refit time and scope, filter start, relabelling, failures, logs |
| Raw-feature HMM measures the clock | Core-hours observations; x1, x3 normalised by same-hour medians; x2 uses a full-daily-cycle window |
| Directional gate is noise | No directional gate; direction tags are logging only (section 5) |
| Gate meaning drifts with K; P_MIN saturated | K-invariant gate on V_T with threshold 0; no probability parameter |
| Seed dependence | Fixed seeds 0..9, best likelihood; seed-stability diagnostic |
| 8-week data hole (blocker) | Segment rule 1.4; data_gap_exit; regime warm-up; fold pro-rating |
| Tick volume corrupt Oct 2024-Jan 2025 | `tv_bad` flag, excluded from training, 2-D marginal emission in filtering |
| Holidays misdetected | Exchange calendar 1.6 |
| Skip rule adversely selects; valid_for_bars < delay | Protocols in section 7; validity 45 min; base case is the delayed fill |
| Bid/ask cost model unspecified | Section 8: triggers by side, gap rules, fill-bar rules, stress applied to triggers |
| Bar spread is probably the minimum | p90 hourly floor, 2x stress, EXIT-3X report, optional tick calibration |
| Swap unbounded | Section 8.6, MISSED-EXIT scenario includes swap |
| Contract size unknown | Owner decision O4; results at both contracts |
| Live closed-bar race | Section 9 and 6.6 (time-based selection, poll, no silent fallback) |
| Live history depth and regime ownership | Section 9, ER-06 |
| Live concurrency differs from backtest | Reference occupancy 7.1, per-index block, pending orders in state (ER-04) |
| Rank-then-filter ambiguity | 4.2 with owner decision O7; symmetric gates remove the same-index-both-sides case |
| Ranking across sessions | Ranking only within the strategy's window set G(T) |
| Missing member changes ranking | Group rule 4.2.2 |
| Hidden degrees of freedom | Section 11 table; strategy specs list their own |
| Trial accounting / DSR too lenient | Section 10.7 (N_raw and N_eff, scenarios not trials, contingent shared sensitivity) |
| Run B leakage; daily series undefined | Embargo and MTM daily series in 10.2 |
| Minimum counts fail structurally | Pro-rated fold minimum; secondary sample |
| Beta / null benchmarks | 10.6 B1 (pass), B2, B3 |
| GER40 ranked alone / US500 | 4.2 honest scope; US500 removed (O6) |
| Session semantics conflict | Session check on T with start <= T < end (ER-12) |
| GER40 trading hours finding | Corrected: demo GER40 trades almost around the clock; freshness is per index |
