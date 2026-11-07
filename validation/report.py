"""GATE 3 validation report generator (COMMON 8.9, 10.2-10.7, PROMPT "MANDATORY VALIDATION").

Given the outputs of the runner (scenario runs, folds, run B, sensitivity grid), the stats
module (bootstrap, shuffle, DSR, PBO), the competition replay and the benchmarks for ONE
strategy, this module writes `docs/validation/<STRATEGY_ID>.md`: provenance, the scenario table,
the per-fold table with the pro-rated minimum, the per-index / side / exit-type / regime tables
under the >= 30 bucket rule, run B, sensitivity (one-at-a-time and the 2-D heatmap with trade
counts and exit-type shares next to it), bootstrap quantiles, the shuffle test, benchmarks B1-B3
with the mandated sentences, the competition replay, DSR and PBO, the COMMON 6.8 regime flags,
and a PASS/FAIL table with one row per COMMON 10.5 criterion 1-9 plus the 10.4 minimums.

Why it is a separate, dumb renderer: the report must never be a place where a decision is
made. It computes no outcome on its own except trivial aggregates of tables it is handed (bucket
means, fold sums), it never reads market data, never runs a backtest and never touches the
holdout. Every threshold is a module constant copied from COMMON 10.4 / 10.5 and is not a
parameter of any function, so a caller cannot lower a standard to get a strategy through.

Interfaces are plain frozen dataclasses defined HERE, so the runner, stats, replay and
benchmark modules (owned by other Engineers) only need to fill them; nothing here imports
their modules. For the cross-checks below it imports (lazily, read-only) core.config,
backtest.scenarios.contract_table, backtest.trials (n_raw, read_log) and the strategies'
registered ParamGrids. Helpers turn the existing backtest outputs (backtest.results trades / daily
frames) into those dataclasses.

Verdict logic: each criterion row is PASS, FAIL or NOT EVALUATED (an input is missing or, for
criterion 9, the competition contract is not confirmed). Report-only rows are INFO and never
affect the verdict. Overall: PASS only if every criterion row is PASS; FAIL if any row FAILS;
otherwise NOT PASSED (incomplete). A NOT EVALUATED row is never treated as a pass.

What the report cross-checks instead of trusting the caller (a pass needs all of it):
  - the frozen settings of every statistic (bootstrap 10,000 resamples / seed 12345 / mean
    block 5; B1 200 draws, seeds 1..200; PBO 16 blocks / 12,870 splits) equal the module
    constants, else the row is NOT EVALUATED;
  - the sensitivity grid is the strategy's pre-registered ParamGrid (strategies.<module>.PARAMS;
    29 configurations): the share's denominator is the full expected set, a missing cell is never
    a pass, and an off-grid or mismatched grid makes criterion 5 NOT EVALUATED;
  - N_raw of the DSR equals the counted configurations in docs/trial_log.csv (all strategies,
    backtest.trials.n_raw), N_eff is in 1..N_raw and PBO ran over the same N trials;
  - whether the competition contract (O4) is confirmed is read from config.yaml
    (rules.competition_usd_per_point), never from the presence of a 'competition' replay, and a
    replay at a contract other than the configured one is ignored;
  - criteria 1, 3, 4, 5, 7 read STRESS-d2 only; the fold set must be exactly folds 1..9;
  - run A is the registered defaults (10.2 "defaults fixed ex ante"): for a strategy with a
    registered grid, Provenance.params must equal ParamGrid.defaults, else every run-A row
    (10.4 minimums, 1, 2, 4, 6, 7, 8) is NOT EVALUATED and a real report is refused;
  - run B covers exactly folds 1..9, every selection is a cell of the one-at-a-time grid with
    its embargo count supplied, and the per-fold test trades / P&L sum to the run-B totals;
  - the stationary bootstrap is of the STRESS-d2 daily series (scenario STRESS-d2, n_obs = the
    run's n_days); the DSR's SR_hat and T_days equal that run's daily Sharpe and n_days; the
    regime-conditional bootstrap's buckets are ACTIVE / CALM and sum to the STRESS-d2 trades;
  - the trades table, the STRESS-d2 trade count and the fold trade counts agree, else the 10.4
    rows and criterion 4 are NOT EVALUATED; no trade in the table is in the holdout;
  - a real (non-synthetic) report needs data_end < 2026-05-01, at least one trial-log row id
    (all present in the log), config.yaml and docs/trial_log.csv from disk, and a strategy with a
    registered grid; MR-TWAP-01 (dropped, A3) is refused.

Public API
    Status constants PASS, FAIL, NOT_EVALUATED, INFO; thresholds (see "Thresholds" below)
    Provenance, ScenarioStats, FoldStats, RunBSelection, RunBStats, SensitivityCell,
    Sensitivity, BootstrapStats, ShuffleStats, BenchmarkStats, ReplayStats, DsrStats, PboStats,
    RegimeDiagnostics, ValidationInputs, CriterionResult           (input / output dataclasses)
    scenario_stats(name, trades, daily, equity=None, start_balance=100_000.0) -> ScenarioStats
    daily_pnl_series(daily, start_balance=100_000.0) -> np.ndarray
    fold_stats(trades, shares, folds=TEST_FOLDS) -> list[FoldStats]   (raises on a filled trade
                                      outside every test window)
    bucket_table(trades, by, min_n=MIN_BUCKET_TRADES) -> pd.DataFrame
    exit_type_shares(trades) -> dict[str, float]
    regime_labels(trades) -> pd.Series  ACTIVE / CALM label per filled trade (feed this to
                                      validation.stats.regime_bootstrap)
    pair_days(trades) -> set[date];  pair_day_summary(trades) -> dict    (SES-IBX-01)
    rolling_trade_day_share(trade_dates, trading_dates, window=20, min_days=5) -> float
    bootstrap_from(res, scenario), shuffle_from(res), dsr_from(rep), pbo_from(res)
                                      adapters from validation.stats results (duck-typed)
    sensitivity_summary(sens, strategy_id=None) -> dict
                                      expected / run / missing cells, positive share over the
                                      expected set, neighbour results, problems, status
    registered_grid(strategy_id) -> ParamGrid | None
    contract_from_config(cfg, which) -> dict | None;  competition_confirmed(cfg) -> bool
    logged_n_raw(inp) -> int            N_raw from docs/trial_log.csv (backtest.trials.n_raw)
    parse_fold_shares(feasibility_md) -> dict[int, float]
    parse_regime_flags(feasibility_md, strategy_id) -> RegimeDiagnostics
    evaluate_criteria(inp) -> list[CriterionResult]
    overall_verdict(rows) -> str
    render_report(inp) -> str
    write_report(inp, out_dir=None) -> Path

SPEC NOTEs (most literal reading; also listed in every generated report):
 1. Pro-rated fold minimum (10.4, A1): a fold meets it when trades >= 4 x data share, with no
    rounding (for integer counts this equals trades >= ceil(4 x share)). A fold with data share
    0 has minimum 0. Criterion 4 (>= 6 of 9 folds with positive $ P&L) is NOT pro-rated: a fold
    with no trade has P&L 0, which is not positive.
 2. Bucket rule (10.4): applied to the per-index, per-side, per-exit-type and per-regime tables.
    Buckets with < 30 trades merge into OTHER; if OTHER itself has < 30 trades only its count is
    shown and its $ / R statistics are withheld.
 3. Immediate grid neighbours (10.5.5): for each parameter, the grid values adjacent to its
    default with every other parameter at its default (the one-at-a-time neighbours); in the
    2-D heatmap, the 4 cells one step from the default cell along one axis (diagonals differ
    in two parameters and are not immediate neighbours). "All grid cells" = the distinct
    configurations of the one-at-a-time grid and the heatmap (29 per spec), each counted once.
    A cell with < 100 trades is "insufficient" and counts as failed (10.4), also as a neighbour.
 4. B2 "mean" (10.6): the mean $ P&L per trade of the B2 runs (pooled over draws), compared
    with the strategy's $ expectancy in the same scenario. "Does not beat" = strategy <= B2.
 5. Criterion 9 needs the competition contract (owner decision O4). While it is not confirmed
    the replay is shown at the demo contract only and criterion 9 is NOT EVALUATED (FAIL if the
    demo replay already fails), so the overall verdict cannot be PASS until O4 is confirmed.
 6. 10.4 competition feasibility (>= 5 trade days in >= 80% of rolling 20-trading-day windows)
    applies to the approved combined set; per strategy it is reported (INFO), not a verdict row.
 7. Criteria 1, 4, 6, 7 and the 10.4 minimums use run A under STRESS-d2 (10.4 names STRESS-d2;
    criterion 1 also STRESS-d3). The daily Sharpe of the helper scenario_stats is
    mean / std (ddof=1) of the end-of-day equity change, non-annualised (10.7).
 8. An undefined DSR or PBO (NaN, e.g. PBO with one trial) makes criterion 8 NOT EVALUATED.
 9. N_raw must EQUAL the trial log's counted configurations at render time (10.7: "all
    configurations ever run"); a later trial makes an earlier DSR stale, so a re-render then
    shows NOT EVALUATED until the DSR / PBO are recomputed over the full log.
10. Max DD % is the largest drawdown as a share of the running peak, computed independently of
    the largest $ drawdown (the two can occur at different times). backtest.runner.max_drawdown
    (the trial log's max_dd_pct) takes the % AT the largest $ drawdown, as a negative number, so
    its magnitude can be smaller than the report's; the report's is the one compared with the
    %-based competition limits.
11. Pair days (SES-IBX-01 "QA reports pair days separately"): a New York local date on which the
    strategy issued candidates on both US30 and NAS100 with opposite sides (any status: the
    tickets were issued whether or not they filled). Pair-day legs and the other filled trades
    are two buckets under the 30-trade rule.
12. Run B evidence (10.2): each selection must be a cell of the one-at-a-time grid (the default
    plus each parameter's other values), its embargoed-train-trade count must be supplied (0 is
    valid), and sum(test trades) / sum(test P&L) must equal the run-B totals.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "docs" / "validation"

# ---------------------------------------------------------------------------------------------
# Status values and thresholds (COMMON 10.4, 10.5; never function parameters)
# ---------------------------------------------------------------------------------------------

PASS, FAIL, NOT_EVALUATED, INFO = "PASS", "FAIL", "NOT EVALUATED", "INFO"
VERDICT_PASS, VERDICT_FAIL, VERDICT_INCOMPLETE = "PASS", "FAIL", "NOT PASSED (incomplete)"

PRIMARY_SCENARIO = "STRESS-d2"                 # 10.3: ticket STATS and 10.4 counts come from here
SCENARIO_ORDER = ("BASE", "STRESS-d1", "STRESS-d2", "STRESS-d3", "MISSED-EXIT", "EXIT-3X")
CRITERION_1_SCENARIOS = ("STRESS-d2", "STRESS-d3")
MISSED_EXIT = "MISSED-EXIT"

MIN_TOTAL_TRADES = 150                         # 10.4
MIN_TRADES_PER_FULL_FOLD = 4                   # 10.4 (pro-rated by data share, A1)
MIN_FOLDS_MEETING_MINIMUM = 7                  # 10.4: "in at least 7 of 9 folds"
MIN_BUCKET_TRADES = 30                         # 10.4 bucket rule
MIN_SENSITIVITY_CELL_TRADES = 100              # 10.4: else "insufficient" = failed cell
MIN_POSITIVE_FOLDS = 6                         # 10.5.4
MIN_POSITIVE_CELL_SHARE = 0.80                 # 10.5.5
DSR_MIN = 0.95                                 # 10.5.8
PBO_MAX = 0.5                                  # 10.5.8 (PBO < 0.5)
PBO_FLAG = 0.2                                 # 10.5.8: 0.2-0.5 flagged
REPLAY_MAX_TOTAL_LOSS_WINDOWS = 0              # 10.5.9: zero windows breaching 10% total loss
REPLAY_MAX_DAILY_BREACH_SHARE = 0.01           # 10.5.9: 5% daily-loss breach in <= 1% of windows
TRADE_DAY_WINDOW, TRADE_DAY_MIN, TRADE_DAY_SHARE = 20, 5, 0.80   # 10.4 competition feasibility
HOLDOUT_START = date(2026, 5, 1)               # COMMON 1.2

# Frozen statistic settings (COMMON 10.5.6, 10.5.10, 10.6, 10.7); a result computed with any
# other setting is not evidence for the criterion.
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 12345
BOOTSTRAP_MEAN_BLOCK = 5.0
BOOTSTRAP_METHOD = "stationary_block"
SHUFFLE_PERMUTATIONS = 10_000
SHUFFLE_SEED = 12345
B1_DRAWS = 200
B1_SEEDS = tuple(range(1, 201))
PBO_BLOCKS = 16
PBO_SPLITS = math.comb(16, 8)                  # 12,870

# Strategies with a pre-registered sensitivity grid (strategies.<module>.PARAMS). MR-TWAP-01 is
# dropped (A3) and never reported.
GRID_MODULES = {"RS-MOM-01": "strategies.rs_mom_01", "SES-IBX-01": "strategies.ses_ibx_01"}
DROPPED = ("MR-TWAP-01",)

N_FOLDS = 9
FOLD_NUMBERS = tuple(range(1, N_FOLDS + 1))
OTHER = "OTHER"
OUTSIDE_FOLD = -1                              # backtest.folds.OUTSIDE: T in no test window
REGIME_BUCKETS = ("ACTIVE", "CALM")            # 10.5.6 regime-conditional buckets
US_PAIR = ("US30", "NAS100")                   # SES-IBX-01 US window set (pair days)
PAIR_TZ = "America/New_York"
PAIR_STRATEGY = "SES-IBX-01"
# Rows that are statements about run A (STRESS-d2 / d3 / MISSED-EXIT at the defaults).
RUN_A_ROWS = ("10.4 trades", "10.4 folds", "1", "2", "4", "6", "7", "8")
COUNT_ROWS = ("10.4 trades", "10.4 folds", "4")

DEMO_NOTE = ("Demo prices, spreads and contracts may differ from the competition feed: every "
             "number in this report comes from the MT5 DEMO feed and the demo contract.")
HOLDOUT_NOTE = ("The final holdout (bars from 2026-05-01 00:00 UTC) is untouched: no data on or "
                "after that date was loaded for this report (final_holdout was never passed). It "
                "opens once, after the owner approves GATE 3.")
COMPETITION_UNCONFIRMED = ("The competition contract (owner decision O4: USD per point per lot, "
                           "commission, swap) is NOT confirmed; the replay below is at the demo "
                           "contract only.")
COMPETITION_IGNORED = ("A replay labelled 'competition' was supplied although config.yaml "
                       "rules.competition_usd_per_point is not set (O4 not confirmed): it is "
                       "shown as UNCONFIRMED and ignored by criterion 9.")
SENTENCE_RS = "the relative-strength selection adds nothing"
SENTENCE_GATE = "the gate adds nothing"

SPEC_NOTES = (
    "Pro-rated fold minimum: a fold meets it when trades >= 4 x data share (no rounding); a fold "
    "with data share 0 has minimum 0. Criterion 4 is not pro-rated: a fold without trades is not "
    "positive.",
    "Bucket rule: per-index, per-side, per-exit-type and per-regime buckets with < 30 trades merge "
    "into OTHER; if OTHER has < 30 trades its statistics are withheld.",
    "Immediate grid neighbours: the adjacent grid values of each parameter with the others at "
    "default, and the 4 axis-adjacent cells of the default in the heatmap; all grid cells = the "
    "distinct configurations of both grids, each counted once; insufficient (< 100 trades) cells "
    "count as failed.",
    "B2 mean: mean $ P&L per trade of the B2 runs (pooled over draws), compared with the "
    "strategy's $ expectancy in the same scenario; the strategy must exceed it.",
    "Criterion 9 needs the competition contract (O4); while it is not confirmed the criterion is "
    "NOT EVALUATED (FAIL if the demo replay fails) and the overall verdict cannot be PASS.",
    "10.4 competition feasibility applies to the approved combined set; per strategy it is "
    "reported only.",
    "Criteria 1, 4, 6, 7 and the 10.4 minimums use run A under STRESS-d2 (criterion 1 also "
    "STRESS-d3); run B, sensitivity and benchmarks must also be STRESS-d2.",
    "An undefined DSR or PBO (NaN) makes criterion 8 NOT EVALUATED.",
    "N_raw must equal the counted configurations in docs/trial_log.csv at render time (all "
    "strategies); otherwise criterion 8 is NOT EVALUATED until DSR / PBO are recomputed.",
    "Max DD % is the largest drawdown relative to the running peak, independent of the largest "
    "$ drawdown; the trial log's max_dd_pct is measured at the largest $ drawdown (negative "
    "sign) and can be smaller in magnitude.",
    "Run A must be the registered default parameters; any other vector makes every run-A row "
    "NOT EVALUATED.",
    "Run B: selections for exactly folds 1..9, each a one-at-a-time grid cell with its embargo "
    "count; per-fold test trades and P&L must sum to the run-B totals.",
    "Criterion 6 uses the stationary bootstrap of the STRESS-d2 daily series (n_obs = its "
    "days); criterion 8 needs the DSR's SR_hat and T_days to equal the STRESS-d2 daily Sharpe "
    "and days; the regime-conditional buckets must be ACTIVE / CALM.",
    "The trades table, the STRESS-d2 trade count and the fold trade counts must agree; "
    "otherwise the 10.4 rows and criterion 4 are NOT EVALUATED.",
    "Pair days (SES-IBX-01): New York dates with issued US30 and NAS100 candidates of opposite "
    "sides (any status); pair-day legs vs other filled trades under the 30-trade rule.",
)

# ---------------------------------------------------------------------------------------------
# Input dataclasses (filled by runner / stats / replay / benchmarks)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """Where the numbers came from. data_end is the last bar date used (must be < 2026-05-01).
    data_source is printed verbatim; anything other than real pre-holdout data is bannered."""
    strategy_id: str
    git_sha: str
    data_hash: str
    config_hash: str
    params: Mapping[str, Any]
    data_source: str = "MT5 DEMO, BarStore.load pre-holdout"
    data_end: date | None = None
    trial_ids: Sequence[str] = ()
    generated_utc: datetime | None = None
    final_holdout_used: bool = False
    synthetic: bool = False
    notes: Sequence[str] = ()


@dataclass(frozen=True)
class ScenarioStats:
    """Run A over the concatenated test windows under one COMMON 10.3 scenario. n_days = length
    of the daily series (runner metrics 'n_days'); win_rate = share of filled trades with
    $ P&L > 0 (the ticket STATS 'out-of-sample win rate' comes from STRESS-d2, 10.3)."""
    name: str
    trades: int
    expectancy_usd: float
    expectancy_r: float
    daily_sharpe: float
    total_usd: float
    max_dd_usd: float
    max_dd_pct: float = float("nan")
    n_days: int | None = None
    win_rate: float = float("nan")


@dataclass(frozen=True)
class FoldStats:
    """One walk-forward test window of run A (STRESS-d2); trades belong to the fold of T."""
    n: int
    window: str
    data_share: float
    trades: int
    pnl_usd: float
    expectancy_r: float = float("nan")

    @property
    def prorated_minimum(self) -> float:
        # An unknown share counts as a full fold (the conservative reading).
        return MIN_TRADES_PER_FULL_FOLD * _num(self.data_share, 1.0)

    @property
    def meets_minimum(self) -> bool:
        return self.trades >= self.prorated_minimum

    @property
    def positive(self) -> bool:
        return _finite(self.pnl_usd) and self.pnl_usd > 0


@dataclass(frozen=True)
class RunBSelection:
    """One fold of run B: the one-at-a-time grid point chosen in the train window, the number
    of filled train trades the embargo dropped for it (runner run_b table 'n_embargoed'; None =
    not supplied, which makes criterion 3 NOT EVALUATED), and its test-window trades / P&L."""
    fold: int
    params: Mapping[str, Any]
    train_sharpe: float
    test_trades: int
    test_pnl_usd: float
    n_embargoed: int | None = None


@dataclass(frozen=True)
class RunBStats:
    """Walk-forward selected run (10.2 run B), out-of-sample, one scenario (STRESS-d2)."""
    selections: Sequence[RunBSelection]
    trades: int
    expectancy_usd: float
    expectancy_r: float
    total_usd: float
    daily_sharpe: float
    scenario: str = PRIMARY_SCENARIO


@dataclass(frozen=True)
class SensitivityCell:
    """One grid configuration, run A STRESS-d2. params = the full parameter vector."""
    params: Mapping[str, float]
    trades: int
    expectancy_usd: float
    expectancy_r: float = float("nan")
    exit_shares: Mapping[str, float] = field(default_factory=dict)

    @property
    def sufficient(self) -> bool:
        return self.trades >= MIN_SENSITIVITY_CELL_TRADES

    @property
    def passes(self) -> bool:
        return self.sufficient and _finite(self.expectancy_usd) and self.expectancy_usd > 0


@dataclass(frozen=True)
class Sensitivity:
    """All grid cells (one-at-a-time and heatmap, duplicates allowed: they are merged by params).
    heatmap_x / heatmap_y name the two heatmap parameters (k x R; ib_minutes x tp_ib_mult).
    grid = {parameter: grid values}; for a strategy in GRID_MODULES the registered ParamGrid is
    authoritative and a supplied grid must equal it; for any other id it is required."""
    defaults: Mapping[str, float]
    cells: Sequence[SensitivityCell]
    heatmap_x: str
    heatmap_y: str
    scenario: str = PRIMARY_SCENARIO
    grid: Mapping[str, Sequence[float]] | None = None


@dataclass(frozen=True)
class BootstrapStats:
    """Quantiles {5: ., 50: ., 95: .} of total $ P&L and of max drawdown $. scenario = the run
    whose series was resampled; n_obs = its length (days for the stationary bootstrap, trades
    for the regime-conditional one). Both are needed to tie the result to run A STRESS-d2."""
    total_pnl_q: Mapping[int, float]
    max_dd_q: Mapping[int, float]
    n_resamples: int = 10_000
    seed: int = 12345
    mean_block_length: float | None = 5.0
    bucket_counts: Mapping[str, int] = field(default_factory=dict)   # regime-conditional only
    method: str = BOOTSTRAP_METHOD               # 'stationary_block' | 'regime_conditional'
    scenario: str | None = None
    n_obs: int | None = None


@dataclass(frozen=True)
class ShuffleStats:
    worst_streak_p95: float
    max_dd_p95_usd: float
    n_permutations: int = 10_000
    seed: int = 12345


@dataclass(frozen=True)
class BenchmarkStats:
    """B1-B3 (10.6) in one scenario; run A values are read from that scenario's row."""
    scenario: str = PRIMARY_SCENARIO
    b1_draws: int = 200
    b1_p95_total_usd: float = float("nan")
    b1_mean_total_usd: float = float("nan")
    b2_mean_usd: float = float("nan")
    b2_trades: int = 0
    b3_expectancy_usd: float = float("nan")
    b3_trades: int = 0
    b3_total_usd: float = float("nan")
    b1_seeds: Sequence[int] | None = None         # must be 1..200 (10.6) for criterion 7


@dataclass(frozen=True)
class ReplayStats:
    """Competition-rule replay (10.5.9), 30-calendar-day windows starting every Monday."""
    contract: str                     # "demo" or "competition"
    usd_per_point: Mapping[str, float] | None
    n_windows: int
    total_loss_breach_windows: int
    daily_loss_breach_windows: int
    personal_stop_breach_windows: int
    windows_ge5_trade_days: int

    def share(self, n: int) -> float:
        return n / self.n_windows if self.n_windows else float("nan")


@dataclass(frozen=True)
class DsrStats:
    sr_hat: float
    t_days: int
    skew: float
    kurtosis: float
    n_raw: int
    dsr_raw: float
    v_raw: float
    n_eff: int
    dsr_eff: float
    v_eff: float


@dataclass(frozen=True)
class PboStats:
    pbo: float
    n_trials: int
    n_splits: int = 12_870
    blocks: int = 16


@dataclass(frozen=True)
class RegimeDiagnostics:
    """COMMON 6.8 gate flags, from the feasibility gate (docs/feasibility_gate.md)."""
    gate: str
    block_share: float
    flag: str
    per_index: Mapping[str, tuple[float, str]] = field(default_factory=dict)
    non_functional_folds: Sequence[int] = ()
    source: str = "docs/feasibility_gate.md"

    @property
    def functional(self) -> bool:
        return self.flag.strip().lower() == "functional"


@dataclass(frozen=True)
class ValidationInputs:
    """Everything the report needs for one strategy. trades = the run A STRESS-d2 trades table
    (backtest.results TRADE_COLUMNS; only status 'filled' rows are used)."""
    provenance: Provenance
    scenarios: Sequence[ScenarioStats] = ()
    folds: Sequence[FoldStats] = ()
    trades: pd.DataFrame | None = None
    run_b: RunBStats | None = None
    sensitivity: Sensitivity | None = None
    bootstrap: BootstrapStats | None = None
    regime_bootstrap: BootstrapStats | None = None
    shuffle: ShuffleStats | None = None
    benchmarks: BenchmarkStats | None = None
    replay: Sequence[ReplayStats] = ()
    dsr: DsrStats | None = None
    pbo: PboStats | None = None
    regime: RegimeDiagnostics | None = None
    trade_day_share: float | None = None
    secondary: ScenarioStats | None = None      # 2023-03-01..2026-04-30, reported only
    spec_notes: Sequence[str] = ()
    # Synthetic reports only: a config dict / trial-log path instead of config.yaml and
    # docs/trial_log.csv. A real report always reads both from disk (ValueError otherwise).
    config: Mapping[str, Any] | None = None
    trial_log: Path | str | None = None

    def scenario(self, name: str) -> ScenarioStats | None:
        for s in self.scenarios:
            if s.name == name:
                return s
        return None

    def replay_for(self, contract: str) -> ReplayStats | None:
        for r in self.replay:
            if r.contract == contract:
                return r
        return None


@dataclass(frozen=True)
class CriterionResult:
    criterion: str
    description: str
    measured: str
    threshold: str
    status: str


# ---------------------------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------------------------

def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _num(x: Any, default: float = float("nan")) -> float:
    return float(x) if _finite(x) else default


def _usd(x: Any) -> str:
    return f"{float(x):+,.2f}" if _finite(x) else "n/a"


def _usd0(x: Any) -> str:
    return f"{float(x):,.2f}" if _finite(x) else "n/a"


def _r(x: Any) -> str:
    return f"{float(x):+.3f}R" if _finite(x) else "n/a"


def _f(x: Any, nd: int = 3) -> str:
    return f"{float(x):.{nd}f}" if _finite(x) else "n/a"


def _pct(x: Any, nd: int = 1) -> str:
    return f"{100 * float(x):.{nd}f}%" if _finite(x) else "n/a"


def _cell(x: Any) -> str:
    return str(x).replace("|", "/").replace("\n", " ")


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(_cell(h) for h in header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _params(p: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={_pv(v)}" for k, v in p.items()) if p else "n/a"


def _pv(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return f"{v:g}"
    return f"{v:g}" if isinstance(v, float) else str(v)


def _kv(v: Any) -> Any:
    try:
        return round(float(v), 9)
    except (TypeError, ValueError):
        return str(v)


def _key(params: Mapping[str, Any]) -> tuple:
    return tuple(sorted((k, _kv(v)) for k, v in params.items()))


def _in_scope(trades: pd.DataFrame) -> pd.DataFrame:
    """Candidate rows the runner counts: a 'fold' column of OUTSIDE_FOLD (T in no test window,
    e.g. a bar-pad candidate) is out of scope, as in backtest.runner.ConfigRun.filled."""
    if "fold" in trades.columns:
        return trades[pd.to_numeric(trades["fold"], errors="coerce") != OUTSIDE_FOLD]
    return trades


def _filled(trades: pd.DataFrame | None) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=["key", "side", "exit_type", "pnl_usd", "r_multiple",
                                     "regime", "signal_time"])
    t = _in_scope(trades)
    if "status" in t.columns:
        return t[t["status"] == "filled"]
    return t


# ---------------------------------------------------------------------------------------------
# Helpers from backtest outputs
# ---------------------------------------------------------------------------------------------

def daily_pnl_series(daily: pd.DataFrame, start_balance: float = 100_000.0) -> np.ndarray:
    """End-of-day MTM equity change (10.2), the first day measured from the start balance."""
    eq = pd.to_numeric(daily["equity"], errors="coerce").to_numpy(dtype="float64")
    if len(eq) == 0:
        return eq
    return np.diff(eq, prepend=start_balance)


def _max_dd(equity: np.ndarray, start_balance: float) -> tuple[float, float]:
    """(largest $ drawdown, largest drawdown as a share of the running peak). The two maxima are
    taken independently: after a big run-up a larger $ drawdown can be a smaller %, and the
    competition limits are % based (SPEC NOTE 10)."""
    if len(equity) == 0:
        return 0.0, 0.0
    path = np.concatenate([[start_balance], equity])
    peak = np.maximum.accumulate(path)
    dd = peak - path
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(peak > 0, dd / peak, np.nan)
    pct = float(np.nanmax(rel)) if np.isfinite(rel).any() else float("nan")
    return float(np.nanmax(dd)), pct


def scenario_stats(name: str, trades: pd.DataFrame, daily: pd.DataFrame,
                   equity: pd.DataFrame | None = None,
                   start_balance: float = 100_000.0) -> ScenarioStats:
    """ScenarioStats from a backtest.results.BacktestResult's frames. Max drawdown uses the
    worst-intrabar MTM equity (8.7) when `equity` is given, else the end-of-day equity."""
    f = _filled(trades)
    pnl = pd.to_numeric(f["pnl_usd"], errors="coerce")
    r = pd.to_numeric(f["r_multiple"], errors="coerce")
    d = daily_pnl_series(daily, start_balance)
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
    sharpe = float(np.mean(d)) / sd if _finite(sd) and sd > 0 else float("nan")
    curve = equity["equity"] if equity is not None and len(equity) else daily["equity"]
    dd_usd, dd_pct = _max_dd(pd.to_numeric(curve, errors="coerce").to_numpy(dtype="float64"),
                             start_balance)
    return ScenarioStats(name=name, trades=int(len(f)),
                         expectancy_usd=float(pnl.mean()) if len(f) else float("nan"),
                         expectancy_r=float(r.mean()) if len(f) else float("nan"),
                         daily_sharpe=sharpe, total_usd=float(pnl.sum()),
                         max_dd_usd=dd_usd, max_dd_pct=dd_pct, n_days=int(len(d)),
                         win_rate=float((pnl > 0).mean()) if len(f) else float("nan"))


def fold_stats(trades: pd.DataFrame, shares: Mapping[int, float],
               folds: Sequence[Any] | None = None) -> list[FoldStats]:
    """Per-fold rows from run A STRESS-d2 trades; a trade belongs to the fold of its signal time
    T (10.2). `folds` defaults to backtest.feasibility.TEST_FOLDS (objects with n, label,
    contains(t)). A filled trade (fold column, if any, not OUTSIDE) whose T is in no window
    raises: dropping it silently would make the fold counts disagree with the run."""
    if folds is None:
        from backtest.feasibility import TEST_FOLDS
        folds = TEST_FOLDS
    f = _filled(trades)
    t = pd.to_datetime(f["signal_time"], utc=True) if len(f) else pd.Series([], dtype=object)
    masks = {fo.n: (np.array([fo.contains(x) for x in t], dtype=bool) if len(f)
                    else np.array([], bool)) for fo in folds}
    if len(f):
        hit = np.logical_or.reduce(list(masks.values())) if masks else np.zeros(len(f), bool)
        if not hit.all():
            raise ValueError(f"{int((~hit).sum())} filled trade(s) with signal time outside "
                             f"every test window, first at {t[~hit].min()}")
    out = []
    for fo in folds:
        mask = masks[fo.n]
        sub = f[mask] if len(f) else f
        pnl = pd.to_numeric(sub["pnl_usd"], errors="coerce")
        out.append(FoldStats(n=fo.n, window=fo.label, data_share=_num(shares.get(fo.n)),
                             trades=int(len(sub)), pnl_usd=float(pnl.sum()),
                             expectancy_r=float(pd.to_numeric(sub["r_multiple"],
                                                              errors="coerce").mean())
                             if len(sub) else float("nan")))
    return out


def _regime_bucket(x: Any) -> str:
    s = str(x) if x is not None and not (isinstance(x, float) and math.isnan(x)) else "n/a"
    return s.split("/", 1)[0] or "n/a"


def bucket_table(trades: pd.DataFrame, by: str, min_n: int = MIN_BUCKET_TRADES) -> pd.DataFrame:
    """$ and R per bucket with the 10.4 rule: buckets with < min_n trades merge into OTHER.
    by = 'key', 'side', 'exit_type' or 'regime' (regime buckets on ACTIVE/CALM, the part of
    Candidate.regime before '/'). Columns: bucket, n, share, total_usd, mean_usd, mean_r,
    merged, withheld. min_n is the 10.4 constant; it is a parameter only for unit tests."""
    f = _filled(trades).copy()
    cols = ["bucket", "n", "share", "total_usd", "mean_usd", "mean_r", "merged", "withheld"]
    if len(f) == 0:
        return pd.DataFrame(columns=cols)
    f["_b"] = f[by].map(_regime_bucket) if by == "regime" else f[by].fillna("n/a").astype(str)
    f["_pnl"] = pd.to_numeric(f["pnl_usd"], errors="coerce")
    f["_r"] = pd.to_numeric(f["r_multiple"], errors="coerce")
    n_all = len(f)
    counts = f["_b"].value_counts()
    big = sorted(b for b, n in counts.items() if n >= min_n)
    small = sorted(b for b, n in counts.items() if n < min_n)
    rows = []
    for b in big:
        g = f[f["_b"] == b]
        rows.append(dict(bucket=b, n=len(g), share=len(g) / n_all, total_usd=g["_pnl"].sum(),
                         mean_usd=g["_pnl"].mean(), mean_r=g["_r"].mean(), merged="",
                         withheld=False))
    if small:
        g = f[f["_b"].isin(small)]
        withheld = len(g) < min_n
        nan = float("nan")
        rows.append(dict(bucket=OTHER, n=len(g), share=len(g) / n_all,
                         total_usd=nan if withheld else g["_pnl"].sum(),
                         mean_usd=nan if withheld else g["_pnl"].mean(),
                         mean_r=nan if withheld else g["_r"].mean(),
                         merged=", ".join(small), withheld=withheld))
    return pd.DataFrame(rows, columns=cols)


def exit_type_shares(trades: pd.DataFrame) -> dict[str, float]:
    """Share of filled trades per exit type (counts, not outcome statistics; 10.5.5)."""
    f = _filled(trades)
    if len(f) == 0:
        return {}
    vc = f["exit_type"].fillna("n/a").astype(str).value_counts(normalize=True)
    return {str(k): float(v) for k, v in vc.items()}


def regime_labels(trades: pd.DataFrame) -> pd.Series:
    """The 10.5.6 bucket (ACTIVE / CALM, the part of Candidate.regime before '/') of every
    filled trade, in table order. Pass these, not the raw labels, to
    validation.stats.regime_bootstrap: it buckets by the label string, and the raw label
    ('ACTIVE/ACTIVE_UP') would bucket per HMM state instead of within ACTIVE / CALM."""
    return _filled(trades)["regime"].map(_regime_bucket)


def pair_days(trades: pd.DataFrame) -> set[date]:
    """SES-IBX-01 pair days (SPEC NOTE 11): New York dates with issued candidates (any status)
    on both US30 and NAS100 with opposite sides."""
    if trades is None or len(trades) == 0:
        return set()
    t = _in_scope(trades)
    t = t[t["key"].isin(US_PAIR)]
    if len(t) == 0:
        return set()
    d = pd.to_datetime(t["signal_time"], utc=True).dt.tz_convert(PAIR_TZ).dt.date
    g = pd.DataFrame({"d": d.to_numpy(), "key": t["key"].to_numpy(),
                      "side": t["side"].to_numpy()}).groupby("d")
    return {k for k, x in g if set(x["key"]) == set(US_PAIR) and set(x["side"]) == {"BUY", "SELL"}}


def pair_day_summary(trades: pd.DataFrame) -> dict[str, Any]:
    """Pair days, pair days with both legs filled, and the bucket table of filled pair-day legs
    vs every other filled trade (30-trade rule)."""
    days = pair_days(trades)
    f = _filled(trades).copy()
    if len(f):
        d = pd.to_datetime(f["signal_time"], utc=True).dt.tz_convert(PAIR_TZ).dt.date
        on = d.isin(days) & f["key"].isin(US_PAIR)
        f["day_type"] = np.where(on, "pair-day leg", "single ticket")
        both = sum(1 for x in days if (d[on] == x).sum() >= 2)
    else:
        f["day_type"] = pd.Series(dtype=object)
        both = 0
    return dict(n_pair_days=len(days), n_both_filled=both,
                table=bucket_table(f, "day_type"))


def rolling_trade_day_share(trade_dates: Iterable[date], trading_dates: Sequence[date],
                            window: int = TRADE_DAY_WINDOW, min_days: int = TRADE_DAY_MIN) -> float:
    """Share of rolling `window`-trading-day windows (over the sorted trading calendar) that
    contain at least `min_days` distinct trade days (10.4 competition feasibility)."""
    cal = sorted(set(trading_dates))
    if len(cal) < window:
        return float("nan")
    traded = set(trade_dates)
    hit = np.array([d in traded for d in cal], dtype="int64")
    csum = np.concatenate([[0], np.cumsum(hit)])
    per_window = csum[window:] - csum[:-window]
    return float(np.mean(per_window >= min_days))


# ---------------------------------------------------------------------------------------------
# Adapters from validation.stats results (duck-typed: only attribute names are relied on, so
# this module never imports the stats module and cannot be broken by its internals)
# ---------------------------------------------------------------------------------------------

def _qmap(q: Any) -> dict[int, float]:
    return {5: float(q.p5), 50: float(q.p50), 95: float(q.p95)}


def bootstrap_from(res: Any, scenario: str) -> BootstrapStats:
    """From validation.stats.BootstrapResult (stationary or regime-conditional). `scenario` =
    the run whose daily series / trades were resampled (must be STRESS-d2 for criterion 6);
    n_obs is copied so the report can tie the result to that run's length."""
    return BootstrapStats(total_pnl_q=_qmap(res.total_pnl), max_dd_q=_qmap(res.max_drawdown),
                          n_resamples=int(res.n_resamples), seed=int(res.seed),
                          mean_block_length=res.mean_block,
                          bucket_counts=dict(res.buckets or {}),
                          method=str(res.method), scenario=str(scenario),
                          n_obs=int(res.n_obs))


def shuffle_from(res: Any) -> ShuffleStats:
    """From validation.stats.ShuffleResult."""
    return ShuffleStats(worst_streak_p95=float(res.worst_streak.p95),
                        max_dd_p95_usd=float(res.max_drawdown.p95),
                        n_permutations=int(res.n_permutations), seed=int(res.seed))


def dsr_from(rep: Any) -> DsrStats:
    """From validation.stats.DSRReport (raw / eff DSRResult)."""
    raw, eff = rep.raw, rep.eff
    return DsrStats(sr_hat=float(raw.sr_hat), t_days=int(raw.t_days), skew=float(raw.skew),
                    kurtosis=float(raw.kurtosis), n_raw=int(raw.n_trials), dsr_raw=float(raw.dsr),
                    v_raw=float(raw.sr_variance), n_eff=int(eff.n_trials),
                    dsr_eff=float(eff.dsr), v_eff=float(eff.sr_variance))


def pbo_from(res: Any) -> PboStats:
    """From validation.stats.PBOResult."""
    return PboStats(pbo=float(res.pbo), n_trials=int(res.n_trials), n_splits=int(res.n_splits),
                    blocks=int(res.n_blocks))


# ---------------------------------------------------------------------------------------------
# Parsers for the descriptive feasibility report (no outcome information in it)
# ---------------------------------------------------------------------------------------------

def _md_rows(lines: Sequence[str]) -> list[list[str]]:
    rows = []
    for ln in lines:
        s = ln.strip()
        if not s.startswith("|") or set(s.replace("|", "").strip()) <= {"-", " "}:
            continue
        rows.append([c.strip() for c in s.strip("|").split("|")])
    return rows


def _share(s: str) -> float:
    m = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)\s*%", s)
    return float(m.group(1)) / 100 if m else float("nan")


def parse_fold_shares(feasibility_md: str) -> dict[int, float]:
    """Fold -> data share from the 'Test windows and data coverage' table."""
    lines = feasibility_md.splitlines()
    try:
        i = next(k for k, ln in enumerate(lines) if ln.startswith("## Test windows"))
    except StopIteration:
        raise ValueError("feasibility report has no 'Test windows' section") from None
    block = []
    for ln in lines[i + 1:]:
        if ln.startswith("## "):
            break
        block.append(ln)
    rows = _md_rows(block)
    if not rows or rows[0][:1] != ["Fold"]:
        raise ValueError("fold coverage table not found")
    col = rows[0].index("Data share")
    return {int(r[0]): _share(r[col]) for r in rows[1:] if r and r[0].isdigit()}


def parse_regime_flags(feasibility_md: str, strategy_id: str) -> RegimeDiagnostics:
    """COMMON 6.8 flags of one strategy: the verdict row (gate, ALL block share, flag), the
    per-index rows and the folds flagged NON-FUNCTIONAL in the strategy's section."""
    lines = feasibility_md.splitlines()
    verdict = None
    for r in _md_rows(lines):
        if r and r[0] == strategy_id and len(r) >= 7:
            verdict = r
            break
    if verdict is None:
        raise ValueError(f"{strategy_id} not in the feasibility verdict table")
    try:
        start = next(k for k, ln in enumerate(lines) if ln.strip() == f"## {strategy_id}")
    except StopIteration:
        raise ValueError(f"no '## {strategy_id}' section") from None
    section = []
    for ln in lines[start + 1:]:
        if ln.startswith("## "):
            break
        section.append(ln)
    per_index: dict[str, tuple[float, str]] = {}
    nf_folds: list[int] = []
    for r in _md_rows(section):
        if len(r) < 7 or r[0] in ("Index", "Fold", "ALL"):
            continue
        if r[0].isdigit():
            if "NON-FUNCTIONAL" in r[6]:
                nf_folds.append(int(r[0]))
        elif "%" in r[5]:
            per_index[r[0]] = (_share(r[5]), r[6])
    return RegimeDiagnostics(gate=verdict[2], block_share=_share(verdict[5]), flag=verdict[6],
                             per_index=per_index, non_functional_folds=tuple(nf_folds))


# ---------------------------------------------------------------------------------------------
# Cross-checks against the repository (config.yaml, the trial log, the registered grids)
# ---------------------------------------------------------------------------------------------

def _config(inp: ValidationInputs) -> Mapping[str, Any]:
    """config.yaml from disk; a supplied dict is accepted for synthetic reports only, so a real
    report cannot claim a contract the owner has not confirmed."""
    if inp.config is not None:
        if not inp.provenance.synthetic:
            raise ValueError("a real report reads config.yaml itself; `config` is for synthetic "
                             "reports only")
        return inp.config
    from core.config import load_config
    return load_config()


def contract_from_config(cfg: Mapping[str, Any], which: str) -> dict[str, float] | None:
    """USD per point per lot by index from the config: 'demo' (costs.usd_per_point_demo) or
    'competition' (rules.competition_usd_per_point; None while O4 is not confirmed)."""
    from backtest.scenarios import contract_table
    try:
        return contract_table(dict(cfg), which)
    except ValueError:
        return None


def competition_confirmed(cfg: Mapping[str, Any]) -> bool:
    """O4 is confirmed only when config rules.competition_usd_per_point is set."""
    return contract_from_config(cfg, "competition") is not None


def _trial_log_path(inp: ValidationInputs) -> Path:
    from backtest.trials import TRIAL_LOG
    if inp.trial_log is None:
        return TRIAL_LOG
    p = Path(inp.trial_log)
    if not inp.provenance.synthetic and p.resolve() != Path(TRIAL_LOG).resolve():
        raise ValueError(f"a real report counts trials in {TRIAL_LOG}, not {p}")
    return p


def logged_n_raw(inp: ValidationInputs) -> int:
    """N_raw from the trial log (backtest.trials.n_raw: distinct counted configurations, all
    strategies)."""
    from backtest.trials import n_raw
    return n_raw(_trial_log_path(inp))


def registered_grid(strategy_id: str) -> Any | None:
    """The strategy's pre-registered ParamGrid (strategies.<module>.PARAMS), or None."""
    mod = GRID_MODULES.get(strategy_id)
    if mod is None:
        return None
    import importlib
    return importlib.import_module(mod).PARAMS


def _same_contract(a: Mapping[str, float] | None, b: Mapping[str, float] | None) -> bool:
    if a is None or b is None or set(a) != set(b):
        return False
    return all(math.isclose(float(a[k]), float(b[k]), rel_tol=1e-9, abs_tol=1e-12) for k in a)


# ---------------------------------------------------------------------------------------------
# Criteria (COMMON 10.4, 10.5)
# ---------------------------------------------------------------------------------------------

def _status(ok: bool | None) -> str:
    return NOT_EVALUATED if ok is None else (PASS if ok else FAIL)


def _gt0(x: Any) -> bool:
    return _finite(x) and float(x) > 0


def _vals(xs: Iterable[Any]) -> tuple[float, ...]:
    return tuple(sorted({round(float(v), 9) for v in xs}))


def _resolve_grid(sens: Sensitivity, strategy_id: str | None
                  ) -> tuple[dict[str, tuple[float, ...]], list[str]]:
    """The expected grid {parameter: values} and the reasons it cannot be trusted. The
    registered ParamGrid wins; a supplied grid must equal it."""
    problems: list[str] = []
    reg = registered_grid(strategy_id) if strategy_id else None
    supplied = ({k: _vals(v) for k, v in sens.grid.items()} if sens.grid is not None else None)
    if reg is not None:
        grid = {k: _vals(v) for k, v in reg.grid.items()}
        if supplied is not None and supplied != grid:
            problems.append(f"supplied grid differs from the registered {strategy_id} grid")
        if _key(sens.defaults) != _key(reg.defaults):
            problems.append(f"defaults {_params(sens.defaults)} differ from the registered "
                            f"{_params(reg.defaults)}")
        if reg.heatmap is not None and {sens.heatmap_x, sens.heatmap_y} != set(reg.heatmap):
            problems.append(f"heatmap {sens.heatmap_x} x {sens.heatmap_y} is not the registered "
                            f"{reg.heatmap[0]} x {reg.heatmap[1]}")
    elif supplied is not None:
        grid = supplied
    else:
        grid = {p: _vals(c.params[p] for c in sens.cells if p in c.params) for p in sens.defaults}
        problems.append("no registered or supplied grid: the expected cell set is unknown")
    if set(grid) != set(sens.defaults):
        problems.append("grid parameters differ from the default parameters")
    for p, dv in sens.defaults.items():
        if p in grid and round(float(dv), 9) not in grid[p]:
            problems.append(f"default {p}={_pv(dv)} is not in its grid")
    if sens.heatmap_x not in grid or sens.heatmap_y not in grid:
        problems.append("heatmap parameters are not grid parameters")
    if sens.scenario != PRIMARY_SCENARIO:
        problems.append(f"scenario {sens.scenario}, not {PRIMARY_SCENARIO}")
    return grid, problems


def _expected_cells(defaults: Mapping[str, Any], grid: Mapping[str, tuple[float, ...]],
                    x: str, y: str) -> dict[tuple, dict[str, float]]:
    """Every pre-registered configuration once: one-at-a-time values and the x by y heatmap."""
    d = {k: round(float(v), 9) for k, v in defaults.items()}
    out: dict[tuple, dict[str, float]] = {}
    for p in d:
        for v in grid.get(p, ()):
            pv = dict(d)
            pv[p] = v
            out.setdefault(_key(pv), pv)
    if x in grid and y in grid:
        for xv in grid[x]:
            for yv in grid[y]:
                pv = dict(d)
                pv[x], pv[y] = xv, yv
                out.setdefault(_key(pv), pv)
    return out


def _neighbour_keys(defaults: Mapping[str, Any], grid: Mapping[str, tuple[float, ...]]
                    ) -> list[tuple]:
    """See SPEC NOTE 3: the adjacent grid values of each parameter, the others at default."""
    d = {k: round(float(v), 9) for k, v in defaults.items()}
    out: list[tuple] = []
    for p, dv in d.items():
        vals = grid.get(p, ())
        if dv not in vals:
            continue
        i = vals.index(dv)
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                nb = dict(d)
                nb[p] = vals[j]
                k = _key(nb)
                if k not in out:
                    out.append(k)
    return out


def sensitivity_summary(sens: Sensitivity, strategy_id: str | None = None) -> dict[str, Any]:
    """Criterion 5 over the full pre-registered cell set (29 per spec). The share's
    denominator is the number of EXPECTED cells: a missing cell is never a pass. Status: NOT
    EVALUATED when the grid cannot be trusted (problems); FAIL when the criterion fails even if
    every missing cell passed; NOT EVALUATED when cells are missing; else PASS / FAIL."""
    grid, problems = _resolve_grid(sens, strategy_id)
    supplied: dict[tuple, SensitivityCell] = {}
    for c in sens.cells:
        supplied.setdefault(_key(c.params), c)
    expected = _expected_cells(sens.defaults, grid, sens.heatmap_x, sens.heatmap_y)
    cells = {k: supplied[k] for k in expected if k in supplied}
    missing = [k for k in expected if k not in supplied]
    unexpected = [k for k in supplied if k not in expected]
    if unexpected:
        problems.append(f"{len(unexpected)} cell(s) off the registered grid: "
                        + "; ".join(_params(dict(k)) for k in unexpected))
    n_exp = len(expected)
    n_pass = sum(c.passes for c in cells.values())
    neighbours = [(k, cells.get(k)) for k in _neighbour_keys(sens.defaults, grid)]
    nb_failed = any(c is not None and not c.passes for _, c in neighbours)
    nb_ok = bool(neighbours) and all(c is not None and c.passes for _, c in neighbours)
    share = n_pass / n_exp if n_exp else float("nan")
    best = (n_pass + len(missing)) / n_exp if n_exp else float("nan")
    if problems or not n_exp:
        status = NOT_EVALUATED
    elif nb_failed or best < MIN_POSITIVE_CELL_SHARE:
        status = FAIL                       # fails whatever the missing cells would show
    elif missing:
        status = NOT_EVALUATED
    else:
        status = _status(nb_ok and share >= MIN_POSITIVE_CELL_SHARE)
    return dict(grid=grid, problems=problems, expected=expected, cells=cells, missing=missing,
                unexpected=unexpected, n_expected=n_exp, n_cells=len(cells), n_pass=n_pass,
                n_insufficient=sum(not c.sufficient for c in cells.values()), share=share,
                neighbours=neighbours, neighbours_ok=nb_ok, status=status)


def _bootstrap_problems(b: BootstrapStats, method: str,
                        prim: ScenarioStats | None) -> list[str]:
    """Frozen settings, and the tie to run A STRESS-d2: the stationary bootstrap resamples its
    n_days daily values, the regime-conditional one its filled trades in ACTIVE / CALM
    buckets."""
    out = []
    if b.method != method:
        out.append(f"method {b.method}, not {method}")
    if b.n_resamples != BOOTSTRAP_RESAMPLES:
        out.append(f"{b.n_resamples} resamples, not {BOOTSTRAP_RESAMPLES}")
    if b.seed != BOOTSTRAP_SEED:
        out.append(f"seed {b.seed}, not {BOOTSTRAP_SEED}")
    if method == BOOTSTRAP_METHOD and not (
            b.mean_block_length is not None and _finite(b.mean_block_length)
            and math.isclose(float(b.mean_block_length), BOOTSTRAP_MEAN_BLOCK)):
        out.append(f"mean block {b.mean_block_length}, not {_pv(BOOTSTRAP_MEAN_BLOCK)}")
    if b.scenario != PRIMARY_SCENARIO:
        out.append(f"resampled series is {b.scenario or 'unlabelled'}, not run A "
                   f"{PRIMARY_SCENARIO}")
    if method == BOOTSTRAP_METHOD:
        need = prim.n_days if prim is not None else None
        if b.n_obs is None or need is None or b.n_obs != need:
            out.append(f"n_obs {b.n_obs} != {need} days of the {PRIMARY_SCENARIO} daily series")
    else:
        need = prim.trades if prim is not None else None
        if b.n_obs is not None and b.n_obs != need:
            out.append(f"n_obs {b.n_obs} != {need} {PRIMARY_SCENARIO} filled trades")
        labels = set(b.bucket_counts)
        if not labels or not labels <= set(REGIME_BUCKETS):
            out.append(f"buckets {sorted(labels)} are not within {'/'.join(REGIME_BUCKETS)} "
                       "(use regime_labels(trades))")
        if sum(int(v) for v in b.bucket_counts.values()) != need:
            out.append(f"bucket counts sum to {sum(int(v) for v in b.bucket_counts.values())}, "
                       f"not the {need} {PRIMARY_SCENARIO} filled trades")
    return out


def _shuffle_problems(s: ShuffleStats) -> list[str]:
    out = []
    if s.n_permutations != SHUFFLE_PERMUTATIONS:
        out.append(f"{s.n_permutations} permutations, not {SHUFFLE_PERMUTATIONS}")
    if s.seed != SHUFFLE_SEED:
        out.append(f"seed {s.seed}, not {SHUFFLE_SEED}")
    return out


def _benchmark_problems(bm: BenchmarkStats) -> list[str]:
    out = []
    if bm.scenario != PRIMARY_SCENARIO:
        out.append(f"run under {bm.scenario}, not {PRIMARY_SCENARIO}")
    if bm.b1_draws != B1_DRAWS:
        out.append(f"{bm.b1_draws} B1 draws, not {B1_DRAWS}")
    if bm.b1_seeds is None:
        out.append("B1 seeds not supplied")
    elif tuple(sorted(int(s) for s in bm.b1_seeds)) != B1_SEEDS:
        out.append("B1 seeds are not 1..200")
    return out


def _trial_problems(d: DsrStats, p: PboStats, logged: int) -> list[str]:
    out = []
    if d.n_raw != logged:
        out.append(f"N_raw {d.n_raw} != {logged} counted configurations in the trial log")
    if not 1 <= d.n_eff <= d.n_raw:
        out.append(f"N_eff {d.n_eff} outside 1..N_raw")
    if p.n_trials != d.n_raw:
        out.append(f"PBO over {p.n_trials} trials, not N_raw {d.n_raw}")
    if p.blocks != PBO_BLOCKS:
        out.append(f"PBO S = {p.blocks}, not {PBO_BLOCKS}")
    if p.n_splits != PBO_SPLITS:
        out.append(f"PBO {p.n_splits} splits, not {PBO_SPLITS}")
    return out


def _run_a_problems(inp: ValidationInputs) -> list[str]:
    """10.2: run A uses the defaults fixed ex ante. A registered strategy's Provenance.params
    must equal its ParamGrid.defaults; a grid cell presented as run A is selection leakage."""
    reg = registered_grid(inp.provenance.strategy_id)
    if reg is None:
        return []
    if _key(inp.provenance.params) != _key(reg.defaults):
        return [f"run A parameters {_params(inp.provenance.params)} are not the registered "
                f"defaults {_params(reg.defaults)}"]
    return []


def _oat_keys(inp: ValidationInputs) -> set[tuple] | None:
    """Keys of the one-at-a-time grid (default + each parameter's other values): the registered
    ParamGrid, else (synthetic, unregistered ids) the supplied sensitivity grid; None if
    unknown."""
    reg = registered_grid(inp.provenance.strategy_id)
    if reg is not None:
        return {_key(p) for p in reg.one_at_a_time()}
    sens = inp.sensitivity
    if sens is None or sens.grid is None:
        return None
    out = set()
    for p, vals in sens.grid.items():
        for v in vals:
            out.add(_key({**dict(sens.defaults), p: v}))
    return out


def _run_b_problems(inp: ValidationInputs) -> list[str]:
    """Run B is evidence only if it covers folds 1..9, every selection is a one-at-a-time grid
    point with its embargo count, and the fold rows add up to the totals (SPEC NOTE 12)."""
    rb = inp.run_b
    assert rb is not None
    out = []
    if rb.scenario != PRIMARY_SCENARIO:
        out.append(f"run under {rb.scenario}, not {PRIMARY_SCENARIO}")
    folds = sorted(s.fold for s in rb.selections)
    if tuple(folds) != FOLD_NUMBERS:
        out.append(f"selections for folds {folds} (need exactly 1..{N_FOLDS})")
    oat = _oat_keys(inp)
    if oat is None:
        out.append("one-at-a-time grid unknown: selections cannot be checked")
    else:
        off = [s for s in rb.selections if _key(s.params) not in oat]
        if off:
            out.append("selection(s) off the one-at-a-time grid: "
                       + "; ".join(f"fold {s.fold} {_params(s.params)}" for s in off))
    no_emb = [s.fold for s in rb.selections if s.n_embargoed is None]
    if no_emb:
        out.append(f"embargo count not supplied for fold(s) {sorted(no_emb)}")
    n = sum(int(s.test_trades) for s in rb.selections)
    if n != rb.trades:
        out.append(f"fold test trades sum to {n}, not the run-B total {rb.trades}")
    pnl = sum(float(s.test_pnl_usd) for s in rb.selections)
    if not (_finite(pnl) and _finite(rb.total_usd)
            and math.isclose(pnl, float(rb.total_usd), rel_tol=1e-9, abs_tol=0.01)):
        out.append(f"fold test P&L sums to {_usd(pnl)}, not the run-B total {_usd(rb.total_usd)}")
    return out


def _count_problems(inp: ValidationInputs, prim: ScenarioStats | None) -> list[str]:
    """The trades table, the STRESS-d2 count and the fold counts must describe the same run."""
    out = []
    if prim is None:
        return out
    if inp.trades is not None and len(_filled(inp.trades)) != prim.trades:
        out.append(f"trades table has {len(_filled(inp.trades))} filled trades, "
                   f"{PRIMARY_SCENARIO} has {prim.trades}")
    if inp.folds and sum(f.trades for f in inp.folds) != prim.trades:
        out.append(f"fold trades sum to {sum(f.trades for f in inp.folds)}, {PRIMARY_SCENARIO} "
                   f"has {prim.trades}")
    return out


def _dsr_tie_problems(d: DsrStats, prim: ScenarioStats | None) -> list[str]:
    """10.7: the DSR is of the selected strategy's (run A STRESS-d2) daily series."""
    if prim is None:
        return [f"no {PRIMARY_SCENARIO} run to tie SR_hat to"]
    out = []
    if not (_finite(d.sr_hat) and _finite(prim.daily_sharpe)
            and math.isclose(d.sr_hat, prim.daily_sharpe, rel_tol=1e-6, abs_tol=1e-12)):
        out.append(f"SR_hat {_f(d.sr_hat, 4)} != {PRIMARY_SCENARIO} daily Sharpe "
                   f"{_f(prim.daily_sharpe, 4)}")
    if prim.n_days is None or d.t_days != prim.n_days:
        out.append(f"T_days {d.t_days} != {prim.n_days} {PRIMARY_SCENARIO} days")
    return out


def _downgrade(rows: list[CriterionResult], crits: Sequence[str],
               problems: Sequence[str]) -> list[CriterionResult]:
    """Rows whose evidence is not about the right run become NOT EVALUATED (never PASS, and a
    FAIL measured on the wrong run is no evidence either)."""
    if not problems:
        return rows
    tag = f" [NOT EVALUATED: {'; '.join(problems)}]"
    return [replace(r, measured=r.measured + tag, status=NOT_EVALUATED)
            if r.criterion in crits and r.status != INFO else r for r in rows]


def _folds_complete(folds: Sequence[FoldStats]) -> bool:
    return tuple(sorted(f.n for f in folds)) == FOLD_NUMBERS


def _fold_numbers(folds: Sequence[FoldStats]) -> str:
    return (f"fold numbers {sorted(f.n for f in folds)} supplied (need exactly 1..{N_FOLDS})"
            if folds else "no folds supplied")


def _not_frozen(problems: Sequence[str]) -> str:
    return f" [NOT EVALUATED: {'; '.join(problems)}]" if problems else ""


def _replay_rows(inp: ValidationInputs, cfg: Mapping[str, Any]
                 ) -> list[tuple[str, ReplayStats | None, str]]:
    """(contract, the replay usable for criterion 9 or None, why not) for demo and competition.
    A replay is usable only at the contract config.yaml defines for it; a 'competition' replay
    while O4 is not confirmed is ignored."""
    out = []
    for contract in ("demo", "competition"):
        table = contract_from_config(cfg, contract)
        r = inp.replay_for(contract)
        if contract == "competition" and table is None:
            why = "contract NOT confirmed (O4)" + (
                "; the supplied 'competition' replay is ignored" if r is not None else "")
            out.append((contract, None, why))
        elif r is None:
            out.append((contract, None, "not run"))
        elif table is None:
            out.append((contract, None, f"no {contract} contract in config.yaml to check against"))
        elif not _same_contract(r.usd_per_point, table):
            out.append((contract, None, f"replay contract {r.usd_per_point} differs from config "
                                        f"{table}: ignored"))
        else:
            out.append((contract, r, ""))
    return out


def evaluate_criteria(inp: ValidationInputs) -> list[CriterionResult]:
    rows: list[CriterionResult] = []
    add = rows.append
    folds_ok = _folds_complete(inp.folds)

    # 10.4 minimums ---------------------------------------------------------------------------
    prim = inp.scenario(PRIMARY_SCENARIO)
    add(CriterionResult(
        "10.4 trades", f"Filled out-of-sample trades, run A {PRIMARY_SCENARIO}",
        str(prim.trades) if prim else "not run", f">= {MIN_TOTAL_TRADES}",
        _status(None if prim is None else prim.trades >= MIN_TOTAL_TRADES)))
    if folds_ok:
        met = sum(f.meets_minimum for f in inp.folds)
        add(CriterionResult(
            "10.4 folds", "Folds with >= 4 trades per full fold (pro-rated by data share)",
            f"{met} of {N_FOLDS}", f">= {MIN_FOLDS_MEETING_MINIMUM} of {N_FOLDS}",
            _status(met >= MIN_FOLDS_MEETING_MINIMUM)))
    else:
        add(CriterionResult("10.4 folds", "Folds with >= 4 trades per full fold (pro-rated)",
                            _fold_numbers(inp.folds), f">= {MIN_FOLDS_MEETING_MINIMUM} "
                            f"of {N_FOLDS}", NOT_EVALUATED))

    # 1 --------------------------------------------------------------------------------------
    parts, oks = [], []
    for name in CRITERION_1_SCENARIOS:
        s = inp.scenario(name)
        if s is None:
            parts.append(f"{name}: not run")
            oks.append(None)
            continue
        parts.append(f"{name}: {_usd(s.expectancy_usd)} $/trade, {_r(s.expectancy_r)}, "
                     f"Sharpe {_f(s.daily_sharpe, 4)}")
        oks.append(_gt0(s.expectancy_usd) and _gt0(s.expectancy_r) and _gt0(s.daily_sharpe))
    ok1 = False if False in oks else (None if None in oks else True)
    add(CriterionResult("1", "Run A expectancy > 0 in $ and R, daily Sharpe > 0, STRESS-d2 and "
                        "STRESS-d3", "; ".join(parts), "> 0 each", _status(ok1)))

    # 2 --------------------------------------------------------------------------------------
    me = inp.scenario(MISSED_EXIT)
    add(CriterionResult("2", "MISSED-EXIT expectancy > 0 in $",
                        f"{_usd(me.expectancy_usd)} $/trade" if me else "not run", "> 0",
                        _status(None if me is None else _gt0(me.expectancy_usd))))

    # 3 --------------------------------------------------------------------------------------
    rb = inp.run_b
    if rb is None:
        add(CriterionResult("3", "Run B out-of-sample expectancy > 0 in $", "not run", "> 0",
                            NOT_EVALUATED))
    else:
        bad = _run_b_problems(inp)
        add(CriterionResult("3", f"Run B out-of-sample expectancy > 0 in $ ({PRIMARY_SCENARIO})",
                            f"{_usd(rb.expectancy_usd)} $/trade ({rb.trades} trades)"
                            + _not_frozen(bad), "> 0",
                            NOT_EVALUATED if bad else _status(_gt0(rb.expectancy_usd))))

    # 4 --------------------------------------------------------------------------------------
    if folds_ok:
        pos = sum(f.positive for f in inp.folds)
        add(CriterionResult("4", f"Folds with positive $ P&L, run A {PRIMARY_SCENARIO}",
                            f"{pos} of {N_FOLDS}", f">= {MIN_POSITIVE_FOLDS} of {N_FOLDS}",
                            _status(pos >= MIN_POSITIVE_FOLDS)))
    else:
        add(CriterionResult("4", "Folds with positive $ P&L", _fold_numbers(inp.folds),
                            f">= {MIN_POSITIVE_FOLDS} of {N_FOLDS}", NOT_EVALUATED))

    # 5 --------------------------------------------------------------------------------------
    if inp.sensitivity is None or not inp.sensitivity.cells:
        add(CriterionResult("5", "Sensitivity: neighbours positive, >= 80% of cells positive",
                            "not run", "all neighbours > 0; >= 80%", NOT_EVALUATED))
    else:
        ss = sensitivity_summary(inp.sensitivity, inp.provenance.strategy_id)
        n_nb_ok = sum(c is not None and c.passes for _, c in ss["neighbours"])
        miss = f", {len(ss['missing'])} missing" if ss["missing"] else ""
        add(CriterionResult(
            "5", "Sensitivity: every immediate neighbour of each default has $ expectancy > 0 "
            "(>= 100 trades); >= 80% of all grid cells positive",
            f"neighbours {n_nb_ok} of {len(ss['neighbours'])} pass; cells {ss['n_pass']} of "
            f"{ss['n_expected']} positive ({_pct(ss['share'])}), {ss['n_insufficient']} "
            f"insufficient{miss}" + _not_frozen(ss["problems"]),
            f"all neighbours; >= {_pct(MIN_POSITIVE_CELL_SHARE, 0)} of {ss['n_expected']}",
            ss["status"]))

    # 6 --------------------------------------------------------------------------------------
    bs = inp.bootstrap
    if bs is None:
        add(CriterionResult("6", "Stationary block bootstrap: 5th percentile of total $ P&L > 0",
                            "not run", "> 0", NOT_EVALUATED))
    else:
        p5 = bs.total_pnl_q.get(5)
        bad = _bootstrap_problems(bs, BOOTSTRAP_METHOD, prim)
        add(CriterionResult("6", "Stationary block bootstrap of the run A "
                            f"{PRIMARY_SCENARIO} daily series (10,000 resamples, mean block 5, "
                            "seed 12345): 5th percentile of total $ P&L > 0",
                            _usd(p5) + _not_frozen(bad), "> 0",
                            NOT_EVALUATED if bad else _status(_gt0(p5))))

    # 7 --------------------------------------------------------------------------------------
    bm = inp.benchmarks
    if bm is None or prim is None or not _finite(bm.b1_p95_total_usd):
        add(CriterionResult("7", "Run A total $ P&L > 95th percentile of placebo B1",
                            "not run", "> B1 p95", NOT_EVALUATED))
    else:
        bad = _benchmark_problems(bm)
        add(CriterionResult("7", f"Run A total $ P&L > 95th percentile of placebo B1 "
                            f"({PRIMARY_SCENARIO}, 200 draws, seeds 1..200)",
                            f"{_usd(prim.total_usd)} vs B1 p95 {_usd(bm.b1_p95_total_usd)}"
                            + _not_frozen(bad), "> B1 p95",
                            NOT_EVALUATED if bad else
                            _status(_finite(prim.total_usd)
                                    and prim.total_usd > bm.b1_p95_total_usd)))

    # 8 --------------------------------------------------------------------------------------
    d, p = inp.dsr, inp.pbo
    if d is None or p is None:
        add(CriterionResult("8", "DSR >= 0.95 (N_raw and N_eff), PBO < 0.5",
                            "not run", "DSR >= 0.95; PBO < 0.5", NOT_EVALUATED))
    else:
        logged = logged_n_raw(inp)
        bad = _trial_problems(d, p, logged) + _dsr_tie_problems(d, prim)
        vals = (d.dsr_raw, d.dsr_eff, p.pbo)
        ok8 = (None if bad or not all(_finite(v) for v in vals)   # undefined: never a pass
               else d.dsr_raw >= DSR_MIN and d.dsr_eff >= DSR_MIN and p.pbo < PBO_MAX)
        flag = " (flagged: 0.2-0.5)" if _finite(p.pbo) and PBO_FLAG <= p.pbo < PBO_MAX else ""
        add(CriterionResult("8", "DSR >= 0.95 under N_raw and N_eff; PBO < 0.5",
                            f"DSR {_f(d.dsr_raw)} (N_raw {d.n_raw}; trial log {logged}), "
                            f"{_f(d.dsr_eff)} (N_eff {d.n_eff}); PBO {_f(p.pbo)}{flag}"
                            + _not_frozen(bad),
                            f"DSR >= {DSR_MIN}; PBO < {PBO_MAX}", _status(ok8)))

    # 9 --------------------------------------------------------------------------------------
    parts9, oks9 = [], []
    for contract, r, why in _replay_rows(inp, _config(inp)):
        if r is None:
            parts9.append(f"{contract}: {why}")
            oks9.append(None)
            continue
        dshare = r.share(r.daily_loss_breach_windows)
        parts9.append(f"{contract}: {r.total_loss_breach_windows} total-loss windows, daily-loss "
                      f"{_pct(dshare, 2)} of {r.n_windows}")
        oks9.append(r.n_windows > 0
                    and r.total_loss_breach_windows <= REPLAY_MAX_TOTAL_LOSS_WINDOWS
                    and _finite(dshare) and dshare <= REPLAY_MAX_DAILY_BREACH_SHARE)
    ok9 = False if False in oks9 else (None if None in oks9 else True)
    add(CriterionResult("9", "Competition replay at the demo AND the competition contract",
                        "; ".join(parts9), "0 total-loss windows; daily-loss <= 1% of windows",
                        _status(ok9)))

    # Report-only rows ------------------------------------------------------------------------
    sh = inp.shuffle
    add(CriterionResult("10 (report)", "Shuffle test: p95 worst losing streak and max DD",
                        f"streak {_f(sh.worst_streak_p95, 1)}, DD {_usd0(sh.max_dd_p95_usd)}"
                        + (f" [settings not frozen: {'; '.join(_shuffle_problems(sh))}]"
                           if _shuffle_problems(sh) else "")
                        if sh else "not run", "report only", INFO))
    add(CriterionResult("10.4 (report)", "Trade days: share of rolling 20-trading-day windows "
                        "with >= 5 trade days (applies to the combined set)",
                        _pct(inp.trade_day_share) if inp.trade_day_share is not None
                        else "not run", f">= {_pct(TRADE_DAY_SHARE, 0)} (combined set)", INFO))
    rows = _downgrade(rows, COUNT_ROWS, _count_problems(inp, prim))
    return _downgrade(rows, RUN_A_ROWS, _run_a_problems(inp))


def overall_verdict(rows: Sequence[CriterionResult]) -> str:
    judged = [r for r in rows if r.status != INFO]
    if judged and all(r.status == PASS for r in judged):
        return VERDICT_PASS
    if any(r.status == FAIL for r in judged):
        return VERDICT_FAIL
    return VERDICT_INCOMPLETE


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------

def _check_provenance(inp: ValidationInputs) -> None:
    """The report always states the holdout is untouched and names its trials; refuse inputs
    that contradict that or that cannot prove it."""
    p = inp.provenance
    if p.strategy_id in DROPPED:
        raise ValueError(f"{p.strategy_id} is dropped (COMMON 12C, A3) and is never reported")
    if p.final_holdout_used:
        raise ValueError("final_holdout was used: GATE 3 reports must not touch the holdout")
    if p.data_end is not None and p.data_end >= HOLDOUT_START:
        raise ValueError(f"data_end {p.data_end} is inside the holdout (from {HOLDOUT_START})")
    if inp.trades is not None and len(inp.trades):
        # Every row, whatever its status or fold: a holdout row means holdout bars were loaded.
        hs = pd.Timestamp(HOLDOUT_START, tz="UTC")
        for col in ("signal_time", "fill_time", "exit_time"):
            if col in inp.trades.columns:
                t = pd.to_datetime(inp.trades[col], utc=True, errors="coerce")
                if (t >= hs).any():
                    raise ValueError(f"trades table has a row with {col} >= {HOLDOUT_START} "
                                     "(holdout)")
    if p.synthetic:
        return
    bad = _run_a_problems(inp)
    if bad:
        raise ValueError(f"{bad[0]} (COMMON 10.2: run A uses the defaults fixed ex ante)")
    if p.data_end is None:
        raise ValueError("a real report needs data_end (the last bar date, < 2026-05-01)")
    if p.strategy_id not in GRID_MODULES:
        raise ValueError(f"{p.strategy_id} has no registered parameter grid; runnable: "
                         f"{sorted(GRID_MODULES)}")
    if not p.trial_ids:
        raise ValueError("a real report needs the trial-log row ids it is built from")
    from backtest.trials import read_log
    logged = set(read_log(_trial_log_path(inp))["trial_id"])
    unknown = [t for t in p.trial_ids if t not in logged]
    if unknown:
        raise ValueError(f"trial ids {unknown} are not in the trial log")


def _scenario_section(inp: ValidationInputs) -> list[str]:
    order = {n: i for i, n in enumerate(SCENARIO_ORDER)}
    sc = sorted(inp.scenarios, key=lambda s: order.get(s.name, len(order)))
    out = ["## Scenarios (run A, concatenated test windows; COMMON 10.3)", ""]
    if not sc:
        return out + ["Not run.", ""]
    rows = [(s.name, s.trades, _pct(s.win_rate), _usd(s.expectancy_usd), _r(s.expectancy_r),
             _f(s.daily_sharpe, 4), s.n_days if s.n_days is not None else "n/a",
             _usd(s.total_usd), _usd0(s.max_dd_usd), _pct(s.max_dd_pct, 2)) for s in sc]
    out.append(_table(["Scenario", "Trades", "Win rate", "Expectancy $", "Expectancy R",
                       "Daily Sharpe", "Days", "Total $", "Max DD $", "Max DD %"], rows))
    out += ["", "Win rate = share of filled trades with $ P&L > 0. Daily Sharpe = mean / std of "
            "the end-of-day MTM equity change over 'Days' calendar days, non-annualised. Max DD % "
            "= the largest drawdown relative to the running peak (it can occur at a different "
            "time than the largest $ drawdown); docs/trial_log.csv's max_dd_pct is measured at "
            "the largest $ drawdown (negative sign) and can be smaller in magnitude. BASE, "
            "STRESS-d1 and EXIT-3X are report-only; STRESS-d2, STRESS-d3 and MISSED-EXIT are "
            "pass/fail. Ticket STATS (win rate, expectancy) come from STRESS-d2.", ""]
    if inp.secondary is not None:
        s = inp.secondary
        out += [f"Secondary sample 2023-03-01..2026-04-30 ({s.name}, report only): {s.trades} "
                f"trades, {_usd(s.expectancy_usd)} $/trade, {_r(s.expectancy_r)}, Sharpe "
                f"{_f(s.daily_sharpe, 4)}, total {_usd(s.total_usd)}, max DD "
                f"{_usd0(s.max_dd_usd)}.", ""]
    return out


def _fold_section(inp: ValidationInputs) -> list[str]:
    out = [f"## Walk-forward folds (run A, {PRIMARY_SCENARIO}; COMMON 10.2, 10.4)", ""]
    if not inp.folds:
        return out + ["Not run.", ""]
    rows = [(f.n, f.window, _pct(f.data_share), f.trades, _f(f.prorated_minimum, 2),
             "yes" if f.meets_minimum else "NO", _usd(f.pnl_usd), _r(f.expectancy_r),
             "yes" if f.positive else "no") for f in sorted(inp.folds, key=lambda f: f.n)]
    out.append(_table(["Fold", "Window", "Data share", "Trades", "Pro-rated minimum",
                       "Minimum met", "P&L $", "Mean R", "Positive"], rows))
    out += ["", "Pro-rated minimum = 4 trades per full fold x data share (amendment A1). A trade "
            "belongs to the fold of its signal time T.", ""]
    if not _folds_complete(inp.folds):
        out += [f"WARNING: {_fold_numbers(inp.folds)}; the fold criteria are NOT EVALUATED.", ""]
    bad = _count_problems(inp, inp.scenario(PRIMARY_SCENARIO))
    if bad:
        out += [f"WARNING (10.4 rows and criterion 4 NOT EVALUATED): {'; '.join(bad)}.", ""]
    return out


def _bucket_section(inp: ValidationInputs) -> list[str]:
    out = [f"## Breakdowns (run A, {PRIMARY_SCENARIO}; COMMON 8.9, 10.4 bucket rule)", ""]
    if inp.trades is None:
        return out + ["Not run.", ""]
    titles = (("key", "Per index"), ("side", "Per side"), ("exit_type", "Per exit type"),
              ("regime", "Per regime (ACTIVE / CALM at signal time)"))
    for by, title in titles:
        bt = bucket_table(inp.trades, by)
        out += [f"### {title}", ""]
        if bt.empty:
            out += ["No filled trades.", ""]
            continue
        out += [_table(["Bucket", "Trades", "Share", "Total $", "Mean $", "Mean R"],
                       _bucket_rows(bt)), ""]
    out += [f"Buckets with fewer than {MIN_BUCKET_TRADES} trades are merged into OTHER; OTHER's "
            "statistics are withheld if it has fewer than 30 trades.", ""]
    if inp.provenance.strategy_id == PAIR_STRATEGY:
        out += _pair_day_lines(inp.trades)
    return out


def _bucket_rows(bt: pd.DataFrame) -> list[tuple]:
    rows = []
    for _, r in bt.iterrows():
        name = r["bucket"] + (f" ({r['merged']})" if r["merged"] else "")
        if r["withheld"]:
            rows.append((name, r["n"], _pct(r["share"]), "withheld (n < 30)", "", ""))
        else:
            rows.append((name, r["n"], _pct(r["share"]), _usd(r["total_usd"]),
                         _usd(r["mean_usd"]), _r(r["mean_r"])))
    return rows


def _pair_day_lines(trades: pd.DataFrame) -> list[str]:
    """SES-IBX-01: pair days reported separately (spec QA table)."""
    ps = pair_day_summary(trades)
    out = ["### Pair days (SES-IBX-01: US30 and NAS100 tickets of opposite sides on one New York "
           "date)", "",
           f"Pair days: {ps['n_pair_days']}; with both legs filled: {ps['n_both_filled']}. "
           "A pair day is two tickets, each with its own risk; both legs can lose.", ""]
    bt = ps["table"]
    if bt.empty:
        return out + ["No filled trades.", ""]
    return out + [_table(["Trades", "N", "Share", "Total $", "Mean $", "Mean R"],
                         _bucket_rows(bt)), ""]


def _run_b_section(inp: ValidationInputs) -> list[str]:
    out = ["## Run B (walk-forward selected; COMMON 10.2)", ""]
    rb = inp.run_b
    if rb is None:
        return out + ["Not run.", ""]
    rows = [(s.fold, _params(s.params), _f(s.train_sharpe, 4),
             s.n_embargoed if s.n_embargoed is not None else "not supplied", s.test_trades,
             _usd(s.test_pnl_usd)) for s in sorted(rb.selections, key=lambda s: s.fold)]
    if rows:
        out += [_table(["Fold", "Selected (one-at-a-time grid, train window)",
                        "Train daily Sharpe", "Embargoed train trades", "Test trades",
                        "Test P&L $"], rows), ""]
    else:
        out += ["No fold selections supplied.", ""]
    out += [f"Out-of-sample ({rb.scenario}): {rb.trades} trades, {_usd(rb.expectancy_usd)} "
            f"$/trade, {_r(rb.expectancy_r)}, total {_usd(rb.total_usd)}, daily Sharpe "
            f"{_f(rb.daily_sharpe, 4)}. Embargo: filled train trades exiting after the train "
            "window end are dropped from selection (10.2).", ""]
    bad = _run_b_problems(inp)
    if bad:
        out += [f"WARNING (criterion 3 NOT EVALUATED): {'; '.join(bad)}.", ""]
    return out


def _exits(shares: Mapping[str, float]) -> str:
    if not shares:
        return "n/a"
    short = {"time_exit": "TE", "data_gap_exit": "GAP", "end_of_data": "END"}
    return " ".join(f"{short.get(k, k)} {100 * v:.0f}%"
                    for k, v in sorted(shares.items(), key=lambda kv: -kv[1]))


def _cell_text(c: SensitivityCell | None) -> str:
    if c is None:
        return "not run"
    if not c.sufficient:
        return f"insufficient (n={c.trades})"
    return f"{_usd(c.expectancy_usd)} (n={c.trades})"


def _sensitivity_section(inp: ValidationInputs) -> list[str]:
    out = ["## Parameter sensitivity (COMMON 10.5.5)", ""]
    sens = inp.sensitivity
    if sens is None or not sens.cells:
        return out + ["Not run.", ""]
    ss = sensitivity_summary(sens, inp.provenance.strategy_id)
    cells: dict[tuple, SensitivityCell] = ss["cells"]
    grid: dict[str, tuple[float, ...]] = ss["grid"]
    defaults = dict(sens.defaults)
    dkey = _key(defaults)
    out += [f"Scenario {sens.scenario}. Defaults: {_params(defaults)}. Pre-registered "
            f"configurations: {ss['n_expected']}. Cell value = $ expectancy per trade (trade "
            f"count); a cell with fewer than {MIN_SENSITIVITY_CELL_TRADES} trades is "
            "insufficient and counts as failed; a cell not run is never a pass.", "",
            "### One at a time", ""]
    nb = {k for k, _ in ss["neighbours"]}
    rows = []
    for p in defaults:
        for v in grid.get(p, ()):
            pv = {k: round(float(x), 9) for k, x in defaults.items()}
            pv[p] = v
            k = _key(pv)
            c = cells.get(k)
            role = ("default" if k == dkey else "neighbour" if k in nb else "")
            rows.append((p, _pv(v), role, c.trades if c else "n/a",
                         _usd(c.expectancy_usd) if c else "not run",
                         _r(c.expectancy_r) if c else "n/a",
                         "PASS" if c and c.passes else ("insufficient" if c and not c.sufficient
                                                        else "FAIL" if c else "NOT RUN"),
                         _exits(c.exit_shares) if c else "n/a"))
    out += [_table(["Parameter", "Value", "Role", "Trades", "Expectancy $", "Expectancy R",
                    "Cell", "Exit-type shares"], rows), ""]

    x, y = sens.heatmap_x, sens.heatmap_y
    xs, ys = grid.get(x, ()), grid.get(y, ())
    out += [f"### Heatmap {x} x {y}", "",
            f"Rows {y}, columns {x}; the default cell is in bold. Exit-type shares of each cell "
            "are next to the expectancy grid (SL, TP, TE = time exit, GAP = data_gap_exit, "
            "END = end_of_data).", ""]
    hdr = [f"{y} \\ {x}"] + [f"{x}={_pv(v)}" for v in xs] + [f"exits {x}={_pv(v)}" for v in xs]
    rows = []
    for yv in reversed(ys):
        vals, ex = [], []
        for xv in xs:
            pv = {k: round(float(v), 9) for k, v in defaults.items()}
            pv[x], pv[y] = xv, yv
            k = _key(pv)
            c = cells.get(k)
            t = _cell_text(c)
            vals.append(f"**{t}**" if k == dkey else t)
            ex.append(_exits(c.exit_shares) if c else "n/a")
        rows.append([f"{y}={_pv(yv)}"] + vals + ex)
    out += [_table(hdr, rows), ""]
    out += [f"Pre-registered cells: {ss['n_expected']}; run: {ss['n_cells']}; positive and "
            f"sufficient: {ss['n_pass']} ({_pct(ss['share'])} of {ss['n_expected']}); "
            f"insufficient: {ss['n_insufficient']}; missing: {len(ss['missing'])}. Immediate "
            f"neighbours of the default: {sum(c is not None and c.passes for _, c in ss['neighbours'])}"
            f" of {len(ss['neighbours'])} pass. A flat response of a parameter that rarely binds "
            "(for example a TP that is rarely reached) is not robustness evidence: read the "
            "exit-type shares.", ""]
    if ss["missing"]:
        out += ["Cells not run (counted as not positive): "
                + "; ".join(_params(ss["expected"][k]) for k in ss["missing"]) + ".", ""]
    if ss["problems"]:
        out += ["WARNING (criterion 5 NOT EVALUATED): " + "; ".join(ss["problems"]) + ".", ""]
    return out


def _q(qs: Mapping[int, float], fmt=_usd) -> tuple[str, str, str]:
    return tuple(fmt(qs.get(k)) for k in (5, 50, 95))  # type: ignore[return-value]


def _bootstrap_section(inp: ValidationInputs) -> list[str]:
    out = ["## Bootstrap (COMMON 10.5.6)", ""]
    rows, warn = [], []
    prim = inp.scenario(PRIMARY_SCENARIO)
    for label, b, method in (("Stationary block bootstrap of the daily series", inp.bootstrap,
                              BOOTSTRAP_METHOD),
                             ("Regime-conditional bootstrap of trades", inp.regime_bootstrap,
                              "regime_conditional")):
        if b is None:
            rows.append((label, "not run", "", "", "", "", "", ""))
            continue
        rows.append((label, *_q(b.total_pnl_q), *_q(b.max_dd_q, _usd0),
                     f"{b.scenario or 'unlabelled'}, n_obs {b.n_obs if b.n_obs is not None else 'n/a'}"
                     f", {b.n_resamples} resamples, seed {b.seed}"
                     + (f", mean block {_pv(float(b.mean_block_length))} days"
                        if b.mean_block_length else "")))
        bad = _bootstrap_problems(b, method, prim)
        if bad:
            warn.append(f"WARNING: {label} not at the frozen settings or not of run A "
                        f"{PRIMARY_SCENARIO} ({'; '.join(bad)}).")
    out += [_table(["Method", "Total $ p5", "Total $ p50", "Total $ p95", "Max DD $ p5",
                    "Max DD $ p50", "Max DD $ p95", "Settings"], rows), ""]
    for w in warn:
        out += [w, ""]
    rb = inp.regime_bootstrap
    if rb is not None and rb.bucket_counts:
        out += ["Regime-conditional bucket counts (kept fixed): "
                + ", ".join(f"{k} {v}" for k, v in sorted(rb.bucket_counts.items())) + ".", ""]
    if inp.regime is not None and not inp.regime.functional:
        out += [f"The {inp.regime.gate} gate is non-functional for this strategy (COMMON 6.8), so "
                "no regime-conditional claim may be made from these numbers.", ""]
    return out


def _shuffle_section(inp: ValidationInputs) -> list[str]:
    out = ["## Shuffle test (COMMON 10.5.10, report only)", ""]
    sh = inp.shuffle
    if sh is None:
        return out + ["Not run.", ""]
    out += [f"{sh.n_permutations} permutations of trade order (seed {sh.seed}): 95th "
            f"percentile of the worst losing streak {_f(sh.worst_streak_p95, 1)} trades; of "
            f"max drawdown {_usd0(sh.max_dd_p95_usd)} $.", ""]
    bad = _shuffle_problems(sh)
    if bad:
        out += [f"WARNING: shuffle test not at the frozen settings ({'; '.join(bad)}).", ""]
    return out


def _benchmark_section(inp: ValidationInputs) -> list[str]:
    out = ["## Benchmarks (COMMON 10.6; never selectable, not counted in N)", ""]
    bm = inp.benchmarks
    if bm is None:
        return out + ["Not run.", ""]
    sa = inp.scenario(PRIMARY_SCENARIO)
    exp = sa.expectancy_usd if sa else float("nan")
    tot = sa.total_usd if sa else float("nan")
    seeds = ("seeds n/a" if bm.b1_seeds is None else
             f"seeds {min(bm.b1_seeds)}..{max(bm.b1_seeds)}" if len(bm.b1_seeds) else "no seeds")
    rows = [(f"Strategy (run A, {PRIMARY_SCENARIO})", _usd(exp), _usd(tot),
             sa.trades if sa else "n/a", ""),
            (f"B1 placebo ({bm.b1_draws} draws, {seeds})", "", f"mean "
             f"{_usd(bm.b1_mean_total_usd)}; p95 {_usd(bm.b1_p95_total_usd)}", "",
             "pass criterion 7"),
            ("B2 random index", f"mean {_usd(bm.b2_mean_usd)}", "", bm.b2_trades, "report"),
            ("B3 gate off", _usd(bm.b3_expectancy_usd), _usd(bm.b3_total_usd), bm.b3_trades,
             "report")]
    out += [f"Benchmarks run under {bm.scenario}.", "",
            _table(["Run", "Expectancy $ / trade", "Total $", "Trades", "Role"], rows), ""]
    bad = _benchmark_problems(bm)
    if bad:
        out += [f"WARNING (criterion 7 NOT EVALUATED): {'; '.join(bad)}.", ""]
    if bm.scenario != PRIMARY_SCENARIO:
        out.append(f"B2 / B3: not evaluable (benchmarks not run under {PRIMARY_SCENARIO}).")
        return out + [""]
    if not (_finite(exp) and _finite(bm.b2_mean_usd)):
        out.append("B2: not evaluable (missing values).")
    elif exp <= bm.b2_mean_usd:
        out.append(f"The strategy does not beat B2's mean: {SENTENCE_RS}.")
    else:
        out.append("The strategy beats B2's mean $ per trade.")
    if not (_finite(exp) and _finite(bm.b3_expectancy_usd)):
        out.append("B3: not evaluable (missing values).")
    elif bm.b3_expectancy_usd >= exp:
        out.append(f"B3 (gate off) >= the gated run in $ expectancy: {SENTENCE_GATE}. The gate "
                   "is not changed in response.")
    else:
        out.append("The gated run beats B3 (gate off) in $ expectancy.")
    return out + [""]


def _replay_section(inp: ValidationInputs, cfg: Mapping[str, Any]) -> list[str]:
    out = ["## Competition replay (COMMON 10.5.9)", "",
           "30-calendar-day windows starting every Monday in the test span.", ""]
    confirmed = competition_confirmed(cfg)
    if not confirmed:
        out += [COMPETITION_UNCONFIRMED, ""]
        if inp.replay_for("competition") is not None:
            out += [COMPETITION_IGNORED, ""]
    usable = {c: (r, why) for c, r, why in _replay_rows(inp, cfg)}
    rows = []
    for r in sorted(inp.replay, key=lambda r: (r.contract != "demo", r.contract)):
        upp = (", ".join(f"{k} {v:g}" for k, v in r.usd_per_point.items())
               if r.usd_per_point else "n/a")
        used, why = usable.get(r.contract, (None, "unknown contract label"))
        label = r.contract if used is r else f"{r.contract} (IGNORED: {why})"
        rows.append((label, upp, r.n_windows,
                     f"{r.total_loss_breach_windows} ({_pct(r.share(r.total_loss_breach_windows), 2)})",
                     f"{r.daily_loss_breach_windows} ({_pct(r.share(r.daily_loss_breach_windows), 2)})",
                     f"{r.personal_stop_breach_windows} "
                     f"({_pct(r.share(r.personal_stop_breach_windows), 2)})",
                     _pct(r.share(r.windows_ge5_trade_days))))
    if rows:
        out += [_table(["Contract", "USD per point per lot", "Windows", "10% total-loss breaches",
                        "5% daily-loss breaches", "Personal 3% stop breaches",
                        "Windows with >= 5 trade days"], rows), ""]
    else:
        out += ["Not run.", ""]
    return out


def _dsr_section(inp: ValidationInputs) -> list[str]:
    out = ["## Deflated Sharpe and PBO (COMMON 10.7)", ""]
    d, p = inp.dsr, inp.pbo
    logged = logged_n_raw(inp) if d is not None or p is not None else None
    if d is None:
        out += ["DSR: not run.", ""]
    else:
        out += [_table(["Trial count", "N", "V (variance of SR)", "DSR", "Threshold"],
                       [("N_raw (all configurations ever run)", d.n_raw, _f(d.v_raw, 6),
                         _f(d.dsr_raw, 4), f">= {DSR_MIN}"),
                        ("N_eff (clusters at correlation 0.7)", d.n_eff, _f(d.v_eff, 6),
                         _f(d.dsr_eff, 4), f">= {DSR_MIN}")]), "",
                f"Trial log: {logged} counted configurations (all strategies) at render time; "
                f"N_raw used by the DSR: {d.n_raw}.", "",
                f"SR_hat (daily, non-annualised) {_f(d.sr_hat, 4)} over {d.t_days} days; skew "
                f"{_f(d.skew)}, kurtosis (non-excess) {_f(d.kurtosis)}.", ""]
    if p is None:
        out += ["PBO: not run.", ""]
    else:
        flag = ""
        if _finite(p.pbo) and PBO_FLAG <= p.pbo < PBO_MAX:
            flag = " FLAGGED (0.2-0.5)."
        out += [f"PBO (CSCV, S = {p.blocks} blocks, {p.n_splits} splits, {p.n_trials} trials): "
                f"{_f(p.pbo)}; threshold < {PBO_MAX}.{flag}", ""]
    if d is not None and p is not None:
        bad = _trial_problems(d, p, logged) + _dsr_tie_problems(
            d, inp.scenario(PRIMARY_SCENARIO))
        if bad:
            out += [f"WARNING (criterion 8 NOT EVALUATED): {'; '.join(bad)}.", ""]
    return out


def _regime_section(inp: ValidationInputs) -> list[str]:
    out = ["## Regime diagnostics (COMMON 6.8)", ""]
    rg = inp.regime
    if rg is None:
        return out + ["Not supplied.", ""]
    out += [f"Source: {rg.source}. Gate {rg.gate}: blocks {_pct(rg.block_share)} of in-window "
            f"cycles: **{rg.flag}**.", ""]
    if rg.per_index:
        out += [_table(["Index", "Block share", "5-95% check"],
                       [(k, _pct(s), fl) for k, (s, fl) in rg.per_index.items()]), ""]
    if rg.non_functional_folds:
        out += ["Folds where every in-window cycle is blocked (regime UNAVAILABLE: no model or "
                "warm-up): " + ", ".join(str(f) for f in rg.non_functional_folds) + ".", ""]
    if not rg.functional:
        out += [f"The {rg.gate} gate is NON-FUNCTIONAL for this strategy (blocks < 5% or > 95%): "
                "no regime-conditional claim may be made.", ""]
    return out


def _criteria_section(rows: Sequence[CriterionResult], verdict: str) -> list[str]:
    out = ["## Pass / fail (COMMON 10.4 minimums, 10.5 criteria 1-9)", ""]
    out.append(_table(["Criterion", "Requirement", "Measured", "Threshold", "Result"],
                      [(r.criterion, r.description, r.measured, r.threshold, f"**{r.status}**")
                       for r in rows]))
    out += ["", f"**Overall verdict: {verdict}.** PASS only if every row except the report-only "
            "rows passes; NOT EVALUATED never counts as a pass. Standards are never lowered.", ""]
    return out


def render_report(inp: ValidationInputs) -> str:
    """The full Markdown report for one strategy (see the module docstring)."""
    pv = inp.provenance
    _check_provenance(inp)
    cfg = _config(inp)
    confirmed = competition_confirmed(cfg)
    rows = evaluate_criteria(inp)
    verdict = overall_verdict(rows)
    gen = pv.generated_utc or datetime.now(timezone.utc)
    out = [f"# Validation report: {pv.strategy_id}", ""]
    if pv.synthetic:
        out += ["> **SYNTHETIC DATA: this report was generated from fabricated inputs for tooling "
                "tests. It says nothing about the strategy.**", ""]
    out += [f"**Overall verdict: {verdict}**", "",
            f"- {DEMO_NOTE}", f"- {HOLDOUT_NOTE}"]
    if not confirmed:
        out.append(f"- {COMPETITION_UNCONFIRMED}")
    out += ["", "## Provenance", "",
            f"- Strategy: {pv.strategy_id}; parameters: {_params(pv.params)}",
            f"- Git SHA: {pv.git_sha}",
            f"- Data SHA-256: {pv.data_hash}",
            f"- config.yaml SHA-256: {pv.config_hash}",
            f"- Data: {pv.data_source}" + (f", last bar date {pv.data_end}" if pv.data_end else ""),
            f"- Trial log rows: {', '.join(pv.trial_ids) if pv.trial_ids else 'none supplied'}",
            f"- Competition contract (O4): {'confirmed' if confirmed else 'NOT confirmed'} "
            "(config.yaml rules.competition_usd_per_point)",
            f"- Generated (UTC): {gen:%Y-%m-%d %H:%M}"]
    out += [f"- {n}" for n in pv.notes]
    out.append("")
    ra = _run_a_problems(inp)
    if ra:
        out += [f"WARNING (every run-A row NOT EVALUATED): {ra[0]}. COMMON 10.2 run A uses the "
                "defaults fixed ex ante.", ""]
    for sec in (_scenario_section, _fold_section, _bucket_section, _run_b_section,
                _sensitivity_section, _bootstrap_section, _shuffle_section, _benchmark_section):
        out += sec(inp)
    out += _replay_section(inp, cfg)
    out += _dsr_section(inp)
    out += _regime_section(inp)
    out += _criteria_section(rows, verdict)
    out += ["## SPEC NOTEs (report generator readings)", ""]
    out += [f"{i}. {n}" for i, n in enumerate(list(SPEC_NOTES) + list(inp.spec_notes), 1)]
    out += ["", "## Limitations", "", f"- {DEMO_NOTE}", f"- {HOLDOUT_NOTE}",
            "- Swap and tick value are current demo values applied to all history (COMMON 8.6, "
            "8.7).", ""]
    return "\n".join(out)


def write_report(inp: ValidationInputs, out_dir: Path | str | None = None) -> Path:
    """Write docs/validation/<STRATEGY_ID>.md (or <out_dir>/<STRATEGY_ID>.md) and return it."""
    text = render_report(inp)
    d = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    sid = re.sub(r"[^A-Za-z0-9_.-]", "_", inp.provenance.strategy_id)
    path = d / f"{sid}.md"
    path.write_text(text, encoding="utf-8")
    return path
