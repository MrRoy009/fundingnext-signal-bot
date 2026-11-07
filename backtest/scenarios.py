"""Backtest scenarios (COMMON 10.3) and contract tables (COMMON 8.7, owner decision O4).

A Scenario is one set of execution assumptions: manual delay d, spread multiplier c, slippage
multiplier s, commission level, negative-swap multiplier, and the two exit variants. Scenarios
are pass/fail checks on one configuration; they never select between configurations and are not
trials (COMMON 10.3), so they live here as fixed presets, not in config.

The contract table is USD per index point per lot. The demo contract is frozen in config
`costs.usd_per_point_demo`; the competition contract is config
`rules.competition_usd_per_point`, null until the owner confirms it (then only the demo result
exists).
"""
from __future__ import annotations

from dataclasses import dataclass

# COMMON 10.3, EXIT-3X: "spread 3 on SL and time-exit fills only".
EXIT_3X_SPREAD_MULT = 3.0

COMMISSION_BASE = "base"
COMMISSION_STRESS = "stress"


@dataclass(frozen=True)
class Scenario:
    """Execution assumptions for one backtest run (COMMON 8, 10.3).

    d:            manual delay in M15 bars; placement P = open of the first bar >= T + d bars.
    spread_mult:  c in eff_spread = c * max(spread_pts, floor), on triggers, exits and cost.
    slip_mult:    s in slip = s * max(0.5 floor, 0.02 ATR).
    commission:   'base' (owner-confirmed competition commission) or 'stress' (COMMON 8.6).
    swap_mult:    multiplier on NEGATIVE swap only (stress doubles it; positive swap is 0).
    missed_exit:  True disables the manual time exit (MISSED-EXIT: bracket held to SL/TP,
                  overnight and weekend).
    exit_3x:      True prices SL and time-exit fills with spread x3 (EXIT-3X).
    """
    name: str = "BASE"
    d: int = 2
    spread_mult: float = 1.0
    slip_mult: float = 1.0
    commission: str = COMMISSION_BASE
    swap_mult: float = 1.0
    missed_exit: bool = False
    exit_3x: bool = False

    def __post_init__(self) -> None:
        if self.d not in (1, 2, 3):
            raise ValueError(f"delay d must be 1, 2 or 3 bars (COMMON 8.3), not {self.d}")
        if self.commission not in (COMMISSION_BASE, COMMISSION_STRESS):
            raise ValueError(f"commission must be 'base' or 'stress', not {self.commission!r}")
        if self.spread_mult <= 0 or self.slip_mult < 0 or self.swap_mult < 0:
            raise ValueError("spread_mult must be > 0; slip_mult and swap_mult >= 0")

    @property
    def exit_spread_mult(self) -> float:
        """Spread multiplier for SL and time-exit FILL prices.
        SPEC NOTE: COMMON 10.3 gives EXIT-3X as 'spread 3 on SL and time-exit fills only'. Read
        literally, the x3 applies to those fill prices; triggers (including the short SL
        trigger), TP, entries, MTM and data-gap / end-of-data closes keep the scenario's c (2 in
        the preset). A long exit sells at the bid, so x3 changes only short SL gap fills and
        short time exits. Whether a short SL gapped (ask open at c >= SL) is a trigger too:
        it uses c; only the gap fill price uses x3."""
        return EXIT_3X_SPREAD_MULT if self.exit_3x else self.spread_mult


def _stress(name: str, d: int, **over) -> Scenario:
    return Scenario(name=name, d=d, spread_mult=2.0, slip_mult=2.0,
                    commission=COMMISSION_STRESS, swap_mult=2.0, **over)


# COMMON 10.3 table.
SCENARIOS: dict[str, Scenario] = {
    "BASE": Scenario(name="BASE", d=2),
    "STRESS-d2": _stress("STRESS-d2", 2),
    "STRESS-d3": _stress("STRESS-d3", 3),
    "STRESS-d1": _stress("STRESS-d1", 1),
    "MISSED-EXIT": _stress("MISSED-EXIT", 2, missed_exit=True),
    "EXIT-3X": _stress("EXIT-3X", 2, exit_3x=True),
}


def contract_table(cfg: dict, which: str = "demo") -> dict[str, float] | None:
    """USD per point per lot by index: 'demo' (config costs.usd_per_point_demo) or
    'competition' (config rules.competition_usd_per_point; None while it is null)."""
    if which == "demo":
        table = (cfg.get("costs") or {}).get("usd_per_point_demo")
        if not table:
            raise ValueError("config costs.usd_per_point_demo is not set (COMMON 8.7)")
    elif which == "competition":
        table = (cfg.get("rules") or {}).get("competition_usd_per_point")
        if table is None:
            return None
    else:
        raise ValueError(f"contract must be 'demo' or 'competition', not {which!r}")
    return {str(k): float(v) for k, v in table.items()}
