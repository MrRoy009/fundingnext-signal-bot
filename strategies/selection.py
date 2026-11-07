"""COMMON 4.2 index selection, shared by every strategy (live generate() and batch scan()).

One call per strategy and decision time T. The caller supplies, for each member of the window
set G(T) (COMMON 4.2.1; see strategies.context for building it):
  - the rank score at T (RS_4h or RS_day, COMMON 4.1),
  - a data-completeness cause, if the member lacks its M15 bar closing at T or has a NaN rank
    score or NaN strategy input because of missing data (4.2.2),
  - its regime status (OK or not, with the cause) and whether it passes the strategy's gate (4.2.3;
    a strategy may fold its own eligibility conditions into the gate, e.g. SES-IBX-01).
`select_index` then applies 4.2.2-4.2.8 exactly and returns a `Selection`:

    incomplete group     -> no candidate for the whole group, cause
                            "incomplete group: <index>: <cause>" (no partial ranking)
    E(T)                 = members with regime status OK that pass the gate (filter, then rank:
                            owner decision O7 default)
    LONG-SELECTED        = argmax of the rank score over E(T); SHORT-SELECTED = argmin
    ties (|a - b| <= 1e-12) go to config instrument order: the earlier index is higher for the
                            long and lower for the short
    |E(T)| = 1           -> BUY only if score > 0, SELL only if score < 0, neither if = 0
    empty E(T)           -> no candidate, with the reason per index
    rank_score           = the chosen index's score rounded to 2 dp

The function never promotes the next index: a strategy may give a BUY only to `long` and a SELL
only to `short`, and if that index then fails an entry condition there is no candidate on that
side (4.2.6). `Selection.allows(key, side)` is the check the scan driver enforces.

Instruments with `tradeable: false` (US500, COMMON 1.1 / O6) are never ranked: they are not in
`config_order(cfg)`, and a member outside that order is a programming error (ValueError).

Public API
    config_order(cfg) -> tuple[str, ...]
    MemberInput(key, score, incomplete=None, regime_ok=True, regime_cause="", gate_pass=True,
                gate_cause="")
    select_index(T, members, order, strategy_id="", excluded=None) -> Selection
    Selection: .T .strategy_id .members .scores .eligible .long .short .incomplete .reasons
               .note .excluded .has_side .sole .allows(key, side) .rank_score(key)
               .set_text() .log_line()
    TIE_TOL = 1e-12
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import pandas as pd

TIE_TOL = 1e-12          # COMMON 4.2.4
RANK_SCORE_DP = 2        # COMMON 4.2.8

BUY, SELL = "BUY", "SELL"


def config_order(cfg: dict) -> tuple[str, ...]:
    """Ranked instruments in config order (US30, NAS100, GER40). `tradeable: false` instruments
    (US500) are left out, so they can never enter G(T) or win a tie."""
    return tuple(i["key"] for i in cfg["instruments"] if i.get("tradeable", True))


@dataclass(frozen=True)
class MemberInput:
    """One member of G(T) as seen at T.

    incomplete: data-completeness cause (COMMON 4.2.2), None if complete. A NaN score is
        incomplete even without a cause ("rank score NaN"): it cannot be ranked.
    regime_ok / regime_cause: regime status OK, else the cause ("regime unavailable: ...").
    gate_pass / gate_cause: passes the strategy's regime gate (and any eligibility condition the
        strategy adds); gate_cause says why not. Only read when regime_ok is True.
    """
    key: str
    score: float
    incomplete: str | None = None
    regime_ok: bool = True
    regime_cause: str = ""
    gate_pass: bool = True
    gate_cause: str = ""


def _fmt(x: float) -> str:
    return "nan" if not math.isfinite(x) else f"{x:+.4f}"


@dataclass(frozen=True)
class Selection:
    """Outcome of COMMON 4.2 for one strategy at one T. `long` / `short` are the keys of
    LONG-SELECTED / SHORT-SELECTED that may receive a BUY / SELL (None: no candidate on that
    side). They can be the same index only on an exact tie (see select_index)."""
    T: pd.Timestamp
    strategy_id: str
    members: tuple[str, ...]                  # G(T), config order
    scores: Mapping[str, float]               # raw rank scores of every member (logged)
    eligible: tuple[str, ...] = ()            # E(T), config order
    long: str | None = None
    short: str | None = None
    incomplete: str | None = None             # "incomplete group: <index>: <cause>"
    reasons: Mapping[str, str] = field(default_factory=dict)   # members not in E(T)
    note: str = ""                            # e.g. the sole index has score 0
    # Indices in their window at T but outside G(T) (not a valid trading date, COMMON 4.2.1 +
    # 1.6), with the reason; logged, never ranked.
    excluded: Mapping[str, str] = field(default_factory=dict)

    @property
    def has_side(self) -> bool:
        return self.long is not None or self.short is not None

    @property
    def sole(self) -> bool:
        """True when E(T) has one index (COMMON 4.2.5); strategies word the reason line
        'sole index' then."""
        return len(self.eligible) == 1

    def allows(self, key: str, side: str) -> bool:
        """COMMON 4.2.6: only LONG-SELECTED may get a BUY, only SHORT-SELECTED a SELL."""
        if side == BUY:
            return key == self.long
        if side == SELL:
            return key == self.short
        raise ValueError(f"side must be BUY or SELL, not {side!r}")

    def rank_score(self, key: str) -> float:
        """Candidate.rank_score: the index's rank score rounded to 2 dp (COMMON 4.2.8)."""
        return round(float(self.scores[key]), RANK_SCORE_DP)

    def set_text(self, keys: Sequence[str] | None = None) -> str:
        """'{US30, NAS100}' for reason lines; default G(T)."""
        return "{" + ", ".join(self.members if keys is None else keys) + "}"

    def log_line(self) -> str:
        """The per-cycle log line: every member's score (COMMON 4.2.8), E(T), the selection,
        and the reason per index outside E(T) (4.2.7) or the incomplete-group cause (4.2.2)."""
        scores = ", ".join(f"{k} {_fmt(self.scores[k])}" for k in self.members)
        head = f"{self.strategy_id} {self.T:%Y-%m-%d %H:%M} UTC G(T) {{{scores}}}"
        if self.excluded:
            head += "; outside G(T): " + "; ".join(f"{k}: {r}" for k, r in self.excluded.items())
        if self.incomplete:
            return f"{head}; no candidate: {self.incomplete}"
        why = "; ".join(f"{k}: {r}" for k, r in self.reasons.items())
        out = (f"{head}; E(T) {self.set_text(self.eligible)}; LONG-SELECTED {self.long or '-'}"
               f", SHORT-SELECTED {self.short or '-'}")
        if not self.eligible:
            out += "; E(T) empty, no candidate"
        if why:
            out += f"; not eligible: {why}"
        if self.note:
            out += f"; {self.note}"
        return out


def _pick(eligible: Sequence[str], scores: Mapping[str, float], highest: bool) -> str:
    """argmax (highest) or argmin over `eligible` (already in config order). Every index whose
    score is within TIE_TOL of the extreme is tied, and the first in config order wins: the
    earlier index counts as higher for the long and lower for the short (COMMON 4.2.4)."""
    vals = [scores[k] for k in eligible]
    best = max(vals) if highest else min(vals)
    for k, v in zip(eligible, vals):
        if abs(v - best) <= TIE_TOL:
            return k
    raise AssertionError("unreachable: the extreme is always within tolerance of itself")


def select_index(T: pd.Timestamp, members: Sequence[MemberInput], order: Sequence[str],
                 strategy_id: str = "", excluded: Mapping[str, str] | None = None) -> Selection:
    """COMMON 4.2 steps 2-8 for one strategy at T. `members` is G(T) (any order); `order` is
    `config_order(cfg)`. Empty G(T) gives an empty Selection (no candidate, nothing to log).
    `excluded`: indices in their window at T that are not in G(T) because T's date is not a
    valid trading date for them (COMMON 4.2.1 + 1.6), with the reason; only carried for the log.

    SPEC NOTE: a NaN rank score always makes the group incomplete ("rank score NaN" unless the
    caller gives a more precise cause), including a NaN from a lookback that is still warming
    up: a NaN cannot be ranked, and COMMON 4.2.2 forbids ranking the rest without it.
    SPEC NOTE: when more than one member is incomplete, every cause is kept, joined by '; ',
    each in the literal form 'incomplete group: <index>: <cause>' (config order).
    SPEC NOTE: on an exact tie (within 1e-12) in a group of two, the literal 4.2.4 rule makes
    the earlier index both the highest (LONG-SELECTED) and the lowest (SHORT-SELECTED); the
    later index is then selected for neither side.
    """
    rank = {k: n for n, k in enumerate(order)}
    for m in members:
        if m.key not in rank:
            raise ValueError(f"{m.key} is not a ranked instrument (config order {tuple(order)}; "
                             "tradeable: false instruments never enter G(T))")
    keys = [m.key for m in members]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate member in G(T): {keys}")
    ex = dict(excluded or {})
    for k in ex:
        if k not in rank or k in keys:
            raise ValueError(f"excluded index {k} must be ranked and not a member of G(T)")
    ex_ro = MappingProxyType({k: ex[k] for k in sorted(ex, key=rank.__getitem__)})
    ms = sorted(members, key=lambda m: rank[m.key])
    T = pd.Timestamp(T)
    g = tuple(m.key for m in ms)
    scores = MappingProxyType({m.key: float(m.score) for m in ms})

    # 4.2.2 data completeness: one incomplete member removes the whole group.
    causes = []
    for m in ms:
        cause = m.incomplete or (None if math.isfinite(m.score) else "rank score NaN")
        if cause:
            causes.append(f"incomplete group: {m.key}: {cause}")
    if causes:
        return Selection(T, strategy_id, g, scores, incomplete="; ".join(causes),
                         excluded=ex_ro)

    # 4.2.3 regime filter (filter first, then rank: O7 default).
    reasons: dict[str, str] = {}
    eligible = []
    for m in ms:
        if not m.regime_ok:
            reasons[m.key] = m.regime_cause or "regime unavailable"
        elif not m.gate_pass:
            reasons[m.key] = m.gate_cause or "regime gate not passed"
        else:
            eligible.append(m.key)
    e = tuple(eligible)
    reasons_ro = MappingProxyType(reasons)
    if not e:                                                   # 4.2.7
        return Selection(T, strategy_id, g, scores, e, reasons=reasons_ro, excluded=ex_ro)

    if len(e) == 1:                                             # 4.2.5 single index
        k, s = e[0], scores[e[0]]
        if s > 0:
            return Selection(T, strategy_id, g, scores, e, long=k, reasons=reasons_ro,
                             excluded=ex_ro)
        if s < 0:
            return Selection(T, strategy_id, g, scores, e, short=k, reasons=reasons_ro,
                             excluded=ex_ro)
        return Selection(T, strategy_id, g, scores, e, reasons=reasons_ro,
                         note=f"sole index {k} has rank score 0: no side", excluded=ex_ro)

    return Selection(T, strategy_id, g, scores, e,                # 4.2.4
                     long=_pick(e, scores, highest=True), short=_pick(e, scores, highest=False),
                     reasons=reasons_ro, excluded=ex_ro)
