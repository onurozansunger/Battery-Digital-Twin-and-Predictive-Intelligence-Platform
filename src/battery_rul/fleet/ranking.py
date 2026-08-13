"""Fleet ranking and the composite priority score.

What this score is
------------------
A **configurable decision-support policy**. It combines seven normalised
components with weights an operator sets, and it exists so a fleet page can put
128 cells in an order rather than in alphabetical order. It is not an optimum, it
has not been validated against real maintenance outcomes (this platform has
none), and two operators with different weights will legitimately get different
orders.

Because of that, every score is returned with its breakdown: the raw value, the
normalisation applied, the weight, and the resulting contribution, per component.
A ranking nobody can interrogate is a ranking nobody should act on.

Missing components
------------------
A cell with no risk probability is not the same as a cell with zero risk. Missing
components are marked ``available=False``, contribute nothing, and are excluded
from the weight denominator — so the score stays on the same 0-``score_scale``
axis instead of being quietly deflated by the absence of evidence. The number of
available components travels in the breakdown, so a score built from two
components is visibly different from one built from seven.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from battery_rul.config import FleetRankingConfig
from battery_rul.fleet.domain import (
    FleetBatteryRecord,
    MaintenancePriority,
    ScoreComponent,
)

__all__ = [
    "RANKING_KEYS",
    "PriorityScore",
    "compute_priority_score",
    "rank_batteries",
]

RankingKey = Literal[
    "priority",
    "priority_score",
    "failure_risk",
    "rul",
    "rul_lower_bound",
    "soh",
    "soh_trend",
    "fade_trend",
    "temperature_trend",
    "resistance_trend",
    "data_quality",
    "uncertainty",
]

#: Every supported ranking key, for API validation and the dashboard's picker.
RANKING_KEYS: tuple[str, ...] = (
    "priority",
    "priority_score",
    "failure_risk",
    "rul",
    "rul_lower_bound",
    "soh",
    "soh_trend",
    "fade_trend",
    "temperature_trend",
    "resistance_trend",
    "data_quality",
    "uncertainty",
)


@dataclass(frozen=True, slots=True)
class PriorityScore:
    """A score and the argument for it."""

    score: float
    components: list[ScoreComponent]
    available_weight: float
    critical_override_applied: bool = False

    @property
    def n_available(self) -> int:
        return sum(1 for c in self.components if c.available)


def _linear_urgency(value: float | None, reference: float) -> float | None:
    """Map "cycles remaining" to urgency in [0, 1]: 0 cycles -> 1, ref -> 0."""
    if value is None or not np.isfinite(value):
        return None
    return float(np.clip(1.0 - (value / reference), 0.0, 1.0))


def _saturating(value: float | None, reference: float) -> float | None:
    """Map a magnitude to [0, 1], saturating at ``reference``."""
    if value is None or not np.isfinite(value):
        return None
    return float(np.clip(value / reference, 0.0, 1.0))


def compute_priority_score(
    record: FleetBatteryRecord,
    cfg: FleetRankingConfig,
    *,
    critical_override: bool = False,
) -> PriorityScore:
    """Score one cell, returning the full component breakdown.

    ``critical_override`` is set by the maintenance engine when a hard rule
    fires. It raises the score to at least ``critical_override_score`` so a cell
    the rules call critical cannot be sorted below a cell that merely scores
    well — the rule is the decision, the score is the ordering within it.
    """
    weights = cfg.weights()
    specs: list[tuple[str, float | None, float | None, str]] = [
        (
            "risk",
            record.failure_risk if not record.risk_is_experimental else None,
            record.failure_risk if not record.risk_is_experimental else None,
            "calibrated end-of-life probability within the horizon, used directly as a "
            "value in [0, 1]"
            + (
                "; withheld because the risk model failed its acceptance gate"
                if record.risk_is_experimental
                else ""
            ),
        ),
        (
            "rul",
            record.predicted_rul,
            _linear_urgency(record.predicted_rul, cfg.rul_reference_cycles),
            f"1 - RUL/{cfg.rul_reference_cycles}, clipped to [0, 1]",
        ),
        (
            "rul_lower_bound",
            record.rul_lower_bound,
            _linear_urgency(record.rul_lower_bound, cfg.rul_reference_cycles),
            f"1 - RUL_lower/{cfg.rul_reference_cycles}, clipped to [0, 1]; the "
            "conservative planning quantity",
        ),
        (
            "soh",
            record.measured_soh,
            _soh_severity(record.measured_soh, cfg),
            f"({cfg.soh_reference_high} - SOH) / "
            f"({cfg.soh_reference_high} - {cfg.soh_reference_low}), clipped to [0, 1]",
        ),
        (
            "trend",
            record.fade_trend_pct_per_10,
            _saturating(record.fade_trend_pct_per_10, cfg.trend_reference_pct_per_10),
            f"capacity-fade trend (% per 10 cycles) / {cfg.trend_reference_pct_per_10}, "
            "clipped to [0, 1]; a falling-capacity trend is positive",
        ),
        (
            "uncertainty",
            record.interval_width,
            _saturating(record.interval_width, cfg.uncertainty_reference_cycles),
            f"RUL interval width / {cfg.uncertainty_reference_cycles} cycles, clipped "
            "to [0, 1]; a wide interval raises priority because the cell is poorly "
            "characterised, not because it is worse",
        ),
        (
            "data_quality",
            record.data_quality_score,
            (
                None
                if record.data_quality_score is None
                else float(np.clip(1.0 - record.data_quality_score, 0.0, 1.0))
            ),
            "1 - input data-quality score, clipped to [0, 1]",
        ),
    ]

    components: list[ScoreComponent] = []
    weighted_sum = 0.0
    available_weight = 0.0
    for name, raw, normalised, transformation in specs:
        weight = float(weights[name])
        available = normalised is not None
        value = float(normalised) if normalised is not None else 0.0
        contribution = weight * value if available else 0.0
        if available:
            weighted_sum += contribution
            available_weight += weight
        components.append(
            ScoreComponent(
                name=name,
                raw_value=None if raw is None or not np.isfinite(raw) else round(float(raw), 6),
                normalised=round(value, 6),
                weight=round(weight, 6),
                contribution=round(contribution, 6),
                transformation=transformation,
                available=available,
            )
        )

    # Renormalised by the weight actually available, so a cell missing its risk
    # probability is not scored as though its risk were zero.
    score = 0.0 if available_weight <= 0 else cfg.score_scale * weighted_sum / available_weight

    applied = False
    if critical_override and score < cfg.critical_override_score:
        score = cfg.critical_override_score
        applied = True

    return PriorityScore(
        score=round(float(score), 4),
        components=components,
        available_weight=round(available_weight, 6),
        critical_override_applied=applied,
    )


def _soh_severity(soh: float | None, cfg: FleetRankingConfig) -> float | None:
    if soh is None or not np.isfinite(soh):
        return None
    span = cfg.soh_reference_high - cfg.soh_reference_low
    return float(np.clip((cfg.soh_reference_high - soh) / span, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def _missing_last(value: float | None, *, descending: bool) -> tuple[int, float]:
    """Sort key that always puts missing values last, in either direction."""
    if value is None or not np.isfinite(value):
        return (1, 0.0)
    return (0, -float(value) if descending else float(value))


_KEY_FUNCTIONS: dict[str, Callable[[FleetBatteryRecord], tuple[int, float]]] = {
    "priority_score": lambda r: _missing_last(r.priority_score, descending=True),
    "failure_risk": lambda r: _missing_last(
        None if r.risk_is_experimental else r.failure_risk, descending=True
    ),
    "rul": lambda r: _missing_last(r.predicted_rul, descending=False),
    "rul_lower_bound": lambda r: _missing_last(r.rul_lower_bound, descending=False),
    "soh": lambda r: _missing_last(r.measured_soh, descending=False),
    # Trends: "fastest degradation first". A falling SOH trend is negative, so it
    # is ranked ascending; fade is expressed as a positive number, so descending.
    "soh_trend": lambda r: _missing_last(r.soh_trend_pct_per_10, descending=False),
    "fade_trend": lambda r: _missing_last(r.fade_trend_pct_per_10, descending=True),
    "temperature_trend": lambda r: _missing_last(r.temperature_trend_c_per_10, descending=True),
    "resistance_trend": lambda r: _missing_last(r.resistance_trend_pct_per_10, descending=True),
    "data_quality": lambda r: _missing_last(r.data_quality_score, descending=False),
    "uncertainty": lambda r: _missing_last(r.interval_width, descending=True),
}


def rank_batteries(
    records: Sequence[FleetBatteryRecord],
    *,
    by: str = "priority",
    limit: int | None = None,
    include_unevaluated: bool = False,
) -> list[FleetBatteryRecord]:
    """Order cells by one criterion.

    Ties break on ``battery_id`` so the same fleet always produces the same
    order — an unstable ranking makes "the top ten changed" unreadable.

    Cells that could not be evaluated are excluded by default: they have no
    values to rank on, and placing them anywhere in a severity order is a claim
    about them. ``include_unevaluated`` appends them at the end instead.
    """
    if by not in RANKING_KEYS:
        raise ValueError(f"Unknown ranking key {by!r}. Supported: {', '.join(RANKING_KEYS)}")

    evaluated = [r for r in records if r.is_evaluated]
    others = [r for r in records if not r.is_evaluated]

    if by == "priority":
        ordered = sorted(
            evaluated,
            key=lambda r: (
                MaintenancePriority(r.priority).severity,
                -float(r.priority_score),
                r.battery_id,
            ),
        )
    else:
        key_fn = _KEY_FUNCTIONS[by]
        ordered = sorted(evaluated, key=lambda r: (*key_fn(r), r.battery_id))

    if include_unevaluated:
        ordered = [*ordered, *sorted(others, key=lambda r: r.battery_id)]
    if limit is not None:
        ordered = ordered[: max(int(limit), 0)]
    return ordered
