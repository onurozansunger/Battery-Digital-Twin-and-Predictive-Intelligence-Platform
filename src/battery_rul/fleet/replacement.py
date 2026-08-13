"""Replacement planning and maintenance-workload forecasting.

Advisory, and the word matters. This module answers "which cells might need
replacing within 20 / 50 / 100 cycles, and how confident is that?" — it does not
schedule anything, does not raise a purchase order, and does not convert cells
into money. Cost per replacement, downtime, spare availability and fleet
utilisation are all inputs this platform does not have; a savings figure computed
without them would be fiction with a currency symbol on it.

Uncertainty-aware counts
------------------------
Every horizon reports three counts:

``count``        cells whose planning quantity falls inside the horizon
``lower_count``  cells whose **upper** RUL bound falls inside it — the optimistic
                 reading, and the smallest defensible number
``upper_count``  cells whose **lower** RUL bound falls inside it — the
                 conservative reading, and the largest defensible number

A single number would be an assertion the intervals do not support. Three
numbers make the width of the intervals visible in the plan itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from battery_rul.config import ExperimentConfig
from battery_rul.fleet.domain import (
    FleetBatteryRecord,
    FleetReplacementSummary,
    FleetWorkloadForecast,
    MaintenancePriority,
    ReplacementCandidate,
    ReplacementHorizon,
    WorkloadBucket,
)
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "ReplacementPlanner",
    "summarise_replacements",
    "workload_forecast",
]

_CAVEATS = (
    "Replacement horizons are advisory planning input derived from model "
    "predictions and configurable thresholds, not a maintenance schedule.",
    "Remaining-life predictions carry interval-width uncertainty; the lower and "
    "upper counts bracket the plan under those intervals.",
    "No cost, downtime or spares-availability assumption is applied, so these "
    "counts must not be converted into financial figures without them.",
)


@dataclass
class ReplacementPlanner:
    """Classifies one cell into a replacement horizon, with its evidence."""

    cfg: ExperimentConfig

    def evaluate(self, record: FleetBatteryRecord) -> ReplacementCandidate:
        policy = self.cfg.fleet.replacement

        if not record.is_evaluated:
            return ReplacementCandidate(
                battery_id=record.battery_id,
                replacement_candidate=False,
                replacement_horizon=ReplacementHorizon.UNKNOWN,
                confidence="unknown",
                planning_category="not_evaluated",
                evidence=[
                    f"No replacement assessment: the cell was not evaluated "
                    f"(status {record.status.value})."
                ],
                caveats=list(_CAVEATS),
            )

        planning_rul = _planning_rul(record, use_lower_bound=policy.use_lower_bound)
        evidence: list[str] = []
        triggers: list[str] = []

        horizon = ReplacementHorizon.NOT_FLAGGED
        horizon_cycles: int | None = None
        if planning_rul is not None:
            basis = (
                "RUL interval lower bound"
                if (policy.use_lower_bound and record.rul_lower_bound is not None)
                else "RUL point estimate"
            )
            evidence.append(f"Planning remaining life {planning_rul:.0f} cycles ({basis}).")
            if planning_rul <= policy.near_term_cycles:
                horizon, horizon_cycles = ReplacementHorizon.NEAR_TERM, policy.near_term_cycles
                triggers.append(f"planning RUL <= {policy.near_term_cycles}")
            elif planning_rul <= policy.medium_term_cycles:
                horizon, horizon_cycles = ReplacementHorizon.MEDIUM_TERM, policy.medium_term_cycles
                triggers.append(f"planning RUL <= {policy.medium_term_cycles}")
            elif planning_rul <= policy.long_term_cycles:
                horizon, horizon_cycles = ReplacementHorizon.LONG_TERM, policy.long_term_cycles
                triggers.append(f"planning RUL <= {policy.long_term_cycles}")
        else:
            evidence.append("No remaining-life estimate is available for this cell.")

        risk = None if record.risk_is_experimental else record.failure_risk
        if risk is not None and risk >= policy.risk_candidate_threshold:
            triggers.append(f"calibrated risk {risk:.2f} >= {policy.risk_candidate_threshold}")
            evidence.append(
                f"Calibrated probability of reaching end of life within "
                f"{record.risk_horizon_cycles} cycles: {100 * risk:.0f} %."
            )
            if horizon in (ReplacementHorizon.NOT_FLAGGED, ReplacementHorizon.UNKNOWN):
                horizon, horizon_cycles = ReplacementHorizon.MEDIUM_TERM, policy.medium_term_cycles

        if (
            record.measured_soh is not None
            and record.measured_soh <= policy.soh_candidate_threshold
        ):
            triggers.append(
                f"measured SOH {record.measured_soh:.3f} <= {policy.soh_candidate_threshold}"
            )
            evidence.append(f"Measured state of health {100 * record.measured_soh:.1f} %.")
            if horizon in (ReplacementHorizon.NOT_FLAGGED, ReplacementHorizon.UNKNOWN):
                horizon, horizon_cycles = ReplacementHorizon.NEAR_TERM, policy.near_term_cycles

        if record.priority in (MaintenancePriority.P0_CRITICAL, MaintenancePriority.P1_URGENT):
            triggers.append(f"maintenance priority {record.priority.value}")
            if horizon in (ReplacementHorizon.NOT_FLAGGED, ReplacementHorizon.UNKNOWN):
                horizon, horizon_cycles = ReplacementHorizon.NEAR_TERM, policy.near_term_cycles

        candidate = horizon in (
            ReplacementHorizon.NEAR_TERM,
            ReplacementHorizon.MEDIUM_TERM,
            ReplacementHorizon.LONG_TERM,
        )
        confidence = self._confidence(record, planning_rul)
        if triggers:
            evidence.append("Triggered: " + "; ".join(triggers) + ".")

        caveats = list(_CAVEATS)
        if record.risk_is_experimental and record.failure_risk is not None:
            caveats.append(
                "The failure-risk model is marked experimental and was not used as "
                "evidence for this classification."
            )
        if record.rul_lower_bound is None and planning_rul is not None:
            caveats.append(
                "No prediction interval was available, so the point estimate was used "
                "as the planning quantity. That is less conservative than intended."
            )

        return ReplacementCandidate(
            battery_id=record.battery_id,
            replacement_candidate=candidate,
            replacement_horizon=horizon,
            horizon_cycles=horizon_cycles,
            confidence=confidence,
            planning_category=_category(horizon, candidate),
            evidence=evidence,
            caveats=caveats,
            rul_point=record.predicted_rul,
            rul_lower_bound=record.rul_lower_bound,
            rul_upper_bound=record.rul_upper_bound,
        )

    def _confidence(
        self, record: FleetBatteryRecord, planning_rul: float | None
    ) -> Literal["high", "medium", "low", "unknown"]:
        """How much weight the horizon can bear.

        Driven by interval width relative to the estimate and by input quality —
        not by the model's confidence in itself, which it does not report.
        """
        if planning_rul is None:
            return "unknown"
        if record.data_quality_class in ("POOR", "INSUFFICIENT"):
            return "low"
        width = record.interval_width
        if width is None:
            return "low"
        point = record.predicted_rul or 0.0
        ratio = width / max(point, 1.0)
        if ratio >= self.cfg.fleet.replacement.wide_interval_ratio:
            return "low"
        if ratio >= 0.5 * self.cfg.fleet.replacement.wide_interval_ratio:
            return "medium"
        return "high" if record.data_quality_class == "GOOD" else "medium"


def _planning_rul(record: FleetBatteryRecord, *, use_lower_bound: bool) -> float | None:
    candidates = (
        (record.rul_lower_bound, record.predicted_rul)
        if use_lower_bound
        else (record.predicted_rul,)
    )
    for value in candidates:
        if value is not None and np.isfinite(value):
            return float(value)
    return None


def _category(horizon: ReplacementHorizon, candidate: bool) -> str:
    if not candidate:
        return "no_action" if horizon is ReplacementHorizon.NOT_FLAGGED else "not_evaluated"
    return {
        ReplacementHorizon.NEAR_TERM: "near_term_replacement",
        ReplacementHorizon.MEDIUM_TERM: "medium_term_replacement",
        ReplacementHorizon.LONG_TERM: "long_term_monitoring",
    }[horizon]


def summarise_replacements(
    records: Sequence[FleetBatteryRecord],
    candidates: Sequence[ReplacementCandidate],
    cfg: ExperimentConfig,
) -> FleetReplacementSummary:
    """Aggregate candidates by horizon, with uncertainty-aware brackets."""
    policy = cfg.fleet.replacement
    evaluated = [r for r in records if r.is_evaluated]
    by_id = {r.battery_id: r for r in evaluated}

    counts: dict[str, int] = {h.value: 0 for h in ReplacementHorizon}
    for candidate in candidates:
        counts[candidate.replacement_horizon.value] = (
            counts.get(candidate.replacement_horizon.value, 0) + 1
        )

    horizons = {
        ReplacementHorizon.NEAR_TERM.value: policy.near_term_cycles,
        ReplacementHorizon.MEDIUM_TERM.value: policy.medium_term_cycles,
        ReplacementHorizon.LONG_TERM.value: policy.long_term_cycles,
    }
    # Optimistic reading: give every cell its longest defensible life, so the
    # fewest cells fall inside the horizon. Conservative reading: the shortest.
    optimistic = [
        _finite(r.rul_upper_bound if r.rul_upper_bound is not None else r.predicted_rul)
        for r in evaluated
    ]
    conservative = [
        _finite(r.rul_lower_bound if r.rul_lower_bound is not None else r.predicted_rul)
        for r in evaluated
    ]
    lower: dict[str, int] = {}
    upper: dict[str, int] = {}
    for name, cycles in horizons.items():
        lower[name] = sum(1 for value in optimistic if value is not None and value <= cycles)
        upper[name] = sum(1 for value in conservative if value is not None and value <= cycles)

    flagged = [
        c.battery_id for c in candidates if c.replacement_candidate and c.battery_id in by_id
    ]
    return FleetReplacementSummary(
        counts_by_horizon=counts,
        lower_counts_by_horizon=lower,
        upper_counts_by_horizon=upper,
        candidate_count=len(flagged),
        denominator=len(evaluated),
        candidate_battery_ids=sorted(flagged),
        caveats=list(_CAVEATS),
    )


def workload_forecast(
    records: Sequence[FleetBatteryRecord], cfg: ExperimentConfig
) -> FleetWorkloadForecast:
    """Group evaluated cells into maintenance-demand horizons.

    "Immediate" is priority-driven (P0/P1 need attention whatever their RUL
    says); the remaining buckets are driven by the planning remaining-life
    quantity. Cells with no remaining-life estimate land in ``monitor_only`` or
    ``insufficient_data`` rather than being assigned a horizon they do not
    support.
    """
    horizons = cfg.fleet.workload.horizons_cycles
    evaluated = [r for r in records if r.is_evaluated]
    excluded = [r for r in records if not r.is_evaluated]
    denominator = max(len(evaluated), 1)

    buckets: dict[str, list[FleetBatteryRecord]] = {"immediate": []}
    for horizon in horizons:
        buckets[f"next_{horizon}_cycles"] = []
    buckets[f"beyond_{horizons[-1]}_cycles" if horizons else "beyond"] = []
    buckets["monitor_only"] = []
    buckets["insufficient_data"] = list(excluded)

    for record in evaluated:
        if record.priority in (MaintenancePriority.P0_CRITICAL, MaintenancePriority.P1_URGENT):
            buckets["immediate"].append(record)
            continue
        if record.priority is MaintenancePriority.INSUFFICIENT_DATA:
            buckets["insufficient_data"].append(record)
            continue
        planning = _planning_rul(record, use_lower_bound=True)
        if planning is None:
            buckets["monitor_only"].append(record)
            continue
        placed = False
        for horizon in horizons:
            if planning <= horizon:
                buckets[f"next_{horizon}_cycles"].append(record)
                placed = True
                break
        if not placed:
            buckets[f"beyond_{horizons[-1]}_cycles" if horizons else "beyond"].append(record)

    horizon_cycles = {"immediate": 0}
    for horizon in horizons:
        horizon_cycles[f"next_{horizon}_cycles"] = horizon

    out: list[WorkloadBucket] = []
    for label, members in buckets.items():
        cycles = horizon_cycles.get(label)
        lower_count = upper_count = None
        if cycles:
            lower_count = sum(
                1
                for r in evaluated
                if _below(
                    r.rul_upper_bound if r.rul_upper_bound is not None else r.predicted_rul, cycles
                )
            )
            upper_count = sum(
                1
                for r in evaluated
                if _below(
                    r.rul_lower_bound if r.rul_lower_bound is not None else r.predicted_rul, cycles
                )
            )
        priority_counts: dict[str, int] = {}
        for record in members:
            priority_counts[record.priority.value] = (
                priority_counts.get(record.priority.value, 0) + 1
            )
        out.append(
            WorkloadBucket(
                label=label,
                horizon_cycles=cycles,
                battery_count=len(members),
                percent_of_evaluated=round(100.0 * len(members) / denominator, 2),
                priority_counts=priority_counts,
                lower_count=lower_count,
                upper_count=upper_count,
                battery_ids=sorted(r.battery_id for r in members)[:200],
            )
        )

    return FleetWorkloadForecast(
        buckets=out,
        evaluated_count=len(evaluated),
        excluded_count=len(excluded),
        basis=(
            "Cells at priority P0/P1 are counted as immediate; the remaining cells are "
            "bucketed by the RUL interval lower bound where one exists, otherwise by "
            "the point estimate."
        ),
        caveats=[
            "A workload forecast, not a schedule: it describes expected demand under "
            "the configured policy, and does not account for crew availability, "
            "spares, or operational constraints.",
            "Percentages are of the evaluated fleet, not of the submitted fleet; "
            f"{len(excluded)} cell(s) could not be evaluated.",
        ],
    )


def _finite(value: float | None) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None


def _below(value: float | None, threshold: float) -> bool:
    finite = _finite(value)
    return finite is not None and finite <= threshold
