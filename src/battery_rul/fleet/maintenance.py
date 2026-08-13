"""The fleet maintenance-priority engine.

Deterministic rules over model *outputs*, kept out of the model for the same
three reasons the battery-level recommendation engine is (auditability,
changeability, separate failure modes) plus a fourth that only appears at fleet
scale: an operator comparing 128 cells needs the comparison to be stable. A
learned policy that re-ranks the fleet when it is retrained gives an operations
team no way to say "what changed since yesterday".

The ladder
----------
``P0_CRITICAL``        critical measured health, or very high risk together with
                       a very low RUL lower bound, or a critical rule-based data
                       warning
``P1_URGENT``          high risk *and* an RUL lower bound below the urgent
                       threshold
``P2_HIGH``            warning health class, or an elevated degradation trend
``P3_MEDIUM``          slight degradation with a meaningful trend
``P4_LOW``             stable and healthy
``P5_MONITOR``         healthy, but poorly characterised: a wide interval, thin
                       history, or out-of-distribution features
``INSUFFICIENT_DATA``  the input cannot support a maintenance decision

Rules fire on the **lower bound** of the RUL interval wherever one exists. A
point estimate of 45 cycles with a lower bound of 12 is not a 45-cycle
situation, and a fleet plan built on the middle of wide intervals strands cells.

An experimental risk model contributes nothing. Milestone 2 gates the risk
probability on beating a cycle-index baseline out of fold; when it does not, the
service marks it experimental and this engine treats it as absent rather than as
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from battery_rul.config import ExperimentConfig
from battery_rul.digital_twin.domain import Provenance
from battery_rul.fleet.domain import (
    BatteryPriorityRecord,
    FleetBatteryRecord,
    InspectionRecommendation,
    MaintenancePriority,
)
from battery_rul.fleet.ranking import compute_priority_score
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "FLEET_ACTIONS",
    "MaintenancePriorityEngine",
    "inspection_recommendation",
]

#: The closed set of fleet-level actions. Deliberately a superset of nothing:
#: each maps to exactly one priority, so an action can never disagree with a
#: severity.
FLEET_ACTIONS: dict[MaintenancePriority, tuple[str, str]] = {
    MaintenancePriority.P0_CRITICAL: (
        "IMMEDIATE_ENGINEERING_REVIEW",
        "Remove from planned duty and review immediately",
    ),
    MaintenancePriority.P1_URGENT: ("PLAN_REPLACEMENT", "Begin replacement planning"),
    MaintenancePriority.P2_HIGH: ("SCHEDULE_INSPECTION", "Schedule an inspection"),
    MaintenancePriority.P3_MEDIUM: (
        "MONITOR_MORE_FREQUENTLY",
        "Increase monitoring cadence",
    ),
    MaintenancePriority.P4_LOW: ("NORMAL_OPERATION", "Continue normal operation"),
    MaintenancePriority.P5_MONITOR: (
        "MONITOR_UNCERTAIN_ESTIMATE",
        "Keep under observation — the estimate is poorly characterised",
    ),
    MaintenancePriority.INSUFFICIENT_DATA: (
        "INSUFFICIENT_DATA",
        "No maintenance decision — the input cannot support one",
    ),
}


@dataclass
class MaintenancePriorityEngine:
    """Pure, deterministic, separately testable.

    Constructed from an :class:`ExperimentConfig`; every threshold it reads lives
    in ``fleet.maintenance``. Given the same record it always returns the same
    verdict, which is what lets each rule be tested on its own.
    """

    cfg: ExperimentConfig

    def evaluate(
        self,
        record: FleetBatteryRecord,
        *,
        cycles_per_day: float | None = None,
        critical_data_warning: bool = False,
    ) -> BatteryPriorityRecord:
        """Assign a priority to one cell and show the whole argument."""
        policy = self.cfg.fleet.maintenance
        triggered: list[str] = []

        if not record.is_evaluated or record.data_quality_class in (
            policy.insufficient_quality_classes
        ):
            return self._insufficient(record, triggered)

        risk = None if record.risk_is_experimental else record.failure_risk
        if record.risk_is_experimental and record.failure_risk is not None:
            triggered.append(
                "risk_withheld: the failure-risk model failed its out-of-fold "
                "acceptance gate, so its probability is not used as evidence"
            )
        # The conservative planning quantity: the lower bound when one exists.
        planning_rul = _first_finite(record.rul_lower_bound, record.predicted_rul)
        planning_basis = (
            "RUL interval lower bound"
            if record.rul_lower_bound is not None
            else "RUL point estimate (no interval available)"
        )

        priority = self._classify(record, risk=risk, planning_rul=planning_rul, triggered=triggered)
        if critical_data_warning:
            triggered.append("critical_data_warning: a rule-based data warning forced P0")
            priority = MaintenancePriority.P0_CRITICAL

        override = priority is MaintenancePriority.P0_CRITICAL
        scored = compute_priority_score(record, self.cfg.fleet.ranking, critical_override=override)
        action_code, action_title = FLEET_ACTIONS[priority]
        inspection = inspection_recommendation(
            priority, self.cfg, cycles_per_day=cycles_per_day, planning_rul=planning_rul
        )

        return BatteryPriorityRecord(
            battery_id=record.battery_id,
            priority=priority,
            priority_score=scored.score,
            score_breakdown=scored.components,
            triggered_rules=triggered,
            evidence=self._evidence(record, risk=risk, planning_basis=planning_basis),
            recommended_action=action_code,
            action_title=action_title,
            inspection=inspection,
            critical_override=scored.critical_override_applied,
            disclaimer=policy.disclaimer,
            provenance=Provenance.RULE_BASED,
        )

    # -- the ladder --------------------------------------------------------
    def _classify(
        self,
        record: FleetBatteryRecord,
        *,
        risk: float | None,
        planning_rul: float | None,
        triggered: list[str],
    ) -> MaintenancePriority:
        policy = self.cfg.fleet.maintenance

        if _at_or_below(record.measured_soh, policy.critical_soh):
            triggered.append(
                f"critical_soh: measured SOH {record.measured_soh:.3f} is at or below "
                f"{policy.critical_soh}"
            )
            return MaintenancePriority.P0_CRITICAL
        if _at_or_above(risk, policy.critical_risk) and _at_or_below(
            planning_rul, policy.critical_rul_lower_cycles
        ):
            triggered.append(
                f"critical_risk_and_rul: risk {risk:.3f} >= {policy.critical_risk} with a "
                f"planning RUL of {planning_rul:.1f} <= {policy.critical_rul_lower_cycles} cycles"
            )
            return MaintenancePriority.P0_CRITICAL
        if _at_or_below(planning_rul, policy.critical_rul_lower_cycles) and risk is None:
            # No usable risk probability, but the interval itself already places
            # the cell within a handful of cycles of end of life.
            triggered.append(
                f"critical_rul: planning RUL {planning_rul:.1f} <= "
                f"{policy.critical_rul_lower_cycles} cycles"
            )
            return MaintenancePriority.P0_CRITICAL

        if _at_or_above(risk, policy.urgent_risk) and _at_or_below(
            planning_rul, policy.urgent_rul_lower_cycles
        ):
            triggered.append(
                f"urgent: risk {risk:.3f} >= {policy.urgent_risk} and planning RUL "
                f"{planning_rul:.1f} <= {policy.urgent_rul_lower_cycles} cycles"
            )
            return MaintenancePriority.P1_URGENT
        if risk is None and _at_or_below(planning_rul, policy.urgent_rul_lower_cycles):
            triggered.append(
                f"urgent_rul_only: planning RUL {planning_rul:.1f} <= "
                f"{policy.urgent_rul_lower_cycles} cycles, with no usable risk probability"
            )
            return MaintenancePriority.P1_URGENT

        if record.health_class == "warning":
            triggered.append("high_warning_health: measured health class is 'warning'")
            return MaintenancePriority.P2_HIGH
        if _at_or_below(planning_rul, policy.high_rul_lower_cycles):
            triggered.append(
                f"high_rul: planning RUL {planning_rul:.1f} <= "
                f"{policy.high_rul_lower_cycles} cycles"
            )
            return MaintenancePriority.P2_HIGH
        if _at_or_above(record.fade_trend_pct_per_10, policy.high_fade_trend_pct_per_10):
            triggered.append(
                f"high_trend: capacity fade {record.fade_trend_pct_per_10:.2f} % per 10 "
                f"cycles >= {policy.high_fade_trend_pct_per_10}"
            )
            return MaintenancePriority.P2_HIGH

        if record.health_class == "slightly_degraded" and _at_or_above(
            record.fade_trend_pct_per_10, policy.medium_fade_trend_pct_per_10
        ):
            triggered.append(
                "medium_degrading: slight degradation with a fade trend of "
                f"{record.fade_trend_pct_per_10:.2f} % per 10 cycles >= "
                f"{policy.medium_fade_trend_pct_per_10}"
            )
            return MaintenancePriority.P3_MEDIUM

        wide = _at_or_above(record.interval_width, policy.monitor_interval_width_cycles)
        thin = record.data_quality_class in ("POOR", "ACCEPTABLE")
        ood = record.out_of_distribution_feature_count > 0
        if wide or thin or ood:
            reasons = []
            if wide:
                reasons.append(
                    f"interval width {record.interval_width:.0f} cycles >= "
                    f"{policy.monitor_interval_width_cycles}"
                )
            if thin:
                reasons.append(f"data quality {record.data_quality_class}")
            if ood:
                reasons.append(
                    f"{record.out_of_distribution_feature_count} feature(s) outside the "
                    "training range"
                )
            triggered.append("monitor_uncertain: " + "; ".join(reasons))
            return MaintenancePriority.P5_MONITOR

        triggered.append("low: no rule threshold is met")
        return MaintenancePriority.P4_LOW

    # -- helpers -----------------------------------------------------------
    def _insufficient(
        self, record: FleetBatteryRecord, triggered: list[str]
    ) -> BatteryPriorityRecord:
        policy = self.cfg.fleet.maintenance
        reason = (
            f"data quality is {record.data_quality_class}"
            if record.data_quality_class in policy.insufficient_quality_classes
            else f"the cell was not successfully evaluated (status: {record.status.value})"
        )
        triggered.append(f"insufficient_data: {reason}")
        action_code, action_title = FLEET_ACTIONS[MaintenancePriority.INSUFFICIENT_DATA]
        return BatteryPriorityRecord(
            battery_id=record.battery_id,
            priority=MaintenancePriority.INSUFFICIENT_DATA,
            priority_score=0.0,
            score_breakdown=[],
            triggered_rules=triggered,
            evidence=[
                f"No maintenance priority was assigned because {reason}.",
                *(record.errors or []),
            ],
            recommended_action=action_code,
            action_title=action_title,
            inspection=InspectionRecommendation(
                recommended_cycles=None,
                recommended_label="insufficient_data",
                basis="No priority could be assigned, so no inspection window is implied.",
                assumptions=["Supply more cycles of history, or verify sensor coverage."],
            ),
            disclaimer=policy.disclaimer,
        )

    def _evidence(
        self, record: FleetBatteryRecord, *, risk: float | None, planning_basis: str
    ) -> list[str]:
        items: list[str] = []
        if record.measured_soh is not None:
            items.append(
                f"Measured state of health {100 * record.measured_soh:.1f} % "
                f"({record.health_class}) — derived from measured capacity."
            )
        if record.predicted_rul is not None:
            if record.rul_lower_bound is not None and record.rul_upper_bound is not None:
                items.append(
                    f"Predicted remaining useful life {record.predicted_rul:.0f} cycles "
                    f"(interval {record.rul_lower_bound:.0f}–{record.rul_upper_bound:.0f})."
                )
            else:
                items.append(
                    f"Predicted remaining useful life {record.predicted_rul:.0f} cycles "
                    "(no prediction interval available)."
                )
        items.append(f"Planning quantity: {planning_basis}.")
        if risk is not None:
            items.append(
                f"Calibrated probability of reaching end of life within "
                f"{record.risk_horizon_cycles} cycles: {100 * risk:.0f}% ({record.risk_class})."
            )
        elif record.failure_risk is not None:
            items.append(
                f"Failure-risk probability {100 * record.failure_risk:.0f}% is reported but "
                "was withheld from these rules: the model is marked experimental."
            )
        for label, value, unit in (
            ("Capacity-fade trend", record.fade_trend_pct_per_10, "% per 10 cycles"),
            ("Temperature trend", record.temperature_trend_c_per_10, "°C per 10 cycles"),
            ("Resistance trend", record.resistance_trend_pct_per_10, "% per 10 cycles"),
        ):
            if value is not None:
                items.append(f"{label} {value:+.2f} {unit}.")
        items.append(
            f"Input data quality: {record.data_quality_class}"
            + (
                f" (score {record.data_quality_score:.2f})."
                if record.data_quality_score is not None
                else "."
            )
        )
        return items


def inspection_recommendation(
    priority: MaintenancePriority,
    cfg: ExperimentConfig,
    *,
    cycles_per_day: float | None = None,
    planning_rul: float | None = None,
) -> InspectionRecommendation:
    """Translate a priority into an inspection window.

    Cycles are the primary output because cycles are what the model reasons in.
    A calendar estimate is produced **only** when a recent cycles-per-day rate
    was measurable from timestamps; without one there is no conversion, and
    inventing a duty cycle would turn "within 10 cycles" into a date that means
    nothing.
    """
    policy = cfg.fleet.maintenance
    windows = policy.inspection_window_cycles
    assumptions: list[str] = []

    if priority is MaintenancePriority.INSUFFICIENT_DATA:
        return InspectionRecommendation(
            recommended_cycles=None,
            recommended_label="insufficient_data",
            basis="No priority could be assigned.",
            assumptions=["Supply more cycles of history, or verify sensor coverage."],
        )

    cycles = windows.get(priority.value)
    if priority is MaintenancePriority.P0_CRITICAL:
        label = "immediate_engineering_review"
        cycles = 0
        basis = "P0_CRITICAL: review before further cycling."
    elif priority in (MaintenancePriority.P4_LOW, MaintenancePriority.P5_MONITOR):
        label = "next_scheduled_inspection"
        cycles = cycles if cycles else None
        basis = (
            f"{priority.value}: no separate inspection is implied; review at the next "
            "scheduled maintenance."
        )
    else:
        label = f"within_{cycles}_cycles"
        basis = (
            f"{priority.value}: configured inspection window of {cycles} cycles from "
            "fleet.maintenance.inspection_window_cycles."
        )

    if planning_rul is not None and cycles:
        # Never recommend inspecting after the cell is expected to be gone.
        bounded = int(min(cycles, max(planning_rul, 0)))
        if bounded < cycles:
            assumptions.append(
                f"Window shortened from {cycles} to {bounded} cycles: the planning "
                "remaining-life estimate is shorter than the policy window."
            )
            cycles = bounded

    estimated_days: float | None = None
    if cycles is not None and cycles_per_day and cycles_per_day > 0:
        estimated_days = round(float(cycles) / float(cycles_per_day), 2)
        assumptions.append(
            f"Calendar estimate assumes the recent duty cycle continues "
            f"({cycles_per_day:.2f} cycles/day, measured from the supplied timestamps)."
        )
    elif cycles is not None:
        assumptions.append(
            "No calendar estimate: the supplied history does not carry enough "
            "timestamps to measure a cycles-per-day rate, and a cycle-to-day "
            "conversion is not assumed."
        )

    return InspectionRecommendation(
        recommended_cycles=cycles,
        recommended_label=label,
        estimated_days=estimated_days,
        cycles_per_day=cycles_per_day if (cycles_per_day and cycles_per_day > 0) else None,
        basis=basis,
        assumptions=assumptions,
    )


def _first_finite(*values: float | None) -> float | None:
    for value in values:
        if value is not None and np.isfinite(value):
            return float(value)
    return None


def _at_or_below(value: float | None, threshold: float) -> bool:
    return value is not None and bool(np.isfinite(value)) and value <= threshold


def _at_or_above(value: float | None, threshold: float) -> bool:
    return value is not None and bool(np.isfinite(value)) and value >= threshold
