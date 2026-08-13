"""Fleet-level data-quality monitoring.

Milestone 2 assesses one cell's input before scoring it. This aggregates those
assessments across a fleet and asks a different question: not "can I score this
cell?" but "is this fleet's telemetry healthy?".

Kept strictly separate from drift. A sensor that stopped reporting and a
population that has genuinely aged both move the numbers, and the remedies are
opposite — one is fixed by an engineer with a screwdriver, the other by
retraining or by accepting the change. Filing them under one heading guarantees
the wrong remedy gets applied about half the time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from battery_rul.config import ExperimentConfig
from battery_rul.digital_twin.domain import DataQualityAssessment
from battery_rul.fleet.domain import (
    FleetBatteryRecord,
    FleetDataQualitySummary,
    MonitoringStatus,
    ProcessingStatus,
)
from battery_rul.monitoring.domain import FleetQualityReport

__all__ = ["fleet_quality_report", "summarise_fleet_data_quality"]


def summarise_fleet_data_quality(
    records: Sequence[FleetBatteryRecord],
    cfg: ExperimentConfig,
    assessments: Mapping[str, DataQualityAssessment] | None = None,
) -> FleetDataQualitySummary:
    """Aggregate per-battery quality into a fleet verdict.

    ``assessments`` (battery id -> the Milestone 2 assessment) unlocks the
    per-feature and per-check detail. Without it the summary still reports class
    counts and scores, because a fleet run that could not collect the detail
    should still say what it knows rather than reporting nothing.
    """
    policy = cfg.monitoring.data_quality
    total = len(records)
    if total == 0:
        return FleetDataQualitySummary(
            status=MonitoringStatus.UNKNOWN,
            warnings=["No batteries were submitted; input quality cannot be assessed."],
        )

    class_counts: dict[str, int] = {}
    for record in records:
        label = record.data_quality_class or "unknown"
        class_counts[label] = class_counts.get(label, 0) + 1

    scores = np.asarray(
        [r.data_quality_score for r in records if r.data_quality_score is not None], dtype=float
    )
    poor_or_worse = sum(
        1 for r in records if r.data_quality_class in ("POOR", "INSUFFICIENT")
    ) + sum(1 for r in records if r.status is ProcessingStatus.FAILED)
    insufficient = sum(
        1
        for r in records
        if r.data_quality_class == "INSUFFICIENT" or r.status is ProcessingStatus.INSUFFICIENT_DATA
    )
    ood_batteries = sorted(r.battery_id for r in records if r.out_of_distribution_feature_count > 0)

    per_feature_missing: dict[str, float] = {}
    check_failure_rates: dict[str, float] = {}
    schema_mismatch: list[str] = []
    missing_fractions: list[float] = []

    if assessments:
        feature_counts: dict[str, int] = {}
        check_counts: dict[str, int] = {}
        check_totals: dict[str, int] = {}
        for battery_id, assessment in assessments.items():
            missing_fractions.append(float(assessment.missing_feature_fraction))
            if assessment.missing_features:
                schema_mismatch.append(battery_id)
            for name in assessment.missing_features:
                feature_counts[name] = feature_counts.get(name, 0) + 1
            for name, payload in (assessment.checks or {}).items():
                check_totals[name] = check_totals.get(name, 0) + 1
                if not payload.get("passed", True):
                    check_counts[name] = check_counts.get(name, 0) + 1
        denominator = max(len(assessments), 1)
        per_feature_missing = {
            name: round(count / denominator, 4)
            for name, count in sorted(feature_counts.items(), key=lambda kv: -kv[1])[:50]
        }
        check_failure_rates = {
            name: round(check_counts.get(name, 0) / max(check_totals[name], 1), 4)
            for name in sorted(check_totals)
        }

    poor_fraction = poor_or_worse / total
    insufficient_fraction = insufficient / total
    mean_missing = float(np.mean(missing_fractions)) if missing_fractions else None

    warnings: list[str] = []
    status = MonitoringStatus.OK
    if poor_fraction >= policy.critical_poor_fraction:
        status = MonitoringStatus.CRITICAL
        warnings.append(
            f"{poor_fraction:.0%} of the fleet has POOR or worse input quality "
            f"(critical threshold {policy.critical_poor_fraction:.0%})."
        )
    elif poor_fraction >= policy.warning_poor_fraction:
        status = MonitoringStatus.WARNING
        warnings.append(
            f"{poor_fraction:.0%} of the fleet has POOR or worse input quality "
            f"(warning threshold {policy.warning_poor_fraction:.0%})."
        )

    if insufficient_fraction >= policy.critical_insufficient_fraction:
        status = MonitoringStatus.CRITICAL
        warnings.append(
            f"{insufficient_fraction:.0%} of the fleet has insufficient data to score "
            f"(critical threshold {policy.critical_insufficient_fraction:.0%})."
        )
    elif insufficient_fraction >= policy.warning_insufficient_fraction:
        status = MonitoringStatus.worst([status, MonitoringStatus.WARNING])
        warnings.append(
            f"{insufficient_fraction:.0%} of the fleet has insufficient data to score "
            f"(warning threshold {policy.warning_insufficient_fraction:.0%})."
        )

    if mean_missing is not None:
        if mean_missing >= policy.critical_missing_feature_fraction:
            status = MonitoringStatus.CRITICAL
            warnings.append(
                f"On average {mean_missing:.0%} of required features are absent "
                f"(critical threshold {policy.critical_missing_feature_fraction:.0%})."
            )
        elif mean_missing >= policy.warning_missing_feature_fraction:
            status = MonitoringStatus.worst([status, MonitoringStatus.WARNING])
            warnings.append(
                f"On average {mean_missing:.0%} of required features are absent "
                f"(warning threshold {policy.warning_missing_feature_fraction:.0%})."
            )

    if ood_batteries:
        warnings.append(
            f"{len(ood_batteries)} cell(s) have at least one feature outside the "
            "training reference range; those predictions are extrapolations. This is "
            "an input-range observation, not a drift verdict — see the feature-drift "
            "report for that."
        )

    return FleetDataQualitySummary(
        status=status,
        quality_class_counts=class_counts,
        mean_quality_score=round(float(scores.mean()), 4) if scores.size else None,
        poor_or_worse_fraction=round(poor_fraction, 4),
        insufficient_fraction=round(insufficient_fraction, 4),
        mean_missing_feature_fraction=(
            round(mean_missing, 4) if mean_missing is not None else None
        ),
        per_feature_missing_rate=per_feature_missing,
        batteries_with_schema_mismatch=sorted(schema_mismatch)[:50],
        batteries_with_ood_features=ood_batteries[:50],
        check_failure_rates=check_failure_rates,
        denominator=total,
        warnings=warnings,
    )


def fleet_quality_report(
    fleet_id: str,
    records: Sequence[FleetBatteryRecord],
    cfg: ExperimentConfig,
    assessments: Mapping[str, DataQualityAssessment] | None = None,
) -> FleetQualityReport:
    """The per-battery quality detail behind the fleet summary."""
    summary = summarise_fleet_data_quality(records, cfg, assessments)
    per_battery = [
        {
            "battery_id": record.battery_id,
            "status": record.status.value,
            "quality_class": record.data_quality_class,
            "quality_score": record.data_quality_score,
            "n_cycles": record.n_cycles,
            "out_of_distribution_features": record.out_of_distribution_feature_count,
            "missing_feature_fraction": (
                assessments[record.battery_id].missing_feature_fraction
                if assessments and record.battery_id in assessments
                else None
            ),
            "warnings": (
                assessments[record.battery_id].warnings[:10]
                if assessments and record.battery_id in assessments
                else record.warnings[:10]
            ),
        }
        for record in records
    ]
    return FleetQualityReport(
        fleet_id=fleet_id,
        status=summary.status,
        per_battery=per_battery,
        summary=summary.model_dump(mode="json"),
        warnings=summary.warnings,
    )
