"""Fleet aggregation — statistics that carry their own denominators.

The rule this module exists to enforce: **a failed or unevaluated cell never
enters a predicted-quantity aggregate, and the number of cells excluded is always
reported beside the number that were included.** "Fleet median RUL 94 cycles" is
a different claim depending on whether it was computed over 128 cells or over the
103 that could be scored, and only the second one is true.

Distributions of measured quantities (health class, SOH) use a separate
denominator from distributions of predicted quantities (RUL, risk), because a
cell can have a measured SOH and no prediction. Merging the two denominators
would quietly change what "the fleet" means between two adjacent numbers on the
same page.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from battery_rul.config import ExperimentConfig
from battery_rul.fleet.domain import (
    FleetBatteryRecord,
    FleetHealthDistribution,
    FleetMaintenanceSummary,
    FleetRiskDistribution,
    FleetStatistics,
    MaintenancePriority,
)

__all__ = [
    "health_distribution",
    "maintenance_summary",
    "risk_distribution",
    "fleet_statistics",
]


def _values(records: Sequence[FleetBatteryRecord], attribute: str) -> np.ndarray:
    raw = [getattr(r, attribute) for r in records if r.is_evaluated]
    finite = [float(v) for v in raw if v is not None and np.isfinite(v)]
    return np.asarray(finite, dtype=float)


def _quantiles(values: np.ndarray, quantiles: Sequence[float]) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {f"q{int(round(100 * q))}": round(float(np.quantile(values, q)), 4) for q in quantiles}


def health_distribution(records: Sequence[FleetBatteryRecord]) -> FleetHealthDistribution:
    """Counts by measured health class over the cells that have one."""
    counts: dict[str, int] = {}
    unknown = 0
    denominator = 0
    for record in records:
        if not record.is_evaluated:
            continue
        label = record.health_class or "unknown"
        if label == "unknown" or record.measured_soh is None:
            unknown += 1
            counts["unknown"] = counts.get("unknown", 0) + 1
            continue
        denominator += 1
        counts[label] = counts.get(label, 0) + 1
    return FleetHealthDistribution(counts=counts, denominator=denominator, unknown_count=unknown)


def risk_distribution(
    records: Sequence[FleetBatteryRecord], cfg: ExperimentConfig
) -> FleetRiskDistribution:
    """Counts by risk class plus the shape of the probability distribution."""
    counts: dict[str, int] = {}
    unknown = 0
    denominator = 0
    experimental = False
    for record in records:
        if not record.is_evaluated:
            continue
        experimental = experimental or record.risk_is_experimental
        if record.failure_risk is None:
            unknown += 1
            counts["unknown"] = counts.get("unknown", 0) + 1
            continue
        denominator += 1
        counts[record.risk_class] = counts.get(record.risk_class, 0) + 1

    values = _values(records, "failure_risk")
    above = {
        str(threshold): int((values >= threshold).sum())
        for threshold in cfg.fleet.high_risk_thresholds
    }
    return FleetRiskDistribution(
        counts=counts,
        denominator=denominator,
        unknown_count=unknown,
        quantiles=_quantiles(values, cfg.fleet.quantiles),
        mean=round(float(values.mean()), 5) if values.size else None,
        experimental_model=experimental,
        above_thresholds=above,
    )


def maintenance_summary(
    records: Sequence[FleetBatteryRecord], cfg: ExperimentConfig
) -> FleetMaintenanceSummary:
    """Priority and action counts over the whole submitted fleet."""
    priority_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for record in records:
        priority_counts[record.priority.value] = priority_counts.get(record.priority.value, 0) + 1
        action_counts[record.recommended_action] = (
            action_counts.get(record.recommended_action, 0) + 1
        )

    critical_levels = set(cfg.fleet.critical_priorities)
    critical = [r for r in records if r.priority.value in critical_levels]
    inspection = [
        r
        for r in records
        if r.priority
        in (
            MaintenancePriority.P0_CRITICAL,
            MaintenancePriority.P1_URGENT,
            MaintenancePriority.P2_HIGH,
        )
    ]
    return FleetMaintenanceSummary(
        priority_counts=priority_counts,
        action_counts=action_counts,
        critical_count=len(critical),
        inspection_recommended_count=len(inspection),
        insufficient_data_count=priority_counts.get(MaintenancePriority.INSUFFICIENT_DATA.value, 0),
        denominator=len(records),
        high_risk_battery_ids=sorted(r.battery_id for r in critical),
    )


def fleet_statistics(
    records: Sequence[FleetBatteryRecord], cfg: ExperimentConfig
) -> FleetStatistics:
    """Numeric summaries, each over the cells that actually carry the quantity."""
    quantiles = cfg.fleet.quantiles
    submitted = max(len(records), 1)

    soh = _values(records, "measured_soh")
    rul = _values(records, "predicted_rul")
    rul_lower = _values(records, "rul_lower_bound")
    width = _values(records, "interval_width")
    risk = _values(records, "failure_risk")

    below = {
        str(threshold): int((rul <= threshold).sum()) for threshold in cfg.fleet.low_rul_thresholds
    }
    above = {
        str(threshold): int((risk >= threshold).sum())
        for threshold in cfg.fleet.high_risk_thresholds
    }
    missingness = {
        "measured_soh": round(1.0 - soh.size / submitted, 4),
        "predicted_rul": round(1.0 - rul.size / submitted, 4),
        "rul_lower_bound": round(1.0 - rul_lower.size / submitted, 4),
        "failure_risk": round(1.0 - risk.size / submitted, 4),
    }

    return FleetStatistics(
        soh_median=round(float(np.median(soh)), 5) if soh.size else None,
        soh_mean=round(float(soh.mean()), 5) if soh.size else None,
        soh_quantiles=_quantiles(soh, quantiles),
        soh_denominator=int(soh.size),
        rul_median=round(float(np.median(rul)), 3) if rul.size else None,
        rul_mean=round(float(rul.mean()), 3) if rul.size else None,
        rul_quantiles=_quantiles(rul, quantiles),
        rul_denominator=int(rul.size),
        rul_lower_bound_median=round(float(np.median(rul_lower)), 3) if rul_lower.size else None,
        interval_width_median=round(float(np.median(width)), 3) if width.size else None,
        risk_median=round(float(np.median(risk)), 5) if risk.size else None,
        risk_mean=round(float(risk.mean()), 5) if risk.size else None,
        risk_denominator=int(risk.size),
        below_rul_threshold_counts=below,
        above_risk_threshold_counts=above,
        missingness=missingness,
    )
