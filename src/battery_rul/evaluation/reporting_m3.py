"""The Milestone 3 fleet report.

Renders stored snapshots into Markdown. It reads persisted artifacts and does
not recompute anything: a report that recalculates its own numbers can disagree
with the snapshot it claims to describe, and then nobody knows which is right.

Every table states its denominator, every predicted quantity is labelled, and
demo fleets are marked at the top of the document rather than in a footnote.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from battery_rul.config import ExperimentConfig
from battery_rul.fleet.domain import FleetSnapshot
from battery_rul.monitoring.domain import Alert, MonitoringSnapshot
from battery_rul.utils.io import atomic_write_text
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["render_fleet_report", "write_fleet_report"]


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f} %"


def render_fleet_report(
    cfg: ExperimentConfig,
    snapshot: FleetSnapshot,
    monitoring: MonitoringSnapshot | None = None,
    alerts: Sequence[Alert] = (),
) -> str:
    """Build the report text."""
    summary = snapshot.summary
    statistics = snapshot.fleet_statistics
    lines: list[str] = []

    lines.append(f"# Fleet report — {snapshot.fleet_id}")
    lines.append("")
    lines.append(f"*Generated {snapshot.generated_at_utc} · snapshot `{snapshot.snapshot_id}`*")
    lines.append("")

    if snapshot.identity.is_demo_data:
        # The notice comes from the fleet's own identity: a derived demo fleet
        # and a synthetic one are different claims, and a hardcoded banner would
        # describe one of them wrongly.
        notice = snapshot.identity.data_notice or (
            "DEMONSTRATION FLEET. This data does not describe a real fleet and no "
            "number in this document is a research result."
        )
        lines.append(f"> **{notice}**")
        lines.append("")

    lines.append("## Executive summary")
    lines.append("")
    lines.append("| Quantity | Value | Basis |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Batteries submitted | {summary.battery_count} | observed |")
    lines.append(
        f"| Successfully evaluated | {summary.successfully_processed_count} | "
        "produced a prediction |"
    )
    lines.append(f"| Failed | {summary.failed_count} | excluded from all aggregates |")
    lines.append(
        f"| Insufficient data | {summary.insufficient_data_count} | input cannot support a "
        "prediction |"
    )
    lines.append(f"| Healthy | {summary.healthy_count} | measured SOH band |")
    lines.append(f"| Slightly degraded | {summary.slightly_degraded_count} | measured SOH band |")
    lines.append(f"| Warning | {summary.warning_count} | measured SOH band |")
    lines.append(f"| Critical | {summary.critical_count} | measured SOH band |")
    lines.append(
        f"| Inspection recommended | {summary.inspection_recommended_count} | rule-based |"
    )
    lines.append(
        f"| Replacement planning | {summary.replacement_planning_count} | rule-based, advisory |"
    )
    lines.append(
        f"| Median SOH | {_percent(statistics.soh_median)} | derived, n={statistics.soh_denominator} |"
    )
    lines.append(
        f"| Median RUL | {_fmt(statistics.rul_median, 1, ' cycles')} | predicted, "
        f"n={statistics.rul_denominator} |"
    )
    lines.append(f"| Data-quality status | {summary.data_quality_status.value} | monitoring |")
    lines.append(f"| Drift status | {summary.drift_status.value} | monitoring |")
    lines.append(
        f"| Active model version | `{summary.active_model_version or 'none'}` | registry/bundle |"
    )
    lines.append("")
    lines.append(
        f"Denominators differ by quantity on purpose: {summary.battery_count} cells were "
        f"submitted, {summary.successfully_processed_count} produced a prediction, and "
        "only those enter the predicted-quantity statistics."
    )
    lines.append("")

    # -- priorities ---------------------------------------------------------
    lines.append("## Maintenance priority distribution")
    lines.append("")
    lines.append("| Priority | Count |")
    lines.append("| --- | --- |")
    for name, count in sorted(snapshot.maintenance_summary.priority_counts.items()):
        lines.append(f"| {name} | {count} |")
    lines.append("")

    # -- top cells ----------------------------------------------------------
    ranked = sorted(
        [r for r in snapshot.batteries if r.is_evaluated],
        key=lambda r: (r.priority.severity, -r.priority_score, r.battery_id),
    )[:15]
    if ranked:
        lines.append("## Highest-priority cells")
        lines.append("")
        lines.append(
            "| Battery | Priority | Score | RUL (pred.) | RUL lower | SOH (meas.) | Risk | "
            "Quality | Action |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for record in ranked:
            risk = "—"
            if record.failure_risk is not None:
                risk = _percent(record.failure_risk) + (
                    " *(experimental)*" if record.risk_is_experimental else ""
                )
            lines.append(
                f"| `{record.battery_id}` | {record.priority.value} | "
                f"{record.priority_score:.1f} | {_fmt(record.predicted_rul, 1)} | "
                f"{_fmt(record.rul_lower_bound, 1)} | {_percent(record.measured_soh)} | "
                f"{risk} | {record.data_quality_class} | {record.recommended_action} |"
            )
        lines.append("")

    # -- workload -----------------------------------------------------------
    lines.append("## Maintenance workload forecast")
    lines.append("")
    lines.append("| Horizon | Cells | % of evaluated | Lower | Upper |")
    lines.append("| --- | --- | --- | --- | --- |")
    for bucket in snapshot.workload_forecast.buckets:
        lines.append(
            f"| {bucket.label} | {bucket.battery_count} | {bucket.percent_of_evaluated:.1f} % | "
            f"{_fmt(bucket.lower_count, 0)} | {_fmt(bucket.upper_count, 0)} |"
        )
    lines.append("")
    lines.append(
        "Lower and upper counts bracket the forecast under the prediction intervals: "
        "lower uses each cell's most optimistic remaining life, upper its most "
        "conservative. This is a workload forecast, not a schedule."
    )
    lines.append("")

    # -- replacement --------------------------------------------------------
    replacement = snapshot.replacement_summary
    lines.append("## Replacement planning (advisory)")
    lines.append("")
    lines.append("| Horizon | Candidates | Lower | Upper |")
    lines.append("| --- | --- | --- | --- |")
    for horizon in ("near_term", "medium_term", "long_term"):
        lines.append(
            f"| {horizon} | {replacement.counts_by_horizon.get(horizon, 0)} | "
            f"{replacement.lower_counts_by_horizon.get(horizon, 0)} | "
            f"{replacement.upper_counts_by_horizon.get(horizon, 0)} |"
        )
    lines.append("")
    for caveat in replacement.caveats:
        lines.append(f"- {caveat}")
    lines.append("")

    # -- monitoring ---------------------------------------------------------
    if monitoring is not None:
        lines.append("## Monitoring")
        lines.append("")
        lines.append(f"- Monitoring snapshot: `{monitoring.snapshot_id}`")
        lines.append(f"- Overall status: **{monitoring.overall_status.value}**")
        drift = monitoring.feature_drift_summary or {}
        if drift:
            lines.append(
                f"- Feature drift: {drift.get('status', 'UNKNOWN')} — "
                f"{drift.get('n_features_drifted', 0)} of "
                f"{drift.get('n_features_tested', 0)} tested features flagged "
                f"(reference `{drift.get('reference_id')}`)"
            )
        prediction = monitoring.prediction_drift_summary or {}
        if prediction:
            lines.append(
                f"- Prediction drift: {prediction.get('status', 'UNKNOWN')} — "
                f"{prediction.get('n_drifted', 0)} quantity/quantities shifted"
            )
        performance = monitoring.performance_summary or {}
        if performance:
            lines.append(
                f"- Delayed-label performance: {performance.get('status')} "
                f"({performance.get('n_labels_joined', 0)} labels joined, coverage "
                f"{performance.get('label_coverage', 0):.1%})"
            )
        lines.append("")
        lines.append(
            "Feature or prediction drift is not evidence that the model has become "
            "less accurate. Only labelled outcomes can show that, and the "
            "delayed-label section above states how many are available."
        )
        lines.append("")

    if alerts:
        lines.append("## Active alerts")
        lines.append("")
        lines.append("| Severity | Type | Message | Recommended human action |")
        lines.append("| --- | --- | --- | --- |")
        for alert in alerts[:20]:
            lines.append(
                f"| {alert.severity.value} | {alert.type.value} | {alert.message} | "
                f"{alert.recommended_human_action} |"
            )
        lines.append("")

    if snapshot.warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in snapshot.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Definitions in force")
    lines.append("")
    lines.append(
        f"- End of life: smoothed capacity at or below {cfg.data.eol_threshold:.0%} of the "
        f"{cfg.data.eol_reference} reference for {cfg.target.eol_persistence} consecutive cycles."
    )
    lines.append(
        f"- Failure risk: projected end-of-life crossing within {cfg.risk.horizon_cycles} "
        "cycles — a derived label, not an observed safety event."
    )
    lines.append(
        f"- Prediction intervals: {cfg.uncertainty.method} at "
        f"{cfg.uncertainty.coverage:.0%} target coverage."
    )
    lines.append(
        "- Priority score: configurable weighted policy "
        f"(weights {cfg.fleet.ranking.weights()}), normalised to "
        f"0–{cfg.fleet.ranking.score_scale:.0f}. Not an optimum."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(snapshot.disclaimer)
    lines.append("")
    return "\n".join(lines)


def write_fleet_report(
    cfg: ExperimentConfig,
    snapshot: FleetSnapshot,
    monitoring: MonitoringSnapshot | None = None,
    alerts: Sequence[Alert] = (),
) -> Path:
    path = cfg.paths.reports_dir / "milestone_3" / "fleet_report.md"
    atomic_write_text(path, render_fleet_report(cfg, snapshot, monitoring, alerts))
    logger.info("Fleet report -> %s", path)
    return path
