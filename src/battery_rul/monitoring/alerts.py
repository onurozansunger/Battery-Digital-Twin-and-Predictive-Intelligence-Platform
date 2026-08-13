"""Alert policy — turning findings into things a human should look at.

Two constraints shape this module.

**No external notification.** Alerts are written locally and served from an
endpoint. Wiring a pager needs credentials and an on-call rota this repository
does not have, and a notifier configured with placeholder values is worse than
none: it looks like coverage.

**Every alert names a human action.** ``recommended_human_action`` is a required
field. Nothing here retrains, promotes, quarantines or takes a cell out of
service — a drift alert is evidence for a decision, and the decision belongs to
an engineer who can also see the sensor logs and the duty schedule.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from battery_rul.config import ExperimentConfig
from battery_rul.fleet.domain import FleetSnapshot, MonitoringStatus
from battery_rul.monitoring.domain import (
    Alert,
    AlertSeverity,
    AlertType,
    FeatureDriftReport,
    PerformanceReport,
    PerformanceStatus,
    PredictionDriftReport,
)
from battery_rul.observability.metrics import METRICS
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["AlertPolicy", "evaluate_alerts"]


def _alert_id(*parts: str) -> str:
    """Deterministic id from the alert's content.

    Deterministic on purpose: the same finding in two runs produces the same id,
    so a persistence layer can deduplicate and an acknowledgement survives the
    next run instead of being buried under a fresh UUID.
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"alert-{digest[:16]}"


@dataclass
class AlertPolicy:
    """Maps monitoring findings onto alerts, under configurable muting."""

    cfg: ExperimentConfig

    def build(
        self,
        *,
        fleet_snapshot: FleetSnapshot | None = None,
        feature_drift: FeatureDriftReport | None = None,
        prediction_drift: PredictionDriftReport | None = None,
        performance: PerformanceReport | None = None,
        readiness: dict | None = None,
        model_version: str | None = None,
    ) -> list[Alert]:
        policy = self.cfg.monitoring.alerts
        if not policy.enabled:
            return []

        alerts: list[Alert] = []
        fleet_id = fleet_snapshot.fleet_id if fleet_snapshot else None
        batch_id = fleet_snapshot.batch_id if fleet_snapshot else None
        version = model_version or (
            fleet_snapshot.model_metadata.active_model_version if fleet_snapshot else None
        )

        def add(
            alert_type: AlertType,
            severity: AlertSeverity,
            message: str,
            *,
            evidence: Sequence[str] = (),
            threshold: float | None = None,
            observed: float | None = None,
            action: str,
            key: str = "",
        ) -> None:
            if alert_type.value in policy.muted_types:
                return
            alerts.append(
                Alert(
                    alert_id=_alert_id(alert_type.value, fleet_id or "", batch_id or "", key),
                    type=alert_type,
                    severity=severity,
                    generated_at_utc=datetime.now(UTC).isoformat(),
                    fleet_id=fleet_id,
                    model_version=version,
                    batch_id=batch_id,
                    message=message,
                    evidence=list(evidence),
                    threshold=threshold,
                    observed_value=observed,
                    recommended_human_action=action,
                )
            )

        if fleet_snapshot is not None:
            self._fleet_alerts(fleet_snapshot, add)
        if feature_drift is not None:
            self._feature_drift_alerts(feature_drift, add)
        if prediction_drift is not None:
            self._prediction_drift_alerts(prediction_drift, add)
        if performance is not None:
            self._performance_alerts(performance, add)
        if readiness is not None and not readiness.get("ready", True):
            add(
                AlertType.MODEL_UNAVAILABLE,
                AlertSeverity.CRITICAL,
                "No model artifact is loaded; prediction endpoints cannot answer.",
                evidence=[f"{k}: {v}" for k, v in (readiness.get("errors") or {}).items()][:10],
                action=(
                    "Check the artifact directory and the bundle compatibility errors, "
                    "then restart the service. Do not route traffic to this instance "
                    "until /ready returns 200."
                ),
                key="readiness",
            )

        alerts = alerts[: policy.max_alerts_per_run]
        for alert in alerts:
            METRICS.increment(
                "monitoring_alerts_total",
                labels={"type": alert.type.value, "severity": alert.severity.value},
                help_text="Alerts raised by the monitoring policy.",
            )
        return alerts

    # -- individual sources ------------------------------------------------
    def _fleet_alerts(self, snapshot: FleetSnapshot, add) -> None:  # type: ignore[no-untyped-def]
        quality = snapshot.data_quality
        if quality.status is MonitoringStatus.CRITICAL:
            add(
                AlertType.DATA_QUALITY_CRITICAL,
                AlertSeverity.CRITICAL,
                "Fleet input data quality is CRITICAL.",
                evidence=quality.warnings[:10],
                observed=quality.poor_or_worse_fraction,
                threshold=self.cfg.monitoring.data_quality.critical_poor_fraction,
                action=(
                    "Investigate telemetry collection before reading any prediction "
                    "from this batch. This is an input problem; retraining will not "
                    "fix a sensor that stopped reporting."
                ),
                key="quality",
            )
        elif quality.status is MonitoringStatus.WARNING:
            add(
                AlertType.DATA_QUALITY_WARNING,
                AlertSeverity.WARNING,
                "Fleet input data quality is degraded.",
                evidence=quality.warnings[:10],
                observed=quality.poor_or_worse_fraction,
                threshold=self.cfg.monitoring.data_quality.warning_poor_fraction,
                action="Review which cells are affected and why their telemetry is thin.",
                key="quality",
            )

        if snapshot.failed_count:
            add(
                AlertType.FLEET_PROCESSING_FAILURE,
                (
                    AlertSeverity.CRITICAL
                    if snapshot.failed_count > snapshot.battery_count / 2
                    else AlertSeverity.WARNING
                ),
                f"{snapshot.failed_count} of {snapshot.battery_count} batteries failed "
                "to process.",
                evidence=[
                    f"{record.battery_id}: {record.errors[0]}"
                    for record in snapshot.batteries
                    if record.errors
                ][:10],
                observed=float(snapshot.failed_count),
                action=(
                    "Read the per-battery errors in the snapshot. Failures are excluded "
                    "from every aggregate, so the fleet numbers describe only the cells "
                    "that succeeded."
                ),
                key="failures",
            )

        critical = snapshot.maintenance_summary.critical_count
        limit = self.cfg.fleet.high_critical_count_alert
        if critical >= limit > 0:
            add(
                AlertType.HIGH_CRITICAL_BATTERY_COUNT,
                AlertSeverity.WARNING,
                f"{critical} batteries are at a critical maintenance priority.",
                evidence=snapshot.maintenance_summary.high_risk_battery_ids[:20],
                observed=float(critical),
                threshold=float(limit),
                action=(
                    "Review the critical cells individually against their measured "
                    "capacity before scheduling any work."
                ),
                key="critical_count",
            )

    def _feature_drift_alerts(self, report: FeatureDriftReport, add) -> None:  # type: ignore[no-untyped-def]
        if report.status is MonitoringStatus.CRITICAL:
            add(
                AlertType.FEATURE_DRIFT_CRITICAL,
                AlertSeverity.CRITICAL,
                f"{report.n_features_drifted} of {report.n_features_tested} tested "
                "features have drifted from the training reference.",
                evidence=report.drifted_features[:20],
                observed=round(report.drifted_fraction, 4),
                threshold=self.cfg.monitoring.drift.fleet_critical_fraction,
                action=(
                    "Establish whether the inputs changed (pipeline, sensors, units) or "
                    "the population did (an older fleet). Feature drift alone is not "
                    "evidence the model is less accurate — check the performance report "
                    "before considering a retrain."
                ),
                key="feature_drift",
            )
        elif report.status is MonitoringStatus.WARNING:
            add(
                AlertType.FEATURE_DRIFT_WARNING,
                AlertSeverity.WARNING,
                f"{report.n_features_drifted} feature(s) have drifted from the "
                "training reference.",
                evidence=report.drifted_features[:20],
                observed=round(report.drifted_fraction, 4),
                threshold=self.cfg.monitoring.drift.fleet_warning_fraction,
                action="Review the drifted features against recent telemetry changes.",
                key="feature_drift",
            )

    def _prediction_drift_alerts(self, report: PredictionDriftReport, add) -> None:  # type: ignore[no-untyped-def]
        if report.status in (MonitoringStatus.WARNING, MonitoringStatus.CRITICAL):
            add(
                AlertType.PREDICTION_DRIFT_WARNING,
                (
                    AlertSeverity.WARNING
                    if report.status is MonitoringStatus.WARNING
                    else AlertSeverity.CRITICAL
                ),
                f"The model's output distribution has shifted on {report.n_drifted} "
                "monitored quantity/quantities.",
                evidence=[
                    f"{r.output_name} {r.metric}={r.drift_value} (threshold {r.threshold})"
                    for r in report.results
                    if r.drift_detected
                ][:20],
                observed=float(report.n_drifted),
                action=(
                    "Compare against the feature-drift report and the fleet's age "
                    "profile. A fleet that has aged should show prediction drift; that "
                    "is the model working, not failing."
                ),
                key="prediction_drift",
            )

    def _performance_alerts(self, report: PerformanceReport, add) -> None:  # type: ignore[no-untyped-def]
        if report.status is PerformanceStatus.DEGRADED:
            add(
                AlertType.MODEL_PERFORMANCE_DEGRADED,
                AlertSeverity.CRITICAL,
                "Production accuracy has crossed a degraded threshold on labelled outcomes.",
                evidence=report.breaches[:10],
                action=(
                    "Review the breached metrics against label coverage and life-stage "
                    "breakdown before acting. Do not retrain on one crossed threshold: "
                    "check whether the labelled cells are representative first."
                ),
                key="performance",
            )
        elif report.status is PerformanceStatus.WARNING:
            add(
                AlertType.MODEL_PERFORMANCE_WARNING,
                AlertSeverity.WARNING,
                "Production accuracy has crossed a warning threshold on labelled outcomes.",
                evidence=report.breaches[:10],
                action="Monitor the trend over the next few label batches.",
                key="performance",
            )


def evaluate_alerts(cfg: ExperimentConfig, **kwargs) -> list[Alert]:  # type: ignore[no-untyped-def]
    """Convenience wrapper around :class:`AlertPolicy`."""
    return AlertPolicy(cfg=cfg).build(**kwargs)
