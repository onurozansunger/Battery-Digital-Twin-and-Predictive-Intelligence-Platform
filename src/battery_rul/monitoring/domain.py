"""Monitoring domain objects.

Every report here is serialisable, versioned and persisted. That is not
bookkeeping: a drift alert is only actionable next to the run that produced it,
the reference it compared against, and the model version that was live at the
time. A monitoring system that reports "PSI 0.31" without those three facts has
told nobody anything.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from battery_rul.fleet.domain import MonitoringStatus

__all__ = [
    "MONITORING_SNAPSHOT_SCHEMA_VERSION",
    "Alert",
    "AlertSeverity",
    "AlertType",
    "FeatureDriftReport",
    "FeatureDriftResult",
    "FleetQualityReport",
    "MonitoringSnapshot",
    "PerformanceReport",
    "PerformanceStatus",
    "PredictionDriftReport",
    "PredictionDriftResult",
]

MONITORING_SNAPSHOT_SCHEMA_VERSION = "3.0"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------
class FeatureDriftResult(_Model):
    """One feature's drift verdict under one metric."""

    feature_name: str
    feature_type: Literal["numerical", "categorical"] = "numerical"
    drift_metric: str
    drift_value: float | None = None
    p_value: float | None = None
    adjusted_p_value: float | None = Field(
        default=None,
        description="After the configured multiple-comparison correction. Testing "
        "200 features at alpha=0.05 yields ~10 false positives by construction; the "
        "raw p-value alone would make that look like drift.",
    )
    threshold: float | None = None
    drift_detected: bool = False
    severity: MonitoringStatus = MonitoringStatus.OK
    reference_sample_size: int = Field(default=0, ge=0)
    sample_size: int = Field(default=0, ge=0)
    reference_summary: dict[str, float] = Field(default_factory=dict)
    current_summary: dict[str, float] = Field(default_factory=dict)
    reliable: bool = Field(
        default=True,
        description="False when the sample is too small, the feature is constant, or "
        "it is absent from one side. An unreliable result is reported, never scored.",
    )
    warnings: list[str] = Field(default_factory=list)


class FeatureDriftReport(_Model):
    """Fleet-level feature drift against a versioned reference."""

    generated_at_utc: str = Field(default_factory=_utc_now)
    reference_id: str
    reference_fingerprint: str | None = None
    reference_partition: str | None = None
    reference_window: str = ""
    current_window: str = ""
    status: MonitoringStatus = MonitoringStatus.UNKNOWN
    results: list[FeatureDriftResult] = Field(default_factory=list)
    n_features_tested: int = Field(default=0, ge=0)
    n_features_drifted: int = Field(default=0, ge=0)
    n_features_skipped: int = Field(default=0, ge=0)
    drifted_features: list[str] = Field(default_factory=list)
    multiple_comparison: str = "none"
    method_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def drifted_fraction(self) -> float:
        return self.n_features_drifted / self.n_features_tested if self.n_features_tested else 0.0


class PredictionDriftResult(_Model):
    """One output quantity's distribution shift."""

    output_name: str
    metric: str
    reference_value: float | None = None
    current_value: float | None = None
    drift_value: float | None = None
    threshold: float | None = None
    drift_detected: bool = False
    severity: MonitoringStatus = MonitoringStatus.OK
    reference_sample_size: int = Field(default=0, ge=0)
    sample_size: int = Field(default=0, ge=0)
    detail: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PredictionDriftReport(_Model):
    """Changes in what the model is saying, not in whether it is right."""

    generated_at_utc: str = Field(default_factory=_utc_now)
    reference_id: str | None = None
    model_version: str | None = None
    status: MonitoringStatus = MonitoringStatus.UNKNOWN
    results: list[PredictionDriftResult] = Field(default_factory=list)
    n_drifted: int = Field(default=0, ge=0)
    sample_size: int = Field(default=0, ge=0)
    interpretation: str = (
        "Prediction drift does not prove model degradation. It shows that the "
        "model's output distribution or the population it is scoring has changed. "
        "Only labelled outcomes can establish that accuracy has fallen."
    )
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------
class FleetQualityReport(_Model):
    """Per-battery quality detail behind the fleet summary."""

    generated_at_utc: str = Field(default_factory=_utc_now)
    fleet_id: str
    status: MonitoringStatus = MonitoringStatus.UNKNOWN
    per_battery: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Performance with delayed labels
# ---------------------------------------------------------------------------
class PerformanceStatus(StrEnum):
    """Explicit states, including the two that mean "we cannot say"."""

    NO_LABELS = "NO_LABELS"
    INSUFFICIENT_LABELS = "INSUFFICIENT_LABELS"
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    DEGRADED = "DEGRADED"


class PerformanceReport(_Model):
    """Accuracy against outcomes that became observable after the prediction."""

    generated_at_utc: str = Field(default_factory=_utc_now)
    model_version: str | None = None
    status: PerformanceStatus = PerformanceStatus.NO_LABELS
    n_predictions: int = Field(default=0, ge=0)
    n_labels_joined: int = Field(default=0, ge=0)
    label_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    evaluation_delay_cycles: dict[str, float] = Field(
        default_factory=dict,
        description="How long the labels took to arrive, in cycles. A metric "
        "computed on labels that arrived instantly is measuring a different "
        "problem from one whose labels took 40 cycles.",
    )
    rul_metrics: dict[str, Any] = Field(default_factory=dict)
    rul_error_by_life_stage: list[dict[str, Any]] = Field(default_factory=list)
    interval_coverage: dict[str, Any] = Field(default_factory=dict)
    soh_metrics: dict[str, Any] = Field(default_factory=dict)
    risk_metrics: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    breaches: list[str] = Field(default_factory=list)
    comparison_note: str = ""
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
class AlertType(StrEnum):
    DATA_QUALITY_WARNING = "DATA_QUALITY_WARNING"
    DATA_QUALITY_CRITICAL = "DATA_QUALITY_CRITICAL"
    FEATURE_DRIFT_WARNING = "FEATURE_DRIFT_WARNING"
    FEATURE_DRIFT_CRITICAL = "FEATURE_DRIFT_CRITICAL"
    PREDICTION_DRIFT_WARNING = "PREDICTION_DRIFT_WARNING"
    MODEL_PERFORMANCE_WARNING = "MODEL_PERFORMANCE_WARNING"
    MODEL_PERFORMANCE_DEGRADED = "MODEL_PERFORMANCE_DEGRADED"
    ARTIFACT_MISMATCH = "ARTIFACT_MISMATCH"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    FLEET_PROCESSING_FAILURE = "FLEET_PROCESSING_FAILURE"
    HIGH_CRITICAL_BATTERY_COUNT = "HIGH_CRITICAL_BATTERY_COUNT"


class AlertSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Alert(_Model):
    """One finding that a human should look at.

    ``recommended_human_action`` is required, and it is always a human action.
    Nothing in this platform retrains, promotes or takes a cell out of service on
    an alert: a threshold crossing is evidence for a decision, not the decision.
    """

    alert_id: str
    type: AlertType
    severity: AlertSeverity
    generated_at_utc: str = Field(default_factory=_utc_now)
    fleet_id: str | None = None
    model_version: str | None = None
    batch_id: str | None = None
    message: str
    evidence: list[str] = Field(default_factory=list)
    threshold: float | None = None
    observed_value: float | None = None
    recommended_human_action: str
    acknowledged: bool = False
    acknowledged_at_utc: str | None = None
    acknowledged_by: str | None = None


# ---------------------------------------------------------------------------
# The snapshot
# ---------------------------------------------------------------------------
class MonitoringSnapshot(_Model):
    """One monitoring run, complete and self-describing."""

    snapshot_id: str
    generated_at_utc: str = Field(default_factory=_utc_now)
    schema_version: str = MONITORING_SNAPSHOT_SCHEMA_VERSION
    fleet_id: str
    model_version: str | None = None
    data_version: str | None = None
    batch_id: str | None = None
    input_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    data_quality_summary: dict[str, Any] = Field(default_factory=dict)
    feature_drift_summary: dict[str, Any] = Field(default_factory=dict)
    prediction_drift_summary: dict[str, Any] = Field(default_factory=dict)
    performance_summary: dict[str, Any] = Field(default_factory=dict)
    alert_summary: dict[str, Any] = Field(default_factory=dict)
    alerts: list[Alert] = Field(default_factory=list)
    overall_status: MonitoringStatus = MonitoringStatus.UNKNOWN
    warnings: list[str] = Field(default_factory=list)
    report_paths: dict[str, str] = Field(
        default_factory=dict,
        description="Relative artifact paths. Relative on purpose: an absolute "
        "path in a persisted document leaks a filesystem layout into every "
        "consumer, including API responses.",
    )

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
