"""Production monitoring: data quality, drift, performance, alerts.

Four questions, deliberately kept apart because conflating them is how a team
retrains a healthy model or ignores a broken sensor:

``data_quality``      is the *input* usable? (missing sensors, short histories,
                      duplicate cycles) — a data problem, not a model problem
``drift``             have the *inputs* moved away from what training saw?
``prediction_drift``  has the model's *output* distribution moved?
``performance``       given labels that arrived later, is the model still
                      accurate? — the only one of the four that can answer that

Feature drift with stable performance is a population change the model handles.
Stable inputs with degraded performance is a model or a world change. Only the
combination is diagnostic, so :class:`MonitoringSnapshot` reports all four side
by side and never collapses them into one score.
"""

from __future__ import annotations

from battery_rul.monitoring.alerts import AlertPolicy, evaluate_alerts
from battery_rul.monitoring.domain import (
    MONITORING_SNAPSHOT_SCHEMA_VERSION,
    Alert,
    AlertSeverity,
    AlertType,
    FeatureDriftReport,
    FeatureDriftResult,
    MonitoringSnapshot,
    PerformanceReport,
    PerformanceStatus,
    PredictionDriftReport,
)
from battery_rul.monitoring.drift import detect_feature_drift
from battery_rul.monitoring.performance import evaluate_delayed_labels
from battery_rul.monitoring.prediction_drift import detect_prediction_drift
from battery_rul.monitoring.reference import (
    ReferenceDistribution,
    build_reference_distribution,
    load_reference,
    save_reference,
)

__all__ = [
    "MONITORING_SNAPSHOT_SCHEMA_VERSION",
    "Alert",
    "AlertPolicy",
    "AlertSeverity",
    "AlertType",
    "FeatureDriftReport",
    "FeatureDriftResult",
    "MonitoringSnapshot",
    "PerformanceReport",
    "PerformanceStatus",
    "PredictionDriftReport",
    "ReferenceDistribution",
    "build_reference_distribution",
    "detect_feature_drift",
    "detect_prediction_drift",
    "evaluate_alerts",
    "evaluate_delayed_labels",
    "load_reference",
    "save_reference",
]
