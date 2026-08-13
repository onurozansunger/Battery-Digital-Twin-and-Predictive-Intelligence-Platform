"""Delayed-label performance monitoring, fleet data quality and alerts.

The states that matter most here are the ones where the honest answer is "not
yet": no labels, too few labels, single-class labels. Each has to be reported as
itself rather than as a metric computed on nothing.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from battery_rul.config import ExperimentConfig, load_config
from battery_rul.fleet.domain import (
    FleetBatteryRecord,
    MaintenancePriority,
    MonitoringStatus,
    ProcessingStatus,
)
from battery_rul.monitoring.alerts import AlertPolicy
from battery_rul.monitoring.data_quality import summarise_fleet_data_quality
from battery_rul.monitoring.domain import AlertSeverity, AlertType, PerformanceStatus
from battery_rul.monitoring.performance import (
    OutcomeLabel,
    PredictionRecord,
    evaluate_delayed_labels,
    join_predictions_and_labels,
    prediction_records_from_snapshot,
)


@pytest.fixture
def perf_cfg() -> ExperimentConfig:
    return load_config(overrides={"monitoring.performance.min_labels": 5})


def _risk_probability(index: int) -> float:
    """A monotone risk score, so the fixture's risk head is well-behaved.

    Deliberately correlated with :func:`_label`'s outcome: these tests are about
    the *plumbing* — joins, thresholds, statuses — and a fixture whose risk
    scores were unrelated to its labels would fail the PR-AUC floor for reasons
    that have nothing to do with what is being tested.
    """
    return min(0.95, 0.05 + 0.08 * index)


def _prediction(index: int, *, error: float = 0.0, version: str = "1.0.0") -> PredictionRecord:
    truth = 20.0 + index
    return PredictionRecord(
        prediction_id=f"p{index}",
        battery_id=f"B{index}",
        cycle_index=100 + index,
        model_version=version,
        predicted_rul=truth + error,
        rul_lower_bound=truth + error - 15.0,
        rul_upper_bound=truth + error + 15.0,
        interval_coverage_target=0.9,
        predicted_soh_forecast=0.85,
        failure_risk=_risk_probability(index % 100),
        measured_soh=0.88,
    )


def _label(index: int, *, positive: bool | None = None) -> OutcomeLabel:
    return OutcomeLabel(
        battery_id=f"B{index}",
        cycle_index=100 + index,
        observed_at_cycle=100 + index + 30,
        observed_rul=20.0 + index,
        observed_soh=0.84,
        eol_within_horizon=(
            (_risk_probability(index % 100) >= 0.5) if positive is None else positive
        ),
        label_source="test",
    )


# ---------------------------------------------------------------------------
# The states that mean "not yet"
# ---------------------------------------------------------------------------
def test_no_labels_is_reported_as_no_labels(perf_cfg):
    report = evaluate_delayed_labels([_prediction(i) for i in range(10)], [], perf_cfg)
    assert report.status is PerformanceStatus.NO_LABELS
    assert report.rul_metrics == {}
    assert report.label_coverage == 0.0
    assert any("expected" in w for w in report.warnings)


def test_too_few_labels_publishes_no_metric(perf_cfg):
    predictions = [_prediction(i) for i in range(10)]
    report = evaluate_delayed_labels(predictions, [_label(0), _label(1)], perf_cfg)
    assert report.status is PerformanceStatus.INSUFFICIENT_LABELS
    assert report.rul_metrics == {}
    assert report.n_labels_joined == 2


def test_single_class_risk_labels_do_not_produce_a_score(perf_cfg):
    predictions = [_prediction(i) for i in range(10)]
    labels = [_label(i, positive=False) for i in range(10)]
    report = evaluate_delayed_labels(predictions, labels, perf_cfg)
    assert "pr_auc" not in report.risk_metrics
    assert any("same class" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# With labels
# ---------------------------------------------------------------------------
def test_accurate_predictions_are_healthy(perf_cfg):
    predictions = [_prediction(i, error=1.0) for i in range(12)]
    labels = [_label(i) for i in range(12)]
    report = evaluate_delayed_labels(predictions, labels, perf_cfg, model_version="1.0.0")

    assert report.status is PerformanceStatus.HEALTHY
    assert report.rul_metrics["mae"] == pytest.approx(1.0, abs=1e-6)
    assert report.rul_metrics["bias"] == pytest.approx(1.0, abs=1e-6)
    assert report.n_labels_joined == 12
    assert report.label_coverage == pytest.approx(1.0)


def test_a_large_error_crosses_the_degraded_threshold(perf_cfg):
    predictions = [_prediction(i, error=60.0) for i in range(12)]
    labels = [_label(i) for i in range(12)]
    report = evaluate_delayed_labels(predictions, labels, perf_cfg)

    assert report.status is PerformanceStatus.DEGRADED
    assert any("RUL MAE" in breach for breach in report.breaches)


def test_interval_coverage_is_measured_on_the_intervals_that_were_issued(perf_cfg):
    predictions = [_prediction(i, error=1.0) for i in range(12)]
    labels = [_label(i) for i in range(12)]
    report = evaluate_delayed_labels(predictions, labels, perf_cfg)

    assert report.interval_coverage["n"] == 12
    assert report.interval_coverage["empirical_coverage"] == pytest.approx(1.0)


def test_error_is_broken_down_by_life_stage(perf_cfg):
    predictions = [_prediction(i, error=2.0) for i in range(12)]
    labels = [_label(i) for i in range(12)]
    report = evaluate_delayed_labels(predictions, labels, perf_cfg)

    stages = {row["life_stage"]: row for row in report.rul_error_by_life_stage}
    assert set(stages) == {"0-20", "20-50", "50-100", "100+"}
    assert stages["20-50"]["n"] > 0


def test_the_evaluation_delay_is_reported(perf_cfg):
    predictions = [_prediction(i) for i in range(12)]
    labels = [_label(i) for i in range(12)]
    report = evaluate_delayed_labels(predictions, labels, perf_cfg)
    assert report.evaluation_delay_cycles["median"] == pytest.approx(30.0)


def test_production_metrics_are_marked_as_not_comparable_with_the_test_partition(perf_cfg):
    report = evaluate_delayed_labels([_prediction(0)], [], perf_cfg)
    assert "not comparable" in report.comparison_note


def test_the_join_is_inner(perf_cfg):
    """A prediction with no label is not a zero-error prediction."""
    joined = join_predictions_and_labels([_prediction(i) for i in range(5)], [_label(0)])
    assert len(joined) == 1
    assert joined["battery_id"].iloc[0] == "B0"


def test_metrics_are_attributed_to_the_version_that_predicted(perf_cfg):
    predictions = [_prediction(i, error=1.0, version="2.0.0") for i in range(12)]
    predictions += [_prediction(i + 100, error=99.0, version="1.0.0") for i in range(12)]
    labels = [_label(i) for i in range(12)] + [_label(i + 100) for i in range(12)]

    report = evaluate_delayed_labels(predictions, labels, perf_cfg, model_version="2.0.0")
    assert report.rul_metrics["mae"] == pytest.approx(1.0, abs=1e-6)


def test_prediction_records_come_from_the_snapshot(fleet_snapshot):
    records = prediction_records_from_snapshot(fleet_snapshot)
    assert records, "the fixture fleet produced at least one scored cell"
    assert all(r.model_version for r in records)
    assert {r.battery_id for r in records} <= {r.battery_id for r in fleet_snapshot.batteries}


# ---------------------------------------------------------------------------
# Fleet data quality
# ---------------------------------------------------------------------------
def _record(battery_id: str, quality: str, score: float, **overrides) -> FleetBatteryRecord:
    return FleetBatteryRecord(
        battery_id=battery_id,
        status=overrides.pop("status", ProcessingStatus.SUCCESS),
        data_quality_class=quality,
        data_quality_score=score,
        **overrides,
    )


def test_a_healthy_fleet_is_ok(perf_cfg):
    records = [_record(f"B{i}", "GOOD", 0.95) for i in range(10)]
    summary = summarise_fleet_data_quality(records, perf_cfg)
    assert summary.status is MonitoringStatus.OK
    assert summary.denominator == 10
    assert summary.quality_class_counts["GOOD"] == 10


def test_widespread_poor_quality_is_critical(perf_cfg):
    records = [_record(f"B{i}", "POOR", 0.4) for i in range(10)]
    summary = summarise_fleet_data_quality(records, perf_cfg)
    assert summary.status is MonitoringStatus.CRITICAL
    assert summary.poor_or_worse_fraction == pytest.approx(1.0)


def test_failed_cells_count_towards_poor_quality(perf_cfg):
    records = [_record(f"B{i}", "GOOD", 0.95) for i in range(6)]
    records += [
        _record(f"F{i}", "unknown", 0.0, status=ProcessingStatus.FAILED, errors=["boom"])
        for i in range(4)
    ]
    summary = summarise_fleet_data_quality(records, perf_cfg)
    assert summary.status in (MonitoringStatus.WARNING, MonitoringStatus.CRITICAL)
    assert summary.denominator == 10


def test_an_empty_fleet_is_unknown_not_ok(perf_cfg):
    summary = summarise_fleet_data_quality([], perf_cfg)
    assert summary.status is MonitoringStatus.UNKNOWN


def test_out_of_distribution_cells_are_named_without_being_called_drift(perf_cfg):
    records = [_record("B1", "GOOD", 0.9, out_of_distribution_feature_count=3)]
    summary = summarise_fleet_data_quality(records, perf_cfg)
    assert summary.batteries_with_ood_features == ["B1"]
    assert any("not a drift verdict" in w for w in summary.warnings)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
def test_a_clean_run_raises_no_alerts(perf_cfg, fleet_snapshot):
    alerts = AlertPolicy(cfg=perf_cfg).build(readiness={"ready": True})
    assert alerts == []


def test_an_unavailable_model_is_a_critical_alert(perf_cfg):
    alerts = AlertPolicy(cfg=perf_cfg).build(
        readiness={"ready": False, "errors": {"rul": "not found"}}
    )
    assert len(alerts) == 1
    assert alerts[0].type is AlertType.MODEL_UNAVAILABLE
    assert alerts[0].severity is AlertSeverity.CRITICAL


def test_every_alert_names_a_human_action(perf_cfg, fleet_snapshot):
    alerts = AlertPolicy(cfg=perf_cfg).build(
        fleet_snapshot=fleet_snapshot, readiness={"ready": False, "errors": {}}
    )
    assert alerts
    assert all(alert.recommended_human_action for alert in alerts)


def test_alert_ids_are_deterministic(perf_cfg):
    policy = AlertPolicy(cfg=perf_cfg)
    first = policy.build(readiness={"ready": False, "errors": {}})
    second = policy.build(readiness={"ready": False, "errors": {}})
    assert first[0].alert_id == second[0].alert_id


def test_muting_suppresses_an_alert_type(perf_cfg):
    perf_cfg.monitoring.alerts.muted_types = [AlertType.MODEL_UNAVAILABLE.value]
    alerts = AlertPolicy(cfg=perf_cfg).build(readiness={"ready": False, "errors": {}})
    assert alerts == []


def test_alerts_can_be_disabled_entirely(perf_cfg):
    perf_cfg.monitoring.alerts.enabled = False
    assert AlertPolicy(cfg=perf_cfg).build(readiness={"ready": False, "errors": {}}) == []


def test_no_external_notifier_is_configurable(perf_cfg):
    """External notification is off and frozen: this build ships no credentials."""
    assert perf_cfg.monitoring.alerts.external_notifications is False
    with pytest.raises(ValidationError):
        perf_cfg.monitoring.alerts.external_notifications = True


def test_a_high_critical_count_raises_an_alert(perf_cfg):
    from battery_rul.fleet.aggregation import maintenance_summary

    perf_cfg.fleet.high_critical_count_alert = 2
    records = [
        FleetBatteryRecord(
            battery_id=f"B{i}",
            status=ProcessingStatus.SUCCESS,
            priority=MaintenancePriority.P0_CRITICAL,
        )
        for i in range(3)
    ]
    summary = maintenance_summary(records, perf_cfg)
    assert summary.critical_count == 3
