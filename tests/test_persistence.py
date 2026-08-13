"""Persistence, observability and experiment tracking.

Everything runs against a temporary database or an in-memory one — never the
developer's real store, which is what anchoring the configured paths to
``paths.root`` buys.
"""

from __future__ import annotations

import json
import logging

import pytest

from battery_rul.config import ExperimentConfig, load_config
from battery_rul.monitoring.domain import (
    Alert,
    AlertSeverity,
    AlertType,
    MonitoringSnapshot,
)
from battery_rul.monitoring.performance import OutcomeLabel, PredictionRecord
from battery_rul.observability.logging import (
    StructuredFormatter,
    bind_context,
    current_context,
    log_event,
)
from battery_rul.observability.metrics import MetricsRegistry
from battery_rul.persistence import ReadOnlyStoreError, build_repository
from battery_rul.persistence.base import SCHEMA_VERSION, Repository
from battery_rul.persistence.sqlite import SQLiteRepository


@pytest.fixture
def store_cfg(tmp_path) -> ExperimentConfig:
    return load_config(overrides={"paths.root": str(tmp_path)})


@pytest.fixture
def repository(store_cfg) -> Repository:
    return build_repository(store_cfg)


def _alert(alert_id: str = "alert-1", fleet_id: str = "F1") -> Alert:
    return Alert(
        alert_id=alert_id,
        type=AlertType.DATA_QUALITY_WARNING,
        severity=AlertSeverity.WARNING,
        fleet_id=fleet_id,
        message="Input quality is degraded.",
        recommended_human_action="Check the telemetry pipeline.",
    )


# ---------------------------------------------------------------------------
# Contract and lifecycle
# ---------------------------------------------------------------------------
def test_the_sqlite_repository_satisfies_the_protocol(repository):
    assert isinstance(repository, Repository)
    assert repository.schema_version() == SCHEMA_VERSION


def test_the_in_memory_backend_is_the_same_implementation(store_cfg):
    store_cfg.persistence.backend = "memory"
    memory = build_repository(store_cfg)
    assert isinstance(memory, SQLiteRepository)
    assert memory.schema_version() == SCHEMA_VERSION


def test_initialising_twice_is_safe(store_cfg):
    build_repository(store_cfg)
    again = build_repository(store_cfg)
    assert again.list_fleet_snapshots() == []


# ---------------------------------------------------------------------------
# Fleet snapshots
# ---------------------------------------------------------------------------
def test_a_fleet_snapshot_round_trips(repository, fleet_snapshot):
    repository.save_fleet_snapshot(fleet_snapshot)
    loaded = repository.get_fleet_snapshot(fleet_snapshot.snapshot_id)

    assert loaded is not None
    assert loaded.snapshot_id == fleet_snapshot.snapshot_id
    assert loaded.battery_count == fleet_snapshot.battery_count
    assert loaded.to_json_dict() == fleet_snapshot.to_json_dict()


def test_the_latest_snapshot_is_the_newest(repository, fleet_snapshot):
    older = fleet_snapshot.model_copy(
        update={"snapshot_id": "old", "generated_at_utc": "2020-01-01T00:00:00+00:00"}
    )
    newer = fleet_snapshot.model_copy(
        update={"snapshot_id": "new", "generated_at_utc": "2030-01-01T00:00:00+00:00"}
    )
    repository.save_fleet_snapshot(older)
    repository.save_fleet_snapshot(newer)
    assert repository.latest_fleet_snapshot(fleet_snapshot.fleet_id).snapshot_id == "new"


def test_listing_returns_metadata_not_whole_snapshots(repository, fleet_snapshot):
    repository.save_fleet_snapshot(fleet_snapshot)
    rows = repository.list_fleet_snapshots(fleet_snapshot.fleet_id)
    assert len(rows) == 1
    assert "batteries" not in rows[0]
    assert rows[0]["battery_count"] == fleet_snapshot.battery_count


def test_an_unknown_snapshot_is_none_not_an_error(repository):
    assert repository.get_fleet_snapshot("nope") is None
    assert repository.latest_fleet_snapshot("no-such-fleet") is None


# ---------------------------------------------------------------------------
# Monitoring snapshots and alerts
# ---------------------------------------------------------------------------
def test_a_monitoring_snapshot_round_trips(repository):
    snapshot = MonitoringSnapshot(snapshot_id="mon-1", fleet_id="F1", input_count=5)
    repository.save_monitoring_snapshot(snapshot)
    loaded = repository.latest_monitoring_snapshot("F1")
    assert loaded is not None
    assert loaded.snapshot_id == "mon-1"


def test_alerts_round_trip_and_filter(repository):
    repository.save_alerts([_alert("a1"), _alert("a2")])
    assert len(repository.list_alerts("F1")) == 2
    assert repository.list_alerts("other-fleet") == []


def test_acknowledging_an_alert_survives_a_repeated_finding(repository):
    """Alert ids are deterministic, so re-raising must not wipe the ack."""
    repository.save_alerts([_alert("a1")])
    assert repository.acknowledge_alert("a1", by="alice") is True

    repository.save_alerts([_alert("a1")])
    acknowledged = repository.list_alerts("F1", acknowledged=True)
    assert len(acknowledged) == 1
    assert acknowledged[0].acknowledged_by == "alice"


def test_acknowledging_an_unknown_alert_returns_false(repository):
    assert repository.acknowledge_alert("missing", by="alice") is False


# ---------------------------------------------------------------------------
# Predictions and labels
# ---------------------------------------------------------------------------
def test_prediction_records_and_labels_round_trip(repository):
    records = [
        PredictionRecord(
            prediction_id=f"p{i}",
            battery_id=f"B{i}",
            cycle_index=100 + i,
            model_version="1.0.0",
            predicted_rul=30.0,
        )
        for i in range(3)
    ]
    repository.save_prediction_records(records)
    repository.save_outcome_labels(
        [OutcomeLabel(battery_id="B0", cycle_index=100, observed_rul=28.0)]
    )

    assert len(repository.list_prediction_records()) == 3
    assert len(repository.list_prediction_records(model_version="2.0.0")) == 0
    assert len(repository.list_outcome_labels()) == 1


def test_a_label_for_the_same_cycle_replaces_rather_than_duplicates(repository):
    first = OutcomeLabel(battery_id="B0", cycle_index=100, observed_rul=28.0)
    second = OutcomeLabel(battery_id="B0", cycle_index=100, observed_rul=31.0)
    repository.save_outcome_labels([first])
    repository.save_outcome_labels([second])

    labels = repository.list_outcome_labels()
    assert len(labels) == 1
    assert labels[0].observed_rul == 31.0


# ---------------------------------------------------------------------------
# Read-only mode
# ---------------------------------------------------------------------------
def test_a_read_only_deployment_refuses_writes_explicitly(store_cfg, fleet_snapshot):
    repository = build_repository(store_cfg)
    store_cfg.deployment.read_only = True

    with pytest.raises(ReadOnlyStoreError, match="read_only"):
        repository.save_fleet_snapshot(fleet_snapshot)
    with pytest.raises(ReadOnlyStoreError):
        repository.save_alerts([_alert()])


def test_reads_still_work_in_read_only_mode(store_cfg, fleet_snapshot):
    repository = build_repository(store_cfg)
    repository.save_fleet_snapshot(fleet_snapshot)
    store_cfg.deployment.read_only = True
    assert repository.latest_fleet_snapshot(fleet_snapshot.fleet_id) is not None


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
def test_metrics_record_counters_gauges_and_histograms():
    registry = MetricsRegistry()
    registry.increment("things_total", labels={"kind": "a"})
    registry.increment("things_total", 2, labels={"kind": "a"})
    registry.set_gauge("temperature", 21.5)
    registry.observe("duration_seconds", 0.05)

    assert registry.counter_value("things_total", {"kind": "a"}) == 3.0
    assert registry.gauge_value("temperature") == 21.5
    assert registry.histogram_stats("duration_seconds")["count"] == 1.0


def test_a_counter_cannot_decrease():
    registry = MetricsRegistry()
    with pytest.raises(ValueError, match="cannot decrease"):
        registry.increment("things_total", -1)


def test_prometheus_rendering_is_well_formed():
    registry = MetricsRegistry()
    registry.increment("requests_total", labels={"path": "/v1/fleet/snapshot"}, help_text="reqs")
    registry.observe("latency_seconds", 0.2)
    text = registry.render_prometheus()

    assert "# TYPE requests_total counter" in text
    assert 'requests_total{path="/v1/fleet/snapshot"} 1.0' in text
    assert "latency_seconds_bucket" in text
    assert 'le="+Inf"' in text


def test_metric_labels_are_escaped():
    registry = MetricsRegistry()
    registry.increment("weird_total", labels={"name": 'a"b'})
    assert '\\"' in registry.render_prometheus()


def test_a_disabled_registry_records_nothing():
    registry = MetricsRegistry()
    registry.enabled = False
    registry.increment("things_total")
    assert registry.counter_value("things_total") == 0.0


def test_log_context_is_inherited_and_restored():
    with bind_context(fleet_id="F1", batch_id="b1"):
        assert current_context().fleet_id == "F1"
        with bind_context(battery_id="B7"):
            inner = current_context()
            assert inner.battery_id == "B7"
            assert inner.fleet_id == "F1", "the outer context is inherited"
        assert current_context().battery_id is None
    assert current_context().fleet_id is None


def test_an_unknown_context_field_is_refused():
    with pytest.raises(ValueError, match="Unknown log-context field"), bind_context(nonsense="x"):
        pass


def test_structured_logs_are_json_and_carry_the_context():
    formatter = StructuredFormatter()
    record = logging.LogRecord(
        "test", logging.INFO, __file__, 1, "fleet_batch_completed", None, None
    )
    record.event = "fleet_batch_completed"
    record.duration_ms = 12.5
    with bind_context(fleet_id="F1", model_version="1.0.0"):
        payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["fleet_id"] == "F1"
    assert payload["model_version"] == "1.0.0"
    assert payload["duration_ms"] == 12.5


def test_raw_measurement_data_is_never_logged():
    """An accidental `extra={"history": frame}` must not leak a cell's telemetry."""
    formatter = StructuredFormatter()
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "x", None, None)
    record.history = [{"capacity_ah": 1.9}] * 500
    payload = json.loads(formatter.format(record))
    assert payload["history"] == "<omitted: raw measurement data is never logged>"


def test_log_event_emits_a_stable_event_name(caplog):
    logger = logging.getLogger("test.events")
    with caplog.at_level(logging.INFO, logger="test.events"):
        log_event(logger, "fleet_batch_started", battery_count=5)
    assert caplog.records[0].event == "fleet_batch_started"
    assert caplog.records[0].battery_count == 5


# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------
def test_a_file_tracked_run_records_provenance(store_cfg):
    from battery_rul.tracking import FileTracker, build_tracker

    tracker = build_tracker(store_cfg)
    assert isinstance(tracker, FileTracker)

    tracker.start_run("unit-test", stage="test")
    tracker.log_params({"model": "ridge", "alpha": 0.5})
    tracker.log_metrics({"mae": 12.3})
    run = tracker.end_run()

    assert run is not None
    assert run.status == "FINISHED"
    assert run.data_fingerprint
    assert run.params["model"] == "ridge"
    assert run.metrics["mae"] == 12.3
    assert (tracker.root / f"{run.run_id}.json").is_file()


def test_raw_data_is_never_tracked(store_cfg):
    from battery_rul.tracking import FileTracker

    tracker = FileTracker(cfg=store_cfg)
    tracker.start_run("unit-test")
    tracker.log_params({"history": [{"capacity_ah": 1.9}]})
    run = tracker.end_run()
    assert "omitted" in run.params["history"]


def test_artifact_paths_are_recorded_relative_to_the_root(store_cfg, tmp_path):
    from battery_rul.tracking import FileTracker

    artifact = tmp_path / "reports" / "metrics.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}")

    tracker = FileTracker(cfg=store_cfg)
    tracker.start_run("unit-test")
    tracker.log_artifact("metrics", artifact)
    run = tracker.end_run()

    assert run.artifacts["metrics"] == "reports/metrics.json"


def test_logging_without_an_active_run_is_an_error(store_cfg):
    from battery_rul.tracking import FileTracker

    with pytest.raises(RuntimeError, match="No tracking run is active"):
        FileTracker(cfg=store_cfg).log_metrics({"mae": 1.0})


def test_runs_can_be_compared(store_cfg):
    from battery_rul.tracking import FileTracker, compare_runs

    for index in range(2):
        tracker = FileTracker(cfg=store_cfg)
        tracker.start_run(f"run-{index}")
        tracker.log_metrics({"mae": 10.0 + index})
        tracker.end_run()

    rows = compare_runs(store_cfg)
    assert len(rows) == 2
    assert {row["metric.mae"] for row in rows} == {10.0, 11.0}
