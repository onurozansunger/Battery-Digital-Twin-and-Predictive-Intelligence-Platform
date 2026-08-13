"""Batch fleet inference over real (small) bundles trained on synthetic cells.

Fixture metrics are not model performance and nothing here asserts a quality
number. What is asserted is the architecture: the fleet path *reuses* the
battery-level service, loads its artifacts once, isolates per-cell failures,
keeps every denominator honest, and produces a snapshot that survives a JSON
round trip.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from battery_rul.digital_twin.service import BatteryDigitalTwinService
from battery_rul.fleet.aggregation import fleet_statistics, health_distribution
from battery_rul.fleet.domain import (
    FLEET_SNAPSHOT_SCHEMA_VERSION,
    FleetSnapshot,
    MaintenancePriority,
    ProcessingStatus,
)
from battery_rul.fleet.ingestion import BatteryHistoryInput, FleetIngestor


# ---------------------------------------------------------------------------
# The snapshot
# ---------------------------------------------------------------------------
def test_the_snapshot_accounts_for_every_submitted_cell(fleet_snapshot, fleet_histories):
    ingestion, histories = fleet_histories
    assert fleet_snapshot.battery_count == len(fleet_snapshot.batteries)
    assert fleet_snapshot.battery_count >= len(histories)
    assert (
        fleet_snapshot.successfully_processed_count
        + fleet_snapshot.failed_count
        + fleet_snapshot.insufficient_data_count
        == fleet_snapshot.battery_count
    )


def test_no_battery_appears_twice(fleet_snapshot):
    identifiers = [record.battery_id for record in fleet_snapshot.batteries]
    assert len(identifiers) == len(set(identifiers))


def test_the_snapshot_is_json_serialisable_and_reloadable(fleet_snapshot):
    payload = fleet_snapshot.to_json_dict()
    text = json.dumps(payload)
    restored = FleetSnapshot(**json.loads(text))

    assert restored.snapshot_id == fleet_snapshot.snapshot_id
    assert restored.schema_version == FLEET_SNAPSHOT_SCHEMA_VERSION
    assert restored.to_json_dict() == payload


def test_every_record_carries_its_model_version(fleet_snapshot):
    for record in fleet_snapshot.evaluated():
        assert record.model_version, "a decision must be attributable to a model version"
    assert fleet_snapshot.model_metadata.active_model_version


def test_the_snapshot_reports_its_processing_cost_and_fingerprint(fleet_snapshot):
    assert fleet_snapshot.processing_duration_ms is not None
    assert fleet_snapshot.processing_duration_ms >= 0
    assert fleet_snapshot.data_fingerprint


def test_the_disclaimer_travels_with_the_payload(fleet_snapshot):
    assert "decision support" in fleet_snapshot.disclaimer or "policy" in fleet_snapshot.disclaimer


# ---------------------------------------------------------------------------
# Per-battery invariants
# ---------------------------------------------------------------------------
def test_prediction_intervals_bracket_the_point_estimate(fleet_snapshot):
    for record in fleet_snapshot.evaluated():
        if record.rul_lower_bound is not None and record.rul_upper_bound is not None:
            assert record.rul_lower_bound <= record.predicted_rul <= record.rul_upper_bound


def test_risk_probabilities_stay_in_the_unit_interval(fleet_snapshot):
    for record in fleet_snapshot.batteries:
        if record.failure_risk is not None:
            assert 0.0 <= record.failure_risk <= 1.0


def test_priority_scores_stay_inside_the_configured_range(fleet_snapshot, m3_config):
    scale = m3_config.fleet.ranking.score_scale
    for record in fleet_snapshot.batteries:
        assert 0.0 <= record.priority_score <= scale


def test_every_evaluated_cell_has_a_priority_and_an_argument(fleet_snapshot):
    for record in fleet_snapshot.evaluated():
        assert record.priority is not MaintenancePriority.INSUFFICIENT_DATA
        assert record.priority_record is not None
        assert record.priority_record.triggered_rules
        assert record.priority_record.evidence
        assert record.replacement is not None


# ---------------------------------------------------------------------------
# Aggregation honesty
# ---------------------------------------------------------------------------
def test_aggregates_exclude_unevaluated_cells(m3_config, fleet_snapshot):
    statistics = fleet_statistics(fleet_snapshot.batteries, m3_config)
    evaluated_with_rul = [r for r in fleet_snapshot.evaluated() if r.predicted_rul is not None]
    assert statistics.rul_denominator == len(evaluated_with_rul)


def test_denominators_are_reported_beside_the_statistics(m3_config, fleet_snapshot):
    statistics = fleet_statistics(fleet_snapshot.batteries, m3_config)
    assert statistics.soh_denominator >= 0
    assert set(statistics.missingness) >= {"predicted_rul", "measured_soh"}


def test_a_failed_cell_never_enters_a_predicted_aggregate(m3_config, fleet_snapshot):
    from battery_rul.fleet.domain import FleetBatteryRecord

    poisoned = [
        *fleet_snapshot.batteries,
        FleetBatteryRecord(
            battery_id="FAILED-1",
            status=ProcessingStatus.FAILED,
            errors=["synthetic failure"],
        ),
    ]
    before = fleet_statistics(fleet_snapshot.batteries, m3_config)
    after = fleet_statistics(poisoned, m3_config)

    assert after.rul_denominator == before.rul_denominator
    assert after.rul_median == before.rul_median


def test_health_distribution_counts_only_cells_with_a_measured_soh(fleet_snapshot):
    distribution = health_distribution(fleet_snapshot.batteries)
    total = sum(distribution.counts.values())
    assert total == len(fleet_snapshot.evaluated())
    assert distribution.denominator + distribution.unknown_count == total


# ---------------------------------------------------------------------------
# Partial failure
# ---------------------------------------------------------------------------
def test_one_broken_cell_does_not_destroy_the_others(m3_config, fleet_service, fleet_histories):
    _, histories = fleet_histories
    broken = BatteryHistoryInput(
        battery_id="BROKEN",
        history=pd.DataFrame({"cycle_index": [1, 2, 3], "capacity_ah": [np.nan, np.nan, np.nan]}),
    )
    snapshot = fleet_service.create_fleet_snapshot(
        "PARTIAL", [*histories[:2], broken], batch_id="partial-1"
    )

    assert snapshot.battery_count == 3
    assert snapshot.successfully_processed_count >= 1, "the healthy cells still scored"
    broken_record = snapshot.battery("BROKEN")
    assert broken_record is not None
    assert broken_record.status is not ProcessingStatus.SUCCESS


def test_cells_rejected_at_ingestion_still_appear_in_the_snapshot(
    m3_config, fleet_service, fleet_histories
):
    _, histories = fleet_histories
    frame = histories[0].history.copy()
    frame["cycle_index"] = 1  # duplicated cycles: rejected at ingestion

    ingestion, accepted = FleetIngestor(cfg=m3_config).from_records(
        "PARTIAL",
        [("GOOD", histories[0].history), ("BAD", frame)],
        source="test",
    )
    snapshot = fleet_service.create_fleet_snapshot(
        "PARTIAL", accepted, ingestion=ingestion, batch_id="partial-2"
    )

    assert snapshot.battery_count == 2
    bad = snapshot.battery("BAD")
    assert bad is not None
    assert bad.status is ProcessingStatus.FAILED
    assert bad.errors
    assert any("failed" in warning for warning in snapshot.warnings)


def test_an_empty_fleet_is_reported_rather_than_crashing(fleet_service):
    snapshot = fleet_service.create_fleet_snapshot("EMPTY", [], batch_id="empty-1")
    assert snapshot.battery_count == 0
    assert snapshot.fleet_statistics.rul_denominator == 0
    assert any("No battery" in warning for warning in snapshot.warnings)


def test_a_fleet_larger_than_the_batch_limit_is_refused_explicitly(
    m3_config, fleet_service, fleet_histories
):
    _, histories = fleet_histories
    with pytest.raises(ValueError, match="limit"):
        fleet_service.create_fleet_snapshot("BIG", histories, max_batteries=1)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
def test_the_fleet_service_reuses_the_battery_level_service(m3_config):
    """One inference path, one set of loaded artifacts."""
    from battery_rul.fleet.inference import FleetInferenceService

    twin = BatteryDigitalTwinService.create(m3_config)
    service = FleetInferenceService.create(m3_config, twin=twin)
    assert service.twin is twin


def test_bundles_are_loaded_once_not_once_per_battery(
    m3_config, fleet_service, fleet_histories, monkeypatch
):
    """A fleet path that reloads a pickle per cell turns minutes into hours."""
    import battery_rul.models.bundle as bundle_module

    calls: list[str] = []
    original = bundle_module.load_bundle

    def _counting_load(path, cfg=None, **kwargs):
        calls.append(str(path))
        return original(path, cfg, **kwargs)

    monkeypatch.setattr(bundle_module, "load_bundle", _counting_load)
    _, histories = fleet_histories
    fleet_service.create_fleet_snapshot("REUSE", histories, batch_id="reuse-1")

    assert calls == [], "scoring a fleet must not load a single bundle"


def test_concurrency_does_not_change_the_result(m3_config, fleet_histories):
    from battery_rul.fleet.inference import FleetInferenceService

    _, histories = fleet_histories
    serial_cfg = m3_config.model_copy(deep=True)
    serial_cfg.fleet.max_concurrency = 1
    parallel_cfg = m3_config.model_copy(deep=True)
    parallel_cfg.fleet.max_concurrency = 4

    serial = FleetInferenceService.create(serial_cfg).create_fleet_snapshot(
        "C", histories, batch_id="c1"
    )
    parallel = FleetInferenceService.create(parallel_cfg).create_fleet_snapshot(
        "C", histories, batch_id="c1"
    )

    assert [r.battery_id for r in serial.batteries] == [r.battery_id for r in parallel.batteries]
    assert [r.predicted_rul for r in serial.batteries] == [
        r.predicted_rul for r in parallel.batteries
    ]


def test_running_the_same_fleet_twice_gives_the_same_numbers(fleet_service, fleet_histories):
    _, histories = fleet_histories
    first = fleet_service.create_fleet_snapshot("D", histories, batch_id="d1")
    second = fleet_service.create_fleet_snapshot("D", histories, batch_id="d1")

    assert [r.predicted_rul for r in first.batteries] == [r.predicted_rul for r in second.batteries]
    assert first.fleet_statistics.rul_median == second.fleet_statistics.rul_median


def test_a_demo_fleet_is_labelled_everywhere(m3_config):
    from battery_rul.fleet.demo import DemoFleetSpec, demo_fleet_identity, ingest_demo_fleet
    from battery_rul.fleet.inference import FleetInferenceService

    spec = DemoFleetSpec(fleet_id="DEMO-TEST", n_batteries=6)
    ingestion, histories = ingest_demo_fleet(m3_config, spec)
    snapshot = FleetInferenceService.create(m3_config).create_fleet_snapshot(
        "DEMO-TEST",
        histories,
        ingestion=ingestion,
        identity=demo_fleet_identity(spec),
        batch_id="demo-1",
    )

    assert snapshot.identity.is_demo_data is True
    assert snapshot.summary.is_demo_data is True
    assert any("DEMO DATA" in warning for warning in snapshot.warnings)
    assert all(record.battery_id.startswith("DEMO-") for record in snapshot.batteries)


def test_the_demo_fleet_is_deterministic(m3_config):
    from battery_rul.fleet.demo import DemoFleetSpec, generate_demo_fleet

    spec = DemoFleetSpec(fleet_id="DEMO-TEST", n_batteries=4)
    first = generate_demo_fleet(m3_config, spec)
    second = generate_demo_fleet(m3_config, spec)

    for a, b in zip(first, second, strict=True):
        assert a.battery_id == b.battery_id
        pd.testing.assert_frame_equal(a.history, b.history)
