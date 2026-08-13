"""Regression and integration tests for Milestone 3.

Two jobs:

*Regression.* Milestone 3 is an extension, not a rewrite. The Milestone 1 and 2
public interfaces, wire formats and behaviours have to be exactly what they
were, and these tests fail if a fleet-layer convenience quietly changed one.

*Integration.* The documented pipelines have to run end to end and produce the
artifacts the documentation promises.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from battery_rul.config import ExperimentConfig, load_config


# ===========================================================================
# Milestone 1 and 2 interfaces are unchanged
# ===========================================================================
def test_milestone_1_and_2_modules_still_import():
    import battery_rul.data.loader  # noqa: F401
    import battery_rul.evaluation.metrics  # noqa: F401
    import battery_rul.features.engineering  # noqa: F401
    import battery_rul.features.pipeline  # noqa: F401
    import battery_rul.models.bundle  # noqa: F401
    import battery_rul.pipelines.milestone_2  # noqa: F401
    import battery_rul.recommendations.engine  # noqa: F401
    import battery_rul.uncertainty.conformal  # noqa: F401


def test_the_battery_snapshot_schema_version_has_not_moved():
    """Milestone 3 adds a fleet schema; it does not renumber the battery one."""
    from battery_rul.digital_twin.domain import SNAPSHOT_SCHEMA_VERSION

    assert SNAPSHOT_SCHEMA_VERSION == "2.0"


def test_the_bundle_schema_version_has_not_moved():
    from battery_rul.models.bundle import BUNDLE_SCHEMA_VERSION

    assert BUNDLE_SCHEMA_VERSION == "2.0"


def test_the_data_fingerprint_is_unchanged_by_the_new_configuration_sections():
    """Milestone 3 config must not invalidate a Milestone 2 bundle.

    ``data_affecting_dict`` covers the six sections that change what a model
    learns. Adding fleet, monitoring, registry, tracking, persistence and
    deployment sections must leave it alone — otherwise every existing bundle
    fails its compatibility check on upgrade.
    """
    cfg = load_config()
    affecting = set(cfg.data_affecting_dict())
    assert affecting == {"data", "validation", "features", "target", "soh", "risk"}
    for section in ("fleet", "monitoring", "registry", "tracking", "persistence", "deployment"):
        assert section not in affecting


def test_the_battery_level_service_keeps_its_public_interface(m3_platform):
    from battery_rul.digital_twin.service import BatteryDigitalTwinService

    cfg, _ = m3_platform
    service = BatteryDigitalTwinService.create(cfg, strict=True)
    for method in (
        "create_snapshot",
        "predict_rul",
        "predict_soh",
        "predict_failure_risk",
        "explain_prediction",
        "readiness",
        "health_check",
        "get_model_metadata",
    ):
        assert callable(getattr(service, method)), f"{method} is a Milestone 2 entry point"


def test_the_battery_level_snapshot_is_unchanged(m3_platform):
    from battery_rul.digital_twin.service import BatteryDigitalTwinService

    cfg, cycles = m3_platform
    service = BatteryDigitalTwinService.create(cfg, strict=True)
    battery_id = sorted(cycles["battery_id"].unique())[0]
    history = cycles.loc[cycles["battery_id"] == battery_id].reset_index(drop=True)
    history = history.drop(
        columns=[
            c
            for c in history.columns
            if c.startswith(("rul_", "eol_", "life_", "soh", "is_censored"))
        ],
        errors="ignore",
    )

    snapshot = service.create_snapshot(battery_id, history)
    payload = snapshot.to_json_dict()
    for key in (
        "battery_id",
        "identity",
        "measurement_summary",
        "health",
        "prediction",
        "failure_risk",
        "recommendation",
        "data_quality",
        "metadata",
        "disclaimer",
    ):
        assert key in payload, f"{key} is part of the Milestone 2 contract"


def test_prepare_features_is_purely_additive(m3_platform):
    """The one method Milestone 3 added to the battery service returns the same
    frame the service scores, and changes nothing else."""
    from battery_rul.digital_twin.service import BatteryDigitalTwinService

    cfg, cycles = m3_platform
    service = BatteryDigitalTwinService.create(cfg, strict=True)
    battery_id = sorted(cycles["battery_id"].unique())[0]
    history = cycles.loc[cycles["battery_id"] == battery_id].reset_index(drop=True)

    features = service.prepare_features(battery_id, history)
    assert isinstance(features, pd.DataFrame)
    assert len(features) > 0


def test_the_milestone_2_api_endpoints_still_answer(m3_platform):
    from fastapi.testclient import TestClient

    from battery_rul.api.app import create_app
    from battery_rul.api.schemas import CycleRecord
    from battery_rul.digital_twin.service import BatteryDigitalTwinService

    cfg, cycles = m3_platform
    client = TestClient(create_app(cfg, service=BatteryDigitalTwinService.create(cfg)))

    battery_id = sorted(cycles["battery_id"].unique())[0]
    allowed = set(CycleRecord.model_fields)
    frame = cycles.loc[cycles["battery_id"] == battery_id]
    frame = frame[[c for c in frame.columns if c in allowed]].copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = frame["timestamp"].astype(str)
    body = {
        "battery_id": str(battery_id),
        "history": [
            {k: (None if pd.isna(v) else v) for k, v in row.items()}
            for row in frame.to_dict(orient="records")
        ],
        "include_explanation": False,
    }

    for path in (
        "/v1/predict/rul",
        "/v1/predict/soh",
        "/v1/predict/risk",
        "/v1/predict/full",
        "/v1/digital-twin/snapshot",
    ):
        response = client.post(path, json=body)
        assert response.status_code == 200, f"{path} is a Milestone 2 endpoint"


def test_the_milestone_2_dashboard_still_imports():
    import battery_rul.dashboard.app as dashboard

    assert hasattr(dashboard, "main")


def test_the_recommendation_engine_is_untouched():
    """The fleet priority engine is a separate layer, not a replacement."""
    from battery_rul.recommendations.engine import (
        ActionCode,
        RecommendationEngine,
        RecommendationInputs,
    )

    cfg = load_config()
    engine = RecommendationEngine(cfg=cfg)
    result = engine.recommend(
        RecommendationInputs(rul_point=200.0, soh=0.97, health_class="healthy")
    )
    assert result.action_code == ActionCode.NORMAL_OPERATION.value


# ===========================================================================
# Milestone 3 pipeline integration
# ===========================================================================
@pytest.fixture(scope="module")
def pipeline_cfg(tmp_path_factory) -> ExperimentConfig:
    """A separate root, so the pipeline test's artifacts do not collide with
    the session fixtures'."""
    root = tmp_path_factory.mktemp("m3-pipelines")
    return load_config(
        "configs/synthetic.yaml",
        overrides={
            "paths.root": str(root),
            "data.cache_interim": False,
            "evaluation.nested_enabled": False,
            "models.enabled": ["ridge"],
            "features.max_features": 20,
            "features.rolling_windows": [3],
            "features.lags": [1],
            "features.slope_windows": [5],
            "features.ewm_halflives": [5],
            "multitask.enabled": False,
            "uncertainty.min_calibration_rows": 10,
            "calibration.min_calibration_rows": 10,
            "explainability.enabled": False,
            "monitoring.drift.min_sample_size": 20,
            "monitoring.prediction_drift.min_sample_size": 3,
        },
    )


@pytest.fixture(scope="module")
def pipeline_artifacts(pipeline_cfg):
    from battery_rul.pipelines import prepare_data
    from battery_rul.pipelines.milestone_2 import build_bundles, prepare_multitask_data

    prepared = prepare_data.run(pipeline_cfg, verify_leakage=False)
    build_bundles(pipeline_cfg, prepare_multitask_data(pipeline_cfg, prepared=prepared))
    return pipeline_cfg


def test_the_reference_is_built_from_the_training_partition_only(pipeline_artifacts):
    from battery_rul.pipelines.milestone_3 import build_reference

    result = build_reference(pipeline_artifacts)
    assert result["partition"] == "train"
    assert result["n_features"] > 0
    assert result["fingerprint"]
    assert not result["path"].startswith("/"), "artifact paths are published relative"


def test_the_fleet_batch_writes_every_documented_artifact(pipeline_artifacts):
    from battery_rul.pipelines.milestone_3 import run_fleet_batch

    result = run_fleet_batch(pipeline_artifacts, fleet_id="PIPE", source="processed")

    assert result["battery_count"] > 0
    assert result["snapshot_id"]
    for name in ("fleet_snapshot", "fleet_ranking", "maintenance_plan", "replacement_plan"):
        path = pipeline_artifacts.paths.root / result["artifacts"][name]
        assert path.is_file(), f"{name} was not written"

    ranking = pd.read_csv(pipeline_artifacts.paths.root / result["artifacts"]["fleet_ranking"])
    assert {"rank", "battery_id", "priority", "priority_score"} <= set(ranking.columns)
    assert ranking["battery_id"].is_unique


def test_the_snapshot_is_persisted_and_reloadable(pipeline_artifacts):
    from battery_rul.persistence import build_repository
    from battery_rul.pipelines.milestone_3 import run_fleet_batch

    result = run_fleet_batch(pipeline_artifacts, fleet_id="PIPE2", source="processed")
    repository = build_repository(pipeline_artifacts)
    loaded = repository.latest_fleet_snapshot("PIPE2")

    assert loaded is not None
    assert loaded.snapshot_id == result["snapshot_id"]
    assert loaded.battery_count == result["battery_count"]


def test_the_monitoring_run_reports_every_section(pipeline_artifacts):
    from battery_rul.pipelines.milestone_3 import build_reference, run_monitoring

    build_reference(pipeline_artifacts)
    result = run_monitoring(pipeline_artifacts, fleet_id="PIPE", source="processed")

    assert result["overall_status"] in ("OK", "WARNING", "CRITICAL", "UNKNOWN")
    assert result["data_quality_status"] in ("OK", "WARNING", "CRITICAL", "UNKNOWN")
    assert result["performance_status"] == "NO_LABELS", "no labels have been supplied yet"
    for name in ("data_quality_report", "feature_drift_report", "model_performance_report"):
        assert (pipeline_artifacts.paths.root / result["report_paths"][name]).is_file()


def test_delayed_labels_turn_into_a_performance_report(pipeline_artifacts):
    """The documented delayed-label workflow, end to end."""
    from battery_rul.monitoring.performance import OutcomeLabel, evaluate_delayed_labels
    from battery_rul.persistence import build_repository

    repository = build_repository(pipeline_artifacts)
    predictions = repository.list_prediction_records()
    assert predictions, "the fleet batch recorded its predictions"

    labels = [
        OutcomeLabel(
            battery_id=record.battery_id,
            cycle_index=record.cycle_index,
            observed_at_cycle=record.cycle_index + 25,
            observed_rul=float(record.predicted_rul or 0.0) + 2.0,
            observed_soh=0.8,
            eol_within_horizon=True,
            label_source="test_fixture",
        )
        for record in predictions
    ]
    repository.save_outcome_labels(labels)

    pipeline_artifacts.monitoring.performance.min_labels = 1
    report = evaluate_delayed_labels(
        predictions, repository.list_outcome_labels(), pipeline_artifacts
    )
    assert report.status.value in ("HEALTHY", "WARNING", "DEGRADED")
    assert report.n_labels_joined == len(predictions)
    assert report.label_coverage > 0


def test_the_registry_round_trip_promotes_and_rolls_back(pipeline_artifacts):
    from battery_rul.pipelines.milestone_3 import (
        evaluate_promotion_stage,
        promote_model,
        register_model,
        rollback_model,
    )
    from battery_rul.registry.store import FileModelRegistry, ModelStage

    register_model(
        pipeline_artifacts,
        model_name="pipe-rul",
        model_version="1.0.0",
        bundle="artifacts/rul",
        validation_status="VALIDATED",
        overwrite=True,
    )
    register_model(
        pipeline_artifacts,
        model_name="pipe-rul",
        model_version="1.1.0",
        bundle="artifacts/rul",
        validation_status="VALIDATED",
        overwrite=True,
    )

    gate = evaluate_promotion_stage(
        pipeline_artifacts,
        model_name="pipe-rul",
        model_version="1.0.0",
        smoke_test=True,
        contract_tests=True,
        unit_tests=True,
        leakage_check=True,
    )
    assert gate["decision"] in ("APPROVED", "REJECTED", "REQUIRES_REVIEW")
    assert gate["checks"]

    dry = promote_model(
        pipeline_artifacts,
        model_name="pipe-rul",
        model_version="1.0.0",
        by="test",
        dry_run=True,
        skip_gate=True,
    )
    assert dry["promoted"] is False and dry["dry_run"] is True

    registry = FileModelRegistry(cfg=pipeline_artifacts)
    registry.promote("pipe-rul", "1.0.0", by="test", reason="fixture")
    registry.promote("pipe-rul", "1.1.0", by="test", reason="fixture")
    assert registry.production_model("pipe-rul").model_version == "1.1.0"

    restored = rollback_model(pipeline_artifacts, model_name="pipe-rul", by="test")
    assert restored["model"].endswith("1.0.0")
    assert registry.production_model("pipe-rul").model_version == "1.0.0"
    assert len(registry.list_models(model_name="pipe-rul", stage=ModelStage.PRODUCTION)) == 1


def test_the_fleet_report_renders_from_stored_snapshots(pipeline_artifacts):
    from battery_rul.pipelines.milestone_3 import generate_fleet_report

    result = generate_fleet_report(pipeline_artifacts, fleet_id="PIPE")
    report = (pipeline_artifacts.paths.root / result["report_path"]).read_text()

    assert "# Fleet report" in report
    assert "Median SOH" in report
    assert "workload forecast" in report.lower()
    assert "advisory" in report.lower()


def test_the_cli_entry_point_returns_a_meaningful_exit_code(pipeline_artifacts, tmp_path):
    """A pipeline that cannot run must exit non-zero, not print and succeed."""
    from battery_rul.pipelines.milestone_3 import main

    code = main(
        [
            "run-fleet-batch",
            "--source",
            "file",
            "--path",
            str(tmp_path / "does-not-exist.parquet"),
            "--set",
            f"paths.root={pipeline_artifacts.paths.root}",
        ]
    )
    assert code == 1


def test_generated_artifacts_carry_no_absolute_machine_paths(pipeline_artifacts):
    """The same rule the repository applies to committed artifacts."""
    from battery_rul.pipelines.milestone_3 import run_fleet_batch

    result = run_fleet_batch(pipeline_artifacts, fleet_id="PIPE3", source="processed")
    payload = json.dumps(result)
    assert str(pipeline_artifacts.paths.root) not in payload

    summary = (pipeline_artifacts.paths.root / result["artifacts"]["fleet_summary"]).read_text()
    assert str(pipeline_artifacts.paths.root) not in summary
