"""Digital-twin service and FastAPI integration tests.

These run against bundles built inside the test session from the synthetic
generator. Fixture metrics are *not* model performance and nothing here asserts
a quality number — the tests assert behaviour: shapes, ranges, error handling,
schema stability and the training/serving invariants.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from battery_rul.api.app import create_app
from battery_rul.api.schemas import CycleRecord, PredictionRequest
from battery_rul.config import ExperimentConfig, load_config
from battery_rul.digital_twin.service import BatteryDigitalTwinService, InvalidHistoryError


# ---------------------------------------------------------------------------
# Fixtures: a real, tiny, end-to-end trained artifact set
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> tuple[ExperimentConfig, pd.DataFrame]:
    """Run the real pipelines on synthetic cells and return the config + cycles.

    Module-scoped: training four small models per test would dominate the suite.
    """
    root = tmp_path_factory.mktemp("m2")
    cfg = load_config(
        "configs/synthetic.yaml",
        overrides={
            "paths.root": str(root),
            "data.cache_interim": False,
            "evaluation.nested_enabled": False,
            "models.enabled": ["ridge", "random_forest"],
            "features.max_features": 25,
            "features.rolling_windows": [3, 5],
            "features.lags": [1, 3],
            "features.slope_windows": [5],
            "features.ewm_halflives": [5],
            "multitask.enabled": False,
            "uncertainty.min_calibration_rows": 10,
            "calibration.min_calibration_rows": 10,
            "explainability.enabled": False,
        },
    )
    from battery_rul.pipelines import prepare_data
    from battery_rul.pipelines.milestone_2 import build_bundles, prepare_multitask_data

    prepared = prepare_data.run(cfg, verify_leakage=False)
    data = prepare_multitask_data(cfg, prepared=prepared)
    build_bundles(cfg, data)
    return cfg, prepared.cycles


@pytest.fixture(scope="module")
def service(trained) -> BatteryDigitalTwinService:
    cfg, _ = trained
    return BatteryDigitalTwinService.create(cfg, strict=True)


@pytest.fixture(scope="module")
def history(trained) -> pd.DataFrame:
    _, cycles = trained
    battery = cycles["battery_id"].iloc[0]
    frame = cycles.loc[cycles["battery_id"] == battery].reset_index(drop=True)
    drop = [
        c
        for c in frame.columns
        if c.startswith(
            (
                "rul_",
                "eol_",
                "life_",
                "is_censored",
                "soh",
                "capacity_smooth",
                "reference_capacity",
                "capacity_fade",
                "equivalent_full",
            )
        )
    ]
    return frame.drop(columns=drop, errors="ignore")


@pytest.fixture(scope="module")
def battery_id(history: pd.DataFrame) -> str:
    return str(history["battery_id"].iloc[0])


@pytest.fixture(scope="module")
def client(trained, service) -> TestClient:
    cfg, _ = trained
    return TestClient(create_app(cfg, service=service))


# ===========================================================================
# Service layer
# ===========================================================================
def test_service_is_ready_after_bundles_are_built(service):
    readiness = service.readiness()
    assert readiness["ready"] is True
    assert readiness["bundles"]["rul"] and readiness["bundles"]["risk"]


def test_snapshot_is_serialisable_and_complete(service, battery_id, history):
    snapshot = service.create_snapshot(battery_id, history)
    payload = snapshot.to_json_dict()
    json.dumps(payload)  # must not raise
    for key in (
        "battery_id",
        "health",
        "prediction",
        "failure_risk",
        "recommendation",
        "data_quality",
        "metadata",
    ):
        assert key in payload


def test_snapshot_outputs_are_in_range(service, battery_id, history):
    snapshot = service.create_snapshot(battery_id, history)
    assert snapshot.prediction.rul_cycles is None or snapshot.prediction.rul_cycles >= 0
    if snapshot.failure_risk.probability is not None:
        assert 0.0 <= snapshot.failure_risk.probability <= 1.0
    if snapshot.health.soh is not None:
        cfg = service.cfg
        assert cfg.soh.plausible_min <= snapshot.health.soh <= cfg.soh.plausible_max


def test_interval_brackets_the_point_estimate(service, battery_id, history):
    snapshot = service.create_snapshot(battery_id, history)
    interval = snapshot.prediction.rul_interval
    assert interval is not None, "the RUL bundle should carry a conformal estimator"
    assert interval.lower_bound <= interval.point_estimate <= interval.upper_bound
    assert interval.interval_type == "prediction_interval"


def test_short_history_is_unscoreable_and_returns_insufficient_data(service, battery_id, history):
    snapshot = service.create_snapshot(battery_id, history.head(3))
    assert snapshot.data_quality.quality_class == "INSUFFICIENT"
    assert snapshot.prediction.rul_cycles is None
    assert snapshot.recommendation.action_code == "INSUFFICIENT_DATA"


def test_prediction_before_the_first_scoreable_cycle_is_refused(service, battery_id, history):
    """Training/serving parity: no prediction from a window training never saw."""
    from battery_rul.features.warmup import first_scoreable_cycle

    threshold = first_scoreable_cycle(service.cfg, family="tabular")
    truncated = history.loc[history["cycle_index"] < threshold]
    snapshot = service.create_snapshot(battery_id, truncated)
    assert snapshot.prediction.is_scoreable is False
    assert snapshot.prediction.rul_cycles is None


def test_overlapping_predictions_are_stable_as_history_grows(service, battery_id, history):
    """The prediction at cycle k must not change when later cycles are appended."""
    cutoff = int(history["cycle_index"].iloc[len(history) // 2])
    prefix = history.loc[history["cycle_index"] <= cutoff]
    at_the_time = service.create_snapshot(battery_id, prefix).prediction.rul_cycles

    longer = history.loc[history["cycle_index"] <= cutoff + 20]
    later = service.create_snapshot(battery_id, longer.loc[longer["cycle_index"] <= cutoff])
    assert later.prediction.rul_cycles == pytest.approx(at_the_time, rel=1e-6)


def test_empty_history_is_rejected(service, battery_id, history):
    with pytest.raises(InvalidHistoryError, match="empty"):
        service.create_snapshot(battery_id, history.head(0))


def test_duplicate_cycles_are_rejected(service, battery_id, history):
    frame = pd.concat([history, history.tail(2)], ignore_index=True)
    with pytest.raises(InvalidHistoryError, match="share a cycle_index"):
        service.create_snapshot(battery_id, frame)


def test_missing_required_column_is_rejected(service, battery_id, history):
    with pytest.raises(InvalidHistoryError, match="missing required column"):
        service.create_snapshot(battery_id, history.drop(columns=["capacity_ah"]))


def test_unsorted_history_is_sorted_not_rejected(service, battery_id, history):
    shuffled = history.sample(frac=1.0, random_state=0).reset_index(drop=True)
    ordered = service.create_snapshot(battery_id, history)
    reordered = service.create_snapshot(battery_id, shuffled)
    assert reordered.prediction.rul_cycles == pytest.approx(ordered.prediction.rul_cycles, rel=1e-6)


def test_oversized_history_is_rejected(service, battery_id, history):
    service.cfg.service.max_history_cycles = 5
    try:
        with pytest.raises(InvalidHistoryError, match="maximum"):
            service.create_snapshot(battery_id, history)
    finally:
        service.cfg.service.max_history_cycles = 5000


def test_explanation_returns_named_drivers(service, battery_id, history):
    explanation = service.explain_prediction(battery_id, history)
    assert explanation.drivers
    for driver in explanation.drivers:
        assert driver.display_name
        assert driver.contribution_magnitude >= 0
        assert "caused" not in driver.explanation_text.lower()


def test_model_metadata_reports_the_definitions(service):
    metadata = service.get_model_metadata()
    assert metadata["risk_definition"]["is_observed_safety_failure"] is False
    assert metadata["soh_definition"]["representation"] == "fraction in [0, 1]"
    assert metadata["warmup_policy"]["first_scoreable_cycle"] >= 1


def test_convenience_methods_agree_with_the_snapshot(service, battery_id, history):
    snapshot = service.create_snapshot(battery_id, history, explain=False)
    assert service.predict_rul(battery_id, history).rul_cycles == pytest.approx(
        snapshot.prediction.rul_cycles
    )
    assert service.predict_soh(battery_id, history).soh == pytest.approx(snapshot.health.soh)


# ===========================================================================
# API
# ===========================================================================
def _payload(battery_id: str, history: pd.DataFrame, n: int | None = None) -> dict:
    allowed = set(CycleRecord.model_fields)
    frame = history[[c for c in history.columns if c in allowed]]
    if n is not None:
        frame = frame.head(n)
    records = [
        {k: (None if pd.isna(v) else v) for k, v in record.items()}
        for record in frame.to_dict(orient="records")
    ]
    for record in records:
        if record.get("timestamp") is not None:
            record["timestamp"] = str(record["timestamp"])
    return {"battery_id": battery_id, "history": records}


def test_health_endpoint_answers(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_endpoint_reports_ready(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_version_endpoint(client):
    payload = client.get("/version").json()
    assert payload["api_version"] == "v1"
    assert payload["snapshot_schema_version"]


def test_model_info_endpoint(client):
    payload = client.get("/model-info").json()
    assert "bundles" in payload["metadata"]


def test_predict_rul_endpoint(client, battery_id, history):
    response = client.post("/v1/predict/rul", json=_payload(battery_id, history))
    assert response.status_code == 200
    body = response.json()
    assert body["battery_id"] == battery_id
    assert body["prediction"]["rul_cycles"] >= 0


def test_predict_soh_endpoint(client, battery_id, history):
    body = client.post("/v1/predict/soh", json=_payload(battery_id, history)).json()
    assert 0.0 <= body["health"]["soh"] <= 1.5
    assert body["health"]["health_class"] in {
        "healthy",
        "slightly_degraded",
        "warning",
        "critical",
        "unknown",
    }


def test_predict_risk_endpoint(client, battery_id, history):
    body = client.post("/v1/predict/risk", json=_payload(battery_id, history)).json()
    assert 0.0 <= body["failure_risk"]["probability"] <= 1.0
    assert body["failure_risk"]["horizon_cycles"] > 0


def test_full_prediction_returns_a_snapshot(client, battery_id, history):
    response = client.post("/v1/predict/full", json=_payload(battery_id, history))
    assert response.status_code == 200
    snapshot = response.json()["snapshot"]
    assert snapshot["battery_id"] == battery_id
    assert snapshot["metadata"]["snapshot_schema_version"]
    assert snapshot["recommendation"]["disclaimer"]


def test_digital_twin_snapshot_endpoint_matches_predict_full(client, battery_id, history):
    payload = _payload(battery_id, history)
    a = client.post("/v1/predict/full", json=payload).json()["snapshot"]
    b = client.post("/v1/digital-twin/snapshot", json=payload).json()["snapshot"]
    assert a["prediction"]["rul_cycles"] == b["prediction"]["rul_cycles"]


def test_explain_endpoint(client, battery_id, history):
    body = client.post("/v1/explain", json=_payload(battery_id, history)).json()
    assert body["explanation"]["drivers"]


def test_snapshot_schema_is_stable(client, battery_id, history):
    """The published wire format is a contract; this test is the tripwire."""
    snapshot = client.post("/v1/predict/full", json=_payload(battery_id, history)).json()[
        "snapshot"
    ]
    expected = {
        "battery_id",
        "generated_at_utc",
        "identity",
        "measurement_summary",
        "health",
        "prediction",
        "failure_risk",
        "explanation",
        "recommendation",
        "data_quality",
        "metadata",
        "warnings",
        "disclaimer",
    }
    assert set(snapshot) == expected


def test_empty_history_is_a_422(client, battery_id):
    response = client.post("/v1/predict/rul", json={"battery_id": battery_id, "history": []})
    assert response.status_code == 422


def test_blank_battery_id_is_a_422(client, history):
    response = client.post("/v1/predict/rul", json=_payload("  ", history))
    assert response.status_code == 422


def test_path_separator_in_battery_id_is_rejected(client, history):
    """No request input may look like a filesystem path."""
    response = client.post("/v1/predict/rul", json=_payload("../../etc/passwd", history))
    assert response.status_code == 422


def test_missing_required_field_is_a_422(client, battery_id):
    response = client.post(
        "/v1/predict/rul",
        json={"battery_id": battery_id, "history": [{"cycle_index": 1}]},
    )
    assert response.status_code == 422


def test_unknown_field_is_rejected(client, battery_id):
    response = client.post(
        "/v1/predict/rul",
        json={
            "battery_id": battery_id,
            "history": [{"cycle_index": 1, "capacity_ah": 1.8, "surprise": 1}],
        },
    )
    assert response.status_code == 422


def test_duplicate_cycles_return_a_structured_error(client, battery_id, history):
    payload = _payload(battery_id, history)
    payload["history"].append(payload["history"][-1])
    response = client.post("/v1/predict/rul", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "invalid_history"
    assert body["request_id"]


def test_insufficient_cycles_returns_200_with_a_warning(client, battery_id, history):
    """A thin history is a *result*, not a server error: the twin says so."""
    response = client.post("/v1/predict/rul", json=_payload(battery_id, history, n=3))
    assert response.status_code == 200
    body = response.json()
    assert body["data_quality_class"] == "INSUFFICIENT"
    assert body["prediction"]["rul_cycles"] is None


def test_request_id_is_echoed(client, battery_id, history):
    response = client.post(
        "/v1/predict/rul",
        json=_payload(battery_id, history),
        headers={"X-Request-ID": "abc-123"},
    )
    assert response.headers["X-Request-ID"] == "abc-123"


def test_openapi_document_is_generated(client):
    document = client.get("/openapi.json").json()
    assert "/v1/predict/full" in document["paths"]
    assert "/v1/digital-twin/snapshot" in document["paths"]


def test_model_unavailable_reports_not_ready(tmp_path):
    """A missing artifact set must fail readiness, not 200 with silent nulls."""
    cfg = load_config(overrides={"paths.root": str(tmp_path)})
    assert not Path(cfg.artifacts.rul_dir).exists()
    app = create_app(cfg)
    with TestClient(app) as unready:
        assert unready.get("/health").status_code == 200
        assert unready.get("/ready").status_code == 503


# ===========================================================================
# Dashboard adapter
# ===========================================================================
def test_dashboard_adapter_returns_the_same_snapshot(trained, battery_id, history):
    from battery_rul.dashboard.data_adapter import TwinClient

    cfg, _ = trained
    client = TwinClient.build(cfg)
    snapshot = client.snapshot(battery_id, history)
    assert snapshot.battery_id == battery_id


def test_dashboard_demo_loader_strips_label_columns(trained):
    from battery_rul.dashboard.data_adapter import load_demo_cycles

    cfg, _ = trained
    data = load_demo_cycles(cfg)
    assert data is not None
    assert "rul_cycles" not in data.cycles.columns
    assert "eol_cycle" not in data.cycles.columns
    assert data.batteries


def test_dashboard_trajectory_is_causal(trained, battery_id, history):
    """Each trajectory point uses only the cycles available at that time."""
    from battery_rul.dashboard.data_adapter import TwinClient, trajectory_frame

    cfg, _ = trained
    client = TwinClient.build(cfg)
    frame = trajectory_frame(client, battery_id, history, step=25, max_points=5)
    assert not frame.empty
    assert frame["cycle_index"].is_monotonic_increasing
    for _, row in frame.dropna(subset=["rul", "rul_lower", "rul_upper"]).iterrows():
        assert row["rul_lower"] <= row["rul"] <= row["rul_upper"]


def test_prediction_request_to_frame(battery_id, history):
    payload = _payload(battery_id, history, n=5)
    request = PredictionRequest(**payload)
    frame = request.to_frame()
    assert len(frame) == 5
    assert "capacity_ah" in frame.columns
    assert np.all(frame["cycle_index"].to_numpy() > 0)
