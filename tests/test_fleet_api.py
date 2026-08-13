"""Fleet API contract tests.

These pin the *published shapes*: status codes, response keys, pagination
metadata, error bodies and the security defaults. A change here is a change to
a contract a client depends on, and should be a deliberate one with a version
bump behind it.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from battery_rul.api.app import create_app
from battery_rul.api.fleet_schemas import MAX_BATTERIES_PER_REQUEST
from battery_rul.api.schemas import CycleRecord
from battery_rul.persistence import build_repository


@pytest.fixture(scope="module")
def api(m3_platform, fleet_service):
    cfg, cycles = m3_platform
    repository = build_repository(cfg)
    app = create_app(cfg, service=fleet_service.twin, repository=repository)
    app.state.fleet_service = fleet_service
    return TestClient(app), cfg, cycles, repository


@pytest.fixture(scope="module")
def payload(api):
    """Two cells' histories in the request shape a client would send."""
    _, _, cycles, _ = api
    allowed = set(CycleRecord.model_fields)
    batteries = []
    for battery_id in sorted(cycles["battery_id"].unique())[:2]:
        frame = cycles.loc[cycles["battery_id"] == battery_id]
        frame = frame[[c for c in frame.columns if c in allowed]].copy()
        if "timestamp" in frame.columns:
            frame["timestamp"] = frame["timestamp"].astype(str)
        records = [
            {k: (None if pd.isna(v) else v) for k, v in row.items()}
            for row in frame.to_dict(orient="records")
        ]
        batteries.append({"battery_id": str(battery_id), "history": records})
    return {"fleet_id": "API-FLEET", "batteries": batteries}


# ---------------------------------------------------------------------------
# Operational endpoints
# ---------------------------------------------------------------------------
def test_health_answers_and_names_the_service(api):
    client, _, _, _ = api
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["service"] == "battery-digital-twin"


def test_ready_reports_the_loaded_bundles(api):
    client, _, _, _ = api
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_version_reports_both_schema_versions(api):
    client, _, _, _ = api
    body = client.get("/version").json()
    assert body["api_version"] == "v1"
    assert body["snapshot_schema_version"]


def test_metrics_are_exposed_in_prometheus_format(api):
    client, _, _, _ = api
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "api_requests_total" in response.text


def test_every_response_carries_a_request_id(api):
    client, cfg, _, _ = api
    response = client.get("/health")
    assert response.headers[cfg.service.request_id_header]


# ---------------------------------------------------------------------------
# POST /v1/fleet/snapshot
# ---------------------------------------------------------------------------
def test_the_snapshot_endpoint_returns_the_documented_shape(api, payload):
    client, _, _, _ = api
    body = client.post("/v1/fleet/snapshot", json=payload).json()

    for key in (
        "fleet_id",
        "snapshot_id",
        "generated_at_utc",
        "schema_version",
        "summary",
        "health_distribution",
        "risk_distribution",
        "maintenance_summary",
        "replacement_summary",
        "workload_forecast",
        "fleet_statistics",
        "data_quality",
        "drift_status",
        "model_metadata",
        "batteries",
        "disclaimer",
    ):
        assert key in body, f"{key} is part of the published contract"
    assert body["battery_count"] == 2


def test_battery_records_are_paged_with_honest_metadata(api, payload):
    client, _, _, _ = api
    body = client.post("/v1/fleet/snapshot", json={**payload, "page_size": 1}).json()

    pagination = body["batteries"]["pagination"]
    assert len(body["batteries"]["items"]) == 1
    assert pagination["total_items"] == 2
    assert pagination["total_pages"] == 2
    assert pagination["has_next"] is True

    second = client.post("/v1/fleet/snapshot", json={**payload, "page_size": 1, "page": 2}).json()
    assert second["batteries"]["pagination"]["has_next"] is False
    assert (
        second["batteries"]["items"][0]["battery_id"] != body["batteries"]["items"][0]["battery_id"]
    )


def test_aggregates_are_never_paged(api, payload):
    """A summary of a page is not a summary of the fleet."""
    client, _, _, _ = api
    body = client.post("/v1/fleet/snapshot", json={**payload, "page_size": 1}).json()
    assert body["summary"]["battery_count"] == 2
    assert body["fleet_statistics"]["rul_denominator"] <= 2


def test_records_can_be_omitted_for_a_cheap_poll(api, payload):
    client, _, _, _ = api
    body = client.post(
        "/v1/fleet/snapshot", json={**payload, "include_battery_records": False}
    ).json()
    assert body["batteries"] is None
    assert body["summary"]["battery_count"] == 2


def test_a_partial_failure_is_a_200_with_the_failure_reported(api, payload):
    client, _, _, _ = api
    broken = {
        "battery_id": "BROKEN",
        "history": [{"cycle_index": 1, "capacity_ah": 2.0}] * 2,
    }
    body = client.post(
        "/v1/fleet/snapshot", json={**payload, "batteries": [*payload["batteries"], broken]}
    )
    assert body.status_code == 200

    data = body.json()
    assert data["battery_count"] == 3
    records = {r["battery_id"]: r for r in data["ingestion_records"]}
    assert records["BROKEN"]["status"] == "failed"
    assert records["BROKEN"]["errors"]


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------
def test_duplicate_battery_ids_are_rejected_before_inference(api, payload):
    client, _, _, _ = api
    duplicated = {**payload, "batteries": [payload["batteries"][0], payload["batteries"][0]]}
    response = client.post("/v1/fleet/snapshot", json=duplicated)
    assert response.status_code == 422
    assert "Duplicate" in response.text


def test_an_empty_fleet_is_rejected(api):
    client, _, _, _ = api
    assert (
        client.post("/v1/fleet/snapshot", json={"fleet_id": "X", "batteries": []}).status_code
        == 422
    )


def test_a_fleet_id_with_a_path_separator_is_rejected(api, payload):
    client, _, _, _ = api
    response = client.post("/v1/fleet/snapshot", json={**payload, "fleet_id": "../etc"})
    assert response.status_code == 422


def test_unknown_request_fields_are_rejected(api, payload):
    client, _, _, _ = api
    response = client.post("/v1/fleet/snapshot", json={**payload, "artifact_path": "/etc/passwd"})
    assert response.status_code == 422


def test_the_online_batch_limit_is_enforced(api, payload):
    client, cfg, _, _ = api
    original = cfg.fleet.max_batteries_per_request
    cfg.fleet.max_batteries_per_request = 1
    try:
        response = client.post("/v1/fleet/snapshot", json=payload)
        assert response.status_code == 413
        assert "batch pipeline" in response.text
    finally:
        cfg.fleet.max_batteries_per_request = original


def test_the_schema_caps_the_request_independently_of_configuration():
    from battery_rul.api.fleet_schemas import FleetRequest

    assert MAX_BATTERIES_PER_REQUEST == 500
    assert FleetRequest.model_fields["batteries"].metadata


# ---------------------------------------------------------------------------
# Ranking, plans, monitoring
# ---------------------------------------------------------------------------
def test_ranking_names_its_criterion_and_its_limits(api, payload):
    client, _, _, _ = api
    body = client.post("/v1/fleet/rank", json={**payload, "rank_by": "rul"}).json()

    assert body["rank_by"] == "rul"
    assert "not an optimum" in body["methodology_note"]
    identifiers = [item["battery_id"] for item in body["ranking"]["items"]]
    assert len(identifiers) == len(set(identifiers))


def test_an_unsupported_ranking_key_is_rejected(api, payload):
    client, _, _, _ = api
    assert client.post("/v1/fleet/rank", json={**payload, "rank_by": "vibes"}).status_code == 422


def test_the_maintenance_plan_carries_its_disclaimer(api, payload):
    client, _, _, _ = api
    body = client.post("/v1/fleet/maintenance-plan", json=payload).json()
    assert body["disclaimer"]
    assert body["workload_forecast"]["buckets"]


def test_the_replacement_plan_carries_its_caveats(api, payload):
    client, _, _, _ = api
    body = client.post("/v1/fleet/replacement-plan", json=payload).json()
    assert body["caveats"]
    assert "candidates" in body


def test_the_online_monitoring_run_does_not_claim_a_performance_verdict(api, payload):
    client, _, _, _ = api
    body = client.post("/v1/fleet/monitoring/run", json=payload).json()
    assert body["performance_status"] == "NOT_EVALUATED_ONLINE"
    assert any("delayed-label" in w.lower() for w in body["warnings"])


# ---------------------------------------------------------------------------
# Stored reads
# ---------------------------------------------------------------------------
def test_a_stored_snapshot_can_be_read_back(api, fleet_snapshot):
    client, _, _, repository = api
    repository.save_fleet_snapshot(fleet_snapshot)

    body = client.get(f"/v1/fleet/{fleet_snapshot.fleet_id}/latest").json()
    assert body["snapshot_id"] == fleet_snapshot.snapshot_id

    summary = client.get(f"/v1/fleet/{fleet_snapshot.fleet_id}/summary").json()
    assert summary["summary"]["battery_count"] == fleet_snapshot.battery_count


def test_an_unknown_fleet_is_a_404_that_says_what_to_run(api):
    client, _, _, _ = api
    response = client.get("/v1/fleet/NO-SUCH-FLEET/summary")
    assert response.status_code == 404
    assert "run_fleet_batch" in response.json()["detail"]


def test_critical_batteries_are_listed_with_the_policy_that_defined_them(api, fleet_snapshot):
    client, cfg, _, repository = api
    repository.save_fleet_snapshot(fleet_snapshot)
    body = client.get(f"/v1/fleet/{fleet_snapshot.fleet_id}/critical-batteries").json()
    assert body["critical_priorities"] == list(cfg.fleet.critical_priorities)


def test_alerts_are_paginated(api, fleet_snapshot):
    client, _, _, _ = api
    body = client.get(f"/v1/fleet/{fleet_snapshot.fleet_id}/alerts").json()
    assert "pagination" in body
    assert "human review" in body["note"]


# ---------------------------------------------------------------------------
# Registry endpoints
# ---------------------------------------------------------------------------
def test_the_model_list_never_publishes_a_filesystem_path(api):
    client, _, _, _ = api
    body = client.get("/v1/models").json()
    assert "models" in body
    for entry in body["models"]:
        assert "bundle_path" not in entry


def test_no_production_model_is_a_404_that_says_how_to_promote_one(api):
    client, _, _, _ = api
    response = client.get("/v1/models/production")
    if response.status_code == 404:
        assert "promote_model" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Security defaults
# ---------------------------------------------------------------------------
def test_administrative_endpoints_are_disabled_by_default(api):
    client, cfg, _, _ = api
    assert cfg.deployment.admin_endpoints_enabled is False
    response = client.post(
        "/v1/admin/models/promote",
        json={"model_name": "x", "model_version": "1.0.0", "by": "tester"},
    )
    assert response.status_code == 403
    assert "disabled" in response.json()["detail"]


def test_no_cross_origin_access_by_default(api):
    client, cfg, _, _ = api
    assert cfg.deployment.cors_allow_origins == []
    response = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in response.headers}


def test_error_bodies_do_not_leak_internal_paths(api):
    client, _, _, _ = api
    response = client.get("/v1/fleet/NO-SUCH-FLEET/latest")
    assert response.status_code == 404
    assert "/Users/" not in response.text
    assert "/home/" not in response.text


def test_the_openapi_document_describes_the_fleet_endpoints(api):
    client, _, _, _ = api
    paths = client.get("/openapi.json").json()["paths"]
    for path in (
        "/v1/fleet/snapshot",
        "/v1/fleet/rank",
        "/v1/fleet/maintenance-plan",
        "/v1/fleet/replacement-plan",
        "/v1/fleet/{fleet_id}/latest",
        "/v1/fleet/{fleet_id}/summary",
        "/v1/fleet/{fleet_id}/critical-batteries",
        "/v1/fleet/{fleet_id}/alerts",
        "/v1/models",
        "/v1/monitoring/latest",
    ):
        assert path in paths, f"{path} is part of the published API"
