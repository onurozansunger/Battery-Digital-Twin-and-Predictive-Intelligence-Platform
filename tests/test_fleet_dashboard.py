"""The fleet dashboard's data adapter and presentation helpers.

Streamlit layout is not unit-testable without a script runner, which is exactly
why all the data access and table building lives in the adapter. These tests
exercise that; the dashboard script itself is covered by an import smoke test,
here and in CI.
"""

from __future__ import annotations

from battery_rul.dashboard.fleet_adapter import (
    FleetDashboardAdapter,
    battery_table,
    score_breakdown_table,
    workload_table,
)
from battery_rul.fleet.analytics import fleet_trend_series


def test_the_dashboard_script_imports_and_declares_its_pages():
    import battery_rul.dashboard.fleet_app as app

    assert hasattr(app, "main")
    assert len(app.PAGES) == 14
    assert "Executive Fleet Overview" in app.PAGES
    assert "Architecture & Limitations" in app.PAGES


def test_the_adapter_reads_a_stored_snapshot(m3_config, fleet_snapshot):
    from battery_rul.persistence import build_repository

    build_repository(m3_config).save_fleet_snapshot(fleet_snapshot)
    adapter = FleetDashboardAdapter.build(m3_config)
    loaded = adapter.latest_snapshot(fleet_snapshot.fleet_id)

    assert loaded is not None
    assert loaded.snapshot_id == fleet_snapshot.snapshot_id


def test_an_unknown_fleet_reads_as_none_not_an_invented_one(m3_config):
    adapter = FleetDashboardAdapter.build(m3_config)
    assert adapter.latest_snapshot("NO-SUCH-FLEET") is None


def test_the_battery_table_labels_measured_and_predicted_columns(fleet_snapshot):
    frame = battery_table(fleet_snapshot)
    assert len(frame) == fleet_snapshot.battery_count
    assert "SOH (measured)" in frame.columns
    assert "RUL (predicted)" in frame.columns
    assert "risk (predicted)" in frame.columns
    assert frame["battery_id"].is_unique


def test_the_score_breakdown_table_shows_every_component(fleet_snapshot):
    evaluated = fleet_snapshot.evaluated()
    if not evaluated:
        return
    frame = score_breakdown_table(fleet_snapshot, evaluated[0].battery_id)
    assert not frame.empty
    assert {"component", "normalised", "weight", "contribution", "transformation"} <= set(
        frame.columns
    )


def test_the_score_breakdown_of_an_unknown_cell_is_empty_not_an_error(fleet_snapshot):
    assert score_breakdown_table(fleet_snapshot, "NOPE").empty


def test_the_workload_table_carries_the_uncertainty_bracket(fleet_snapshot):
    frame = workload_table(fleet_snapshot)
    assert "lower (optimistic)" in frame.columns
    assert "upper (conservative)" in frame.columns
    assert len(frame) == len(fleet_snapshot.workload_forecast.buckets)


def test_the_adapter_scores_a_demo_fleet_and_labels_it(m3_config):
    adapter = FleetDashboardAdapter.build(m3_config)
    snapshot = adapter.run_fleet(source="demo", fleet_id="DASH-DEMO", demo_size=4)

    assert snapshot.identity.is_demo_data is True
    assert snapshot.battery_count == 4


def test_trend_series_need_at_least_two_snapshots(m3_config, fleet_snapshot):
    points = fleet_trend_series([fleet_snapshot], metric="median_soh")
    assert len(points) == 1
    assert points[0].denominator is not None

    second = fleet_snapshot.model_copy(
        update={"snapshot_id": "later", "generated_at_utc": "2030-01-01T00:00:00+00:00"}
    )
    points = fleet_trend_series([second, fleet_snapshot], metric="median_rul")
    assert [p.generated_at_utc for p in points] == sorted(p.generated_at_utc for p in points)


def test_the_adapter_reports_readiness_without_raising(m3_config):
    adapter = FleetDashboardAdapter.build(m3_config)
    assert "ready" in adapter.readiness()


def test_the_adapter_reads_the_registry(m3_config):
    adapter = FleetDashboardAdapter.build(m3_config)
    payload = adapter.models()
    assert "models" in payload
    assert isinstance(payload["models"], list)


# ---------------------------------------------------------------------------
# Battery Passport (optional demonstration layer)
# ---------------------------------------------------------------------------
def test_a_passport_never_invents_manufacturing_facts(m3_config, fleet_snapshot):
    from battery_rul.fleet.passport import build_passport

    record = fleet_snapshot.batteries[0]
    passport = build_passport(record, m3_config)

    assert passport.manufacturer is None
    assert passport.chemistry is None
    assert passport.carbon_footprint_kg_co2e is None
    sources = {s.field_group: s.source for s in passport.field_sources}
    assert sources["identity_and_manufacturing"] == "unavailable"
    assert sources["carbon_footprint"] == "unavailable"


def test_a_passport_marks_itself_as_not_compliance(m3_config, fleet_snapshot):
    from battery_rul.fleet.passport import build_passport

    passport = build_passport(fleet_snapshot.batteries[0], m3_config)
    assert "not a regulatory battery passport" in passport.compliance_notice
    assert any("not a regulatory" in c for c in passport.caveats)


def test_supplied_metadata_is_labelled_as_supplied(m3_config, fleet_snapshot):
    from battery_rul.fleet.passport import SuppliedBatteryMetadata, build_passport

    supplied = SuppliedBatteryMetadata(
        chemistry="LCO 18650", manufacturer="<operator-supplied>", nominal_capacity_ah=2.0
    )
    passport = build_passport(fleet_snapshot.batteries[0], m3_config, supplied)

    assert passport.manufacturer == "<operator-supplied>"
    sources = {s.field_group: s.source for s in passport.field_sources}
    assert sources["identity_and_manufacturing"] == "supplied"


def test_a_passport_is_json_serialisable(m3_config, fleet_snapshot):
    import json

    from battery_rul.fleet.passport import build_passport

    payload = build_passport(fleet_snapshot.batteries[0], m3_config).to_json_dict()
    assert json.loads(json.dumps(payload))["battery_id"]
