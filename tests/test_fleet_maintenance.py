"""The maintenance-priority engine, inspection windows and replacement planning.

Each rule is exercised on its own. That is the point of keeping the engine
deterministic and free of model calls: "why is this cell P1?" has to be
answerable by reading one rule and one threshold, and a test has to be able to
fire exactly that rule.
"""

from __future__ import annotations

import pandas as pd
import pytest

from battery_rul.config import ExperimentConfig, load_config
from battery_rul.fleet.analytics import battery_trends, cycles_per_day, trend_per_10_cycles
from battery_rul.fleet.domain import (
    FleetBatteryRecord,
    MaintenancePriority,
    ProcessingStatus,
    ReplacementHorizon,
)
from battery_rul.fleet.maintenance import (
    FLEET_ACTIONS,
    MaintenancePriorityEngine,
    inspection_recommendation,
)
from battery_rul.fleet.replacement import (
    ReplacementPlanner,
    summarise_replacements,
    workload_forecast,
)


@pytest.fixture
def policy_cfg() -> ExperimentConfig:
    return load_config()


def _record(**overrides) -> FleetBatteryRecord:
    point = overrides.pop("rul", 60.0)
    lower = overrides.pop("rul_lower", None)
    lower = (point - 20.0) if lower is None and point is not None else lower
    base = {
        "battery_id": overrides.pop("battery_id", "B1"),
        "status": ProcessingStatus.SUCCESS,
        "latest_cycle": 100,
        "n_cycles": 100,
        "measured_soh": 0.90,
        "health_class": "healthy",
        "predicted_rul": point,
        "rul_lower_bound": lower,
        "rul_upper_bound": None if point is None else point + 30.0,
        "interval_width": None if point is None else 40.0,
        "failure_risk": 0.05,
        "fade_trend_pct_per_10": 0.1,
        "data_quality_class": "GOOD",
        "data_quality_score": 0.95,
    }
    return FleetBatteryRecord(**{**base, **overrides})


# ---------------------------------------------------------------------------
# The ladder, rule by rule
# ---------------------------------------------------------------------------
def test_critical_soh_forces_p0(policy_cfg):
    engine = MaintenancePriorityEngine(cfg=policy_cfg)
    result = engine.evaluate(_record(measured_soh=0.65, health_class="critical"))
    assert result.priority is MaintenancePriority.P0_CRITICAL
    assert any("critical_soh" in rule for rule in result.triggered_rules)


def test_very_high_risk_with_a_very_low_lower_bound_forces_p0(policy_cfg):
    engine = MaintenancePriorityEngine(cfg=policy_cfg)
    result = engine.evaluate(_record(rul=6.0, rul_lower=2.0, failure_risk=0.95))
    assert result.priority is MaintenancePriority.P0_CRITICAL
    assert any("critical_risk_and_rul" in rule for rule in result.triggered_rules)


def test_high_risk_with_a_low_lower_bound_is_p1(policy_cfg):
    engine = MaintenancePriorityEngine(cfg=policy_cfg)
    result = engine.evaluate(_record(rul=25.0, rul_lower=12.0, failure_risk=0.7))
    assert result.priority is MaintenancePriority.P1_URGENT


def test_a_warning_health_class_is_p2(policy_cfg):
    engine = MaintenancePriorityEngine(cfg=policy_cfg)
    result = engine.evaluate(_record(measured_soh=0.75, health_class="warning", rul=200.0))
    assert result.priority is MaintenancePriority.P2_HIGH


def test_an_elevated_fade_trend_is_p2(policy_cfg):
    engine = MaintenancePriorityEngine(cfg=policy_cfg)
    result = engine.evaluate(_record(rul=300.0, fade_trend_pct_per_10=2.0))
    assert result.priority is MaintenancePriority.P2_HIGH
    assert any("high_trend" in rule for rule in result.triggered_rules)


def test_slight_degradation_with_a_meaningful_trend_is_p3(policy_cfg):
    engine = MaintenancePriorityEngine(cfg=policy_cfg)
    result = engine.evaluate(
        _record(
            measured_soh=0.85,
            health_class="slightly_degraded",
            rul=300.0,
            fade_trend_pct_per_10=0.7,
        )
    )
    assert result.priority is MaintenancePriority.P3_MEDIUM


def test_a_stable_healthy_cell_is_p4(policy_cfg):
    engine = MaintenancePriorityEngine(cfg=policy_cfg)
    result = engine.evaluate(
        _record(measured_soh=0.97, rul=300.0, interval_width=10.0, fade_trend_pct_per_10=0.05)
    )
    assert result.priority is MaintenancePriority.P4_LOW


def test_a_healthy_cell_with_a_wide_interval_is_watched_not_filed(policy_cfg):
    engine = MaintenancePriorityEngine(cfg=policy_cfg)
    result = engine.evaluate(
        _record(measured_soh=0.97, rul=300.0, interval_width=200.0, fade_trend_pct_per_10=0.05)
    )
    assert result.priority is MaintenancePriority.P5_MONITOR
    assert any("monitor_uncertain" in rule for rule in result.triggered_rules)


def test_insufficient_quality_short_circuits_the_ladder(policy_cfg):
    engine = MaintenancePriorityEngine(cfg=policy_cfg)
    result = engine.evaluate(_record(measured_soh=0.5, data_quality_class="INSUFFICIENT"))
    assert result.priority is MaintenancePriority.INSUFFICIENT_DATA
    assert result.priority_score == 0.0
    assert result.inspection.recommended_cycles is None


def test_a_failed_record_never_gets_a_priority(policy_cfg):
    engine = MaintenancePriorityEngine(cfg=policy_cfg)
    result = engine.evaluate(
        _record(status=ProcessingStatus.FAILED, errors=["boom"], rul=None, rul_lower=None)
    )
    assert result.priority is MaintenancePriority.INSUFFICIENT_DATA


def test_a_critical_data_warning_overrides_the_ladder(policy_cfg):
    engine = MaintenancePriorityEngine(cfg=policy_cfg)
    result = engine.evaluate(_record(rul=300.0), critical_data_warning=True)
    assert result.priority is MaintenancePriority.P0_CRITICAL
    assert any("critical_data_warning" in rule for rule in result.triggered_rules)


def test_an_experimental_risk_model_is_not_used_as_evidence(policy_cfg):
    """A model that lost to a cycle counter must not trigger a replacement."""
    engine = MaintenancePriorityEngine(cfg=policy_cfg)
    result = engine.evaluate(
        _record(rul=300.0, rul_lower=250.0, failure_risk=0.99, risk_is_experimental=True)
    )
    assert result.priority is not MaintenancePriority.P0_CRITICAL
    assert any("risk_withheld" in rule for rule in result.triggered_rules)


def test_rules_fire_on_the_lower_bound_not_the_point_estimate(policy_cfg):
    """A 45-cycle point estimate with a 12-cycle lower bound is not a 45-cycle
    situation."""
    engine = MaintenancePriorityEngine(cfg=policy_cfg)
    conservative = engine.evaluate(_record(rul=45.0, rul_lower=12.0, failure_risk=0.7))
    optimistic = engine.evaluate(_record(rul=45.0, rul_lower=44.0, failure_risk=0.7))
    assert conservative.priority.severity < optimistic.priority.severity


def test_every_result_carries_its_argument(policy_cfg):
    engine = MaintenancePriorityEngine(cfg=policy_cfg)
    result = engine.evaluate(_record())
    assert result.triggered_rules
    assert result.evidence
    assert result.score_breakdown
    assert result.recommended_action == FLEET_ACTIONS[result.priority][0]
    assert result.disclaimer


def test_the_engine_is_deterministic(policy_cfg):
    engine = MaintenancePriorityEngine(cfg=policy_cfg)
    record = _record(rul=30.0, failure_risk=0.5)
    first, second = engine.evaluate(record), engine.evaluate(record)
    assert first.priority is second.priority
    assert first.priority_score == second.priority_score


def test_thresholds_are_configurable(policy_cfg):
    policy_cfg.fleet.maintenance.critical_soh = 0.95
    engine = MaintenancePriorityEngine(cfg=policy_cfg)
    assert engine.evaluate(_record(measured_soh=0.94)).priority is MaintenancePriority.P0_CRITICAL


# ---------------------------------------------------------------------------
# Inspection windows
# ---------------------------------------------------------------------------
def test_p0_means_immediate_review(policy_cfg):
    window = inspection_recommendation(MaintenancePriority.P0_CRITICAL, policy_cfg)
    assert window.recommended_cycles == 0
    assert window.recommended_label == "immediate_engineering_review"


def test_a_calendar_estimate_needs_a_measured_duty_rate(policy_cfg):
    without = inspection_recommendation(MaintenancePriority.P2_HIGH, policy_cfg)
    assert without.estimated_days is None
    assert any("No calendar estimate" in a for a in without.assumptions)

    with_rate = inspection_recommendation(
        MaintenancePriority.P2_HIGH, policy_cfg, cycles_per_day=2.0
    )
    assert with_rate.estimated_days == pytest.approx(with_rate.recommended_cycles / 2.0, rel=1e-6)
    assert any("duty cycle" in a for a in with_rate.assumptions)


def test_the_window_never_exceeds_the_remaining_life(policy_cfg):
    window = inspection_recommendation(MaintenancePriority.P3_MEDIUM, policy_cfg, planning_rul=4.0)
    assert window.recommended_cycles == 4
    assert any("shortened" in a for a in window.assumptions)


def test_low_priority_implies_no_separate_window(policy_cfg):
    window = inspection_recommendation(MaintenancePriority.P4_LOW, policy_cfg)
    assert window.recommended_label == "next_scheduled_inspection"


def test_insufficient_data_has_no_window(policy_cfg):
    window = inspection_recommendation(MaintenancePriority.INSUFFICIENT_DATA, policy_cfg)
    assert window.recommended_cycles is None


# ---------------------------------------------------------------------------
# Replacement planning
# ---------------------------------------------------------------------------
def test_replacement_horizons_follow_the_planning_quantity(policy_cfg):
    planner = ReplacementPlanner(cfg=policy_cfg)
    near = planner.evaluate(_record(rul=25.0, rul_lower=10.0))
    medium = planner.evaluate(_record(rul=60.0, rul_lower=40.0))
    long_term = planner.evaluate(_record(rul=110.0, rul_lower=90.0))
    none = planner.evaluate(_record(rul=400.0, rul_lower=380.0, measured_soh=0.98))

    assert near.replacement_horizon is ReplacementHorizon.NEAR_TERM
    assert medium.replacement_horizon is ReplacementHorizon.MEDIUM_TERM
    assert long_term.replacement_horizon is ReplacementHorizon.LONG_TERM
    assert none.replacement_horizon is ReplacementHorizon.NOT_FLAGGED
    assert none.replacement_candidate is False


def test_an_unevaluated_cell_is_not_a_candidate(policy_cfg):
    planner = ReplacementPlanner(cfg=policy_cfg)
    candidate = planner.evaluate(
        _record(status=ProcessingStatus.FAILED, rul=None, rul_lower=None, errors=["x"])
    )
    assert candidate.replacement_candidate is False
    assert candidate.replacement_horizon is ReplacementHorizon.UNKNOWN
    assert candidate.confidence == "unknown"


def test_a_wide_interval_lowers_the_planning_confidence(policy_cfg):
    planner = ReplacementPlanner(cfg=policy_cfg)
    tight = planner.evaluate(_record(rul=30.0, rul_lower=28.0, interval_width=5.0))
    wide = planner.evaluate(_record(rul=30.0, rul_lower=5.0, interval_width=90.0))
    assert wide.confidence == "low"
    assert tight.confidence in ("high", "medium")


def test_replacement_carries_caveats_and_no_cost_claim(policy_cfg):
    planner = ReplacementPlanner(cfg=policy_cfg)
    candidate = planner.evaluate(_record(rul=15.0, rul_lower=5.0))
    assert candidate.caveats
    assert any("cost" in c for c in candidate.caveats)


def test_the_replacement_summary_brackets_the_count_by_uncertainty(policy_cfg):
    planner = ReplacementPlanner(cfg=policy_cfg)
    records = [
        _record(battery_id="B1", rul=25.0, rul_lower=8.0),
        _record(battery_id="B2", rul=80.0, rul_lower=45.0),
        _record(battery_id="B3", rul=300.0, rul_lower=280.0, measured_soh=0.98),
    ]
    candidates = [planner.evaluate(r) for r in records]
    for record, candidate in zip(records, candidates, strict=True):
        record.replacement = candidate

    summary = summarise_replacements(records, candidates, policy_cfg)
    assert summary.denominator == 3
    for horizon in ("near_term", "medium_term", "long_term"):
        assert summary.lower_counts_by_horizon[horizon] <= summary.upper_counts_by_horizon[horizon]


# ---------------------------------------------------------------------------
# Workload forecast
# ---------------------------------------------------------------------------
def test_the_workload_forecast_covers_every_cell_exactly_once(policy_cfg):
    records = [
        _record(battery_id="B1", rul=5.0, rul_lower=1.0, priority=MaintenancePriority.P0_CRITICAL),
        _record(battery_id="B2", rul=25.0, rul_lower=8.0, priority=MaintenancePriority.P2_HIGH),
        _record(battery_id="B3", rul=200.0, rul_lower=180.0, priority=MaintenancePriority.P4_LOW),
        _record(
            battery_id="B4",
            status=ProcessingStatus.FAILED,
            rul=None,
            rul_lower=None,
            errors=["x"],
        ),
    ]
    forecast = workload_forecast(records, policy_cfg)
    total = sum(bucket.battery_count for bucket in forecast.buckets)
    assert total == len(records)
    assert forecast.evaluated_count == 3
    assert forecast.excluded_count == 1
    assert forecast.caveats


def test_workload_percentages_are_of_the_evaluated_fleet(policy_cfg):
    records = [
        _record(battery_id="B1", rul=5.0, rul_lower=1.0, priority=MaintenancePriority.P0_CRITICAL),
        _record(
            battery_id="B2",
            status=ProcessingStatus.FAILED,
            rul=None,
            rul_lower=None,
            errors=["x"],
        ),
    ]
    forecast = workload_forecast(records, policy_cfg)
    immediate = next(b for b in forecast.buckets if b.label == "immediate")
    assert immediate.percent_of_evaluated == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Trend analytics
# ---------------------------------------------------------------------------
def test_a_missing_trend_is_none_not_zero():
    """A flat trend and an absent one are different facts."""
    frame = pd.DataFrame({"cycle_index": range(30), "capacity_ah": [2.0] * 30})
    assert trend_per_10_cycles(frame, "capacity_ah", relative=False) == pytest.approx(0.0)
    assert trend_per_10_cycles(frame, "not_a_column", relative=False) is None


def test_a_falling_capacity_produces_a_positive_fade_trend():
    frame = pd.DataFrame(
        {"cycle_index": range(30), "capacity_ah": [2.0 - 0.01 * k for k in range(30)]}
    )
    trends = battery_trends(frame)
    assert trends["fade_trend_pct_per_10"] > 0


def test_too_few_points_yields_no_trend():
    frame = pd.DataFrame({"cycle_index": [1, 2], "capacity_ah": [2.0, 1.9]})
    assert trend_per_10_cycles(frame, "capacity_ah", relative=False) is None


def test_a_duty_rate_needs_timestamps():
    frame = pd.DataFrame({"cycle_index": range(20), "capacity_ah": [2.0] * 20})
    assert cycles_per_day(frame) is None

    stamped = frame.assign(
        timestamp=pd.date_range("2026-01-01", periods=20, freq="12h").astype(str)
    )
    assert cycles_per_day(stamped) == pytest.approx(2.0, rel=0.1)
