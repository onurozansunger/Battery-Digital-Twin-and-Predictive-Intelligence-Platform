"""The composite priority score and the ranking methods.

Two things are being protected here. First, that the score's *properties* hold —
bounded range, missing components excluded rather than treated as zero, a
breakdown that reconstructs the number. Second, that ranking is total and
deterministic: no cell appears twice, ties break the same way every run.
"""

from __future__ import annotations

import pytest

from battery_rul.config import FleetRankingConfig, load_config
from battery_rul.fleet.domain import (
    FleetBatteryRecord,
    MaintenancePriority,
    ProcessingStatus,
)
from battery_rul.fleet.ranking import (
    RANKING_KEYS,
    compute_priority_score,
    rank_batteries,
)


def _record(battery_id: str = "B1", **overrides) -> FleetBatteryRecord:
    """A valid fleet record, with the interval kept consistent by construction.

    ``FleetBatteryRecord`` rejects an interval that does not bracket the point
    estimate, so a helper that let a test set ``predicted_rul`` alone would fail
    validation rather than testing anything. Bounds follow the point estimate
    unless the test names them explicitly.
    """
    point = overrides.get("predicted_rul", 50.0)
    base = {
        "battery_id": battery_id,
        "status": ProcessingStatus.SUCCESS,
        "latest_cycle": 100,
        "n_cycles": 100,
        "measured_soh": 0.85,
        "health_class": "slightly_degraded",
        "predicted_rul": point,
        "rul_lower_bound": None if point is None else max(point - 20.0, 0.0),
        "rul_upper_bound": None if point is None else point + 30.0,
        "interval_width": None if point is None else 50.0,
        "failure_risk": 0.3,
        "fade_trend_pct_per_10": 0.5,
        "data_quality_class": "GOOD",
        "data_quality_score": 0.95,
    }
    return FleetBatteryRecord(**{**base, **overrides})


@pytest.fixture
def ranking_cfg() -> FleetRankingConfig:
    return load_config().fleet.ranking


# ---------------------------------------------------------------------------
# Score properties
# ---------------------------------------------------------------------------
def test_the_score_stays_inside_the_configured_range(ranking_cfg):
    for soh, rul, risk in ((1.0, 500.0, 0.0), (0.5, 0.0, 1.0), (0.85, 50.0, 0.4)):
        score = compute_priority_score(
            _record(measured_soh=soh, predicted_rul=rul, rul_lower_bound=rul, failure_risk=risk),
            ranking_cfg,
        )
        assert 0.0 <= score.score <= ranking_cfg.score_scale


def test_a_worse_cell_scores_higher(ranking_cfg):
    healthy = compute_priority_score(
        _record(measured_soh=0.98, predicted_rul=200.0, rul_lower_bound=180.0, failure_risk=0.02),
        ranking_cfg,
    )
    degraded = compute_priority_score(
        _record(measured_soh=0.72, predicted_rul=8.0, rul_lower_bound=2.0, failure_risk=0.9),
        ranking_cfg,
    )
    assert degraded.score > healthy.score


def test_the_breakdown_reconstructs_the_score(ranking_cfg):
    scored = compute_priority_score(_record(), ranking_cfg)
    available = [c for c in scored.components if c.available]
    total = sum(c.contribution for c in available)
    weight = sum(c.weight for c in available)
    assert scored.score == pytest.approx(ranking_cfg.score_scale * total / weight, rel=1e-4)


def test_every_component_states_its_transformation(ranking_cfg):
    scored = compute_priority_score(_record(), ranking_cfg)
    assert all(c.transformation for c in scored.components)
    assert {c.name for c in scored.components} == set(ranking_cfg.weights())


def test_a_missing_component_is_excluded_rather_than_scored_as_zero(ranking_cfg):
    """A cell with no risk probability is not a cell with zero risk."""
    with_risk = compute_priority_score(_record(failure_risk=0.5), ranking_cfg)
    without = compute_priority_score(_record(failure_risk=None), ranking_cfg)

    missing = next(c for c in without.components if c.name == "risk")
    assert missing.available is False
    assert missing.contribution == 0.0
    assert without.available_weight < with_risk.available_weight
    # The remaining components are renormalised, so the score is not deflated
    # towards zero purely by the absence of evidence.
    assert without.score > 0.0


def test_an_experimental_risk_model_is_withheld_from_the_score(ranking_cfg):
    scored = compute_priority_score(
        _record(failure_risk=0.99, risk_is_experimental=True), ranking_cfg
    )
    component = next(c for c in scored.components if c.name == "risk")
    assert component.available is False
    assert "acceptance gate" in component.transformation


def test_a_critical_override_lifts_the_score_above_the_floor(ranking_cfg):
    scored = compute_priority_score(
        _record(measured_soh=0.99, predicted_rul=500.0, rul_lower_bound=480.0, failure_risk=0.0),
        ranking_cfg,
        critical_override=True,
    )
    assert scored.score >= ranking_cfg.critical_override_score
    assert scored.critical_override_applied is True


def test_a_record_with_nothing_scoreable_gets_zero(ranking_cfg):
    empty = _record(
        measured_soh=None,
        predicted_rul=None,
        rul_lower_bound=None,
        rul_upper_bound=None,
        interval_width=None,
        failure_risk=None,
        fade_trend_pct_per_10=None,
        data_quality_score=None,
    )
    scored = compute_priority_score(empty, ranking_cfg)
    assert scored.score == 0.0
    assert scored.n_available == 0


def test_weights_are_configurable(ranking_cfg):
    """The policy is configuration, not code."""
    risk_only = ranking_cfg.model_copy(
        update={
            "risk_weight": 1.0,
            "rul_weight": 0.0,
            "rul_lower_weight": 0.0,
            "soh_weight": 0.0,
            "trend_weight": 0.0,
            "uncertainty_weight": 0.0,
            "quality_weight": 0.0,
        }
    )
    scored = compute_priority_score(_record(failure_risk=0.75), risk_only)
    assert scored.score == pytest.approx(75.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def _fleet() -> list[FleetBatteryRecord]:
    return [
        _record(
            "B1", predicted_rul=10.0, priority=MaintenancePriority.P1_URGENT, priority_score=80
        ),
        _record("B2", predicted_rul=90.0, priority=MaintenancePriority.P4_LOW, priority_score=20),
        _record("B3", predicted_rul=50.0, priority=MaintenancePriority.P2_HIGH, priority_score=55),
        _record(
            "B4",
            status=ProcessingStatus.FAILED,
            predicted_rul=None,
            rul_lower_bound=None,
            rul_upper_bound=None,
            interval_width=None,
            measured_soh=None,
            priority=MaintenancePriority.INSUFFICIENT_DATA,
        ),
    ]


@pytest.mark.parametrize("key", RANKING_KEYS)
def test_every_ranking_key_produces_a_total_order_without_duplicates(key):
    ordered = rank_batteries(_fleet(), by=key, include_unevaluated=True)
    identifiers = [r.battery_id for r in ordered]
    assert len(identifiers) == len(set(identifiers)), "no battery appears twice"
    assert set(identifiers) == {"B1", "B2", "B3", "B4"}


def test_unevaluated_cells_are_excluded_by_default():
    ordered = rank_batteries(_fleet(), by="priority")
    assert "B4" not in [r.battery_id for r in ordered]


def test_unevaluated_cells_sort_last_when_included():
    ordered = rank_batteries(_fleet(), by="priority", include_unevaluated=True)
    assert ordered[-1].battery_id == "B4"


def test_priority_ranking_is_by_severity_then_score():
    ordered = rank_batteries(_fleet(), by="priority")
    assert [r.battery_id for r in ordered] == ["B1", "B3", "B2"]


def test_lowest_rul_first():
    ordered = rank_batteries(_fleet(), by="rul")
    assert [r.battery_id for r in ordered] == ["B1", "B3", "B2"]


def test_missing_values_sort_last_not_first():
    records = [
        _record("A", predicted_rul=None),
        _record("B", predicted_rul=5.0),
    ]
    ordered = rank_batteries(records, by="rul")
    assert [r.battery_id for r in ordered] == ["B", "A"]


def test_ranking_is_deterministic_across_calls():
    records = _fleet()
    first = [r.battery_id for r in rank_batteries(records, by="priority_score")]
    second = [r.battery_id for r in rank_batteries(list(reversed(records)), by="priority_score")]
    assert first == second


def test_the_limit_truncates_from_the_top():
    ordered = rank_batteries(_fleet(), by="priority", limit=2)
    assert [r.battery_id for r in ordered] == ["B1", "B3"]


def test_an_unknown_ranking_key_is_refused():
    with pytest.raises(ValueError, match="Unknown ranking key"):
        rank_batteries(_fleet(), by="vibes")
