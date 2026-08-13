"""Digital-twin domain model, data quality, recommendations and bundles."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from battery_rul.config import ExperimentConfig
from battery_rul.digital_twin.domain import (
    SNAPSHOT_SCHEMA_VERSION,
    BatteryHealthState,
    BatteryIdentity,
    BatteryMeasurements,
    BatteryPrediction,
    BatteryRecommendation,
    BatteryRiskAssessment,
    BatteryTwinSnapshot,
    BatteryUncertainty,
    DataQualityAssessment,
    Provenance,
    TwinMetadata,
)
from battery_rul.digital_twin.quality import assess_data_quality
from battery_rul.models.bundle import (
    ArtifactCompatibilityError,
    BundleMetadata,
    load_bundle,
    save_bundle,
)
from battery_rul.recommendations.engine import (
    ActionCode,
    RecommendationEngine,
    RecommendationInputs,
)


# ===========================================================================
# Domain model
# ===========================================================================
def _snapshot(**overrides) -> BatteryTwinSnapshot:
    base = {
        "battery_id": "B0007",
        "identity": BatteryIdentity(battery_id="B0007"),
        "measurement_summary": BatteryMeasurements(
            latest_cycle=121, n_cycles_supplied=121, first_cycle=1
        ),
        "health": BatteryHealthState(soh=0.846, health_class="warning"),
        "prediction": BatteryPrediction(rul_cycles=38.0),
        "failure_risk": BatteryRiskAssessment(horizon_cycles=30, probability=0.71),
        "recommendation": BatteryRecommendation(
            action_code="SCHEDULE_INSPECTION",
            priority="medium",
            title="Schedule an inspection",
            explanation="…",
            disclaimer="…",
        ),
        "data_quality": DataQualityAssessment(
            quality_score=0.9, quality_class="GOOD", n_cycles=121, missing_feature_fraction=0.0
        ),
        "metadata": TwinMetadata(),
    }
    return BatteryTwinSnapshot(**{**base, **overrides})


def test_snapshot_is_json_serialisable():
    payload = _snapshot().to_json_dict()
    text = json.dumps(payload)
    assert json.loads(text)["battery_id"] == "B0007"


def test_snapshot_schema_version_is_reported():
    assert _snapshot().metadata.snapshot_schema_version == SNAPSHOT_SCHEMA_VERSION


def test_soh_percent_is_derived_from_the_fraction():
    """One internal representation; the percentage is a rendering of it."""
    state = BatteryHealthState(soh=0.846)
    assert state.soh_percent == pytest.approx(84.6)


def test_every_value_carries_a_provenance_tag():
    snapshot = _snapshot()
    assert snapshot.measurement_summary.provenance is Provenance.OBSERVED
    assert snapshot.prediction.provenance is Provenance.PREDICTED
    assert snapshot.recommendation.provenance is Provenance.RULE_BASED
    assert snapshot.data_quality.provenance is Provenance.DERIVED


def test_uncertainty_must_bracket_the_point_estimate():
    with pytest.raises(ValueError, match="bracket"):
        BatteryUncertainty(
            point_estimate=38.0,
            lower_bound=40.0,
            upper_bound=48.0,
            interval_coverage_target=0.9,
            uncertainty_method="split_conformal",
            calibration_sample_size=100,
        )


def test_risk_probability_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        BatteryRiskAssessment(horizon_cycles=30, probability=1.4)


def test_negative_rul_is_rejected():
    with pytest.raises(ValueError):
        BatteryPrediction(rul_cycles=-5.0)


def test_blank_battery_id_is_rejected():
    with pytest.raises(ValueError):
        _snapshot(battery_id="   ")


def test_unknown_snapshot_fields_are_rejected():
    with pytest.raises(ValueError):
        _snapshot(unexpected_field=1)


def test_risk_assessment_states_the_label_is_derived():
    assessment = BatteryRiskAssessment(horizon_cycles=30)
    assert "not an observed safety failure" in assessment.label_definition.lower()


# ===========================================================================
# Data quality
# ===========================================================================
def _history(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "battery_id": "A",
            "cycle_index": np.arange(1, n + 1),
            "capacity_ah": np.linspace(1.9, 1.3, n),
            "voltage_mean_v": np.full(n, 3.6),
            "temperature_max_c": np.linspace(30.0, 40.0, n),
        }
    )


def test_good_history_scores_well(cfg: ExperimentConfig):
    assessment = assess_data_quality(_history(80), cfg, min_history_cycles=10)
    assert assessment.quality_class == "GOOD"
    assert assessment.quality_score >= cfg.quality.good_min_score


def test_short_history_is_insufficient(cfg: ExperimentConfig):
    assessment = assess_data_quality(_history(4), cfg, min_history_cycles=25)
    assert assessment.quality_class == "INSUFFICIENT"
    assert any("Insufficient history" in w for w in assessment.warnings)


def test_duplicate_cycles_are_penalised(cfg: ExperimentConfig):
    frame = pd.concat([_history(60), _history(60).tail(3)], ignore_index=True)
    assessment = assess_data_quality(frame, cfg, min_history_cycles=10)
    assert assessment.checks["duplicate_cycles"]["passed"] is False
    assert assessment.quality_score < 1.0


def test_large_cycle_gap_is_flagged(cfg: ExperimentConfig):
    frame = _history(60)
    frame.loc[frame.index[30:], "cycle_index"] += 200
    assessment = assess_data_quality(frame, cfg, min_history_cycles=10)
    assert assessment.checks["cycle_gaps"]["passed"] is False


def test_implausible_values_are_flagged(cfg: ExperimentConfig):
    frame = _history(60)
    frame.loc[frame.index[5], "voltage_mean_v"] = 900.0
    assessment = assess_data_quality(frame, cfg, min_history_cycles=10)
    assert assessment.checks["value_plausibility"]["passed"] is False


def test_missing_required_features_drive_the_class_down(cfg: ExperimentConfig):
    frame = _history(60)
    required = [f"feature_{i}" for i in range(10)]
    assessment = assess_data_quality(frame, cfg, required_features=required, min_history_cycles=10)
    assert assessment.quality_class == "INSUFFICIENT"
    assert len(assessment.missing_features) == 10


def test_out_of_distribution_features_are_flagged(cfg: ExperimentConfig):
    frame = _history(60)
    features = pd.DataFrame({"x": np.full(60, 1e6)})
    assessment = assess_data_quality(
        frame,
        cfg,
        required_features=["x"],
        feature_frame=features,
        reference_ranges={"x": (0.0, 1.0)},
        min_history_cycles=10,
    )
    assert "x" in assessment.out_of_distribution_flags


def test_quality_score_is_bounded(cfg: ExperimentConfig):
    frame = _history(60)
    frame.loc[frame.index[5], "voltage_mean_v"] = 900.0
    frame = pd.concat([frame, frame.tail(5)], ignore_index=True)
    assessment = assess_data_quality(frame, cfg, min_history_cycles=10)
    assert 0.0 <= assessment.quality_score <= 1.0


# ===========================================================================
# Recommendation engine
# ===========================================================================
def test_insufficient_data_never_yields_a_maintenance_action(cfg: ExperimentConfig):
    """The whole point of the class: no confident action from poor input."""
    engine = RecommendationEngine(cfg=cfg)
    result = engine.recommend(
        RecommendationInputs(rul_point=3.0, risk_probability=0.99, quality_class="INSUFFICIENT")
    )
    assert result.action_code == ActionCode.INSUFFICIENT_DATA.value


def test_unscoreable_cell_yields_insufficient_data(cfg: ExperimentConfig):
    engine = RecommendationEngine(cfg=cfg)
    result = engine.recommend(RecommendationInputs(is_scoreable=False))
    assert result.action_code == ActionCode.INSUFFICIENT_DATA.value


def test_healthy_cell_gets_normal_operation(cfg: ExperimentConfig):
    engine = RecommendationEngine(cfg=cfg)
    result = engine.recommend(
        RecommendationInputs(
            rul_point=200.0,
            rul_lower_bound=150.0,
            soh=0.97,
            health_class="healthy",
            risk_probability=0.01,
            risk_class="low",
        )
    )
    assert result.action_code == ActionCode.NORMAL_OPERATION.value
    assert result.priority == "none"


def test_very_low_remaining_life_escalates_to_urgent(cfg: ExperimentConfig):
    engine = RecommendationEngine(cfg=cfg)
    result = engine.recommend(
        RecommendationInputs(rul_point=6.0, rul_lower_bound=2.0, health_class="critical")
    )
    assert result.action_code == ActionCode.IMMEDIATE_ENGINEERING_REVIEW.value
    assert result.priority == "urgent"


def test_rules_use_the_lower_bound_not_the_point_estimate(cfg: ExperimentConfig):
    """A wide interval must not be planned against its middle."""
    engine = RecommendationEngine(cfg=cfg)
    optimistic = engine.recommend(RecommendationInputs(rul_point=45.0, rul_lower_bound=45.0))
    uncertain = engine.recommend(RecommendationInputs(rul_point=45.0, rul_lower_bound=12.0))
    assert optimistic.action_code == ActionCode.NORMAL_OPERATION.value
    assert uncertain.action_code == ActionCode.PLAN_REPLACEMENT.value


def test_high_risk_alone_triggers_inspection(cfg: ExperimentConfig):
    engine = RecommendationEngine(cfg=cfg)
    result = engine.recommend(
        RecommendationInputs(rul_point=500.0, rul_lower_bound=400.0, risk_probability=0.35)
    )
    assert result.action_code == ActionCode.SCHEDULE_INSPECTION.value


def test_temperature_trend_produces_a_thermal_advisory(cfg: ExperimentConfig):
    engine = RecommendationEngine(cfg=cfg)
    result = engine.recommend(
        RecommendationInputs(
            rul_point=500.0,
            rul_lower_bound=400.0,
            health_class="healthy",
            temperature_trend_c_per_10=2.0,
        )
    )
    assert result.action_code == ActionCode.REDUCE_HIGH_TEMPERATURE_OPERATION.value


def test_resistance_trend_produces_a_charging_advisory(cfg: ExperimentConfig):
    engine = RecommendationEngine(cfg=cfg)
    result = engine.recommend(
        RecommendationInputs(
            rul_point=500.0,
            rul_lower_bound=400.0,
            health_class="healthy",
            resistance_trend_pct_per_10=8.0,
        )
    )
    assert result.action_code == ActionCode.REDUCE_AGGRESSIVE_CHARGING.value


def test_thresholds_are_configurable(cfg: ExperimentConfig):
    cfg.recommendations.inspection_rul_cycles = 300
    engine = RecommendationEngine(cfg=cfg)
    result = engine.recommend(RecommendationInputs(rul_point=250.0, rul_lower_bound=250.0))
    assert result.action_code == ActionCode.SCHEDULE_INSPECTION.value


def test_recommendation_is_deterministic(cfg: ExperimentConfig):
    engine = RecommendationEngine(cfg=cfg)
    inputs = RecommendationInputs(rul_point=30.0, rul_lower_bound=20.0, risk_probability=0.4)
    first = engine.recommend(inputs)
    second = engine.recommend(inputs)
    assert first.model_dump() == second.model_dump()


def test_every_recommendation_carries_the_disclaimer(cfg: ExperimentConfig):
    engine = RecommendationEngine(cfg=cfg)
    for inputs in (
        RecommendationInputs(quality_class="INSUFFICIENT"),
        RecommendationInputs(rul_point=1000.0, rul_lower_bound=900.0),
        RecommendationInputs(rul_point=2.0, rul_lower_bound=1.0),
    ):
        assert engine.recommend(inputs).disclaimer == cfg.recommendations.disclaimer


def test_evidence_is_populated(cfg: ExperimentConfig):
    engine = RecommendationEngine(cfg=cfg)
    result = engine.recommend(
        RecommendationInputs(rul_point=30.0, rul_lower_bound=20.0, soh=0.78, risk_probability=0.4)
    )
    assert any("remaining useful life" in item for item in result.evidence)
    assert any("state of health" in item for item in result.evidence)


# ===========================================================================
# Model bundles
# ===========================================================================
@pytest.fixture
def fitted_bundle(labelled_cycles: pd.DataFrame, cfg: ExperimentConfig, tmp_path):
    from battery_rul.features.engineering import build_features, feature_columns
    from battery_rul.features.pipeline import FeaturePipeline
    from battery_rul.models.base import TrainingData, build_model

    frame, _ = build_features(labelled_cycles, cfg.features)
    features = feature_columns(frame)
    y = frame[cfg.target.name].to_numpy(dtype=float)
    pipeline = FeaturePipeline(cfg=cfg.features).fit(frame[features], y)
    model = build_model("ridge", cfg).fit(
        TrainingData(
            X=pipeline.transform(frame[features]),
            y=y,
            frame=frame,
            feature_names=pipeline.feature_names,
        )
    )
    path = save_bundle(
        tmp_path / "rul",
        model=model,
        preprocessing=pipeline,
        cfg=cfg,
        task="rul_regression",
        model_name="ridge",
    )
    return path, cfg


def test_bundle_round_trips(fitted_bundle):
    path, cfg = fitted_bundle
    bundle = load_bundle(path, cfg)
    assert bundle.metadata.model_name == "ridge"
    assert bundle.metadata.feature_names == bundle.preprocessing.feature_names
    assert bundle.metadata.data_fingerprint == cfg.data_fingerprint()


def test_bundle_records_full_metadata(fitted_bundle):
    path, cfg = fitted_bundle
    bundle = load_bundle(path, cfg)
    metadata = bundle.metadata
    assert metadata.schema_version
    assert metadata.preprocessing_fingerprint
    assert metadata.target_definition["eol_persistence"] == cfg.target.eol_persistence
    assert metadata.dependencies


def test_missing_bundle_directory_fails_clearly(cfg: ExperimentConfig, tmp_path):
    with pytest.raises(FileNotFoundError, match="Model bundle not found"):
        load_bundle(tmp_path / "nope", cfg)


def test_incomplete_bundle_fails_clearly(fitted_bundle):
    path, cfg = fitted_bundle
    (path / "model.pkl").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        load_bundle(path, cfg)


def test_incompatible_configuration_is_refused(fitted_bundle):
    """The defect this exists to prevent: serving under a different EOL definition."""
    path, cfg = fitted_bundle
    cfg.data.eol_threshold = 0.60
    with pytest.raises(ArtifactCompatibilityError, match="different data-affecting"):
        load_bundle(path, cfg)


def test_incompatibility_message_names_the_field(fitted_bundle):
    path, cfg = fitted_bundle
    cfg.target.eol_persistence = 7
    with pytest.raises(ArtifactCompatibilityError, match="target.eol_persistence"):
        load_bundle(path, cfg)


def test_strict_compatibility_can_be_disabled(fitted_bundle):
    path, cfg = fitted_bundle
    cfg.data.eol_threshold = 0.60
    cfg.artifacts.strict_compatibility = False
    assert load_bundle(path, cfg) is not None


def test_a_bundle_records_the_interpreter_that_pickled_it(fitted_bundle):
    """Pickles are not portable across Python minor versions, so the version is
    part of the bundle's contract."""
    import sys

    from battery_rul.utils.io import load_json

    path, _ = fitted_bundle
    recorded = load_json(path / "metadata.json")["python_version"]
    assert recorded.startswith(f"{sys.version_info.major}.{sys.version_info.minor}")


def test_a_bundle_from_another_python_minor_version_is_refused(fitted_bundle):
    """The alternative is `ModuleNotFoundError: No module named 'pathlib._local'`,
    which names nothing useful and sends the reader hunting for a dependency."""
    from battery_rul.utils.io import load_json, save_json

    path, cfg = fitted_bundle
    payload = load_json(path / "metadata.json")
    payload["python_version"] = "3.7.9"
    save_json(payload, path / "metadata.json")

    with pytest.raises(ArtifactCompatibilityError, match="pickled by Python 3.7.9"):
        load_bundle(path, cfg)


def test_a_bundle_without_a_recorded_interpreter_still_loads(fitted_bundle):
    """Bundles built before the field existed are not retroactively rejected."""
    from battery_rul.utils.io import load_json, save_json

    path, cfg = fitted_bundle
    payload = load_json(path / "metadata.json")
    payload["python_version"] = ""
    save_json(payload, path / "metadata.json")
    assert load_bundle(path, cfg) is not None


def test_unsupported_schema_version_is_refused(fitted_bundle):
    path, cfg = fitted_bundle
    from battery_rul.utils.io import load_json, save_json

    payload = load_json(path / "metadata.json")
    payload["schema_version"] = "99.0"
    save_json(payload, path / "metadata.json")
    with pytest.raises(ArtifactCompatibilityError, match="schema version"):
        load_bundle(path, cfg)


def test_missing_required_metadata_is_refused(fitted_bundle):
    path, cfg = fitted_bundle
    from battery_rul.utils.io import load_json, save_json

    payload = load_json(path / "metadata.json")
    payload["preprocessing_fingerprint"] = ""
    save_json(payload, path / "metadata.json")
    with pytest.raises(ArtifactCompatibilityError, match="missing required field"):
        load_bundle(path, cfg)


def test_missing_calibrator_is_refused_when_required(fitted_bundle):
    path, cfg = fitted_bundle
    with pytest.raises(ArtifactCompatibilityError, match="no calibration artifact"):
        load_bundle(path, cfg, require_calibrator=True)


def test_feature_schema_mismatch_is_detected(fitted_bundle):
    """Artifacts written apart must not be served together."""
    path, cfg = fitted_bundle
    from battery_rul.utils.io import load_json, save_json

    payload = load_json(path / "metadata.json")
    payload["feature_names"] = payload["feature_names"][:-1]
    save_json(payload, path / "metadata.json")
    with pytest.raises(ArtifactCompatibilityError, match="were not written together"):
        load_bundle(path, cfg)


def test_metadata_ignores_unknown_keys_with_a_warning():
    metadata = BundleMetadata.from_dict(
        {
            "model_name": "x",
            "model_version": "1",
            "task": "t",
            "unknown_future_field": 1,
        }
    )
    assert metadata.model_name == "x"
