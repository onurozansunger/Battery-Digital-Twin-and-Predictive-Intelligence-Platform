"""Drift detection: the statistics, and the edge cases that break naive ones.

The behaviours under test are mostly about *refusing to answer*: a constant
feature, a 12-row sample and a feature that vanished from the batch all have to
be reported without a drift verdict, because in each case the statistic would be
an artifact rather than a measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_rul.config import ExperimentConfig, load_config
from battery_rul.fleet.domain import MonitoringStatus
from battery_rul.monitoring.drift import (
    benjamini_hochberg,
    detect_feature_drift,
    jensen_shannon_divergence,
    population_stability_index,
)
from battery_rul.monitoring.prediction_drift import (
    class_frequency_distance,
    detect_prediction_drift,
    summarise_predictions,
)
from battery_rul.monitoring.reference import (
    build_reference_distribution,
    load_reference,
    reference_path,
    save_reference,
)


@pytest.fixture
def monitoring_cfg(tmp_path) -> ExperimentConfig:
    return load_config(
        overrides={
            "paths.root": str(tmp_path),
            "monitoring.drift.min_sample_size": 30,
            "monitoring.prediction_drift.min_sample_size": 5,
        }
    )


def _frame(n: int = 500, *, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "battery_id": [f"B{i % 5}" for i in range(n)],
            "capacity_ah": rng.normal(2.0 + shift, 0.1, n),
            "temperature_max_c": rng.normal(35.0 + 5 * shift, 2.0, n),
            "constant_feature": np.full(n, 1.0),
            "flag": rng.integers(0, 2, n).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# The statistics themselves
# ---------------------------------------------------------------------------
def test_psi_is_zero_for_identical_distributions():
    frequencies = [0.2, 0.3, 0.5]
    assert population_stability_index(frequencies, frequencies) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_shift():
    reference = [0.25, 0.25, 0.25, 0.25]
    small = population_stability_index(reference, [0.30, 0.25, 0.25, 0.20])
    large = population_stability_index(reference, [0.70, 0.15, 0.10, 0.05])
    assert 0 < small < large


def test_psi_is_finite_when_a_bin_is_empty():
    """An empty bin is a division artifact, not infinite drift."""
    value = population_stability_index([0.5, 0.5], [1.0, 0.0])
    assert np.isfinite(value)


def test_psi_rejects_mismatched_bin_counts():
    with pytest.raises(ValueError, match="matching bin counts"):
        population_stability_index([0.5, 0.5], [0.3, 0.3, 0.4])


def test_jensen_shannon_is_bounded_and_symmetric():
    p, q = [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]
    forward = jensen_shannon_divergence(p, q)
    backward = jensen_shannon_divergence(q, p)
    assert forward == pytest.approx(backward, rel=1e-9)
    assert 0.0 <= forward <= np.log(2) + 1e-9


def test_benjamini_hochberg_is_monotone_and_preserves_order():
    raw = [0.001, 0.01, 0.04, 0.2, None]
    adjusted = benjamini_hochberg(raw)
    assert adjusted[-1] is None, "a test that could not run stays unadjusted"
    values = list(adjusted[:-1])
    assert all(a >= r for a, r in zip(values, raw[:-1], strict=True))
    assert values == sorted(values)


def test_benjamini_hochberg_handles_an_empty_input():
    assert benjamini_hochberg([None, None]) == [None, None]


# ---------------------------------------------------------------------------
# The reference artifact
# ---------------------------------------------------------------------------
def test_the_reference_round_trips_as_json(monitoring_cfg):
    reference = build_reference_distribution(_frame(), monitoring_cfg, reference_id="unit_ref")
    path = save_reference(reference, monitoring_cfg)

    assert path.suffix == ".json", "never a pickle: a reference is read by a service"
    loaded = load_reference(monitoring_cfg, "unit_ref")
    assert loaded.feature_names == reference.feature_names
    assert loaded.fingerprint() == reference.fingerprint()


def test_a_reference_id_can_never_become_a_path(monitoring_cfg):
    with pytest.raises(ValueError, match="Invalid reference id"):
        reference_path(monitoring_cfg, "../../etc/passwd")


def test_building_a_reference_from_an_empty_frame_is_refused(monitoring_cfg):
    with pytest.raises(ValueError, match="empty"):
        build_reference_distribution(pd.DataFrame(), monitoring_cfg)


def test_a_constant_feature_is_recorded_as_constant(monitoring_cfg):
    reference = build_reference_distribution(_frame(), monitoring_cfg)
    assert reference.feature_stats["constant_feature"]["constant"] is True


# ---------------------------------------------------------------------------
# Feature drift
# ---------------------------------------------------------------------------
def test_no_drift_between_samples_of_the_same_distribution(monitoring_cfg):
    reference = build_reference_distribution(_frame(seed=1), monitoring_cfg)
    report = detect_feature_drift(_frame(seed=2), reference, monitoring_cfg)

    assert report.status in (MonitoringStatus.OK, MonitoringStatus.WARNING)
    drifted = {r.feature_name for r in report.results if r.drift_detected}
    assert "capacity_ah" not in drifted


def test_a_shifted_distribution_is_detected(monitoring_cfg):
    reference = build_reference_distribution(_frame(seed=1), monitoring_cfg)
    report = detect_feature_drift(_frame(shift=1.0, seed=2), reference, monitoring_cfg)

    assert report.status is MonitoringStatus.CRITICAL
    assert "capacity_ah" in report.drifted_features
    assert report.n_features_drifted >= 1


def test_a_constant_feature_never_raises_and_is_never_scored(monitoring_cfg):
    reference = build_reference_distribution(_frame(seed=1), monitoring_cfg)
    report = detect_feature_drift(_frame(seed=2), reference, monitoring_cfg)

    results = [r for r in report.results if r.feature_name == "constant_feature"]
    assert results, "the feature is reported"
    assert all(not r.reliable for r in results)
    assert all(not r.drift_detected for r in results)


def test_a_small_sample_is_reported_unscored(monitoring_cfg):
    reference = build_reference_distribution(_frame(seed=1), monitoring_cfg)
    report = detect_feature_drift(_frame(n=10, seed=2), reference, monitoring_cfg)

    capacity = [r for r in report.results if r.feature_name == "capacity_ah"]
    assert all(not r.reliable for r in capacity)
    assert all("minimum" in " ".join(r.warnings) for r in capacity)
    assert report.status is MonitoringStatus.UNKNOWN


def test_a_feature_missing_from_the_batch_is_a_finding(monitoring_cfg):
    reference = build_reference_distribution(_frame(seed=1), monitoring_cfg)
    current = _frame(seed=2).drop(columns=["temperature_max_c"])
    report = detect_feature_drift(current, reference, monitoring_cfg)

    missing = [r for r in report.results if r.feature_name == "temperature_max_c"]
    assert missing and missing[0].drift_metric == "unavailable"
    assert report.n_features_skipped >= 1
    assert any("absent" in w for w in report.warnings)


def test_a_new_feature_is_reported_not_tested(monitoring_cfg):
    reference = build_reference_distribution(_frame(seed=1), monitoring_cfg)
    current = _frame(seed=2).assign(brand_new=1.5)
    report = detect_feature_drift(current, reference, monitoring_cfg)
    assert any("absent from the" in w for w in report.warnings)


def test_the_report_states_its_reference_and_its_methods(monitoring_cfg):
    reference = build_reference_distribution(_frame(seed=1), monitoring_cfg, reference_id="ref_a")
    report = detect_feature_drift(_frame(seed=2), reference, monitoring_cfg)

    assert report.reference_id == "ref_a"
    assert report.reference_fingerprint
    assert report.method_notes
    assert report.multiple_comparison == monitoring_cfg.monitoring.drift.multiple_comparison


def test_thresholds_are_configurable(monitoring_cfg):
    reference = build_reference_distribution(_frame(seed=1), monitoring_cfg)
    monitoring_cfg.monitoring.drift.psi_thresholds = (0.0001, 0.0002)
    report = detect_feature_drift(_frame(seed=2), reference, monitoring_cfg)
    assert report.n_features_drifted >= 1, "a tighter threshold flags more"


# ---------------------------------------------------------------------------
# Prediction drift
# ---------------------------------------------------------------------------
def test_class_frequency_distance_is_a_proportion():
    assert class_frequency_distance({"a": 1.0}, {"a": 1.0}) == pytest.approx(0.0)
    assert class_frequency_distance({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)
    assert class_frequency_distance({"a": 0.5, "b": 0.5}, {"a": 0.7, "b": 0.3}) == pytest.approx(
        0.2
    )


def test_prediction_drift_needs_a_minimum_sample(monitoring_cfg, fleet_snapshot):
    monitoring_cfg.monitoring.prediction_drift.min_sample_size = 1000
    summary = summarise_predictions(fleet_snapshot.batteries, monitoring_cfg)
    report = detect_prediction_drift(summary, summary, monitoring_cfg)
    assert report.status is MonitoringStatus.UNKNOWN
    assert any("below the configured minimum" in w for w in report.warnings)


def test_identical_batches_show_no_prediction_drift(monitoring_cfg, fleet_snapshot):
    summary = summarise_predictions(fleet_snapshot.batteries, monitoring_cfg)
    report = detect_prediction_drift(summary, summary, monitoring_cfg)
    assert report.n_drifted == 0
    assert report.status in (MonitoringStatus.OK, MonitoringStatus.UNKNOWN)


def test_a_shifted_output_distribution_is_detected(monitoring_cfg, fleet_snapshot):
    current = summarise_predictions(fleet_snapshot.batteries, monitoring_cfg)
    reference = dict(current)
    reference["predicted_rul"] = {
        **current["predicted_rul"],
        "mean": float(current["predicted_rul"]["mean"]) + 500.0,
    }
    report = detect_prediction_drift(current, reference, monitoring_cfg)
    assert report.n_drifted >= 1
    assert any(r.output_name == "predicted_rul" for r in report.results if r.drift_detected)


def test_the_prediction_drift_report_states_what_drift_does_not_mean(
    monitoring_cfg, fleet_snapshot
):
    summary = summarise_predictions(fleet_snapshot.batteries, monitoring_cfg)
    report = detect_prediction_drift(summary, summary, monitoring_cfg)
    assert "does not prove model degradation" in report.interpretation
