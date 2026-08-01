"""Conformal prediction intervals and probability calibration."""

from __future__ import annotations

import numpy as np
import pytest

from battery_rul.calibration.probability import (
    ProbabilityCalibrator,
    brier_score,
    expected_calibration_error,
    reliability_curve,
    risk_metrics,
    tune_threshold,
)
from battery_rul.config import CalibrationConfig, UncertaintyConfig
from battery_rul.uncertainty.conformal import (
    ConformalIntervalEstimator,
    PredictionInterval,
    conformal_quantile,
    coverage_report,
)


# ---------------------------------------------------------------------------
# Conformal intervals
# ---------------------------------------------------------------------------
@pytest.fixture
def uncertainty_cfg() -> UncertaintyConfig:
    return UncertaintyConfig(
        method="split_conformal",
        coverage=0.9,
        normalise_by_life_stage=False,
        min_calibration_rows=10,
    )


def test_conformal_quantile_applies_the_finite_sample_correction():
    """Without the (n+1) correction the interval under-covers at small n."""
    residuals = np.arange(1.0, 11.0)
    corrected = conformal_quantile(residuals, 0.9)
    naive = float(np.quantile(residuals, 0.9, method="higher"))
    assert corrected >= naive


def test_conformal_achieves_nominal_coverage_on_exchangeable_data(uncertainty_cfg):
    rng = np.random.default_rng(0)
    truth = rng.normal(100.0, 20.0, size=2000)
    noise = rng.normal(0.0, 5.0, size=2000)
    predicted = truth + noise

    estimator = ConformalIntervalEstimator(cfg=uncertainty_cfg)
    estimator.fit(truth[:1000], predicted[:1000])
    lower, upper = estimator.intervals(predicted[1000:])
    covered = (truth[1000:] >= lower) & (truth[1000:] <= upper)
    # 90 % nominal; the tolerance is Monte-Carlo noise on 1000 held-out rows.
    assert 0.86 <= covered.mean() <= 0.96


def test_interval_always_brackets_the_point_estimate(uncertainty_cfg):
    rng = np.random.default_rng(1)
    truth = rng.normal(50.0, 10.0, size=200)
    estimator = ConformalIntervalEstimator(cfg=uncertainty_cfg)
    estimator.fit(truth, truth + rng.normal(0, 3, size=200))
    for point in (0.0, 5.0, 50.0, 500.0):
        interval = estimator.interval(point)
        assert interval.lower_bound <= interval.point_estimate <= interval.upper_bound


def test_lower_bound_is_clipped_at_zero_for_rul(uncertainty_cfg):
    """Negative remaining life is not a physical statement."""
    rng = np.random.default_rng(2)
    truth = rng.normal(50.0, 10.0, size=200)
    estimator = ConformalIntervalEstimator(cfg=uncertainty_cfg)
    estimator.fit(truth, truth + rng.normal(0, 8, size=200))
    assert estimator.interval(2.0).lower_bound >= 0.0


def test_interval_ordering_is_enforced_by_the_type():
    with pytest.raises(ValueError, match="bracket the point estimate"):
        PredictionInterval(
            point_estimate=10.0,
            lower_bound=12.0,
            upper_bound=20.0,
            interval_coverage_target=0.9,
            uncertainty_method="split_conformal",
            calibration_sample_size=100,
        )


def test_stage_conditioning_produces_different_widths():
    """Heteroscedastic residuals must not get one global width.

    The stage variable is **measured SOH** — a healthy cell (high SOH) has the
    least predictable remaining life and must get the widest interval.
    """
    cfg = UncertaintyConfig(coverage=0.9, normalise_by_life_stage=True, min_calibration_rows=10)
    rng = np.random.default_rng(3)
    n = 900
    soh = np.concatenate([np.full(300, 0.95), np.full(300, 0.85), np.full(300, 0.72)])
    scale = np.concatenate([np.full(300, 20.0), np.full(300, 8.0), np.full(300, 2.0)])
    truth = rng.normal(100.0, 10.0, size=n)
    predicted = truth + rng.normal(0.0, scale)

    estimator = ConformalIntervalEstimator(cfg=cfg)
    estimator.fit(truth, predicted, life_fraction=soh)
    assert set(estimator.stage_quantiles) == {"early", "mid", "late"}
    assert estimator.stage_quantiles["early"] > estimator.stage_quantiles["late"]


def test_stage_variable_is_observable_at_serving_time():
    """The regression guard for the flaw this replaced.

    Conditioning on life fraction (cycle / eol_cycle) needs a label, so a served
    cell can never supply it and every interval would quietly use the global
    quantile. Bucketing must therefore work from a measured SOH value alone.
    """
    cfg = UncertaintyConfig(coverage=0.9, normalise_by_life_stage=True, min_calibration_rows=10)
    rng = np.random.default_rng(7)
    soh = rng.uniform(0.70, 1.0, size=400)
    truth = rng.normal(80.0, 15.0, size=400)
    estimator = ConformalIntervalEstimator(cfg=cfg)
    estimator.fit(truth, truth + rng.normal(0, 6, size=400), life_fraction=soh)

    healthy = estimator.interval(50.0, life_fraction=0.95)
    worn = estimator.interval(50.0, life_fraction=0.72)
    assert healthy.life_stage == "early"
    assert worn.life_stage == "late"
    assert "by_life_stage" in healthy.uncertainty_method


def test_unknown_stage_falls_back_to_the_global_quantile():
    cfg = UncertaintyConfig(coverage=0.9, normalise_by_life_stage=True, min_calibration_rows=10)
    rng = np.random.default_rng(8)
    soh = rng.uniform(0.70, 1.0, size=400)
    truth = rng.normal(80.0, 15.0, size=400)
    estimator = ConformalIntervalEstimator(cfg=cfg)
    estimator.fit(truth, truth + rng.normal(0, 6, size=400), life_fraction=soh)

    quantile, stage = estimator.quantile_for(None)
    assert stage is None
    assert quantile == pytest.approx(estimator.global_quantile)


def test_thin_conformal_calibration_set_is_refused(uncertainty_cfg):
    uncertainty_cfg.min_calibration_rows = 50
    with pytest.raises(ValueError, match="at least 50"):
        ConformalIntervalEstimator(cfg=uncertainty_cfg).fit(np.arange(10.0), np.arange(10.0) + 1)


def test_unfitted_estimator_refuses_to_produce_intervals(uncertainty_cfg):
    with pytest.raises(RuntimeError, match="not fitted"):
        ConformalIntervalEstimator(cfg=uncertainty_cfg).interval(10.0)


def test_coverage_report_breaks_down_by_group():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "battery_id": ["A"] * 10 + ["B"] * 10,
            "life_stage": ["early"] * 10 + ["late"] * 10,
            "y_true": [5.0] * 20,
            "lower_bound": [0.0] * 10 + [6.0] * 10,
            "upper_bound": [10.0] * 10 + [8.0] * 10,
        }
    )
    report = coverage_report(frame)
    assert report["empirical_coverage"] == pytest.approx(0.5)
    assert report["by_battery_id"]["A"]["empirical_coverage"] == pytest.approx(1.0)
    assert report["by_battery_id"]["B"]["empirical_coverage"] == pytest.approx(0.0)


def test_interval_is_labelled_a_prediction_interval(uncertainty_cfg):
    rng = np.random.default_rng(4)
    truth = rng.normal(50.0, 10.0, size=100)
    estimator = ConformalIntervalEstimator(cfg=uncertainty_cfg)
    estimator.fit(truth, truth + rng.normal(0, 3, size=100))
    payload = estimator.interval(50.0).to_dict()
    assert payload["interval_type"] == "prediction_interval"
    assert "confidence interval" in payload["note"]


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------
def _miscalibrated(n: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Labels plus deliberately over-confident scores."""
    rng = np.random.default_rng(seed)
    probability = rng.uniform(0.0, 1.0, size=n)
    labels = rng.binomial(1, probability).astype(float)
    # Push scores toward the extremes: the classic tree-ensemble failure.
    over_confident = np.clip(probability**0.4 * 1.15 - 0.05, 0.0, 1.0)
    return labels, over_confident


def test_isotonic_calibration_improves_brier_and_ece():
    labels, scores = _miscalibrated()
    calibrator = ProbabilityCalibrator(cfg=CalibrationConfig(method="isotonic"))
    calibrator.fit(labels, scores)
    assert calibrator.metrics_after["brier"] <= calibrator.metrics_before["brier"]
    assert calibrator.metrics_after["ece"] < calibrator.metrics_before["ece"]


def test_platt_calibration_runs_and_stays_in_range():
    labels, scores = _miscalibrated(seed=1)
    calibrator = ProbabilityCalibrator(cfg=CalibrationConfig(method="platt"))
    calibrator.fit(labels, scores)
    out = calibrator.transform(scores)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_calibrated_probabilities_are_always_in_range():
    labels, scores = _miscalibrated(seed=2)
    calibrator = ProbabilityCalibrator(cfg=CalibrationConfig(method="isotonic"))
    calibrator.fit(labels, scores)
    out = calibrator.transform(np.array([-5.0, 0.0, 0.5, 1.0, 7.0]))
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_single_class_calibration_set_does_not_fit_a_degenerate_mapping():
    calibrator = ProbabilityCalibrator(cfg=CalibrationConfig(method="isotonic"))
    calibrator.fit(np.zeros(100), np.linspace(0.0, 1.0, 100))
    assert calibrator.method == "none_single_class"
    np.testing.assert_allclose(calibrator.transform(np.array([0.3])), [0.3])


def test_thin_probability_calibration_set_is_refused():
    calibrator = ProbabilityCalibrator(cfg=CalibrationConfig(min_calibration_rows=50))
    with pytest.raises(ValueError, match="at least 50"):
        calibrator.fit(np.array([0.0, 1.0]), np.array([0.2, 0.8]))


def test_unfitted_calibrator_refuses_to_transform():
    calibrator = ProbabilityCalibrator(cfg=CalibrationConfig())
    with pytest.raises(RuntimeError, match="not fitted"):
        calibrator.transform(np.array([0.5]))


def test_reliability_curve_bins_sum_to_the_sample():
    labels, scores = _miscalibrated(seed=3)
    curve = reliability_curve(labels, scores, n_bins=10)
    assert int(curve["n"].sum()) == len(labels)


def test_perfect_probabilities_have_zero_brier():
    assert brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)


def test_ece_is_zero_for_a_perfectly_calibrated_forecaster():
    rng = np.random.default_rng(5)
    probability = rng.uniform(0.05, 0.95, size=20000)
    labels = rng.binomial(1, probability).astype(float)
    assert expected_calibration_error(labels, probability, n_bins=10) < 0.02


def test_threshold_tuning_maximises_the_named_objective():
    labels = np.array([0.0] * 50 + [1.0] * 50)
    scores = np.concatenate([np.linspace(0.0, 0.5, 50), np.linspace(0.5, 1.0, 50)])
    threshold, stats = tune_threshold(labels, scores, objective="f1")
    assert 0.0 <= threshold <= 1.0
    assert stats["f1"] > 0.8


def test_precision_at_recall_objective_respects_the_floor():
    rng = np.random.default_rng(6)
    labels = rng.binomial(1, 0.3, size=500).astype(float)
    scores = np.clip(labels * 0.5 + rng.normal(0.3, 0.2, size=500), 0, 1)
    _, stats = tune_threshold(labels, scores, objective="precision_at_recall", min_recall=0.8)
    assert stats["recall"] >= 0.8


def test_risk_metrics_are_nan_not_fabricated_on_a_single_class_set():
    """A degenerate set has no AUC; reporting 0.5 would be an invented number."""
    metrics = risk_metrics(np.zeros(50), np.linspace(0, 1, 50), threshold=0.5)
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])
    assert metrics["n_positive"] == 0


def test_risk_metrics_report_the_confusion_matrix():
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = risk_metrics(labels, scores, threshold=0.5)
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 1
