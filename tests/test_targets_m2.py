"""Milestone 2 target generation: state of health and failure risk."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_rul.config import ExperimentConfig
from battery_rul.targets.risk import (
    RiskClass,
    attach_failure_risk_target,
    classify_risk,
    risk_target_column,
)
from battery_rul.targets.soh import (
    HealthClass,
    attach_soh_target,
    classify_soh,
    reference_capacity,
)


# ---------------------------------------------------------------------------
# SOH reference strategies
# ---------------------------------------------------------------------------
def _cell(capacities: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "battery_id": "X",
            "cycle_index": np.arange(1, len(capacities) + 1),
            "capacity_ah": capacities,
            "capacity_smooth_ah": capacities,
        }
    )


def test_nominal_reference_uses_the_rating(cfg: ExperimentConfig):
    cfg.soh.reference_strategy = "nominal"
    cfg.data.nominal_capacity_ah = 2.0
    assert reference_capacity(_cell([1.5, 1.4, 1.3]), cfg) == pytest.approx(2.0)


def test_first_cycle_reference_uses_the_opening_measurement(cfg: ExperimentConfig):
    cfg.soh.reference_strategy = "first_cycle"
    assert reference_capacity(_cell([1.8, 1.7, 1.6]), cfg) == pytest.approx(1.8)


def test_first_n_mean_reference_is_robust_to_one_bad_opening_reading(cfg: ExperimentConfig):
    """The reason this is the default: one aborted discharge must not set the scale."""
    cfg.soh.reference_strategy = "first_n_cycle_mean"
    cfg.soh.reference_cycles = 5
    capacities = [1.0, 1.9, 1.9, 1.9, 1.9, 1.8]
    first_cycle_value = 1.0
    n_mean = reference_capacity(_cell(capacities), cfg)
    assert n_mean == pytest.approx(np.mean(capacities[:5]))
    assert n_mean > first_cycle_value * 1.5


def test_reference_ignores_non_positive_readings(cfg: ExperimentConfig):
    cfg.soh.reference_strategy = "first_cycle"
    assert reference_capacity(_cell([0.0, 1.8, 1.7]), cfg) == pytest.approx(1.8)


def test_reference_raises_when_nothing_is_measurable(cfg: ExperimentConfig):
    cfg.soh.reference_strategy = "first_cycle"
    with pytest.raises(ValueError, match="no valid capacity"):
        reference_capacity(_cell([np.nan, np.nan]), cfg)


def test_reference_is_causal(cfg: ExperimentConfig):
    """Truncating the future must not move the reference."""
    cfg.soh.reference_strategy = "first_n_cycle_mean"
    capacities = [1.9, 1.88, 1.86, 1.84, 1.82, 1.5, 1.2]
    full = reference_capacity(_cell(capacities), cfg)
    prefix = reference_capacity(_cell(capacities[:5]), cfg)
    assert full == pytest.approx(prefix)


# ---------------------------------------------------------------------------
# SOH target and banding
# ---------------------------------------------------------------------------
def test_soh_target_is_a_fraction(labelled_cycles: pd.DataFrame, cfg: ExperimentConfig):
    frame, report = attach_soh_target(labelled_cycles, cfg)
    values = frame[cfg.soh.target_name].to_numpy(dtype=float)
    assert values.min() >= cfg.soh.plausible_min
    assert values.max() <= cfg.soh.plausible_max
    assert report.to_dict()["representation"] == "fraction in [0, 1]"


def test_soh_target_matches_capacity_over_reference(
    labelled_cycles: pd.DataFrame, cfg: ExperimentConfig
):
    cfg.soh.reference_strategy = "nominal"
    frame, _ = attach_soh_target(labelled_cycles, cfg)
    expected = frame["capacity_smooth_ah"] / cfg.data.nominal_capacity_ah
    np.testing.assert_allclose(
        frame[cfg.soh.target_name].to_numpy(dtype=float),
        np.clip(expected.to_numpy(dtype=float), cfg.soh.plausible_min, cfg.soh.plausible_max),
        rtol=1e-4,
    )


def test_soh_values_outside_the_plausible_range_are_clipped(cfg: ExperimentConfig):
    cfg.soh.reference_strategy = "nominal"
    cfg.data.nominal_capacity_ah = 2.0
    frame = _cell([10.0, 1.5, 0.01])
    out, report = attach_soh_target(frame, cfg)
    assert report.n_clipped >= 1
    # float32 storage: compare with the representation tolerance, not exactly.
    assert out[cfg.soh.target_name].max() == pytest.approx(cfg.soh.plausible_max, rel=1e-6)
    assert out[cfg.soh.target_name].min() == pytest.approx(cfg.soh.plausible_min, rel=1e-6)


@pytest.mark.parametrize(
    ("soh", "expected"),
    [
        (0.99, HealthClass.HEALTHY),
        (0.90, HealthClass.HEALTHY),
        (0.85, HealthClass.SLIGHTLY_DEGRADED),
        (0.80, HealthClass.SLIGHTLY_DEGRADED),
        (0.75, HealthClass.WARNING),
        (0.70, HealthClass.WARNING),
        (0.60, HealthClass.CRITICAL),
    ],
)
def test_health_bands_are_inclusive_at_the_lower_edge(
    soh: float, expected: HealthClass, cfg: ExperimentConfig
):
    assert classify_soh(soh, cfg.soh) is expected


def test_unknown_soh_is_not_silently_healthy(cfg: ExperimentConfig):
    """A missing measurement must not render as a green tile."""
    assert classify_soh(None, cfg.soh) is HealthClass.UNKNOWN
    assert classify_soh(float("nan"), cfg.soh) is HealthClass.UNKNOWN


def test_health_bands_are_configurable(cfg: ExperimentConfig):
    cfg.soh.healthy_min = 0.95
    assert classify_soh(0.93, cfg.soh) is HealthClass.SLIGHTLY_DEGRADED


def test_band_ordering_is_validated():
    from battery_rul.config import SOHConfig

    with pytest.raises(ValueError, match="SOH bands must satisfy"):
        SOHConfig(healthy_min=0.7, slightly_degraded_min=0.8, warning_min=0.9)


# ---------------------------------------------------------------------------
# Failure risk
# ---------------------------------------------------------------------------
def test_risk_label_is_rul_below_horizon(labelled_cycles: pd.DataFrame, cfg: ExperimentConfig):
    cfg.risk.horizon_cycles = 30
    frame, _ = attach_failure_risk_target(labelled_cycles, cfg)
    rul = frame[cfg.target.name].to_numpy(dtype=float)
    label = pd.to_numeric(frame[cfg.risk.target_name], errors="coerce").to_numpy(dtype=float)
    expected = (rul <= 30).astype(float)
    np.testing.assert_allclose(label, expected)


def test_additional_horizons_are_attached(labelled_cycles: pd.DataFrame, cfg: ExperimentConfig):
    cfg.risk.horizon_cycles = 30
    cfg.risk.additional_horizons = [20, 50]
    frame, report = attach_failure_risk_target(labelled_cycles, cfg)
    for horizon in (20, 30, 50):
        assert risk_target_column(horizon, cfg.risk.target_name) in frame.columns
    assert report.horizons == [20, 30, 50]


def test_longer_horizons_have_at_least_as_many_positives(
    labelled_cycles: pd.DataFrame, cfg: ExperimentConfig
):
    cfg.risk.horizon_cycles = 30
    cfg.risk.additional_horizons = [20, 50]
    _, report = attach_failure_risk_target(labelled_cycles, cfg)
    assert report.positives[20] <= report.positives[30] <= report.positives[50]


def test_risk_target_requires_the_rul_column(labelled_cycles: pd.DataFrame, cfg: ExperimentConfig):
    """The risk label must be a thresholding of RUL, never an independent derivation."""
    frame = labelled_cycles.drop(columns=[cfg.target.name])
    with pytest.raises(KeyError, match="needs the RUL column"):
        attach_failure_risk_target(frame, cfg)


def test_risk_report_states_the_label_is_derived(
    labelled_cycles: pd.DataFrame, cfg: ExperimentConfig
):
    _, report = attach_failure_risk_target(labelled_cycles, cfg)
    assert report.to_dict()["is_observed_safety_failure"] is False


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, RiskClass.LOW),
        (0.19, RiskClass.LOW),
        (0.20, RiskClass.MEDIUM),
        (0.49, RiskClass.MEDIUM),
        (0.50, RiskClass.HIGH),
        (0.79, RiskClass.HIGH),
        (0.80, RiskClass.VERY_HIGH),
        (1.0, RiskClass.VERY_HIGH),
    ],
)
def test_risk_bands(probability: float, expected: RiskClass, cfg: ExperimentConfig):
    assert classify_risk(probability, cfg.risk) is expected


def test_unknown_risk_is_not_low(cfg: ExperimentConfig):
    assert classify_risk(None, cfg.risk) is RiskClass.UNKNOWN


def test_attach_all_targets_wires_the_three_together(
    raw_cycles: pd.DataFrame, cfg: ExperimentConfig
):
    from battery_rul.data.loader import derive_health
    from battery_rul.targets import attach_all_targets

    frame, reports = attach_all_targets(derive_health(raw_cycles, cfg), cfg)
    assert cfg.target.name in frame.columns
    assert cfg.soh.target_name in frame.columns
    assert cfg.risk.target_name in frame.columns
    assert set(reports) == {"rul", "soh", "risk"}
