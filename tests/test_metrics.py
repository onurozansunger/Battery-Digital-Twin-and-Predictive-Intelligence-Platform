"""Evaluation metrics, including the prognostics-specific ones."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_rul.evaluation.metrics import (
    METRIC_DIRECTION,
    bootstrap_metric_ci,
    compute_metrics,
    per_battery_metrics,
    prognostic_horizon,
    residual_summary,
)


def test_perfect_predictions_are_perfect():
    y = np.array([100.0, 50.0, 25.0, 10.0, 0.0])
    m = compute_metrics(y, y)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["mape"] == pytest.approx(0.0)
    assert m["alpha_lambda"] == pytest.approx(1.0)


def test_known_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx((2 + 2 + 3) / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert m["max_error"] == pytest.approx(3.0)
    assert m["bias"] == pytest.approx((2 - 2 + 3) / 3)


def test_mape_is_finite_at_zero_rul():
    """RUL legitimately hits zero; MAPE must not blow up."""
    y_true = np.array([0.0, 5.0, 10.0])
    y_pred = np.array([2.0, 5.0, 10.0])
    m = compute_metrics(y_true, y_pred, mape_epsilon=1.0)
    assert np.isfinite(m["mape"])
    assert m["mape"] > 0


def test_nan_predictions_are_excluded_not_counted():
    """Sequence models emit NaN for un-windowable rows."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([np.nan, 20.0, 30.0, 40.0])
    m = compute_metrics(y_true, y_pred)
    assert m["n"] == 3
    assert m["mae"] == pytest.approx(0.0)


def test_all_nan_returns_nan_not_crash():
    m = compute_metrics([1.0, 2.0], [np.nan, np.nan])
    assert m["n"] == 0
    assert np.isnan(m["mae"])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_metrics([1.0, 2.0], [1.0])


def test_bias_sign_convention():
    """Positive bias == optimistic == predicting more life than remains."""
    assert compute_metrics([10.0], [20.0])["bias"] > 0
    assert compute_metrics([20.0], [10.0])["bias"] < 0


def test_alpha_lambda_uses_a_relative_cone():
    y_true = np.array([100.0, 10.0])
    # 15 cycles off: inside the 20 % cone at RUL 100, outside it at RUL 10.
    y_pred = np.array([115.0, 25.0])
    assert compute_metrics(y_true, y_pred, alpha=0.20)["alpha_lambda"] == pytest.approx(0.5)


def test_metric_direction_covers_reported_metrics():
    m = compute_metrics([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
    for key in m:
        if key in {"n", "mse", "std_residual", "median_ae", "smape"}:
            continue
        assert key in METRIC_DIRECTION, f"{key} has no optimisation direction"


# ---------------------------------------------------------------------------
def test_prognostic_horizon_perfect_model():
    y = np.arange(100, 0, -1, dtype=float)
    assert prognostic_horizon(y, y) == pytest.approx(100.0)


def test_prognostic_horizon_is_nan_when_never_accurate():
    y_true = np.arange(50, 0, -1, dtype=float)
    assert np.isnan(prognostic_horizon(y_true, y_true * 5))


def test_prognostic_horizon_detects_late_convergence():
    """A model that is wrong early but right near EOL has a short horizon."""
    y_true = np.arange(100, 0, -1, dtype=float)
    y_pred = y_true.copy()
    y_pred[y_true > 40] += 60  # badly wrong while RUL is high
    horizon = prognostic_horizon(y_true, y_pred, alpha=0.2)
    assert 30 <= horizon <= 45


# ---------------------------------------------------------------------------
def test_per_battery_metrics_are_independent():
    frame = pd.DataFrame(
        {
            "battery_id": ["A"] * 5 + ["B"] * 5,
            "y_true": [10.0, 8, 6, 4, 2] * 2,
            "y_pred": [10.0, 8, 6, 4, 2] + [20.0, 18, 16, 14, 12],
        }
    )
    out = per_battery_metrics(frame)
    assert len(out) == 2
    a = out.set_index("battery_id").loc["A"]
    b = out.set_index("battery_id").loc["B"]
    assert a["mae"] == pytest.approx(0.0)
    assert b["mae"] == pytest.approx(10.0)


def test_residual_summary_reports_quantiles_and_shape():
    rng = np.random.default_rng(0)
    y_true = np.linspace(100, 0, 200)
    y_pred = y_true + rng.normal(0, 5, 200)
    summary = residual_summary(y_true, y_pred)
    assert {"q05", "q50", "q95", "mean", "std", "skew", "kurtosis"} <= set(summary)
    assert summary["q05"] < summary["q50"] < summary["q95"]


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y_true = np.linspace(100, 0, 300)
    y_pred = y_true + rng.normal(0, 8, 300)
    ci = bootstrap_metric_ci(y_true, y_pred, metric="rmse", n_samples=200, seed=3)
    assert ci["lower"] <= ci["point"] <= ci["upper"]
    assert ci["n_bootstrap"] > 0


def test_bootstrap_disabled_returns_empty():
    assert bootstrap_metric_ci([1.0, 2.0], [1.0, 2.0], n_samples=0) == {}
