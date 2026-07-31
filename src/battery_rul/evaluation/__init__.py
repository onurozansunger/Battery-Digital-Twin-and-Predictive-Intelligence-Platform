"""Metrics, scoring and report rendering."""

from __future__ import annotations

from battery_rul.evaluation.evaluator import (
    EvaluationResult,
    compare_models,
    compare_models_common_rows,
    cross_validate_by_battery,
    evaluate_model,
    learning_curve,
)
from battery_rul.evaluation.metrics import (
    METRIC_DIRECTION,
    bootstrap_metric_ci,
    compute_metrics,
    per_battery_metrics,
    prognostic_horizon,
    residual_summary,
)
from battery_rul.evaluation.reporting import render_evaluation_report, to_markdown_table

__all__ = [
    "METRIC_DIRECTION",
    "EvaluationResult",
    "bootstrap_metric_ci",
    "compare_models",
    "compare_models_common_rows",
    "compute_metrics",
    "cross_validate_by_battery",
    "evaluate_model",
    "learning_curve",
    "per_battery_metrics",
    "prognostic_horizon",
    "render_evaluation_report",
    "residual_summary",
    "to_markdown_table",
]
