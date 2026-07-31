"""Figure generation: shared style, EDA and result plots."""

from __future__ import annotations

from battery_rul.visualization.eda import generate_eda_figures
from battery_rul.visualization.results import (
    plot_learning_curve,
    plot_model_comparison,
    plot_predictions,
    plot_residual_analysis,
    plot_training_history,
)
from battery_rul.visualization.style import apply_style, battery_palette, figure, save_figure

__all__ = [
    "apply_style",
    "battery_palette",
    "figure",
    "generate_eda_figures",
    "plot_learning_curve",
    "plot_model_comparison",
    "plot_predictions",
    "plot_residual_analysis",
    "plot_training_history",
    "save_figure",
]
