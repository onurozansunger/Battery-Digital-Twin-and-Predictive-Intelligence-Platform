"""Result figures: predictions, residuals, comparison and learning curves."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.evaluation.evaluator import EvaluationResult
from battery_rul.utils.logging import get_logger
from battery_rul.visualization.style import battery_palette, figure

logger = get_logger(__name__)

__all__ = [
    "plot_learning_curve",
    "plot_model_comparison",
    "plot_predictions",
    "plot_residual_analysis",
]


def plot_predictions(
    result: EvaluationResult, cfg: ExperimentConfig, *, outdir: Path | None = None
) -> list[Path]:
    """Prediction-vs-truth scatter plus per-cell RUL trajectories."""
    outdir = Path(outdir or cfg.paths.figures_dir / "results")
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = cfg.viz.figure_format
    paths: list[Path] = []

    data = result.predictions.dropna(subset=["y_pred"])
    if data.empty:
        logger.warning("No scoreable predictions for %s; skipping figures", result.model_name)
        return paths

    batteries = sorted(data["battery_id"].unique().tolist())
    colours = battery_palette(batteries)

    # -- scatter -----------------------------------------------------------
    path = outdir / f"pred_vs_truth_{result.model_name}.{suffix}"
    with figure(nrows=1, ncols=1, figsize=(7.5, 7), path=path, cfg=cfg.viz) as (fig, ax):
        limit = float(max(data["y_true"].max(), data["y_pred"].max())) * 1.05
        alpha = cfg.evaluation.alpha
        xs = np.linspace(0, limit, 200)
        ax.fill_between(
            xs,
            xs * (1 - alpha),
            xs * (1 + alpha),
            color="#009E73",
            alpha=0.13,
            label=f"±{alpha:.0%} α-λ cone",
        )
        ax.plot([0, limit], [0, limit], ls="--", color="#444444", lw=1.3, label="perfect")
        for battery in batteries:
            g = data[data["battery_id"] == battery]
            ax.scatter(
                g["y_true"],
                g["y_pred"],
                s=22,
                alpha=0.8,
                color=colours[battery],
                label=battery,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("True RUL (cycles)")
        ax.set_ylabel("Predicted RUL (cycles)")
        ax.set_title(
            f"{result.model_name} — predicted vs true RUL ({result.partition})\n"
            f"MAE {result.metrics['mae']:.1f} · RMSE {result.metrics['rmse']:.1f} · "
            f"R² {result.metrics['r2']:.3f}"
        )
        ax.legend(loc="upper left", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
    paths.append(path)

    # -- trajectories --------------------------------------------------------
    n = len(batteries)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    path = outdir / f"rul_trajectories_{result.model_name}.{suffix}"
    with figure(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.2 * ncols, 4.0 * nrows),
        path=path,
        cfg=cfg.viz,
        squeeze=False,
    ) as (fig, axes):
        flat = axes.ravel()
        for ax, battery in zip(flat, batteries, strict=False):
            g = data[data["battery_id"] == battery].sort_values("cycle_index")
            ax.plot(g["cycle_index"], g["y_true"], color="#333333", lw=2.2, label="true RUL")
            ax.plot(
                g["cycle_index"],
                g["y_pred"],
                color=colours[battery],
                lw=1.8,
                ls="--",
                marker="o",
                markersize=3,
                label="predicted",
            )
            ax.fill_between(
                g["cycle_index"],
                g["y_true"],
                g["y_pred"],
                color=colours[battery],
                alpha=0.16,
            )
            mae = float(g["abs_error"].mean())
            ax.set_title(f"{battery} — MAE {mae:.1f} cycles")
            ax.set_xlabel("Discharge cycle")
            ax.set_ylabel("RUL (cycles)")
            ax.legend(fontsize=8)
        for ax in flat[n:]:
            ax.set_visible(False)
        fig.suptitle(
            f"{result.model_name} — RUL trajectory on held-out cells",
            fontsize=14,
            fontweight="semibold",
        )
    paths.append(path)
    return paths


def plot_residual_analysis(
    result: EvaluationResult, cfg: ExperimentConfig, *, outdir: Path | None = None
) -> Path | None:
    """Four-panel residual diagnostic."""
    outdir = Path(outdir or cfg.paths.figures_dir / "results")
    outdir.mkdir(parents=True, exist_ok=True)
    data = result.predictions.dropna(subset=["y_pred"])
    if data.empty:
        return None

    residual = data["residual"].to_numpy()
    path = outdir / f"residual_analysis_{result.model_name}.{cfg.viz.figure_format}"
    with figure(nrows=2, ncols=2, figsize=(13, 9), path=path, cfg=cfg.viz) as (fig, axes):
        ax0, ax1, ax2, ax3 = axes.ravel()

        ax0.hist(residual, bins=cfg.evaluation.residual_bins, color="#0072B2", alpha=0.85)
        ax0.axvline(0, color="#B00020", ls="--")
        ax0.axvline(
            residual.mean(), color="#E69F00", ls="-", lw=2, label=f"bias {residual.mean():+.1f}"
        )
        ax0.set_title("Error distribution")
        ax0.set_xlabel("Predicted − true (cycles)")
        ax0.legend(fontsize=8)

        colours = battery_palette(sorted(data["battery_id"].unique().tolist()))
        for battery, g in data.groupby("battery_id"):
            ax1.scatter(
                g["y_true"], g["residual"], s=18, alpha=0.75, color=colours[battery], label=battery
            )
        ax1.axhline(0, color="#B00020", ls="--")
        ax1.set_title("Residual vs true RUL — reveals level-dependent bias")
        ax1.set_xlabel("True RUL (cycles)")
        ax1.set_ylabel("Residual (cycles)")
        ax1.legend(fontsize=8, ncol=2)

        # Q-Q against the normal: heavy tails mean the RMSE is driven by a few rows.
        ordered = np.sort(residual)
        from scipy import stats

        theoretical = stats.norm.ppf(
            np.linspace(0.5 / len(ordered), 1 - 0.5 / len(ordered), len(ordered))
        )
        theoretical = theoretical * ordered.std() + ordered.mean()
        ax2.scatter(theoretical, ordered, s=16, color="#009E73", alpha=0.8)
        lo, hi = float(min(theoretical.min(), ordered.min())), float(
            max(theoretical.max(), ordered.max())
        )
        ax2.plot([lo, hi], [lo, hi], ls="--", color="#444444")
        ax2.set_title("Q–Q plot vs normal")
        ax2.set_xlabel("Theoretical quantile")
        ax2.set_ylabel("Observed residual")

        if "soh" in data.columns and data["soh"].notna().any():
            ax3.scatter(data["soh"] * 100, data["abs_error"], s=18, alpha=0.7, color="#D55E00")
            ax3.set_xlabel("State of health (%)")
            ax3.set_title("Absolute error vs cell condition")
        else:
            ax3.scatter(data["cycle_index"], data["abs_error"], s=18, alpha=0.7, color="#D55E00")
            ax3.set_xlabel("Discharge cycle")
            ax3.set_title("Absolute error vs cycle")
        ax3.set_ylabel("|error| (cycles)")

        fig.suptitle(
            f"{result.model_name} — residual diagnostics ({result.partition})",
            fontsize=14,
            fontweight="semibold",
        )
    return path


def plot_model_comparison(
    comparison: pd.DataFrame, cfg: ExperimentConfig, *, outdir: Path | None = None
) -> Path | None:
    """Bar chart of the headline metrics across every trained model."""
    outdir = Path(outdir or cfg.paths.figures_dir / "results")
    outdir.mkdir(parents=True, exist_ok=True)
    if comparison is None or comparison.empty:
        return None

    metrics = [m for m in ("mae", "rmse", "r2", "alpha_lambda") if m in comparison.columns]
    path = outdir / f"model_comparison.{cfg.viz.figure_format}"
    with figure(
        nrows=1,
        ncols=len(metrics),
        figsize=(4.2 * len(metrics), 5.5),
        path=path,
        cfg=cfg.viz,
        squeeze=False,
    ) as (fig, axes):
        table = comparison.sort_values("rmse")
        for ax, metric in zip(axes.ravel(), metrics, strict=True):
            values = table[metric].to_numpy(dtype=float)
            best = np.nanargmin(values) if metric in {"mae", "rmse"} else np.nanargmax(values)
            colours = ["#009E73" if i == best else "#0072B2" for i in range(len(values))]
            bars = ax.barh(table["model"].astype(str), values, color=colours, alpha=0.9)
            ax.bar_label(bars, fmt="%.3g", fontsize=8, padding=3)
            ax.set_title(metric.replace("_", " ").upper())
            ax.grid(axis="y", visible=False)
            ax.invert_yaxis()
            if metric == "r2":
                ax.set_xlim(min(0.0, float(np.nanmin(values)) * 1.1), 1.05)
        fig.suptitle(
            "Model comparison on the held-out test cells (green = best)",
            fontsize=14,
            fontweight="semibold",
        )
    return path


def plot_learning_curve(
    curve: pd.DataFrame, model_name: str, cfg: ExperimentConfig, *, outdir: Path | None = None
) -> Path | None:
    outdir = Path(outdir or cfg.paths.figures_dir / "results")
    outdir.mkdir(parents=True, exist_ok=True)
    if curve is None or curve.empty:
        return None

    path = outdir / f"learning_curve_{model_name}.{cfg.viz.figure_format}"
    with figure(nrows=1, ncols=2, figsize=(12, 5), path=path, cfg=cfg.viz) as (fig, axes):
        ax0, ax1 = axes
        ax0.plot(curve["n_train_rows"], curve["train_rmse"], marker="o", label="train")
        ax0.plot(curve["n_train_rows"], curve["test_rmse"], marker="s", label="test")
        ax0.set_xlabel("Training rows")
        ax0.set_ylabel("RMSE (cycles)")
        ax0.set_title("Learning curve — RMSE")
        ax0.legend()

        ax1.plot(curve["n_train_rows"], curve["train_r2"], marker="o", label="train")
        ax1.plot(curve["n_train_rows"], curve["test_r2"], marker="s", label="test")
        ax1.set_xlabel("Training rows")
        ax1.set_ylabel("R²")
        ax1.set_title("Learning curve — R²")
        ax1.legend()

        fig.suptitle(
            f"{model_name} — learning curve (training data subsampled by cell)",
            fontsize=14,
            fontweight="semibold",
        )
    return path


def plot_training_history(
    history: dict[str, list[float]],
    model_name: str,
    cfg: ExperimentConfig,
    *,
    outdir: Path | None = None,
) -> Path | None:
    """Loss curves for the neural models."""
    outdir = Path(outdir or cfg.paths.figures_dir / "results")
    outdir.mkdir(parents=True, exist_ok=True)
    if not history or "train_loss" not in history:
        return None

    path = outdir / f"training_history_{model_name}.{cfg.viz.figure_format}"
    with figure(nrows=1, ncols=1, figsize=(9, 5), path=path, cfg=cfg.viz) as (fig, ax):
        epochs = range(len(history["train_loss"]))
        ax.plot(epochs, history["train_loss"], label="train loss")
        if history.get("val_loss"):
            ax.plot(epochs, history["val_loss"], label="validation loss")
            best = int(np.argmin(history["val_loss"]))
            ax.axvline(best, ls=":", color="#B00020", label=f"best epoch {best}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{cfg.models.training.loss.upper()} loss")
        ax.set_title(f"{model_name} — training history")
        ax.legend()
    return path
