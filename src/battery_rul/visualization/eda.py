"""Exploratory data analysis figures.

Every plot here answers a question that shaped a modelling decision documented
elsewhere in the repo — this is EDA as evidence, not as a gallery.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.utils.logging import get_logger
from battery_rul.visualization.style import battery_palette, figure

logger = get_logger(__name__)

__all__ = ["generate_eda_figures"]


def generate_eda_figures(
    df: pd.DataFrame, cfg: ExperimentConfig, *, outdir: Path | None = None
) -> list[Path]:
    """Render the full EDA set. Returns the paths written."""
    outdir = Path(outdir or cfg.paths.figures_dir / "eda")
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = cfg.viz.figure_format

    batteries = sorted(df["battery_id"].unique().tolist())
    colours = battery_palette(batteries)
    shown = batteries[: cfg.viz.max_batteries_per_plot]
    paths: list[Path] = []

    def _emit(name: str) -> Path:
        path = outdir / f"{name}.{suffix}"
        paths.append(path)
        return path

    # 1 ------------------------------------------------------------------
    with figure(
        nrows=2,
        ncols=1,
        figsize=(11, 9),
        path=_emit("01_capacity_degradation"),
        cfg=cfg.viz,
        sharex=True,
    ) as (fig, axes):
        ax_top, ax_bottom = axes
        eol = cfg.eol_capacity_ah
        for battery in shown:
            g = df[df["battery_id"] == battery]
            ax_top.plot(
                g["cycle_index"], g["capacity_ah"], alpha=0.28, color=colours[battery], lw=1.0
            )
            ax_top.plot(
                g["cycle_index"],
                g["capacity_smooth_ah"],
                color=colours[battery],
                label=battery,
                lw=2.0,
            )
            ax_bottom.plot(g["cycle_index"], g["soh"] * 100, color=colours[battery], label=battery)
        ax_top.axhline(eol, ls="--", color="#B00020", lw=1.5)
        ax_top.text(
            0.995,
            eol,
            f"  EOL = {eol:.2f} Ah",
            transform=ax_top.get_yaxis_transform(),
            va="bottom",
            ha="right",
            color="#B00020",
            fontsize=9,
            fontweight="bold",
        )
        ax_top.set_ylabel("Discharge capacity (Ah)")
        ax_top.set_title(
            "Capacity fade — faint lines are raw measurements, bold lines the causal "
            "trailing median"
        )
        ax_top.legend(ncol=4, loc="upper right")
        ax_bottom.axhline(cfg.data.eol_threshold * 100, ls="--", color="#B00020", lw=1.5)
        ax_bottom.set_xlabel("Discharge cycle")
        ax_bottom.set_ylabel("State of health (%)")
        ax_bottom.set_title("State of health relative to the EOL threshold")

    # 2 ------------------------------------------------------------------
    with figure(nrows=2, ncols=2, figsize=(13, 9), path=_emit("02_signal_trends"), cfg=cfg.viz) as (
        fig,
        axes,
    ):
        panels = [
            ("voltage_min_v", "Discharge cut-off voltage (V)", "Voltage"),
            ("temperature_max_c", "Peak cell temperature (°C)", "Temperature"),
            ("internal_resistance_ohm", "Electrolyte resistance Re (Ω)", "Internal resistance"),
            ("cc_ct_ratio", "CC time / total charge time", "Constant-current fraction"),
        ]
        for ax, (column, ylabel, title) in zip(axes.ravel(), panels, strict=True):
            if column not in df.columns:
                ax.set_visible(False)
                continue
            for battery in shown:
                g = df[df["battery_id"] == battery]
                ax.plot(g["cycle_index"], g[column], color=colours[battery], alpha=0.85, lw=1.4)
            ax.set_xlabel("Discharge cycle")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        fig.suptitle("Degradation signatures beyond capacity", fontsize=14, fontweight="semibold")

    # 3 ------------------------------------------------------------------
    with figure(
        nrows=1, ncols=3, figsize=(15, 4.6), path=_emit("03_current_and_temperature"), cfg=cfg.viz
    ) as (fig, axes):
        ax0, ax1, ax2 = axes
        for battery in shown:
            g = df[df["battery_id"] == battery]
            ax0.plot(g["cycle_index"], g["current_mean_a"], color=colours[battery], lw=1.2)
            ax1.plot(g["cycle_index"], g["temperature_mean_c"], color=colours[battery], lw=1.2)
            ax2.plot(
                g["cycle_index"], g["discharge_duration_s"] / 60.0, color=colours[battery], lw=1.2
            )
        ax0.set_title("Mean load current")
        ax0.set_ylabel("Current (A)")
        ax1.set_title("Mean cell temperature")
        ax1.set_ylabel("Temperature (°C)")
        ax2.set_title("Discharge duration")
        ax2.set_ylabel("Minutes")
        for ax in axes:
            ax.set_xlabel("Discharge cycle")

    # 4 ------------------------------------------------------------------
    with figure(
        nrows=1, ncols=3, figsize=(15, 4.6), path=_emit("04_distributions"), cfg=cfg.viz
    ) as (fig, axes):
        ax0, ax1, ax2 = axes
        counts = df.groupby("battery_id").size().sort_values()
        ax0.barh(
            counts.index.astype(str), counts.to_numpy(), color=[colours[b] for b in counts.index]
        )
        ax0.set_title("Cycles per cell")
        ax0.set_xlabel("Discharge cycles")
        ax0.grid(axis="y", visible=False)

        ax1.hist(df["capacity_ah"].dropna(), bins=40, color="#0072B2", alpha=0.85)
        ax1.axvline(cfg.eol_capacity_ah, ls="--", color="#B00020", lw=1.5)
        ax1.set_title("Capacity distribution (all cells)")
        ax1.set_xlabel("Capacity (Ah)")

        ax2.hist(df["soh"].dropna() * 100, bins=40, color="#009E73", alpha=0.85)
        ax2.axvline(cfg.data.eol_threshold * 100, ls="--", color="#B00020", lw=1.5)
        ax2.set_title("State-of-health distribution")
        ax2.set_xlabel("SoH (%)")

    # 5 ------------------------------------------------------------------
    numeric = df.select_dtypes(include=[np.number])
    outlier_columns = [
        c
        for c in (
            "capacity_ah",
            "voltage_min_v",
            "temperature_max_c",
            "internal_resistance_ohm",
            "discharge_duration_s",
            "energy_throughput_wh",
            "coulombic_efficiency",
        )
        if c in numeric.columns
    ]
    if outlier_columns:
        with figure(
            nrows=1, ncols=1, figsize=(12, 5.5), path=_emit("05_outlier_analysis"), cfg=cfg.viz
        ) as (fig, ax):
            # Robust z-scores: median/MAD rather than mean/std, because the mean is
            # itself dragged by the outliers we are trying to see.
            data = []
            for column in outlier_columns:
                series = numeric[column].dropna()
                median = series.median()
                mad = (series - median).abs().median()
                scale = 1.4826 * mad if mad > 0 else series.std()
                data.append(
                    ((series - median) / scale).to_numpy() if scale else np.zeros(len(series))
                )
            bp = ax.boxplot(
                data,
                labels=[c.replace("_", "\n") for c in outlier_columns],
                showfliers=True,
                patch_artist=True,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("#0072B2")
                patch.set_alpha(0.35)
            ax.axhline(3, ls=":", color="#B00020")
            ax.axhline(-3, ls=":", color="#B00020")
            ax.set_ylabel("Robust z-score (median / MAD)")
            ax.set_title("Outlier structure — dotted lines mark ±3 robust σ")
            ax.set_ylim(-12, 12)

    # 6 ------------------------------------------------------------------
    corr_columns = [
        c
        for c in numeric.columns
        if c not in {"cycle_index", "n_samples_discharge", "reference_capacity_ah"}
        and numeric[c].notna().sum() > 10
        and numeric[c].std() > 0
    ][:22]
    if len(corr_columns) > 2:
        corr = numeric[corr_columns].corr()
        with figure(
            nrows=1, ncols=1, figsize=(11.5, 10), path=_emit("06_correlation_matrix"), cfg=cfg.viz
        ) as (fig, ax):
            im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr_columns)))
            ax.set_yticks(range(len(corr_columns)))
            ax.set_xticklabels(corr_columns, rotation=90, fontsize=8)
            ax.set_yticklabels(corr_columns, fontsize=8)
            ax.grid(visible=False)
            fig.colorbar(im, ax=ax, shrink=0.78, label="Pearson ρ")
            ax.set_title("Correlation between raw cycle-level signals")

    # 7 ------------------------------------------------------------------
    summary = (
        df.groupby("battery_id")
        .agg(
            n_cycles=("cycle_index", "size"),
            cap_start=("capacity_smooth_ah", "first"),
            cap_end=("capacity_smooth_ah", "last"),
            temp_max=("temperature_max_c", "max"),
            ambient=("ambient_temperature_c", "mean"),
        )
        .reset_index()
    )
    summary["fade_pct"] = 100 * (summary["cap_start"] - summary["cap_end"]) / summary["cap_start"]
    with figure(
        nrows=1, ncols=2, figsize=(13, 5), path=_emit("07_battery_comparison"), cfg=cfg.viz
    ) as (fig, axes):
        ax0, ax1 = axes
        bars = ax0.bar(
            summary["battery_id"].astype(str),
            summary["fade_pct"],
            color=[colours[b] for b in summary["battery_id"]],
        )
        ax0.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
        ax0.set_title("Total capacity fade over the record")
        ax0.set_ylabel("Fade (%)")
        ax0.tick_params(axis="x", rotation=45)
        ax0.grid(axis="x", visible=False)

        scatter = ax1.scatter(
            summary["ambient"],
            summary["fade_pct"],
            s=summary["n_cycles"] * 1.2,
            c=summary["temp_max"],
            cmap="magma",
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        for _, row in summary.iterrows():
            ax1.annotate(
                row["battery_id"],
                (row["ambient"], row["fade_pct"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
        fig.colorbar(scatter, ax=ax1, label="Peak cell temperature (°C)")
        ax1.set_xlabel("Ambient temperature (°C)")
        ax1.set_ylabel("Fade (%)")
        ax1.set_title("Fade vs test conditions (marker size = cycles)")

    # 8 ------------------------------------------------------------------
    if "rul_cycles" in df.columns:
        with figure(
            nrows=1, ncols=2, figsize=(13, 5), path=_emit("08_target_distribution"), cfg=cfg.viz
        ) as (fig, axes):
            ax0, ax1 = axes
            ax0.hist(df["rul_cycles"].dropna(), bins=40, color="#D55E00", alpha=0.85)
            ax0.set_title("RUL target distribution")
            ax0.set_xlabel("Remaining useful life (cycles)")
            ax0.set_ylabel("Rows")
            for battery in shown:
                g = df[df["battery_id"] == battery]
                ax1.plot(g["cycle_index"], g["rul_cycles"], color=colours[battery], label=battery)
            ax1.set_title("RUL trajectory per cell")
            ax1.set_xlabel("Discharge cycle")
            ax1.set_ylabel("RUL (cycles)")
            ax1.legend(ncol=3, fontsize=8)

    logger.info("Wrote %d EDA figures to %s", len(paths), outdir)
    return paths
