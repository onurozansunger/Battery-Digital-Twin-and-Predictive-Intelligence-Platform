"""Markdown report generation.

The evaluation report is a deliverable, not a log dump: it states the setup, the
numbers, and — importantly — what the numbers do *not* establish. Every table is
rendered from the same objects the pipeline used, so the report cannot drift out
of sync with ``metrics.json``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.evaluation.evaluator import EvaluationResult
from battery_rul.utils.io import atomic_write_text, environment_fingerprint
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["render_evaluation_report", "to_markdown_table"]


def to_markdown_table(df: pd.DataFrame, *, max_rows: int = 60, floatfmt: str = "{:.4f}") -> str:
    """DataFrame -> GitHub-flavoured markdown, without a tabulate dependency."""
    if df is None or df.empty:
        return "_(no data)_"

    frame = df.head(max_rows).copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            values = frame[column]
            # A column that is float only because it carries NaN (cycle counts,
            # EOL indices) should not be rendered as "127.0000".
            finite = values.dropna()
            integral = not finite.empty and (finite % 1 == 0).all()
            fmt = "{:.0f}" if integral else floatfmt
            frame[column] = values.map(lambda v, _f=fmt: "—" if pd.isna(v) else _f.format(v))
        else:
            frame[column] = frame[column].astype(str)

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in frame.astype(str).to_numpy())
    suffix = f"\n\n_… {len(df) - max_rows} more rows omitted._" if len(df) > max_rows else ""
    return f"{header}\n{divider}\n{body}{suffix}"


def render_evaluation_report(
    *,
    cfg: ExperimentConfig,
    comparison: pd.DataFrame,
    comparison_common: pd.DataFrame | None = None,
    cv_metrics: dict[str, Any] | None = None,
    cv_per_fold: pd.DataFrame | None = None,
    results: dict[str, EvaluationResult],
    champion: str,
    dataset_summary: pd.DataFrame,
    split_info: dict[str, Any],
    target_info: dict[str, Any],
    feature_info: dict[str, Any],
    validation_info: dict[str, Any],
    tuning_info: dict[str, Any] | None = None,
    learning_curves: pd.DataFrame | None = None,
    timings: dict[str, float] | None = None,
    figures: list[str] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Write ``reports/evaluation_report.md`` and return its path."""
    output_path = Path(output_path or cfg.paths.reports_dir / "evaluation_report.md")
    env = environment_fingerprint()
    champion_result = results.get(champion)

    lines: list[str] = []
    add = lines.append

    add(f"# Evaluation Report — {cfg.experiment_name}")
    add("")
    add(f"_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}_")
    add("")
    add(f"> {cfg.description}")
    add("")

    # -- headline ----------------------------------------------------------
    add("## 1. Headline result")
    add("")
    if champion_result is not None:
        m = champion_result.metrics
        ci = champion_result.confidence_interval
        ci_text = f" (95 % CI {ci['lower']:.2f}–{ci['upper']:.2f})" if ci and "lower" in ci else ""
        add(
            f"**{champion}** is the champion model, selected by "
            f"`{cfg.models.select_by}` on the **validation** partition and reported "
            f"here on the untouched **test** partition."
        )
        add("")
        add(f"- **MAE** — {m['mae']:.2f} cycles")
        add(f"- **RMSE** — {m['rmse']:.2f} cycles{ci_text}")
        add(f"- **R²** — {m['r2']:.3f}")
        add(
            f"- **MAPE** — {m['mape']:.1f} % (denominator floored at {cfg.evaluation.mape_epsilon:g} cycle)"
        )
        add(
            f"- **α-λ accuracy (α={cfg.evaluation.alpha:.0%})** — "
            f"{m['alpha_lambda']:.1%} of predictions inside the relative error cone"
        )
        add(f"- **Predictions within 10 cycles** — {m['within_10_cycles']:.1%}")
        add(
            f"- **Bias** — {m['bias']:+.2f} cycles "
            f"({'optimistic — predicts more life than remains' if m['bias'] > 0 else 'conservative'})"
        )
        ph = m.get("prognostic_horizon")
        n_ph = m.get("prognostic_horizon_cells", 0)
        if ph is not None and pd.notna(ph):
            add(
                f"- **Prognostic horizon** — predictions settle inside the α cone "
                f"{ph:.0f} cycles before end of life, averaged over the {n_ph} test "
                "cell(s) where they settle at all"
            )
        else:
            add(
                "- **Prognostic horizon** — not reached: predictions never settle "
                f"inside the ±{cfg.evaluation.alpha:.0%} relative cone and stay there. "
                "The cone tightens to a couple of cycles near end of life, which is a "
                "demanding bar at this error level."
            )
    add("")

    # -- setup --------------------------------------------------------------
    add("## 2. Experimental setup")
    add("")
    add("### 2.1 Dataset")
    add("")
    add(to_markdown_table(dataset_summary))
    add("")
    add("### 2.2 Target definition")
    add("")
    add(
        f"RUL(k) = k_EOL − k, where k_EOL is the first **persistent** cycle at which "
        f"trailing-median-smoothed capacity falls to or below "
        f"**{target_info.get('eol_capacity_ah', cfg.eol_capacity_ah):.3f} Ah** "
        f"({cfg.data.eol_threshold:.0%} of the {cfg.data.eol_reference} reference capacity)."
    )
    add("")
    add(f"- Labelled rows: **{target_info.get('n_rows', '—')}**")
    add(
        f"- RUL range: **{target_info.get('rul_min', '—')} – {target_info.get('rul_max', '—')} cycles**"
    )
    add(f"- Mean RUL: **{target_info.get('rul_mean', '—')} cycles**")
    censored = target_info.get("censored_batteries") or []
    if censored:
        add(f"- Excluded as right-censored (never reached EOL): **{', '.join(censored)}**")
    eol_map = target_info.get("eol_cycles") or {}
    if eol_map:
        add("")
        add("End-of-life cycle per cell:")
        add("")
        add(
            to_markdown_table(
                pd.DataFrame(sorted(eol_map.items()), columns=["battery_id", "eol_cycle"])
            )
        )
    add("")

    add("### 2.3 Split")
    add("")
    add(f"**Strategy:** `{split_info.get('strategy')}` — {split_info.get('notes', '')}")
    add("")
    sizes = split_info.get("sizes", {})
    add(f"- train: **{sizes.get('train', 0)}** rows, cells `{split_info.get('train_batteries')}`")
    add(f"- val: **{sizes.get('val', 0)}** rows, cells `{split_info.get('val_batteries')}`")
    add(f"- test: **{sizes.get('test', 0)}** rows, cells `{split_info.get('test_batteries')}`")
    add("")
    add(
        "No random row-level splitting is used anywhere in this project. Consecutive "
        "cycles of one cell are near-duplicates, so a random split lets a model "
        "interpolate between neighbouring rows and produces an R² that means nothing."
    )
    add("")

    add("### 2.4 Features")
    add("")
    add(
        f"- Generated: **{feature_info.get('n_generated', '—')}** causal features from "
        f"{len(cfg.features.base_signals)} base signals"
    )
    add(f"- After unsupervised pruning: **{feature_info.get('n_after_pruning', '—')}**")
    add(
        f"- After supervised top-K selection (train partition only): "
        f"**{feature_info.get('n_selected', '—')}**"
    )
    add(f"- Scaler: `{cfg.features.scaler}`")
    add(f"- Warm-up rows dropped: **{feature_info.get('warmup_rows_dropped', 0)}**")
    add("")
    add(
        "Every feature at cycle *k* is a function of cycles ≤ *k* of the same cell. "
        "This is verified mechanically by `assert_no_leakage`, which rebuilds the "
        "features on a truncated history and requires bit-identical values."
    )
    add("")

    # -- validation ----------------------------------------------------------
    add("## 3. Data quality")
    add("")
    add(
        f"- Rows in: **{validation_info.get('n_rows_in', '—')}** → out: "
        f"**{validation_info.get('n_rows_out', '—')}**"
    )
    add(
        f"- Cells in: **{validation_info.get('n_batteries_in', '—')}** → out: "
        f"**{validation_info.get('n_batteries_out', '—')}**"
    )
    issues = validation_info.get("issues") or []
    if issues:
        add("")
        add(to_markdown_table(pd.DataFrame(issues)[["check", "severity", "message"]], max_rows=25))
    add("")

    # -- comparison ------------------------------------------------------------
    add("## 4. Model comparison (test partition)")
    add("")
    add(to_markdown_table(comparison))
    add("")
    add(
        "`n_unscored` counts rows the model could not score. Sequence models need a "
        "full window of history, so the first *w−1* cycles of every test cell are "
        "unscoreable by construction — they are reported, never silently dropped."
    )
    add("")
    if comparison_common is not None and not comparison_common.empty:
        add("### 4.1 Like-for-like: rows every model can score")
        add("")
        add(
            "The table above compares models on different row counts, and the rows "
            "the sequence models skip are the early-life ones — the hardest. That "
            "difference alone can reorder a ranking. This table restricts every "
            "model to the intersection, so the ordering reflects the models rather "
            "than their input requirements."
        )
        add("")
        add(to_markdown_table(comparison_common))
        add("")

    # -- cross-validation ----------------------------------------------------
    if cv_metrics:
        add("### 4.2 Leave-one-battery-out cross-validation")
        add("")
        add(
            f"The cohort is {cv_metrics.get('n_folds', '—')} cells, so the single "
            "holdout above puts **one** cell in the test partition — one sample. "
            "Leave-one-battery-out holds out each cell in turn, re-fitting the "
            "feature pipeline inside every fold, and pools the out-of-fold "
            "predictions. It uses every row for evaluation instead of a fifth of "
            "them, and the spread across folds is a far more honest uncertainty "
            "statement than a bootstrap over correlated rows."
        )
        add("")
        add(
            f"**Pooled ({champion}):** MAE {cv_metrics.get('mae', float('nan')):.2f} · "
            f"RMSE {cv_metrics.get('rmse', float('nan')):.2f} · "
            f"R² {cv_metrics.get('r2', float('nan')):.3f} · "
            f"bias {cv_metrics.get('bias', float('nan')):+.2f} cycles"
        )
        add("")
        add(
            f"**Spread across folds:** MAE σ = {cv_metrics.get('mae_across_folds_std', float('nan')):.2f}, "
            f"RMSE σ = {cv_metrics.get('rmse_across_folds_std', float('nan')):.2f} cycles. "
            "Read that as the real uncertainty on the headline number."
        )
        add("")
        if cv_per_fold is not None and not cv_per_fold.empty:
            keep = [
                c
                for c in ("battery_id", "n", "mae", "rmse", "mape", "r2", "bias", "alpha_lambda")
                if c in cv_per_fold.columns
            ]
            add(to_markdown_table(cv_per_fold[keep]))
            add("")

    # -- per battery --------------------------------------------------------
    add("## 5. Per-cell breakdown")
    add("")
    add(
        "With only a handful of held-out cells, the aggregate number can hide a cell "
        "the model gets badly wrong. This table is the honest view of the result."
    )
    add("")
    for name, result in sorted(results.items()):
        if result.per_battery.empty:
            continue
        add(f"### {name}")
        add("")
        add(to_markdown_table(result.per_battery.drop(columns=["mse"], errors="ignore")))
        add("")

    # -- residuals -----------------------------------------------------------
    if champion_result is not None and champion_result.residuals:
        add("## 6. Residual analysis — champion")
        add("")
        residual_frame = pd.DataFrame(
            sorted(champion_result.residuals.items()), columns=["statistic", "value"]
        )
        add(to_markdown_table(residual_frame))
        add("")
        corr = champion_result.residuals.get("residual_rul_corr")
        if corr is not None:
            direction = "under-predicts" if corr < 0 else "over-predicts"
            add(
                f"Residuals correlate with true RUL at ρ = {corr:+.3f}: the model "
                f"{direction} at high RUL. This is the expected signature of "
                "regression-to-the-mean on a bounded target and is the main reason "
                "early-life predictions should be treated as a range, not a number."
            )
        add("")

    # -- learning curves ------------------------------------------------------
    if learning_curves is not None and not learning_curves.empty:
        add("## 7. Learning curve — champion")
        add("")
        add(to_markdown_table(learning_curves))
        add("")
        add(
            "Training data is subsampled **by cell**, keeping the earliest cycles, so "
            "each point remains a valid temporal split."
        )
        add("")

    # -- tuning ---------------------------------------------------------------
    if tuning_info:
        add("## 8. Hyperparameter optimisation")
        add("")
        add(
            f"Optuna, sampler `{cfg.tuning.sampler}`, pruner `{cfg.tuning.pruner}`, "
            f"{cfg.tuning.n_trials} trials per model, objective `{cfg.tuning.metric}` "
            f"under {cfg.tuning.cv_folds}-fold **battery-grouped** cross-validation."
        )
        add("")
        for model_name, info in sorted(tuning_info.items()):
            add(
                f"**{model_name}** — best {cfg.tuning.metric} = {info.get('best_value', float('nan')):.4f}"
            )
            add("")
            add("```json")
            add(_pretty(info.get("best_params", {})))
            add("```")
            add("")

    # -- figures ----------------------------------------------------------------
    if figures:
        add("## 9. Figures")
        add("")
        for figure in sorted(figures):
            name = Path(figure).stem.replace("_", " ")
            add(f"- `{figure}` — {name}")
        add("")

    # -- limitations --------------------------------------------------------------
    add("## 10. What these numbers do and do not establish")
    add("")
    add(
        "- The test cells were never seen during training, scaling, feature selection "
        "or model choice. The metric is therefore an honest estimate **for cells of "
        "this chemistry and format tested on this rig**."
    )
    add(
        f"- The cohort is **{validation_info.get('n_batteries_out', '—')} cells**. That "
        "is a small sample by any standard; the per-cell table matters more than the "
        "aggregate, and the bootstrap interval is computed over rows (which are "
        "correlated within a cell) so it *understates* true uncertainty."
    )
    add(
        "- Cells that never reached end of life are excluded rather than imputed. "
        "Handling them properly requires survival analysis, which is out of scope here."
    )
    add(
        "- The NASA cells were aged under constant-current profiles in a temperature "
        "chamber. Real duty cycles are irregular; transfer to field data is unproven "
        "and should be measured, not assumed."
    )
    add("")

    add("## 11. Reproducibility")
    add("")
    add("```json")
    add(_pretty({**env, "seed": cfg.seed, "config": cfg.experiment_name}))
    add("```")
    add("")
    if timings:
        add(to_markdown_table(pd.DataFrame(sorted(timings.items()), columns=["stage", "seconds"])))
        add("")

    atomic_write_text(output_path, "\n".join(lines) + "\n")
    logger.info("Evaluation report -> %s", output_path)
    return output_path


def _pretty(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, default=str)
