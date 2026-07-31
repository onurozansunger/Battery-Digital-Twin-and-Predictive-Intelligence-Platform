"""Model explanation: SHAP, permutation importance, native importance, error analysis.

A caveat that is stated in the generated report as well as here: the feature set
is highly collinear by construction (a rolling mean over 5 cycles and over 10
cycles of the same signal are close to redundant). Under collinearity, both SHAP
and permutation importance *distribute* credit among correlated features rather
than identifying a unique cause. The right reading of these figures is at the
level of **signal families** — capacity, resistance, charge timing, temperature —
not individual columns.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.models.base import BaseModel, TrainingData
from battery_rul.utils.logging import get_logger
from battery_rul.visualization.style import figure

logger = get_logger(__name__)

__all__ = ["ExplanationResult", "explain_model", "signal_family"]

#: Maps an engineered column back to the physical signal it derives from.
_SUFFIX_RE = re.compile(
    r"_(rmean|rstd|rmin|rmax|rrange|rdev|ewm|lag|diff|pct|slope|cummean|cummin|cummax|cumstd)_?\d*$"
)
_TAIL_RE = re.compile(r"_(ratio_to_initial|delta_from_initial)$")


def signal_family(feature: str) -> str:
    """Group an engineered feature under its originating physical signal."""
    name = _TAIL_RE.sub("", _SUFFIX_RE.sub("", feature))
    families = {
        "capacity": ("capacity", "soh", "energy", "coulombic"),
        "resistance": ("resistance", "ohm"),
        "voltage": ("voltage", "dvdt"),
        "temperature": ("temperature", "ambient"),
        "charge_timing": ("charge_duration", "cc_", "cv_", "charge_cc", "charge_cv"),
        "discharge_timing": ("discharge_duration", "time_to_min", "discharge_charge"),
        "current": ("current",),
        "cycle_position": ("cycle_index", "cum_"),
    }
    lowered = name.lower()
    for family, needles in families.items():
        if any(needle in lowered for needle in needles):
            return family
    return "other"


@dataclass
class ExplanationResult:
    """Importance rankings and error diagnostics for one model."""

    model_name: str
    native_importance: pd.Series | None = None
    permutation_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    shap_values: np.ndarray | None = field(default=None, repr=False)
    shap_importance: pd.Series | None = None
    family_importance: pd.Series | None = None
    error_analysis: dict[str, Any] = field(default_factory=dict)
    figures: list[Path] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "native_importance_top": _top(self.native_importance),
            "permutation_importance_top": (
                self.permutation_importance.head(25).round(5).to_dict(orient="records")
                if not self.permutation_importance.empty
                else []
            ),
            "shap_importance_top": _top(self.shap_importance),
            "family_importance": _top(self.family_importance, k=20),
            "error_analysis": self.error_analysis,
            "figures": [str(p) for p in self.figures],
        }


def _top(series: pd.Series | None, k: int = 25) -> dict[str, float]:
    if series is None or series.empty:
        return {}
    return {str(name): round(float(value), 6) for name, value in series.head(k).items()}


def explain_model(
    model: BaseModel,
    train: TrainingData,
    test: TrainingData,
    cfg: ExperimentConfig,
    predictions: pd.DataFrame | None = None,
    *,
    outdir: Path | None = None,
) -> ExplanationResult:
    """Run every enabled explanation method and render the figures."""
    outdir = Path(outdir or cfg.paths.figures_dir / "explainability")
    outdir.mkdir(parents=True, exist_ok=True)
    result = ExplanationResult(model_name=model.name)
    ex_cfg = cfg.explainability

    if not ex_cfg.enabled:
        logger.info("Explainability disabled by config")
        return result

    feature_names = list(test.feature_names)

    # -- native importance --------------------------------------------------
    result.native_importance = model.feature_importance()

    # -- permutation importance ----------------------------------------------
    result.permutation_importance = _permutation_importance(model, test, feature_names, cfg)

    # -- SHAP ------------------------------------------------------------------
    if ex_cfg.shap_enabled:
        result.shap_values, result.shap_importance = _shap_values(
            model, train, test, feature_names, cfg
        )

    # -- family roll-up ---------------------------------------------------------
    ranking = result.shap_importance
    if ranking is None or ranking.empty:
        ranking = (
            result.permutation_importance.set_index("feature")["importance_mean"]
            if not result.permutation_importance.empty
            else result.native_importance
        )
    if ranking is not None and not ranking.empty:
        families = pd.Series(
            {name: signal_family(str(name)) for name in ranking.index}, name="family"
        )
        result.family_importance = ranking.groupby(families).sum().sort_values(ascending=False)

    # -- error analysis -----------------------------------------------------------
    if predictions is not None and not predictions.empty:
        result.error_analysis = _error_analysis(predictions, cfg)

    # -- figures -------------------------------------------------------------------
    result.figures = _render(result, test, cfg, outdir)
    return result


# ---------------------------------------------------------------------------
def _permutation_importance(
    model: BaseModel, data: TrainingData, feature_names: list[str], cfg: ExperimentConfig
) -> pd.DataFrame:
    """Model-agnostic importance by shuffling one column at a time.

    Implemented directly rather than via sklearn so it works for the sequence
    models too: the shuffle is applied to the row matrix *before* windowing, so a
    permuted feature is corrupted consistently across every window it appears in.
    Shuffling is done **within each cell** to preserve the marginal distribution
    of a signal that differs systematically between cells.
    """
    from battery_rul.evaluation.metrics import compute_metrics
    from battery_rul.features.target import inverse_transform_target

    if data.is_empty:
        return pd.DataFrame()

    y_true = inverse_transform_target(data.y, cfg)
    baseline_pred = inverse_transform_target(model.predict(data), cfg)
    baseline = compute_metrics(y_true, baseline_pred, mape_epsilon=cfg.evaluation.mape_epsilon)[
        "rmse"
    ]
    if not np.isfinite(baseline):
        return pd.DataFrame()

    rng = np.random.default_rng(cfg.seed)
    battery_ids = data.battery_ids
    repeats = cfg.explainability.permutation_repeats
    rows = []

    for j, name in enumerate(feature_names):
        scores = np.empty(repeats)
        for r in range(repeats):
            corrupted = data.X.copy()
            for battery in np.unique(battery_ids):
                mask = battery_ids == battery
                column = corrupted[mask, j]
                corrupted[mask, j] = rng.permutation(column)
            shuffled = TrainingData(
                X=corrupted, y=data.y, frame=data.frame, feature_names=feature_names
            )
            pred = inverse_transform_target(model.predict(shuffled), cfg)
            scores[r] = compute_metrics(y_true, pred, mape_epsilon=cfg.evaluation.mape_epsilon)[
                "rmse"
            ]
        rows.append(
            {
                "feature": name,
                "importance_mean": float(np.mean(scores) - baseline),
                "importance_std": float(np.std(scores)),
                "baseline_rmse": float(baseline),
            }
        )

    table = (
        pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    )
    logger.info(
        "Permutation importance for %s: top feature %s (+%.3f RMSE)",
        model.name,
        table.iloc[0]["feature"] if len(table) else "—",
        table.iloc[0]["importance_mean"] if len(table) else float("nan"),
    )
    return table


def _shap_values(
    model: BaseModel,
    train: TrainingData,
    test: TrainingData,
    feature_names: list[str],
    cfg: ExperimentConfig,
) -> tuple[np.ndarray | None, pd.Series | None]:
    """SHAP values, using the exact TreeExplainer where possible."""
    try:
        import shap
    except ImportError:  # pragma: no cover
        logger.warning("shap is not installed; skipping SHAP")
        return None, None

    ex_cfg = cfg.explainability
    n = min(len(test), ex_cfg.shap_max_samples)
    if n == 0:
        return None, None
    X = test.X[:n]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model.is_tree and getattr(model, "estimator", None) is not None:
                explainer = shap.TreeExplainer(model.estimator)
                values = explainer.shap_values(X, check_additivity=False)
            elif model.is_sequence:
                # Sequence models take 3-D tensors; explaining them faithfully
                # needs DeepExplainer over windows, which is out of scope for this
                # milestone. Permutation importance covers them instead.
                logger.info(
                    "Skipping SHAP for sequence model %s (permutation used instead)", model.name
                )
                return None, None
            else:
                background = shap.kmeans(
                    train.X, min(ex_cfg.shap_background_samples, max(len(train) // 2, 1))
                )
                # KernelExplainer feeds synthetic coalitions in batches far larger
                # than the evaluation set, so the callback must build metadata to
                # match `arr`, not reuse the test frame. Tabular models ignore the
                # frame entirely; tiling a template keeps TrainingData's row-count
                # invariant satisfied without pretending the ids are meaningful.
                template = test.frame.iloc[:1]

                def _predict(arr: np.ndarray) -> np.ndarray:
                    arr = np.atleast_2d(arr)
                    frame = pd.concat([template] * len(arr), ignore_index=True)
                    frame["cycle_index"] = np.arange(1, len(arr) + 1)
                    return model.predict(
                        TrainingData(
                            X=arr,
                            y=np.zeros(len(arr)),
                            frame=frame,
                            feature_names=feature_names,
                        )
                    )

                explainer = shap.KernelExplainer(_predict, background)
                values = explainer.shap_values(X[: min(n, 200)], nsamples=100, silent=True)
    except Exception as exc:  # noqa: BLE001 - explainability must never break a run
        logger.warning("SHAP failed for %s: %s", model.name, exc)
        return None, None

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., 0]
    importance = pd.Series(
        np.abs(values).mean(axis=0), index=feature_names[: values.shape[1]]
    ).sort_values(ascending=False)
    logger.info("SHAP computed for %s over %d samples", model.name, values.shape[0])
    return values, importance


def _error_analysis(predictions: pd.DataFrame, cfg: ExperimentConfig) -> dict[str, Any]:
    """Where does the model fail, and is the failure structured?"""
    data = predictions.dropna(subset=["y_pred"]).copy()
    if data.empty:
        return {}

    threshold = float(data["abs_error"].quantile(cfg.explainability.error_analysis_quantile))
    worst = data[data["abs_error"] >= threshold]

    # Bin by remaining life: errors concentrated far from EOL are tolerable,
    # errors close to EOL are the ones that cost money.
    bins = [0, 10, 25, 50, 100, np.inf]
    labels = ["0–10", "10–25", "25–50", "50–100", "100+"]
    data["rul_band"] = pd.cut(data["y_true"], bins=bins, labels=labels, right=False)
    by_band = (
        data.groupby("rul_band", observed=True)
        .agg(
            n=("abs_error", "size"),
            mae=("abs_error", "mean"),
            bias=("residual", "mean"),
            p90_abs_error=("abs_error", lambda s: float(np.quantile(s, 0.9))),
        )
        .round(3)
        .reset_index()
    )

    return {
        "error_quantile": cfg.explainability.error_analysis_quantile,
        "abs_error_threshold": round(threshold, 3),
        "n_worst_rows": int(len(worst)),
        "worst_batteries": worst["battery_id"].value_counts().to_dict(),
        "worst_mean_true_rul": round(float(worst["y_true"].mean()), 2),
        "overall_mean_true_rul": round(float(data["y_true"].mean()), 2),
        "by_rul_band": by_band.to_dict(orient="records"),
        "worst_rows": worst.nlargest(15, "abs_error")[
            ["battery_id", "cycle_index", "y_true", "y_pred", "abs_error"]
        ]
        .round(2)
        .to_dict(orient="records"),
    }


def _render(
    result: ExplanationResult, test: TrainingData, cfg: ExperimentConfig, outdir: Path
) -> list[Path]:
    paths: list[Path] = []
    suffix = cfg.viz.figure_format
    k = cfg.explainability.top_k_features

    # -- importance comparison ------------------------------------------------
    panels = [
        ("Native importance", result.native_importance),
        (
            "Permutation importance (ΔRMSE)",
            (
                result.permutation_importance.set_index("feature")["importance_mean"]
                if not result.permutation_importance.empty
                else None
            ),
        ),
        ("Mean |SHAP|", result.shap_importance),
    ]
    panels = [(t, s) for t, s in panels if s is not None and not s.empty]
    if panels:
        path = outdir / f"feature_importance_{result.model_name}.{suffix}"
        with figure(
            nrows=1,
            ncols=len(panels),
            figsize=(6.0 * len(panels), 0.28 * k + 2.5),
            path=path,
            cfg=cfg.viz,
            squeeze=False,
        ) as (fig, axes):
            for ax, (title, series) in zip(axes.ravel(), panels, strict=True):
                top = series.head(k).iloc[::-1]
                ax.barh(
                    [str(i)[:44] for i in top.index],
                    top.to_numpy(),
                    color="#0072B2",
                    alpha=0.9,
                )
                ax.set_title(title)
                ax.tick_params(axis="y", labelsize=7)
                ax.grid(axis="y", visible=False)
            fig.suptitle(
                f"{result.model_name} — feature importance (three independent views)",
                fontsize=14,
                fontweight="semibold",
            )
        paths.append(path)

    # -- family roll-up -------------------------------------------------------
    if result.family_importance is not None and not result.family_importance.empty:
        path = outdir / f"signal_family_importance_{result.model_name}.{suffix}"
        with figure(nrows=1, ncols=1, figsize=(8.5, 5), path=path, cfg=cfg.viz) as (fig, ax):
            top = result.family_importance.iloc[::-1]
            bars = ax.barh([str(i) for i in top.index], top.to_numpy(), color="#009E73", alpha=0.9)
            ax.bar_label(bars, fmt="%.3g", fontsize=8, padding=3)
            ax.set_title(
                f"{result.model_name} — importance aggregated by physical signal family\n"
                "(the collinearity-robust reading of the rankings above)"
            )
            ax.grid(axis="y", visible=False)
        paths.append(path)

    # -- SHAP beeswarm / summary ------------------------------------------------
    if result.shap_values is not None:
        try:
            import shap

            path = outdir / f"shap_summary_{result.model_name}.{suffix}"
            import matplotlib.pyplot as plt

            from battery_rul.visualization.style import apply_style, save_figure

            apply_style(cfg.viz)
            n = result.shap_values.shape[0]
            frame = pd.DataFrame(
                test.X[:n, : result.shap_values.shape[1]],
                columns=test.feature_names[: result.shap_values.shape[1]],
            )
            plt.figure(figsize=(10, 0.26 * k + 3))
            shap.summary_plot(result.shap_values, frame, max_display=k, show=False, plot_size=None)
            fig = plt.gcf()
            fig.suptitle(
                f"{result.model_name} — SHAP value distribution", fontsize=13, fontweight="semibold"
            )
            save_figure(fig, path, cfg.viz)
            plt.close(fig)
            paths.append(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SHAP summary plot failed: %s", exc)

    # -- error analysis ----------------------------------------------------------
    bands = result.error_analysis.get("by_rul_band")
    if bands:
        path = outdir / f"error_by_rul_band_{result.model_name}.{suffix}"
        table = pd.DataFrame(bands)
        with figure(nrows=1, ncols=2, figsize=(12, 5), path=path, cfg=cfg.viz) as (fig, axes):
            ax0, ax1 = axes
            bars = ax0.bar(table["rul_band"].astype(str), table["mae"], color="#D55E00", alpha=0.9)
            ax0.bar_label(bars, fmt="%.1f", fontsize=9, padding=2)
            ax0.set_title("MAE by remaining-life band")
            ax0.set_xlabel("True RUL (cycles)")
            ax0.set_ylabel("MAE (cycles)")
            ax0.grid(axis="x", visible=False)

            colours = ["#B00020" if v > 0 else "#0072B2" for v in table["bias"]]
            bars = ax1.bar(table["rul_band"].astype(str), table["bias"], color=colours, alpha=0.9)
            ax1.bar_label(bars, fmt="%+.1f", fontsize=9, padding=2)
            ax1.axhline(0, color="#333333", lw=1)
            ax1.set_title("Bias by band — red = optimistic (predicts too much life)")
            ax1.set_xlabel("True RUL (cycles)")
            ax1.set_ylabel("Mean signed error (cycles)")
            ax1.grid(axis="x", visible=False)
            fig.suptitle(
                f"{result.model_name} — error structure across the life curve",
                fontsize=14,
                fontweight="semibold",
            )
        paths.append(path)

    return paths
