"""Typed, validated, YAML-backed configuration for the battery RUL platform.

Every tunable in the project lives here. Nothing in ``src/`` reads an environment
variable or hardcodes a filesystem path: pipelines receive an
:class:`ExperimentConfig` and everything downstream is derived from it.

Usage
-----
>>> from battery_rul.config import load_config
>>> cfg = load_config("configs/default.yaml", overrides={"training.epochs": 5})
>>> cfg.paths.processed_dir
PosixPath('data/processed')
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "AlertPolicyConfig",
    "ArtifactConfig",
    "CalibrationConfig",
    "DataConfig",
    "DataQualityConfig",
    "DeploymentConfig",
    "DriftConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ExplainabilityConfig",
    "FeatureConfig",
    "FleetConfig",
    "FleetRankingConfig",
    "MaintenancePolicyConfig",
    "ModelZooConfig",
    "MonitoringConfig",
    "MultiTaskConfig",
    "PathsConfig",
    "PerformanceMonitoringConfig",
    "PersistenceConfig",
    "PredictionDriftConfig",
    "PromotionGateConfig",
    "RecommendationConfig",
    "RegistryConfig",
    "ReplacementPlanningConfig",
    "RiskConfig",
    "SOHConfig",
    "ServiceConfig",
    "SplitConfig",
    "TargetConfig",
    "TrackingConfig",
    "TrainingConfig",
    "TuningConfig",
    "UncertaintyConfig",
    "ValidationConfig",
    "WorkloadForecastConfig",
    "load_config",
    "project_root",
]


def project_root() -> Path:
    """Return the repository root.

    Resolved relative to this file (``src/battery_rul/config.py`` -> two levels up
    from ``src``), so the package behaves identically whether it is invoked from
    the repo root, from ``scripts/``, or from an installed wheel used in-place.
    An explicit ``BATTERY_RUL_ROOT`` environment variable wins when present, which
    is what container and CI runners use.
    """
    env = os.environ.get("BATTERY_RUL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


class _Base(BaseModel):
    """Strict base model: unknown keys are configuration bugs, not features."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=False)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
class PathsConfig(_Base):
    """All filesystem locations, expressed relative to the project root."""

    root: Path = Field(default_factory=project_root)
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    figures_dir: Path = Path("figures")
    reports_dir: Path = Path("reports")
    artifacts_dir: Path = Path("reports/artifacts")

    @model_validator(mode="after")
    def _absolutise(self) -> PathsConfig:
        root = Path(self.root).expanduser().resolve()
        object.__setattr__(self, "root", root)
        for field in (
            "raw_dir",
            "interim_dir",
            "processed_dir",
            "models_dir",
            "figures_dir",
            "reports_dir",
            "artifacts_dir",
        ):
            value = Path(getattr(self, field))
            if not value.is_absolute():
                value = root / value
            object.__setattr__(self, field, value)
        return self

    def ensure(self) -> None:
        """Create every configured directory. Idempotent, and best-effort.

        A directory that cannot be created is logged and skipped rather than
        raising. The reason is containers: a job running with a read-only root
        filesystem and two writable mounts genuinely cannot create ``data/raw``,
        and it does not need to — a monitoring run reads processed cycles and
        writes reports. Failing here would stop that run for a directory it
        never touches.

        This is not a silent fallback. Nothing is created that was not asked
        for, and the *use* of a missing directory still fails loudly at the
        write, with a message naming the actual path.
        """
        import logging

        for field in self.model_fields:
            path = getattr(self, field)
            if not isinstance(path, Path):
                continue
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logging.getLogger(__name__).warning(
                    "Could not create %s (%s). Continuing: this deployment may not "
                    "need it. Any write to it will fail with this path named.",
                    path,
                    exc.strerror or exc,
                )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class DataConfig(_Base):
    """Which dataset to ingest and how strictly to police it."""

    source: str = Field(
        default="nasa",
        description="Registered data-source key. See battery_rul.data.registry.",
    )
    subdir: str = Field(default="nasa", description="Sub-folder of raw_dir holding the source.")
    batteries: list[str] | None = Field(
        default=None,
        description="Explicit battery whitelist. None means 'every battery discovered'.",
    )
    exclude_batteries: list[str] = Field(default_factory=list)
    min_cycles: int = Field(default=30, ge=1, description="Drop batteries shorter than this.")
    nominal_capacity_ah: float = Field(default=2.0, gt=0.0)
    eol_threshold: float = Field(
        default=0.70,
        gt=0.0,
        lt=1.0,
        description="End-of-life as a fraction of the reference capacity (0.70 => 70 % SoH).",
    )
    eol_reference: Literal["nominal", "initial"] = Field(
        default="nominal",
        description="Denominator for SoH and the EOL threshold. 'nominal' uses the "
        "manufacturer rating (the convention in the NASA-dataset literature); "
        "'initial' uses each cell's own beginning-of-life capacity, which is fairer "
        "across a heterogeneous fleet whose cells did not all start healthy.",
    )
    capacity_smoothing_window: int = Field(
        default=5, ge=1, description="Median-filter window used to de-noise the capacity series."
    )
    trim_leading_outliers: bool = Field(
        default=True,
        description="Drop leading cycles whose capacity sits far below the cell's "
        "early-life level. On the NASA rig the first one to seven discharges of "
        "some cells are partial or aborted runs, not degradation.",
    )
    leading_outlier_ratio: float = Field(
        default=0.90,
        gt=0.0,
        le=1.0,
        description="A leading cycle is an artifact if its capacity is below this "
        "fraction of the median of the following `leading_outlier_lookahead` cycles.",
    )
    leading_outlier_lookahead: int = Field(default=10, ge=2)
    min_start_soh: float = Field(
        default=0.80,
        ge=0.0,
        le=1.5,
        description="Reject cells that are already degraded at beginning of life. "
        "RUL is undefined for a cell that starts below (or barely above) its own "
        "end-of-life threshold.",
    )
    min_fade_fraction: float = Field(
        default=0.02,
        ge=0.0,
        description="Reject cells whose capacity never falls by at least this "
        "fraction of reference — a flat series carries no degradation signal.",
    )
    truncate_at_collapse: bool = Field(
        default=True,
        description="End a cell's record where its capacity collapses to a level "
        "no ageing cell reaches gradually. On the NASA rig, cells moved into a 4 degC "
        "chamber mid-experiment report ~0.07 Ah — the discharge test aborts almost "
        "immediately — which is a regime change, not end of life.",
    )
    collapse_fraction: float = Field(
        default=0.25,
        gt=0.0,
        lt=1.0,
        description="Capacity below this fraction of the reference marks a collapse.",
    )
    collapse_persistence: int = Field(
        default=5,
        ge=1,
        description="Cycles the collapse must persist before the record is truncated. "
        "Guards against truncating on a single bad reading.",
    )
    drop_incomplete_cycles: bool = True
    allow_synthetic_fallback: bool = Field(
        default=True,
        description="If the raw source is missing, generate a physics-informed surrogate "
        "so the pipeline remains runnable. Always logged loudly.",
    )
    cache_interim: bool = True

    @field_validator("source")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.strip().lower()


class ValidationConfig(_Base):
    """Thresholds for the data-integrity gate."""

    max_missing_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    max_capacity_jump: float = Field(
        default=0.35,
        gt=0.0,
        description="Reject a cycle whose capacity moves more than this fraction "
        "of nominal in a single step (sensor glitch).",
    )
    voltage_bounds_v: tuple[float, float] = (0.0, 5.0)
    temperature_bounds_c: tuple[float, float] = (-40.0, 100.0)
    require_monotonic_cycle_index: bool = True
    fail_fast: bool = Field(
        default=False, description="Raise on violation instead of quarantining rows."
    )
    missingness_indicators: bool = Field(
        default=True,
        description="Emit a ``<column>_is_missing`` flag for every sensor column that "
        "has at least one missing observation. Absence of a reading is itself "
        "information (an EIS sweep that was never run, an aborted charge step), and "
        "an imputed value hides it.",
    )
    indicator_columns: list[str] | None = Field(
        default=None,
        description="Restrict missingness indicators to these columns. None means "
        "'every numeric column that has a missing value'.",
    )


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
class FeatureConfig(_Base):
    """Feature-engineering knobs.

    Every window here is *causal*: at cycle *k* only cycles ``<= k`` are read.
    See :mod:`battery_rul.features.engineering` for the leakage guarantees.
    """

    rolling_windows: list[int] = Field(default_factory=lambda: [3, 5, 10, 20])
    lags: list[int] = Field(default_factory=lambda: [1, 2, 3, 5, 10])
    slope_windows: list[int] = Field(default_factory=lambda: [5, 10, 20])
    ewm_halflives: list[int] = Field(default_factory=lambda: [5, 15])
    base_signals: list[str] = Field(
        default_factory=lambda: [
            "capacity_ah",
            "soh",
            "discharge_duration_s",
            "charge_duration_s",
            "voltage_mean_v",
            "voltage_min_v",
            "voltage_std_v",
            "voltage_slope_v_per_s",
            "current_mean_a",
            "temperature_mean_c",
            "temperature_max_c",
            "internal_resistance_ohm",
            "energy_throughput_wh",
            "cc_ct_ratio",
        ]
    )
    include_cycle_index: bool = True
    include_cumulative: bool = True
    drop_warmup_cycles: int = Field(
        default=10,
        ge=0,
        description="Discard the first N cycles per battery: their rolling windows "
        "are only partially populated and would inject survivorship noise.",
    )
    min_periods_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    variance_threshold: float = Field(default=1e-10, ge=0.0)
    correlation_prune_threshold: float | None = Field(
        default=0.98,
        description="Drop one of any feature pair above |rho|. None disables. Fitted "
        "on the TRAINING partition only, inside FeaturePipeline — see Milestone 1.1.",
    )
    fallback_imputation: Literal["median", "mean", "zero"] = Field(
        default="median",
        description="Fleet-level fallback for a feature that has no observed value "
        "anywhere in a cell's history. Learned from TRAINING rows only, persisted "
        "with the fitted pipeline and replayed unchanged at validation, test and "
        "serving time.",
    )
    scaler: Literal["standard", "robust", "minmax", "none"] = "robust"
    max_features: int | None = Field(
        default=80,
        description="Keep only the top-K features by supervised importance. Fitted "
        "on the TRAINING partition only (inside FeaturePipeline), never on the full "
        "dataset. None disables selection.",
    )
    selection_method: Literal["tree_importance", "mutual_info", "f_regression"] = "tree_importance"

    @field_validator("rolling_windows", "lags", "slope_windows", "ewm_halflives")
    @classmethod
    def _positive_sorted(cls, v: list[int]) -> list[int]:
        if any(x < 1 for x in v):
            raise ValueError("window/lag values must be >= 1")
        return sorted(set(v))


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
class TargetConfig(_Base):
    """Definition of the supervised target.

    RUL(k) = EOL_cycle - k, where EOL_cycle is the first cycle at which the
    smoothed capacity falls to or below ``eol_threshold * nominal_capacity`` and
    stays there. Full derivation in :mod:`battery_rul.features.target`.
    """

    name: str = "rul_cycles"
    eol_persistence: int = Field(
        default=3,
        ge=1,
        description="Complete consecutive observations at or below the end-of-life "
        "threshold required for a crossing to count. A record that ends before "
        "this many observations exist is right-censored, not confirmed end of life.",
    )
    clip_negative: bool = True
    cap_at: int | None = Field(
        default=None,
        description="Optional upper clip on RUL (piecewise-linear target, common in "
        "prognostics literature). None keeps the raw remaining-cycle count.",
    )
    log_transform: bool = Field(
        default=False, description="Train on log1p(RUL) and invert at predict time."
    )
    drop_post_eol: bool = Field(
        default=True, description="Discard cycles recorded after end-of-life is reached."
    )
    require_eol_reached: bool = Field(
        default=True,
        description="Only keep batteries that actually reach EOL; otherwise RUL is "
        "right-censored and unlearnable without survival modelling.",
    )
    min_labelled_cycles: int = Field(
        default=25,
        ge=1,
        description="Minimum labelled (pre-EOL) rows a cell must contribute. Cells "
        "that cross the threshold almost immediately — the cold-chamber cells whose "
        "*delivered* capacity is low without the cell being worn out — carry a few "
        "unrepresentative rows and distort both training and the holdout metric.",
    )


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
class SplitConfig(_Base):
    """Leakage-free partitioning.

    ``strategy``:
      * ``battery_holdout`` — entire batteries are held out (the honest setting
        for "will this new cell survive?").
      * ``chronological`` — per battery, the first p % of cycles train, the tail tests.
      * ``walk_forward`` — expanding-origin cross-validation over cycles.
    """

    strategy: Literal["battery_holdout", "chronological", "walk_forward"] = "battery_holdout"
    test_batteries: list[str] | None = None
    val_batteries: list[str] | None = None
    test_size: float = Field(default=0.25, gt=0.0, lt=1.0)
    val_size: float = Field(default=0.20, ge=0.0, lt=1.0)
    n_folds: int = Field(default=5, ge=2)
    walk_forward_min_train_fraction: float = Field(default=0.4, gt=0.0, lt=1.0)
    walk_forward_horizon: int = Field(default=50, ge=1)
    gap_cycles: int = Field(
        default=0,
        ge=0,
        description="Purge window between train and test in chronological mode; "
        "guards against rolling-window bleed across the boundary.",
    )
    seed: int = 42


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class SequenceConfig(_Base):
    """Windowing for the sequence models (LSTM / GRU / Transformer)."""

    window: int = Field(default=20, ge=2)
    stride: int = Field(default=1, ge=1)
    hidden_size: int = Field(default=96, ge=8)
    num_layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.15, ge=0.0, lt=1.0)
    bidirectional: bool = False
    # Transformer-only
    d_model: int = Field(default=96, ge=8)
    nhead: int = Field(default=4, ge=1)
    dim_feedforward: int = Field(default=192, ge=8)

    @model_validator(mode="after")
    def _head_divides_dmodel(self) -> SequenceConfig:
        if self.d_model % self.nhead:
            raise ValueError(f"d_model={self.d_model} must be divisible by nhead={self.nhead}")
        return self


class TrainingConfig(_Base):
    """Optimisation settings shared by the neural models."""

    epochs: int = Field(default=80, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    grad_clip: float | None = 1.0
    early_stopping_patience: int = Field(default=12, ge=1)
    lr_scheduler: Literal["none", "cosine", "plateau"] = "plateau"
    loss: Literal["mse", "huber", "mae"] = "huber"
    huber_delta: float = Field(default=1.0, gt=0.0)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    num_workers: int = Field(default=0, ge=0)
    seed: int = 42


class ModelZooConfig(_Base):
    """Which estimators to train and their (pre-tuning) hyperparameters."""

    enabled: list[str] = Field(
        default_factory=lambda: [
            "linear_regression",
            "ridge",
            "random_forest",
            "xgboost",
            "lightgbm",
            "catboost",
            "lstm",
            "gru",
            "transformer",
        ]
    )
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    sequence: SequenceConfig = Field(default_factory=SequenceConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    select_by: str = Field(default="rmse", description="Metric used to crown the champion.")
    select_mode: Literal["min", "max"] = "min"

    def params_for(self, name: str) -> dict[str, Any]:
        return copy.deepcopy(self.params.get(name, {}))


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
class TuningConfig(_Base):
    """Optuna study configuration. Search spaces live in
    :mod:`battery_rul.models.search_spaces` so they are versioned with the code."""

    enabled: bool = False
    models: list[str] = Field(default_factory=lambda: ["xgboost", "lightgbm", "random_forest"])
    n_trials: int = Field(default=40, ge=1)
    timeout_s: int | None = 1800
    direction: Literal["minimize", "maximize"] = "minimize"
    metric: str = "rmse"
    sampler: Literal["tpe", "random", "cmaes"] = "tpe"
    pruner: Literal["median", "hyperband", "none"] = "median"
    cv_folds: int = Field(default=3, ge=2)
    study_name: str = "battery_rul"
    storage: str | None = None
    seed: int = 42


# ---------------------------------------------------------------------------
# Evaluation / explainability
# ---------------------------------------------------------------------------
class EvaluationConfig(_Base):
    metrics: list[str] = Field(
        default_factory=lambda: ["mae", "rmse", "mape", "smape", "r2", "max_error", "alpha_lambda"]
    )
    mape_epsilon: float = Field(
        default=1.0, gt=0.0, description="Floor on the denominator so RUL->0 doesn't blow MAPE up."
    )
    alpha: float = Field(
        default=0.20, gt=0.0, lt=1.0, description="alpha-lambda prognostics accuracy band."
    )
    residual_bins: int = Field(default=40, ge=5)
    bootstrap_samples: int = Field(default=1000, ge=0)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    per_battery_breakdown: bool = True
    learning_curve_fractions: list[float] = Field(
        default_factory=lambda: [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
    )
    nested_enabled: bool = Field(
        default=True,
        description="Run the nested leave-one-battery-out comparison. This is what "
        "makes the headline metric an estimate of the whole procedure rather than "
        "of an already-selected model.",
    )
    nested_candidates: list[str] = Field(
        default_factory=lambda: [
            "cohort_median_life",
            "capacity_fade_extrapolation",
            "soh_analogue",
            "elastic_net",
            "ridge",
            "random_forest",
            "xgboost",
            "lightgbm",
            "gru",
            "transformer",
        ],
        description="Families compared under identical outer folds. Baselines are "
        "not optional here: a learned model that cannot beat them has not earned "
        "its complexity.",
    )
    nested_select_by: str = Field(
        default="mae", description="Metric the inner selection loop optimises."
    )


class ExplainabilityConfig(_Base):
    enabled: bool = True
    shap_enabled: bool = True
    shap_max_samples: int = Field(default=2000, ge=50)
    shap_background_samples: int = Field(default=200, ge=10)
    permutation_repeats: int = Field(default=15, ge=1)
    top_k_features: int = Field(default=25, ge=1)
    error_analysis_quantile: float = Field(default=0.90, gt=0.0, lt=1.0)


# ---------------------------------------------------------------------------
# Milestone 2 — digital twin
# ---------------------------------------------------------------------------
class SOHConfig(_Base):
    """State-of-health definition and health banding.

    The internal representation is a **fraction in [0, 1]** everywhere in the
    codebase; percentages exist only in rendered text. See docs/SOH_DEFINITION.md.
    """

    reference_strategy: Literal["nominal", "first_cycle", "first_n_cycle_mean"] = Field(
        default="first_n_cycle_mean",
        description="Denominator of SOH. 'nominal' uses the manufacturer rating; "
        "'first_cycle' the cell's own first valid measurement; 'first_n_cycle_mean' "
        "the mean of its first N valid cycles (robust to one bad opening reading).",
    )
    reference_cycles: int = Field(default=5, ge=1)
    target_name: str = "soh_target"
    forecast_horizon_cycles: int = Field(
        default=30,
        ge=1,
        description="The SOH *model* forecasts SOH this many cycles ahead. Modelling "
        "SOH at the current cycle is not a prediction problem: the target is "
        "measured capacity over a per-cell constant, and measured capacity is an "
        "input, so the model would learn a rescaling and report a flattering error. "
        "The twin reports current SOH deterministically and uses the model only for "
        "the forecast.",
    )
    forecast_target_name: str = "soh_target_future"
    #: Health bands, expressed as fractions. Read as: class C applies when
    #: ``lower <= SOH < upper`` of the band above it. Demonstration values.
    healthy_min: float = Field(default=0.90, gt=0.0, le=1.5)
    slightly_degraded_min: float = Field(default=0.80, gt=0.0, le=1.5)
    warning_min: float = Field(default=0.70, gt=0.0, le=1.5)
    plausible_min: float = Field(default=0.20, ge=0.0, description="Physical clip, lower.")
    plausible_max: float = Field(default=1.20, gt=0.0, description="Physical clip, upper.")

    @model_validator(mode="after")
    def _ordered(self) -> SOHConfig:
        if not (self.warning_min < self.slightly_degraded_min < self.healthy_min):
            raise ValueError(
                "SOH bands must satisfy warning_min < slightly_degraded_min < healthy_min, "
                f"got {self.warning_min} / {self.slightly_degraded_min} / {self.healthy_min}"
            )
        if self.plausible_min >= self.plausible_max:
            raise ValueError("soh.plausible_min must be below soh.plausible_max")
        return self


class RiskConfig(_Base):
    """Future end-of-life risk target, model and decision threshold."""

    horizon_cycles: int = Field(
        default=30, ge=1, description="H in 'reaches end of life within the next H cycles'."
    )
    additional_horizons: list[int] = Field(default_factory=lambda: [20, 50])
    target_name: str = "failure_within_horizon"
    model: Literal["auto", "logistic_regression", "random_forest", "lightgbm", "xgboost"] = Field(
        default="auto",
        description="Classifier family. 'auto' selects the highest PR-AUC family by "
        "leave-one-battery-out predictions over non-test cells only.",
    )
    candidates: list[Literal["logistic_regression", "random_forest", "lightgbm", "xgboost"]] = (
        Field(
            default=[
                "logistic_regression",
                "random_forest",
                "lightgbm",
                "xgboost",
            ],
            min_length=1,
            description="Families compared when model='auto'. The held-out test cell never "
            "participates in family selection.",
        )
    )
    class_weight_balanced: bool = True
    threshold: float | None = Field(
        default=None,
        description="Decision threshold. None means 'tune on the validation/calibration "
        "partition by the configured objective'. Never tuned on test.",
    )
    threshold_objective: Literal["f1", "youden", "precision_at_recall"] = "f1"
    min_recall: float = Field(default=0.80, gt=0.0, le=1.0)
    #: Risk banding on the calibrated probability.
    low_max: float = Field(default=0.20, gt=0.0, lt=1.0)
    medium_max: float = Field(default=0.50, gt=0.0, lt=1.0)
    high_max: float = Field(default=0.80, gt=0.0, lt=1.0)
    require_beating_baseline: bool = Field(
        default=True,
        description="Refuse to let the risk probability drive recommendations unless "
        "it beats the cycle-index baseline out of fold. Because the label is "
        "'RUL <= H', cycle index ranks a cell's positives perfectly, so a model that "
        "loses to it has demonstrated nothing — and a demonstrated-nothing model must "
        "not trigger an inspection or a replacement. It is still reported, marked "
        "experimental.",
    )

    @model_validator(mode="after")
    def _ordered(self) -> RiskConfig:
        if not (self.low_max < self.medium_max < self.high_max):
            raise ValueError("risk bands must satisfy low_max < medium_max < high_max")
        if len(self.candidates) != len(set(self.candidates)):
            raise ValueError("risk.candidates must not contain duplicates")
        return self


class MultiTaskConfig(_Base):
    """Shared sequence encoder with RUL / SOH / risk heads."""

    enabled: bool = True
    encoder: Literal["transformer", "lstm", "gru"] = "transformer"
    window: int = Field(default=20, ge=2)
    stride: int = Field(default=1, ge=1)
    d_model: int = Field(default=96, ge=8)
    nhead: int = Field(default=4, ge=1)
    num_layers: int = Field(default=2, ge=1)
    dim_feedforward: int = Field(default=192, ge=8)
    hidden_size: int = Field(default=96, ge=8)
    dropout: float = Field(default=0.15, ge=0.0, lt=1.0)
    head_hidden: int = Field(default=64, ge=4)
    rul_loss: Literal["mae", "mse", "huber"] = "huber"
    soh_loss: Literal["mae", "mse", "huber"] = "huber"
    risk_loss: Literal["bce", "focal"] = "bce"
    focal_gamma: float = Field(default=2.0, ge=0.0)
    rul_weight: float = Field(default=1.0, ge=0.0)
    soh_weight: float = Field(default=1.0, ge=0.0)
    risk_weight: float = Field(default=0.5, ge=0.0)
    #: RUL is scaled by this before the loss so the three tasks are commensurate.
    rul_scale: float = Field(default=100.0, gt=0.0)
    epochs: int = Field(default=120, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=2e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    grad_clip: float | None = 1.0
    early_stopping_patience: int = Field(default=20, ge=1)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    seed: int = 42

    @model_validator(mode="after")
    def _head_divides_dmodel(self) -> MultiTaskConfig:
        if self.encoder == "transformer" and self.d_model % self.nhead:
            raise ValueError(f"d_model={self.d_model} must be divisible by nhead={self.nhead}")
        if self.rul_weight + self.soh_weight + self.risk_weight <= 0:
            raise ValueError("At least one multitask loss weight must be positive")
        return self


class UncertaintyConfig(_Base):
    """Prediction-interval construction for RUL (and optionally SOH)."""

    method: Literal["split_conformal", "none"] = "split_conformal"
    coverage: float = Field(default=0.90, gt=0.0, lt=1.0)
    calibration_unit: Literal["row", "battery"] = Field(
        default="battery",
        description="Exchangeability unit used to calibrate residuals. 'battery' "
        "uses the worst residual per held-out cell, so hundreds of autocorrelated "
        "cycles cannot masquerade as hundreds of independent calibration examples. "
        "'row' is retained for genuinely exchangeable tabular data and unit tests.",
    )
    normalise_by_life_stage: bool = Field(
        default=True,
        description="Fit separate conformal quantiles per degradation-stage bucket. "
        "RUL residuals are strongly heteroscedastic — wide early in life, tight near "
        "end of life — so one global quantile is too loose where it matters most.",
    )
    stage_edges: list[float] = Field(
        default_factory=lambda: [0.90, 0.80],
        description="**State-of-health** cut points, descending, defining the "
        "conditioning buckets: early (SOH >= first), mid (>= second), late (below). "
        "SOH is used rather than life fraction on purpose — life fraction needs the "
        "end-of-life cycle, which is a label, so conditioning on it would work in "
        "evaluation and silently fall back to a single global quantile at serving "
        "time, exactly where the interval matters.",
    )
    min_calibration_rows: int = Field(default=20, ge=5)
    min_calibration_batteries: int = Field(
        default=3,
        ge=2,
        description="Minimum independent cells required for a stage-specific "
        "battery-block quantile. Thinner stages fall back to the global cell-block "
        "quantile instead of claiming group-conditional evidence from one cell.",
    )
    cross_conformal_coverage: bool = Field(
        default=True,
        description="Estimate coverage with leave-one-cell-out conformal rather than "
        "on the rows the quantile was fitted from. Fitting the quantile on a set and "
        "then measuring coverage on that same set is close to circular — it recovers "
        "the nominal level by construction and says nothing about a new cell.",
    )


class CalibrationConfig(_Base):
    """Probability calibration for the failure-risk classifier."""

    method: Literal["isotonic", "platt", "none"] = "isotonic"
    min_calibration_rows: int = Field(default=30, ge=10)
    n_bins: int = Field(default=10, ge=2, description="Reliability-diagram / ECE bins.")


class DataQualityConfig(_Base):
    """Thresholds for the input-quality gate applied at serving time."""

    min_cycles_for_prediction: int = Field(default=15, ge=1)
    min_cycles_good: int = Field(default=40, ge=1)
    max_missing_fraction_acceptable: float = Field(default=0.30, gt=0.0, le=1.0)
    max_missing_fraction_good: float = Field(default=0.05, ge=0.0, le=1.0)
    max_cycle_gap: int = Field(default=25, ge=1)
    good_min_score: float = Field(default=0.80, gt=0.0, le=1.0)
    acceptable_min_score: float = Field(default=0.60, gt=0.0, le=1.0)
    poor_min_score: float = Field(default=0.35, gt=0.0, le=1.0)
    ood_z_threshold: float = Field(default=5.0, gt=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> DataQualityConfig:
        if not (self.poor_min_score < self.acceptable_min_score < self.good_min_score):
            raise ValueError("quality score thresholds must be strictly increasing")
        return self


class RecommendationConfig(_Base):
    """Deterministic rule thresholds for the decision-support layer.

    These are engineering rules, not model outputs, and they live outside the
    model on purpose: they must be auditable and changeable without retraining.
    """

    inspection_rul_cycles: int = Field(default=40, ge=1)
    replacement_rul_cycles: int = Field(default=15, ge=1)
    urgent_rul_cycles: int = Field(default=5, ge=1)
    inspection_risk: float = Field(default=0.30, gt=0.0, lt=1.0)
    replacement_risk: float = Field(default=0.60, gt=0.0, lt=1.0)
    urgent_risk: float = Field(default=0.85, gt=0.0, lt=1.0)
    temperature_trend_c_per_10_cycles: float = Field(default=0.5, gt=0.0)
    resistance_trend_pct_per_10_cycles: float = Field(default=2.0, gt=0.0)
    fade_trend_pct_per_10_cycles: float = Field(default=1.0, gt=0.0)
    inspection_window_cycles: tuple[int, int] = (10, 20)
    disclaimer: str = (
        "Engineering decision support from a research prototype. Not a substitute "
        "for battery-management-system protection and not validated for "
        "safety-critical or autonomous use. Review by a qualified engineer is required."
    )


class ArtifactConfig(_Base):
    """Where Milestone 2 model bundles live and how strictly they are policed."""

    root: Path = Path("artifacts")
    rul_dir: Path = Path("artifacts/rul")
    soh_dir: Path = Path("artifacts/soh")
    risk_dir: Path = Path("artifacts/risk")
    multitask_dir: Path = Path("artifacts/multitask")
    calibration_dir: Path = Path("artifacts/calibration")
    bundle_dir: Path = Path("artifacts/bundles")
    # -- milestone 3 ------------------------------------------------------
    registry_dir: Path = Path("artifacts/registry")
    monitoring_dir: Path = Path("artifacts/monitoring")
    fleet_dir: Path = Path("artifacts/fleet")
    tracking_dir: Path = Path("artifacts/tracking")
    persistence_dir: Path = Path("artifacts/persistence")
    schema_version: str = "2.0"
    strict_compatibility: bool = Field(
        default=True,
        description="Refuse to serve a bundle whose persisted training configuration "
        "disagrees with the runtime configuration on any data-affecting field.",
    )

    #: Directory fields, resolved against ``paths.root`` by ExperimentConfig.
    _DIR_FIELDS: ClassVar[tuple[str, ...]] = (
        "root",
        "rul_dir",
        "soh_dir",
        "risk_dir",
        "multitask_dir",
        "calibration_dir",
        "bundle_dir",
        "registry_dir",
        "monitoring_dir",
        "fleet_dir",
        "tracking_dir",
        "persistence_dir",
    )

    def resolve_against(self, root: Path) -> None:
        """Make every relative artifact path absolute under ``root``.

        Called by :class:`ExperimentConfig` rather than by a validator here,
        because the anchor is ``paths.root`` — which a test or a CI runner
        overrides. Resolving against the repository root instead would silently
        point a temp-directory experiment at the developer's real artifacts,
        which is exactly the kind of cross-contamination the split discipline
        exists to prevent.
        """
        for field_name in self._DIR_FIELDS:
            value = Path(getattr(self, field_name))
            if not value.is_absolute():
                object.__setattr__(self, field_name, root / value)

    def ensure(self) -> None:
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, Path):
                value.mkdir(parents=True, exist_ok=True)


class ServiceConfig(_Base):
    """API and dashboard runtime settings."""

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_title: str = "Battery Digital Twin API"
    api_root_path: str = ""
    max_history_cycles: int = Field(
        default=5000, ge=1, description="Upper bound on rows accepted in one request."
    )
    dashboard_api_url: str = "http://127.0.0.1:8000"
    dashboard_mode: Literal["service", "api"] = Field(
        default="service",
        description="'service' calls BatteryDigitalTwinService in-process; 'api' "
        "goes through the HTTP client. Both use the same service layer code path.",
    )
    request_id_header: str = "X-Request-ID"


# ---------------------------------------------------------------------------
# Milestone 3 — fleet intelligence
# ---------------------------------------------------------------------------
class FleetRankingConfig(_Base):
    """Weights and normalisation scales for the composite priority score.

    The score is a **configurable decision-support policy**, not an optimum. Each
    component is mapped into [0, 1] with the reference scale named below, then
    combined with these weights and multiplied by ``score_scale``. Every
    component and its transformation is reported in the score breakdown, so a
    reviewer can see how a ranking was produced rather than trusting the number.
    """

    risk_weight: float = Field(default=0.30, ge=0.0)
    rul_weight: float = Field(default=0.20, ge=0.0)
    rul_lower_weight: float = Field(default=0.20, ge=0.0)
    soh_weight: float = Field(default=0.15, ge=0.0)
    trend_weight: float = Field(default=0.05, ge=0.0)
    uncertainty_weight: float = Field(default=0.05, ge=0.0)
    quality_weight: float = Field(default=0.05, ge=0.0)
    #: RUL at or above this many cycles contributes zero urgency; 0 cycles
    #: contributes 1.0. Linear in between.
    rul_reference_cycles: int = Field(default=100, ge=1)
    #: Interval width at or above this many cycles saturates the uncertainty term.
    uncertainty_reference_cycles: int = Field(default=100, ge=1)
    #: SOH at or below this fraction saturates the health-severity term; the term
    #: is zero at ``soh_reference_high``.
    soh_reference_low: float = Field(default=0.70, gt=0.0, lt=1.5)
    soh_reference_high: float = Field(default=1.00, gt=0.0, le=1.5)
    #: Capacity-fade trend (percent per 10 cycles) that saturates the trend term.
    trend_reference_pct_per_10: float = Field(default=3.0, gt=0.0)
    score_scale: float = Field(default=100.0, gt=0.0)
    #: Score floor applied when a critical rule fires, so a rule override cannot
    #: be out-ranked by a smoothly-scored cell.
    critical_override_score: float = Field(default=95.0, ge=0.0)

    @model_validator(mode="after")
    def _weights_positive(self) -> FleetRankingConfig:
        if self.total_weight <= 0:
            raise ValueError("At least one ranking weight must be positive")
        if self.soh_reference_low >= self.soh_reference_high:
            raise ValueError("soh_reference_low must be below soh_reference_high")
        if self.critical_override_score > self.score_scale:
            raise ValueError("critical_override_score must not exceed score_scale")
        return self

    @property
    def total_weight(self) -> float:
        return (
            self.risk_weight
            + self.rul_weight
            + self.rul_lower_weight
            + self.soh_weight
            + self.trend_weight
            + self.uncertainty_weight
            + self.quality_weight
        )

    def weights(self) -> dict[str, float]:
        return {
            "risk": self.risk_weight,
            "rul": self.rul_weight,
            "rul_lower_bound": self.rul_lower_weight,
            "soh": self.soh_weight,
            "trend": self.trend_weight,
            "uncertainty": self.uncertainty_weight,
            "data_quality": self.quality_weight,
        }


class MaintenancePolicyConfig(_Base):
    """Thresholds for the fleet maintenance-priority engine.

    Engineering policy, deliberately outside the model: a fleet operator
    tightening its inspection rules edits this, and nothing is retrained. The
    defaults are demonstration values, not an industry standard.
    """

    critical_soh: float = Field(default=0.70, gt=0.0, lt=1.5)
    critical_rul_lower_cycles: int = Field(default=5, ge=0)
    critical_risk: float = Field(default=0.85, gt=0.0, le=1.0)
    urgent_rul_lower_cycles: int = Field(default=15, ge=0)
    urgent_risk: float = Field(default=0.60, gt=0.0, le=1.0)
    high_rul_lower_cycles: int = Field(default=40, ge=0)
    high_fade_trend_pct_per_10: float = Field(default=1.0, gt=0.0)
    medium_fade_trend_pct_per_10: float = Field(default=0.5, gt=0.0)
    #: A healthy cell whose interval is wider than this is watched, not filed.
    monitor_interval_width_cycles: float = Field(default=60.0, gt=0.0)
    #: Data-quality classes that force INSUFFICIENT_DATA regardless of outputs.
    insufficient_quality_classes: list[str] = Field(default_factory=lambda: ["INSUFFICIENT"])
    #: Inspection windows in cycles, per priority level. Cycle-based is the
    #: primary output; calendar days are only estimated when a usage rate exists.
    inspection_window_cycles: dict[str, int] = Field(
        default_factory=lambda: {
            "P0_CRITICAL": 0,
            "P1_URGENT": 5,
            "P2_HIGH": 10,
            "P3_MEDIUM": 20,
            "P4_LOW": 0,
            "P5_MONITOR": 0,
        },
        description="Recommended inspection window per priority, in cycles. 0 for "
        "P0 means immediate engineering review; 0 for P4/P5 means 'next scheduled "
        "inspection', with no separate window.",
    )
    min_cycles_for_rate_estimate: int = Field(
        default=10,
        ge=2,
        description="Cycles of timestamped history required before a cycles-per-day "
        "rate is estimated. Without it no calendar-day window is produced — a "
        "cycle-to-day conversion must never be invented.",
    )
    disclaimer: str = (
        "Fleet maintenance priority is deterministic decision support derived from "
        "model outputs and configurable engineering rules. It is not a schedule, not "
        "a safety control, and requires review by a qualified engineer."
    )


class ReplacementPlanningConfig(_Base):
    """Horizons and evidence rules for replacement planning (advisory only)."""

    near_term_cycles: int = Field(default=20, ge=1)
    medium_term_cycles: int = Field(default=50, ge=1)
    long_term_cycles: int = Field(default=100, ge=1)
    #: A candidate is flagged when the RUL lower bound falls inside a horizon.
    #: The point estimate is reported beside it but does not drive the flag.
    use_lower_bound: bool = True
    #: Risk at or above this level makes a cell a candidate regardless of RUL.
    risk_candidate_threshold: float = Field(default=0.60, gt=0.0, le=1.0)
    soh_candidate_threshold: float = Field(default=0.72, gt=0.0, lt=1.5)
    #: Interval width above this fraction of the point estimate downgrades the
    #: planning confidence to "low".
    wide_interval_ratio: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> ReplacementPlanningConfig:
        if not (self.near_term_cycles < self.medium_term_cycles < self.long_term_cycles):
            raise ValueError("replacement horizons must be strictly increasing")
        return self


class WorkloadForecastConfig(_Base):
    """Buckets for the maintenance-workload planning view."""

    horizons_cycles: list[int] = Field(default_factory=lambda: [10, 30, 50])

    @field_validator("horizons_cycles")
    @classmethod
    def _sorted_positive(cls, v: list[int]) -> list[int]:
        if any(x < 1 for x in v):
            raise ValueError("workload horizons must be >= 1")
        return sorted(set(v))


class FleetConfig(_Base):
    """Fleet orchestration limits and policy sub-sections."""

    default_fleet_id: str = "DEMO-FLEET-01"
    max_batteries_per_request: int = Field(
        default=100, ge=1, description="Upper bound on batteries in one online (HTTP) request."
    )
    max_batteries_per_batch: int = Field(
        default=10_000, ge=1, description="Upper bound for an offline batch run."
    )
    max_concurrency: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Worker threads for batch inference. The default is 1: the "
        "estimators are already BLAS-parallel and the ordering guarantee is easier "
        "to reason about serially. Results are re-ordered deterministically either way.",
    )
    min_cycles_per_battery: int = Field(
        default=2, ge=1, description="Rows below which an ingested battery is rejected outright."
    )
    max_upload_bytes: int = Field(
        default=25 * 1024 * 1024, ge=1024, description="Cap on an uploaded fleet file."
    )
    page_size: int = Field(default=50, ge=1, le=1000)
    max_page_size: int = Field(default=500, ge=1, le=5000)
    critical_priorities: list[str] = Field(
        default_factory=lambda: ["P0_CRITICAL", "P1_URGENT"],
        description="Priority levels counted as 'critical' in the fleet summary.",
    )
    low_rul_thresholds: list[int] = Field(
        default_factory=lambda: [20, 50, 100],
        description="Report how many evaluated cells sit below each RUL threshold.",
    )
    high_risk_thresholds: list[float] = Field(default_factory=lambda: [0.5, 0.8])
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    high_critical_count_alert: int = Field(
        default=5, ge=0, description="Critical-battery count that raises a fleet-level alert."
    )
    ranking: FleetRankingConfig = Field(default_factory=FleetRankingConfig)
    maintenance: MaintenancePolicyConfig = Field(default_factory=MaintenancePolicyConfig)
    replacement: ReplacementPlanningConfig = Field(default_factory=ReplacementPlanningConfig)
    workload: WorkloadForecastConfig = Field(default_factory=WorkloadForecastConfig)

    @model_validator(mode="after")
    def _page_bounds(self) -> FleetConfig:
        if self.page_size > self.max_page_size:
            raise ValueError("fleet.page_size must not exceed fleet.max_page_size")
        return self


# ---------------------------------------------------------------------------
# Milestone 3 — monitoring
# ---------------------------------------------------------------------------
NumericalDriftMetric = Literal["psi", "ks", "wasserstein", "js"]
CategoricalDriftMetric = Literal["chi2", "js", "unseen_rate"]


def _default_numerical_drift_metrics() -> list[NumericalDriftMetric]:
    return ["psi", "ks"]


def _default_categorical_drift_metrics() -> list[CategoricalDriftMetric]:
    return ["chi2", "unseen_rate"]


class DriftConfig(_Base):
    """Batch feature-drift detection against a versioned reference artifact."""

    enabled: bool = True
    numerical_metrics: list[NumericalDriftMetric] = Field(
        default_factory=_default_numerical_drift_metrics
    )
    categorical_metrics: list[CategoricalDriftMetric] = Field(
        default_factory=_default_categorical_drift_metrics
    )
    #: Alert thresholds per metric: (warning, critical).
    psi_thresholds: tuple[float, float] = (0.10, 0.25)
    ks_thresholds: tuple[float, float] = (0.10, 0.20)
    wasserstein_thresholds: tuple[float, float] = (0.10, 0.25)
    js_thresholds: tuple[float, float] = (0.10, 0.25)
    unseen_rate_thresholds: tuple[float, float] = (0.01, 0.05)
    n_bins: int = Field(default=10, ge=2)
    min_sample_size: int = Field(
        default=50,
        ge=5,
        description="Below this many current observations a feature is reported as "
        "UNRELIABLE rather than drifted: a small sample moves these statistics on "
        "its own.",
    )
    max_features: int = Field(
        default=200, ge=1, description="Cap on features tested, for runtime and readability."
    )
    #: Multiple-comparison control for the p-value-bearing tests (KS, chi-square).
    multiple_comparison: Literal["benjamini_hochberg", "bonferroni", "none"] = "benjamini_hochberg"
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    #: Fleet status: fraction of tested features that must be flagged.
    fleet_warning_fraction: float = Field(default=0.10, gt=0.0, le=1.0)
    fleet_critical_fraction: float = Field(default=0.25, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> DriftConfig:
        for name in (
            "psi_thresholds",
            "ks_thresholds",
            "wasserstein_thresholds",
            "js_thresholds",
            "unseen_rate_thresholds",
        ):
            warning, critical = getattr(self, name)
            if warning >= critical:
                raise ValueError(f"{name}: warning threshold must be below the critical one")
        if self.fleet_warning_fraction >= self.fleet_critical_fraction:
            raise ValueError("fleet_warning_fraction must be below fleet_critical_fraction")
        return self


class PredictionDriftConfig(_Base):
    """Thresholds for changes in the model's own output distribution."""

    enabled: bool = True
    #: Absolute mean shift, in units of the reference standard deviation.
    standardised_mean_shift_thresholds: tuple[float, float] = (0.25, 0.50)
    psi_thresholds: tuple[float, float] = (0.10, 0.25)
    #: Total-variation distance between class frequency vectors.
    class_frequency_thresholds: tuple[float, float] = (0.15, 0.30)
    #: Ratio of current to reference median interval width.
    interval_width_ratio_thresholds: tuple[float, float] = (1.25, 1.75)
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    min_sample_size: int = Field(default=20, ge=3)
    n_bins: int = Field(default=10, ge=2)


class PerformanceMonitoringConfig(_Base):
    """Delayed-label performance monitoring thresholds."""

    enabled: bool = True
    min_labels: int = Field(
        default=20,
        ge=1,
        description="Below this many joined labels the status is INSUFFICIENT_LABELS "
        "and no metric is published. Metrics on a handful of rows are noise.",
    )
    rul_mae_thresholds: tuple[float, float] = Field(
        default=(15.0, 30.0), description="(warning, degraded) MAE in cycles."
    )
    soh_mae_thresholds: tuple[float, float] = (0.03, 0.06)
    brier_thresholds: tuple[float, float] = (0.15, 0.25)
    pr_auc_floor: float = Field(default=0.50, ge=0.0, le=1.0)
    #: Empirical interval coverage may fall this far below the nominal level.
    coverage_tolerance: float = Field(default=0.05, ge=0.0, lt=1.0)
    label_delay_note: str = (
        "Production monitoring metrics are computed on labels that became available "
        "after the prediction was made. They are not comparable with the Milestone 1 "
        "and Milestone 2 held-out test metrics, which describe a different partition."
    )


class FleetDataQualityConfig(_Base):
    """Fleet-level aggregation thresholds for input data quality."""

    warning_poor_fraction: float = Field(default=0.10, gt=0.0, le=1.0)
    critical_poor_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    warning_insufficient_fraction: float = Field(default=0.10, gt=0.0, le=1.0)
    critical_insufficient_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    warning_missing_feature_fraction: float = Field(default=0.10, gt=0.0, le=1.0)
    critical_missing_feature_fraction: float = Field(default=0.30, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> FleetDataQualityConfig:
        pairs = (
            (self.warning_poor_fraction, self.critical_poor_fraction),
            (self.warning_insufficient_fraction, self.critical_insufficient_fraction),
            (self.warning_missing_feature_fraction, self.critical_missing_feature_fraction),
        )
        if any(warning >= critical for warning, critical in pairs):
            raise ValueError("data-quality warning fractions must be below the critical ones")
        return self


class AlertPolicyConfig(_Base):
    """Which monitoring findings become alerts, and how loud."""

    enabled: bool = True
    #: Alert types suppressed entirely, by name.
    muted_types: list[str] = Field(default_factory=list)
    #: Alerts are written locally. No external notifier is configured or called;
    #: doing so would need credentials this repository deliberately does not hold.
    external_notifications: bool = Field(default=False, frozen=True)
    max_alerts_per_run: int = Field(default=200, ge=1)


class MonitoringConfig(_Base):
    """Reference artifacts, detectors and alerting."""

    enabled: bool = True
    reference_id: str = Field(
        default="training_reference",
        description="Name of the reference-distribution artifact used by the drift "
        "detectors. Built from the TRAINING partition only.",
    )
    reference_partition: Literal["train", "train_val"] = Field(
        default="train",
        description="Partition the reference distribution is fitted on. The final "
        "test partition is never used: a reference fitted on it would make the "
        "held-out result part of the monitoring machinery.",
    )
    max_reference_samples: int = Field(default=50_000, ge=100)
    data_quality: FleetDataQualityConfig = Field(default_factory=FleetDataQualityConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    prediction_drift: PredictionDriftConfig = Field(default_factory=PredictionDriftConfig)
    performance: PerformanceMonitoringConfig = Field(default_factory=PerformanceMonitoringConfig)
    alerts: AlertPolicyConfig = Field(default_factory=AlertPolicyConfig)


# ---------------------------------------------------------------------------
# Milestone 3 — registry, promotion, tracking, persistence, deployment
# ---------------------------------------------------------------------------
class PromotionGateConfig(_Base):
    """Conditions a candidate must satisfy before it may reach PRODUCTION.

    Every gate is a *comparison against the current production model* where one
    exists, and an absolute floor where it does not. No single metric promotes a
    model on its own, and promotion never happens implicitly in CI.
    """

    require_validation_status: bool = True
    require_artifact_checksum: bool = True
    require_feature_schema_compatible: bool = True
    require_leakage_check: bool = True
    require_inference_smoke_test: bool = True
    require_contract_tests: bool = True
    require_tests_passed: bool = True
    #: Relative RUL MAE regression tolerated against production (0.05 = 5 % worse).
    max_rul_mae_regression: float = Field(default=0.05, ge=0.0)
    max_soh_mae_regression: float = Field(default=0.05, ge=0.0)
    max_brier_regression: float = Field(default=0.05, ge=0.0)
    #: PR-AUC may fall by at most this much in absolute terms.
    max_pr_auc_regression: float = Field(default=0.02, ge=0.0)
    min_interval_coverage: float = Field(default=0.80, ge=0.0, le=1.0)
    min_worst_cell_interval_coverage: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum empirical coverage on the worst individually reported held-out "
            "cell. Marginal coverage can hide a severe cohort-specific miss."
        ),
    )
    max_latency_regression_ratio: float = Field(default=1.50, gt=1.0)
    #: Absolute floors applied when there is no production model to compare with.
    max_absolute_rul_mae: float | None = Field(default=None)
    allow_auto_promotion: bool = Field(
        default=False,
        description="Never true in CI by default. Promotion is an explicit human "
        "action; a gate that promotes on green turns a metric into a deploy button.",
    )


class RegistryConfig(_Base):
    """Model registry backend and stage policy."""

    backend: Literal["file"] = Field(
        default="file",
        description="Local JSON-file registry. MLflow's Model Registry needs a "
        "tracking server with a database backend; adding one for a single-node "
        "prototype would be infrastructure for appearance.",
    )
    dir: Path = Path("artifacts/registry")
    registry_file: str = "models.json"
    allow_multiple_production: bool = Field(
        default=False,
        description="Permit several PRODUCTION entries for one serving task. Keep false "
        "unless an external traffic router, rather than this service, selects among them.",
    )
    verify_checksums: bool = True
    promotion: PromotionGateConfig = Field(default_factory=PromotionGateConfig)


class TrackingConfig(_Base):
    """Experiment tracking. Local file backend by default; MLflow when installed."""

    backend: Literal["file", "mlflow"] = "file"
    dir: Path = Path("artifacts/tracking")
    experiment_name: str = "battery-fleet"
    #: Only used when ``backend='mlflow'``. Empty means MLflow's local ``mlruns``.
    mlflow_tracking_uri: str | None = None
    log_figures: bool = True
    #: Raw cycle measurements are never logged as parameters or artifacts.
    log_raw_data: bool = Field(default=False, frozen=True)


class PersistenceConfig(_Base):
    """Where fleet snapshots, monitoring snapshots and alerts are stored."""

    backend: Literal["sqlite", "memory"] = "sqlite"
    database_path: Path = Path("artifacts/persistence/platform.db")
    schema_version: str = "3.0"
    retention_days: int | None = Field(
        default=365, description="None disables pruning. Nothing is deleted automatically."
    )
    busy_timeout_s: float = Field(default=5.0, gt=0.0)


class DeploymentConfig(_Base):
    """Runtime switches a container or CI runner sets."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["console", "json"] = "console"
    metrics_enabled: bool = True
    #: Serve the Prometheus text exposition on ``/metrics``.
    metrics_endpoint_enabled: bool = True
    read_only: bool = Field(
        default=False,
        description="Refuse every write to persistence and artifacts. For a "
        "container running with a read-only root filesystem.",
    )
    admin_endpoints_enabled: bool = Field(
        default=False, description="Model promotion/rollback over HTTP. Off by default."
    )
    cors_allow_origins: list[str] = Field(
        default_factory=list,
        description="Explicit origin allow-list. Empty means no cross-origin access.",
    )
    max_request_bytes: int = Field(default=32 * 1024 * 1024, ge=1024)


class VizConfig(_Base):
    dpi: int = Field(default=160, ge=72)
    figure_format: Literal["png", "pdf", "svg"] = "png"
    style: str = "battery_rul"
    palette: str = "viridis"
    figsize: tuple[float, float] = (10.0, 6.0)
    max_batteries_per_plot: int = Field(default=12, ge=1)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
class ExperimentConfig(_Base):
    """Root configuration object handed to every pipeline stage."""

    experiment_name: str = "battery_rul_v1"
    description: str = "NASA li-ion remaining-useful-life baseline milestone."
    seed: int = 42

    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    models: ModelZooConfig = Field(default_factory=ModelZooConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    viz: VizConfig = Field(default_factory=VizConfig)

    # -- milestone 2 ------------------------------------------------------
    soh: SOHConfig = Field(default_factory=SOHConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    multitask: MultiTaskConfig = Field(default_factory=MultiTaskConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    recommendations: RecommendationConfig = Field(default_factory=RecommendationConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)

    # -- milestone 3 ------------------------------------------------------
    fleet: FleetConfig = Field(default_factory=FleetConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)

    @model_validator(mode="after")
    def _propagate_seed(self) -> ExperimentConfig:
        """A single ``seed:`` at the root drives every stochastic component unless
        a stage explicitly overrides it in YAML."""
        for stage in (self.split, self.tuning):
            if stage.seed == 42 and self.seed != 42:
                stage.seed = self.seed
        if self.models.training.seed == 42 and self.seed != 42:
            self.models.training.seed = self.seed
        if self.multitask.seed == 42 and self.seed != 42:
            self.multitask.seed = self.seed
        self.artifacts.resolve_against(self.paths.root)
        self._resolve_milestone_3_paths()
        return self

    def _resolve_milestone_3_paths(self) -> None:
        """Anchor the Milestone 3 storage paths to ``paths.root``.

        Same reasoning as :meth:`ArtifactConfig.resolve_against`: a test or CI
        run that redirects ``paths.root`` at a temporary directory must not have
        its registry, tracking store or database silently point at the
        developer's real artifacts.
        """
        root = self.paths.root
        for section, field_name in (
            (self.registry, "dir"),
            (self.tracking, "dir"),
            (self.persistence, "database_path"),
        ):
            value = Path(getattr(section, field_name))
            if not value.is_absolute():
                object.__setattr__(section, field_name, root / value)

    # -- data-affecting fingerprint ---------------------------------------
    def data_affecting_dict(self) -> dict[str, Any]:
        """The configuration subset that changes what the model actually learns.

        Used to key interim caches and to check a persisted model bundle against
        the runtime configuration. Deliberately excludes cosmetics (figure DPI,
        report paths, thread counts): those must not invalidate a cache or block
        a load, and including them would train reviewers to ignore the check.
        """
        return {
            "data": self.data.model_dump(mode="json", exclude={"cache_interim"}),
            "validation": self.validation.model_dump(mode="json"),
            "features": self.features.model_dump(mode="json"),
            "target": self.target.model_dump(mode="json"),
            "soh": self.soh.model_dump(mode="json"),
            "risk": self.risk.model_dump(mode="json", exclude={"threshold"}),
        }

    def data_fingerprint(self) -> str:
        """Stable short hash of :meth:`data_affecting_dict`."""
        import hashlib

        payload = json.dumps(self.data_affecting_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # -- convenience ------------------------------------------------------
    @property
    def eol_capacity_ah(self) -> float:
        """Absolute capacity (Ah) at which a cell is declared end-of-life."""
        return self.data.nominal_capacity_ah * self.data.eol_threshold

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce_scalar(text: str) -> Any:
    """Parse a CLI override value with YAML semantics (``true``/``3``/``[1,2]``)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _expand_dotted(overrides: dict[str, Any]) -> dict[str, Any]:
    """``{"training.epochs": 5}`` -> ``{"training": {"epochs": 5}}``."""
    nested: dict[str, Any] = {}
    for dotted, value in overrides.items():
        cursor = nested
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _coerce_scalar(value) if isinstance(value, str) else value
    return nested


def load_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    base: str | Path | None = None,
) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from YAML plus dotted overrides.

    Parameters
    ----------
    path:
        YAML file to load. ``None`` yields the pure-default configuration.
    overrides:
        Dotted-key overrides applied last, e.g. ``{"models.training.epochs": 3}``.
        Values given as strings are parsed with YAML semantics.
    base:
        Optional parent YAML merged underneath ``path``. If ``path`` itself
        contains a top-level ``extends:`` key, that wins.

    Raises
    ------
    FileNotFoundError, pydantic.ValidationError
    """
    payload: dict[str, Any] = {}

    if base is not None:
        payload = _deep_merge(payload, _read_yaml(base))

    if path is not None:
        payload = _deep_merge(payload, _resolve_extends(path))

    if overrides:
        payload = _deep_merge(payload, _expand_dotted(overrides))

    return ExperimentConfig(**payload)


def _resolve_extends(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML file, recursively merging its ``extends:`` ancestry underneath.

    Chains are supported (``synthetic.yaml`` -> ``fast.yaml`` -> ``default.yaml``)
    and cycles are rejected rather than recursing forever. Relative ``extends``
    paths resolve against the *referring* file's directory, so a config folder can
    be moved wholesale.
    """
    resolved = Path(path).resolve()
    if resolved in _seen:
        chain = " -> ".join(p.name for p in (*_seen, resolved))
        raise ValueError(f"Circular config inheritance: {chain}")

    raw = _read_yaml(resolved)
    parent = raw.pop("extends", None)
    if parent is None:
        return raw

    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    return _deep_merge(_resolve_extends(parent_path, (*_seen, resolved)), raw)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping, got {type(data).__name__}: {p}")
    return data
