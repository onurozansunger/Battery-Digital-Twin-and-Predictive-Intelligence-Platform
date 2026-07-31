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
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ExplainabilityConfig",
    "FeatureConfig",
    "ModelZooConfig",
    "PathsConfig",
    "SplitConfig",
    "TargetConfig",
    "TrainingConfig",
    "TuningConfig",
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
        """Create every configured directory. Idempotent."""
        for field in self.model_fields:
            path = getattr(self, field)
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)


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
        default=0.98, description="Drop one of any feature pair above |rho|. None disables."
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


class ExplainabilityConfig(_Base):
    enabled: bool = True
    shap_enabled: bool = True
    shap_max_samples: int = Field(default=2000, ge=50)
    shap_background_samples: int = Field(default=200, ge=10)
    permutation_repeats: int = Field(default=15, ge=1)
    top_k_features: int = Field(default=25, ge=1)
    error_analysis_quantile: float = Field(default=0.90, gt=0.0, lt=1.0)


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

    @model_validator(mode="after")
    def _propagate_seed(self) -> ExperimentConfig:
        """A single ``seed:`` at the root drives every stochastic component unless
        a stage explicitly overrides it in YAML."""
        for stage in (self.split, self.tuning):
            if stage.seed == 42 and self.seed != 42:
                stage.seed = self.seed
        if self.models.training.seed == 42 and self.seed != 42:
            self.models.training.seed = self.seed
        return self

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
