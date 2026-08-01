"""Deployable model bundles: artifact + preprocessing + contract, or nothing.

A pickled estimator on its own is not deployable. It carries no record of which
columns it expects, in which order, under which end-of-life definition, or with
which target transform — so a configuration drift between training and serving
produces confident nonsense rather than an error. Milestone 1.1 makes the
contract explicit and enforced.

A bundle directory contains:

``model.pkl``          the fitted estimator (joblib)
``preprocessing.pkl``  the fitted :class:`FeaturePipeline`
``metadata.json``      :class:`BundleMetadata` — schema/model version, git
                       revision, data and preprocessing fingerprints, target
                       definition, feature schema, warm-up policy, dependency
                       versions, metrics
``calibration.pkl``    optional probability calibrator
``uncertainty.pkl``    optional conformal interval estimator

Loading validates all of it. ``strict_compatibility`` (default on) refuses a
bundle whose persisted training configuration disagrees with the runtime
configuration on any data-affecting field. The alternative — warn and continue —
means the warning is read once and ignored forever.

Security note: bundles are joblib pickles and are only ever loaded from local
paths this process configured. No API input selects a path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from battery_rul.config import ExperimentConfig
from battery_rul.features.pipeline import FeaturePipeline
from battery_rul.utils.io import environment_fingerprint, load_json, load_pickle, save_json
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "ArtifactCompatibilityError",
    "BundleMetadata",
    "ModelBundle",
    "load_bundle",
    "save_bundle",
]

#: Bumped whenever the on-disk layout changes incompatibly.
BUNDLE_SCHEMA_VERSION = "2.0"

MODEL_FILE = "model.pkl"
PREPROCESSING_FILE = "preprocessing.pkl"
METADATA_FILE = "metadata.json"
CALIBRATION_FILE = "calibration.pkl"
UNCERTAINTY_FILE = "uncertainty.pkl"


class ArtifactCompatibilityError(RuntimeError):
    """Raised when a bundle cannot be served under the runtime configuration."""


@dataclass
class BundleMetadata:
    """The contract a bundle publishes about itself."""

    model_name: str
    model_version: str
    task: str
    schema_version: str = BUNDLE_SCHEMA_VERSION
    created_at_utc: str = ""
    git_revision: str | None = None
    #: Hash of the data-affecting configuration at training time.
    data_fingerprint: str = ""
    #: Hash of the fitted feature schema and preprocessing decisions.
    preprocessing_fingerprint: str = ""
    #: Hash of the source cycle table the model was trained on.
    dataset_fingerprint: str = ""
    feature_names: list[str] = field(default_factory=list)
    sequence_length: int | None = None
    first_scoreable_cycle: int | None = None
    warmup_policy: dict[str, Any] = field(default_factory=dict)
    target_definition: dict[str, Any] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    dependencies: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BundleMetadata:
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            logger.warning("Bundle metadata carries unknown keys (ignored): %s", unknown)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class ModelBundle:
    """A loaded, validated bundle."""

    model: Any
    preprocessing: FeaturePipeline
    metadata: BundleMetadata
    calibrator: Any | None = None
    uncertainty: Any | None = None
    path: Path | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(self.metadata.feature_names)

    def describe(self) -> dict[str, Any]:
        return {
            "model_name": self.metadata.model_name,
            "model_version": self.metadata.model_version,
            "task": self.metadata.task,
            "schema_version": self.metadata.schema_version,
            "created_at_utc": self.metadata.created_at_utc,
            "git_revision": self.metadata.git_revision,
            "n_features": len(self.metadata.feature_names),
            "sequence_length": self.metadata.sequence_length,
            "first_scoreable_cycle": self.metadata.first_scoreable_cycle,
            "has_calibrator": self.calibrator is not None,
            "has_uncertainty": self.uncertainty is not None,
            "data_fingerprint": self.metadata.data_fingerprint,
            "preprocessing_fingerprint": self.metadata.preprocessing_fingerprint,
        }


def _required_metadata_fields() -> tuple[str, ...]:
    return (
        "model_name",
        "model_version",
        "task",
        "schema_version",
        "data_fingerprint",
        "preprocessing_fingerprint",
        "feature_names",
    )


def save_bundle(
    path: str | Path,
    *,
    model: Any,
    preprocessing: FeaturePipeline,
    cfg: ExperimentConfig,
    task: str,
    model_name: str,
    model_version: str = "1.0.0",
    dataset_fingerprint: str = "",
    metrics: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
    sequence_length: int | None = None,
    first_scoreable_cycle: int | None = None,
    warmup_policy: dict[str, Any] | None = None,
    calibrator: Any | None = None,
    uncertainty: Any | None = None,
    notes: str = "",
) -> Path:
    """Write a complete bundle. Partial bundles are never written."""
    from battery_rul.utils.io import save_pickle

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    environment = environment_fingerprint()

    metadata = BundleMetadata(
        model_name=model_name,
        model_version=model_version,
        task=task,
        schema_version=BUNDLE_SCHEMA_VERSION,
        created_at_utc=datetime.now(UTC).isoformat(),
        git_revision=environment.get("git_revision"),
        data_fingerprint=cfg.data_fingerprint(),
        preprocessing_fingerprint=preprocessing.fingerprint(),
        dataset_fingerprint=dataset_fingerprint,
        feature_names=list(preprocessing.feature_names),
        sequence_length=sequence_length,
        first_scoreable_cycle=first_scoreable_cycle,
        warmup_policy=warmup_policy or {},
        target_definition={
            "name": cfg.target.name,
            "eol_threshold": cfg.data.eol_threshold,
            "eol_reference": cfg.data.eol_reference,
            "eol_persistence": cfg.target.eol_persistence,
            "log_transform": cfg.target.log_transform,
            "cap_at": cfg.target.cap_at,
            "clip_negative": cfg.target.clip_negative,
            "soh_reference_strategy": cfg.soh.reference_strategy,
            "soh_reference_cycles": cfg.soh.reference_cycles,
            "failure_horizon_cycles": cfg.risk.horizon_cycles,
        },
        training_config=cfg.data_affecting_dict(),
        thresholds=thresholds or {},
        metrics=metrics or {},
        dependencies=environment.get("packages", {}),
        notes=notes,
    )

    save_pickle(model, path / MODEL_FILE)
    save_pickle(preprocessing, path / PREPROCESSING_FILE)
    if calibrator is not None:
        save_pickle(calibrator, path / CALIBRATION_FILE)
    if uncertainty is not None:
        save_pickle(uncertainty, path / UNCERTAINTY_FILE)
    save_json(metadata.to_dict(), path / METADATA_FILE)
    logger.info("Bundle written: %s (%s / %s)", path, model_name, task)
    return path


def load_bundle(
    path: str | Path,
    cfg: ExperimentConfig | None = None,
    *,
    require_calibrator: bool = False,
    require_uncertainty: bool = False,
) -> ModelBundle:
    """Load and validate a bundle. Fails loudly rather than serving a mismatch.

    Raises
    ------
    FileNotFoundError
        Bundle directory or a required artifact is absent.
    ArtifactCompatibilityError
        Metadata is incomplete, the schema version is unsupported, the feature
        schema disagrees with the persisted preprocessing artifact, or — when
        ``cfg`` is supplied and ``artifacts.strict_compatibility`` is set — the
        runtime configuration differs from the training configuration on a
        data-affecting field.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Model bundle not found: {path}. Build it with "
            "`python -m battery_rul.pipelines.build_model_bundle`."
        )

    for required in (MODEL_FILE, PREPROCESSING_FILE, METADATA_FILE):
        if not (path / required).is_file():
            raise FileNotFoundError(f"Bundle {path} is incomplete: {required} is missing")

    payload = load_json(path / METADATA_FILE)
    if not isinstance(payload, dict):
        raise ArtifactCompatibilityError(f"{path / METADATA_FILE} is not a JSON object")
    missing = [f for f in _required_metadata_fields() if not payload.get(f)]
    if missing:
        raise ArtifactCompatibilityError(
            f"Bundle {path} metadata is missing required field(s): {missing}"
        )
    metadata = BundleMetadata.from_dict(payload)

    major = metadata.schema_version.split(".")[0]
    if major != BUNDLE_SCHEMA_VERSION.split(".")[0]:
        raise ArtifactCompatibilityError(
            f"Bundle {path} uses schema version {metadata.schema_version}, this build "
            f"supports {BUNDLE_SCHEMA_VERSION}. Rebuild the bundle."
        )

    model = load_pickle(path / MODEL_FILE)
    preprocessing = load_pickle(path / PREPROCESSING_FILE)
    if not isinstance(preprocessing, FeaturePipeline):
        raise ArtifactCompatibilityError(
            f"{path / PREPROCESSING_FILE} does not contain a FeaturePipeline "
            f"(got {type(preprocessing).__name__})"
        )
    if list(preprocessing.feature_names) != list(metadata.feature_names):
        raise ArtifactCompatibilityError(
            f"Bundle {path}: the persisted preprocessing artifact expects "
            f"{len(preprocessing.feature_names)} features but the metadata declares "
            f"{len(metadata.feature_names)}. The artifacts were not written together."
        )

    calibrator = None
    if (path / CALIBRATION_FILE).is_file():
        calibrator = load_pickle(path / CALIBRATION_FILE)
    elif require_calibrator:
        raise ArtifactCompatibilityError(
            f"Bundle {path} has no calibration artifact, but calibrated probabilities "
            "were requested. Run the calibration pipeline."
        )

    uncertainty = None
    if (path / UNCERTAINTY_FILE).is_file():
        uncertainty = load_pickle(path / UNCERTAINTY_FILE)
    elif require_uncertainty:
        raise ArtifactCompatibilityError(
            f"Bundle {path} has no uncertainty artifact, but prediction intervals "
            "were requested. Run the conformal calibration pipeline."
        )

    if cfg is not None:
        _check_runtime_compatibility(metadata, cfg, path)

    logger.info(
        "Loaded bundle %s: %s v%s (task=%s, %d features)",
        path.name,
        metadata.model_name,
        metadata.model_version,
        metadata.task,
        len(metadata.feature_names),
    )
    return ModelBundle(
        model=model,
        preprocessing=preprocessing,
        metadata=metadata,
        calibrator=calibrator,
        uncertainty=uncertainty,
        path=path,
    )


def _check_runtime_compatibility(
    metadata: BundleMetadata, cfg: ExperimentConfig, path: Path
) -> None:
    """Compare the persisted training configuration with the runtime one."""
    runtime = cfg.data_affecting_dict()
    trained = metadata.training_config or {}
    differences = _diff(trained, runtime)

    if not differences:
        return

    message = (
        f"Bundle {path} was trained under a different data-affecting configuration.\n"
        + "\n".join(
            f"  {key}: trained={old!r} runtime={new!r}" for key, old, new in differences[:20]
        )
    )
    if len(differences) > 20:
        message += f"\n  … and {len(differences) - 20} more"

    if cfg.artifacts.strict_compatibility:
        raise ArtifactCompatibilityError(
            message + "\nRetrain the bundle, or set artifacts.strict_compatibility=false if "
            "you have verified the difference is immaterial."
        )
    logger.warning("%s\n(strict_compatibility is off; continuing)", message)


def _diff(
    trained: dict[str, Any], runtime: dict[str, Any], prefix: str = ""
) -> list[tuple[str, Any, Any]]:
    """Recursive dictionary difference, reported as dotted paths."""
    out: list[tuple[str, Any, Any]] = []
    for key in sorted(set(trained) | set(runtime)):
        left, right = trained.get(key), runtime.get(key)
        dotted = f"{prefix}{key}"
        if isinstance(left, dict) and isinstance(right, dict):
            out.extend(_diff(left, right, prefix=f"{dotted}."))
        elif json.dumps(left, sort_keys=True, default=str) != json.dumps(
            right, sort_keys=True, default=str
        ):
            out.append((dotted, left, right))
    return out
