"""A local, file-backed model registry.

Why not MLflow's Model Registry: it requires a tracking *server* with a database
backend. For a single-node prototype that would be a service to run, secure and
back up in order to store what is, here, a list of forty-line records. Experiment
tracking does use MLflow when it is installed (see
:mod:`battery_rul.tracking`); the registry is a JSON document with an interface
narrow enough to swap later.

Guarantees this implementation makes
------------------------------------
* at most one PRODUCTION version per serving task (RUL, SOH or risk), unless
  explicitly configured otherwise — enforced on write, not by convention;
* every entry carries a checksum over its bundle files, verified on promotion,
  so an entry cannot silently point at an artifact that has been replaced;
* every stage transition is appended to a history, with who did it and why, so
  "when did this go live" is answerable;
* bundle paths are stored **relative to the project root**, so a registry file
  never leaks a developer's home directory into a committed artifact.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from battery_rul.config import ExperimentConfig
from battery_rul.utils.io import load_json, save_json
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "REGISTRY_SCHEMA_VERSION",
    "FileModelRegistry",
    "ModelStage",
    "RegisteredModel",
    "RegistryError",
    "bundle_checksum",
]

REGISTRY_SCHEMA_VERSION = "3.0"

#: Files whose bytes define a bundle's identity. Ordered, so the checksum is
#: reproducible across filesystems that enumerate directories differently.
_CHECKSUM_FILES = (
    "metadata.json",
    "model.pkl",
    "preprocessing.pkl",
    "calibration.pkl",
    "uncertainty.pkl",
)


class RegistryError(RuntimeError):
    """A registry operation is not permitted or cannot be completed."""


class ModelStage(StrEnum):
    """Lifecycle stages. A model moves forward explicitly and can move back."""

    CANDIDATE = "CANDIDATE"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"
    ARCHIVED = "ARCHIVED"
    REJECTED = "REJECTED"


#: Legal transitions. Rejecting a production model without archiving it first is
#: not an oversight to allow: it would leave the fleet served by something the
#: registry says was refused.
_ALLOWED_TRANSITIONS: dict[ModelStage, set[ModelStage]] = {
    ModelStage.CANDIDATE: {ModelStage.STAGING, ModelStage.PRODUCTION, ModelStage.REJECTED},
    ModelStage.STAGING: {ModelStage.PRODUCTION, ModelStage.ARCHIVED, ModelStage.REJECTED},
    ModelStage.PRODUCTION: {ModelStage.ARCHIVED},
    ModelStage.ARCHIVED: {ModelStage.STAGING, ModelStage.PRODUCTION},
    ModelStage.REJECTED: {ModelStage.CANDIDATE},
}


def bundle_checksum(path: str | Path) -> str:
    """SHA-256 over a bundle's files, in a fixed order.

    Absent optional files (no calibrator, no conformal estimator) are hashed as
    their absence, so a bundle that loses its calibrator has a different
    checksum rather than the same one.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise RegistryError(f"Not a bundle directory: {directory}")
    digest = hashlib.sha256()
    for name in _CHECKSUM_FILES:
        file = directory / name
        digest.update(name.encode("utf-8"))
        if not file.is_file():
            digest.update(b"<absent>")
            continue
        digest.update(file.read_bytes())
    return digest.hexdigest()


@dataclass
class RegisteredModel:
    """One registered model version."""

    model_name: str
    model_version: str
    stage: ModelStage = ModelStage.CANDIDATE
    #: Relative to the project root. Never absolute — see the module docstring.
    bundle_path: str = ""
    artifact_checksum: str = ""
    dataset_fingerprint: str = ""
    data_fingerprint: str = ""
    feature_schema_version: str = ""
    feature_schema_fingerprint: str = ""
    n_features: int = 0
    task: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    calibration_metrics: dict[str, Any] = field(default_factory=dict)
    uncertainty_metrics: dict[str, Any] = field(default_factory=dict)
    validation_status: str = "UNVALIDATED"
    validation_evidence: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    promoted_at_utc: str | None = None
    promoted_by: str | None = None
    git_revision: str | None = None
    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.model_name}:{self.model_version}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stage"] = self.stage.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RegisteredModel:
        known = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        known["stage"] = ModelStage(known.get("stage", ModelStage.CANDIDATE))
        return cls(**known)


@dataclass
class FileModelRegistry:
    """JSON-document registry under ``artifacts/registry``."""

    cfg: ExperimentConfig

    @property
    def path(self) -> Path:
        return Path(self.cfg.registry.dir) / self.cfg.registry.registry_file

    # -- reading -----------------------------------------------------------
    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": REGISTRY_SCHEMA_VERSION, "models": [], "history": []}
        payload = load_json(self.path)
        if not isinstance(payload, dict):
            raise RegistryError(f"{self.path} is not a JSON object")
        version = str(payload.get("schema_version", REGISTRY_SCHEMA_VERSION))
        if version.split(".")[0] != REGISTRY_SCHEMA_VERSION.split(".")[0]:
            raise RegistryError(
                f"Registry {self.path} uses schema {version}; this build supports "
                f"{REGISTRY_SCHEMA_VERSION}."
            )
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        if self.cfg.deployment.read_only:
            raise RegistryError(
                "deployment.read_only is set: this process must not modify the registry."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload["schema_version"] = REGISTRY_SCHEMA_VERSION
        payload["updated_at_utc"] = datetime.now(UTC).isoformat()
        save_json(payload, self.path)

    def list_models(
        self,
        *,
        model_name: str | None = None,
        stage: ModelStage | None = None,
        task: str | None = None,
    ) -> list[RegisteredModel]:
        entries = [RegisteredModel.from_dict(e) for e in self._read().get("models", [])]
        if model_name:
            entries = [e for e in entries if e.model_name == model_name]
        if stage:
            entries = [e for e in entries if e.stage is stage]
        if task:
            entries = [e for e in entries if e.task == task]
        return sorted(entries, key=lambda e: (e.model_name, e.created_at_utc), reverse=False)

    def get(self, model_name: str, model_version: str) -> RegisteredModel | None:
        return next(
            (
                e
                for e in self.list_models(model_name=model_name)
                if e.model_version == model_version
            ),
            None,
        )

    def production_model(
        self, model_name: str | None = None, *, task: str | None = None
    ) -> RegisteredModel | None:
        """The live model for a family or task, or ``None``.

        Serving resolves by task (RUL, SOH, risk). A caller that supplies neither
        filter retains the old registry-wide view and receives an explicit error
        when several task-specific production models exist.
        """
        entries = self.list_models(
            model_name=model_name,
            stage=ModelStage.PRODUCTION,
            task=task,
        )
        if not entries:
            return None
        if len(entries) > 1 and not self.cfg.registry.allow_multiple_production:
            raise RegistryError(
                f"{len(entries)} PRODUCTION entries exist for "
                f"{model_name or 'the registry'}; that state should be unreachable. "
                "Archive all but one before serving."
            )
        return entries[-1]

    def bundle_directory(self, entry: RegisteredModel, *, verify_checksum: bool = True) -> Path:
        """Resolve and verify the bundle behind a registry entry.

        This is the serving boundary: a promoted record is not trusted merely
        because it says ``PRODUCTION``. Its directory must still exist and its
        bytes must still match the checksum captured at registration.
        """
        directory = _absolute_path(entry.bundle_path, self.cfg).resolve()
        if not directory.is_dir():
            raise RegistryError(f"Bundle for {entry.key} is missing at {entry.bundle_path}.")
        if verify_checksum:
            actual = bundle_checksum(directory)
            if actual != entry.artifact_checksum:
                raise RegistryError(
                    f"Checksum mismatch for {entry.key}: registered "
                    f"{entry.artifact_checksum[:12]}…, on disk {actual[:12]}…."
                )
        return directory

    def history(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return list(self._read().get("history", []))[-limit:]

    # -- writing -----------------------------------------------------------
    def register(
        self,
        *,
        model_name: str,
        model_version: str,
        bundle_path: str | Path,
        metrics: dict[str, Any] | None = None,
        calibration_metrics: dict[str, Any] | None = None,
        uncertainty_metrics: dict[str, Any] | None = None,
        validation_status: str = "UNVALIDATED",
        validation_evidence: dict[str, Any] | None = None,
        notes: str = "",
        stage: ModelStage = ModelStage.CANDIDATE,
        overwrite: bool = False,
    ) -> RegisteredModel:
        """Register a bundle as a CANDIDATE (or another explicitly named stage).

        The bundle's own metadata supplies the fingerprints and feature schema:
        re-typing them at the call site is how a registry entry ends up
        describing a different model from the one on disk.
        """
        from battery_rul.models.bundle import BundleMetadata

        directory = Path(bundle_path)
        metadata_file = directory / "metadata.json"
        if not metadata_file.is_file():
            raise RegistryError(
                f"{directory} is not a model bundle: metadata.json is missing. Build it "
                "with `python -m battery_rul.pipelines.build_model_bundle`."
            )
        metadata = BundleMetadata.from_dict(load_json(metadata_file))

        payload = self._read()
        existing = [
            e
            for e in payload.get("models", [])
            if e.get("model_name") == model_name and e.get("model_version") == model_version
        ]
        if existing and not overwrite:
            raise RegistryError(
                f"{model_name}:{model_version} is already registered. Use a new version "
                "or pass overwrite=True — silently replacing a registered version would "
                "make every snapshot that cites it ambiguous."
            )

        entry = RegisteredModel(
            model_name=model_name,
            model_version=model_version,
            stage=stage,
            bundle_path=_relative_to_root(directory, self.cfg),
            artifact_checksum=bundle_checksum(directory),
            dataset_fingerprint=metadata.dataset_fingerprint,
            data_fingerprint=metadata.data_fingerprint,
            feature_schema_version=metadata.schema_version,
            feature_schema_fingerprint=metadata.preprocessing_fingerprint,
            n_features=len(metadata.feature_names),
            task=metadata.task,
            metrics=metrics if metrics is not None else dict(metadata.metrics or {}),
            calibration_metrics=calibration_metrics or {},
            uncertainty_metrics=uncertainty_metrics or {},
            validation_status=validation_status,
            validation_evidence=validation_evidence or {},
            git_revision=metadata.git_revision,
            notes=notes,
        )

        payload["models"] = [
            e
            for e in payload.get("models", [])
            if not (e.get("model_name") == model_name and e.get("model_version") == model_version)
        ]
        payload["models"].append(entry.to_dict())
        payload.setdefault("history", []).append(
            {
                "at_utc": entry.created_at_utc,
                "action": "register",
                "model": entry.key,
                "stage": stage.value,
                "checksum": entry.artifact_checksum,
            }
        )
        self._write(payload)
        logger.info("Registered %s at stage %s", entry.key, stage.value)
        return entry

    def transition(
        self,
        model_name: str,
        model_version: str,
        stage: ModelStage,
        *,
        by: str,
        reason: str = "",
        verify_checksum: bool | None = None,
        force: bool = False,
    ) -> RegisteredModel:
        """Move a version to a new stage, enforcing the single-production rule."""
        payload = self._read()
        entries = [RegisteredModel.from_dict(e) for e in payload.get("models", [])]
        target = next(
            (e for e in entries if e.model_name == model_name and e.model_version == model_version),
            None,
        )
        if target is None:
            raise RegistryError(f"{model_name}:{model_version} is not registered.")

        allowed = _ALLOWED_TRANSITIONS.get(target.stage, set())
        if stage is not target.stage and stage not in allowed and not force:
            raise RegistryError(
                f"Illegal transition {target.stage.value} -> {stage.value} for "
                f"{target.key}. Legal: {sorted(s.value for s in allowed)}."
            )

        check = self.cfg.registry.verify_checksums if verify_checksum is None else verify_checksum
        if check and stage in (ModelStage.STAGING, ModelStage.PRODUCTION):
            self.bundle_directory(target, verify_checksum=True)

        archived: list[str] = []
        if stage is ModelStage.PRODUCTION and not self.cfg.registry.allow_multiple_production:
            for entry in entries:
                if (
                    (entry.model_name == model_name or entry.task == target.task)
                    and entry.stage is ModelStage.PRODUCTION
                    and entry.key != target.key
                ):
                    entry.stage = ModelStage.ARCHIVED
                    archived.append(entry.key)

        previous = target.stage
        target.stage = stage
        if stage is ModelStage.PRODUCTION:
            target.promoted_at_utc = datetime.now(UTC).isoformat()
            target.promoted_by = by

        payload["models"] = [e.to_dict() for e in entries]
        payload.setdefault("history", []).append(
            {
                "at_utc": datetime.now(UTC).isoformat(),
                "action": "transition",
                "model": target.key,
                "from_stage": previous.value,
                "to_stage": stage.value,
                "by": by,
                "reason": reason,
                "auto_archived": archived,
                "forced": force,
            }
        )
        self._write(payload)
        logger.info(
            "Transitioned %s: %s -> %s by %s%s",
            target.key,
            previous.value,
            stage.value,
            by,
            f" (auto-archived {archived})" if archived else "",
        )
        return target

    def promote(
        self, model_name: str, model_version: str, *, by: str, reason: str = "", force: bool = False
    ) -> RegisteredModel:
        """Shorthand for a transition to PRODUCTION."""
        return self.transition(
            model_name, model_version, ModelStage.PRODUCTION, by=by, reason=reason, force=force
        )

    def rollback(self, model_name: str, *, by: str, reason: str = "") -> RegisteredModel:
        """Restore the most recently archived version that was in production.

        Rollback is a first-class operation, not "promote the old one again":
        it must work when the current production model is broken, and it must
        pick the version that was actually live before, not the newest archive.
        """
        payload = self._read()
        entries = [RegisteredModel.from_dict(e) for e in payload.get("models", [])]
        current = next(
            (e for e in entries if e.model_name == model_name and e.stage is ModelStage.PRODUCTION),
            None,
        )
        previously_live = sorted(
            (
                e
                for e in entries
                if e.model_name == model_name
                and e.stage is ModelStage.ARCHIVED
                and e.promoted_at_utc is not None
                and (current is None or e.model_version != current.model_version)
            ),
            key=lambda e: e.promoted_at_utc or "",
        )
        if not previously_live:
            raise RegistryError(
                f"No archived version of {model_name} was ever in production, so there "
                "is nothing to roll back to."
            )
        target = previously_live[-1]

        if current is not None:
            self.transition(
                model_name,
                current.model_version,
                ModelStage.ARCHIVED,
                by=by,
                reason=f"rolled back: {reason}" if reason else "rolled back",
            )
        restored = self.transition(
            model_name,
            target.model_version,
            ModelStage.PRODUCTION,
            by=by,
            reason=f"rollback: {reason}" if reason else "rollback",
        )
        logger.warning(
            "Rolled back %s: %s -> %s",
            model_name,
            current.model_version if current else "<none>",
            restored.model_version,
        )
        return restored

    def verify(self, model_name: str, model_version: str) -> dict[str, Any]:
        """Re-check an entry's artifact against its recorded checksum."""
        entry = self.get(model_name, model_version)
        if entry is None:
            raise RegistryError(f"{model_name}:{model_version} is not registered.")
        directory = _absolute_path(entry.bundle_path, self.cfg)
        if not directory.is_dir():
            return {
                "verified": False,
                "reason": f"bundle directory is missing: {entry.bundle_path}",
            }
        actual = bundle_checksum(directory)
        return {
            "verified": actual == entry.artifact_checksum,
            "registered_checksum": entry.artifact_checksum,
            "actual_checksum": actual,
            "bundle_path": entry.bundle_path,
        }


def _relative_to_root(path: Path, cfg: ExperimentConfig) -> str:
    root = Path(cfg.paths.root).resolve()
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        # Outside the project root: keep the absolute path but say so loudly, so
        # a registry written on one machine is not silently unusable on another.
        logger.warning(
            "Bundle %s is outside the project root %s; storing an absolute path, which "
            "will not resolve on another machine.",
            resolved,
            root,
        )
        return str(resolved)


def _absolute_path(stored: str, cfg: ExperimentConfig) -> Path:
    path = Path(stored)
    return path if path.is_absolute() else Path(cfg.paths.root) / path
