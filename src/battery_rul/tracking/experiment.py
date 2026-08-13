"""Run tracking: what was run, on what, with what result.

A run record answers the question a metric alone cannot: *could I reproduce
this?* Git revision, dataset fingerprint, configuration, seed, environment and
the artifact paths are therefore part of the record, not optional extras.

MLflow is supported and never required. A remote tracking server is a service to
run; this project's default is a directory of JSON files, which is enough to
compare runs, survives being copied around, and can be read without starting
anything.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from battery_rul.config import ExperimentConfig
from battery_rul.utils.io import environment_fingerprint, save_json
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "ExperimentTracker",
    "FileTracker",
    "MLflowTracker",
    "RunRecord",
    "build_tracker",
    "compare_runs",
]

#: Parameter names that must never be logged, whatever a call site passes.
#: Tracking stores are widely readable; measurement data is not.
_FORBIDDEN_PARAMS = frozenset({"history", "cycles", "frame", "records", "raw_data"})


@dataclass
class RunRecord:
    """One tracked run."""

    run_id: str
    experiment_name: str
    started_at_utc: str
    status: str = "RUNNING"
    finished_at_utc: str | None = None
    git_revision: str | None = None
    dataset_fingerprint: str = ""
    data_fingerprint: str = ""
    model_type: str = ""
    seed: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    features: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    tables: dict[str, str] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class ExperimentTracker(Protocol):
    """The tracking contract both backends satisfy."""

    def start_run(self, name: str, **tags: str) -> RunRecord: ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, Any]) -> None: ...

    def log_artifact(self, name: str, path: str | Path) -> None: ...

    def end_run(self, status: str = "FINISHED") -> RunRecord | None: ...


def _sanitise(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _FORBIDDEN_PARAMS:
            out[key] = "<omitted: raw measurement data is never tracked>"
            continue
        if isinstance(value, str | int | float | bool | type(None)):
            out[key] = value
        elif isinstance(value, dict | list | tuple):
            out[key] = json.loads(json.dumps(value, default=str))
        else:
            out[key] = str(value)
    return out


@dataclass
class FileTracker:
    """JSON run store under ``artifacts/tracking/<experiment>/<run_id>.json``."""

    cfg: ExperimentConfig
    _run: RunRecord | None = field(default=None, init=False, repr=False)

    @property
    def root(self) -> Path:
        return Path(self.cfg.tracking.dir) / _safe(self.cfg.tracking.experiment_name)

    def start_run(self, name: str, **tags: str) -> RunRecord:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}-{_safe(name)}"
        environment = environment_fingerprint()
        self._run = RunRecord(
            run_id=run_id,
            experiment_name=self.cfg.tracking.experiment_name,
            started_at_utc=datetime.now(UTC).isoformat(),
            git_revision=environment.get("git_revision"),
            data_fingerprint=self.cfg.data_fingerprint(),
            seed=self.cfg.seed,
            tags={**tags, "run_name": name},
            environment=environment,
        )
        logger.info("Tracking run started: %s", run_id)
        return self._run

    def log_params(self, params: dict[str, Any]) -> None:
        self._require_run().params.update(_sanitise(params))

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        self._require_run().metrics.update(_sanitise(metrics))

    def log_artifact(self, name: str, path: str | Path) -> None:
        """Record an artifact's *relative* path; the file is not copied.

        Copying would double every model bundle on disk. The path is stored
        relative to the project root so a run record never carries a developer's
        home directory into a committed file.
        """
        run = self._require_run()
        resolved = Path(path)
        try:
            relative = str(resolved.resolve().relative_to(Path(self.cfg.paths.root).resolve()))
        except ValueError:
            relative = str(resolved)
        target = run.figures if resolved.suffix in (".png", ".pdf", ".svg") else run.artifacts
        if resolved.suffix in (".csv", ".parquet"):
            target = run.tables
        target[name] = relative

    def log_features(self, features: list[str]) -> None:
        self._require_run().features = list(features)

    def end_run(self, status: str = "FINISHED") -> RunRecord | None:
        run = self._run
        if run is None:
            return None
        run.status = status
        run.finished_at_utc = datetime.now(UTC).isoformat()
        path = self.root / f"{run.run_id}.json"
        save_json(run.to_dict(), path)
        logger.info("Tracking run %s -> %s (%s)", run.run_id, path.name, status)
        self._run = None
        return run

    def list_runs(self, *, limit: int = 100) -> list[RunRecord]:
        if not self.root.is_dir():
            return []
        runs: list[RunRecord] = []
        for path in sorted(self.root.glob("*.json"), reverse=True)[:limit]:
            payload = json.loads(path.read_text(encoding="utf-8"))
            known = {k: v for k, v in payload.items() if k in RunRecord.__dataclass_fields__}
            runs.append(RunRecord(**known))
        return runs

    def _require_run(self) -> RunRecord:
        if self._run is None:
            raise RuntimeError("No tracking run is active. Call start_run() first.")
        return self._run


@dataclass
class MLflowTracker:
    """MLflow-backed tracker. Only constructed when MLflow is installed."""

    cfg: ExperimentConfig
    _active: Any = field(default=None, init=False, repr=False)
    _record: RunRecord | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import mlflow  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "tracking.backend='mlflow' but MLflow is not installed. Install it "
                "(`pip install mlflow`) or use the default file backend."
            ) from exc

    def start_run(self, name: str, **tags: str) -> RunRecord:
        import mlflow

        uri = (
            self.cfg.tracking.mlflow_tracking_uri
            or f"file:{Path(self.cfg.tracking.dir) / 'mlruns'}"
        )
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(self.cfg.tracking.experiment_name)
        self._active = mlflow.start_run(run_name=name, tags=dict(tags))
        environment = environment_fingerprint()
        self._record = RunRecord(
            run_id=self._active.info.run_id,
            experiment_name=self.cfg.tracking.experiment_name,
            started_at_utc=datetime.now(UTC).isoformat(),
            git_revision=environment.get("git_revision"),
            data_fingerprint=self.cfg.data_fingerprint(),
            seed=self.cfg.seed,
            tags={**tags, "run_name": name},
            environment=environment,
        )
        return self._record

    def log_params(self, params: dict[str, Any]) -> None:
        import mlflow

        clean = _sanitise(params)
        mlflow.log_params({k: str(v)[:500] for k, v in clean.items()})
        if self._record:
            self._record.params.update(clean)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        import mlflow

        clean = _sanitise(metrics)
        numeric = {k: float(v) for k, v in clean.items() if isinstance(v, int | float)}
        if numeric:
            mlflow.log_metrics(numeric)
        if self._record:
            self._record.metrics.update(clean)

    def log_artifact(self, name: str, path: str | Path) -> None:
        import mlflow

        if Path(path).exists():
            mlflow.log_artifact(str(path))
        if self._record:
            self._record.artifacts[name] = str(path)

    def end_run(self, status: str = "FINISHED") -> RunRecord | None:
        import mlflow

        mlflow.end_run(status=status)
        record = self._record
        if record is not None:
            record.status = status
            record.finished_at_utc = datetime.now(UTC).isoformat()
        self._active = None
        self._record = None
        return record


def build_tracker(cfg: ExperimentConfig) -> ExperimentTracker:
    """The configured tracker, falling back to the file backend with a warning.

    Falls back rather than raising: a missing optional dependency must not stop
    a training run, and a run that is tracked to a file is strictly better than
    a run that did not happen.
    """
    if cfg.tracking.backend == "mlflow":
        try:
            return MLflowTracker(cfg=cfg)
        except RuntimeError as exc:
            logger.warning("%s Falling back to the file tracker.", exc)
    return FileTracker(cfg=cfg)


@contextmanager
def tracked_run(cfg: ExperimentConfig, name: str, **tags: str) -> Iterator[ExperimentTracker]:
    """Run a block inside a tracked run, recording failure as failure."""
    tracker = build_tracker(cfg)
    tracker.start_run(name, **tags)
    try:
        yield tracker
    except Exception:
        tracker.end_run(status="FAILED")
        raise
    else:
        tracker.end_run(status="FINISHED")


def compare_runs(cfg: ExperimentConfig, *, limit: int = 20) -> list[dict[str, Any]]:
    """A flat comparison table over the file-backed runs."""
    tracker = FileTracker(cfg=cfg)
    rows: list[dict[str, Any]] = []
    for run in tracker.list_runs(limit=limit):
        row: dict[str, Any] = {
            "run_id": run.run_id,
            "status": run.status,
            "started_at_utc": run.started_at_utc,
            "model_type": run.model_type,
            "git_revision": run.git_revision,
            "data_fingerprint": run.data_fingerprint,
            "seed": run.seed,
        }
        for key, value in run.metrics.items():
            if isinstance(value, int | float):
                row[f"metric.{key}"] = value
        rows.append(row)
    return rows


def _safe(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in text)[:64]
