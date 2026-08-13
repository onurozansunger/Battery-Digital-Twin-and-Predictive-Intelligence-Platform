"""Milestone 3 pipelines: fleet batch, monitoring, registry and reporting.

Same shape as :mod:`battery_rul.pipelines.milestone_2` — one module holding the
implementations, thin ``python -m`` aliases per documented command — because
splitting eight coherent stages across eight near-empty files scatters a pipeline
for the sake of the command names.

Every stage is config-driven, logs structurally, returns a meaningful exit code,
and writes its artifacts under ``artifacts/`` and ``reports/milestone_3/``.
Nothing here trains a model during serving, and nothing promotes a model without
being told to.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from battery_rul.config import ExperimentConfig, load_config
from battery_rul.digital_twin.service import BatteryDigitalTwinService
from battery_rul.fleet.demo import DemoFleetSpec, demo_fleet_identity, ingest_demo_fleet
from battery_rul.fleet.domain import (
    FleetDriftStatus,
    FleetSnapshot,
    MonitoringStatus,
)
from battery_rul.fleet.inference import FleetInferenceService, new_batch_id
from battery_rul.fleet.ingestion import BatteryHistoryInput, FleetIngestor
from battery_rul.fleet.ranking import rank_batteries
from battery_rul.monitoring.alerts import AlertPolicy
from battery_rul.monitoring.domain import (
    MONITORING_SNAPSHOT_SCHEMA_VERSION,
    MonitoringSnapshot,
    PerformanceStatus,
)
from battery_rul.monitoring.drift import detect_feature_drift
from battery_rul.monitoring.performance import (
    evaluate_delayed_labels,
    prediction_records_from_snapshot,
)
from battery_rul.monitoring.prediction_drift import (
    detect_prediction_drift,
    summarise_predictions,
)
from battery_rul.monitoring.reference import (
    ReferenceDistribution,
    build_reference_distribution,
    load_reference,
    save_reference,
)
from battery_rul.observability.logging import bind_context, log_event
from battery_rul.persistence import build_repository
from battery_rul.registry.promotion import PromotionDecision, PromotionGate
from battery_rul.registry.store import FileModelRegistry, ModelStage
from battery_rul.utils.io import save_json, write_table
from battery_rul.utils.logging import get_logger, log_section, setup_logging

logger = get_logger(__name__)

__all__ = [
    "build_reference",
    "evaluate_promotion_stage",
    "generate_fleet_report",
    "main",
    "promote_model",
    "register_model",
    "rollback_model",
    "run_fleet_batch",
    "run_monitoring",
]

MILESTONE_3_REPORTS = "milestone_3"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _reports_dir(cfg: ExperimentConfig) -> Path:
    path = cfg.paths.reports_dir / MILESTONE_3_REPORTS
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relative(path: Path, cfg: ExperimentConfig) -> str:
    """Artifact paths are published relative to the project root, never absolute."""
    try:
        return str(Path(path).resolve().relative_to(Path(cfg.paths.root).resolve()))
    except ValueError:
        return Path(path).name


def _ingest(
    cfg: ExperimentConfig,
    *,
    source: str,
    fleet_id: str,
    path: str | None = None,
    demo_size: int = 24,
) -> tuple[Any, list[BatteryHistoryInput], Any]:
    """Resolve a fleet source into ``(ingestion_result, histories, identity)``."""
    ingestor = FleetIngestor(cfg=cfg)
    if source == "demo":
        from battery_rul.fleet.demo import resolve_construction

        spec = DemoFleetSpec(fleet_id=fleet_id, n_batteries=demo_size)
        result, histories = ingest_demo_fleet(cfg, spec)
        identity = demo_fleet_identity(spec, construction=resolve_construction(cfg, spec))
        return result, histories, identity
    if source == "processed":
        result, histories = ingestor.from_processed_cycles(
            fleet_id, path=Path(path) if path else None
        )
        return result, histories, None
    if source == "file":
        if not path:
            raise ValueError("--path is required for --source file")
        result, histories = ingestor.from_file(fleet_id, path)
        return result, histories, None
    if source == "directory":
        if not path:
            raise ValueError("--path is required for --source directory")
        result, histories = ingestor.from_directory(fleet_id, path)
        return result, histories, None
    raise ValueError(f"Unknown fleet source {source!r}: use demo, processed, file or directory.")


# ---------------------------------------------------------------------------
# Stage — reference distribution
# ---------------------------------------------------------------------------
def build_reference(cfg: ExperimentConfig, *, reference_id: str | None = None) -> dict[str, Any]:
    """Build the drift reference from the TRAINING partition only.

    Reads the Milestone 2 multi-task dataset, which carries the ``split`` column
    assigned by the Milestone 1 battery-holdout partitioning, and keeps only the
    rows the models were actually fitted on. The test partition is never read
    here: a drift reference fitted on it would fold the held-out result into the
    serving machinery.
    """
    log_section(logger, "milestone 3 — build drift reference")
    dataset = cfg.paths.processed_dir / "multitask_dataset.parquet"
    if not dataset.is_file():
        raise FileNotFoundError(
            f"No multi-task dataset at {dataset}. Run "
            "`python -m battery_rul.pipelines.run_milestone_2` first."
        )
    frame = pd.read_parquet(dataset)
    if "split" not in frame.columns:
        raise ValueError(
            f"{dataset.name} has no 'split' column, so the training partition cannot be "
            "isolated. Rebuild it with the Milestone 2 pipeline."
        )

    partitions = ["train"] if cfg.monitoring.reference_partition == "train" else ["train", "val"]
    reference_frame = frame.loc[frame["split"].isin(partitions)]
    if reference_frame.empty:
        raise ValueError(f"No rows in partition(s) {partitions}; cannot build a reference.")

    feature_names = _bundle_feature_names(cfg) or [
        c for c in reference_frame.columns if pd.api.types.is_numeric_dtype(reference_frame[c])
    ]
    feature_names = [c for c in feature_names if c in reference_frame.columns]

    reference = build_reference_distribution(
        reference_frame,
        cfg,
        feature_names=feature_names,
        reference_id=reference_id,
        partition="+".join(partitions),
        notes=(
            "Fitted on the Milestone 1 battery-holdout training partition. The test "
            "partition is excluded by construction."
        ),
    )
    path = save_reference(reference, cfg)
    return {
        "reference_id": reference.reference_id,
        "partition": reference.partition,
        "n_rows": reference.n_rows,
        "n_batteries": reference.n_batteries,
        "n_features": len(reference.feature_stats),
        "fingerprint": reference.fingerprint(),
        "path": _relative(path, cfg),
    }


def _bundle_feature_names(cfg: ExperimentConfig) -> list[str]:
    """Feature names from whichever bundle metadata is available."""
    from battery_rul.utils.io import load_json

    for directory in (cfg.artifacts.rul_dir, cfg.artifacts.risk_dir, cfg.artifacts.soh_dir):
        metadata = Path(directory) / "metadata.json"
        if metadata.is_file():
            payload = load_json(metadata)
            names = payload.get("feature_names") or []
            if names:
                return [str(n) for n in names]
    logger.warning("No bundle metadata found; the reference will cover every numeric column.")
    return []


# ---------------------------------------------------------------------------
# Stage — fleet batch
# ---------------------------------------------------------------------------
def run_fleet_batch(
    cfg: ExperimentConfig,
    *,
    fleet_id: str | None = None,
    source: str = "processed",
    path: str | None = None,
    demo_size: int = 24,
    persist: bool = True,
    write_reports: bool = True,
) -> dict[str, Any]:
    """Score a fleet offline and write the snapshot, rankings and plans."""
    log_section(logger, "milestone 3 — fleet batch")
    fleet = fleet_id or cfg.fleet.default_fleet_id
    batch_id = new_batch_id()
    started_at = datetime.now(UTC).isoformat()

    with bind_context(fleet_id=fleet, batch_id=batch_id):
        ingestion, histories, identity = _ingest(
            cfg, source=source, fleet_id=fleet, path=path, demo_size=demo_size
        )
        service = FleetInferenceService.create(cfg)
        result = service.run_batch(
            fleet, histories, ingestion=ingestion, batch_id=batch_id, identity=identity
        )
        snapshot = result.snapshot

        stored: dict[str, Any] = {}
        if persist:
            repository = build_repository(cfg)
            repository.save_fleet_snapshot(snapshot)
            records = prediction_records_from_snapshot(snapshot)
            repository.save_prediction_records(records)
            repository.save_batch(
                batch_id,
                {
                    "fleet_id": fleet,
                    "started_at_utc": started_at,
                    "finished_at_utc": datetime.now(UTC).isoformat(),
                    "status": "completed",
                    "source": source,
                    "battery_count": snapshot.battery_count,
                    "success_count": snapshot.successfully_processed_count,
                    "failed_count": snapshot.failed_count,
                },
            )
            stored = {"snapshot_id": snapshot.snapshot_id, "prediction_records": len(records)}

        artifacts: dict[str, str] = {}
        if write_reports:
            artifacts = _write_fleet_artifacts(cfg, snapshot)

        log_event(
            logger,
            "fleet_batch_persisted",
            battery_count=snapshot.battery_count,
            success_count=snapshot.successfully_processed_count,
            failed_count=snapshot.failed_count,
        )

    return {
        "fleet_id": fleet,
        "batch_id": batch_id,
        "snapshot_id": snapshot.snapshot_id,
        "source": source,
        "is_demo_data": snapshot.identity.is_demo_data,
        "battery_count": snapshot.battery_count,
        "successfully_processed_count": snapshot.successfully_processed_count,
        "failed_count": snapshot.failed_count,
        "insufficient_data_count": snapshot.insufficient_data_count,
        "critical_count": snapshot.maintenance_summary.critical_count,
        "median_soh": snapshot.fleet_statistics.soh_median,
        "median_rul": snapshot.fleet_statistics.rul_median,
        "processing_duration_ms": snapshot.processing_duration_ms,
        "active_model_version": snapshot.model_metadata.active_model_version,
        "persisted": stored,
        "artifacts": artifacts,
    }


def _write_fleet_artifacts(cfg: ExperimentConfig, snapshot: FleetSnapshot) -> dict[str, str]:
    """Write the snapshot, ranking, maintenance plan and replacement plan."""
    fleet_dir = Path(cfg.artifacts.fleet_dir)
    written: dict[str, str] = {}

    snapshot_path = fleet_dir / "snapshots" / f"{snapshot.snapshot_id}.json"
    save_json(snapshot.to_json_dict(), snapshot_path)
    written["fleet_snapshot"] = _relative(snapshot_path, cfg)

    ranking = pd.DataFrame(
        [
            {
                "rank": index + 1,
                "battery_id": record.battery_id,
                "priority": record.priority.value,
                "priority_score": record.priority_score,
                "failure_risk": record.failure_risk,
                "risk_is_experimental": record.risk_is_experimental,
                "predicted_rul": record.predicted_rul,
                "rul_lower_bound": record.rul_lower_bound,
                "rul_upper_bound": record.rul_upper_bound,
                "measured_soh": record.measured_soh,
                "health_class": record.health_class,
                "fade_trend_pct_per_10": record.fade_trend_pct_per_10,
                "data_quality_class": record.data_quality_class,
                "recommended_action": record.recommended_action,
                "inspection_window_cycles": (
                    record.priority_record.inspection.recommended_cycles
                    if record.priority_record and record.priority_record.inspection
                    else None
                ),
                "model_version": record.model_version,
            }
            for index, record in enumerate(
                rank_batteries(snapshot.batteries, by="priority", include_unevaluated=True)
            )
        ]
    )
    ranking_path = fleet_dir / "rankings" / f"{snapshot.snapshot_id}_ranking.csv"
    write_table(ranking, ranking_path)
    written["fleet_ranking"] = _relative(ranking_path, cfg)

    maintenance = pd.DataFrame(
        [
            {
                "battery_id": record.battery_id,
                "priority": record.priority.value,
                "priority_score": record.priority_score,
                "recommended_action": record.recommended_action,
                "inspection_cycles": (
                    record.priority_record.inspection.recommended_cycles
                    if record.priority_record and record.priority_record.inspection
                    else None
                ),
                "inspection_estimated_days": (
                    record.priority_record.inspection.estimated_days
                    if record.priority_record and record.priority_record.inspection
                    else None
                ),
                "triggered_rules": (
                    " | ".join(record.priority_record.triggered_rules)
                    if record.priority_record
                    else ""
                ),
            }
            for record in snapshot.batteries
        ]
    )
    maintenance_path = fleet_dir / "maintenance_plans" / f"{snapshot.snapshot_id}_maintenance.csv"
    write_table(maintenance, maintenance_path)
    written["maintenance_plan"] = _relative(maintenance_path, cfg)

    replacement = pd.DataFrame(
        [
            {
                "battery_id": record.battery_id,
                "replacement_candidate": record.replacement.replacement_candidate,
                "replacement_horizon": record.replacement.replacement_horizon.value,
                "horizon_cycles": record.replacement.horizon_cycles,
                "confidence": record.replacement.confidence,
                "planning_category": record.replacement.planning_category,
                "rul_point": record.replacement.rul_point,
                "rul_lower_bound": record.replacement.rul_lower_bound,
                "rul_upper_bound": record.replacement.rul_upper_bound,
            }
            for record in snapshot.batteries
            if record.replacement is not None
        ]
    )
    replacement_path = fleet_dir / "replacement_plans" / f"{snapshot.snapshot_id}_replacement.csv"
    write_table(replacement, replacement_path)
    written["replacement_plan"] = _relative(replacement_path, cfg)

    summary_path = _reports_dir(cfg) / "fleet_summary.json"
    save_json(
        {
            "summary": snapshot.summary.model_dump(mode="json"),
            "health_distribution": snapshot.health_distribution.model_dump(mode="json"),
            "risk_distribution": snapshot.risk_distribution.model_dump(mode="json"),
            "maintenance_summary": snapshot.maintenance_summary.model_dump(mode="json"),
            "replacement_summary": snapshot.replacement_summary.model_dump(mode="json"),
            "workload_forecast": snapshot.workload_forecast.model_dump(mode="json"),
            "fleet_statistics": snapshot.fleet_statistics.model_dump(mode="json"),
            "data_quality": snapshot.data_quality.model_dump(mode="json"),
            "model_metadata": snapshot.model_metadata.model_dump(mode="json"),
            "warnings": snapshot.warnings,
            "disclaimer": snapshot.disclaimer,
        },
        summary_path,
    )
    written["fleet_summary"] = _relative(summary_path, cfg)
    return written


# ---------------------------------------------------------------------------
# Stage — monitoring
# ---------------------------------------------------------------------------
def run_monitoring(
    cfg: ExperimentConfig,
    *,
    fleet_id: str | None = None,
    source: str = "processed",
    path: str | None = None,
    demo_size: int = 24,
    reference_id: str | None = None,
    persist: bool = True,
    rows_per_battery: int | None = None,
) -> dict[str, Any]:
    """Run the full monitoring suite over a fresh fleet batch.

    Order matters: the fleet is scored first, because data quality, prediction
    drift and the performance join all describe *that* batch. Feature drift is
    computed from the same batch's engineered features, so every finding in the
    resulting snapshot refers to one identifiable set of inputs.
    """
    log_section(logger, "milestone 3 — monitoring run")
    fleet = fleet_id or cfg.fleet.default_fleet_id
    batch_id = new_batch_id()
    warnings: list[str] = []

    with bind_context(fleet_id=fleet, batch_id=batch_id):
        ingestion, histories, identity = _ingest(
            cfg, source=source, fleet_id=fleet, path=path, demo_size=demo_size
        )
        twin = BatteryDigitalTwinService.create(cfg)
        service = FleetInferenceService.create(cfg, twin=twin)
        batch = service.run_batch(
            fleet, histories, ingestion=ingestion, batch_id=batch_id, identity=identity
        )
        snapshot = batch.snapshot
        model_version = snapshot.model_metadata.active_model_version

        # --- feature drift -------------------------------------------------
        feature_drift = None
        reference: ReferenceDistribution | None = None
        try:
            reference = load_reference(cfg, reference_id)
        except (FileNotFoundError, ValueError) as exc:
            warnings.append(
                f"Feature drift was not assessed: {exc} Build a reference with "
                "`python -m battery_rul.pipelines.build_reference`."
            )
        if reference is not None:
            current = _current_features(twin, histories, rows_per_battery)
            if current.empty:
                warnings.append(
                    "No engineered feature rows could be produced from this batch, so "
                    "feature drift was not assessed."
                )
            else:
                feature_drift = detect_feature_drift(
                    current,
                    reference,
                    cfg,
                    current_window=f"{len(current)} rows from batch {batch_id}",
                )

        # --- prediction drift ----------------------------------------------
        current_predictions = summarise_predictions(snapshot.batteries, cfg)
        prediction_drift = None
        if reference is not None and reference.prediction_stats:
            prediction_drift = detect_prediction_drift(
                current_predictions,
                reference.prediction_stats,
                cfg,
                model_version=model_version,
                reference_id=reference.reference_id,
            )
        else:
            warnings.append(
                "Prediction drift was not assessed: the reference artifact carries no "
                "prediction distribution. Seed one with "
                "`python -m battery_rul.pipelines.run_monitoring --set-prediction-reference`."
            )

        # --- performance with delayed labels --------------------------------
        repository = build_repository(cfg)
        predictions = repository.list_prediction_records(model_version=model_version)
        if not predictions:
            predictions = prediction_records_from_snapshot(snapshot)
        labels = repository.list_outcome_labels()
        performance = evaluate_delayed_labels(predictions, labels, cfg, model_version=model_version)

        # --- alerts ----------------------------------------------------------
        alerts = AlertPolicy(cfg=cfg).build(
            fleet_snapshot=snapshot,
            feature_drift=feature_drift,
            prediction_drift=prediction_drift,
            performance=performance,
            readiness=service.readiness(),
            model_version=model_version,
        )

        # --- assemble and persist --------------------------------------------
        report_paths = _write_monitoring_artifacts(
            cfg, snapshot, feature_drift, prediction_drift, performance, alerts
        )
        statuses = [snapshot.data_quality.status]
        if feature_drift is not None:
            statuses.append(feature_drift.status)
        if prediction_drift is not None:
            statuses.append(prediction_drift.status)
        if performance.status is PerformanceStatus.DEGRADED:
            statuses.append(MonitoringStatus.CRITICAL)
        elif performance.status is PerformanceStatus.WARNING:
            statuses.append(MonitoringStatus.WARNING)

        monitoring = MonitoringSnapshot(
            snapshot_id=f"mon-{batch_id}",
            schema_version=MONITORING_SNAPSHOT_SCHEMA_VERSION,
            fleet_id=fleet,
            model_version=model_version,
            data_version=snapshot.data_fingerprint,
            batch_id=batch_id,
            input_count=snapshot.battery_count,
            success_count=snapshot.successfully_processed_count,
            failed_count=snapshot.failed_count,
            data_quality_summary=snapshot.data_quality.model_dump(mode="json"),
            feature_drift_summary=(
                _drift_summary(feature_drift) if feature_drift is not None else {}
            ),
            prediction_drift_summary=(
                prediction_drift.model_dump(mode="json") if prediction_drift is not None else {}
            ),
            performance_summary=performance.model_dump(mode="json"),
            alert_summary=_alert_summary(alerts),
            alerts=alerts,
            overall_status=MonitoringStatus.worst(statuses),
            warnings=warnings,
            report_paths=report_paths,
        )

        # The drift verdict only exists after the batch has been scored, so the
        # snapshot is re-saved with it. Both the detail block and the summary's
        # headline field are updated: a report that reads one and a dashboard
        # that reads the other must not disagree.
        drift = FleetDriftStatus(
            status=monitoring.overall_status,
            feature_drift_status=(
                feature_drift.status if feature_drift is not None else MonitoringStatus.UNKNOWN
            ),
            prediction_drift_status=(
                prediction_drift.status
                if prediction_drift is not None
                else MonitoringStatus.UNKNOWN
            ),
            n_features_tested=(feature_drift.n_features_tested if feature_drift else 0),
            n_features_drifted=(feature_drift.n_features_drifted if feature_drift else 0),
            top_drifted_features=(feature_drift.drifted_features[:10] if feature_drift else []),
            reference_id=reference.reference_id if reference else None,
            monitoring_snapshot_id=monitoring.snapshot_id,
        )
        snapshot = snapshot.model_copy(
            update={
                "drift_status": drift,
                "summary": snapshot.summary.model_copy(update={"drift_status": drift.status}),
            }
        )

        if persist:
            repository.save_monitoring_snapshot(monitoring)
            repository.save_alerts(alerts)
            repository.save_fleet_snapshot(snapshot)

        log_event(
            logger,
            "monitoring_run_completed",
            status=monitoring.overall_status.value,
            alerts=len(alerts),
            drifted_features=(feature_drift.n_features_drifted if feature_drift else 0),
            performance_status=performance.status.value,
        )

    return {
        "fleet_id": fleet,
        "batch_id": batch_id,
        "monitoring_snapshot_id": monitoring.snapshot_id,
        "overall_status": monitoring.overall_status.value,
        "data_quality_status": snapshot.data_quality.status.value,
        "feature_drift_status": (feature_drift.status.value if feature_drift else "UNKNOWN"),
        "n_features_tested": feature_drift.n_features_tested if feature_drift else 0,
        "n_features_drifted": feature_drift.n_features_drifted if feature_drift else 0,
        "prediction_drift_status": (
            prediction_drift.status.value if prediction_drift else "UNKNOWN"
        ),
        "performance_status": performance.status.value,
        "label_coverage": performance.label_coverage,
        "n_alerts": len(alerts),
        "warnings": warnings,
        "report_paths": report_paths,
    }


def _current_features(
    twin: BatteryDigitalTwinService,
    histories: list[BatteryHistoryInput],
    rows_per_battery: int | None,
) -> pd.DataFrame:
    """Engineered features for the current batch, via the serving pipeline.

    Uses the twin's own feature preparation rather than rebuilding features, so
    drift is measured on the feature space the model actually sees.
    """
    frames: list[pd.DataFrame] = []
    for item in histories:
        try:
            features = twin.prepare_features(item.battery_id, item.history)
        except Exception as exc:  # noqa: BLE001 - one cell must not sink monitoring
            logger.warning("Feature preparation failed for %s: %s", item.battery_id, exc)
            continue
        if features is None or features.empty:
            continue
        frames.append(features.tail(rows_per_battery) if rows_per_battery else features)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _drift_summary(report: Any) -> dict[str, Any]:
    """The drift report without its per-feature rows, for the snapshot document."""
    payload = report.model_dump(mode="json")
    payload["results"] = [
        r for r in payload.get("results", []) if r.get("drift_detected") or not r.get("reliable")
    ][:50]
    payload["results_truncated"] = True
    return payload


def _alert_summary(alerts: list[Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for alert in alerts:
        counts[alert.severity.value] = counts.get(alert.severity.value, 0) + 1
    return {
        "total": len(alerts),
        "by_severity": counts,
        "types": sorted({alert.type.value for alert in alerts}),
    }


def _write_monitoring_artifacts(
    cfg: ExperimentConfig,
    snapshot: FleetSnapshot,
    feature_drift: Any,
    prediction_drift: Any,
    performance: Any,
    alerts: list[Any],
) -> dict[str, str]:
    root = Path(cfg.artifacts.monitoring_dir)
    reports = _reports_dir(cfg)
    written: dict[str, str] = {}

    quality_path = reports / "data_quality_report.json"
    save_json(snapshot.data_quality.model_dump(mode="json"), quality_path)
    written["data_quality_report"] = _relative(quality_path, cfg)

    if feature_drift is not None:
        path = reports / "feature_drift_report.json"
        save_json(feature_drift.model_dump(mode="json"), path)
        written["feature_drift_report"] = _relative(path, cfg)
    if prediction_drift is not None:
        path = reports / "prediction_drift_report.json"
        save_json(prediction_drift.model_dump(mode="json"), path)
        written["prediction_drift_report"] = _relative(path, cfg)

    performance_path = root / "performance_reports" / f"{snapshot.snapshot_id}.json"
    save_json(performance.model_dump(mode="json"), performance_path)
    written["model_performance_report"] = _relative(performance_path, cfg)
    save_json(performance.model_dump(mode="json"), reports / "model_performance_report.json")

    alerts_path = root / "alerts" / f"{snapshot.snapshot_id}.json"
    save_json([a.model_dump(mode="json") for a in alerts], alerts_path)
    written["alerts"] = _relative(alerts_path, cfg)
    save_json([a.model_dump(mode="json") for a in alerts], reports / "active_alerts.json")
    return written


def set_prediction_reference(
    cfg: ExperimentConfig, *, fleet_id: str | None = None, reference_id: str | None = None
) -> dict[str, Any]:
    """Adopt the latest stored fleet snapshot as the prediction-drift reference.

    Explicit and manual on purpose. A reference that updated itself on every run
    would compare each batch with the previous one, which detects a step change
    once and then treats the new behaviour as normal for ever.
    """
    repository = build_repository(cfg)
    fleet = fleet_id or cfg.fleet.default_fleet_id
    snapshot = repository.latest_fleet_snapshot(fleet)
    if snapshot is None:
        raise FileNotFoundError(
            f"No stored fleet snapshot for {fleet}. Run "
            "`python -m battery_rul.pipelines.run_fleet_batch` first."
        )
    reference = load_reference(cfg, reference_id)
    reference.prediction_stats = dict(summarise_predictions(snapshot.batteries, cfg))
    reference.notes = (
        f"{reference.notes} Prediction reference adopted from fleet snapshot "
        f"{snapshot.snapshot_id} ({snapshot.successfully_processed_count} scored cells)."
    ).strip()
    path = save_reference(reference, cfg)
    return {
        "reference_id": reference.reference_id,
        "from_snapshot": snapshot.snapshot_id,
        "n_scored": snapshot.successfully_processed_count,
        "path": _relative(path, cfg),
    }


# ---------------------------------------------------------------------------
# Stage — registry
# ---------------------------------------------------------------------------
def register_model(
    cfg: ExperimentConfig,
    *,
    model_name: str,
    model_version: str,
    bundle: str,
    validation_status: str = "UNVALIDATED",
    notes: str = "",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Register a built bundle as a CANDIDATE."""
    log_section(logger, "milestone 3 — register model")
    registry = FileModelRegistry(cfg=cfg)
    bundle_path = Path(bundle)
    if not bundle_path.is_absolute():
        bundle_path = Path(cfg.paths.root) / bundle_path

    metrics, calibration, uncertainty = _bundle_metrics(bundle_path)
    entry = registry.register(
        model_name=model_name,
        model_version=model_version,
        bundle_path=bundle_path,
        metrics=metrics,
        calibration_metrics=calibration,
        uncertainty_metrics=uncertainty,
        validation_status=validation_status,
        notes=notes,
        overwrite=overwrite,
    )
    return {
        "model": entry.key,
        "stage": entry.stage.value,
        "bundle_path": entry.bundle_path,
        "artifact_checksum": entry.artifact_checksum,
        "validation_status": entry.validation_status,
        "registry_file": _relative(registry.path, cfg),
    }


def _bundle_metrics(bundle_path: Path) -> tuple[dict, dict, dict]:
    """Pull metrics out of a bundle's metadata into the registry's three buckets."""
    from battery_rul.utils.io import load_json

    metadata_file = bundle_path / "metadata.json"
    if not metadata_file.is_file():
        return {}, {}, {}
    payload = load_json(metadata_file)
    metrics = payload.get("metrics") or {}
    calibration = metrics.get("out_of_fold_calibrated") or metrics.get("test_calibrated") or {}
    uncertainty = metrics.get("out_of_fold_coverage") or metrics.get("test_coverage") or {}
    return metrics, calibration, uncertainty


def evaluate_promotion_stage(
    cfg: ExperimentConfig,
    *,
    model_name: str,
    model_version: str,
    smoke_test: bool | None = None,
    contract_tests: bool | None = None,
    unit_tests: bool | None = None,
    leakage_check: bool | None = None,
) -> dict[str, Any]:
    """Evaluate the promotion gate and write the report. Never promotes."""
    log_section(logger, "milestone 3 — evaluate promotion gate")
    registry = FileModelRegistry(cfg=cfg)
    candidate = registry.get(model_name, model_version)
    if candidate is None:
        raise ValueError(f"{model_name}:{model_version} is not registered.")
    production = next(
        (
            e
            for e in registry.list_models(model_name=model_name, stage=ModelStage.PRODUCTION)
            if e.model_version != model_version
        ),
        None,
    )
    report = PromotionGate(cfg=cfg).evaluate(
        candidate,
        production,
        smoke_test_passed=smoke_test,
        contract_tests_passed=contract_tests,
        unit_tests_passed=unit_tests,
        leakage_check_passed=leakage_check,
    )
    path = (
        Path(cfg.registry.dir)
        / "promotion_reports"
        / f"{model_name}_{model_version}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    save_json(report.to_dict(), path)
    save_json(report.to_dict(), _reports_dir(cfg) / "model_promotion_report.json")
    logger.info("Promotion gate: %s (%s)", report.decision.value, "; ".join(report.reasons[:3]))
    return {**report.to_dict(), "report_path": _relative(path, cfg)}


def promote_model(
    cfg: ExperimentConfig,
    *,
    model_name: str,
    model_version: str,
    by: str,
    reason: str = "",
    dry_run: bool = False,
    force: bool = False,
    skip_gate: bool = False,
) -> dict[str, Any]:
    """Promote a version to PRODUCTION after re-checking the gate.

    The gate is re-evaluated here even if it was evaluated in CI: the artifact
    may have changed since, and a promotion that trusts an old verdict is a
    promotion of whatever is on disk now.
    """
    log_section(logger, "milestone 3 — promote model")
    gate: dict[str, Any] | None = None
    if not skip_gate:
        gate = evaluate_promotion_stage(cfg, model_name=model_name, model_version=model_version)
        if gate["decision"] == PromotionDecision.REJECTED.value and not force:
            return {
                "promoted": False,
                "decision": gate["decision"],
                "reasons": gate["reasons"],
                "hint": "Fix the failing gates, or pass --force with an explicit reason.",
            }

    if dry_run:
        return {
            "promoted": False,
            "dry_run": True,
            "decision": gate["decision"] if gate else "SKIPPED",
            "would_promote": f"{model_name}:{model_version}",
        }

    registry = FileModelRegistry(cfg=cfg)
    entry = registry.promote(model_name, model_version, by=by, reason=reason, force=force)
    return {
        "promoted": True,
        "model": entry.key,
        "stage": entry.stage.value,
        "promoted_at_utc": entry.promoted_at_utc,
        "promoted_by": entry.promoted_by,
        "decision": gate["decision"] if gate else "SKIPPED",
        "forced": force,
        "activation": "next_service_start_or_explicit_reload",
    }


def rollback_model(
    cfg: ExperimentConfig, *, model_name: str, by: str, reason: str = ""
) -> dict[str, Any]:
    """Restore the previously live version of a model family."""
    log_section(logger, "milestone 3 — rollback model")
    registry = FileModelRegistry(cfg=cfg)
    before = registry.production_model(model_name)
    entry = registry.rollback(model_name, by=by, reason=reason)
    return {
        "rolled_back": True,
        "model": entry.key,
        "previous_production": before.key if before else None,
        "stage": entry.stage.value,
        "activation": "next_service_start_or_explicit_reload",
    }


# ---------------------------------------------------------------------------
# Stage — report
# ---------------------------------------------------------------------------
def generate_fleet_report(cfg: ExperimentConfig, *, fleet_id: str | None = None) -> dict[str, Any]:
    """Render the Markdown fleet report from stored snapshots.

    Reads persisted artifacts rather than recomputing anything: a report that
    recomputes its own numbers can disagree with the snapshot it claims to
    describe.
    """
    log_section(logger, "milestone 3 — fleet report")
    fleet = fleet_id or cfg.fleet.default_fleet_id
    repository = build_repository(cfg)
    snapshot = repository.latest_fleet_snapshot(fleet)
    if snapshot is None:
        raise FileNotFoundError(
            f"No stored fleet snapshot for {fleet}. Run "
            "`python -m battery_rul.pipelines.run_fleet_batch` first."
        )
    monitoring = repository.latest_monitoring_snapshot(fleet)
    alerts = repository.list_alerts(fleet, limit=25)

    from battery_rul.evaluation.reporting_m3 import write_fleet_report

    path = write_fleet_report(cfg, snapshot, monitoring, alerts)
    return {
        "fleet_id": fleet,
        "snapshot_id": snapshot.snapshot_id,
        "report_path": _relative(path, cfg),
        "monitoring_snapshot_id": monitoring.snapshot_id if monitoring else None,
        "n_alerts": len(alerts),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_STAGES = {
    "build-reference": lambda cfg, args: build_reference(cfg, reference_id=args.reference_id),
    "run-fleet-batch": lambda cfg, args: run_fleet_batch(
        cfg,
        fleet_id=args.fleet_id,
        source=args.source,
        path=args.path,
        demo_size=args.demo_size,
        persist=not args.no_persist,
    ),
    "run-monitoring": lambda cfg, args: (
        set_prediction_reference(cfg, fleet_id=args.fleet_id, reference_id=args.reference_id)
        if args.set_prediction_reference
        else run_monitoring(
            cfg,
            fleet_id=args.fleet_id,
            source=args.source,
            path=args.path,
            demo_size=args.demo_size,
            reference_id=args.reference_id,
            persist=not args.no_persist,
        )
    ),
    "generate-fleet-report": lambda cfg, args: generate_fleet_report(cfg, fleet_id=args.fleet_id),
    "register-model": lambda cfg, args: register_model(
        cfg,
        model_name=args.model_name,
        model_version=args.model_version,
        bundle=args.bundle,
        validation_status=args.validation_status,
        notes=args.notes,
        overwrite=args.overwrite,
    ),
    "evaluate-promotion": lambda cfg, args: evaluate_promotion_stage(
        cfg,
        model_name=args.model_name,
        model_version=args.model_version,
        smoke_test=args.smoke_test,
        contract_tests=args.contract_tests,
        unit_tests=args.unit_tests,
        leakage_check=args.leakage_check,
    ),
    "promote-model": lambda cfg, args: promote_model(
        cfg,
        model_name=args.model_name,
        model_version=args.model_version,
        by=args.by,
        reason=args.reason,
        dry_run=args.dry_run,
        force=args.force,
        skip_gate=args.skip_gate,
    ),
    "rollback-model": lambda cfg, args: rollback_model(
        cfg, model_name=args.model_name, by=args.by, reason=args.reason
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="battery-rul milestone-3", description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument("stage", choices=sorted(_STAGES), nargs="?", default="run-fleet-batch")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", default=[])
    parser.add_argument("--fleet-id", default=None)
    parser.add_argument(
        "--source",
        default="processed",
        choices=["processed", "file", "directory", "demo"],
        help="Fleet source. 'demo' generates clearly-labelled synthetic cells.",
    )
    parser.add_argument(
        "--path", default=None, help="File or directory for --source file/directory."
    )
    parser.add_argument("--demo-size", type=int, default=24)
    parser.add_argument("--reference-id", default=None)
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument(
        "--set-prediction-reference",
        action="store_true",
        help="Adopt the latest stored snapshot as the prediction-drift reference.",
    )
    # -- registry ----------------------------------------------------------
    parser.add_argument("--model-name", default="battery-rul")
    parser.add_argument("--model-version", default="1.0.0")
    parser.add_argument("--bundle", default="artifacts/rul")
    parser.add_argument("--validation-status", default="UNVALIDATED")
    parser.add_argument("--notes", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--by", default="cli", help="Who is performing this transition.")
    parser.add_argument("--reason", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    for name in ("smoke-test", "contract-tests", "unit-tests", "leakage-check"):
        attribute = name.replace("-", "_")
        parser.add_argument(f"--{name}", dest=attribute, action="store_true", default=None)
        parser.add_argument(f"--no-{name}", dest=attribute, action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point shared by every ``python -m battery_rul.pipelines.*`` alias."""
    args = build_parser().parse_args(argv)

    overrides: dict[str, Any] = {}
    for pair in args.set:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()

    path = Path(args.config)
    cfg = load_config(path if path.is_file() else None, overrides=overrides)
    cfg.paths.ensure()
    setup_logging(log_file=cfg.paths.reports_dir / "pipeline.log", force=True)

    try:
        result = _STAGES[args.stage](cfg, args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report and exit non-zero
        logger.exception("Stage %s failed: %s", args.stage, exc)
        return 1

    save_json(result, _reports_dir(cfg) / f"{args.stage}.json")
    logger.info("Stage %s complete.", args.stage)
    if isinstance(result, dict) and result.get("decision") == PromotionDecision.REJECTED.value:
        # A rejected gate is a successful evaluation, but the exit code has to
        # be actionable in a pipeline: 2 means "ran fine, verdict is no".
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    from battery_rul.pipelines.milestone_3 import main as _main

    sys.exit(_main())
