"""Fleet endpoints.

Built as a router factory rather than a module-level ``APIRouter`` so the
dependencies (fleet service, repository, configuration) are injected by
``create_app`` and a test can build the whole surface over fixture artifacts and
a temporary database.

Boundaries this module keeps
----------------------------
*Online means small.* Every POST here is a bounded, synchronous request capped
by ``fleet.max_batteries_per_request``. Large fleets go through the batch CLI;
the error says so rather than timing out.

*No filesystem paths in, no filesystem paths out.* A request cannot name a file,
a bundle or a directory, and responses carry relative artifact paths at most.

*Partial success is a 200.* A fleet where nine cells failed is a successful
request that returns nine failures. Turning it into a 500 would lose the 119
that worked.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from battery_rul.api.fleet_schemas import (
    AlertListResponse,
    CriticalBatteriesResponse,
    FleetPage,
    FleetRankRequest,
    FleetRankResponse,
    FleetRequest,
    FleetSnapshotResponse,
    FleetSummaryResponse,
    MaintenancePlanResponse,
    ModelListResponse,
    MonitoringRunResponse,
    PageMeta,
    PromotionRequest,
    ReplacementPlanResponse,
    RollbackRequest,
)
from battery_rul.config import ExperimentConfig
from battery_rul.fleet.domain import FleetBatteryRecord, FleetSnapshot, MaintenancePriority
from battery_rul.fleet.inference import FleetInferenceService
from battery_rul.fleet.ingestion import FleetIngestor
from battery_rul.fleet.ranking import rank_batteries
from battery_rul.observability.logging import bind_context
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["build_fleet_router"]


def _paginate(
    records: Sequence[FleetBatteryRecord], *, page: int, page_size: int, total: int | None = None
) -> FleetPage:
    start = (page - 1) * page_size
    return FleetPage(
        items=list(records[start : start + page_size]),
        pagination=PageMeta.build(
            page=page, page_size=page_size, total=total if total is not None else len(records)
        ),
    )


def _page_size(cfg: ExperimentConfig, requested: int | None) -> int:
    if requested is None:
        return cfg.fleet.page_size
    return min(int(requested), cfg.fleet.max_page_size)


def build_fleet_router(
    cfg: ExperimentConfig,
    *,
    get_fleet_service: Any,
    get_repository: Any,
) -> APIRouter:
    """Assemble the ``/v1/fleet`` and related routers."""
    router = APIRouter(prefix="/v1", tags=["fleet"])

    # -- request -> snapshot ------------------------------------------------
    def _snapshot_from_request(
        body: FleetRequest, service: FleetInferenceService, request: Request
    ) -> tuple[FleetSnapshot, Any]:
        limit = cfg.fleet.max_batteries_per_request
        if len(body.batteries) > limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"{len(body.batteries)} batteries submitted; this endpoint accepts "
                    f"at most {limit}. Use the batch pipeline "
                    "(`python -m battery_rul.pipelines.run_fleet_batch`) for larger fleets."
                ),
            )
        ingestion, histories = FleetIngestor(cfg=cfg).from_records(
            body.fleet_id,
            [(item.battery_id, item.to_frame()) for item in body.batteries],
            source="api",
        )
        request_id = getattr(request.state, "request_id", None)
        with bind_context(request_id=request_id, fleet_id=body.fleet_id):
            snapshot = service.create_fleet_snapshot(body.fleet_id, histories, ingestion=ingestion)
        return snapshot, ingestion

    def _snapshot_response(
        snapshot: FleetSnapshot,
        ingestion: Any,
        *,
        include_records: bool,
        page: int,
        page_size: int,
    ) -> FleetSnapshotResponse:
        return FleetSnapshotResponse(
            fleet_id=snapshot.fleet_id,
            snapshot_id=snapshot.snapshot_id,
            generated_at_utc=snapshot.generated_at_utc,
            schema_version=snapshot.schema_version,
            identity=snapshot.identity,
            battery_count=snapshot.battery_count,
            successfully_processed_count=snapshot.successfully_processed_count,
            failed_count=snapshot.failed_count,
            insufficient_data_count=snapshot.insufficient_data_count,
            summary=snapshot.summary,
            health_distribution=snapshot.health_distribution,
            risk_distribution=snapshot.risk_distribution,
            maintenance_summary=snapshot.maintenance_summary,
            replacement_summary=snapshot.replacement_summary,
            workload_forecast=snapshot.workload_forecast,
            fleet_statistics=snapshot.fleet_statistics,
            data_quality=snapshot.data_quality,
            drift_status=snapshot.drift_status,
            model_metadata=snapshot.model_metadata,
            batteries=(
                _paginate(snapshot.batteries, page=page, page_size=page_size)
                if include_records
                else None
            ),
            ingestion_records=list(ingestion.records) if ingestion is not None else [],
            batch_id=snapshot.batch_id,
            processing_duration_ms=snapshot.processing_duration_ms,
            warnings=snapshot.warnings,
            disclaimer=snapshot.disclaimer,
        )

    # -- POST /v1/fleet/snapshot -------------------------------------------
    @router.post("/fleet/snapshot", response_model=FleetSnapshotResponse)
    def fleet_snapshot(
        body: FleetRequest,
        request: Request,
        service: FleetInferenceService = Depends(get_fleet_service),
    ) -> FleetSnapshotResponse:
        """Score a fleet synchronously and return the full snapshot.

        Partial success is normal: batteries that failed ingestion or inference
        appear in ``ingestion_records`` and in the per-battery records with a
        ``failed`` status, and are excluded from every aggregate denominator.
        """
        snapshot, ingestion = _snapshot_from_request(body, service, request)
        return _snapshot_response(
            snapshot,
            ingestion,
            include_records=body.include_battery_records,
            page=body.page,
            page_size=_page_size(cfg, body.page_size),
        )

    # -- POST /v1/fleet/rank ------------------------------------------------
    @router.post("/fleet/rank", response_model=FleetRankResponse)
    def fleet_rank(
        body: FleetRankRequest,
        request: Request,
        service: FleetInferenceService = Depends(get_fleet_service),
    ) -> FleetRankResponse:
        """Order a fleet by one criterion. Ties break on battery id."""
        snapshot, _ = _snapshot_from_request(body, service, request)
        ordered = rank_batteries(
            snapshot.batteries,
            by=body.rank_by,
            limit=body.limit,
            include_unevaluated=body.include_unevaluated,
        )
        excluded = snapshot.battery_count - len([r for r in snapshot.batteries if r.is_evaluated])
        return FleetRankResponse(
            fleet_id=snapshot.fleet_id,
            snapshot_id=snapshot.snapshot_id,
            generated_at_utc=snapshot.generated_at_utc,
            rank_by=body.rank_by,
            ranking=_paginate(ordered, page=body.page, page_size=_page_size(cfg, body.page_size)),
            excluded_unevaluated_count=0 if body.include_unevaluated else excluded,
            warnings=snapshot.warnings,
        )

    # -- POST /v1/fleet/maintenance-plan -----------------------------------
    @router.post("/fleet/maintenance-plan", response_model=MaintenancePlanResponse)
    def maintenance_plan(
        body: FleetRequest,
        request: Request,
        service: FleetInferenceService = Depends(get_fleet_service),
    ) -> MaintenancePlanResponse:
        """Priorities, actions and the workload forecast they imply."""
        snapshot, _ = _snapshot_from_request(body, service, request)
        ordered = rank_batteries(snapshot.batteries, by="priority", include_unevaluated=True)
        return MaintenancePlanResponse(
            fleet_id=snapshot.fleet_id,
            snapshot_id=snapshot.snapshot_id,
            generated_at_utc=snapshot.generated_at_utc,
            maintenance_summary=snapshot.maintenance_summary,
            workload_forecast=snapshot.workload_forecast,
            batteries=_paginate(ordered, page=body.page, page_size=_page_size(cfg, body.page_size)),
            disclaimer=cfg.fleet.maintenance.disclaimer,
            warnings=snapshot.warnings,
        )

    # -- POST /v1/fleet/replacement-plan -----------------------------------
    @router.post("/fleet/replacement-plan", response_model=ReplacementPlanResponse)
    def replacement_plan(
        body: FleetRequest,
        request: Request,
        service: FleetInferenceService = Depends(get_fleet_service),
    ) -> ReplacementPlanResponse:
        """Advisory replacement candidates, bracketed by prediction uncertainty."""
        snapshot, _ = _snapshot_from_request(body, service, request)
        candidates = [
            r
            for r in snapshot.batteries
            if r.replacement is not None and r.replacement.replacement_candidate
        ]
        candidates.sort(key=lambda r: (r.priority.severity, -r.priority_score, r.battery_id))
        return ReplacementPlanResponse(
            fleet_id=snapshot.fleet_id,
            snapshot_id=snapshot.snapshot_id,
            generated_at_utc=snapshot.generated_at_utc,
            replacement_summary=snapshot.replacement_summary,
            candidates=_paginate(
                candidates, page=body.page, page_size=_page_size(cfg, body.page_size)
            ),
            caveats=snapshot.replacement_summary.caveats,
            warnings=snapshot.warnings,
        )

    # -- POST /v1/fleet/monitoring/run -------------------------------------
    @router.post("/fleet/monitoring/run", response_model=MonitoringRunResponse, tags=["monitoring"])
    def run_monitoring_endpoint(
        body: FleetRequest,
        request: Request,
        service: FleetInferenceService = Depends(get_fleet_service),
        repository: Any = Depends(get_repository),
    ) -> MonitoringRunResponse:
        """Data-quality and drift monitoring over a submitted fleet.

        The delayed-label performance section is intentionally *not* computed
        here: it depends on labels that arrive later and belongs to the batch
        job, not to a request-scoped call.
        """
        from battery_rul.monitoring.alerts import AlertPolicy
        from battery_rul.monitoring.domain import MonitoringSnapshot
        from battery_rul.monitoring.drift import detect_feature_drift
        from battery_rul.monitoring.prediction_drift import (
            detect_prediction_drift,
            summarise_predictions,
        )
        from battery_rul.monitoring.reference import load_reference

        snapshot, _ = _snapshot_from_request(body, service, request)
        warnings: list[str] = []
        feature_drift = prediction_drift = None
        reference = None
        try:
            reference = load_reference(cfg)
        except (FileNotFoundError, ValueError) as exc:
            warnings.append(f"Feature drift was not assessed: {exc}")

        if reference is not None:
            frames = []
            for item in body.batteries:
                try:
                    features = service.twin.prepare_features(item.battery_id, item.to_frame())
                except Exception as exc:  # noqa: BLE001 - one cell must not sink the run
                    logger.warning("Feature preparation failed for %s: %s", item.battery_id, exc)
                    continue
                if features is not None and not features.empty:
                    frames.append(features)
            if frames:
                import pandas as pd

                feature_drift = detect_feature_drift(
                    pd.concat(frames, ignore_index=True),
                    reference,
                    cfg,
                    current_window=f"online request {snapshot.snapshot_id}",
                )
            else:
                warnings.append("No engineered features could be produced from this request.")
            if reference.prediction_stats:
                prediction_drift = detect_prediction_drift(
                    summarise_predictions(snapshot.batteries, cfg),
                    reference.prediction_stats,
                    cfg,
                    model_version=snapshot.model_metadata.active_model_version,
                    reference_id=reference.reference_id,
                )
            else:
                warnings.append(
                    "Prediction drift was not assessed: the reference carries no "
                    "prediction distribution."
                )

        alerts = AlertPolicy(cfg=cfg).build(
            fleet_snapshot=snapshot,
            feature_drift=feature_drift,
            prediction_drift=prediction_drift,
            readiness=service.readiness(),
        )
        statuses = [snapshot.data_quality.status]
        if feature_drift is not None:
            statuses.append(feature_drift.status)
        if prediction_drift is not None:
            statuses.append(prediction_drift.status)
        from battery_rul.fleet.domain import MonitoringStatus

        monitoring = MonitoringSnapshot(
            snapshot_id=f"mon-{snapshot.snapshot_id}",
            fleet_id=snapshot.fleet_id,
            model_version=snapshot.model_metadata.active_model_version,
            data_version=snapshot.data_fingerprint,
            batch_id=snapshot.batch_id,
            input_count=snapshot.battery_count,
            success_count=snapshot.successfully_processed_count,
            failed_count=snapshot.failed_count,
            data_quality_summary=snapshot.data_quality.model_dump(mode="json"),
            feature_drift_summary=(
                {
                    "status": feature_drift.status.value,
                    "n_features_tested": feature_drift.n_features_tested,
                    "n_features_drifted": feature_drift.n_features_drifted,
                    "drifted_features": feature_drift.drifted_features[:20],
                }
                if feature_drift
                else {}
            ),
            prediction_drift_summary=(
                {
                    "status": prediction_drift.status.value,
                    "n_drifted": prediction_drift.n_drifted,
                }
                if prediction_drift
                else {}
            ),
            alerts=alerts,
            overall_status=MonitoringStatus.worst(statuses),
            warnings=warnings,
        )
        if repository is not None and not cfg.deployment.read_only:
            try:
                repository.save_monitoring_snapshot(monitoring)
                repository.save_alerts(alerts)
            except Exception as exc:  # noqa: BLE001 - surfaced, never silent
                warnings.append(f"Monitoring snapshot was not stored: {exc}")

        return MonitoringRunResponse(
            snapshot_id=monitoring.snapshot_id,
            fleet_id=monitoring.fleet_id,
            generated_at_utc=monitoring.generated_at_utc,
            overall_status=monitoring.overall_status.value,
            data_quality_status=snapshot.data_quality.status.value,
            feature_drift_status=(feature_drift.status.value if feature_drift else "UNKNOWN"),
            prediction_drift_status=(
                prediction_drift.status.value if prediction_drift else "UNKNOWN"
            ),
            performance_status="NOT_EVALUATED_ONLINE",
            n_alerts=len(alerts),
            model_version=monitoring.model_version,
            summary={
                "feature_drift": monitoring.feature_drift_summary,
                "prediction_drift": monitoring.prediction_drift_summary,
            },
            warnings=[
                *warnings,
                "Delayed-label performance is evaluated by the batch monitoring job, "
                "not online: it depends on outcomes that are not yet observable.",
            ],
        )

    # -- stored-snapshot reads ---------------------------------------------
    def _require_snapshot(repository: Any, fleet_id: str) -> FleetSnapshot:
        if repository is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No persistence backend is configured; stored snapshots are unavailable.",
            )
        snapshot = repository.latest_fleet_snapshot(fleet_id)
        if snapshot is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No stored snapshot for fleet {fleet_id!r}. Run a batch with "
                    "`python -m battery_rul.pipelines.run_fleet_batch`."
                ),
            )
        return snapshot

    @router.get("/fleet/{fleet_id}/latest", response_model=FleetSnapshotResponse)
    def latest_snapshot(
        fleet_id: str,
        page: int = Query(default=1, ge=1),
        page_size: int | None = Query(default=None, ge=1, le=1000),
        repository: Any = Depends(get_repository),
    ) -> FleetSnapshotResponse:
        """The most recent stored snapshot for a fleet, battery records paged."""
        snapshot = _require_snapshot(repository, fleet_id)
        return _snapshot_response(
            snapshot,
            None,
            include_records=True,
            page=page,
            page_size=_page_size(cfg, page_size),
        )

    @router.get("/fleet/{fleet_id}/summary", response_model=FleetSummaryResponse)
    def fleet_summary(
        fleet_id: str, repository: Any = Depends(get_repository)
    ) -> FleetSummaryResponse:
        """Aggregates only — the cheap endpoint for a dashboard header."""
        snapshot = _require_snapshot(repository, fleet_id)
        return FleetSummaryResponse(
            summary=snapshot.summary,
            health_distribution=snapshot.health_distribution,
            risk_distribution=snapshot.risk_distribution,
            maintenance_summary=snapshot.maintenance_summary,
            fleet_statistics=snapshot.fleet_statistics,
            data_quality=snapshot.data_quality,
            drift_status=snapshot.drift_status,
            model_metadata=snapshot.model_metadata,
            snapshot_id=snapshot.snapshot_id,
            generated_at_utc=snapshot.generated_at_utc,
            warnings=snapshot.warnings,
        )

    @router.get("/fleet/{fleet_id}/critical-batteries", response_model=CriticalBatteriesResponse)
    def critical_batteries(
        fleet_id: str,
        page: int = Query(default=1, ge=1),
        page_size: int | None = Query(default=None, ge=1, le=1000),
        repository: Any = Depends(get_repository),
    ) -> CriticalBatteriesResponse:
        """Cells at a priority listed in ``fleet.critical_priorities``."""
        snapshot = _require_snapshot(repository, fleet_id)
        critical = [
            r for r in snapshot.batteries if r.priority.value in cfg.fleet.critical_priorities
        ]
        critical.sort(key=lambda r: (MaintenancePriority(r.priority).severity, -r.priority_score))
        return CriticalBatteriesResponse(
            fleet_id=fleet_id,
            snapshot_id=snapshot.snapshot_id,
            generated_at_utc=snapshot.generated_at_utc,
            critical_priorities=list(cfg.fleet.critical_priorities),
            batteries=_paginate(critical, page=page, page_size=_page_size(cfg, page_size)),
            total_critical=len(critical),
        )

    @router.get("/fleet/{fleet_id}/alerts", response_model=AlertListResponse, tags=["monitoring"])
    def fleet_alerts(
        fleet_id: str,
        acknowledged: bool | None = Query(default=None),
        page: int = Query(default=1, ge=1),
        page_size: int | None = Query(default=None, ge=1, le=1000),
        repository: Any = Depends(get_repository),
    ) -> AlertListResponse:
        """Stored alerts for a fleet."""
        if repository is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No persistence backend is configured; alerts are unavailable.",
            )
        alerts = repository.list_alerts(fleet_id, acknowledged=acknowledged, limit=1000)
        size = _page_size(cfg, page_size)
        start = (page - 1) * size
        return AlertListResponse(
            fleet_id=fleet_id,
            alerts=[a.model_dump(mode="json") for a in alerts[start : start + size]],
            pagination=PageMeta.build(page=page, page_size=size, total=len(alerts)),
        )

    @router.get("/monitoring/latest", tags=["monitoring"])
    def latest_monitoring(
        fleet_id: str | None = Query(default=None),
        repository: Any = Depends(get_repository),
    ) -> dict[str, Any]:
        """The most recent monitoring snapshot, or an explicit 404."""
        if repository is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No persistence backend is configured; monitoring history is unavailable.",
            )
        snapshot = repository.latest_monitoring_snapshot(fleet_id)
        if snapshot is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    "No monitoring snapshot has been stored. Run "
                    "`python -m battery_rul.pipelines.run_monitoring`."
                ),
            )
        return snapshot.to_json_dict()

    # -- registry ------------------------------------------------------------
    @router.get("/models", response_model=ModelListResponse, tags=["registry"])
    def list_models() -> ModelListResponse:
        """Registered model versions and their stages."""
        from battery_rul.registry.store import FileModelRegistry, RegistryError

        registry = FileModelRegistry(cfg=cfg)
        try:
            entries = registry.list_models()
            production = registry.production_model()
        except RegistryError as exc:
            return ModelListResponse(registry_available=False, note=str(exc))
        return ModelListResponse(
            models=[_public_entry(e) for e in entries],
            production=_public_entry(production) if production else None,
        )

    @router.get("/models/production", tags=["registry"])
    def production_model() -> dict[str, Any]:
        """The live model, or a 404 that says how to promote one."""
        from battery_rul.registry.store import FileModelRegistry, RegistryError

        try:
            entry = FileModelRegistry(cfg=cfg).production_model()
        except RegistryError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        if entry is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    "No model is at stage PRODUCTION. Promote one with "
                    "`python -m battery_rul.pipelines.promote_model`."
                ),
            )
        return _public_entry(entry)

    # -- administrative (disabled by default) --------------------------------
    # Prefix is relative: this router is included in the ``/v1`` router above,
    # so the served paths are ``/v1/admin/...``.
    admin = APIRouter(prefix="/admin", tags=["admin"])

    @admin.post("/models/promote")
    def admin_promote(body: PromotionRequest) -> dict[str, Any]:
        """Promote a version. Requires ``deployment.admin_endpoints_enabled``."""
        _require_admin()
        from battery_rul.pipelines.milestone_3 import promote_model

        with _registry_errors():
            return promote_model(
                cfg,
                model_name=body.model_name,
                model_version=body.model_version,
                by=body.by,
                reason=body.reason,
                dry_run=body.dry_run,
            )

    @admin.post("/models/rollback")
    def admin_rollback(body: RollbackRequest) -> dict[str, Any]:
        """Roll back to the previously live version."""
        _require_admin()
        from battery_rul.pipelines.milestone_3 import rollback_model

        with _registry_errors():
            return rollback_model(cfg, model_name=body.model_name, by=body.by, reason=body.reason)

    def _require_admin() -> None:
        if not cfg.deployment.admin_endpoints_enabled:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Administrative endpoints are disabled. Set "
                    "deployment.admin_endpoints_enabled=true and put authentication in "
                    "front of this service before enabling them; this build ships none."
                ),
            )
        if cfg.deployment.read_only:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="deployment.read_only is set; this process must not change the registry.",
            )

    router.include_router(admin)
    return router


@contextmanager
def _registry_errors() -> Iterator[None]:
    """Turn registry failures into structured HTTP errors, not stack traces.

    A 500 with an empty body tells an operator nothing; a 404 saying the version
    is not registered, or a 409 saying the transition is illegal, tells them
    what to do next.
    """
    from battery_rul.registry.store import RegistryError

    try:
        yield
    except RegistryError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


def _public_entry(entry: Any) -> dict[str, Any]:
    """A registry entry as published over HTTP.

    ``bundle_path`` is deliberately omitted: a filesystem layout is not a
    consumer's business, and publishing it invites a client to try to read it.
    """
    return {
        "model_name": entry.model_name,
        "model_version": entry.model_version,
        "stage": entry.stage.value,
        "task": entry.task,
        "validation_status": entry.validation_status,
        "artifact_checksum": entry.artifact_checksum,
        "dataset_fingerprint": entry.dataset_fingerprint,
        "feature_schema_fingerprint": entry.feature_schema_fingerprint,
        "n_features": entry.n_features,
        "created_at_utc": entry.created_at_utc,
        "promoted_at_utc": entry.promoted_at_utc,
        "promoted_by": entry.promoted_by,
        "metrics": entry.metrics,
        "notes": entry.notes,
    }
