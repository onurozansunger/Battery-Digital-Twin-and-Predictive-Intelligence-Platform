"""The FastAPI application.

Design notes
------------
*The service is a dependency, not a global.* ``create_app`` builds one
:class:`BatteryDigitalTwinService` and injects it; tests construct an app with a
service built over fixture artifacts and never patch a module-level singleton.

*No training at startup.* Startup loads artifacts and validates them. If they are
missing or incompatible, ``/health`` still answers (the process is alive) while
``/ready`` reports not-ready with the reason — which is what a load balancer
needs in order to keep the instance out of rotation instead of black-holing
requests.

*No filesystem paths from request input.* The artifact directory comes from
configuration only. A request cannot name a model, a path, or a pickle.

*Structured errors.* Every failure returns an :class:`ErrorResponse` with a
request id, so a caller can quote it and it can be found in the logs.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse

from battery_rul import __version__
from battery_rul.api.schemas import (
    ErrorResponse,
    ExplainResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    ReadyResponse,
    RiskResponse,
    RulResponse,
    SnapshotResponse,
    SohResponse,
    VersionResponse,
)
from battery_rul.config import ExperimentConfig, load_config
from battery_rul.digital_twin.domain import SNAPSHOT_SCHEMA_VERSION, BatteryTwinSnapshot
from battery_rul.digital_twin.service import (
    BatteryDigitalTwinService,
    InvalidHistoryError,
    ModelsUnavailableError,
)
from battery_rul.observability.logging import bind_context
from battery_rul.observability.metrics import METRICS, render_prometheus
from battery_rul.utils.io import environment_fingerprint
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["create_app"]

API_VERSION = "v1"


def create_app(
    cfg: ExperimentConfig | None = None,
    *,
    service: BatteryDigitalTwinService | None = None,
    config_path: str | None = "configs/default.yaml",
    repository: Any | None = None,
    enable_fleet: bool = True,
) -> FastAPI:
    """Build the application.

    Parameters
    ----------
    cfg:
        Configuration. Loaded from ``config_path`` when omitted.
    service:
        Pre-built service, for tests. When omitted one is constructed from
        ``cfg``; construction failures are captured so the app still starts and
        can report *why* it is not ready.
    repository:
        Persistence backend for the Milestone 3 fleet reads. Injected in tests;
        built from configuration otherwise. A backend that cannot be opened is
        recorded and the stored-snapshot endpoints answer 503 — the prediction
        endpoints keep working, because they do not need storage.
    enable_fleet:
        Mount the Milestone 3 fleet router. On by default; a deployment that
        only serves battery-level predictions can turn it off.
    """
    from pathlib import Path

    if cfg is None:
        path = Path(config_path) if config_path else None
        cfg = load_config(path if (path and path.is_file()) else None)

    startup_error: str | None = None
    if service is None:
        try:
            service = BatteryDigitalTwinService.create(cfg)
        except Exception as exc:  # noqa: BLE001 - a broken artifact must not stop startup
            startup_error = f"{type(exc).__name__}: {exc}"
            logger.error("Service construction failed at startup: %s", startup_error)
            service = BatteryDigitalTwinService(cfg=cfg)

    app = FastAPI(
        title=cfg.service.api_title,
        version=__version__,
        root_path=cfg.service.api_root_path,
        description=(
            "Digital-twin inference for lithium-ion cells: remaining useful life, "
            "state of health, end-of-life risk within a configurable horizon, "
            "prediction intervals, feature attributions and rule-based engineering "
            "recommendations.\n\n"
            "**This is a research prototype.** Outputs are decision support, not "
            "safety-critical control, and have not been validated for production "
            "deployment. The 'failure risk' label is derived from a capacity "
            "threshold; the source dataset contains no observed safety failures."
        ),
    )
    app.state.cfg = cfg
    app.state.service = service
    app.state.startup_error = startup_error

    # -- Milestone 3 dependencies -------------------------------------------
    # Built once at startup, exactly like the twin service: a fleet service
    # constructed per request would reload every model bundle per request.
    fleet_service = None
    if enable_fleet:
        try:
            from battery_rul.fleet.inference import FleetInferenceService

            fleet_service = FleetInferenceService.create(cfg, twin=service)
        except Exception as exc:  # noqa: BLE001 - fleet absence must not stop startup
            logger.error("Fleet service construction failed: %s", exc)
            app.state.startup_error = f"{startup_error or ''} fleet: {exc}".strip()

    if repository is None and enable_fleet:
        try:
            from battery_rul.persistence import build_repository

            repository = build_repository(cfg)
        except Exception as exc:  # noqa: BLE001 - storage absence must not stop startup
            logger.error("Persistence backend unavailable: %s", exc)
            repository = None

    app.state.fleet_service = fleet_service
    app.state.repository = repository

    def get_service() -> BatteryDigitalTwinService:
        return app.state.service

    def get_fleet_service() -> Any:
        if app.state.fleet_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "The fleet service is not available. Check /ready and the startup "
                    "errors reported there."
                ),
            )
        return app.state.fleet_service

    def get_repository() -> Any:
        return app.state.repository

    # -- CORS ---------------------------------------------------------------
    # Explicit allow-list only. An empty list means no cross-origin access at
    # all, which is the correct default for a service with no authentication.
    if cfg.deployment.cors_allow_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cfg.deployment.cors_allow_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    # -- middleware ---------------------------------------------------------
    @app.middleware("http")
    async def _request_context(request: Request, call_next: Any) -> Any:
        header = cfg.service.request_id_header
        request_id = request.headers.get(header) or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        with bind_context(request_id=request_id, service="battery-rul-api"):
            response = await call_next(request)
        elapsed_ms = 1000 * (time.perf_counter() - started)
        response.headers[header] = request_id
        # Path and status only — request bodies carry cell measurements and are
        # never logged.
        logger.info(
            "%s %s -> %d (%.1f ms) request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        if cfg.deployment.metrics_enabled:
            # The route template, not the URL: labelling by path would create a
            # new time series per fleet id and blow up the metrics cardinality.
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            METRICS.increment(
                "api_requests_total",
                labels={
                    "method": request.method,
                    "path": path,
                    "status": str(response.status_code),
                },
                help_text="HTTP requests handled, by route and status.",
            )
            METRICS.observe(
                "api_request_duration_seconds",
                elapsed_ms / 1000.0,
                labels={"method": request.method, "path": path},
                help_text="HTTP request duration.",
            )
        return response

    # -- error handling ------------------------------------------------------
    def _error(
        request: Request, code: int, error: str, detail: str, hint: str | None = None
    ) -> JSONResponse:
        return JSONResponse(
            status_code=code,
            content=ErrorResponse(
                error=error,
                detail=detail,
                request_id=getattr(request.state, "request_id", None),
                hint=hint,
            ).model_dump(),
        )

    @app.exception_handler(InvalidHistoryError)
    async def _invalid_history(request: Request, exc: InvalidHistoryError) -> JSONResponse:
        return _error(
            request,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "invalid_history",
            str(exc),
            hint="See the canonical cycle schema in battery_rul.data.schema.",
        )

    @app.exception_handler(ModelsUnavailableError)
    async def _models_unavailable(request: Request, exc: ModelsUnavailableError) -> JSONResponse:
        return _error(
            request,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "models_unavailable",
            str(exc),
            hint="Run `python -m battery_rul.pipelines.run_milestone_2`.",
        )

    # -- operational endpoints ------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        """Liveness. Answers even when no model is loaded."""
        return HealthResponse(status="ok", service="battery-digital-twin", version=__version__)

    @app.get("/ready", response_model=ReadyResponse, tags=["operations"])
    def ready(
        svc: BatteryDigitalTwinService = Depends(get_service),
    ) -> JSONResponse:
        """Readiness. 503 when no artifact can answer a prediction request."""
        payload = svc.readiness()
        if app.state.startup_error:
            payload = {
                **payload,
                "errors": {**payload["errors"], "startup": app.state.startup_error},
            }
        body = ReadyResponse(**payload).model_dump()
        code = status.HTTP_200_OK if payload["ready"] else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(status_code=code, content=body)

    @app.get("/version", response_model=VersionResponse, tags=["operations"])
    def version() -> VersionResponse:
        """Versions of the API, the package and the snapshot wire format."""
        return VersionResponse(
            api_version=API_VERSION,
            package_version=__version__,
            snapshot_schema_version=SNAPSHOT_SCHEMA_VERSION,
            git_revision=environment_fingerprint().get("git_revision"),
        )

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["operations"])
    def model_info(svc: BatteryDigitalTwinService = Depends(get_service)) -> ModelInfoResponse:
        """Loaded artifacts, their fingerprints, and the definitions in force."""
        return ModelInfoResponse(metadata=svc.get_model_metadata())

    if cfg.deployment.metrics_enabled and cfg.deployment.metrics_endpoint_enabled:

        @app.get("/metrics", response_class=PlainTextResponse, tags=["operations"])
        def metrics() -> PlainTextResponse:
            """Prometheus text exposition of the in-process metrics.

            Rendered by this project's own tiny registry rather than a client
            library: the exposition format is the only part of that library
            this service needs.
            """
            return PlainTextResponse(
                render_prometheus(), media_type="text/plain; version=0.0.4; charset=utf-8"
            )

    # -- prediction endpoints ---------------------------------------------------
    def _snapshot(
        request_body: PredictionRequest, svc: BatteryDigitalTwinService, *, explain: bool
    ) -> BatteryTwinSnapshot:
        return svc.create_snapshot(
            request_body.battery_id, request_body.to_frame(), explain=explain
        )

    @app.post(
        f"/{API_VERSION}/predict/rul",
        response_model=RulResponse,
        tags=["prediction"],
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    def predict_rul(
        body: PredictionRequest, svc: BatteryDigitalTwinService = Depends(get_service)
    ) -> RulResponse:
        """Remaining useful life in cycles, with a conformal prediction interval."""
        snapshot = _snapshot(body, svc, explain=False)
        return RulResponse(
            battery_id=snapshot.battery_id,
            generated_at_utc=snapshot.generated_at_utc,
            prediction=snapshot.prediction,
            data_quality_class=snapshot.data_quality.quality_class,
            warnings=snapshot.warnings,
        )

    @app.post(
        f"/{API_VERSION}/predict/soh",
        response_model=SohResponse,
        tags=["prediction"],
        responses={422: {"model": ErrorResponse}},
    )
    def predict_soh(
        body: PredictionRequest, svc: BatteryDigitalTwinService = Depends(get_service)
    ) -> SohResponse:
        """State of health as a fraction in [0, 1], plus its engineering band."""
        snapshot = _snapshot(body, svc, explain=False)
        return SohResponse(
            battery_id=snapshot.battery_id,
            generated_at_utc=snapshot.generated_at_utc,
            health=snapshot.health,
            data_quality_class=snapshot.data_quality.quality_class,
            warnings=snapshot.warnings,
        )

    @app.post(
        f"/{API_VERSION}/predict/risk",
        response_model=RiskResponse,
        tags=["prediction"],
        responses={422: {"model": ErrorResponse}},
    )
    def predict_risk(
        body: PredictionRequest, svc: BatteryDigitalTwinService = Depends(get_service)
    ) -> RiskResponse:
        """Calibrated probability of reaching end of life within the horizon."""
        snapshot = _snapshot(body, svc, explain=False)
        return RiskResponse(
            battery_id=snapshot.battery_id,
            generated_at_utc=snapshot.generated_at_utc,
            failure_risk=snapshot.failure_risk,
            data_quality_class=snapshot.data_quality.quality_class,
            warnings=snapshot.warnings,
        )

    @app.post(
        f"/{API_VERSION}/predict/full",
        response_model=SnapshotResponse,
        tags=["prediction"],
        responses={422: {"model": ErrorResponse}},
    )
    def predict_full(
        body: PredictionRequest, svc: BatteryDigitalTwinService = Depends(get_service)
    ) -> SnapshotResponse:
        """Every output at once, as a full digital-twin snapshot."""
        return SnapshotResponse(snapshot=_snapshot(body, svc, explain=body.include_explanation))

    @app.post(
        f"/{API_VERSION}/digital-twin/snapshot",
        response_model=SnapshotResponse,
        tags=["digital-twin"],
        responses={422: {"model": ErrorResponse}},
    )
    def snapshot(
        body: PredictionRequest, svc: BatteryDigitalTwinService = Depends(get_service)
    ) -> SnapshotResponse:
        """Alias of ``/predict/full`` under the digital-twin resource path."""
        return SnapshotResponse(snapshot=_snapshot(body, svc, explain=body.include_explanation))

    @app.post(
        f"/{API_VERSION}/explain",
        response_model=ExplainResponse,
        tags=["explainability"],
        responses={422: {"model": ErrorResponse}},
    )
    def explain(
        body: PredictionRequest, svc: BatteryDigitalTwinService = Depends(get_service)
    ) -> ExplainResponse:
        """Local feature attributions for the latest cycle."""
        snapshot_obj = _snapshot(body, svc, explain=True)
        from battery_rul.digital_twin.domain import BatteryExplanation

        return ExplainResponse(
            battery_id=snapshot_obj.battery_id,
            generated_at_utc=snapshot_obj.generated_at_utc,
            explanation=snapshot_obj.explanation
            or BatteryExplanation(method="unavailable", drivers=[]),
        )

    # -- Milestone 3 fleet endpoints -----------------------------------------
    if enable_fleet:
        from battery_rul.api.fleet_routes import build_fleet_router

        app.include_router(
            build_fleet_router(
                cfg, get_fleet_service=get_fleet_service, get_repository=get_repository
            )
        )

    return app


def main() -> int:  # pragma: no cover - process entry point
    """``python -m battery_rul.api.app`` — run the development server."""
    import uvicorn

    cfg = load_config("configs/default.yaml")
    uvicorn.run(
        create_app(cfg), host=cfg.service.api_host, port=cfg.service.api_port, log_level="info"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
