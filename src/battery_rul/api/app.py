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

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse

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

    def get_service() -> BatteryDigitalTwinService:
        return app.state.service

    # -- middleware ---------------------------------------------------------
    @app.middleware("http")
    async def _request_context(request: Request, call_next: Any) -> Any:
        header = cfg.service.request_id_header
        request_id = request.headers.get(header) or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
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
