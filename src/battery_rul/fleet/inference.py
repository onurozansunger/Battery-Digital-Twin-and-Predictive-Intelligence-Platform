"""Batch digital-twin inference across a fleet.

This service **orchestrates**; it does not infer. Every prediction in a fleet
snapshot comes from
:class:`~battery_rul.digital_twin.service.BatteryDigitalTwinService`, constructed
once and reused for every cell. That is the whole design constraint of this
module, and it is not an aesthetic one:

* a second inference path drifts from the first, and the two disagree in
  production long before anyone notices;
* loading model bundles per battery turns a 128-cell fleet into 128 pickle
  loads, which is minutes of wall clock spent re-reading the same file;
* the warm-up policy, the calibration and the compatibility checks live in the
  battery-level service, and a fleet path that skipped them would produce
  numbers that look identical and mean something different.

Failure isolation
-----------------
One cell raising must not lose the other 127. Each battery is scored inside its
own try/except, and the failure becomes a ``FAILED`` record carrying the
exception type and message — never a silent drop and never a dummy prediction.
A fleet snapshot where 12 cells failed says so in three places: the per-record
status, the ``failed_count``, and the aggregate denominators.

Concurrency
-----------
Bounded, and 1 by default. The estimators are already BLAS-parallel, so extra
threads mostly contend; more importantly, the twin service holds fitted
artifacts whose thread-safety is inherited from scikit-learn/torch rather than
guaranteed here. Results are re-ordered to the input order either way, so a
snapshot is byte-identical regardless of the worker count.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from battery_rul import __version__
from battery_rul.config import ExperimentConfig
from battery_rul.digital_twin.domain import BatteryTwinSnapshot, DataQualityAssessment
from battery_rul.digital_twin.service import (
    BatteryDigitalTwinService,
    InsufficientDataError,
    InvalidHistoryError,
)
from battery_rul.fleet.aggregation import (
    fleet_statistics,
    health_distribution,
    maintenance_summary,
    risk_distribution,
)
from battery_rul.fleet.analytics import battery_trends, cycles_per_day
from battery_rul.fleet.domain import (
    FLEET_SNAPSHOT_SCHEMA_VERSION,
    FleetBatteryRecord,
    FleetDataQualitySummary,
    FleetDriftStatus,
    FleetIdentity,
    FleetIngestionResult,
    FleetModelMetadata,
    FleetSnapshot,
    FleetSummary,
    MaintenancePriority,
    MonitoringStatus,
    ProcessingStatus,
    battery_record_from_snapshot,
)
from battery_rul.fleet.ingestion import BatteryHistoryInput
from battery_rul.fleet.maintenance import MaintenancePriorityEngine
from battery_rul.fleet.replacement import (
    ReplacementPlanner,
    summarise_replacements,
    workload_forecast,
)
from battery_rul.observability.logging import bind_context, log_event
from battery_rul.observability.metrics import METRICS
from battery_rul.utils.io import environment_fingerprint
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["FleetBatchResult", "FleetInferenceService", "new_batch_id"]


def new_batch_id() -> str:
    """A sortable, unique batch identifier: ``20260813T021500Z-ab12cd34``."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True, slots=True)
class FleetBatchResult:
    """A completed batch: the published snapshot plus monitoring detail.

    The per-battery quality assessments are returned alongside rather than
    inside the snapshot, because they are large and only the monitoring layer
    reads them. See :meth:`FleetInferenceService.run_batch`.
    """

    snapshot: FleetSnapshot
    quality_assessments: dict[str, DataQualityAssessment] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _ScoredBattery:
    """One cell's fleet record plus the quality assessment behind it."""

    record: FleetBatteryRecord
    quality: DataQualityAssessment | None

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Unpackable as ``record, quality``."""
        return iter((self.record, self.quality))


@dataclass
class FleetInferenceService:
    """Turns validated histories into a :class:`FleetSnapshot`."""

    cfg: ExperimentConfig
    twin: BatteryDigitalTwinService
    engine: MaintenancePriorityEngine = field(init=False)
    planner: ReplacementPlanner = field(init=False)

    def __post_init__(self) -> None:
        self.engine = MaintenancePriorityEngine(cfg=self.cfg)
        self.planner = ReplacementPlanner(cfg=self.cfg)

    # -- construction ------------------------------------------------------
    @classmethod
    def create(
        cls,
        cfg: ExperimentConfig,
        *,
        twin: BatteryDigitalTwinService | None = None,
        strict: bool = False,
    ) -> FleetInferenceService:
        """Build the service, loading model bundles exactly once.

        ``twin`` is injectable so a caller that already holds a service (the API
        process, a test) does not load the artifacts a second time.
        """
        service = twin or BatteryDigitalTwinService.create(cfg, strict=strict)
        return cls(cfg=cfg, twin=service)

    def readiness(self) -> dict[str, Any]:
        """Delegates: fleet readiness *is* battery-level readiness."""
        return self.twin.readiness()

    # -- the snapshot ------------------------------------------------------
    def create_fleet_snapshot(
        self,
        fleet_id: str,
        battery_histories: Sequence[BatteryHistoryInput],
        *,
        ingestion: FleetIngestionResult | None = None,
        batch_id: str | None = None,
        identity: FleetIdentity | None = None,
        drift_status: FleetDriftStatus | None = None,
        data_quality: FleetDataQualitySummary | None = None,
        max_batteries: int | None = None,
    ) -> FleetSnapshot:
        """Score every supplied cell and assemble the fleet view."""
        return self.run_batch(
            fleet_id,
            battery_histories,
            ingestion=ingestion,
            batch_id=batch_id,
            identity=identity,
            drift_status=drift_status,
            data_quality=data_quality,
            max_batteries=max_batteries,
        ).snapshot

    def run_batch(
        self,
        fleet_id: str,
        battery_histories: Sequence[BatteryHistoryInput],
        *,
        ingestion: FleetIngestionResult | None = None,
        batch_id: str | None = None,
        identity: FleetIdentity | None = None,
        drift_status: FleetDriftStatus | None = None,
        data_quality: FleetDataQualitySummary | None = None,
        max_batteries: int | None = None,
    ) -> FleetBatchResult:
        """The full batch: the snapshot plus the detail monitoring needs.

        ``ingestion`` is optional but strongly recommended: it carries the cells
        that were *rejected* before inference, and without it they vanish from
        the snapshot entirely.
        """
        started = time.perf_counter()
        batch = batch_id or new_batch_id()
        limit = max_batteries or self.cfg.fleet.max_batteries_per_batch
        if len(battery_histories) > limit:
            raise ValueError(
                f"{len(battery_histories)} batteries submitted; the configured limit "
                f"is {limit}. Use the batch pipeline for larger fleets."
            )

        warnings: list[str] = []
        version = self._active_model_version()

        with bind_context(fleet_id=fleet_id, batch_id=batch, model_version=version):
            log_event(
                logger,
                "fleet_batch_started",
                battery_count=len(battery_histories),
                concurrency=self.cfg.fleet.max_concurrency,
            )
            scored_pairs = self._score_all(battery_histories)
            records = [record for record, _ in scored_pairs]
            assessments = {
                record.battery_id: assessment
                for record, assessment in scored_pairs
                if assessment is not None
            }

            # Cells rejected at ingestion are part of the fleet's story.
            if ingestion is not None:
                scored = {r.battery_id for r in records}
                for entry in ingestion.records:
                    if entry.status is ProcessingStatus.FAILED and entry.battery_id not in scored:
                        records.append(
                            FleetBatteryRecord(
                                battery_id=entry.battery_id,
                                status=ProcessingStatus.FAILED,
                                n_cycles=entry.n_rows,
                                errors=list(entry.errors),
                                warnings=list(entry.warnings),
                                priority=MaintenancePriority.INSUFFICIENT_DATA,
                                recommended_action="INSUFFICIENT_DATA",
                            )
                        )
                warnings.extend(ingestion.warnings)

            records.sort(key=lambda r: r.battery_id)
            snapshot = self._assemble(
                fleet_id=fleet_id,
                records=records,
                ingestion=ingestion,
                batch=batch,
                identity=identity,
                drift_status=drift_status,
                data_quality=data_quality,
                assessments=assessments,
                warnings=warnings,
                duration_ms=1000.0 * (time.perf_counter() - started),
                model_version=version,
            )

            METRICS.observe(
                "fleet_batch_duration_seconds",
                (snapshot.processing_duration_ms or 0.0) / 1000.0,
                labels={"fleet_id": fleet_id},
                help_text="Wall-clock duration of one fleet batch.",
            )
            METRICS.increment(
                "battery_inference_total",
                snapshot.successfully_processed_count,
                help_text="Battery-level inferences that produced a prediction.",
            )
            METRICS.increment(
                "battery_inference_failures_total",
                snapshot.failed_count,
                help_text="Battery-level inferences that failed.",
            )
            METRICS.increment(
                "battery_insufficient_data_total",
                snapshot.insufficient_data_count,
                help_text="Batteries whose input could not support a prediction.",
            )
            METRICS.set_gauge(
                "fleet_critical_battery_count",
                snapshot.maintenance_summary.critical_count,
                labels={"fleet_id": fleet_id},
                help_text="Cells at a priority listed in fleet.critical_priorities.",
            )
            log_event(
                logger,
                "fleet_batch_completed",
                duration_ms=snapshot.processing_duration_ms,
                battery_count=snapshot.battery_count,
                success_count=snapshot.successfully_processed_count,
                failed_count=snapshot.failed_count,
                insufficient_count=snapshot.insufficient_data_count,
                critical_count=snapshot.maintenance_summary.critical_count,
            )
        return FleetBatchResult(snapshot=snapshot, quality_assessments=assessments)

    # -- per-battery -------------------------------------------------------
    def _score_all(self, histories: Sequence[BatteryHistoryInput]) -> list[_ScoredBattery]:
        """Score every cell, preserving input order and isolating failures."""
        workers = max(1, min(self.cfg.fleet.max_concurrency, len(histories) or 1))
        if workers == 1 or len(histories) <= 1:
            return [self._score_one(item) for item in histories]

        # Results are indexed and re-sorted, so the worker count never changes
        # the snapshot's content or ordering.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fleet") as pool:
            futures = {
                pool.submit(self._score_one, item): index for index, item in enumerate(histories)
            }
            indexed: list[tuple[int, _ScoredBattery]] = []
            for future, index in futures.items():
                indexed.append((index, future.result()))
        return [scored for _, scored in sorted(indexed, key=lambda pair: pair[0])]

    def _score_one(self, item: BatteryHistoryInput) -> _ScoredBattery:
        """One cell: twin snapshot -> trends -> priority -> replacement.

        Returns the record together with the battery-level data-quality
        assessment. The assessment is not folded into the record because it is
        detail the fleet *monitoring* layer needs and a fleet API response does
        not: 128 cells' worth of per-check payloads is the difference between a
        readable response and a megabyte of it.
        """
        with bind_context(battery_id=item.battery_id):
            started = time.perf_counter()
            try:
                snapshot = self.twin.create_snapshot(item.battery_id, item.history, explain=False)
            except (InvalidHistoryError, InsufficientDataError) as exc:
                return _ScoredBattery(self._failed_record(item, exc, kind="invalid_input"), None)
            except Exception as exc:  # noqa: BLE001 - one cell must not sink the fleet
                logger.exception("Inference failed for %s", item.battery_id)
                return _ScoredBattery(self._failed_record(item, exc, kind="inference_error"), None)

            elapsed_ms = 1000.0 * (time.perf_counter() - started)
            METRICS.observe(
                "battery_inference_duration_seconds",
                elapsed_ms / 1000.0,
                help_text="Per-battery digital-twin inference duration.",
            )
            record = self._build_record(item, snapshot)
            log_event(
                logger,
                "battery_scored",
                duration_ms=elapsed_ms,
                status=record.status.value,
                priority=record.priority.value,
            )
            return _ScoredBattery(record, snapshot.data_quality)

    def _build_record(
        self, item: BatteryHistoryInput, snapshot: BatteryTwinSnapshot
    ) -> FleetBatteryRecord:
        trends = battery_trends(item.history)
        status = _status_for(snapshot)
        record = battery_record_from_snapshot(snapshot, status=status, trends=trends)
        record.warnings = [*record.warnings, *item.warnings]

        priority = self.engine.evaluate(
            record,
            cycles_per_day=cycles_per_day(
                item.history,
                min_cycles=self.cfg.fleet.maintenance.min_cycles_for_rate_estimate,
            ),
        )
        record.priority = priority.priority
        record.priority_score = priority.priority_score
        record.priority_record = priority
        record.recommended_action = priority.recommended_action
        record.replacement = self.planner.evaluate(record)
        return record

    def _failed_record(
        self, item: BatteryHistoryInput, exc: Exception, *, kind: str
    ) -> FleetBatteryRecord:
        log_event(
            logger,
            "battery_scoring_failed",
            status="error",
            error_code=kind,
            error_type=type(exc).__name__,
        )
        return FleetBatteryRecord(
            battery_id=item.battery_id,
            status=ProcessingStatus.FAILED,
            n_cycles=item.n_cycles,
            errors=[f"{type(exc).__name__}: {exc}"],
            warnings=list(item.warnings),
            priority=MaintenancePriority.INSUFFICIENT_DATA,
            recommended_action="INSUFFICIENT_DATA",
        )

    # -- assembly ----------------------------------------------------------
    def _assemble(
        self,
        *,
        fleet_id: str,
        records: list[FleetBatteryRecord],
        ingestion: FleetIngestionResult | None,
        batch: str,
        identity: FleetIdentity | None,
        drift_status: FleetDriftStatus | None,
        data_quality: FleetDataQualitySummary | None,
        assessments: dict[str, DataQualityAssessment],
        warnings: list[str],
        duration_ms: float,
        model_version: str | None,
    ) -> FleetSnapshot:
        from battery_rul.monitoring.data_quality import summarise_fleet_data_quality

        evaluated = [r for r in records if r.is_evaluated]
        failed = [r for r in records if r.status is ProcessingStatus.FAILED]
        insufficient = [r for r in records if r.status is ProcessingStatus.INSUFFICIENT_DATA]

        health = health_distribution(records)
        risk = risk_distribution(records, self.cfg)
        maintenance = maintenance_summary(records, self.cfg)
        candidates = [r.replacement for r in records if r.replacement is not None]
        replacement = summarise_replacements(records, candidates, self.cfg)
        workload = workload_forecast(records, self.cfg)
        statistics = fleet_statistics(records, self.cfg)
        quality = data_quality or summarise_fleet_data_quality(records, self.cfg, assessments)

        if not evaluated:
            warnings.append(
                "No battery in this fleet produced a prediction. Every aggregate below "
                "has a zero denominator; check the readiness endpoint and the "
                "per-battery errors."
            )
        if failed:
            warnings.append(
                f"{len(failed)} battery/batteries failed and are excluded from every "
                "predicted-quantity aggregate: "
                + ", ".join(sorted(r.battery_id for r in failed)[:20])
            )
        if risk.experimental_model:
            warnings.append(
                "The failure-risk model is marked experimental (it did not beat the "
                "cycle-index baseline out of fold). Risk probabilities are reported "
                "but were withheld from the maintenance rules."
            )

        resolved_identity = identity or FleetIdentity(
            fleet_id=fleet_id,
            source=(ingestion.source if ingestion else "unknown"),
            is_demo_data=bool(ingestion.is_demo_data) if ingestion else False,
        )
        if resolved_identity.is_demo_data:
            warnings.append(
                "DEMO DATA: this fleet contains synthetic histories from the "
                "physics-informed generator. It is not measured data and must not be "
                "read as a description of any real fleet."
            )

        summary = FleetSummary(
            fleet_id=fleet_id,
            generated_at_utc=datetime.now(UTC).isoformat(),
            battery_count=len(records),
            successfully_processed_count=len(evaluated),
            failed_count=len(failed),
            insufficient_data_count=len(insufficient),
            healthy_count=health.get("healthy"),
            slightly_degraded_count=health.get("slightly_degraded"),
            warning_count=health.get("warning"),
            critical_count=health.get("critical"),
            inspection_recommended_count=maintenance.inspection_recommended_count,
            replacement_planning_count=replacement.candidate_count,
            high_priority_battery_ids=maintenance.high_risk_battery_ids[:20],
            median_soh=statistics.soh_median,
            median_rul=statistics.rul_median,
            drift_status=(drift_status.status if drift_status else MonitoringStatus.UNKNOWN),
            data_quality_status=quality.status,
            active_model_version=model_version,
            is_demo_data=resolved_identity.is_demo_data,
        )

        return FleetSnapshot(
            fleet_id=fleet_id,
            snapshot_id=batch,
            schema_version=FLEET_SNAPSHOT_SCHEMA_VERSION,
            identity=resolved_identity,
            battery_count=len(records),
            successfully_processed_count=len(evaluated),
            failed_count=len(failed),
            insufficient_data_count=len(insufficient),
            batteries=records,
            summary=summary,
            health_distribution=health,
            risk_distribution=risk,
            maintenance_summary=maintenance,
            replacement_summary=replacement,
            workload_forecast=workload,
            fleet_statistics=statistics,
            data_quality=quality,
            drift_status=drift_status or FleetDriftStatus(),
            model_metadata=self._model_metadata(),
            data_fingerprint=(ingestion.data_fingerprint if ingestion else ""),
            batch_id=batch,
            processing_duration_ms=round(duration_ms, 3),
            warnings=warnings,
        )

    # -- provenance --------------------------------------------------------
    def _active_model_version(self) -> str | None:
        bundle = self.twin.bundles.rul or self.twin.bundles.risk or self.twin.bundles.soh
        return bundle.metadata.model_version if bundle else None

    def _model_metadata(self) -> FleetModelMetadata:
        bundles = self.twin.bundles
        primary = bundles.rul or bundles.risk or bundles.soh
        versions = {
            name: getattr(bundles, name).metadata.model_version
            for name in ("rul", "soh", "risk")
            if getattr(bundles, name) is not None
        }
        registry_name = registry_stage = None
        try:
            from battery_rul.registry.store import FileModelRegistry

            entry = FileModelRegistry(cfg=self.cfg).production_model()
            if entry is not None:
                registry_name = entry.model_name
                registry_stage = entry.stage.value
        except Exception as exc:  # noqa: BLE001 - registry absence is not an error here
            logger.debug("No registry metadata available for this snapshot: %s", exc)

        return FleetModelMetadata(
            active_model_version=primary.metadata.model_version if primary else None,
            active_model_name=primary.metadata.model_name if primary else None,
            registry_model_name=registry_name,
            registry_stage=registry_stage,
            bundle_versions=versions,
            feature_pipeline_fingerprint=(
                primary.metadata.preprocessing_fingerprint if primary else None
            ),
            data_version=primary.metadata.dataset_fingerprint if primary else None,
            battery_snapshot_schema_version=(primary.metadata.schema_version if primary else None),
            git_revision=environment_fingerprint().get("git_revision"),
            package_version=__version__,
            risk_horizon_cycles=self.cfg.risk.horizon_cycles,
            end_of_life_definition={
                "threshold_fraction_of_reference": self.cfg.data.eol_threshold,
                "reference": self.cfg.data.eol_reference,
                "persistence_cycles": self.cfg.target.eol_persistence,
            },
        )


def _status_for(snapshot: BatteryTwinSnapshot) -> ProcessingStatus:
    """Map a battery snapshot onto a fleet processing status.

    A snapshot that produced no RUL is ``INSUFFICIENT_DATA``, not ``SUCCESS``.
    The distinction drives every denominator downstream: a cell with a measured
    SOH but no prediction belongs in the health distribution and not in the RUL
    median, and only an explicit status keeps those two facts apart.
    """
    if snapshot.data_quality.quality_class == "INSUFFICIENT":
        return ProcessingStatus.INSUFFICIENT_DATA
    if snapshot.prediction.rul_cycles is None:
        return ProcessingStatus.INSUFFICIENT_DATA
    return ProcessingStatus.SUCCESS


def histories_from_frame(frame: pd.DataFrame) -> list[BatteryHistoryInput]:
    """Split a validated long-form frame into per-battery inputs.

    A convenience for callers that already trust their frame; ingestion is the
    supported route and performs the validation this skips.
    """
    return [
        BatteryHistoryInput(battery_id=str(battery_id), history=group.reset_index(drop=True))
        for battery_id, group in frame.groupby("battery_id", sort=True)
    ]
