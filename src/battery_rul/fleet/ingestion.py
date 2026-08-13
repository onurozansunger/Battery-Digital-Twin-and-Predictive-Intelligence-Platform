"""Fleet ingestion: many cells in, validated per-cell histories out.

The contract that matters here is **partial success**. A fleet file with one
malformed cell must not fail the other 127, and the malformed cell must not
disappear either — it comes back as a ``FAILED`` record carrying the reason. Both
halves of that are load-bearing: silent tolerance and total failure are the two
ways an ingestion layer loses data.

Accepted shapes
---------------
* one tabular file (``.parquet`` / ``.csv``) containing a ``battery_id`` column
* a directory of per-battery files, the stem naming the cell
* an already-loaded frame (the processed cycle table from Milestone 1)
* in-memory records, which is what the API's request body becomes

Boundaries are enforced per cell: rows are never allowed to cross a
``battery_id``, because every trailing window downstream — rolling means, lags,
slopes — would then read another cell's history as this one's past.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.fleet.domain import (
    BatteryIngestionRecord,
    FleetBatteryReference,
    FleetIngestionResult,
    ProcessingStatus,
)
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "SUPPORTED_SUFFIXES",
    "BatteryHistoryInput",
    "FleetIngestionError",
    "FleetIngestor",
    "frame_fingerprint",
    "resolve_within",
]

#: File types this layer will open. Anything else is refused by name rather than
#: sniffed: a loader that guesses at content is a deserialisation surface.
SUPPORTED_SUFFIXES = (".parquet", ".csv", ".json")

#: The two columns without which a row is not a cycle record.
_MINIMUM_COLUMNS = ("cycle_index", "capacity_ah")


class FleetIngestionError(ValueError):
    """The fleet source itself is unusable (missing file, unreadable, empty)."""


@dataclass(slots=True)
class BatteryHistoryInput:
    """One validated cell history, ready for the twin service.

    Not a Pydantic model on purpose: it holds a DataFrame, and a wire-format
    object holding measurement frames is how large payloads accidentally end up
    in responses and logs.
    """

    battery_id: str
    history: pd.DataFrame
    source: str | None = None
    is_synthetic: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def n_cycles(self) -> int:
        return int(len(self.history))

    def reference(self) -> FleetBatteryReference:
        cycles = pd.to_numeric(self.history["cycle_index"], errors="coerce").dropna()
        return FleetBatteryReference(
            battery_id=self.battery_id,
            n_cycles=self.n_cycles,
            first_cycle=int(cycles.min()) if len(cycles) else None,
            latest_cycle=int(cycles.max()) if len(cycles) else None,
            source=self.source,
            is_synthetic=self.is_synthetic,
        )


def resolve_within(base: Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` and refuse anything outside ``base``.

    Used wherever a path could come from outside the process. The API never
    accepts one, but the batch CLI does, and a traversal check that only exists
    at the HTTP boundary is a check that stops existing the moment someone adds
    a second entry point.
    """
    base_resolved = Path(base).expanduser().resolve()
    target = Path(candidate).expanduser()
    if not target.is_absolute():
        target = base_resolved / target
    target = target.resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise FleetIngestionError(
            f"Refusing to read {target}: it is outside the permitted directory {base_resolved}."
        )
    return target


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """A stable short hash of a frame's content.

    Deterministic across processes: column order is sorted, values are rendered
    through pandas' own hashing, and the result keys a batch so the same input
    always produces the same batch identity.
    """
    if frame.empty:
        return hashlib.sha256(b"empty").hexdigest()[:16]
    ordered = frame[sorted(frame.columns)]
    hashed = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    digest = hashlib.sha256(hashed.tobytes())
    digest.update(",".join(sorted(frame.columns)).encode("utf-8"))
    return digest.hexdigest()[:16]


@dataclass
class FleetIngestor:
    """Validates fleet input and reports what it accepted and what it refused."""

    cfg: ExperimentConfig

    # -- entry points ------------------------------------------------------
    def from_frame(
        self,
        fleet_id: str,
        frame: pd.DataFrame,
        *,
        source: str = "frame",
        is_synthetic: bool = False,
        max_batteries: int | None = None,
    ) -> tuple[FleetIngestionResult, list[BatteryHistoryInput]]:
        """Split one long-form table into validated per-battery histories."""
        if frame is None or len(frame) == 0:
            raise FleetIngestionError("The supplied fleet frame is empty.")
        if "battery_id" not in frame.columns:
            raise FleetIngestionError(
                "A multi-battery table must have a 'battery_id' column. For a "
                "directory of single-cell files, use from_directory()."
            )

        warnings: list[str] = []
        limit = max_batteries or self.cfg.fleet.max_batteries_per_batch
        identifiers = [str(v) for v in pd.unique(frame["battery_id"].astype(str))]
        identifiers.sort()
        if len(identifiers) > limit:
            warnings.append(
                f"{len(identifiers)} batteries supplied; the configured limit is "
                f"{limit}. The first {limit} by identifier are processed and the "
                "remainder are reported as failed rather than dropped."
            )

        records: list[BatteryIngestionRecord] = []
        accepted: list[FleetBatteryReference] = []
        histories: list[BatteryHistoryInput] = []

        for index, battery_id in enumerate(identifiers):
            if index >= limit:
                records.append(
                    BatteryIngestionRecord(
                        battery_id=battery_id,
                        status=ProcessingStatus.FAILED,
                        errors=[
                            f"Not processed: the batch limit of {limit} batteries was reached."
                        ],
                        source=source,
                    )
                )
                continue
            subset = frame.loc[frame["battery_id"].astype(str) == battery_id]
            record, history = self._validate(battery_id, subset, source=source)
            records.append(record)
            if history is not None:
                history.is_synthetic = is_synthetic
                histories.append(history)
                accepted.append(history.reference())

        result = FleetIngestionResult(
            fleet_id=fleet_id,
            source=source,
            accepted=accepted,
            records=records,
            warnings=warnings,
            data_fingerprint=frame_fingerprint(frame),
            source_metadata={
                "n_rows": int(len(frame)),
                "n_batteries_supplied": len(identifiers),
                "columns": sorted(str(c) for c in frame.columns),
            },
            is_demo_data=is_synthetic,
        )
        logger.info(
            "Fleet %s ingestion: %d accepted, %d failed, %d rows from %s",
            fleet_id,
            result.accepted_count,
            result.failed_count,
            len(frame),
            source,
        )
        return result, histories

    def from_file(
        self, fleet_id: str, path: str | Path, *, base_dir: Path | None = None
    ) -> tuple[FleetIngestionResult, list[BatteryHistoryInput]]:
        """Read one tabular file containing several batteries."""
        resolved = (
            resolve_within(base_dir, path) if base_dir is not None else Path(path).expanduser()
        )
        frame = self._read_file(resolved)
        return self.from_frame(fleet_id, frame, source=str(resolved.name))

    def from_directory(
        self, fleet_id: str, directory: str | Path, *, pattern: str = "*"
    ) -> tuple[FleetIngestionResult, list[BatteryHistoryInput]]:
        """Read a directory of per-battery files; the file stem names the cell."""
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise FleetIngestionError(f"Fleet directory not found: {root}")

        paths = sorted(
            p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not paths:
            raise FleetIngestionError(
                f"No {'/'.join(SUPPORTED_SUFFIXES)} files in {root} matching {pattern!r}."
            )

        limit = self.cfg.fleet.max_batteries_per_batch
        records: list[BatteryIngestionRecord] = []
        accepted: list[FleetBatteryReference] = []
        histories: list[BatteryHistoryInput] = []
        warnings: list[str] = []
        fingerprints: list[str] = []
        seen: set[str] = set()

        for index, path in enumerate(paths):
            battery_id = path.stem
            if battery_id in seen:
                records.append(
                    BatteryIngestionRecord(
                        battery_id=battery_id,
                        status=ProcessingStatus.FAILED,
                        errors=[
                            f"Duplicate battery id: {path.name} names a cell already "
                            "read from another file in this directory."
                        ],
                        source=path.name,
                    )
                )
                continue
            seen.add(battery_id)
            if index >= limit:
                records.append(
                    BatteryIngestionRecord(
                        battery_id=battery_id,
                        status=ProcessingStatus.FAILED,
                        errors=[
                            f"Not processed: the batch limit of {limit} batteries was reached."
                        ],
                        source=path.name,
                    )
                )
                continue
            try:
                frame = self._read_file(path)
            except FleetIngestionError as exc:
                records.append(
                    BatteryIngestionRecord(
                        battery_id=battery_id,
                        status=ProcessingStatus.FAILED,
                        errors=[str(exc)],
                        source=path.name,
                    )
                )
                continue
            if "battery_id" in frame.columns:
                distinct = {str(v) for v in frame["battery_id"].dropna().unique()}
                if len(distinct) > 1:
                    records.append(
                        BatteryIngestionRecord(
                            battery_id=battery_id,
                            status=ProcessingStatus.FAILED,
                            errors=[
                                f"{path.name} contains {len(distinct)} battery ids; a "
                                "per-battery file must contain exactly one cell."
                            ],
                            source=path.name,
                        )
                    )
                    continue
                if distinct:
                    battery_id = distinct.pop()
            fingerprints.append(frame_fingerprint(frame))
            record, history = self._validate(battery_id, frame, source=path.name)
            records.append(record)
            if history is not None:
                histories.append(history)
                accepted.append(history.reference())

        result = FleetIngestionResult(
            fleet_id=fleet_id,
            source=str(root.name),
            accepted=accepted,
            records=records,
            warnings=warnings,
            data_fingerprint=hashlib.sha256("".join(fingerprints).encode()).hexdigest()[:16],
            source_metadata={"n_files": len(paths), "directory": root.name},
        )
        return result, histories

    def from_processed_cycles(
        self, fleet_id: str | None = None, *, path: Path | None = None
    ) -> tuple[FleetIngestionResult, list[BatteryHistoryInput]]:
        """Ingest the Milestone 1 processed cycle table as a fleet.

        This is the *real-data* fleet source: the same measured cells the models
        were built from. It is small (the NASA cohort is a handful of cells) and
        it is not a production fleet — see docs/MILESTONE_3_LIMITATIONS.md.
        """
        source = path or (self.cfg.paths.processed_dir / "cycles.parquet")
        if not Path(source).is_file():
            raise FleetIngestionError(
                f"No processed cycle table at {source}. Run "
                "`python scripts/run_pipeline.py --config configs/default.yaml` first."
            )
        frame = pd.read_parquet(source)
        return self.from_frame(
            fleet_id or self.cfg.fleet.default_fleet_id,
            _drop_label_columns(frame),
            source=f"processed:{Path(source).name}",
        )

    def from_records(
        self,
        fleet_id: str,
        batteries: Mapping[str, Sequence[Mapping[str, Any]]] | Iterable[tuple[str, pd.DataFrame]],
        *,
        source: str = "request",
        max_batteries: int | None = None,
    ) -> tuple[FleetIngestionResult, list[BatteryHistoryInput]]:
        """Ingest in-memory per-battery records — what an API request becomes."""
        items: list[tuple[str, pd.DataFrame]] = []
        if isinstance(batteries, Mapping):
            for battery_id, rows in batteries.items():
                items.append((str(battery_id), pd.DataFrame(list(rows))))
        else:
            items = [(str(bid), frame) for bid, frame in batteries]

        limit = max_batteries or self.cfg.fleet.max_batteries_per_request
        records: list[BatteryIngestionRecord] = []
        accepted: list[FleetBatteryReference] = []
        histories: list[BatteryHistoryInput] = []
        seen: set[str] = set()

        for index, (battery_id, frame) in enumerate(items):
            if battery_id in seen:
                records.append(
                    BatteryIngestionRecord(
                        battery_id=battery_id,
                        status=ProcessingStatus.FAILED,
                        errors=["Duplicate battery_id in the submitted fleet."],
                        source=source,
                    )
                )
                continue
            seen.add(battery_id)
            if index >= limit:
                records.append(
                    BatteryIngestionRecord(
                        battery_id=battery_id,
                        status=ProcessingStatus.FAILED,
                        errors=[
                            f"Not processed: this request exceeds the online limit of "
                            f"{limit} batteries. Use the batch pipeline for larger fleets."
                        ],
                        source=source,
                    )
                )
                continue
            record, history = self._validate(battery_id, frame, source=source)
            records.append(record)
            if history is not None:
                histories.append(history)
                accepted.append(history.reference())

        combined = (
            pd.concat([h.history.assign(battery_id=h.battery_id) for h in histories])
            if histories
            else pd.DataFrame()
        )
        return (
            FleetIngestionResult(
                fleet_id=fleet_id,
                source=source,
                accepted=accepted,
                records=records,
                data_fingerprint=frame_fingerprint(combined),
                source_metadata={"n_batteries_supplied": len(items)},
            ),
            histories,
        )

    # -- validation --------------------------------------------------------
    def _validate(
        self, battery_id: str, frame: pd.DataFrame, *, source: str
    ) -> tuple[BatteryIngestionRecord, BatteryHistoryInput | None]:
        """Validate one cell. Returns ``(record, history or None)``.

        Errors reject the cell; warnings travel with it. The distinction is the
        whole design: a non-monotonic cycle index is repairable by sorting and
        worth saying out loud, while a duplicated cycle index is not repairable
        without guessing which row is the real one.
        """
        errors: list[str] = []
        warnings: list[str] = []

        cleaned_id = str(battery_id).strip()
        if not cleaned_id:
            errors.append("Blank battery_id.")
        elif any(ch in cleaned_id for ch in ("/", "\\", "\0")):
            errors.append("battery_id must not contain path separators.")
        elif len(cleaned_id) > 64:
            errors.append("battery_id exceeds 64 characters.")

        if frame is None or len(frame) == 0:
            errors.append("No cycle rows supplied.")
            return self._failed(cleaned_id, errors, source=source, n_rows=0), None

        missing = [c for c in _MINIMUM_COLUMNS if c not in frame.columns]
        if missing:
            errors.append(f"Missing required column(s): {missing}.")
            return (
                self._failed(cleaned_id, errors, source=source, n_rows=len(frame)),
                None,
            )

        working = frame.copy()
        working["battery_id"] = cleaned_id

        cycles = pd.to_numeric(working["cycle_index"], errors="coerce")
        if cycles.isna().any():
            errors.append(
                f"{int(cycles.isna().sum())} row(s) have a missing or non-numeric " "cycle_index."
            )
        if (cycles.dropna() < 0).any():
            errors.append("cycle_index contains negative values.")

        duplicated = int(cycles.duplicated(keep=False).sum())
        if duplicated:
            errors.append(
                f"{duplicated} row(s) share a cycle_index. A cycle record must have "
                "one row per cycle; deduplicate before submitting."
            )

        capacity = pd.to_numeric(working["capacity_ah"], errors="coerce")
        if not np.isfinite(capacity.to_numpy(dtype=float)).any():
            errors.append("capacity_ah has no finite values.")

        if len(working) < self.cfg.fleet.min_cycles_per_battery:
            errors.append(
                f"{len(working)} cycle(s) supplied; at least "
                f"{self.cfg.fleet.min_cycles_per_battery} are required to ingest a cell."
            )
        if len(working) > self.cfg.service.max_history_cycles:
            errors.append(
                f"{len(working)} rows exceed the configured maximum of "
                f"{self.cfg.service.max_history_cycles} per battery."
            )

        if errors:
            return (
                self._failed(cleaned_id, errors, source=source, n_rows=len(working)),
                None,
            )

        ordered = cycles.is_monotonic_increasing
        if not ordered:
            warnings.append("cycle_index was not increasing; rows were sorted before use.")
            working = working.assign(cycle_index=cycles).sort_values("cycle_index", kind="stable")
        working = working.reset_index(drop=True)

        gaps = np.diff(np.sort(cycles.dropna().to_numpy()))
        if gaps.size and int(gaps.max()) > self.cfg.quality.max_cycle_gap:
            warnings.append(
                f"Largest cycle gap is {int(gaps.max())}, above the configured "
                f"{self.cfg.quality.max_cycle_gap}; trailing windows span a "
                "discontinuity in the ageing clock."
            )
        missing_capacity = int(capacity.isna().sum())
        if missing_capacity:
            warnings.append(f"{missing_capacity} cycle(s) have no capacity measurement.")

        record = BatteryIngestionRecord(
            battery_id=cleaned_id,
            status=ProcessingStatus.SUCCESS,
            n_rows=int(len(working)),
            warnings=warnings,
            source=source,
        )
        history = BatteryHistoryInput(
            battery_id=cleaned_id,
            history=working,
            source=source,
            warnings=warnings,
        )
        return record, history

    @staticmethod
    def _failed(
        battery_id: str, errors: list[str], *, source: str, n_rows: int
    ) -> BatteryIngestionRecord:
        logger.warning("Battery %s rejected at ingestion: %s", battery_id or "<blank>", errors)
        return BatteryIngestionRecord(
            battery_id=battery_id or "<blank>",
            status=ProcessingStatus.FAILED,
            n_rows=n_rows,
            errors=errors,
            source=source,
        )

    # -- file reading ------------------------------------------------------
    def _read_file(self, path: Path) -> pd.DataFrame:
        if not path.is_file():
            raise FleetIngestionError(f"Fleet file not found: {path}")
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise FleetIngestionError(
                f"Unsupported fleet file type {suffix!r}. Supported: "
                f"{', '.join(SUPPORTED_SUFFIXES)}."
            )
        size = path.stat().st_size
        if size > self.cfg.fleet.max_upload_bytes:
            raise FleetIngestionError(
                f"{path.name} is {size / 1e6:.1f} MB; the configured limit is "
                f"{self.cfg.fleet.max_upload_bytes / 1e6:.1f} MB."
            )
        try:
            if suffix == ".parquet":
                return pd.read_parquet(path)
            if suffix == ".csv":
                return pd.read_csv(path)
            return pd.read_json(path, orient="records")
        except Exception as exc:  # noqa: BLE001 - report the file, not a stack trace
            raise FleetIngestionError(
                f"Could not read {path.name}: {type(exc).__name__}: {exc}"
            ) from exc


#: Columns that encode the supervised label. A fleet history is *input*, and
#: handing the twin a frame that still carries its own answer would make every
#: downstream number meaningless.
_LABEL_COLUMNS = (
    "rul_cycles",
    "rul_raw_cycles",
    "eol_cycle",
    "life_fraction",
    "is_censored",
    "soh_target",
    "soh_target_future",
    "soh_health_class",
    "soh_reference_capacity_ah",
    "failure_within_horizon",
    "split",
)


def _drop_label_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
        c
        for c in frame.columns
        if c not in _LABEL_COLUMNS and not str(c).startswith("failure_within_horizon")
    ]
    return frame[keep]
