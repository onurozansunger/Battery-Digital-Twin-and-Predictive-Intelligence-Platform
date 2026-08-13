"""SQLite implementation of the repository contract.

Design notes
------------
*Documents in columns.* Each row stores its indexed identifiers as columns and
the full object as a JSON payload. The domain objects are versioned Pydantic
models that will gain fields; normalising them into forty columns would make
every schema addition a migration for no query benefit at this scale.

*WAL mode.* Readers do not block the writer, which matters because the dashboard
polls while a batch writes.

*One connection per thread.* ``sqlite3`` connections are not shareable across
threads, and the API is threaded. A thread-local connection is simpler and safer
than a pool for this volume.

*Explicit failures.* Every write raises :class:`PersistenceError` on failure, and
raises :class:`ReadOnlyStoreError` immediately when the deployment is read-only —
before touching the disk, so a read-only container fails loudly at the first
write rather than at an unpredictable one.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from battery_rul.config import ExperimentConfig
from battery_rul.fleet.domain import FleetSnapshot
from battery_rul.monitoring.domain import Alert, MonitoringSnapshot
from battery_rul.monitoring.performance import OutcomeLabel, PredictionRecord
from battery_rul.persistence.base import (
    SCHEMA_VERSION,
    PersistenceError,
    ReadOnlyStoreError,
)
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["SQLiteRepository"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fleet_snapshots (
    snapshot_id      TEXT PRIMARY KEY,
    fleet_id         TEXT NOT NULL,
    generated_at_utc TEXT NOT NULL,
    model_version    TEXT,
    batch_id         TEXT,
    battery_count    INTEGER NOT NULL DEFAULT 0,
    success_count    INTEGER NOT NULL DEFAULT 0,
    failed_count     INTEGER NOT NULL DEFAULT 0,
    schema_version   TEXT NOT NULL,
    payload          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fleet_snapshots_fleet
    ON fleet_snapshots (fleet_id, generated_at_utc DESC);
CREATE TABLE IF NOT EXISTS monitoring_snapshots (
    snapshot_id      TEXT PRIMARY KEY,
    fleet_id         TEXT NOT NULL,
    generated_at_utc TEXT NOT NULL,
    model_version    TEXT,
    overall_status   TEXT,
    schema_version   TEXT NOT NULL,
    payload          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_monitoring_snapshots_fleet
    ON monitoring_snapshots (fleet_id, generated_at_utc DESC);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id         TEXT PRIMARY KEY,
    type             TEXT NOT NULL,
    severity         TEXT NOT NULL,
    generated_at_utc TEXT NOT NULL,
    fleet_id         TEXT,
    model_version    TEXT,
    batch_id         TEXT,
    acknowledged     INTEGER NOT NULL DEFAULT 0,
    payload          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_fleet
    ON alerts (fleet_id, acknowledged, generated_at_utc DESC);
CREATE TABLE IF NOT EXISTS prediction_records (
    prediction_id    TEXT PRIMARY KEY,
    battery_id       TEXT NOT NULL,
    cycle_index      INTEGER NOT NULL,
    model_version    TEXT,
    fleet_id         TEXT,
    batch_id         TEXT,
    generated_at_utc TEXT NOT NULL,
    payload          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prediction_records_join
    ON prediction_records (battery_id, cycle_index);
CREATE TABLE IF NOT EXISTS outcome_labels (
    battery_id       TEXT NOT NULL,
    cycle_index      INTEGER NOT NULL,
    observed_at_utc  TEXT,
    label_source     TEXT,
    payload          TEXT NOT NULL,
    PRIMARY KEY (battery_id, cycle_index)
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id    TEXT PRIMARY KEY,
    fleet_id    TEXT,
    started_at_utc  TEXT,
    finished_at_utc TEXT,
    status      TEXT,
    payload     TEXT NOT NULL
);
"""


@dataclass
class SQLiteRepository:
    """Concrete repository. Satisfies :class:`~battery_rul.persistence.base.Repository`."""

    cfg: ExperimentConfig
    database: str | Path | None = None
    _local: threading.local = field(default_factory=threading.local, init=False, repr=False)
    _shared: sqlite3.Connection | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = (
            str(self.database)
            if self.database is not None
            else str(Path(self.cfg.persistence.database_path))
        )
        self.is_memory = self.path == ":memory:"
        if not self.is_memory:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    # -- connection --------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        if self.is_memory:
            # One shared connection: a second ":memory:" connection is a second,
            # empty database, which would make the in-memory backend behave
            # nothing like the file one.
            if self._shared is None:
                self._shared = sqlite3.connect(self.path, check_same_thread=False)
                self._shared.row_factory = sqlite3.Row
            return self._shared
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path, timeout=self.cfg.persistence.busy_timeout_s)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA foreign_keys=ON")
            self._local.connection = connection
        return connection

    def _initialise(self) -> None:
        try:
            connection = self._connect()
            with connection:
                connection.executescript(_SCHEMA)
                connection.execute(
                    "INSERT OR IGNORE INTO schema_meta (key, value) VALUES (?, ?)",
                    ("schema_version", SCHEMA_VERSION),
                )
        except sqlite3.Error as exc:
            raise PersistenceError(f"Could not initialise the store at {self.path}: {exc}") from exc

        stored = self.schema_version()
        if stored.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            raise PersistenceError(
                f"Store at {self.path} uses schema {stored}; this build supports "
                f"{SCHEMA_VERSION}. Point persistence.database_path at a new file or "
                "migrate the existing one."
            )

    def _guard_write(self) -> None:
        if self.cfg.deployment.read_only:
            raise ReadOnlyStoreError(
                "deployment.read_only is set: this process must not write to storage. "
                "Unset it, or route writes to the batch job that owns them."
            )

    def _execute(self, sql: str, parameters: Sequence[Any] = ()) -> sqlite3.Cursor:
        try:
            connection = self._connect()
            with connection:
                return connection.execute(sql, tuple(parameters))
        except sqlite3.Error as exc:
            raise PersistenceError(
                f"Store operation failed: {exc}\nSQL: {sql.strip()[:120]}"
            ) from exc

    def _query(self, sql: str, parameters: Sequence[Any] = ()) -> list[sqlite3.Row]:
        try:
            return self._connect().execute(sql, tuple(parameters)).fetchall()
        except sqlite3.Error as exc:
            raise PersistenceError(f"Store query failed: {exc}\nSQL: {sql.strip()[:120]}") from exc

    # -- fleet snapshots ---------------------------------------------------
    def save_fleet_snapshot(self, snapshot: FleetSnapshot) -> str:
        self._guard_write()
        self._execute(
            """INSERT OR REPLACE INTO fleet_snapshots
               (snapshot_id, fleet_id, generated_at_utc, model_version, batch_id,
                battery_count, success_count, failed_count, schema_version, payload)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot.snapshot_id,
                snapshot.fleet_id,
                snapshot.generated_at_utc,
                snapshot.model_metadata.active_model_version,
                snapshot.batch_id,
                snapshot.battery_count,
                snapshot.successfully_processed_count,
                snapshot.failed_count,
                snapshot.schema_version,
                json.dumps(snapshot.to_json_dict()),
            ),
        )
        logger.info(
            "Fleet snapshot %s stored (%s, %d batteries)",
            snapshot.snapshot_id,
            snapshot.fleet_id,
            snapshot.battery_count,
        )
        return snapshot.snapshot_id

    def get_fleet_snapshot(self, snapshot_id: str) -> FleetSnapshot | None:
        rows = self._query(
            "SELECT payload FROM fleet_snapshots WHERE snapshot_id = ?", (snapshot_id,)
        )
        return FleetSnapshot(**json.loads(rows[0]["payload"])) if rows else None

    def latest_fleet_snapshot(self, fleet_id: str) -> FleetSnapshot | None:
        rows = self._query(
            """SELECT payload FROM fleet_snapshots WHERE fleet_id = ?
               ORDER BY generated_at_utc DESC LIMIT 1""",
            (fleet_id,),
        )
        return FleetSnapshot(**json.loads(rows[0]["payload"])) if rows else None

    def list_fleet_snapshots(
        self, fleet_id: str | None = None, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Metadata only — listing 50 full snapshots would be megabytes."""
        sql = """SELECT snapshot_id, fleet_id, generated_at_utc, model_version, batch_id,
                        battery_count, success_count, failed_count, schema_version
                 FROM fleet_snapshots"""
        parameters: list[Any] = []
        if fleet_id:
            sql += " WHERE fleet_id = ?"
            parameters.append(fleet_id)
        sql += " ORDER BY generated_at_utc DESC LIMIT ?"
        parameters.append(int(limit))
        return [dict(row) for row in self._query(sql, parameters)]

    # -- monitoring --------------------------------------------------------
    def save_monitoring_snapshot(self, snapshot: MonitoringSnapshot) -> str:
        self._guard_write()
        self._execute(
            """INSERT OR REPLACE INTO monitoring_snapshots
               (snapshot_id, fleet_id, generated_at_utc, model_version, overall_status,
                schema_version, payload)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot.snapshot_id,
                snapshot.fleet_id,
                snapshot.generated_at_utc,
                snapshot.model_version,
                snapshot.overall_status.value,
                snapshot.schema_version,
                json.dumps(snapshot.to_json_dict()),
            ),
        )
        return snapshot.snapshot_id

    def latest_monitoring_snapshot(self, fleet_id: str | None = None) -> MonitoringSnapshot | None:
        sql = "SELECT payload FROM monitoring_snapshots"
        parameters: list[Any] = []
        if fleet_id:
            sql += " WHERE fleet_id = ?"
            parameters.append(fleet_id)
        sql += " ORDER BY generated_at_utc DESC LIMIT 1"
        rows = self._query(sql, parameters)
        return MonitoringSnapshot(**json.loads(rows[0]["payload"])) if rows else None

    def get_monitoring_snapshot(self, snapshot_id: str) -> MonitoringSnapshot | None:
        rows = self._query(
            "SELECT payload FROM monitoring_snapshots WHERE snapshot_id = ?", (snapshot_id,)
        )
        return MonitoringSnapshot(**json.loads(rows[0]["payload"])) if rows else None

    def list_monitoring_snapshots(
        self, fleet_id: str | None = None, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        sql = """SELECT snapshot_id, fleet_id, generated_at_utc, model_version,
                        overall_status, schema_version FROM monitoring_snapshots"""
        parameters: list[Any] = []
        if fleet_id:
            sql += " WHERE fleet_id = ?"
            parameters.append(fleet_id)
        sql += " ORDER BY generated_at_utc DESC LIMIT ?"
        parameters.append(int(limit))
        return [dict(row) for row in self._query(sql, parameters)]

    # -- alerts ------------------------------------------------------------
    def save_alerts(self, alerts: list[Alert]) -> int:
        self._guard_write()
        if not alerts:
            return 0
        for alert in alerts:
            # INSERT OR IGNORE, not REPLACE: alert ids are deterministic, so a
            # repeated finding must not wipe an existing acknowledgement.
            self._execute(
                """INSERT OR IGNORE INTO alerts
                   (alert_id, type, severity, generated_at_utc, fleet_id, model_version,
                    batch_id, acknowledged, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                (
                    alert.alert_id,
                    alert.type.value,
                    alert.severity.value,
                    alert.generated_at_utc,
                    alert.fleet_id,
                    alert.model_version,
                    alert.batch_id,
                    json.dumps(alert.model_dump(mode="json")),
                ),
            )
        return len(alerts)

    def list_alerts(
        self,
        fleet_id: str | None = None,
        *,
        acknowledged: bool | None = None,
        limit: int = 100,
    ) -> list[Alert]:
        sql = "SELECT payload, acknowledged FROM alerts"
        clauses: list[str] = []
        parameters: list[Any] = []
        if fleet_id:
            clauses.append("fleet_id = ?")
            parameters.append(fleet_id)
        if acknowledged is not None:
            clauses.append("acknowledged = ?")
            parameters.append(1 if acknowledged else 0)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY generated_at_utc DESC LIMIT ?"
        parameters.append(int(limit))

        out: list[Alert] = []
        for row in self._query(sql, parameters):
            payload = json.loads(row["payload"])
            payload["acknowledged"] = bool(row["acknowledged"])
            out.append(Alert(**payload))
        return out

    def acknowledge_alert(self, alert_id: str, *, by: str) -> bool:
        self._guard_write()
        cursor = self._execute("UPDATE alerts SET acknowledged = 1 WHERE alert_id = ?", (alert_id,))
        if cursor.rowcount == 0:
            return False
        rows = self._query("SELECT payload FROM alerts WHERE alert_id = ?", (alert_id,))
        payload = json.loads(rows[0]["payload"])
        payload.update(
            acknowledged=True,
            acknowledged_at_utc=datetime.now(UTC).isoformat(),
            acknowledged_by=by,
        )
        self._execute(
            "UPDATE alerts SET payload = ? WHERE alert_id = ?", (json.dumps(payload), alert_id)
        )
        return True

    # -- predictions and labels -------------------------------------------
    def save_prediction_records(self, records: list[PredictionRecord]) -> int:
        self._guard_write()
        for record in records:
            self._execute(
                """INSERT OR REPLACE INTO prediction_records
                   (prediction_id, battery_id, cycle_index, model_version, fleet_id,
                    batch_id, generated_at_utc, payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.prediction_id,
                    record.battery_id,
                    record.cycle_index,
                    record.model_version,
                    record.fleet_id,
                    record.batch_id,
                    record.generated_at_utc,
                    json.dumps(record.model_dump(mode="json")),
                ),
            )
        return len(records)

    def list_prediction_records(
        self, *, model_version: str | None = None, limit: int = 10_000
    ) -> list[PredictionRecord]:
        sql = "SELECT payload FROM prediction_records"
        parameters: list[Any] = []
        if model_version:
            sql += " WHERE model_version = ?"
            parameters.append(model_version)
        sql += " ORDER BY generated_at_utc DESC LIMIT ?"
        parameters.append(int(limit))
        return [
            PredictionRecord(**json.loads(row["payload"])) for row in self._query(sql, parameters)
        ]

    def save_outcome_labels(self, labels: list[OutcomeLabel]) -> int:
        self._guard_write()
        for label in labels:
            self._execute(
                """INSERT OR REPLACE INTO outcome_labels
                   (battery_id, cycle_index, observed_at_utc, label_source, payload)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    label.battery_id,
                    label.cycle_index,
                    label.observed_at_utc,
                    label.label_source,
                    json.dumps(label.model_dump(mode="json")),
                ),
            )
        return len(labels)

    def list_outcome_labels(self, *, limit: int = 10_000) -> list[OutcomeLabel]:
        rows = self._query(
            "SELECT payload FROM outcome_labels ORDER BY battery_id, cycle_index LIMIT ?",
            (int(limit),),
        )
        return [OutcomeLabel(**json.loads(row["payload"])) for row in rows]

    # -- batches -----------------------------------------------------------
    def save_batch(self, batch_id: str, payload: dict[str, Any]) -> str:
        self._guard_write()
        self._execute(
            """INSERT OR REPLACE INTO batches
               (batch_id, fleet_id, started_at_utc, finished_at_utc, status, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                batch_id,
                payload.get("fleet_id"),
                payload.get("started_at_utc"),
                payload.get("finished_at_utc"),
                payload.get("status", "completed"),
                json.dumps(payload),
            ),
        )
        return batch_id

    def list_batches(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self._query(
                "SELECT batch_id, fleet_id, started_at_utc, finished_at_utc, status "
                "FROM batches ORDER BY started_at_utc DESC LIMIT ?",
                (int(limit),),
            )
        ]

    # -- lifecycle ---------------------------------------------------------
    def schema_version(self) -> str:
        rows = self._query("SELECT value FROM schema_meta WHERE key = 'schema_version'")
        return str(rows[0]["value"]) if rows else SCHEMA_VERSION

    def prune(self, *, older_than_days: int | None = None) -> int:
        """Delete rows older than the retention window. Never automatic.

        Called explicitly by an operator or a scheduled job. Retention that runs
        on its own inside a service is how a monitoring history disappears the
        week before someone needs it.
        """
        self._guard_write()
        days = (
            older_than_days if older_than_days is not None else self.cfg.persistence.retention_days
        )
        if not days:
            return 0
        cutoff = datetime.now(UTC).timestamp() - float(days) * 86400.0
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=UTC).isoformat()
        deleted = 0
        for table in ("fleet_snapshots", "monitoring_snapshots", "alerts"):
            cursor = self._execute(
                f"DELETE FROM {table} WHERE generated_at_utc < ?", (cutoff_iso,)  # noqa: S608
            )
            deleted += cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
        logger.info("Pruned %d row(s) older than %s", deleted, cutoff_iso)
        return deleted

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None
        if self._shared is not None:
            self._shared.close()
            self._shared = None
