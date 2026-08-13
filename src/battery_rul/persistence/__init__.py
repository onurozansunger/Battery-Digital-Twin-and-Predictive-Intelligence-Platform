"""Storage for fleet snapshots, monitoring runs, alerts and prediction records.

A repository interface plus a SQLite implementation. SQLite because it is in the
standard library, needs no server, handles concurrent readers, and is a real
database with real transactions — everything a single-node prototype needs, and
nothing it does not. Introducing PostgreSQL here would add an operational
dependency to justify a schema this small.

Route handlers never touch SQL. They receive a repository, which is what lets
the API tests run against an in-memory store and what makes replacing the
backend a change in one file.
"""

from __future__ import annotations

from battery_rul.config import ExperimentConfig
from battery_rul.persistence.base import (
    PersistenceError,
    ReadOnlyStoreError,
    Repository,
)
from battery_rul.persistence.sqlite import SQLiteRepository

__all__ = [
    "PersistenceError",
    "ReadOnlyStoreError",
    "Repository",
    "SQLiteRepository",
    "build_repository",
]


def build_repository(cfg: ExperimentConfig) -> Repository:
    """Construct the configured repository.

    ``persistence.backend='memory'`` maps to SQLite's in-memory database rather
    than a separate dict-based implementation: one implementation means the
    tests exercise the same SQL the deployment runs.
    """
    if cfg.persistence.backend == "memory":
        return SQLiteRepository(cfg=cfg, database=":memory:")
    return SQLiteRepository(cfg=cfg)
