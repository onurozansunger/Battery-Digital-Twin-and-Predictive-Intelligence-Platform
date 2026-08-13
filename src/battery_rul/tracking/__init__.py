"""Experiment tracking.

One interface, two backends:

``file``    a local run store under ``artifacts/tracking`` — JSON per run, no
            server, no dependency. The default, because a tracking system that
            needs infrastructure to record a five-second experiment does not get
            used.
``mlflow``  MLflow's local file store (or a remote URI when one is configured).
            Used when MLflow is installed and selected; never required.

The interface is the same either way, so a run logged today can be compared
tomorrow regardless of which backend recorded it.

Never logged: raw cycle measurements. Configurations, metrics, fingerprints and
artifact *paths* are provenance; a battery's telemetry in a tracking store is a
copy of someone's operational data in a place nobody is auditing.
"""

from __future__ import annotations

from battery_rul.tracking.experiment import (
    ExperimentTracker,
    FileTracker,
    MLflowTracker,
    RunRecord,
    build_tracker,
    compare_runs,
)

__all__ = [
    "ExperimentTracker",
    "FileTracker",
    "MLflowTracker",
    "RunRecord",
    "build_tracker",
    "compare_runs",
]
