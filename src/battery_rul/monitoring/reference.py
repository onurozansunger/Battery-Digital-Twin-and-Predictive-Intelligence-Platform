"""The reference distribution artifact.

Drift is a comparison, and a comparison needs a fixed, versioned, inspectable
other side. This module builds that: per-feature summary statistics and binned
histograms, written as **JSON**.

JSON rather than a pickle, deliberately. A monitoring artifact is read by a
long-running service, and a pickle is executable content; a reference file that
could be swapped for a malicious one would turn monitoring into a code-execution
path. Everything needed for PSI, KS, Wasserstein and JS survives the round trip
as numbers.

Partition discipline
--------------------
The reference is fitted on the **training** partition (or train+validation when
configured), never on the final test partition. A drift reference fitted on test
rows would make the held-out result part of the serving machinery, and every
subsequent claim about that partition would be contaminated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from battery_rul.config import ExperimentConfig
from battery_rul.utils.io import save_json
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "REFERENCE_SCHEMA_VERSION",
    "ReferenceDistribution",
    "build_reference_distribution",
    "load_reference",
    "reference_path",
    "save_reference",
]

REFERENCE_SCHEMA_VERSION = "3.0"

#: A feature with fewer than this many distinct values is treated as categorical
#: regardless of dtype. One-hot and missingness-indicator columns are numeric in
#: pandas and continuous in nothing.
_CATEGORICAL_MAX_UNIQUE = 12


@dataclass
class ReferenceDistribution:
    """What the training data looked like, in a form drift metrics can use."""

    reference_id: str
    created_at_utc: str
    partition: str
    schema_version: str = REFERENCE_SCHEMA_VERSION
    n_rows: int = 0
    n_batteries: int = 0
    feature_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    prediction_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    dataset_fingerprint: str = ""
    data_fingerprint: str = ""
    model_version: str | None = None
    git_revision: str | None = None
    notes: str = ""

    @property
    def feature_names(self) -> list[str]:
        return sorted(self.feature_stats)

    def fingerprint(self) -> str:
        """Stable hash of the reference's content, recorded in every report."""
        payload = json.dumps(
            {"features": self.feature_stats, "predictions": self.prediction_stats},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReferenceDistribution:
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            logger.warning("Reference artifact carries unknown keys (ignored): %s", unknown)
        return cls(**{k: v for k, v in payload.items() if k in known})


def _summarise_numeric(values: np.ndarray, n_bins: int) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "type": "numerical",
            "count": 0,
            "constant": True,
            "reason": "no finite values in the reference partition",
        }
    low, high = float(finite.min()), float(finite.max())
    constant = bool(np.isclose(low, high))
    if constant:
        # A constant feature has no distribution to compare against. Recorded as
        # such rather than binned into a degenerate histogram that every drift
        # metric then divides by zero on.
        edges: list[float] = [low, high]
        frequencies: list[float] = [1.0]
    else:
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        raw_edges = np.unique(np.quantile(finite, quantiles))
        if raw_edges.size < 3:
            raw_edges = np.linspace(low, high, min(n_bins, 4) + 1)
        # Open the outer edges so unseen extremes in a current batch land in the
        # end bins rather than being dropped.
        edges = [float(-np.inf), *[float(e) for e in raw_edges[1:-1]], float(np.inf)]
        counts, _ = np.histogram(finite, bins=[low - 1e-9, *edges[1:-1], high + 1e-9])
        total = max(int(counts.sum()), 1)
        frequencies = [float(c / total) for c in counts]
    return {
        "type": "numerical",
        "count": int(finite.size),
        "missing_rate": round(float(1.0 - finite.size / max(values.size, 1)), 6),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "min": low,
        "max": high,
        "quantiles": {
            f"q{int(round(100 * q))}": float(np.quantile(finite, q))
            for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        },
        "bin_edges": edges,
        "bin_frequencies": frequencies,
        "constant": constant,
    }


def _summarise_categorical(values: pd.Series) -> dict[str, Any]:
    counts = values.dropna().astype(str).value_counts()
    total = max(int(counts.sum()), 1)
    return {
        "type": "categorical",
        "count": int(total),
        "missing_rate": round(float(values.isna().mean()), 6),
        "categories": {str(k): float(v / total) for k, v in counts.items()},
        "n_categories": int(counts.size),
        "constant": bool(counts.size <= 1),
    }


def build_reference_distribution(
    frame: pd.DataFrame,
    cfg: ExperimentConfig,
    *,
    feature_names: list[str] | None = None,
    reference_id: str | None = None,
    partition: str | None = None,
    prediction_columns: dict[str, str] | None = None,
    dataset_fingerprint: str = "",
    model_version: str | None = None,
    notes: str = "",
) -> ReferenceDistribution:
    """Summarise a reference frame into a drift-comparable artifact.

    ``frame`` must already be restricted to the reference partition; this
    function does not filter, because a helper that silently decided which rows
    were "training" would be the easiest place in the codebase to leak the test
    partition into monitoring.
    """
    from battery_rul.utils.io import environment_fingerprint

    if frame is None or frame.empty:
        raise ValueError("Cannot build a reference distribution from an empty frame.")

    columns = feature_names or [
        c
        for c in frame.columns
        if c not in ("battery_id", "dataset", "split", "timestamp")
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    columns = columns[: cfg.monitoring.drift.max_features]

    limit = cfg.monitoring.max_reference_samples
    working = frame
    if len(frame) > limit:
        # Deterministic subsample: monitoring must not change verdict because a
        # reference was rebuilt on a different random draw.
        working = frame.iloc[:: max(len(frame) // limit, 1)].head(limit)
        logger.info(
            "Reference subsampled deterministically from %d to %d rows", len(frame), len(working)
        )

    n_bins = cfg.monitoring.drift.n_bins
    stats: dict[str, dict[str, Any]] = {}
    for column in columns:
        if column not in working.columns:
            continue
        series = working[column]
        numeric = pd.to_numeric(series, errors="coerce")
        distinct = int(numeric.dropna().nunique())
        if distinct <= _CATEGORICAL_MAX_UNIQUE and distinct > 0:
            stats[column] = _summarise_categorical(series)
        else:
            stats[column] = _summarise_numeric(numeric.to_numpy(dtype=float), n_bins)

    prediction_stats: dict[str, dict[str, Any]] = {}
    for name, column in (prediction_columns or {}).items():
        if column not in working.columns:
            continue
        values = pd.to_numeric(working[column], errors="coerce").to_numpy(dtype=float)
        prediction_stats[name] = _summarise_numeric(values, n_bins)

    return ReferenceDistribution(
        reference_id=reference_id or cfg.monitoring.reference_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        partition=partition or cfg.monitoring.reference_partition,
        n_rows=int(len(working)),
        n_batteries=int(working["battery_id"].nunique()) if "battery_id" in working else 0,
        feature_stats=stats,
        prediction_stats=prediction_stats,
        dataset_fingerprint=dataset_fingerprint,
        data_fingerprint=cfg.data_fingerprint(),
        model_version=model_version,
        git_revision=environment_fingerprint().get("git_revision"),
        notes=notes,
    )


def reference_path(cfg: ExperimentConfig, reference_id: str | None = None) -> Path:
    """Where a named reference artifact lives. Ids are sanitised, not trusted."""
    name = reference_id or cfg.monitoring.reference_id
    safe = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", "."))
    if not safe or safe != name:
        raise ValueError(
            f"Invalid reference id {name!r}: only letters, digits, '-', '_' and '.' "
            "are permitted, so a reference id can never become a path."
        )
    return Path(cfg.artifacts.monitoring_dir) / "reference_distributions" / f"{safe}.json"


def save_reference(reference: ReferenceDistribution, cfg: ExperimentConfig) -> Path:
    path = reference_path(cfg, reference.reference_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = reference.to_dict()
    payload["fingerprint"] = reference.fingerprint()
    save_json(payload, path)
    logger.info(
        "Reference distribution '%s' written: %d features, %d rows -> %s",
        reference.reference_id,
        len(reference.feature_stats),
        reference.n_rows,
        path.name,
    )
    return path


def load_reference(cfg: ExperimentConfig, reference_id: str | None = None) -> ReferenceDistribution:
    """Load a reference artifact, refusing an incompatible schema version."""
    path = reference_path(cfg, reference_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"No reference distribution at {path}. Build one with "
            "`python -m battery_rul.pipelines.build_reference`."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("fingerprint", None)
    reference = ReferenceDistribution.from_dict(payload)
    major = str(reference.schema_version).split(".")[0]
    if major != REFERENCE_SCHEMA_VERSION.split(".")[0]:
        raise ValueError(
            f"Reference artifact {path.name} uses schema {reference.schema_version}; "
            f"this build supports {REFERENCE_SCHEMA_VERSION}. Rebuild it."
        )
    return reference
