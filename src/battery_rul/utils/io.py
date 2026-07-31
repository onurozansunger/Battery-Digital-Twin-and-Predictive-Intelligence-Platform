"""Serialisation helpers.

A single place that knows how the project writes things to disk, so artifact
naming, atomicity and NumPy-JSON quirks are handled once.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "atomic_write_text",
    "environment_fingerprint",
    "load_json",
    "load_pickle",
    "read_table",
    "save_json",
    "save_pickle",
    "timestamp",
    "write_table",
]


def timestamp() -> str:
    """UTC timestamp safe for filenames."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands NumPy scalars, arrays, Paths and datetimes."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            value = float(o)
            return None if (np.isnan(value) or np.isinf(value)) else value
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        if isinstance(o, set):
            return sorted(o)
        return super().default(o)


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write via a temp file + rename so a crash can never leave a half-file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def save_json(obj: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Persist ``obj`` as JSON, NumPy-aware and atomic."""
    return atomic_write_text(path, json.dumps(obj, indent=indent, cls=NumpyJSONEncoder) + "\n")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_pickle(obj: Any, path: str | Path, *, compress: int = 3) -> Path:
    """Persist a Python object with joblib (better for large NumPy payloads)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=compress)
    logger.debug("Wrote %s (%.1f KB)", path, path.stat().st_size / 1024)
    return path


def load_pickle(path: str | Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}. Have you run the training pipeline?")
    return joblib.load(path)


def write_table(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Write a DataFrame, dispatching on extension (``.parquet`` / ``.csv``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False, **kwargs)
    elif suffix == ".csv":
        df.to_csv(path, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported table format: {suffix!r} ({path})")
    logger.debug("Wrote %s rows=%d cols=%d", path, len(df), df.shape[1])
    return path


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Table not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    raise ValueError(f"Unsupported table format: {suffix!r} ({path})")


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None


def environment_fingerprint() -> dict[str, Any]:
    """Provenance block embedded in every metrics/report artifact."""
    versions: dict[str, str] = {}
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "lightgbm", "catboost", "torch", "optuna"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - optional dependency, never fatal
            versions[mod] = "not-installed"
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_revision": _git_revision(),
        "packages": versions,
    }
