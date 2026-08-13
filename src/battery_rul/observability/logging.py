"""Structured logging with an ambient context.

Why a context variable rather than threading identifiers through every call:
a fleet batch calls the twin service once per cell, which calls feature
generation, which logs. Passing ``batch_id`` down that chain would put an
operational concern into six function signatures that have nothing to do with
operations. A :class:`contextvars.ContextVar` binds it once at the boundary
(request handler, batch runner) and every line emitted underneath carries it —
including from library code that knows nothing about this module.

What is deliberately *not* logged
---------------------------------
Raw cycle histories. A battery history is measurement data belonging to whoever
operates the cell, it is large, and a log file is the wrong place for it. The
formatter drops any ``history``/``records``/``frame`` field defensively, so an
accidental ``extra={"history": frame}`` at a call site cannot leak one.
"""

from __future__ import annotations

import contextvars
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "LogContext",
    "StructuredFormatter",
    "bind_context",
    "configure_structured_logging",
    "current_context",
    "log_event",
]

#: Fields that must never reach a log record, whatever a call site passes.
_FORBIDDEN_EXTRA = frozenset({"history", "records", "frame", "cycles", "payload", "raw"})

#: Standard LogRecord attributes, so the formatter can find the custom ones.
_RESERVED = frozenset(
    (
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    )
)


@dataclass(frozen=True, slots=True)
class LogContext:
    """Identifiers shared by every log line emitted inside one unit of work."""

    service: str = "battery-rul"
    request_id: str | None = None
    batch_id: str | None = None
    fleet_id: str | None = None
    battery_id: str | None = None
    model_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# The default is None rather than a LogContext instance: a mutable default on a
# ContextVar is shared by every task that never sets one, and `LogContext` being
# frozen today is not a guarantee it stays frozen.
_CONTEXT: contextvars.ContextVar[LogContext | None] = contextvars.ContextVar(
    "battery_rul_log_context", default=None
)

_EMPTY_CONTEXT = LogContext()


def current_context() -> LogContext:
    """The context in force for this task/thread."""
    return _CONTEXT.get() or _EMPTY_CONTEXT


@contextmanager
def bind_context(**fields: Any) -> Iterator[LogContext]:
    """Bind identifiers for the duration of the block, restoring them after.

    Only the fields supplied are changed; the rest are inherited, so a
    per-battery block inside a batch keeps the batch's identifiers.
    """
    known = {k: v for k, v in fields.items() if k in LogContext.__dataclass_fields__}
    unknown = sorted(set(fields) - set(known))
    if unknown:
        raise ValueError(f"Unknown log-context field(s): {unknown}")
    updated = replace(current_context(), **known)
    token = _CONTEXT.set(updated)
    try:
        yield updated
    finally:
        _CONTEXT.reset(token)


class StructuredFormatter(logging.Formatter):
    """One JSON object per line: timestamp, level, service, context, event."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(current_context().to_dict())

        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            if key in _FORBIDDEN_EXTRA:
                payload[key] = "<omitted: raw measurement data is never logged>"
                continue
            payload[key] = _jsonable(value)

        if record.exc_info:
            payload["error_type"] = getattr(record.exc_info[0], "__name__", "Exception")
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _jsonable(value: Any) -> Any:
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value][:100]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in list(value.items())[:100]}
    return str(value)


def configure_structured_logging(
    level: int | str = logging.INFO, *, log_file: str | None = None
) -> logging.Logger:
    """Replace the root handlers with a JSON stream handler.

    Used by containers and batch jobs, where a log line is going to be parsed by
    a collector rather than read by a person. Interactive runs keep the Rich
    console handler from :mod:`battery_rul.utils.logging`.
    """
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(StructuredFormatter())
    root.addHandler(stream_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(StructuredFormatter())
        root.addHandler(file_handler)
    root.setLevel(level)
    for noisy in ("matplotlib", "PIL", "optuna", "shap", "numba", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return root


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    status: str = "ok",
    duration_ms: float | None = None,
    error_code: str | None = None,
    **fields: Any,
) -> None:
    """Emit one structured event.

    ``event`` is a stable machine-readable name (``fleet_batch_completed``), not
    a sentence: dashboards group on it, and a message that changes wording
    breaks every saved query built on it.
    """
    extra: dict[str, Any] = {"event": event, "status": status}
    if duration_ms is not None:
        extra["duration_ms"] = round(float(duration_ms), 3)
    if error_code is not None:
        extra["error_code"] = error_code
    extra.update({k: v for k, v in fields.items() if k not in _FORBIDDEN_EXTRA})
    logger.log(level, event, extra=extra)
