"""Structured logging and in-process metrics.

Two separate concerns kept in two modules:

:mod:`battery_rul.observability.logging`
    A JSON formatter and a context binder, so every log line from a request or a
    batch carries the same ``request_id`` / ``batch_id`` / ``fleet_id`` /
    ``model_version`` fields without every call site remembering to pass them.

:mod:`battery_rul.observability.metrics`
    A tiny counter/gauge/histogram registry with a Prometheus text rendering. No
    client library is pulled in: the exposition format is a dozen lines, and a
    dependency whose only job is to format text is a dependency to maintain.
"""

from __future__ import annotations

from battery_rul.observability.logging import (
    LogContext,
    bind_context,
    configure_structured_logging,
    current_context,
    log_event,
)
from battery_rul.observability.metrics import (
    METRICS,
    MetricsRegistry,
    render_prometheus,
    timed,
)

__all__ = [
    "METRICS",
    "LogContext",
    "MetricsRegistry",
    "bind_context",
    "configure_structured_logging",
    "current_context",
    "log_event",
    "render_prometheus",
    "timed",
]
