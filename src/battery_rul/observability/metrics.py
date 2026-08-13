"""A minimal in-process metrics registry.

Counters, gauges and histograms with string labels, plus a Prometheus text
rendering. Deliberately not ``prometheus_client``: the only thing this project
needs from it is the exposition format, which is a dozen lines of string
formatting, and the library brings a global default registry that fights with
test isolation.

Thread-safe by a single lock. The volumes here — a few hundred observations per
batch — make contention irrelevant, and a correct simple implementation beats a
lock-free one nobody will audit.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "METRICS",
    "MetricSample",
    "MetricsRegistry",
    "render_prometheus",
    "timed",
]

#: Histogram buckets in seconds, spanning "fast local call" to "slow batch".
DEFAULT_BUCKETS: tuple[float, ...] = (0.005, 0.025, 0.1, 0.25, 1.0, 2.5, 10.0, 30.0, 120.0)


def _key(labels: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))


@dataclass(frozen=True, slots=True)
class MetricSample:
    """One exported time series."""

    name: str
    labels: tuple[tuple[str, str], ...]
    value: float

    def label_dict(self) -> dict[str, str]:
        return dict(self.labels)


@dataclass
class _Histogram:
    buckets: tuple[float, ...]
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * (len(self.buckets) + 1)

    def observe(self, value: float) -> None:
        self.total += value
        self.count += 1
        for index, edge in enumerate(self.buckets):
            if value <= edge:
                self.counts[index] += 1
                return
        self.counts[-1] += 1


class MetricsRegistry:
    """Counters, gauges and histograms, with help text and units."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
        self._gauges: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
        self._histograms: dict[str, dict[tuple[tuple[str, str], ...], _Histogram]] = {}
        self._help: dict[str, str] = {}
        self.enabled = True

    # -- recording ---------------------------------------------------------
    def increment(
        self,
        name: str,
        value: float = 1.0,
        *,
        labels: Mapping[str, str] | None = None,
        help_text: str = "",
    ) -> None:
        if not self.enabled:
            return
        if value < 0:
            raise ValueError(f"Counter {name} cannot decrease (got {value})")
        with self._lock:
            self._help.setdefault(name, help_text)
            series = self._counters.setdefault(name, {})
            series[_key(labels)] = series.get(_key(labels), 0.0) + float(value)

    def set_gauge(
        self,
        name: str,
        value: float,
        *,
        labels: Mapping[str, str] | None = None,
        help_text: str = "",
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._help.setdefault(name, help_text)
            self._gauges.setdefault(name, {})[_key(labels)] = float(value)

    def observe(
        self,
        name: str,
        value: float,
        *,
        labels: Mapping[str, str] | None = None,
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
        help_text: str = "",
    ) -> None:
        if not self.enabled:
            return
        if not math.isfinite(value):
            return
        with self._lock:
            self._help.setdefault(name, help_text)
            series = self._histograms.setdefault(name, {})
            histogram = series.get(_key(labels))
            if histogram is None:
                histogram = _Histogram(buckets=buckets)
                series[_key(labels)] = histogram
            histogram.observe(float(value))

    # -- reading -----------------------------------------------------------
    def counter_value(self, name: str, labels: Mapping[str, str] | None = None) -> float:
        with self._lock:
            return self._counters.get(name, {}).get(_key(labels), 0.0)

    def gauge_value(self, name: str, labels: Mapping[str, str] | None = None) -> float | None:
        with self._lock:
            return self._gauges.get(name, {}).get(_key(labels))

    def histogram_stats(
        self, name: str, labels: Mapping[str, str] | None = None
    ) -> dict[str, float] | None:
        with self._lock:
            histogram = self._histograms.get(name, {}).get(_key(labels))
        if histogram is None:
            return None
        return {
            "count": float(histogram.count),
            "sum": histogram.total,
            "mean": histogram.total / histogram.count if histogram.count else float("nan"),
        }

    def snapshot(self) -> dict[str, Any]:
        """A JSON-serialisable view, for the ``/metrics`` JSON variant and tests."""
        with self._lock:
            return {
                "counters": {
                    name: [
                        {"labels": dict(labels), "value": value}
                        for labels, value in sorted(series.items())
                    ]
                    for name, series in sorted(self._counters.items())
                },
                "gauges": {
                    name: [
                        {"labels": dict(labels), "value": value}
                        for labels, value in sorted(series.items())
                    ]
                    for name, series in sorted(self._gauges.items())
                },
                "histograms": {
                    name: [
                        {
                            "labels": dict(labels),
                            "count": histogram.count,
                            "sum": round(histogram.total, 6),
                        }
                        for labels, histogram in sorted(series.items())
                    ]
                    for name, series in sorted(self._histograms.items())
                },
            }

    def reset(self) -> None:
        """Drop every series. Tests use this; the running service never does."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._help.clear()

    # -- exposition --------------------------------------------------------
    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            counters = {k: dict(v) for k, v in self._counters.items()}
            gauges = {k: dict(v) for k, v in self._gauges.items()}
            histograms = {k: dict(v) for k, v in self._histograms.items()}
            helps = dict(self._help)

        for name, counter_series in sorted(counters.items()):
            lines.append(f"# HELP {name} {helps.get(name, '')}".rstrip())
            lines.append(f"# TYPE {name} counter")
            for labels, value in sorted(counter_series.items()):
                lines.append(f"{name}{_render_labels(labels)} {value}")
        for name, gauge_series in sorted(gauges.items()):
            lines.append(f"# HELP {name} {helps.get(name, '')}".rstrip())
            lines.append(f"# TYPE {name} gauge")
            for labels, value in sorted(gauge_series.items()):
                lines.append(f"{name}{_render_labels(labels)} {value}")
        for name, histogram_series in sorted(histograms.items()):
            lines.append(f"# HELP {name} {helps.get(name, '')}".rstrip())
            lines.append(f"# TYPE {name} histogram")
            for labels, histogram in sorted(histogram_series.items()):
                cumulative = 0
                for index, edge in enumerate(histogram.buckets):
                    cumulative += histogram.counts[index]
                    lines.append(
                        f"{name}_bucket{_render_labels(labels, le=str(edge))} {cumulative}"
                    )
                cumulative += histogram.counts[-1]
                lines.append(f"{name}_bucket{_render_labels(labels, le='+Inf')} {cumulative}")
                lines.append(f"{name}_sum{_render_labels(labels)} {histogram.total}")
                lines.append(f"{name}_count{_render_labels(labels)} {histogram.count}")
        return "\n".join(lines) + ("\n" if lines else "")


def _render_labels(labels: tuple[tuple[str, str], ...], *, le: str | None = None) -> str:
    items = [(k, v) for k, v in labels]
    if le is not None:
        items.append(("le", le))
    if not items:
        return ""
    body = ",".join(f'{k}="{_escape(v)}"' for k, v in items)
    return "{" + body + "}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


#: The process-wide registry. Injectable everywhere it matters; this is the
#: default so a call site does not have to plumb one through.
METRICS = MetricsRegistry()


def render_prometheus(registry: MetricsRegistry | None = None) -> str:
    return (registry or METRICS).render_prometheus()


@contextmanager
def timed(
    name: str,
    *,
    labels: Mapping[str, str] | None = None,
    registry: MetricsRegistry | None = None,
) -> Iterator[dict[str, float]]:
    """Time a block into a histogram (seconds), yielding the elapsed dict.

    The duration is recorded whether the block succeeds or raises — a metric
    that only counts successes hides exactly the latency that matters.
    """
    target = registry or METRICS
    started = time.perf_counter()
    elapsed: dict[str, float] = {}
    try:
        yield elapsed
    finally:
        elapsed["seconds"] = time.perf_counter() - started
        elapsed["ms"] = 1000.0 * elapsed["seconds"]
        target.observe(name, elapsed["seconds"], labels=labels)
