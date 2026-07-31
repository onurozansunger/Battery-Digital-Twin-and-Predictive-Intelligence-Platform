"""Lightweight instrumentation used across pipeline stages."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["StageTimer", "timed"]


@dataclass
class StageTimer:
    """Accumulates wall-clock durations so run summaries can report them."""

    durations: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def __call__(self, label: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.durations[label] = self.durations.get(label, 0.0) + elapsed
            logger.info("[timing] %-34s %8.2f s", label, elapsed)

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in self.durations.items()}

    @property
    def total_seconds(self) -> float:
        return round(sum(self.durations.values()), 3)


@contextmanager
def timed(label: str, *, level: int = 20, **extra: Any) -> Iterator[None]:
    """One-off timing context manager."""
    start = time.perf_counter()
    try:
        yield
    finally:
        suffix = f" ({extra})" if extra else ""
        logger.log(level, "%s finished in %.2f s%s", label, time.perf_counter() - start, suffix)
