"""Data-source abstraction and registry.

Adding CALCE, Oxford or Stanford in a later milestone means writing one subclass
of :class:`BatterySource` and decorating it with :func:`register_source`. No other
file in the repository changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from battery_rul.config import DataConfig
from battery_rul.data.schema import DatasetMetadata, coerce_schema
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["BatterySource", "available_sources", "get_source", "register_source"]

_REGISTRY: dict[str, type[BatterySource]] = {}


def register_source(key: str) -> Callable[[type[BatterySource]], type[BatterySource]]:
    """Class decorator that publishes a loader under ``key``."""

    def _decorate(cls: type[BatterySource]) -> type[BatterySource]:
        normalised = key.strip().lower()
        if normalised in _REGISTRY:
            raise ValueError(f"Data source {normalised!r} is already registered")
        _REGISTRY[normalised] = cls
        cls.key = normalised
        return cls

    return _decorate


def available_sources() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_source(cfg: DataConfig, root: Path) -> BatterySource:
    """Instantiate the loader named by ``cfg.source``."""
    try:
        cls = _REGISTRY[cfg.source]
    except KeyError as exc:
        raise KeyError(
            f"Unknown data source {cfg.source!r}. Registered: {available_sources()}"
        ) from exc
    return cls(cfg=cfg, root=root)


class BatterySource(ABC):
    """Base class for all dataset loaders.

    A subclass implements two things:

    * :meth:`discover` — which battery ids exist on disk;
    * :meth:`load_battery` — one cell's cycles as canonical-schema rows.

    :meth:`load` handles filtering, concatenation, ordering, schema coercion and
    metadata, identically for every dataset.
    """

    key: str = "base"
    nominal_capacity_ah: float = 2.0

    def __init__(self, cfg: DataConfig, root: Path) -> None:
        self.cfg = cfg
        self.root = Path(root)
        self.source_dir = self.root / cfg.subdir

    # -- to implement --------------------------------------------------
    @abstractmethod
    def discover(self) -> list[str]:
        """Battery ids visible to this loader, sorted deterministically."""

    @abstractmethod
    def load_battery(self, battery_id: str) -> pd.DataFrame:
        """Return one cell's cycles. Must include every required schema column."""

    # -- shared --------------------------------------------------------
    def selected_batteries(self) -> list[str]:
        """Apply the whitelist/blacklist from config to the discovered ids."""
        found = self.discover()
        if not found:
            return []

        if self.cfg.batteries:
            requested = list(dict.fromkeys(self.cfg.batteries))
            unknown = [b for b in requested if b not in found]
            if unknown:
                logger.warning(
                    "Configured batteries not present in %s: %s (available: %s)",
                    self.source_dir,
                    unknown,
                    found,
                )
            selected = [b for b in requested if b in found]
        else:
            selected = list(found)

        excluded = set(self.cfg.exclude_batteries)
        return [b for b in selected if b not in excluded]

    def load(self) -> tuple[pd.DataFrame, DatasetMetadata]:
        """Load every selected battery into one canonical table."""
        batteries = self.selected_batteries()
        if not batteries:
            raise FileNotFoundError(
                f"No batteries found for source {self.key!r} under {self.source_dir}. "
                f"Check `data.subdir` in your config, or run scripts/download_data.py."
            )

        frames: list[pd.DataFrame] = []
        skipped: list[tuple[str, str]] = []
        for battery_id in batteries:
            try:
                frame = self.load_battery(battery_id)
            except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the run
                logger.exception("Failed to load battery %s: %s", battery_id, exc)
                skipped.append((battery_id, f"load error: {exc}"))
                continue

            if frame.empty:
                skipped.append((battery_id, "no usable cycles"))
                continue
            if len(frame) < self.cfg.min_cycles:
                skipped.append((battery_id, f"{len(frame)} cycles < min_cycles"))
                continue
            frames.append(frame)

        if skipped:
            for battery_id, reason in skipped:
                logger.info("Skipped battery %s (%s)", battery_id, reason)

        if not frames:
            raise ValueError(
                f"Every battery for source {self.key!r} was rejected. "
                f"Reasons: {skipped}. Consider lowering data.min_cycles."
            )

        df = pd.concat(frames, ignore_index=True)
        df["dataset"] = self.key
        df = coerce_schema(df)
        df = df.sort_values(["battery_id", "cycle_index"], kind="stable").reset_index(drop=True)

        meta = DatasetMetadata(
            dataset=self.key,
            n_batteries=int(df["battery_id"].nunique()),
            n_cycles=int(len(df)),
            batteries=tuple(sorted(df["battery_id"].unique().tolist())),
            nominal_capacity_ah=self.cfg.nominal_capacity_ah,
            eol_threshold=self.cfg.eol_threshold,
            source_files=tuple(self._source_files()),
            synthetic=getattr(self, "is_synthetic", False),
            notes=self.notes(),
        )
        logger.info(
            "Loaded %s: %d batteries, %d discharge cycles (%s)",
            self.key,
            meta.n_batteries,
            meta.n_cycles,
            ", ".join(meta.batteries[:10]) + ("…" if meta.n_batteries > 10 else ""),
        )
        return df, meta

    def _source_files(self) -> Iterable[str]:
        return ()

    def notes(self) -> str:
        return ""
