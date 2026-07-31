"""Physics-informed synthetic battery generator.

Two jobs:

1. **Test fixture.** The unit-test suite must run in seconds on a machine with no
   200 MB dataset checked out. Every test that needs realistic cycle data pulls
   from here.
2. **Fallback source.** If ``data.allow_synthetic_fallback`` is on and the raw
   NASA files are missing, the pipeline still runs end to end — loudly labelled
   as synthetic in the metadata so nobody mistakes the numbers for real results.

Degradation model
-----------------
Capacity follows the widely used double-exponential fade law (Saha & Goebel's
empirical form), plus per-cell variation and measurement noise:

    Q(k) = a * exp(b * k) + c * exp(d * k)

Internal resistance grows roughly as the inverse of capacity; charge CC-time
shrinks with capacity; discharge duration is proportional to capacity; peak
temperature rises as resistance grows. These couplings are what make the
synthetic data useful for exercising feature code — the correlations the feature
pipeline is supposed to find genuinely exist.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

from battery_rul.data.base import BatterySource, register_source
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["SyntheticBatterySource", "make_synthetic_cycles"]


def make_synthetic_cycles(
    battery_id: str,
    *,
    n_cycles: int = 160,
    nominal_capacity_ah: float = 2.0,
    eol_threshold: float = 0.70,
    seed: int = 0,
    ambient_c: float = 24.0,
    noise_scale: float = 1.0,
) -> pd.DataFrame:
    """One synthetic cell's worth of canonical-schema cycle rows."""
    rng = np.random.default_rng(seed)
    k = np.arange(1, n_cycles + 1, dtype=float)

    # Double-exponential fade with per-cell parameter jitter.
    a = nominal_capacity_ah * rng.uniform(0.93, 0.99)
    c = nominal_capacity_ah * rng.uniform(0.02, 0.07)
    d = -rng.uniform(0.006, 0.020)

    # The slow-fade rate `b` is solved for rather than sampled, so that every
    # generated cell actually crosses end of life somewhere inside its record.
    # A surrogate whose cells never reach EOL would produce a fully right-censored
    # dataset — useless as a fixture, since the target could not be built at all.
    eol_capacity = eol_threshold * nominal_capacity_ah
    eol_cycle = max(n_cycles * rng.uniform(0.65, 0.85), 10.0)
    residual = max(eol_capacity - c * np.exp(d * eol_cycle), 0.05 * nominal_capacity_ah)
    b = float(np.log(residual / a) / eol_cycle)

    capacity = a * np.exp(b * k) + c * np.exp(d * k)

    # Rest-period recovery: capacity ticks back up after long pauses. This is the
    # real, physical reason a raw NASA capacity series is non-monotonic.
    recovery = np.zeros_like(k)
    for start in rng.choice(np.arange(10, n_cycles), size=max(n_cycles // 40, 1), replace=False):
        recovery[int(start) :] += rng.uniform(0.005, 0.030) * np.exp(
            -np.arange(n_cycles - int(start)) / 12.0
        )
    capacity = capacity + recovery
    capacity += rng.normal(0.0, 0.010 * noise_scale, size=n_cycles)
    capacity = np.clip(capacity, 0.05, None)

    soh = capacity / nominal_capacity_ah
    resistance = 0.055 / np.clip(soh, 0.3, None) + rng.normal(0.0, 0.0015 * noise_scale, n_cycles)

    discharge_duration = 3600.0 * capacity / 2.0 * rng.uniform(0.97, 1.03, n_cycles)
    charge_duration = 9000.0 + 2500.0 * (1.0 - soh) + rng.normal(0, 90 * noise_scale, n_cycles)
    cc_ratio = np.clip(0.62 * soh + 0.05 + rng.normal(0, 0.012 * noise_scale, n_cycles), 0.05, 0.98)
    cc_duration = cc_ratio * charge_duration

    voltage_mean = 3.55 + 0.30 * soh + rng.normal(0, 0.010 * noise_scale, n_cycles)
    voltage_min = 2.55 + 0.10 * soh + rng.normal(0, 0.012 * noise_scale, n_cycles)
    voltage_slope = -(0.00042 + 0.00030 * (1.0 - soh)) * rng.uniform(0.95, 1.05, n_cycles)
    temperature_mean = (
        ambient_c + 8.0 + 12.0 * (1.0 - soh) + rng.normal(0, 0.4 * noise_scale, n_cycles)
    )

    return pd.DataFrame(
        {
            "battery_id": battery_id,
            "cycle_index": k.astype(int),
            "timestamp": pd.Timestamp("2008-04-02") + pd.to_timedelta(k * 3.7, unit="h"),
            "ambient_temperature_c": ambient_c,
            "capacity_ah": capacity,
            "discharge_duration_s": discharge_duration,
            "voltage_mean_v": voltage_mean,
            "voltage_min_v": voltage_min,
            "voltage_max_v": voltage_mean + 0.55,
            "voltage_std_v": 0.18 + 0.05 * (1.0 - soh),
            "voltage_slope_v_per_s": voltage_slope,
            "voltage_knee_v": 3.05 + 0.15 * soh,
            "voltage_drop_v": 1.35 + 0.20 * (1.0 - soh),
            "current_mean_a": -2.0 + rng.normal(0, 0.01 * noise_scale, n_cycles),
            "current_std_a": 0.35 + rng.normal(0, 0.005 * noise_scale, n_cycles),
            "temperature_mean_c": temperature_mean,
            "temperature_max_c": temperature_mean + 4.5 + 3.0 * (1.0 - soh),
            "temperature_rise_c": 6.0 + 9.0 * (1.0 - soh),
            "energy_throughput_wh": capacity * 3.4 * rng.uniform(0.98, 1.02, n_cycles),
            "time_to_min_voltage_s": discharge_duration * rng.uniform(0.95, 1.0, n_cycles),
            "dvdt_mean_v_per_s": voltage_slope * 1.1,
            "dvdt_min_v_per_s": voltage_slope * 6.0,
            "n_samples_discharge": rng.integers(150, 220, n_cycles),
            "charge_duration_s": charge_duration,
            "charge_cc_duration_s": cc_duration,
            "charge_cv_duration_s": charge_duration - cc_duration,
            "cc_ct_ratio": cc_ratio,
            "charge_voltage_mean_v": 3.95 + 0.05 * soh,
            "charge_current_mean_a": 1.5 * cc_ratio + 0.2,
            "charge_temperature_max_c": temperature_mean + 2.0,
            "charge_energy_wh": capacity * 3.6 * 1.05,
            "coulombic_efficiency": np.clip(
                0.985 + rng.normal(0, 0.004 * noise_scale, n_cycles), 0.8, 1.05
            ),
            "internal_resistance_ohm": resistance,
            "charge_transfer_resistance_ohm": resistance * rng.uniform(1.2, 1.6, n_cycles),
        }
    )


@register_source("synthetic")
class SyntheticBatterySource(BatterySource):
    """Generates a small fleet of surrogate cells. Never touches the filesystem."""

    nominal_capacity_ah = 2.0
    is_synthetic = True

    #: Cell-count / length / ambient recipe for the default fleet.
    DEFAULT_FLEET: tuple[tuple[str, int, float], ...] = (
        ("S0001", 168, 24.0),
        ("S0002", 168, 24.0),
        ("S0003", 132, 24.0),
        ("S0004", 197, 24.0),
        ("S0005", 150, 43.0),
        ("S0006", 185, 24.0),
        ("S0007", 120, 4.0),
        ("S0008", 205, 24.0),
    )

    def discover(self) -> list[str]:
        return [name for name, _, _ in self.DEFAULT_FLEET]

    def load_battery(self, battery_id: str) -> pd.DataFrame:
        recipe = {name: (n, amb) for name, n, amb in self.DEFAULT_FLEET}
        if battery_id not in recipe:
            raise KeyError(f"Unknown synthetic battery {battery_id!r}")
        n_cycles, ambient = recipe[battery_id]
        # A stable hash, not the builtin: PYTHONHASHSEED randomisation would give
        # a different cell every process, quietly destroying reproducibility.
        seed = zlib.crc32(f"{battery_id}:{self.cfg.nominal_capacity_ah}".encode()) % (2**31)
        return make_synthetic_cycles(
            battery_id,
            n_cycles=n_cycles,
            nominal_capacity_ah=self.cfg.nominal_capacity_ah,
            eol_threshold=self.cfg.eol_threshold,
            seed=seed,
            ambient_c=ambient,
        )

    def _source_files(self) -> Iterable[str]:
        return ()

    def notes(self) -> str:
        return (
            "SYNTHETIC DATA — generated from a double-exponential capacity-fade "
            "model. Useful for exercising the pipeline; NOT a source of publishable "
            "accuracy numbers."
        )
