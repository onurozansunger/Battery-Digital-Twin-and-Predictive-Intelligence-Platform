"""NASA Ames PCoE lithium-ion battery aging dataset loader.

Raw layout
----------
Each cell is a MATLAB v5 file, ``B0005.mat`` … ``B0056.mat``. The top-level
struct holds a ``cycle`` struct-array; every element is one experimental step:

``type``                 ``'charge'`` | ``'discharge'`` | ``'impedance'``
``ambient_temperature``  chamber set-point in degC
``time``                 ``[yyyy, mm, dd, HH, MM, SS.ffff]`` at step start
``data``                 step-specific measurement traces

Charge / discharge steps carry ``Voltage_measured``, ``Current_measured``,
``Temperature_measured``, ``Time`` (seconds from step start) plus the load or
charger set-points; discharge additionally carries a scalar ``Capacity`` (Ah).
Impedance steps carry an EIS sweep from which we keep the two physically
meaningful scalars: ``Re`` (electrolyte resistance) and ``Rct`` (charge-transfer
resistance).

Normalisation strategy
----------------------
The unit of analysis is the **discharge** step, because that is where capacity is
measured. Each discharge row is enriched with:

* summary statistics of its own voltage / current / temperature traces;
* summary statistics of the **most recent preceding charge** step;
* the **most recent preceding impedance** sweep.

"Most recent preceding" is enforced strictly — a row never sees a measurement
recorded after its own timestamp. That is the first of several leakage guards in
this repository.

Reference
---------
B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics Data
Repository, NASA Ames Research Center, Moffett Field, CA.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from battery_rul.data.base import BatterySource, register_source
from battery_rul.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["NASABatterySource"]

_BATTERY_FILE_RE = re.compile(r"^(B\d{4})\.mat$", re.IGNORECASE)

# The traces contain occasional rig glitches; these bounds mark a sample unusable.
_VOLTAGE_SANE = (0.0, 6.0)
_TEMPERATURE_SANE = (-50.0, 150.0)


def _flatten(value: Any) -> np.ndarray:
    """MATLAB scalars/arrays arrive nested in object arrays; get a 1-D float view."""
    arr = np.asarray(value)
    while arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    if np.iscomplexobj(arr):
        # EIS sweeps are stored as complex impedance; Re and Rct are the real
        # scalars we consume, so take the real part explicitly rather than let
        # NumPy warn about a lossy cast.
        arr = arr.real
    return np.asarray(arr, dtype=float).ravel()


def _scalar(value: Any, default: float = np.nan) -> float:
    arr = _flatten(value)
    return float(arr[0]) if arr.size else default


def _matlab_time(raw: Any) -> pd.Timestamp | None:
    """``[yyyy, mm, dd, HH, MM, SS.ffff]`` -> pandas Timestamp."""
    parts = _flatten(raw)
    if parts.size < 6 or not np.all(np.isfinite(parts[:6])):
        return None
    year, month, day, hour, minute = (int(p) for p in parts[:5])
    seconds = float(parts[5])
    if not (1 <= month <= 12 and 1 <= day <= 31 and 1900 < year < 2100):
        return None
    try:
        base = datetime(year, month, day, hour % 24, minute % 60)
    except ValueError:  # pragma: no cover - defensive
        return None
    return pd.Timestamp(base) + pd.Timedelta(seconds=min(seconds, 59.999))


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope, NaN-safe and degenerate-safe."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    xs, ys = x[mask], y[mask]
    span = xs.max() - xs.min()
    if span <= 0:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    return float(np.trapezoid(y[mask], x[mask]))


@dataclass(slots=True)
class _ChargeSummary:
    """Feature block extracted from one charge step."""

    duration_s: float = np.nan
    cc_duration_s: float = np.nan
    cv_duration_s: float = np.nan
    cc_ct_ratio: float = np.nan
    voltage_mean_v: float = np.nan
    current_mean_a: float = np.nan
    temperature_max_c: float = np.nan
    energy_wh: float = np.nan
    charge_ah: float = np.nan

    def as_row(self) -> dict[str, float]:
        return {
            "charge_duration_s": self.duration_s,
            "charge_cc_duration_s": self.cc_duration_s,
            "charge_cv_duration_s": self.cv_duration_s,
            "cc_ct_ratio": self.cc_ct_ratio,
            "charge_voltage_mean_v": self.voltage_mean_v,
            "charge_current_mean_a": self.current_mean_a,
            "charge_temperature_max_c": self.temperature_max_c,
            "charge_energy_wh": self.energy_wh,
        }


@dataclass(slots=True)
class _ImpedanceSummary:
    internal_resistance_ohm: float = np.nan
    charge_transfer_resistance_ohm: float = np.nan

    def as_row(self) -> dict[str, float]:
        return {
            "internal_resistance_ohm": self.internal_resistance_ohm,
            "charge_transfer_resistance_ohm": self.charge_transfer_resistance_ohm,
        }


@register_source("nasa")
class NASABatterySource(BatterySource):
    """Loader for the NASA PCoE 18650 aging experiments."""

    nominal_capacity_ah = 2.0
    is_synthetic = False

    #: Sub-folders searched for ``*.mat`` files, in priority order.
    _MAT_SUBDIRS = ("mat", "", "raw", "BatteryAgingARC")

    # -- discovery ------------------------------------------------------
    def _mat_dir(self) -> Path | None:
        for sub in self._MAT_SUBDIRS:
            candidate = self.source_dir / sub if sub else self.source_dir
            if candidate.is_dir() and any(candidate.glob("*.mat")):
                return candidate
        # Fall back to a recursive search — users unzip in creative ways.
        if self.source_dir.is_dir():
            for path in sorted(self.source_dir.rglob("*.mat")):
                return path.parent
        return None

    def discover(self) -> list[str]:
        mat_dir = self._mat_dir()
        if mat_dir is None:
            logger.warning("No .mat files under %s", self.source_dir)
            return []
        ids = []
        for path in mat_dir.glob("*.mat"):
            match = _BATTERY_FILE_RE.match(path.name)
            if match:
                ids.append(match.group(1).upper())
        return sorted(set(ids))

    def _source_files(self) -> Iterable[str]:
        mat_dir = self._mat_dir()
        return [] if mat_dir is None else sorted(p.name for p in mat_dir.glob("*.mat"))

    def notes(self) -> str:
        return (
            "NASA Ames PCoE battery aging data (Saha & Goebel, 2007). One row per "
            "discharge cycle; charge and EIS features are causally back-filled from "
            "the most recent preceding step of that type."
        )

    # -- per-battery parsing --------------------------------------------
    def load_battery(self, battery_id: str) -> pd.DataFrame:
        mat_dir = self._mat_dir()
        if mat_dir is None:  # pragma: no cover - guarded by discover()
            raise FileNotFoundError(f"No .mat directory under {self.source_dir}")

        path = mat_dir / f"{battery_id}.mat"
        if not path.is_file():
            matches = [p for p in mat_dir.glob("*.mat") if p.stem.upper() == battery_id.upper()]
            if not matches:
                raise FileNotFoundError(f"{battery_id}.mat not found in {mat_dir}")
            path = matches[0]

        cycles = self._read_cycle_struct(path)
        rows: list[dict[str, Any]] = []
        last_charge = _ChargeSummary()
        last_impedance = _ImpedanceSummary()
        discharge_counter = 0

        for step in cycles:
            step_type = step["type"]
            if step_type == "charge":
                summary = self._summarise_charge(step)
                if summary is not None:
                    last_charge = summary
            elif step_type == "impedance":
                summary = self._summarise_impedance(step)
                if summary is not None:
                    last_impedance = summary
            elif step_type == "discharge":
                row = self._summarise_discharge(step)
                if row is None:
                    continue
                discharge_counter += 1
                row["battery_id"] = battery_id
                row["cycle_index"] = discharge_counter
                row.update(last_charge.as_row())
                row.update(last_impedance.as_row())
                row["coulombic_efficiency"] = (
                    row["capacity_ah"] / last_charge.charge_ah
                    if np.isfinite(last_charge.charge_ah) and last_charge.charge_ah > 1e-6
                    else np.nan
                )
                rows.append(row)

        if not rows:
            logger.warning("%s: no usable discharge cycles", battery_id)
            return pd.DataFrame()

        return pd.DataFrame(rows)

    # -- MATLAB plumbing -------------------------------------------------
    @staticmethod
    def _read_cycle_struct(path: Path) -> list[dict[str, Any]]:
        """Unwrap the nested MATLAB struct into a list of plain dicts."""
        from scipy.io import loadmat  # imported lazily: scipy is only needed here

        mat = loadmat(str(path))
        keys = [k for k in mat if not k.startswith("__")]
        if not keys:
            raise ValueError(f"{path.name} contains no MATLAB variables")

        container = mat[keys[0]]
        try:
            cycle_array = container["cycle"][0, 0]
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(f"{path.name}: unexpected structure, no 'cycle' field") from exc

        n = cycle_array.shape[1]
        steps: list[dict[str, Any]] = []
        for i in range(n):
            raw_type = cycle_array["type"][0, i]
            step_type = str(np.asarray(raw_type).ravel()[0]).strip().lower()
            steps.append(
                {
                    "type": step_type,
                    "ambient_temperature_c": _scalar(cycle_array["ambient_temperature"][0, i]),
                    "timestamp": _matlab_time(cycle_array["time"][0, i]),
                    "data": cycle_array["data"][0, i],
                }
            )
        return steps

    @staticmethod
    def _field(data: Any, name: str) -> np.ndarray | None:
        names = data.dtype.names or ()
        if name not in names:
            return None
        try:
            return _flatten(data[name][0, 0])
        except (IndexError, TypeError, ValueError):  # pragma: no cover - defensive
            return None

    # -- step summarisers -------------------------------------------------
    def _summarise_discharge(self, step: dict[str, Any]) -> dict[str, Any] | None:
        data = step["data"]
        time_s = self._field(data, "Time")
        voltage = self._field(data, "Voltage_measured")
        current = self._field(data, "Current_measured")
        temperature = self._field(data, "Temperature_measured")
        capacity = self._field(data, "Capacity")

        if time_s is None or voltage is None or time_s.size < 5:
            return None

        capacity_ah = float(capacity[0]) if capacity is not None and capacity.size else np.nan
        if not np.isfinite(capacity_ah) or capacity_ah <= 0:
            # A discharge with no capacity reading cannot anchor a health label.
            return None

        n = min(time_s.size, voltage.size)
        time_s, voltage = time_s[:n], voltage[:n]
        current = current[:n] if current is not None and current.size >= n else np.full(n, np.nan)
        temperature = (
            temperature[:n]
            if temperature is not None and temperature.size >= n
            else np.full(n, np.nan)
        )

        voltage = np.where(
            (voltage >= _VOLTAGE_SANE[0]) & (voltage <= _VOLTAGE_SANE[1]), voltage, np.nan
        )
        temperature = np.where(
            (temperature >= _TEMPERATURE_SANE[0]) & (temperature <= _TEMPERATURE_SANE[1]),
            temperature,
            np.nan,
        )

        dt = np.diff(time_s)
        dv = np.diff(voltage)
        with np.errstate(divide="ignore", invalid="ignore"):
            dvdt = np.where(dt > 0, dv / dt, np.nan)

        finite_v = voltage[np.isfinite(voltage)]
        min_voltage = float(finite_v.min()) if finite_v.size else np.nan
        time_to_min = float(time_s[int(np.nanargmin(voltage))]) if finite_v.size else np.nan
        knee_voltage = (
            float(voltage[int(np.nanargmin(dvdt))]) if np.isfinite(dvdt).any() else np.nan
        )

        power_w = np.abs(voltage * current)
        energy_wh = _trapezoid(power_w, time_s) / 3600.0

        return {
            "timestamp": step["timestamp"],
            "ambient_temperature_c": step["ambient_temperature_c"],
            "capacity_ah": capacity_ah,
            "discharge_duration_s": float(time_s[-1] - time_s[0]),
            "voltage_mean_v": float(np.nanmean(voltage)),
            "voltage_min_v": min_voltage,
            "voltage_max_v": float(np.nanmax(voltage)) if finite_v.size else np.nan,
            "voltage_std_v": float(np.nanstd(voltage)),
            "voltage_slope_v_per_s": _ols_slope(time_s, voltage),
            "voltage_knee_v": knee_voltage,
            "voltage_drop_v": (float(finite_v[0] - finite_v[-1]) if finite_v.size >= 2 else np.nan),
            "current_mean_a": float(np.nanmean(current)),
            "current_std_a": float(np.nanstd(current)),
            "temperature_mean_c": float(np.nanmean(temperature)),
            "temperature_max_c": (
                float(np.nanmax(temperature)) if np.isfinite(temperature).any() else np.nan
            ),
            "temperature_rise_c": (
                float(np.nanmax(temperature) - temperature[np.isfinite(temperature)][0])
                if np.isfinite(temperature).any()
                else np.nan
            ),
            "energy_throughput_wh": energy_wh,
            "time_to_min_voltage_s": time_to_min,
            "dvdt_mean_v_per_s": float(np.nanmean(dvdt)) if np.isfinite(dvdt).any() else np.nan,
            "dvdt_min_v_per_s": float(np.nanmin(dvdt)) if np.isfinite(dvdt).any() else np.nan,
            "n_samples_discharge": int(n),
        }

    def _summarise_charge(self, step: dict[str, Any]) -> _ChargeSummary | None:
        data = step["data"]
        time_s = self._field(data, "Time")
        voltage = self._field(data, "Voltage_measured")
        current = self._field(data, "Current_measured")
        temperature = self._field(data, "Temperature_measured")
        set_current = self._field(data, "Current_charge")

        if time_s is None or time_s.size < 5:
            return None

        n = time_s.size
        voltage = voltage[:n] if voltage is not None and voltage.size >= n else np.full(n, np.nan)
        current = current[:n] if current is not None and current.size >= n else np.full(n, np.nan)
        temperature = (
            temperature[:n]
            if temperature is not None and temperature.size >= n
            else np.full(n, np.nan)
        )

        # CC/CV split: the constant-current phase is where the charger is still
        # delivering (near) its full set-point current. The ratio of CC time to
        # total charge time is one of the strongest published degradation
        # indicators — it shrinks monotonically as the cell ages.
        probe = set_current[:n] if set_current is not None and set_current.size >= n else current
        probe = np.abs(probe)
        duration_s = float(time_s[-1] - time_s[0])
        cc_duration = cv_duration = np.nan
        if np.isfinite(probe).any():
            peak = float(np.nanmax(probe))
            if peak > 1e-3:
                in_cc = probe >= 0.90 * peak
                cc_duration = float(np.nansum(np.diff(time_s) * in_cc[:-1]))
                cv_duration = max(duration_s - cc_duration, 0.0)

        charge_ah = _trapezoid(np.abs(current), time_s) / 3600.0
        energy_wh = _trapezoid(np.abs(voltage * current), time_s) / 3600.0

        return _ChargeSummary(
            duration_s=duration_s,
            cc_duration_s=cc_duration,
            cv_duration_s=cv_duration,
            cc_ct_ratio=(cc_duration / duration_s if duration_s > 0 else np.nan),
            voltage_mean_v=float(np.nanmean(voltage)),
            current_mean_a=float(np.nanmean(current)),
            temperature_max_c=(
                float(np.nanmax(temperature)) if np.isfinite(temperature).any() else np.nan
            ),
            energy_wh=energy_wh,
            charge_ah=charge_ah,
        )

    def _summarise_impedance(self, step: dict[str, Any]) -> _ImpedanceSummary | None:
        data = step["data"]
        re_ohm = self._field(data, "Re")
        rct_ohm = self._field(data, "Rct")
        if re_ohm is None and rct_ohm is None:
            return None

        def _clean(arr: np.ndarray | None) -> float:
            if arr is None or arr.size == 0:
                return np.nan
            value = float(arr[0])
            # Occasional sweeps return non-physical (negative or huge) values.
            return value if 0.0 < value < 10.0 else np.nan

        return _ImpedanceSummary(
            internal_resistance_ohm=_clean(re_ohm),
            charge_transfer_resistance_ohm=_clean(rct_ohm),
        )
