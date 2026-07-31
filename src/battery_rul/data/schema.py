"""The canonical cycle-level schema.

Every data source — NASA today, CALCE/Oxford/Stanford tomorrow — is normalised
into a single long-form table with **one row per discharge cycle per battery**.
Downstream code (features, target, splitting, models) is written against this
schema and nothing else, which is what makes adding a dataset a matter of writing
one loader rather than touching the pipeline.

Column contract
---------------
Identity / index
    ``battery_id``        str  — unique cell identifier ("B0005")
    ``dataset``           str  — source key ("nasa")
    ``cycle_index``       int  — 1-based, gap-free, ordered discharge counter
    ``timestamp``         datetime64[ns] — test-rig clock at cycle start

Health
    ``capacity_ah``       float — measured discharge capacity
    ``capacity_smooth_ah``float — median-filtered capacity (noise is ~2 % on NASA)
    ``soh``               float — capacity_smooth_ah / nominal_capacity_ah

Discharge-segment summaries
    ``discharge_duration_s``, ``voltage_mean_v``, ``voltage_min_v``,
    ``voltage_max_v``, ``voltage_std_v``, ``voltage_slope_v_per_s``,
    ``voltage_knee_v``, ``current_mean_a``, ``current_std_a``,
    ``temperature_mean_c``, ``temperature_max_c``, ``temperature_rise_c``,
    ``energy_throughput_wh``, ``cv_time_s`` …

Charge-segment summaries (from the charge step immediately preceding)
    ``charge_duration_s``, ``charge_cc_duration_s``, ``charge_cv_duration_s``,
    ``cc_ct_ratio``, ``charge_voltage_mean_v``, ``charge_temperature_max_c`` …

Electrochemical impedance (last measurement at or before this cycle — causal)
    ``internal_resistance_ohm`` (Re), ``charge_transfer_resistance_ohm`` (Rct)

Environment
    ``ambient_temperature_c`` float
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

__all__ = [
    "CHARGE_COLUMNS",
    "CYCLE_SCHEMA",
    "DISCHARGE_COLUMNS",
    "IDENTITY_COLUMNS",
    "IMPEDANCE_COLUMNS",
    "REQUIRED_COLUMNS",
    "ColumnSpec",
    "coerce_schema",
    "schema_frame",
]


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """One column of the canonical table."""

    name: str
    dtype: str
    unit: str
    description: str
    required: bool = True
    minimum: float | None = None
    maximum: float | None = None


IDENTITY_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("dataset", "string", "-", "Source dataset key."),
    ColumnSpec("battery_id", "string", "-", "Unique cell identifier within the dataset."),
    ColumnSpec("cycle_index", "int32", "count", "1-based ordered discharge counter.", minimum=1),
    ColumnSpec(
        "timestamp", "datetime64[ns]", "-", "Test-rig clock at cycle start.", required=False
    ),
    ColumnSpec(
        "ambient_temperature_c", "float32", "degC", "Chamber set-point.", minimum=-40, maximum=100
    ),
)

HEALTH_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("capacity_ah", "float32", "Ah", "Measured discharge capacity.", minimum=0.0),
    ColumnSpec(
        "capacity_smooth_ah",
        "float32",
        "Ah",
        "Median-filtered capacity.",
        required=False,
        minimum=0,
    ),
    ColumnSpec("soh", "float32", "-", "State of health vs nominal capacity.", required=False),
)

DISCHARGE_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("discharge_duration_s", "float32", "s", "Length of the discharge step.", minimum=0),
    ColumnSpec("voltage_mean_v", "float32", "V", "Mean terminal voltage under load.", minimum=0),
    ColumnSpec("voltage_min_v", "float32", "V", "Minimum (cut-off) terminal voltage.", minimum=0),
    ColumnSpec("voltage_max_v", "float32", "V", "Maximum terminal voltage.", minimum=0),
    ColumnSpec("voltage_std_v", "float32", "V", "Dispersion of the voltage trace.", minimum=0),
    ColumnSpec("voltage_slope_v_per_s", "float32", "V/s", "OLS slope of the discharge curve."),
    ColumnSpec(
        "voltage_knee_v", "float32", "V", "Voltage at the steepest dV/dt point (knee).", minimum=0
    ),
    ColumnSpec("voltage_drop_v", "float32", "V", "First-to-last voltage delta."),
    ColumnSpec("current_mean_a", "float32", "A", "Mean load current (negative = discharge)."),
    ColumnSpec("current_std_a", "float32", "A", "Load-current dispersion.", minimum=0),
    ColumnSpec("temperature_mean_c", "float32", "degC", "Mean cell-surface temperature."),
    ColumnSpec("temperature_max_c", "float32", "degC", "Peak cell-surface temperature."),
    ColumnSpec("temperature_rise_c", "float32", "degC", "Peak minus initial temperature."),
    ColumnSpec("energy_throughput_wh", "float32", "Wh", "Integral of |V*I| dt over discharge."),
    ColumnSpec(
        "time_to_min_voltage_s", "float32", "s", "Seconds until the cut-off voltage.", minimum=0
    ),
    ColumnSpec("dvdt_mean_v_per_s", "float32", "V/s", "Mean first difference of voltage."),
    ColumnSpec("dvdt_min_v_per_s", "float32", "V/s", "Sharpest voltage collapse rate."),
    ColumnSpec("n_samples_discharge", "int32", "count", "Sample count in the discharge trace."),
)

CHARGE_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        "charge_duration_s", "float32", "s", "Length of the preceding charge step.", required=False
    ),
    ColumnSpec(
        "charge_cc_duration_s", "float32", "s", "Constant-current phase length.", required=False
    ),
    ColumnSpec(
        "charge_cv_duration_s", "float32", "s", "Constant-voltage phase length.", required=False
    ),
    ColumnSpec("cc_ct_ratio", "float32", "-", "CC time / total charge time.", required=False),
    ColumnSpec("charge_voltage_mean_v", "float32", "V", "Mean charge voltage.", required=False),
    ColumnSpec("charge_current_mean_a", "float32", "A", "Mean charge current.", required=False),
    ColumnSpec(
        "charge_temperature_max_c", "float32", "degC", "Peak temperature on charge.", required=False
    ),
    ColumnSpec(
        "charge_energy_wh", "float32", "Wh", "Energy pushed in during charge.", required=False
    ),
    ColumnSpec("coulombic_efficiency", "float32", "-", "Discharge Ah / charge Ah.", required=False),
)

IMPEDANCE_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        "internal_resistance_ohm",
        "float32",
        "ohm",
        "Electrolyte resistance Re, forward-filled from the last EIS sweep at or "
        "before this cycle (never from the future).",
        required=False,
    ),
    ColumnSpec(
        "charge_transfer_resistance_ohm",
        "float32",
        "ohm",
        "Charge-transfer resistance Rct, causally forward-filled.",
        required=False,
    ),
)

CYCLE_SCHEMA: tuple[ColumnSpec, ...] = (
    IDENTITY_COLUMNS + HEALTH_COLUMNS + DISCHARGE_COLUMNS + CHARGE_COLUMNS + IMPEDANCE_COLUMNS
)

REQUIRED_COLUMNS: tuple[str, ...] = tuple(c.name for c in CYCLE_SCHEMA if c.required)

_SPEC_BY_NAME: dict[str, ColumnSpec] = {c.name: c for c in CYCLE_SCHEMA}


def spec(name: str) -> ColumnSpec:
    """Look up a column contract by name."""
    try:
        return _SPEC_BY_NAME[name]
    except KeyError as exc:  # pragma: no cover - programmer error
        raise KeyError(f"{name!r} is not part of the canonical cycle schema") from exc


def schema_frame() -> pd.DataFrame:
    """The schema as a DataFrame — rendered directly into the dataset card."""
    return pd.DataFrame(
        [
            {
                "column": c.name,
                "dtype": c.dtype,
                "unit": c.unit,
                "required": c.required,
                "description": c.description,
            }
            for c in CYCLE_SCHEMA
        ]
    )


def coerce_schema(df: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """Cast a loader's output onto the canonical dtypes and column order.

    Missing optional columns are created as all-NaN so downstream code can rely
    on their presence. Extra columns are preserved (appended after the canonical
    block) unless ``strict``.

    Raises
    ------
    ValueError
        If a *required* column is absent.
    """
    df = df.copy()

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Loader output is missing required schema columns: {missing_required}. "
            f"Present: {sorted(df.columns)}"
        )

    for col in CYCLE_SCHEMA:
        if col.name not in df.columns:
            df[col.name] = pd.NA

    for col in CYCLE_SCHEMA:
        series = df[col.name]
        try:
            if col.dtype.startswith("datetime"):
                df[col.name] = pd.to_datetime(series, errors="coerce")
            elif col.dtype == "string":
                df[col.name] = series.astype("string")
            elif col.dtype.startswith("int"):
                df[col.name] = (
                    pd.to_numeric(series, errors="coerce")
                    .astype("Int32")
                    .astype(col.dtype, errors="ignore")
                )
            else:
                df[col.name] = pd.to_numeric(series, errors="coerce").astype(col.dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column {col.name!r} cannot be cast to {col.dtype}: {exc}") from exc

    canonical = [c.name for c in CYCLE_SCHEMA]
    extras = [] if strict else [c for c in df.columns if c not in canonical]
    return df[canonical + extras]


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Provenance carried alongside every loaded table."""

    dataset: str
    n_batteries: int
    n_cycles: int
    batteries: tuple[str, ...]
    nominal_capacity_ah: float
    eol_threshold: float
    source_files: tuple[str, ...] = field(default_factory=tuple)
    synthetic: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "n_batteries": self.n_batteries,
            "n_cycles": self.n_cycles,
            "batteries": list(self.batteries),
            "nominal_capacity_ah": self.nominal_capacity_ah,
            "eol_threshold": self.eol_threshold,
            "source_files": list(self.source_files),
            "synthetic": self.synthetic,
            "notes": self.notes,
        }
