# Dataset Card — NASA Ames PCoE Li-ion Battery Aging Data

## Overview

| | |
|---|---|
| **Name** | Battery Data Set (NASA Ames Prognostics Data Repository) |
| **Authors** | B. Saha and K. Goebel, NASA Ames Research Center (2007) |
| **Cells** | 34 commercial 18650 lithium-cobalt-oxide cells |
| **Rated capacity** | 2.0 Ah |
| **Voltage window** | ~2.0 V (cut-off) to 4.2 V (fully charged) |
| **Raw format** | MATLAB v5 (`.mat`), one file per cell |
| **Size** | ~209 MB compressed, ~190 MB extracted |
| **Licence** | US Government work; released by NASA for open research use |
| **Retrieved from** | `https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip` |

## Experimental protocol

Cells were run through repeated charge/discharge cycles in a temperature chamber
until they reached an end-of-life criterion:

* **Charge** — constant current at 1.5 A to 4.2 V, then constant voltage until
  the current fell below 20 mA.
* **Discharge** — constant current (typically 2 A, some cells 1 A or 4 A) to a
  per-cell cut-off voltage between 2.0 V and 2.7 V.
* **Impedance** — electrochemical impedance spectroscopy sweeps (0.1 Hz – 5 kHz)
  interleaved between cycles, yielding electrolyte resistance `Re` and
  charge-transfer resistance `Rct`.

Ambient temperature was 4 °C, 24 °C, 43 °C or 44 °C depending on the cell, and
some cells were moved between chambers mid-experiment.

## Raw structure

Each `.mat` holds a `cycle` struct-array; every element is one experimental step:

| Field | Meaning |
|---|---|
| `type` | `'charge'` / `'discharge'` / `'impedance'` |
| `ambient_temperature` | Chamber set-point, °C |
| `time` | `[yyyy, mm, dd, HH, MM, SS.ffff]` at step start |
| `data` | Measurement traces for that step |

Discharge steps additionally carry a scalar `Capacity` (Ah) — the measured
discharge capacity, and the anchor for every health label in this project.

## Normalised schema

This repository converts the raw files into **one row per discharge cycle per
cell**. See `src/battery_rul/data/schema.py` for the authoritative contract; the
column groups are:

| Group | Examples |
|---|---|
| Identity | `dataset`, `battery_id`, `cycle_index`, `timestamp`, `ambient_temperature_c` |
| Health | `capacity_ah`, `capacity_smooth_ah`, `soh`, `reference_capacity_ah` |
| Discharge | `discharge_duration_s`, `voltage_min_v`, `voltage_slope_v_per_s`, `voltage_knee_v`, `temperature_max_c`, `energy_throughput_wh`, `dvdt_min_v_per_s` |
| Charge | `charge_duration_s`, `charge_cc_duration_s`, `cc_ct_ratio`, `charge_energy_wh`, `coulombic_efficiency` |
| Impedance | `internal_resistance_ohm` (Re), `charge_transfer_resistance_ohm` (Rct) |

Charge and impedance features are attached from the **most recent preceding**
step of that type. Nothing is ever back-filled from a future step.

## Cohort selection

Of the 34 cells, the default configuration admits **5**. The exclusions are
mechanical and logged, not hand-picked — the config asks for every cell on disk
and the gates below select the cohort:

| Gate | Config key | Cells removed |
|---|---|---|
| Too few cycles | `data.min_cycles: 30` | B0025–B0028, B0049–B0052 |
| Begins already degraded (< 80 % SoH at cycle 1) | `data.min_start_soh` | B0038–B0041, B0045, B0053–B0056 |
| Never degrades measurably | `data.min_fade_fraction` | B0031 |
| Never reaches end of life (right-censored) | `target.require_eol_reached` | B0007, B0029, B0030, B0032, B0036 |
| Fewer than 25 labelled cycles before EOL | `target.min_labelled_cycles` | B0046, B0047, B0048 |
| Sustained capacity collapse (regime change) | `data.truncate_at_collapse` | B0042, B0043, B0044 truncated to 40 cycles, then right-censored; B0033 truncated to 139 |

**Retained cohort:** B0005, B0006, B0018, B0033, B0034 — all cycled at a constant
24 °C, 520 labelled rows.

### Why cells are excluded, in plain terms

* **The low-SoH group** (B0038–B0041, B0045, B0053–B0056) were run at 4 °C or
  moved between chambers. Their *delivered* capacity is low because the cell is
  cold, not because it is worn out. Against a 2.0 Ah rated reference they appear
  to start life below the end-of-life threshold, which makes "remaining useful
  life" undefined for them.
* **B0046–B0048** cross the threshold within 12–19 cycles for the same reason.
  They contribute a handful of rows describing a cold cell rather than an aged
  one.
* **B0042–B0044** are the subtlest case, and the one that produced a real bug.
  They were moved from a 22 °C to a 4 °C chamber at cycle 41. Measured capacity
  drops from ~1.5 Ah to ~0.07 Ah in a single step and stays there for the
  remaining 67 cycles — at 4 °C the discharge test terminates almost immediately,
  so what is recorded is not a capacity measurement of a working cell at all.
  Left in, the EOL detector reads the collapse as a persistent threshold crossing
  and labels end-of-life at cycle 44, which is wrong by roughly the whole
  remaining life of the cell.

  The single-step jump check (`validation.max_capacity_jump`) does **not** catch
  this: it inspects first differences, so it flags the one transition cycle and
  drops it, after which the series looks perfectly smooth at 0.07 Ah. A
  first-difference test can only see the edge of a level shift, and dropping the
  edge hides the evidence. `data.truncate_at_collapse` looks at the *level*
  instead and ends the record at the collapse. Truncated to 40 cycles, these
  cells no longer reach 1.40 Ah and are excluded as right-censored.
* **The censored group** simply had their experiment stopped early. Their true
  RUL is unknown; training on "RUL = cycles until the lab went home" teaches the
  wrong thing. Handling them properly needs survival analysis (see
  `docs/limitations.md`).

Setting `data.eol_reference: initial` scores each cell against **its own**
beginning-of-life capacity and readmits most of the cold-chamber group. That is a
legitimate alternative convention; it is not the default because the headline
numbers would then not be comparable with the published NASA-dataset literature,
which uses the 2.0 Ah rating.

## Data quality issues encountered

| Issue | Frequency | Handling |
|---|---|---|
| Aborted / partial leading discharges | 9 cells, 1–7 cycles each | Trimmed by `data.trim_leading_outliers`; cycle index re-based to 1 |
| Single-step capacity jumps > 0.7 Ah | 14 cycles | Dropped as rig glitches |
| Sustained capacity collapse from a chamber change | 4 cells | Record truncated at the collapse (`_truncate_at_collapse`) |
| Missing impedance for early cycles | ~8 % of rows | Causal forward-fill from the last preceding sweep |
| Non-physical EIS values (negative or > 10 Ω) | occasional | Nullified, then causally imputed |
| Capacity recovery after rest | pervasive, by design | **Not** removed — it is real physics. Handled via trailing-median smoothing and the EOL persistence rule |

Complete, machine-readable detail lives in `data/processed/manifest.json` under
the `validation` key after any run.

## Statistics of the modelling cohort

Regenerate with `python scripts/prepare_data.py`; the authoritative numbers for a
given run are in `data/processed/manifest.json`.

| | |
|---|---|
| Cells | 5 (B0005, B0006, B0018, B0033, B0034) |
| Labelled rows (pre-EOL) | 520 |
| Rows after the warm-up trim | 495 |
| RUL range | 0 – 126 cycles (mean 52.8) |
| End-of-life cycle | 77 – 127 |
| Ambient conditions | 24 °C throughout, all cells |

## Known biases and limitations

1. **Small cohort.** Five cells is a small sample. The leave-one-battery-out
   spread across folds (σ ≈ 2.3 cycles MAE) is the honest uncertainty statement.
2. **One chemistry, one format.** LCO 18650. Nothing here has been shown to
   transfer to LFP, NMC, or pouch/prismatic formats.
3. **Laboratory duty cycles.** Constant-current discharge in a chamber. Real EV
   and grid-storage duty cycles are irregular and partial; transfer to field data
   is unmeasured.
4. **Accelerated ageing.** Cells were cycled back-to-back. Calendar ageing, which
   dominates in many real deployments, is largely absent.
5. **2008-era instrumentation.** Capacity readings carry a few percent of noise,
   which places a floor on achievable accuracy.
6. **Survivorship in the cohort gates.** The gates above are applied before
   modelling. They are principled and logged, but they do define which cells the
   reported metric describes.

## Citation

```bibtex
@misc{saha2007battery,
  author       = {Saha, B. and Goebel, K.},
  title        = {Battery Data Set},
  year         = {2007},
  publisher    = {NASA Ames Prognostics Data Repository},
  howpublished = {NASA Ames Research Center, Moffett Field, CA},
  note         = {https://www.nasa.gov/intelligent-systems-division/}
}
```

## Adding another dataset

Write one subclass of `BatterySource` implementing `discover()` and
`load_battery()`, decorate it with `@register_source("calce")`, and point
`data.source` at it. No other file changes — schema coercion, validation,
labelling, feature engineering, splitting and evaluation are all dataset-agnostic
by construction.
