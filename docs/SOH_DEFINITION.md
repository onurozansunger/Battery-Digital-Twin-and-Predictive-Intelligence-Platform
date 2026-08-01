# State of Health — definition

## The quantity

For cell *i* at discharge cycle *k*:

```
SOH_i(k) = Q̃_i(k) / Q_ref_i
```

where `Q̃_i(k)` is the **trailing-median-smoothed** discharge capacity (window
`data.capacity_smoothing_window`, never centred — a centred filter would read
future cycles) and `Q_ref_i` is the configured reference capacity.

## One internal representation

**SOH is a fraction in [0, 1] everywhere inside this codebase**: targets, model
outputs, thresholds, API payloads, snapshot fields. Percentages exist only in
rendered strings, and are computed at the point of rendering
(`BatteryHealthState.soh_percent` is derived from `soh`, never set
independently).

This is not a stylistic preference. Mixing fractions and percentages is the most
common way an SOH pipeline produces a plausible-looking number that is wrong by
a factor of one hundred, and the error is invisible in code review because both
representations look reasonable in isolation. One unit, one place it is
converted.

## Reference strategies

Configured by `soh.reference_strategy`:

| Strategy | Reference | Interpretation | Weakness |
|---|---|---|---|
| `nominal` | manufacturer rating (2.0 Ah for the NASA 18650 cells) | absolute condition, comparable across cells | a cell that left the factory at 1.85 Ah reads 92.5 % when new |
| `first_cycle` | the cell's own first valid measurement | fade since beginning of life; every cell starts at exactly 1.0 | one aborted opening discharge poisons the entire series |
| `first_n_cycle_mean` **(default)** | mean of the first `soh.reference_cycles` valid cycles | fade, robust to a single bad opening reading | needs N cycles before it is established |

The default is `first_n_cycle_mean` with N = 5. The NASA rig produces aborted or
partial opening discharges often enough that the loader has a dedicated
leading-artifact trim for them; `first_cycle` would let exactly those readings
set the scale for the whole cell.

**The strategies are not interchangeable and are never mixed within a run.** The
choice is persisted in the model bundle's `target_definition` and the runtime
configuration is checked against it before serving (`artifacts.strict_compatibility`).

## Causality

`first_cycle` and `first_n_cycle_mean` read the *opening* cycles of a cell,
which are in the past for every cycle at which a prediction is made, and the
smoothed capacity is trailing-filtered. So the reference is known from cycle N
onward and no row uses a future observation. Rows before that are inside the
warm-up region discarded by `features.drop_warmup_cycles` in every shipped
configuration. `tests/test_targets_m2.py::test_reference_is_causal` asserts
that truncating a cell's future does not move its reference.

## Health bands

Configured by `soh.healthy_min`, `soh.slightly_degraded_min`, `soh.warning_min`:

| Class | Default range |
|---|---|
| `healthy` | SOH ≥ 0.90 |
| `slightly_degraded` | 0.80 ≤ SOH < 0.90 |
| `warning` | 0.70 ≤ SOH < 0.80 |
| `critical` | SOH < 0.70 |
| `unknown` | SOH missing or non-finite |

These are **project-level engineering categories and configurable
demonstration thresholds**. They are not a standard, not a regulatory
definition, and not a medical-style diagnosis. The 0.70 boundary aligns with the
end-of-life threshold used for the RUL target so the two definitions agree; the
other two are conventional second-life practice.

`unknown` is a distinct class on purpose. A missing measurement must not render
as a green tile — collapsing "we could not tell" into "healthy" is how a broken
sensor becomes a clean dashboard.

## Plausible range

`soh.plausible_min` / `soh.plausible_max` (default 0.20 / 1.20) clip both the
target and every model output. A reading far outside that band is a measurement
problem, not a health state; clipping is logged and counted in the target
report (`n_clipped_to_plausible_range`).

## Where it is implemented

- `src/battery_rul/targets/soh.py` — reference strategies, target, banding
- `src/battery_rul/config.py::SOHConfig` — every threshold
- `tests/test_targets_m2.py` — reference strategies, causality, bands, clipping
