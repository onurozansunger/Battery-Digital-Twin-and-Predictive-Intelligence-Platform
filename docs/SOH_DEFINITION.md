# State of Health — definition

## Two different things, kept apart

| | What it is | Provenance | Source |
|---|---|---|---|
| **Current SOH** | smoothed capacity ÷ reference | `derived` — a measurement | computed deterministically, no model |
| **SOH forecast** | SOH at *t + H* (default H = 30 cycles) | `predicted` | the SOH bundle |

This separation is a correction. Milestone 2 trained a model to predict SOH *at
the current cycle* and published a 1.34 % test MAE for it. That target is
measured capacity divided by a per-cell constant, and measured capacity is a
model input — so the model was learning a rescaling, and the error described how
well a rescaling can be learned, not whether a latent health state can be
inferred. It was target-proxy leakage, not future leakage, and the number was
misleading rather than wrong.

Current SOH is now reported as what it is: a measurement. The model forecasts,
which at cycle *t* is a genuine prediction because nothing in the input reveals
capacity at *t + H*. See `docs/MILESTONE_2_1_REVIEW_FIXES.md` §2.

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

## Reading the forecast metrics

Every forecast metric is reported beside a **persistence baseline** — predicting
that SOH will not change over the horizon — with a `beats_persistence_baseline`
flag and a `skill_vs_persistence` score. On a slowly degrading cell, persistence
is a strong baseline, and a forecaster that cannot beat it has not learned
degradation regardless of how small its MAE looks.

Metrics use `soh_metrics`, not the RUL `compute_metrics`. The latter reports
`within_10_cycles`, `within_25_cycles` and `alpha_lambda`, all defined in
discharge cycles; applied to a fraction in [0, 1] they are meaningless, and
earlier reports published exactly that.

## Where it is implemented

- `src/battery_rul/targets/soh.py` — reference strategies, both targets, banding
- `src/battery_rul/evaluation/metrics.py::soh_metrics` — SOH-appropriate metrics
- `src/battery_rul/config.py::SOHConfig` — every threshold
- `tests/test_targets_m2.py` — reference strategies, causality, bands, clipping
