# Model card — multi-task sequence model

## Overview

| | |
|---|---|
| **Name** | `multitask_transformer` |
| **Task** | Joint remaining-useful-life regression, state-of-health regression, and end-of-life-within-horizon classification |
| **Input** | A window of *W* consecutive cycles of one cell, scaled by the fitted training preprocessing artifact |
| **Output** | Three values for the window's final cycle: RUL in cycles, SOH as a fraction, risk probability |
| **Training data** | NASA randomised-usage li-ion cells, five cells after quality gating, training partition only |
| **Artifacts** | `artifacts/multitask/{model.pkl, preprocessing.pkl, metadata.json}` |

## Architecture

```
window (B, W, F)
   │
   ├─ transformer:  linear projection → sinusoidal positions → pre-norm
   │                encoder stack → attention pooling
   └─ lstm / gru:   recurrent stack → concat(last hidden state, sequence mean)
   │
LayerNorm + dropout  →  shared representation (B, d)
   │
   ├── RUL head   (Linear → GELU → Dropout → Linear)
   ├── SOH head   (same)
   └── risk head  (same, emits a logit)
```

Defaults: `d_model` 96, 4 heads, 2 layers, feed-forward 192, dropout 0.15,
head hidden 64, window 20, stride 1. Fixed sinusoidal positions rather than
learned embeddings — with a few hundred windows, learned positions overfit
immediately and add parameters for nothing.

Pre-norm layers, not post-norm: post-norm Transformers need a warm-up schedule to
train stably, and at this data size the schedule is more trouble than the
architecture is worth.

## Why share an encoder

The three tasks are three views of one latent process: remaining life, present
health and near-term crossing probability are all functions of the same
degradation trajectory. A shared encoder gets three supervision signals for one
parameter set, which matters when the binding constraint is data rather than
capacity. The risk head in particular sees very few positives on its own;
sharing an encoder trained partly by the two dense regression targets is what
makes it learnable at all.

## Loss

```
total = w_rul · L_rul + w_soh · L_soh + w_risk · L_risk
```

Defaults: `w_rul` 1.0, `w_soh` 1.0, `w_risk` 0.5. Huber for both regression
heads (RUL residuals have heavy tails near end of life and squared error there
drags the whole fit toward the last few cycles); weighted binary cross-entropy
for the risk head, with focal loss available.

RUL is divided by `multitask.rul_scale` (100) before its loss so a target
measured in hundreds of cycles does not swamp an SOH target measured in units of
one.

**Each component is logged separately every epoch.** A combined loss that
improves while one component silently degrades is the standard multi-task
failure mode and it is invisible in the total.

Rows with a missing label for a task are masked out of that task's loss, not
imputed: a missing label should cost the task one sample, not teach it a
fabricated one.

## Class imbalance

`pos_weight = n_negative / n_positive` computed on the **training** windows, when
`risk.class_weight_balanced` is set. No oversampling — duplicating rows of a time
series puts near-identical windows in the same batch and inflates everything
computed afterwards.

## Windowing and warm-up

A window ending at cycle *k* holds cycles `[k−W+1, k]` of **one** cell and is
labelled with that cell's targets at *k*. Windows never cross a cell boundary
(`tests/test_multitask.py::test_windows_never_cross_a_battery_boundary`).

The first *W − 1* scoreable cycles of every cell have no full window. Those rows
return **NaN, not a padded guess**, and are counted as unscored in every metric
table. The first scoreable cycle is
`features.drop_warmup_cycles + W`, computed by `features.warmup.WarmupPolicy`
and used identically in training, evaluation and serving.

## Post-processing

Raw head outputs are unconstrained. Applied at prediction time and recorded as
distinct from the raw output:

- RUL clipped at 0 — negative remaining life is not a physical statement
- SOH clipped to `[soh.plausible_min, soh.plausible_max]`
- Risk clipped to [0, 1] (the sigmoid already guarantees this; the clip is
  belt-and-braces against numerical edge cases)

## Reproducibility

`multitask.seed` (default inherited from the root seed) drives weight
initialisation, the data-loader generator and every stochastic component.
`save()` persists the state dict **plus the architecture configuration it was
trained with**, and `load()` rebuilds from the persisted configuration — a
runtime `window` change cannot silently re-window a trained encoder
(`tests/test_multitask.py::test_loaded_model_uses_its_own_training_window`).

## Evaluation

`reports/milestone_2/evaluation_report.md`, section "Multi-task model versus
independent models". Metrics are reported per partition with the scored/unscored
counts and coverage, because the multi-task model and the tabular models do not
score the same rows.

## Intended use

Engineering decision support on laboratory li-ion cells of the chemistry and
duty cycle it was trained on, as one input among several.

## Out of scope

Autonomous or safety-critical control. Replacing BMS protection. Production EV
deployment. Chemistries, formats, duty cycles or thermal environments outside the
training cohort. Predicting safety events — the training data contains none.

## Known weaknesses

- Five training cells. The encoder has ~176 k parameters and a few hundred
  windows; regularisation and early stopping carry a lot of weight.
- Early-life predictions are the least reliable, and are also where the
  conformal interval is widest — read the interval, not the point estimate.
- Attention weights are exposed for diagnostics and are **not** explanations.
- The risk head is uncalibrated in the multi-task model; the calibrated
  probability the twin reports comes from the independent risk bundle when one
  is present.
