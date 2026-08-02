# Failure risk — definition, and what it is not

## The label

At cycle *k* of cell *i*:

```
failure_within_horizon(k) = 1  if RUL_i(k) <= H
                          = 0  otherwise
```

with `H = risk.horizon_cycles` (default 30; 20 and 50 are also attached as
`failure_within_horizon_20` / `_50` so a different horizon can be evaluated
without regenerating the dataset).

## What this label is *not*

It is **not an observed safety failure.** This matters enough to state four ways:

- Nothing in the NASA dataset records a thermal event, a venting incident, an
  internal short, or a pack-level fault. There are no such events to learn from.
- The label is derived arithmetically from the RUL target, which is itself
  derived from a capacity threshold **this project chose** (70 % of reference,
  persisting for 3 cycles).
- A positive label therefore means "smoothed capacity is projected to cross the
  configured end-of-life threshold within H cycles". It does not mean "this cell
  is about to become dangerous."
- The name "failure risk" is inherited from the prognostics literature. Every
  user-facing surface in this repository — the API response
  (`BatteryRiskAssessment.label_definition`), the dashboard's risk tab, the model
  card — restates the derivation next to the number.

A model cannot predict a category of event it has never seen. If you need
safety-event prediction, you need data containing safety events.

## Leakage

The label at cycle *k* is a function of the record's end-of-life cycle, which is
established offline over the complete series. **That is legitimate for a label** —
supervision is always constructed with hindsight. What must never happen is a
*feature* at cycle *k* seeing beyond *k*.

The two concerns live in separate modules so they can be reviewed
independently: label construction in `src/battery_rul/targets/`, feature
causality in `src/battery_rul/features/engineering.py` with a mechanical
verifier (`assert_no_leakage`) that is itself tested against a planted
violation.

Right-censored cells — no confirmed end-of-life crossing — carry no valid risk
label. They are labelled NaN and excluded by the same
`target.require_eol_reached` gate that governs RUL.

## Class imbalance

Positives are rare by construction: only the final H cycles of each cell
qualify. Handling:

- **Class weighting** (`risk.class_weight_balanced`) in the classifier, and
  `pos_weight` in the multi-task risk head.
- **Focal loss** available for the multi-task head (`multitask.risk_loss: focal`).
- **Threshold tuning** on out-of-fold, non-test rows only.
- **Precision–recall analysis** reported alongside ROC.

**Row-level oversampling is deliberately not used.** Duplicating rows of a time
series puts near-identical windows in the same batch, breaks the temporal
structure the sequence models depend on, and inflates every metric computed over
the resampled set.

## The acceptance gate

Reporting that a model loses to a trivial baseline is necessary but not
sufficient. Milestone 2 documented the loss honestly and then let the same
probability trigger inspections and replacements — which is acting on a model
that has demonstrated nothing.

`train_risk` now records `passes_acceptance_gate` in the bundle: does the
calibrated out-of-fold PR-AUC exceed the cycle-index baseline? When it does not,
and `risk.require_beating_baseline` is set, the digital-twin service:

* marks the assessment `is_experimental` and `excluded_from_recommendation`,
* **withholds the probability from the recommendation rules**, so remaining
  life, its lower bound and measured health carry the decision alone,
* adds a snapshot warning saying so.

The probability is still reported. Suppressing it would be its own dishonesty;
the point is that it must not silently drive an action.

## Threshold selection

`risk.threshold` is `null` by default, meaning "tune it". Tuning runs on
**out-of-fold predictions over non-test cells** (leave-one-battery-out within
train + validation) — never on the final test labels. The chosen value is
persisted in the risk bundle's `thresholds.decision_threshold` and replayed at
serving, so the evaluation and the deployed decision use the same number.

Objectives (`risk.threshold_objective`):

- `f1` — balanced default.
- `youden` — maximise `recall + specificity − 1`.
- `precision_at_recall` — the most precise threshold that still achieves
  `risk.min_recall`. Use this when a missed crossing costs more than a needless
  inspection, which for a battery it usually does.

## The trivial baseline every AUC must be read against

Because the label is `RUL(t) ≤ H`, the positives of a cell are **exactly its last
H cycles**. Within a single cell, therefore, *cycle index alone* ranks them
perfectly:

| Cell | rows | positives | model ROC-AUC | ROC-AUC of `cycle_index` |
|---|---|---|---|---|
| B0006 | 106 | 31 | 0.51 | **1.00** |
| B0018 | 94 | 31 | 1.00 | **1.00** |
| B0033 | 101 | 31 | 0.71 | **1.00** |
| B0034 | 72 | 31 | 0.87 | **1.00** |

(Measured on this repository's out-of-fold predictions; regenerate with
`reports/milestone_2/risk_out_of_fold.csv`.)

Two consequences, both enforced in code:

1. **Any AUC computed on a single-cell partition is degenerate.** The Milestone 1
   battery-holdout puts one cell in test, so a headline "test ROC-AUC 0.93"
   describes a model performing *worse than counting cycles* on that cell.
2. `risk_metrics` therefore reports `roc_auc_cycle_index_baseline` and
   `pr_auc_cycle_index_baseline` computed on **exactly the same rows**, plus a
   `beats_cycle_index_baseline` flag, and logs a warning when the model loses.
   The evaluation report leads the risk section with this caveat.

The informative question is not "is the AUC high?" but "does the model beat a
cycle counter, and does it still beat one across cells?" — which is why the
pooled out-of-fold figure over four cells is the one worth reading.

## Reported metrics

PR-AUC, ROC-AUC (each beside its cycle-index baseline), precision, recall, F1,
Brier score, expected calibration error, and the full confusion matrix at the
selected threshold — before **and** after probability calibration, on out-of-fold
rows and on the held-out test cells.
See `reports/milestone_2/evaluation_report.md`.

**The post-calibration Brier and ECE on out-of-fold rows are in-sample.** The
calibrator was fitted on those rows, so its ECE there is often exactly zero and
means nothing. The test-partition figures are the out-of-sample calibration
evidence; the metrics payload flags this with `calibration_is_in_sample`.

ROC-AUC and PR-AUC are reported as `NaN` on a single-class evaluation set rather
than as 0.5 or 1.0. A degenerate set has no AUC; printing one would be an
invented number.

## Risk bands

Applied to the **calibrated** probability (`risk.low_max` / `medium_max` /
`high_max`):

| Class | Default range |
|---|---|
| `low` | p < 0.20 |
| `medium` | 0.20 ≤ p < 0.50 |
| `high` | 0.50 ≤ p < 0.80 |
| `very_high` | p ≥ 0.80 |
| `unknown` | probability unavailable |

## Where it is implemented

- `src/battery_rul/targets/risk.py` — label, bands
- `src/battery_rul/pipelines/milestone_2.py::train_risk` — classifier, calibration, threshold
- `src/battery_rul/calibration/probability.py` — calibration, threshold search, metrics
- `tests/test_targets_m2.py`, `tests/test_uncertainty_calibration.py`
