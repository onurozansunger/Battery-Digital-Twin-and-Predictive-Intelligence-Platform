# Milestone 2 — evaluation design

## Where the numbers are

| Artifact | Contents |
|---|---|
| `reports/milestone_2/metrics.json` | every metric, as JSON. The source of truth. |
| `reports/milestone_2/evaluation_report.md` | the rendered report, a projection of the JSON |
| `reports/milestone_2/rul_out_of_fold_intervals.csv` | per-row RUL prediction + interval + life stage |
| `reports/milestone_2/rul_test_intervals.csv` | the same on the held-out test cells |
| `reports/milestone_2/soh_out_of_fold.csv` | per-row SOH prediction |
| `reports/milestone_2/risk_out_of_fold.csv` | raw and calibrated risk probability per row |
| `reports/milestone_2/risk_reliability_curve.csv` | the reliability diagram, as data |
| `reports/nested_model_comparison*.csv` | the Milestone 1.1 nested RUL comparison |

Every one is written by the run that produced it. Nothing is carried forward.

## Partitions

The Milestone 1 battery-holdout split is reused unchanged, so both milestones
describe the same cells:

- **train** — model fitting.
- **validation** — model-family selection for SOH.
- **calibration** — *not* a fixed partition. Both calibrators are fitted on
  **out-of-fold predictions over the non-test cells**, from leave-one-battery-out
  within train + validation. At five cells, a dedicated calibration cell would
  cost 20 % of the training data and yield one cell's worth of residuals;
  out-of-fold predictions give many more rows, and each is scored by a model that
  never saw its cell.
- **test** — scored once, at the end. No test label enters any calibration fit,
  threshold search or selection decision.

## RUL

Reported: MAE, RMSE, R², bias, MAPE (with a floored denominator, since RUL
reaches 0), α–λ accuracy, within-10-cycles, prognostic horizon.

Reported **out-of-fold** (every non-test cell, each scored by a model that never
saw it) and on the **held-out test cells**. Per battery and per life stage, not
only pooled.

The headline is the **nested** figure from Milestone 1.1: family selection runs
inside every outer fold, so the pooled metric estimates the whole procedure
rather than an already-chosen model. It is higher than the per-candidate best,
and that difference *is* the cost of selection.

## Uncertainty

Empirical coverage against the 90 % target, mean and median interval width, and
both broken down **by battery** and **by life stage**. The breakdowns matter more
than the marginal number: a marginal 90 % that is 99 % early in life and 55 % near
end of life is worse than useless for a maintenance decision.

## SOH

MAE, RMSE, R², **maximum absolute error** and per-battery breakdown, on
out-of-fold rows and on test. Maximum absolute error is reported because a mean
SOH error of 1 % with a 12 % worst case is a different product from a uniform
2 %, and the mean hides it.

Model family selected on validation from elastic net / random forest / LightGBM;
the selection and its scores are recorded in the metrics JSON.

## Failure risk

PR-AUC, ROC-AUC, precision, recall, F1, Brier, expected calibration error, and
the full confusion matrix at the selected threshold — **before and after
calibration**, out-of-fold and on test. The before/after pair is the point: it
shows whether calibration helped, rather than asserting that it did.

ROC-AUC and PR-AUC are `NaN` on a single-class evaluation set. A degenerate set
has no AUC and printing 0.5 would be a fabricated number.

PR-AUC is the metric to read first. The positive class is rare by construction,
and ROC-AUC is optimistic under class imbalance.

## Multi-task versus independent models

Reported per partition with `n_rows`, `n_scored`, `n_unscored` and `coverage`
alongside every metric. Coverage below 1.0 is the sequence warm-up: the first
*window − 1* scoreable cycles of each cell have no full window.

**Those rows are reported as unscored, not dropped from the denominator.** They
are the hardest early-life rows, and quietly excluding them from only the
sequence model's metric would make the comparison an artifact of window length.
The two families are comparable only on the rows both can score.

## How the deployed RUL family is chosen

From the nested comparison's **selection frequency** — the family the inner loop
chose most often across outer folds, restricted to tabular families (a sequence
model cannot score early cycles at serving time either). If no nested result
exists the pipeline falls back to the first configured tabular model and **logs a
warning saying the choice was not evidence-based**, so a reader can tell the two
cases apart.

## What is deliberately not done

- No single average metric decides the final model. The selection criteria are
  the pooled metric, the fold dispersion, the worst fold, coverage, and how often
  the inner loop chose the family. All five are in the tables.
- No metric is reported without its sample size.
- No number is quoted from a previous run.
