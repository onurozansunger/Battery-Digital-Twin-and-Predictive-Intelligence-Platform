# Prediction uncertainty — split conformal

## Why conformal

Quantile regression and Monte-Carlo dropout both produce intervals whose width
is a statement the *model* makes about itself. Whether that statement is true is
an open question you then have to check empirically anyway.

Split conformal inverts the order: take an arbitrary point predictor, measure how
wrong it actually was on held-out data, and build the interval from those
measured residuals. The coverage guarantee comes from the calibration data, not
from the model being well-specified. For a prototype whose headline claim must be
checkable, that is the right trade.

## The construction

For calibration rows *j* with target coverage 1 − α:

```
q̂ = Quantile_{ceil((n+1)(1-α))/n} ( |y_j − ŷ_j| )
interval(ŷ) = [ŷ − q̂, ŷ + q̂]
```

The `(n + 1)` finite-sample correction is not cosmetic. Without it the interval
under-covers at small *n*, and at this cohort size every interval is small-*n*.

## The assumption, stated plainly

Marginal coverage ≥ 1 − α holds **provided calibration and test rows are
exchangeable**. Here that is only approximately true, in two specific ways:

1. **Rows within a cell are strongly autocorrelated.** Consecutive cycles are
   near-duplicates, so *n* overstates the effective sample size and realised
   coverage is noisier than the nominal level suggests.
2. **Calibration cells and the served cell are different physical cells.**
   Exchangeability is a cross-cell assumption, not an i.i.d.-rows one. Two cells
   from the same rig on the same duty cycle are close to exchangeable; a cell
   from a different chemistry or a different thermal environment is not.

Both are restated in the interval's own `note` field, so they travel with the
number rather than living only in this document.

## Prediction interval, not confidence interval

The interval covers a **future observation**, not a parameter. The naming is
consistent throughout: `PredictionInterval`, `interval_type:
"prediction_interval"`, "prediction interval" in the dashboard and the API
description. Calling it a confidence interval would be a category error that
overstates what it delivers.

## Life-stage conditioning

RUL residuals are strongly heteroscedastic: a fresh cell's remaining life is
nearly unpredictable, a nearly-dead cell's is not. A single global quantile is
far too wide near end of life — which is exactly where a maintenance decision is
being made.

With `uncertainty.normalise_by_life_stage` (default on), calibration residuals
are bucketed by life fraction at `uncertainty.life_stage_edges` (default
0.33 / 0.66 → `early` / `mid` / `late`) and a quantile is fitted per bucket. A
bucket with fewer than `uncertainty.min_calibration_rows` residuals falls back to
the global quantile, and the fallback is logged.

This is standard Mondrian (group-conditional) conformal prediction. The guarantee
holds within each group under the same exchangeability assumption.

## Where the calibration data comes from

**Out-of-fold predictions over the non-test cells**, produced by
leave-one-battery-out within train + validation. Two reasons rather than a fixed
held-out slice:

- the cohort is five cells, so a dedicated calibration cell would cost 20 % of
  the training data and still yield one cell's worth of residuals;
- out-of-fold predictions are honest — each row is scored by a model that never
  saw its cell — so the residuals are the ones a deployed model would make.

Test cells are excluded from every conformal fit.

## What is reported

- Empirical coverage against the nominal target
- Mean and median interval width
- **Coverage per battery** and **coverage per life stage**

The last two matter more than the marginal number. A marginal 90 % that is 99 %
early in life and 55 % near end of life is worse than useless for a maintenance
decision, and only the breakdown reveals it. Both are in
`reports/milestone_2/evaluation_report.md` and
`reports/milestone_2/rul_out_of_fold_intervals.csv`.

## Post-processing

The lower bound is clipped at zero (`ConformalIntervalEstimator.lower_clip`):
negative remaining life is not a physical statement. Clipping can pull a bound
past the point estimate when the estimate itself sits outside the physical range;
in that case the point estimate is clipped too, so the interval always contains
it. `PredictionInterval.__post_init__` enforces the ordering.

## Where it is implemented

- `src/battery_rul/uncertainty/conformal.py`
- `src/battery_rul/pipelines/milestone_2.py::calibrate_uncertainty`
- `tests/test_uncertainty_calibration.py` — nominal coverage on exchangeable
  data, ordering, clipping, life-stage widths, thin-calibration refusal
