# Milestone 2 — Battery Digital Twin: evaluation report

Generated at 2026-08-01T08:46:38.230880+00:00 from git revision `82ef951`.

> Every number below was produced by the pipeline run that wrote `reports/milestone_2/metrics.json`. Nothing here is carried over from an earlier run.


## Definitions in force

- **End of life**: smoothed capacity at or below 70% of the nominal reference for 3 complete consecutive cycles.
- **State of health**: fraction in [0, 1], reference strategy `first_n_cycle_mean` over 5 cycle(s).
- **Failure risk**: RUL(t) ≤ 30 cycles. A *derived* label from the capacity threshold — not an observed safety failure.
- **Uncertainty**: split_conformal at 90% target coverage. Prediction intervals, not confidence intervals.
- **Probability calibration**: isotonic, fitted on out-of-fold predictions over non-test cells only.


## Target generation

SOH (first_n_cycle_mean): 520 rows, range [0.689, 1.030], mean 0.888.


Risk label positive rate by horizon: H=20: 0.202, H=30: 0.298, H=50: 0.490


## Remaining useful life and prediction intervals

Deployed family: `random_forest`.


Out-of-fold empirical coverage: **0.917** against a 90% target, mean interval width 46.96 cycles over 373 rows.


Held-out test coverage: **0.803**, mean width 48.60 cycles over 122 rows.


### Coverage by life stage

| life_stage | n | empirical_coverage | mean_interval_width |
| --- | --- | --- | --- |
| early | 159 | 0.9120 | 56.0167 |
| late | 70 | 0.9286 | 44.0381 |
| mid | 144 | 0.9167 | 38.3848 |


### Coverage by battery

| battery_id | n | empirical_coverage | mean_interval_width |
| --- | --- | --- | --- |
| B0006 | 106 | 0.9528 | 46.9236 |
| B0018 | 94 | 1.0000 | 45.8726 |
| B0033 | 101 | 0.8614 | 49.4214 |
| B0034 | 72 | 0.8333 | 44.9896 |


## State of health

Selected model: `lightgbm` (chosen on validation).


### Out-of-fold metrics (non-test cells)

| n | mae | rmse | r2 | max_absolute_error |
| --- | --- | --- | --- | --- |
| 373 | 0.0244 | 0.0320 | 0.8290 | 0.0890 |


### Held-out test metrics

| n | mae | rmse | r2 | max_error |
| --- | --- | --- | --- | --- |
| 122 | 0.0134 | 0.0200 | 0.9375 | 0.0552 |


### Per battery

| battery_id | partition | n | mae | rmse | r2 |
| --- | --- | --- | --- | --- | --- |
| B0005 | test | 122 | 0.0134 | 0.0200 | 0.9375 |
| B0006 | out_of_fold | 106 | 0.0400 | 0.0490 | 0.7341 |
| B0018 | out_of_fold | 94 | 0.0084 | 0.0098 | 0.9801 |
| B0033 | out_of_fold | 101 | 0.0222 | 0.0236 | 0.7563 |
| B0034 | out_of_fold | 72 | 0.0255 | 0.0297 | -0.5357 |


## Failure risk

Model: `lightgbm`, horizon 30 cycles, decision threshold 0.489 (tuned on out-of-fold non-test rows, objective `f1`).


Calibration (isotonic) on 373 rows: Brier 0.2776 → 0.1416, ECE 0.2694 → 0.0000.


> **Read the AUCs against the cycle-index baseline, not against 1.0.** The label is `RUL ≤ H`, so within a single cell the positives are exactly the last H cycles and *cycle index alone* ranks them perfectly. Any AUC on a single-cell partition is degenerate; the `*_cycle_index_baseline` columns are what carry information.


> The post-calibration Brier and ECE on out-of-fold rows are **in-sample** — the calibrator was fitted on those rows, so its ECE there is often exactly zero. The test-partition figures are the out-of-sample calibration evidence.


### Out-of-fold, before and after calibration

| variant | n | n_positive | pr_auc | pr_auc_cycle_index_baseline | roc_auc | roc_auc_cycle_index_baseline | beats_cycle_index_baseline | precision | recall | f1 | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 373 | 124 | 0.6287 | 0.9282 | 0.7849 | 0.9564 | no | 0.5739 | 0.5323 | 0.5523 | 0.2776 | 0.2694 |
| calibrated | 373 | 124 | 0.6467 | 0.9282 | 0.8334 | 0.9564 | no | 0.5519 | 0.9435 | 0.6964 | 0.1417 | 0.0000 |


### Held-out test, calibrated

| n | n_test_cells | n_positive | pr_auc | pr_auc_cycle_index_baseline | roc_auc | roc_auc_cycle_index_baseline | beats_cycle_index_baseline | precision | recall | f1 | brier | ece | true_positive | false_positive | false_negative | true_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122 | 1 | 31 | 0.7209 | 1.0000 | 0.9341 | 1.0000 | no | 0.7209 | 1.0000 | 0.8378 | 0.0924 | 0.1013 | 31 | 12 | 0 | 79 |


## Multi-task model versus independent models

Encoder `transformer`, window 20, 176452 parameters, best epoch 2.

| partition | n_rows | n_scored | coverage | rul_mae | soh_mae | risk_pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| val | 106 | 87 | 0.8208 | 8.7986 | 0.1631 | 1.0000 |
| test | 122 | 103 | 0.8443 | 14.4302 | 0.0547 | 1.0000 |


A multi-task risk PR-AUC near 1.0 on a one-cell partition is not evidence of a good classifier: cycle index alone achieves it, for the reason given in the risk section above. Compare the two columns.


Coverage below 1.0 is the sequence warm-up: the first 19 scoreable cycles of each cell have no full window. Those rows are reported as unscored rather than dropped from the denominator, so this table is comparable with the independent models only on the rows both can score.


## Limitations

See `docs/MILESTONE_2_LIMITATIONS.md`. In short: a five-cell laboratory cohort, one chemistry, one duty cycle, a derived rather than observed failure label, and conformal coverage that assumes an exchangeability between calibration and served cells that physical cells only approximately satisfy.
