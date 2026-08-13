# Milestone 2 — Battery Digital Twin: evaluation report

Generated at 2026-08-13T10:46:15.616049+00:00 from git revision `985b886`.

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

Deployed family: `elastic_net` — leave-one-cell-out over non-test cells. Leave-one-cell-out MAE by family: {'cohort_median_life': 13.49673, 'capacity_fade_extrapolation': 24.85964, 'soh_analogue': 14.29681, 'elastic_net': 8.53366, 'ridge': 12.14476, 'random_forest': 11.40809, 'xgboost': 13.77688, 'lightgbm': 12.82675}.


Out-of-fold empirical coverage: **0.917** against a 90% target, mean interval width 44.81 cycles over 373 rows.


Scheme: leave-one-cell-out cross-conformal; the quantile applied to each cell was fitted on the other non-test cells only. For contrast, applying the quantile back to the residuals it was fitted from gives 1.000 — close to the nominal level by construction, which is why it is not the headline.


Held-out test coverage: **0.951**, mean width 54.16 cycles over 122 rows.


### Coverage by life stage

| life_stage | n | empirical_coverage | mean_interval_width |
| --- | --- | --- | --- |
| early | 159 | 0.8113 | 58.2535 |
| late | 70 | 1.0000 | 54.9727 |
| mid | 144 | 0.9931 | 25.0232 |


### Coverage by battery

| battery_id | n | empirical_coverage | mean_interval_width |
| --- | --- | --- | --- |
| B0006 | 106 | 1.0000 | 56.2956 |
| B0018 | 94 | 1.0000 | 44.9609 |
| B0033 | 101 | 0.7030 | 38.0750 |
| B0034 | 72 | 0.9861 | 37.1458 |


## State of health — forecast

Target: SOH **30 cycles ahead**, not at the current cycle. Current SOH is a measurement, reported by the twin as a derived quantity; modelling it would mean predicting measured capacity divided by a per-cell constant from an input that includes measured capacity, which reports a flattering error and demonstrates nothing.


Selected model: `lightgbm` — leave-one-cell-out over non-test cells.


> Read the MAE against `persistence_baseline_mae`: predicting that SOH will not change over the horizon. A forecaster that does not beat it has not learned degradation.


### Out-of-fold metrics (non-test cells)

| n | mae | mae_percentage_points | persistence_baseline_mae | beats_persistence_baseline | skill_vs_persistence | rmse | r2 | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 253 | 0.0465 | 4.6505 | 0.0659 | yes | 0.2944 | 0.0569 | 0.4063 | 0.1308 |


### Held-out test metrics

| n | n_test_cells | mae | mae_percentage_points | persistence_baseline_mae | beats_persistence_baseline | rmse | r2 | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 92 | 1 | 0.0406 | 4.0646 | 0.0656 | yes | 0.0488 | 0.4956 | 0.1053 |


### Per battery

| battery_id | partition | n | mae | rmse | r2 | max_absolute_error |
| --- | --- | --- | --- | --- | --- | --- |
| B0005 | test | 92 | 0.0406 | 0.0488 | 0.4956 | 0.1053 |
| B0006 | out_of_fold | 76 | 0.0680 | 0.0777 | -0.4278 | 0.1308 |
| B0018 | out_of_fold | 64 | 0.0295 | 0.0359 | 0.4742 | 0.0852 |
| B0033 | out_of_fold | 71 | 0.0400 | 0.0493 | -0.1967 | 0.0967 |
| B0034 | out_of_fold | 42 | 0.0445 | 0.0503 | -2.8434 | 0.0897 |


## Failure risk


> **This model failed its acceptance gate.** It does not beat the cycle-index baseline out of fold, so the twin marks its probability `experimental` and withholds it from the recommendation rules. The numbers below are reported for transparency, not because the model is fit to drive a maintenance decision.

Model: `random_forest`, horizon 30 cycles, decision threshold 0.409 (tuned on out-of-fold non-test rows, objective `f1`).


Calibration (isotonic) on 373 rows: Brier 0.1327 → 0.0999, ECE 0.1439 → 0.0000.


> **Read the AUCs against the cycle-index baseline, not against 1.0.** The label is `RUL ≤ H`, so within a single cell the positives are exactly the last H cycles and *cycle index alone* ranks them perfectly. Any AUC on a single-cell partition is degenerate; the `*_cycle_index_baseline` columns are what carry information.


> The post-calibration Brier and ECE on out-of-fold rows are **in-sample** — the calibrator was fitted on those rows, so its ECE there is often exactly zero. The test-partition figures are the out-of-sample calibration evidence.


### Out-of-fold, before and after calibration

| variant | n | n_positive | pr_auc | pr_auc_cycle_index_baseline | roc_auc | roc_auc_cycle_index_baseline | beats_cycle_index_baseline | precision | recall | f1 | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 373 | 124 | 0.8325 | 0.9282 | 0.9082 | 0.9564 | no | 0.7727 | 0.6855 | 0.7265 | 0.1327 | 0.1439 |
| calibrated | 373 | 124 | 0.8296 | 0.9282 | 0.9228 | 0.9564 | no | 0.6237 | 0.9758 | 0.7610 | 0.0999 | 0.0000 |


### Held-out test, calibrated

| n | n_test_cells | n_positive | pr_auc | pr_auc_cycle_index_baseline | roc_auc | roc_auc_cycle_index_baseline | beats_cycle_index_baseline | precision | recall | f1 | brier | ece | true_positive | false_positive | false_negative | true_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 122 | 1 | 31 | 0.9916 | 1.0000 | 0.9981 | 1.0000 | no | 0.7209 | 1.0000 | 0.8378 | 0.0690 | 0.1382 | 31 | 12 | 0 | 79 |


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
