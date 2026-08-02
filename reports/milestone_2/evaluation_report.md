# Milestone 2 — Battery Digital Twin: evaluation report

Generated at 2026-08-02T11:47:27.877983+00:00 from git revision `cf411df`.

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


> **The 90% interval does not achieve its nominal coverage.** Measured out-of-fold coverage is 0.764 and the held-out cell is 0.713. Treat the interval as indicative, not as a 90 % guarantee. The exchangeability assumption conformal prediction rests on is not satisfied across physically distinct cells at this cohort size — which is the honest reading, and the reason the in-sample figure was replaced.


Out-of-fold empirical coverage: **0.764** against a 90% target, mean interval width 30.62 cycles over 373 rows.


Scheme: leave-one-cell-out cross-conformal; the quantile applied to each cell was fitted on the other non-test cells only. For contrast, applying the quantile back to the residuals it was fitted from gives 0.917 — close to the nominal level by construction, which is why it is not the headline.


Held-out test coverage: **0.713**, mean width 41.76 cycles over 122 rows.


### Coverage by life stage

| life_stage | n | empirical_coverage | mean_interval_width |
| --- | --- | --- | --- |
| early | 159 | 0.5723 | 42.0199 |
| late | 70 | 0.9286 | 25.9852 |
| mid | 144 | 0.8958 | 20.2977 |


### Coverage by battery

| battery_id | n | empirical_coverage | mean_interval_width |
| --- | --- | --- | --- |
| B0006 | 106 | 0.9528 | 37.5789 |
| B0018 | 94 | 0.8830 | 34.6252 |
| B0033 | 101 | 0.2970 | 17.5628 |
| B0034 | 72 | 0.9861 | 33.4864 |


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
