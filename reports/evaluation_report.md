# Evaluation Report — battery_rul_v1

_Generated 2026-07-31 20:28 UTC_

> NASA li-ion remaining-useful-life baseline. Battery-holdout split, causal feature engineering, nine models compared, champion selected on validation.

## 1. Headline result

**transformer** is the champion model, selected by `rmse` on the **validation** partition and reported here on the untouched **test** partition.

- **MAE** — 11.00 cycles
- **RMSE** — 12.99 cycles (95 % CI 11.77–14.20)
- **R²** — 0.790
- **MAPE** — 69.5 % (denominator floored at 1 cycle)
- **α-λ accuracy (α=20%)** — 37.8% of predictions inside the relative error cone
- **Predictions within 10 cycles** — 51.9%
- **Bias** — -6.76 cycles (conservative)
- **Prognostic horizon** — not reached: predictions never settle inside the ±20% relative cone and stay there. The cone tightens to a couple of cycles near end of life, which is a demanding bar at this error level.

## 2. Experimental setup

### 2.1 Dataset

| battery_id | n_cycles | capacity_start_ah | capacity_end_ah | soh_end | eol_cycle | reaches_eol | ambient_c |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 168 | 1.8565 | 1.2935 | 0.6467 | 127.0000 | True | 24.0000 |
| B0006 | 168 | 2.0353 | 1.1644 | 0.5822 | 111.0000 | True | 24.0000 |
| B0007 | 168 | 1.8911 | 1.4063 | 0.7032 | — | False | 24.0000 |
| B0018 | 132 | 1.8550 | 1.3548 | 0.6774 | 99.0000 | True | 24.0000 |
| B0029 | 40 | 1.6975 | 1.6275 | 0.8138 | — | False | 43.0000 |
| B0030 | 40 | 1.6561 | 1.5787 | 0.7894 | — | False | 43.0000 |
| B0032 | 40 | 1.7049 | 1.6634 | 0.8317 | — | False | 43.0000 |
| B0033 | 189 | 1.7132 | 1.3280 | 0.6640 | 106.0000 | True | 24.0000 |
| B0034 | 196 | 1.6623 | 1.3120 | 0.6560 | 77.0000 | True | 24.0000 |
| B0036 | 194 | 1.8011 | 1.5795 | 0.7898 | — | False | 24.0000 |
| B0042 | 109 | 1.7287 | 1.3569 | 0.6784 | 44.0000 | True | 10.6000 |
| B0043 | 109 | 1.7138 | 1.3003 | 0.6502 | 44.0000 | True | 10.6000 |
| B0044 | 109 | 1.6865 | 1.2686 | 0.6343 | 44.0000 | True | 10.6000 |
| B0046 | 69 | 1.7282 | 1.1746 | 0.5873 | 19.0000 | True | 4.0000 |
| B0047 | 69 | 1.6743 | 1.1774 | 0.5887 | 12.0000 | True | 4.0000 |
| B0048 | 69 | 1.6580 | 1.2520 | 0.6260 | 18.0000 | True | 4.0000 |

### 2.2 Target definition

RUL(k) = k_EOL − k, where k_EOL is the first **persistent** cycle at which trailing-median-smoothed capacity falls to or below **1.400 Ah** (70% of the nominal reference capacity).

- Labelled rows: **649**
- RUL range: **0.0 – 126.0 cycles**
- Mean RUL: **46.652 cycles**
- Excluded as right-censored (never reached EOL): **B0007, B0029, B0030, B0032, B0036**

End-of-life cycle per cell:

| battery_id | eol_cycle |
| --- | --- |
| B0005 | 127 |
| B0006 | 111 |
| B0018 | 99 |
| B0033 | 106 |
| B0034 | 77 |
| B0042 | 44 |
| B0043 | 44 |
| B0044 | 44 |

### 2.3 Split

**Strategy:** `battery_holdout` — Entire cells held out. Test cells were never seen during training, scaling or feature selection.

- train: **271** rows, cells `['B0018', 'B0033', 'B0043', 'B0044']`
- val: **144** rows, cells `['B0006', 'B0042']`
- test: **194** rows, cells `['B0005', 'B0034']`

No random row-level splitting is used anywhere in this project. Consecutive cycles of one cell are near-duplicates, so a random split lets a model interpolate between neighbouring rows and produces an R² that means nothing.

### 2.4 Features

- Generated: **709** causal features from 14 base signals
- After unsupervised pruning: **397**
- After supervised top-K selection (train partition only): **80**
- Scaler: `robust`
- Warm-up rows dropped: **40**

Every feature at cycle *k* is a function of cycles ≤ *k* of the same cell. This is verified mechanically by `assert_no_leakage`, which rebuilds the features on a truncated history and requires bit-identical values.

## 3. Data quality

- Rows in: **2551** → out: **1869**
- Cells in: **26** → out: **16**

| check | severity | message |
| --- | --- | --- |
| capacity_jump | warning | 14 cycles move capacity by more than 0.70 Ah in one step (rig glitch) |
| cohort_gates | info | Excluded 10 cell(s): B0031 (fades only 0.0% (< 2.0%)); B0038 (starts at 0.55 SoH (< 0.8)); B0039 (starts at 0.24 SoH (< 0.8)); B0040 (starts at 0.40 SoH (< 0.8)); B0041 (starts at 0.03 SoH (< 0.8)); B0045 (starts at 0.54 SoH (< 0.8)); B0053 (starts at 0.53 SoH (< 0.8)); B0054 (starts at 0.58 SoH (< 0.8)); B0055 (starts at 0.66 SoH (< 0.8)); B0056 (starts at 0.67 SoH (< 0.8)) |

## 4. Model comparison (test partition)

| rank | model | n | mae | rmse | mape | smape | r2 | median_ae | max_error | bias | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon | n_unscored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | transformer | 156 | 10.9964 | 12.9884 | 69.4755 | 39.7529 | 0.7900 | 9.7160 | 29.7565 | -6.7575 | 0.3782 | 0.5192 | 0.9359 | — | 38 |
| 2 | gru | 156 | 12.5205 | 14.6421 | 66.1801 | 45.6397 | 0.7331 | 11.3587 | 29.8535 | -7.0168 | 0.2692 | 0.4103 | 0.8910 | — | 38 |
| 3 | ridge | 194 | 14.1218 | 17.4941 | 67.7696 | 32.8421 | 0.7182 | 13.2701 | 38.8916 | -3.3933 | 0.4536 | 0.4124 | 0.8660 | 0.0000 | 0 |
| 4 | lstm | 156 | 14.9164 | 18.4843 | 91.5129 | 52.2972 | 0.5747 | 10.8865 | 39.9898 | -7.2480 | 0.2821 | 0.4744 | 0.8269 | — | 38 |
| 5 | linear_regression | 194 | 19.7012 | 20.1909 | 96.9495 | 69.2389 | 0.6247 | 20.8711 | 27.6102 | -3.3377 | 0.1392 | 0.0567 | 0.9691 | 0.0000 | 0 |
| 6 | random_forest | 194 | 17.3699 | 20.6771 | 116.4022 | 43.0181 | 0.6064 | 17.4865 | 44.1648 | 0.2605 | 0.3505 | 0.3505 | 0.7320 | — | 0 |
| 7 | catboost | 194 | 17.8143 | 23.3541 | 123.5046 | 44.5482 | 0.4978 | 14.0418 | 56.2626 | -5.3980 | 0.4175 | 0.4330 | 0.7423 | — | 0 |
| 8 | xgboost | 194 | 22.8734 | 25.1737 | 192.0176 | 54.1410 | 0.4165 | 25.4507 | 39.8088 | 5.9226 | 0.1856 | 0.1392 | 0.4691 | — | 0 |
| 9 | lightgbm | 194 | 23.1864 | 25.4871 | 187.5195 | 54.5336 | 0.4019 | 25.9700 | 40.8076 | 5.8406 | 0.1701 | 0.1392 | 0.4485 | — | 0 |

`n_unscored` counts rows the model could not score. Sequence models need a full window of history, so the first *w−1* cycles of every test cell are unscoreable by construction — they are reported, never silently dropped.

### 4.1 Like-for-like: rows every model can score

The table above compares models on different row counts, and the rows the sequence models skip are the early-life ones — the hardest. That difference alone can reorder a ranking. This table restricts every model to the intersection, so the ordering reflects the models rather than their input requirements.

| rank | model | n | mae | rmse | mape | r2 | bias | alpha_lambda | within_10_cycles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | transformer | 156 | 10.9964 | 12.9884 | 69.4755 | 0.7900 | -6.7575 | 0.3782 | 0.5192 |
| 2 | ridge | 156 | 11.8362 | 14.1929 | 78.2207 | 0.7493 | -0.9866 | 0.4744 | 0.4423 |
| 3 | gru | 156 | 12.5205 | 14.6421 | 66.1801 | 0.7331 | -7.0168 | 0.2692 | 0.4103 |
| 4 | random_forest | 156 | 15.4837 | 18.3992 | 138.1223 | 0.5786 | 3.3150 | 0.3718 | 0.4167 |
| 5 | lstm | 156 | 14.9164 | 18.4843 | 91.5129 | 0.5747 | -7.2480 | 0.2821 | 0.4744 |
| 6 | catboost | 156 | 15.4677 | 19.0492 | 147.2523 | 0.5483 | -0.2366 | 0.3974 | 0.4167 |
| 7 | linear_regression | 156 | 19.2061 | 19.7609 | 113.9828 | 0.5139 | -3.9391 | 0.1154 | 0.0705 |
| 8 | lightgbm | 156 | 21.8507 | 24.2679 | 225.0962 | 0.2669 | 9.2958 | 0.2115 | 0.1731 |
| 9 | xgboost | 156 | 21.8933 | 24.3102 | 231.2381 | 0.2644 | 9.4574 | 0.2051 | 0.1731 |

## 5. Per-cell breakdown

With only a handful of held-out cells, the aggregate number can hide a cell the model gets badly wrong. This table is the honest view of the result.

### catboost

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 20.9456 | 26.9034 | 14.9899 | 56.2626 | 91.0586 | 42.5296 | 0.4164 | -14.8582 | 22.4282 | 0.3689 | 0.3770 | 0.6721 | — |
| B0034 | 72 | 12.5086 | 15.5938 | 8.1812 | 28.9612 | 178.4827 | 47.9686 | 0.4370 | 10.6319 | 11.4075 | 0.5000 | 0.5278 | 0.8611 | — |

### gru

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 103 | 13.1551 | 15.5964 | 11.2072 | 29.8535 | 31.8855 | 40.0459 | 0.7248 | -13.1348 | 8.4099 | 0.2816 | 0.4369 | 0.8350 | — |
| B0034 | 53 | 11.2873 | 12.5820 | 11.9809 | 21.8228 | 132.8282 | 56.5107 | 0.3235 | 4.8728 | 11.6001 | 0.2453 | 0.3585 | 1.0000 | — |

### lightgbm

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 21.0140 | 24.3405 | 22.5204 | 40.8076 | 148.6439 | 45.0357 | 0.5223 | -6.5688 | 23.4373 | 0.2705 | 0.2213 | 0.5492 | — |
| B0034 | 72 | 26.8675 | 27.3204 | 27.0744 | 34.3340 | 253.3921 | 70.6272 | -0.7281 | 26.8675 | 4.9536 | 0.0000 | 0.0000 | 0.2778 | — |

### linear_regression

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 18.3178 | 19.0021 | 18.9812 | 27.6102 | 46.1543 | 72.7551 | 0.7089 | -18.3178 | 5.0534 | 0.2213 | 0.0902 | 0.9508 | 0.0000 |
| B0034 | 72 | 22.0453 | 22.0597 | 22.1166 | 22.9941 | 183.0191 | 63.2810 | -0.1267 | 22.0453 | 0.7974 | 0.0000 | 0.0000 | 1.0000 | — |

### lstm

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 103 | 16.4123 | 20.2506 | 11.0666 | 39.9898 | 38.1995 | 50.6247 | 0.5361 | -16.3645 | 11.9286 | 0.2136 | 0.4660 | 0.7379 | — |
| B0034 | 53 | 12.0092 | 14.4468 | 10.1178 | 23.5905 | 195.1219 | 55.5477 | 0.1081 | 10.4689 | 9.9555 | 0.4151 | 0.4906 | 1.0000 | — |

### random_forest

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 15.2799 | 20.0235 | 9.1300 | 44.1648 | 63.2005 | 31.5962 | 0.6767 | -11.9269 | 16.0838 | 0.4754 | 0.5328 | 0.7213 | — |
| B0034 | 72 | 20.9113 | 21.7398 | 20.8636 | 30.7326 | 206.5497 | 62.3719 | -0.0942 | 20.9113 | 5.9443 | 0.1389 | 0.0417 | 0.7500 | — |

### ridge

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 13.9269 | 18.8417 | 6.9987 | 38.8916 | 19.8118 | 22.3544 | 0.7138 | -13.9250 | 12.6927 | 0.6066 | 0.5656 | 0.7869 | 0.0000 |
| B0034 | 72 | 14.4520 | 14.9357 | 14.3008 | 21.3235 | 149.0316 | 50.6129 | 0.4835 | 14.4520 | 3.7701 | 0.1944 | 0.1528 | 1.0000 | — |

### transformer

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 103 | 12.1072 | 14.1058 | 10.1624 | 29.7565 | 29.4933 | 35.6725 | 0.7749 | -12.0776 | 7.2874 | 0.3010 | 0.4660 | 0.9029 | — |
| B0034 | 53 | 8.8378 | 10.4815 | 6.9230 | 17.5244 | 147.1767 | 47.6827 | 0.5305 | 3.5814 | 9.8507 | 0.5283 | 0.6226 | 1.0000 | — |

### xgboost

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 20.9718 | 24.1651 | 23.1968 | 39.8088 | 154.7789 | 45.0231 | 0.5292 | -5.9828 | 23.4128 | 0.2623 | 0.2213 | 0.5328 | — |
| B0034 | 72 | 26.0957 | 26.7963 | 26.1927 | 35.6176 | 255.1166 | 69.5909 | -0.6625 | 26.0957 | 6.0874 | 0.0556 | 0.0000 | 0.3611 | — |

## 6. Residual analysis — champion

| statistic | value |
| --- | --- |
| kurtosis | 0.2905 |
| mean | -6.7575 |
| q05 | -25.4845 |
| q25 | -10.3239 |
| q50 | -7.5723 |
| q75 | -3.5278 |
| q95 | 16.4101 |
| residual_rul_corr | -0.8122 |
| skew | 0.4773 |
| std | 11.0921 |

Residuals correlate with true RUL at ρ = -0.812: the model under-predicts at high RUL. This is the expected signature of regression-to-the-mean on a bounded target and is the main reason early-life predictions should be treated as a range, not a number.

## 7. Learning curve — champion

| fraction | n_train_rows | n_train_batteries | train_rmse | test_rmse | train_mae | test_mae | train_r2 | test_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2500 | 94 | 1 | 0.7388 | 11.9088 | 0.4957 | 9.7825 | 0.9988 | 0.8235 |
| 0.5000 | 195 | 2 | 0.9489 | 14.2085 | 0.8334 | 11.7929 | 0.9983 | 0.7487 |
| 0.7500 | 233 | 3 | 0.5967 | 14.1532 | 0.3913 | 11.7799 | 0.9994 | 0.7507 |
| 1.0000 | 271 | 4 | 0.7673 | 13.8963 | 0.6001 | 11.6305 | 0.9989 | 0.7596 |

Training data is subsampled **by cell**, keeping the earliest cycles, so each point remains a valid temporal split.

## 9. Figures

- `figures/eda/01_capacity_degradation.png` — 01 capacity degradation
- `figures/eda/02_signal_trends.png` — 02 signal trends
- `figures/eda/03_current_and_temperature.png` — 03 current and temperature
- `figures/eda/04_distributions.png` — 04 distributions
- `figures/eda/05_outlier_analysis.png` — 05 outlier analysis
- `figures/eda/06_correlation_matrix.png` — 06 correlation matrix
- `figures/eda/07_battery_comparison.png` — 07 battery comparison
- `figures/eda/08_target_distribution.png` — 08 target distribution
- `figures/explainability/error_by_rul_band_transformer.png` — error by rul band transformer
- `figures/explainability/feature_importance_transformer.png` — feature importance transformer
- `figures/explainability/signal_family_importance_transformer.png` — signal family importance transformer
- `figures/results/learning_curve_transformer.png` — learning curve transformer
- `figures/results/model_comparison.png` — model comparison
- `figures/results/pred_vs_truth_catboost.png` — pred vs truth catboost
- `figures/results/pred_vs_truth_gru.png` — pred vs truth gru
- `figures/results/pred_vs_truth_lightgbm.png` — pred vs truth lightgbm
- `figures/results/pred_vs_truth_linear_regression.png` — pred vs truth linear regression
- `figures/results/pred_vs_truth_lstm.png` — pred vs truth lstm
- `figures/results/pred_vs_truth_random_forest.png` — pred vs truth random forest
- `figures/results/pred_vs_truth_ridge.png` — pred vs truth ridge
- `figures/results/pred_vs_truth_transformer.png` — pred vs truth transformer
- `figures/results/pred_vs_truth_xgboost.png` — pred vs truth xgboost
- `figures/results/residual_analysis_catboost.png` — residual analysis catboost
- `figures/results/residual_analysis_gru.png` — residual analysis gru
- `figures/results/residual_analysis_lightgbm.png` — residual analysis lightgbm
- `figures/results/residual_analysis_linear_regression.png` — residual analysis linear regression
- `figures/results/residual_analysis_lstm.png` — residual analysis lstm
- `figures/results/residual_analysis_random_forest.png` — residual analysis random forest
- `figures/results/residual_analysis_ridge.png` — residual analysis ridge
- `figures/results/residual_analysis_transformer.png` — residual analysis transformer
- `figures/results/residual_analysis_xgboost.png` — residual analysis xgboost
- `figures/results/rul_trajectories_catboost.png` — rul trajectories catboost
- `figures/results/rul_trajectories_gru.png` — rul trajectories gru
- `figures/results/rul_trajectories_lightgbm.png` — rul trajectories lightgbm
- `figures/results/rul_trajectories_linear_regression.png` — rul trajectories linear regression
- `figures/results/rul_trajectories_lstm.png` — rul trajectories lstm
- `figures/results/rul_trajectories_random_forest.png` — rul trajectories random forest
- `figures/results/rul_trajectories_ridge.png` — rul trajectories ridge
- `figures/results/rul_trajectories_transformer.png` — rul trajectories transformer
- `figures/results/rul_trajectories_xgboost.png` — rul trajectories xgboost
- `figures/results/training_history_gru.png` — training history gru
- `figures/results/training_history_lstm.png` — training history lstm
- `figures/results/training_history_transformer.png` — training history transformer

## 10. What these numbers do and do not establish

- The test cells were never seen during training, scaling, feature selection or model choice. The metric is therefore an honest estimate **for cells of this chemistry and format tested on this rig**.
- The cohort is **16 cells**. That is a small sample by any standard; the per-cell table matters more than the aggregate, and the bootstrap interval is computed over rows (which are correlated within a cell) so it *understates* true uncertainty.
- Cells that never reached end of life are excluded rather than imputed. Handling them properly requires survival analysis, which is out of scope here.
- The NASA cells were aged under constant-current profiles in a temperature chamber. Real duty cycles are irregular; transfer to field data is unproven and should be measured, not assumed.

## 11. Reproducibility

```json
{
  "generated_at": "2026-07-31T20:28:44.258019+00:00",
  "python": "3.13.5",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "git_revision": null,
  "packages": {
    "numpy": "2.3.1",
    "pandas": "2.3.2",
    "sklearn": "1.8.0",
    "xgboost": "3.3.0",
    "lightgbm": "4.7.0",
    "catboost": "1.2.10",
    "torch": "2.11.0",
    "optuna": "4.9.0"
  },
  "seed": 42,
  "config": "battery_rul_v1"
}
```

| stage | seconds |
| --- | --- |
| build_partitions | 0.2160 |
| eda_figures | 1.2170 |
| explainability | 4.1860 |
| fit:catboost | 1.3100 |
| fit:gru | 8.1720 |
| fit:lightgbm | 0.8940 |
| fit:linear_regression | 0.0020 |
| fit:lstm | 2.5990 |
| fit:random_forest | 0.1650 |
| fit:ridge | 0.0010 |
| fit:transformer | 4.0670 |
| fit:xgboost | 0.5120 |
| learning_curve | 26.3960 |
| result_figures | 4.2910 |

