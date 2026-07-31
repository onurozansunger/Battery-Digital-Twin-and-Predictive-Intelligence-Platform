# Evaluation Report — battery_rul_v1

_Generated 2026-07-31 20:41 UTC_

> NASA li-ion remaining-useful-life baseline. Battery-holdout split, causal feature engineering, nine models compared, champion selected on validation.

## 1. Headline result

**transformer** is the champion model, selected by `rmse` on the **validation** partition and reported here on the untouched **test** partition.

- **MAE** — 11.62 cycles
- **RMSE** — 13.82 cycles (95 % CI 12.19–15.42)
- **R²** — 0.784
- **MAPE** — 62.4 % (denominator floored at 1 cycle)
- **α-λ accuracy (α=20%)** — 39.8% of predictions inside the relative error cone
- **Predictions within 10 cycles** — 48.5%
- **Bias** — -1.19 cycles (conservative)
- **Prognostic horizon** — not reached: predictions never settle inside the ±20% relative cone and stay there. The cone tightens to a couple of cycles near end of life, which is a demanding bar at this error level.

## 2. Experimental setup

### 2.1 Dataset

| battery_id | n_cycles | capacity_start_ah | capacity_end_ah | soh_end | eol_cycle | reaches_eol | ambient_c |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 168 | 1.8565 | 1.2935 | 0.6467 | 127 | True | 24 |
| B0006 | 168 | 2.0353 | 1.1644 | 0.5822 | 111 | True | 24 |
| B0007 | 168 | 1.8911 | 1.4063 | 0.7032 | — | False | 24 |
| B0018 | 132 | 1.8550 | 1.3548 | 0.6774 | 99 | True | 24 |
| B0029 | 40 | 1.6975 | 1.6275 | 0.8138 | — | False | 43 |
| B0030 | 40 | 1.6561 | 1.5787 | 0.7894 | — | False | 43 |
| B0032 | 40 | 1.7049 | 1.6634 | 0.8317 | — | False | 43 |
| B0033 | 132 | 1.7132 | 1.3704 | 0.6852 | 106 | True | 24 |
| B0034 | 196 | 1.6623 | 1.3120 | 0.6560 | 77 | True | 24 |
| B0036 | 194 | 1.8011 | 1.5795 | 0.7898 | — | False | 24 |
| B0042 | 40 | 1.7287 | 1.5880 | 0.7940 | — | False | 22 |
| B0043 | 40 | 1.7138 | 1.4853 | 0.7427 | — | False | 22 |
| B0044 | 40 | 1.6865 | 1.4313 | 0.7157 | — | False | 22 |
| B0046 | 69 | 1.7282 | 1.1746 | 0.5873 | 19 | True | 4 |
| B0047 | 69 | 1.6743 | 1.1774 | 0.5887 | 12 | True | 4 |
| B0048 | 69 | 1.6580 | 1.2520 | 0.6260 | 18 | True | 4 |

### 2.2 Target definition

RUL(k) = k_EOL − k, where k_EOL is the first **persistent** cycle at which trailing-median-smoothed capacity falls to or below **1.400 Ah** (70% of the nominal reference capacity).

- Labelled rows: **520**
- RUL range: **0.0 – 126.0 cycles**
- Mean RUL: **52.785 cycles**
- Excluded as right-censored (never reached EOL): **B0007, B0029, B0030, B0032, B0036, B0042, B0043, B0044**

End-of-life cycle per cell:

| battery_id | eol_cycle |
| --- | --- |
| B0005 | 127 |
| B0006 | 111 |
| B0018 | 99 |
| B0033 | 106 |
| B0034 | 77 |

### 2.3 Split

**Strategy:** `battery_holdout` — Entire cells held out. Test cells were never seen during training, scaling or feature selection.

- train: **267** rows, cells `['B0018', 'B0033', 'B0034']`
- val: **106** rows, cells `['B0006']`
- test: **122** rows, cells `['B0005']`

No random row-level splitting is used anywhere in this project. Consecutive cycles of one cell are near-duplicates, so a random split lets a model interpolate between neighbouring rows and produces an R² that means nothing.

### 2.4 Features

- Generated: **709** causal features from 14 base signals
- After unsupervised pruning: **392**
- After supervised top-K selection (train partition only): **80**
- Scaler: `robust`
- Warm-up rows dropped: **25**

Every feature at cycle *k* is a function of cycles ≤ *k* of the same cell. This is verified mechanically by `assert_no_leakage`, which rebuilds the features on a truncated history and requires bit-identical values.

## 3. Data quality

- Rows in: **2280** → out: **1605**
- Cells in: **26** → out: **16**

| check | severity | message |
| --- | --- | --- |
| capacity_jump | warning | 7 cycles move capacity by more than 0.70 Ah in one step (rig glitch) |
| cohort_gates | info | Excluded 10 cell(s): B0031 (fades only 0.0% (< 2.0%)); B0038 (starts at 0.55 SoH (< 0.8)); B0039 (starts at 0.24 SoH (< 0.8)); B0040 (starts at 0.40 SoH (< 0.8)); B0041 (starts at 0.03 SoH (< 0.8)); B0045 (starts at 0.54 SoH (< 0.8)); B0053 (starts at 0.53 SoH (< 0.8)); B0054 (starts at 0.58 SoH (< 0.8)); B0055 (starts at 0.66 SoH (< 0.8)); B0056 (starts at 0.67 SoH (< 0.8)) |

## 4. Model comparison (test partition)

| rank | model | n | mae | rmse | mape | smape | r2 | median_ae | max_error | bias | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon | n_unscored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ridge | 122 | 10.8324 | 13.1938 | 96.3736 | 31.3107 | 0.8596 | 8.8119 | 24.8518 | 3.6039 | 0.5902 | 0.5492 | 1.0000 | — | 0 |
| 2 | transformer | 103 | 11.6218 | 13.8230 | 62.3728 | 33.2496 | 0.7839 | 10.0415 | 28.4106 | -1.1937 | 0.3981 | 0.4854 | 0.8835 | — | 19 |
| 3 | gru | 103 | 15.1018 | 16.8616 | 110.9498 | 42.1779 | 0.6784 | 17.1571 | 30.9465 | 3.4151 | 0.3495 | 0.2718 | 0.8932 | — | 19 |
| 4 | random_forest | 122 | 19.1949 | 23.5063 | 139.0120 | 41.8598 | 0.5545 | 17.4770 | 51.2586 | -5.1479 | 0.3607 | 0.3279 | 0.6557 | — | 0 |
| 5 | lightgbm | 122 | 20.7673 | 23.6170 | 129.2976 | 45.1058 | 0.5503 | 19.9472 | 44.0809 | -4.3400 | 0.2377 | 0.2131 | 0.6885 | — | 0 |
| 6 | lstm | 103 | 19.4587 | 23.6505 | 53.2679 | 46.6317 | 0.3673 | 18.7201 | 49.3845 | -6.3232 | 0.2427 | 0.2913 | 0.7379 | 0 | 19 |
| 7 | xgboost | 122 | 21.2152 | 24.2447 | 135.5327 | 45.8236 | 0.5261 | 21.3969 | 46.8821 | -4.5488 | 0.2459 | 0.2213 | 0.6721 | — | 0 |
| 8 | catboost | 122 | 21.8494 | 27.1882 | 161.8206 | 45.6278 | 0.4040 | 19.2482 | 60.2177 | -6.2127 | 0.3443 | 0.3361 | 0.5656 | — | 0 |
| 9 | linear_regression | 122 | 31.1265 | 33.7961 | 68.1172 | 121.2681 | 0.0791 | 31.2931 | 53.0000 | -31.1265 | 0.0082 | 0.0902 | 0.3197 | 0 | 0 |

`n_unscored` counts rows the model could not score. Sequence models need a full window of history, so the first *w−1* cycles of every test cell are unscoreable by construction — they are reported, never silently dropped.

### 4.1 Like-for-like: rows every model can score

The table above compares models on different row counts, and the rows the sequence models skip are the early-life ones — the hardest. That difference alone can reorder a ranking. This table restricts every model to the intersection, so the ordering reflects the models rather than their input requirements.

| rank | model | n | mae | rmse | mape | r2 | bias | alpha_lambda | within_10_cycles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ridge | 103 | 9.1862 | 11.3476 | 110.9273 | 0.8543 | 7.9130 | 0.5728 | 0.6311 |
| 2 | transformer | 103 | 11.6218 | 13.8230 | 62.3728 | 0.7839 | -1.1937 | 0.3981 | 0.4854 |
| 3 | gru | 103 | 15.1018 | 16.8616 | 110.9498 | 0.6784 | 3.4151 | 0.3495 | 0.2718 |
| 4 | random_forest | 103 | 15.1483 | 18.3390 | 157.9083 | 0.6195 | 1.4900 | 0.4272 | 0.3883 |
| 5 | lightgbm | 103 | 17.5996 | 19.7851 | 146.8908 | 0.5572 | 1.8580 | 0.2816 | 0.2524 |
| 6 | xgboost | 103 | 17.8983 | 20.2115 | 154.0777 | 0.5379 | 1.8425 | 0.2913 | 0.2621 |
| 7 | catboost | 103 | 17.1650 | 21.2902 | 183.9289 | 0.4872 | 1.3562 | 0.4078 | 0.3981 |
| 8 | lstm | 103 | 19.4587 | 23.6505 | 53.2679 | 0.3673 | -6.3232 | 0.2427 | 0.2913 |
| 9 | linear_regression | 103 | 31.5050 | 34.4897 | 75.9315 | -0.3456 | -31.5050 | 0.0097 | 0.1068 |

### 4.2 Leave-one-battery-out cross-validation

The cohort is 5 cells, so the single holdout above puts **one** cell in the test partition — one sample. Leave-one-battery-out holds out each cell in turn, re-fitting the feature pipeline inside every fold, and pools the out-of-fold predictions. It uses every row for evaluation instead of a fifth of them, and the spread across folds is a far more honest uncertainty statement than a bootstrap over correlated rows.

**Pooled (transformer):** MAE 8.06 · RMSE 9.93 · R² 0.850 · bias -1.84 cycles

**Spread across folds:** MAE σ = 2.34, RMSE σ = 2.45 cycles. Read that as the real uncertainty on the headline number.

| battery_id | n | mae | rmse | mape | r2 | bias | alpha_lambda |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 103 | 6.6578 | 8.8548 | 22.8916 | 0.9113 | -4.6418 | 0.8252 |
| B0006 | 87 | 6.5184 | 8.1956 | 32.2549 | 0.8935 | -2.4022 | 0.6092 |
| B0018 | 75 | 8.4739 | 9.7658 | 82.5105 | 0.7965 | 8.4569 | 0.3733 |
| B0033 | 82 | 11.9864 | 13.7122 | 74.0944 | 0.6644 | -8.3727 | 0.1951 |
| B0034 | 53 | 6.6583 | 7.4569 | 84.7309 | 0.7624 | 0.0850 | 0.3396 |

## 5. Per-cell breakdown

With only a handful of held-out cells, the aggregate number can hide a cell the model gets badly wrong. This table is the honest view of the result.

### catboost

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 21.8494 | 27.1882 | 19.2482 | 60.2177 | 161.8206 | 45.6278 | 0.4040 | -6.2127 | 26.4688 | 0.3443 | 0.3361 | 0.5656 | — |

### gru

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 103 | 15.1018 | 16.8616 | 17.1571 | 30.9465 | 110.9498 | 42.1779 | 0.6784 | 3.4151 | 16.5121 | 0.3495 | 0.2718 | 0.8932 | — |

### lightgbm

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 20.7673 | 23.6170 | 19.9472 | 44.0809 | 129.2976 | 45.1058 | 0.5503 | -4.3400 | 23.2148 | 0.2377 | 0.2131 | 0.6885 | — |

### linear_regression

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 31.1265 | 33.7961 | 31.2931 | 53 | 68.1172 | 121.2681 | 0.0791 | -31.1265 | 13.1648 | 0.0082 | 0.0902 | 0.3197 | 0 |

### lstm

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 103 | 19.4587 | 23.6505 | 18.7201 | 49.3845 | 53.2679 | 46.6317 | 0.3673 | -6.3232 | 22.7895 | 0.2427 | 0.2913 | 0.7379 | 0 |

### random_forest

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 19.1949 | 23.5063 | 17.4770 | 51.2586 | 139.0120 | 41.8598 | 0.5545 | -5.1479 | 22.9356 | 0.3607 | 0.3279 | 0.6557 | — |

### ridge

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 10.8324 | 13.1938 | 8.8119 | 24.8518 | 96.3736 | 31.3107 | 0.8596 | 3.6039 | 12.6921 | 0.5902 | 0.5492 | 1 | — |

### transformer

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 103 | 11.6218 | 13.8230 | 10.0415 | 28.4106 | 62.3728 | 33.2496 | 0.7839 | -1.1937 | 13.7714 | 0.3981 | 0.4854 | 0.8835 | — |

### xgboost

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 21.2152 | 24.2447 | 21.3969 | 46.8821 | 135.5327 | 45.8236 | 0.5261 | -4.5488 | 23.8142 | 0.2459 | 0.2213 | 0.6721 | — |

## 6. Residual analysis — champion

| statistic | value |
| --- | --- |
| kurtosis | -0.8046 |
| mean | -1.1937 |
| q05 | -27.1385 |
| q25 | -12.8180 |
| q50 | 6.1702 |
| q75 | 9.9198 |
| q95 | 10.9973 |
| residual_rul_corr | -0.9101 |
| skew | -0.9009 |
| std | 13.7714 |

Residuals correlate with true RUL at ρ = -0.910: the model under-predicts at high RUL. This is the expected signature of regression-to-the-mean on a bounded target and is the main reason early-life predictions should be treated as a range, not a number.

## 7. Learning curve — champion

| fraction | n_train_rows | n_train_batteries | train_rmse | test_rmse | train_mae | test_mae | train_r2 | test_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2500 | 70 | 1 | 1.1438 | 20.1470 | 0.8565 | 17.6630 | 0.9940 | 0.5408 |
| 0.5000 | 145 | 2 | 0.6998 | 18.0330 | 0.5703 | 16.0960 | 0.9980 | 0.6321 |
| 0.7500 | 195 | 2 | 0.8136 | 14.2863 | 0.5926 | 11.5355 | 0.9987 | 0.7691 |
| 1.0000 | 267 | 3 | 0.3799 | 13.5114 | 0.2915 | 12.3359 | 0.9997 | 0.7935 |

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
  "generated_at": "2026-07-31T20:41:30.937265+00:00",
  "python": "3.13.5",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "git_revision": "da858c9",
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
| build_partitions | 0.2080 |
| cross_validate | 68.9230 |
| eda_figures | 1.1810 |
| explainability | 4.1960 |
| fit:catboost | 0.8760 |
| fit:gru | 5.3330 |
| fit:lightgbm | 1.2600 |
| fit:linear_regression | 0.0010 |
| fit:lstm | 1.9890 |
| fit:random_forest | 0.1830 |
| fit:ridge | 0.0010 |
| fit:transformer | 5.2000 |
| fit:xgboost | 0.2130 |
| learning_curve | 26.4470 |
| result_figures | 3.5320 |

