# Evaluation Report — battery_rul_v1

_Generated 2026-08-02 11:42 UTC_

> NASA li-ion remaining-useful-life baseline. Battery-holdout split, causal feature engineering, nine models compared, champion selected on validation.

## 1. Headline result

**gru** is the champion model, selected by `rmse` on the **validation** partition and reported here on the untouched **test** partition.

- **MAE** — 11.69 cycles
- **RMSE** — 14.74 cycles (95 % CI 12.75–16.70)
- **R²** — 0.754
- **MAPE** — 78.4 % (denominator floored at 1 cycle)
- **α-λ accuracy (α=20%)** — 44.7% of predictions inside the relative error cone
- **Predictions within 10 cycles** — 48.5%
- **Bias** — -1.48 cycles (conservative)
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
- After unsupervised pruning: **756**
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
| 1 | transformer | 103 | 12.3701 | 13.9545 | 62.6938 | 35.5091 | 0.7797 | 11.7318 | 30.0885 | 2.0941 | 0.3689 | 0.3689 | 0.9223 | — | 19 |
| 2 | gru | 103 | 11.6925 | 14.7362 | 78.4050 | 33.8933 | 0.7543 | 10.0437 | 35.4693 | -1.4810 | 0.4466 | 0.4854 | 0.8738 | — | 19 |
| 3 | lightgbm | 122 | 12.5683 | 17.0717 | 24.2361 | 22.2010 | 0.7650 | 11.4147 | 46.8038 | -10.5812 | 0.5574 | 0.4836 | 0.8689 | — | 0 |
| 4 | ridge | 122 | 13.8966 | 17.5713 | 92.5482 | 33.7024 | 0.7511 | 11.9709 | 43.8250 | -3.1016 | 0.4836 | 0.4426 | 0.8607 | — | 0 |
| 5 | xgboost | 122 | 13.0358 | 17.9528 | 23.3106 | 22.1666 | 0.7401 | 11.6956 | 49.1791 | -11.7592 | 0.5656 | 0.4672 | 0.8689 | — | 0 |
| 6 | random_forest | 122 | 16.0464 | 20.4775 | 96.5946 | 35.7823 | 0.6619 | 13.0123 | 51.4235 | -7.5321 | 0.4590 | 0.3934 | 0.7951 | — | 0 |
| 7 | linear_regression | 122 | 19.8847 | 21.9848 | 52.8044 | 91.0555 | 0.6103 | 18.2881 | 36.6366 | -19.8847 | 0.3197 | 0.1148 | 0.6803 | 0 | 0 |
| 8 | lstm | 103 | 19.9980 | 22.6810 | 156.6691 | 50.2725 | 0.4181 | 22.2987 | 41.0423 | 2.2653 | 0.2621 | 0.2330 | 0.6019 | — | 19 |
| 9 | catboost | 122 | 19.6466 | 24.6615 | 139.4160 | 42.2342 | 0.5096 | 16.1479 | 58.5149 | -6.4909 | 0.3852 | 0.3525 | 0.6393 | — | 0 |

`n_unscored` counts rows the model could not score. Sequence models need a full window of history, so the first *w−1* cycles of every test cell are unscoreable by construction — they are reported, never silently dropped.

### 4.1 Like-for-like: rows every model can score

The table above compares models on different row counts, and the rows the sequence models skip are the early-life ones — the hardest. That difference alone can reorder a ranking. This table restricts every model to the intersection, so the ordering reflects the models rather than their input requirements.

| rank | model | n | mae | rmse | mape | r2 | bias | alpha_lambda | within_10_cycles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | lightgbm | 103 | 8.6979 | 11.2101 | 23.2308 | 0.8578 | -6.3442 | 0.6602 | 0.5728 |
| 2 | xgboost | 103 | 9.0497 | 12.0746 | 21.9606 | 0.8351 | -7.5376 | 0.6699 | 0.5534 |
| 3 | ridge | 103 | 10.2761 | 12.2636 | 104.1383 | 0.8299 | 2.5102 | 0.5728 | 0.5243 |
| 4 | random_forest | 103 | 11.6315 | 13.8770 | 107.8690 | 0.7822 | -1.5467 | 0.5437 | 0.4660 |
| 5 | transformer | 103 | 12.3701 | 13.9545 | 62.6938 | 0.7797 | 2.0941 | 0.3689 | 0.3689 |
| 6 | gru | 103 | 11.6925 | 14.7362 | 78.4050 | 0.7543 | -1.4810 | 0.4466 | 0.4854 |
| 7 | catboost | 103 | 15.0356 | 18.4814 | 157.8215 | 0.6136 | 0.5469 | 0.4563 | 0.4175 |
| 8 | lstm | 103 | 19.9980 | 22.6810 | 156.6691 | 0.4181 | 2.2653 | 0.2621 | 0.2330 |
| 9 | linear_regression | 103 | 21.1652 | 23.2428 | 60.4272 | 0.3889 | -21.1652 | 0.1942 | 0.1068 |

### 4.2 Leave-one-battery-out cross-validation

The cohort is 5 cells, so the single holdout above puts **one** cell in the test partition — one sample. Leave-one-battery-out holds out each cell in turn, re-fitting the feature pipeline inside every fold, and pools the out-of-fold predictions. It uses every row for evaluation instead of a fifth of them, and the spread across folds is a far more honest uncertainty statement than a bootstrap over correlated rows.

**Pooled (gru):** MAE 11.84 · RMSE 13.94 · R² 0.705 · bias -3.47 cycles

**Spread across folds:** MAE σ = 3.60, RMSE σ = 3.95 cycles. Read that as the real uncertainty on the headline number.

| battery_id | n | mae | rmse | mape | r2 | bias | alpha_lambda |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 103 | 11.8453 | 13.8859 | 28.6421 | 0.7819 | -11.8287 | 0.2913 |
| B0006 | 87 | 10.6097 | 11.7014 | 41.2732 | 0.7829 | 0.2631 | 0.5172 |
| B0018 | 75 | 11.4381 | 11.6601 | 91.5323 | 0.7099 | 11.4381 | 0.3467 |
| B0033 | 82 | 16.7982 | 19.5644 | 41.5883 | 0.3168 | -16.7315 | 0.0122 |
| B0034 | 53 | 6.7335 | 9.1053 | 107.2801 | 0.6457 | 6.0584 | 0.4906 |

## 5. Per-cell breakdown

With only a handful of held-out cells, the aggregate number can hide a cell the model gets badly wrong. This table is the honest view of the result.

### catboost

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 19.6466 | 24.6615 | 16.1479 | 58.5149 | 139.4160 | 42.2342 | 0.5096 | -6.4909 | 23.7920 | 0.3852 | 0.3525 | 0.6393 | — |

### gru

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 103 | 11.6925 | 14.7362 | 10.0437 | 35.4693 | 78.4050 | 33.8933 | 0.7543 | -1.4810 | 14.6616 | 0.4466 | 0.4854 | 0.8738 | — |

### lightgbm

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 12.5683 | 17.0717 | 11.4147 | 46.8038 | 24.2361 | 22.2010 | 0.7650 | -10.5812 | 13.3971 | 0.5574 | 0.4836 | 0.8689 | — |

### linear_regression

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 19.8847 | 21.9848 | 18.2881 | 36.6366 | 52.8044 | 91.0555 | 0.6103 | -19.8847 | 9.3771 | 0.3197 | 0.1148 | 0.6803 | 0 |

### lstm

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 103 | 19.9980 | 22.6810 | 22.2987 | 41.0423 | 156.6691 | 50.2725 | 0.4181 | 2.2653 | 22.5675 | 0.2621 | 0.2330 | 0.6019 | — |

### random_forest

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 16.0464 | 20.4775 | 13.0123 | 51.4235 | 96.5946 | 35.7823 | 0.6619 | -7.5321 | 19.0419 | 0.4590 | 0.3934 | 0.7951 | — |

### ridge

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 13.8966 | 17.5713 | 11.9709 | 43.8250 | 92.5482 | 33.7024 | 0.7511 | -3.1016 | 17.2954 | 0.4836 | 0.4426 | 0.8607 | — |

### transformer

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 103 | 12.3701 | 13.9545 | 11.7318 | 30.0885 | 62.6938 | 35.5091 | 0.7797 | 2.0941 | 13.7964 | 0.3689 | 0.3689 | 0.9223 | — |

### xgboost

| battery_id | n | mae | rmse | median_ae | max_error | mape | smape | r2 | bias | std_residual | alpha_lambda | within_10_cycles | within_25_cycles | prognostic_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 122 | 13.0358 | 17.9528 | 11.6956 | 49.1791 | 23.3106 | 22.1666 | 0.7401 | -11.7592 | 13.5656 | 0.5656 | 0.4672 | 0.8689 | — |

## 6. Residual analysis — champion

| statistic | value |
| --- | --- |
| kurtosis | -0.3611 |
| mean | -1.4810 |
| q05 | -31.3449 |
| q25 | -12.1248 |
| q50 | 3.3690 |
| q75 | 10.0050 |
| q95 | 13.5792 |
| residual_rul_corr | -0.9324 |
| skew | -0.9595 |
| std | 14.6616 |

Residuals correlate with true RUL at ρ = -0.932: the model under-predicts at high RUL. This is the expected signature of regression-to-the-mean on a bounded target and is the main reason early-life predictions should be treated as a range, not a number.

## 7. Learning curve — champion

| fraction | n_train_rows | n_train_batteries | train_rmse | test_rmse | train_mae | test_mae | train_r2 | test_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2500 | 70 | 1 | 1.4495 | 12.3191 | 1.0906 | 7.6888 | 0.9903 | 0.8283 |
| 0.5000 | 145 | 2 | 0.7660 | 13.8182 | 0.5741 | 11.8081 | 0.9976 | 0.7840 |
| 0.7500 | 195 | 2 | 0.5813 | 14.1793 | 0.4510 | 12.1949 | 0.9993 | 0.7726 |
| 1.0000 | 267 | 3 | 0.5058 | 14.2030 | 0.3071 | 12.5921 | 0.9995 | 0.7718 |

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
- `figures/explainability/error_by_rul_band_gru.png` — error by rul band gru
- `figures/explainability/feature_importance_gru.png` — feature importance gru
- `figures/explainability/signal_family_importance_gru.png` — signal family importance gru
- `figures/results/learning_curve_gru.png` — learning curve gru
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
  "generated_at": "2026-08-02T11:42:22.276507+00:00",
  "python": "3.13.5",
  "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
  "git_revision": "cf411df",
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
| build_partitions | 0.4090 |
| cross_validate | 151.7650 |
| eda_figures | 1.2050 |
| explainability | 6.9930 |
| fit:catboost | 0.7490 |
| fit:gru | 5.1570 |
| fit:lightgbm | 3.0720 |
| fit:linear_regression | 0.0020 |
| fit:lstm | 1.4750 |
| fit:random_forest | 0.1690 |
| fit:ridge | 0.0030 |
| fit:transformer | 5.2110 |
| fit:xgboost | 0.5470 |
| learning_curve | 41.1220 |
| nested_comparison | 856.0510 |
| result_figures | 3.6130 |

