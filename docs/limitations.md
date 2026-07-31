# Known Limitations & Future Work

Written plainly. A portfolio project that lists no limitations is either
untested or dishonest.

## 1. Data limitations

### 1.1 Eight cells is a small sample
The modelling cohort is 5 cells out of 34, 520 labelled rows. A single
battery-holdout split therefore puts **one** cell in the test partition, and the
metric swings on which cell is drawn — §8.2 of the README shows Ridge finishing
last on validation and first on test in the same run.

**Mitigation:** the headline is leave-one-battery-out cross-validated over all
five cells, and the per-fold spread (σ ≈ 2.3 cycles MAE) is reported alongside
the mean as the real uncertainty. The bootstrap interval is also reported but
resamples **rows**, which are correlated within a cell, so it understates the
spread and should not be quoted on its own. **Fix:** more cells — CALCE, Oxford,
Stanford/Toyota (which has 124).

### 1.2 Right-censored cells are dropped, not modelled
Five cells never reach 70 % SoH; they are excluded. In any real fleet, *most*
cells are healthy and censored, so this discards exactly the population you would
be monitoring. Training on a censored label teaches the model to predict the
experiment's end date.

**Fix:** survival analysis — Cox proportional hazards, DeepSurv, or a
censoring-aware loss. This is the single largest methodological gap.

### 1.3 The cohort gates are a modelling decision
`min_start_soh`, `min_fade_fraction`, `min_labelled_cycles` and
`truncate_at_collapse` remove 29 cells.
Each exclusion is principled and logged, but they collectively define which
population the reported metric describes. A reader who wants the cold-chamber
cells included should set `data.eol_reference: initial`.

### 1.4 One chemistry, one format, one rig
LCO 18650 cells, chamber-cycled with constant-current profiles, 2008-era
instrumentation. Real duty cycles are irregular and partial; calendar ageing is
largely absent. **Nothing here has been shown to transfer to field data.**

### 1.5 Measurement noise sets a floor
NASA capacity readings carry a few percent noise plus genuine rest-recovery
bumps. Trailing-median smoothing handles this causally but lags by a few cycles,
which propagates into the EOL label.

## 2. Methodological limitations

### 2.1 No uncertainty quantification
The model emits a point estimate. For a maintenance decision, "80 ± 30 cycles" is
a far more useful output than "80".

**Fix:** quantile regression (LightGBM supports it natively), conformal
prediction for distribution-free intervals, or MC-dropout / deep ensembles for
the neural models. This is the highest-value next addition.

### 2.2 The EOL threshold is a convention, not a measurement
70 % SoH is the automotive second-life convention. Real end-of-life is
application-defined. The threshold is configurable, but every number in the
report is conditional on it.

### 2.3 Early-life RUL is close to unpredictable
A fresh cell looks almost identical whether it will last 120 or 160 cycles;
degradation only becomes visible after a knee. The error-by-band figure shows
this directly. The literature's usual remedy — a piecewise-linear target cap —
is implemented (`target.cap_at`) but off by default, because it improves the
metric partly by making the problem easier rather than the model better.

### 2.4 Model selection is high-variance
Selection uses validation RMSE over a **single** cell (B0006). This is not a
theoretical concern: in the reported run Ridge is *last* on validation and
*first* on test, while the Transformer is first on validation and second on test.
The cross-validated headline sidesteps the problem for *reporting*, but the
champion that gets persisted to `models/trained_model.pkl` is still chosen on one
cell. Nested cross-validation over cells is the proper fix and is affordable at
this dataset size.

### 2.5 Explanations are confounded by collinearity
SHAP and permutation importance distribute credit among correlated features. The
report aggregates into physical signal families to compensate, but individual
feature rankings should not be over-read.

### 2.6 Sequence models are not SHAP-explained
Faithful attribution for windowed inputs needs DeepExplainer over 3-D tensors.
Currently only permutation importance covers them — so the champion is the
*least* explained model in the zoo.

### 2.7 Tuning is off by default
`configs/default.yaml` uses hand-set hyperparameters. `configs/tuned.yaml` runs
Optuna over the boosting models; the neural models have search spaces defined but
are not tuned by default because each trial trains a network.

## 3. Engineering limitations

### 3.1 No serving layer
`predict.py` is a batch entry point, not an API. No REST/gRPC service, no
containerisation, no model registry, no CI.

### 3.2 No monitoring or drift detection
Deliberately out of scope for milestone 1 — planned for milestone 5.

### 3.3 Single-threaded OpenMP
`_compat.py` pins `OMP_NUM_THREADS=1` to avoid a duplicate-runtime segfault on
the reference environment. Costless at this dataset size; would matter at scale.
The real fix is a clean environment with one OpenMP runtime.

### 3.4 Interim cache is not keyed by config
`data/interim/cycles_<source>.parquet` is reused across runs. Changing a *data*
config (e.g. `min_start_soh`) without clearing the cache reuses the old table.
`data.cache_interim: false` avoids this; a config hash in the filename would fix
it properly.

### 3.5 Feature generation is not incremental
Every run rebuilds ~700 features from scratch. Fine at 600 rows; a bottleneck at
fleet scale.

## 4. What would most improve this work

Ranked by expected value:

1. **Uncertainty quantification** — conformal prediction over the champion.
   Cheap, distribution-free, and turns the output into something actionable.
2. **More cells** — the Stanford/Toyota dataset (124 cells) would change what
   claims are supportable more than any modelling change.
3. **Survival modelling** for censored cells.
4. **Nested cross-validation over cells**, so model selection is not a coin flip.
5. **Physics-informed constraints** — RUL must decrease monotonically within a
   cell; the current models are free to violate this and sometimes do.
6. **Transfer learning across chemistries** — pretrain on NASA, fine-tune on a
   handful of target cells.
7. **SHAP for sequence models** via DeepExplainer.
8. **Multi-horizon output** — predict RUL at several SoH thresholds at once.

## 5. Scope explicitly deferred

Per the milestone plan, none of the following is implemented here: the Digital
Twin, the fleet dashboard, failure-risk classification, maintenance
recommendation, and MLOps monitoring. The architecture anticipates them —
`chronological` splitting and the per-cell prediction path exist precisely so
milestone 2 can build on them — but no partial implementation is included.
