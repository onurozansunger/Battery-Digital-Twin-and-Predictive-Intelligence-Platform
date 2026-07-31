# Battery Digital Twin & Predictive Intelligence Platform

**Milestone 1 — Remaining Useful Life (RUL) prediction for lithium-ion cells.**

An end-to-end, production-style machine-learning pipeline that predicts how many
discharge cycles a lithium-ion battery has left before it reaches end of life,
built on the NASA Ames PCoE battery aging dataset.

```bash
pip install -e ".[dev]"
python scripts/download_data.py
python scripts/run_pipeline.py --config configs/default.yaml
```

---

## Contents

1. [Business problem](#1-business-problem)
2. [What this repository actually claims](#2-what-this-repository-actually-claims)
3. [Dataset](#3-dataset)
4. [Target definition](#4-target-definition)
5. [Pipeline](#5-pipeline)
6. [Preventing leakage](#6-preventing-leakage)
7. [Models](#7-models)
8. [Results](#8-results)
9. [Explainability](#9-explainability)
10. [Reproducing everything](#10-reproducing-everything)
11. [Repository structure](#11-repository-structure)
12. [Configuration](#12-configuration)
13. [Testing](#13-testing)
14. [Limitations](#14-limitations)
15. [Roadmap](#15-roadmap)

---

## 1. Business problem

A lithium-ion cell loses capacity every cycle. Below roughly 70–80 % of rated
capacity it is considered end-of-life for its primary application. Knowing *when*
a cell will get there, from data you already collect, is worth money and safety in
three places:

| Domain | The decision this informs |
|---|---|
| Electric vehicles | Warranty reserves, battery-health disclosure at resale, when to flag a pack for service |
| Grid storage | Replacement scheduling and capacity procurement across thousands of modules |
| Second-life resale | Whether a retired EV pack has enough life left to be worth refurbishing |
| Manufacturing QA | Catching a bad production lot from early-cycle behaviour |

Getting it wrong costs in both directions. Retire a cell too early and you throw
away usable capacity and create avoidable e-waste. Retire it too late and you run
a degraded cell past its safe operating window.

The question this milestone answers is deliberately the hard one:

> Given a battery cell we have **never seen before**, and its charge/discharge
> history so far, how many cycles remain before it drops below 70 % of rated
> capacity?

## 2. What this repository actually claims

Two numbers, kept separate throughout:

* **MAE ≈ 11 cycles, R² ≈ 0.79** on two *entirely unseen cells* under a
  battery-holdout split. This is the deployment-relevant number.
* Under a chronological split — where the model has already seen each test cell's
  early life — the numbers are much better. That is a *different, easier
  question*, and it is reported separately (`configs/chronological.yaml`) rather
  than presented as the headline.

Random row-level splitting is not used anywhere. On this data it produces R²
above 0.99 and means nothing: consecutive cycles of one cell are near-duplicates,
so a random split lets a model interpolate between neighbouring rows instead of
forecasting.

The cohort is **8 cells**. That is a small sample and the per-cell breakdown in
`reports/evaluation_report.md` carries more information than any aggregate.
`docs/limitations.md` is not an afterthought — read it before quoting a number.

## 3. Dataset

NASA Ames Prognostics Data Repository, "Battery Data Set" (Saha & Goebel, 2007):
34 commercial 2.0 Ah 18650 cells cycled to failure in a temperature chamber, with
interleaved electrochemical impedance sweeps.

The loader normalises the raw MATLAB structs into **one row per discharge cycle**,
enriched with summary statistics of that discharge, of the most recent preceding
charge, and of the most recent preceding impedance sweep. Nothing is ever
back-filled from a future step.

Of the 34 cells, 8 pass the cohort gates (too short, begins already degraded,
never degrades, right-censored, too few labelled cycles). Every exclusion is
mechanical and logged — the config asks for *all* cells and the gates select the
cohort deterministically. Full accounting in **[`docs/dataset_card.md`](docs/dataset_card.md)**.

Adding CALCE, Oxford or Stanford later means writing one `BatterySource`
subclass. No other file changes.

## 4. Target definition

```
RUL(k) = k_EOL − k          [discharge cycles]
```

`k_EOL` is the first **persistent** cycle at which trailing-median-smoothed
capacity falls to or below `0.70 × 2.0 Ah = 1.40 Ah`.

Three decisions worth stating explicitly:

* **"Persistent"** means the crossing holds for three consecutive cycles.
  Lithium-ion cells recover capacity after rest periods, so a single dip below
  threshold is routine; taking the first bare crossing systematically
  under-estimates life.
* **Cycles, not calendar time.** The rig ran cells with long idle gaps, so
  wall-clock age reflects lab scheduling rather than physics.
* **The smoother is trailing, never centred.** A centred median would read future
  cycles into the present — the label itself would leak.

## 5. Pipeline

```
raw .mat  →  canonical schema  →  validate  →  label  →  features  →  split
                                                                        │
                                          ┌─────────────────────────────┘
                                          ▼
                        (optional Optuna)  →  train zoo  →  select champion
                                                                        │
                                          ┌─────────────────────────────┘
                                          ▼
                            evaluate + explain  →  report  →  predict
```

Five executable stages, each independently runnable:

| Stage | Command | Produces |
|---|---|---|
| 1. Prepare | `python scripts/prepare_data.py` | `data/processed/{dataset,cycles}.parquet`, `manifest.json` |
| 1b. Tune | `python scripts/tune.py --config configs/tuned.yaml` | `reports/tuning.json` |
| 2. Train | `python scripts/train.py` | `models/trained_model.pkl`, `models/feature_pipeline.pkl`, `reports/metrics.json` |
| 3. Evaluate | `python scripts/evaluate.py` | `figures/**`, `reports/evaluation_report.md` |
| 4. Predict | `python scripts/predict.py` | `reports/predictions.csv` |
| All | `python scripts/run_pipeline.py` | everything above |

**No notebook contains unique logic.** Notebooks in `notebooks/` call the same
`src/` functions the pipeline does, so what they demonstrate is what runs.

Full diagram and module map: **[`docs/architecture.md`](docs/architecture.md)**.

### Feature engineering

~700 features generated from 14 base signals, pruned to ~400 by an unsupervised
filter, then reduced to the top 80 by supervised selection fitted on training rows
only:

rolling mean/std/min/max/range/deviation · EWM · lags · differences · percentage
changes · trailing OLS slopes · ratio and delta vs beginning of life · expanding
statistics · cumulative energy throughput · resistance–capacity interactions ·
temperature excess over ambient · discharge/charge time ratio · cycle-position
transforms.

## 6. Preventing leakage

This is what the codebase is built around. Three distinct boundaries:

**Temporal, within a cell.** Every feature at cycle *k* is a function of cycles
≤ *k* of that cell only. All windows trail; there is no `shift(-n)` anywhere;
capacity smoothing is a trailing median.

`assert_no_leakage()` proves it mechanically on every run: it rebuilds the
features on a truncated history and requires bit-identical values for the rows
present in both. If any feature peeked forward, deleting the future would change
it. The test suite also feeds the checker a deliberately non-causal builder and
asserts that it *fails* — a guard never observed to fail proves nothing.

**Between cells.** Features are computed inside `groupby("battery_id")`. The
default split holds out whole cells.

**Train → validation/test.** Scaler statistics and supervised feature selection
are fitted on training rows only, and re-fitted **inside every Optuna CV fold**.

There is also a subtler trap the code handles explicitly: pruning thresholds are
data-dependent, so a serving batch would prune a *different* column set than
training did. `predict.py` therefore generates features unpruned and lets the
fitted pipeline select the exact training columns — and a test asserts the
serving path reproduces training-time predictions numerically.

## 7. Models

Nine estimators, all given identical features, splits and seeds:

| Family | Models |
|---|---|
| Linear | Linear Regression, Ridge (+ ElasticNet, SVR available) |
| Tree ensembles | Random Forest, XGBoost, LightGBM, CatBoost (+ Gradient Boosting) |
| Sequence | LSTM, GRU, Transformer encoder |

The three neural models share one training loop, so the comparison is
apples-to-apples: same windows, scaler, optimiser, early-stopping rule and seed.
They differ only in their encoder.

Hyperparameter optimisation uses Optuna with battery-grouped CV. Search spaces
live in `src/battery_rul/models/search_spaces.py`, versioned with the code, so a
study is reproducible from a git revision.

## 8. Results

Test cells **B0005** and **B0034** — never seen during training, scaling, feature
selection or model choice. Champion selected on *validation* cells B0006/B0042.

| Rank | Model | MAE | RMSE | MAPE | R² | α-λ (20 %) | ≤10 cycles |
|---|---|---|---|---|---|---|---|
| 1 | **Transformer** | **11.00** | **12.99** | 69.5 % | **0.790** | 0.378 | 51.9 % |
| 2 | GRU | 12.52 | 14.64 | 66.2 % | 0.733 | 0.269 | 41.0 % |
| 3 | Ridge | 14.12 | 17.49 | 67.8 % | 0.718 | 0.454 | 41.2 % |
| 4 | LSTM | 14.92 | 18.48 | 91.5 % | 0.575 | 0.282 | 47.4 % |
| 5 | Linear Regression | 19.70 | 20.19 | 96.9 % | 0.625 | 0.139 | 5.7 % |
| 6 | Random Forest | 17.37 | 20.68 | 116.4 % | 0.606 | 0.351 | 35.1 % |
| 7 | CatBoost | 17.81 | 23.35 | 123.5 % | 0.498 | 0.418 | 43.3 % |
| 8 | XGBoost | 22.87 | 25.17 | 192.0 % | 0.417 | 0.186 | 13.9 % |
| 9 | LightGBM | 23.19 | 25.49 | 187.5 % | 0.402 | 0.170 | 13.9 % |

Errors are in **cycles**. MAPE uses a denominator floored at 1 cycle, because RUL
legitimately reaches zero at end of life.

### Like-for-like

The table above compares models on **different row counts**: sequence models
cannot score a cell's first 19 cycles, and those early rows are the hardest ones.
That difference alone can reorder a ranking, so the pipeline also emits a table
restricted to the 156 rows every model can score:

| Rank | Model | MAE | RMSE | R² | Bias | α-λ (20 %) |
|---|---|---|---|---|---|---|
| 1 | **Transformer** | **11.00** | **12.99** | **0.790** | −6.76 | 0.378 |
| 2 | Ridge | 11.84 | 14.19 | 0.749 | **−0.99** | 0.474 |
| 3 | GRU | 12.52 | 14.64 | 0.733 | −7.02 | 0.269 |
| 4 | Random Forest | 15.48 | 18.40 | 0.579 | +3.32 | 0.372 |
| 5 | LSTM | 14.92 | 18.48 | 0.575 | −7.25 | 0.282 |
| 6 | CatBoost | 15.47 | 19.05 | 0.548 | −0.24 | 0.397 |
| 7 | Linear Regression | 19.21 | 19.76 | 0.514 | −3.94 | 0.115 |
| 8 | LightGBM | 21.85 | 24.27 | 0.267 | +9.30 | 0.212 |
| 9 | XGBoost | 21.89 | 24.31 | 0.264 | +9.46 | 0.205 |

On equal footing **Ridge moves from 3rd to 2nd and is by far the best-calibrated
model** — a bias of −1.0 cycles against the Transformer's −6.8. If you needed an
unbiased estimate rather than the lowest RMSE, Ridge is the better choice, and
its 80-feature linear form is also the easiest to defend to a reliability
engineer. That trade-off is invisible in the headline table.

### Four things these tables are really saying

**Gradient boosting loses, and that is informative.** Trees extrapolate by
returning a constant outside their training range, and each held-out cell sits at
a slightly different capacity scale. The linear and recurrent models extrapolate;
the trees cannot. Swap to `configs/chronological.yaml` — where the model has seen
each test cell's early life — and the ranking reverses. The gap between those two
rankings is a better description of the problem than either number alone.

**The neural models under-predict systematically** (bias ≈ −7 cycles), while the
regularised linear model is nearly unbiased. For maintenance planning
under-prediction is the safe direction, but it is a systematic error, not a
safety feature — and it means the champion's advantage is in variance, not
calibration.

**α-λ accuracy is low (38 %) even for the champion.** The relative error cone
tightens as RUL → 0; near end of life ±20 % is two or three cycles. The model is
useful for "roughly how long left", not for "swap it on Tuesday".

**Error grows with remaining life** — 7.4 cycles MAE at RUL 25–50, 28.8 at RUL
100+. A fresh cell looks nearly identical whether it will last 120 or 160 cycles.
This is the central difficulty of battery prognostics, not a defect of the model.

Per-cell breakdowns, residual diagnostics, learning curves and bootstrap
intervals: **`reports/evaluation_report.md`** (regenerated on every run).

## 9. Explainability

Three independent views — SHAP, permutation importance (computed within-cell so
signal marginals are preserved), and native model importances — plus an error
analysis by remaining-life band.

Importance is **also aggregated into physical signal families**, and that is the
reading to trust. The feature set is deliberately collinear; under collinearity
SHAP and permutation importance distribute credit among correlated features
rather than isolating a cause. Family-level attribution on the default run puts
capacity-derived and current signals well ahead of discharge timing, with
temperature and resistance contributing little.

## 10. Reproducing everything

```bash
# 1. Environment (Python 3.11+; 3.12 recommended)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Data (~209 MB from the NASA PCoE S3 mirror)
python scripts/download_data.py

# 3. Everything: prepare → train → evaluate → predict
python scripts/run_pipeline.py --config configs/default.yaml

# 4. Tests
pytest                          # full suite
pytest -m "not slow"            # skip the real-data parse test
```

Useful variations:

```bash
# One-minute smoke run (reduced zoo, short neural training)
python scripts/run_pipeline.py --config configs/fast.yaml

# No dataset on disk? Runs entirely on the synthetic generator.
python scripts/run_pipeline.py --config configs/synthetic.yaml

# With Optuna hyperparameter search
python scripts/run_pipeline.py --config configs/tuned.yaml

# The forecasting question instead of the cold-start question
python scripts/run_pipeline.py --config configs/chronological.yaml

# Override any config field without editing a file
python scripts/train.py --set models.enabled='[xgboost, lightgbm]' --set seed=7

# Score new cycle data with the persisted champion
python scripts/predict.py --input my_cycles.parquet --output out.csv
```

The CLI is also installed as an entry point: `battery-rul all --config configs/default.yaml`.

**Determinism.** Seed 42 propagates to NumPy, Python, and torch (including cuDNN
determinism). `reports/metrics.json` embeds the git revision, Python version,
platform and every package version used.

### Outputs produced

```
models/trained_model.pkl              champion model
models/feature_pipeline.pkl           fitted scaler + selected columns
models/zoo/*.pkl                      every trained model (so the table is checkable)
reports/metrics.json                  all metrics + full provenance
reports/evaluation_report.md          the written report
reports/model_comparison.csv          comparison table
reports/model_comparison_common_rows.csv   like-for-like comparison
reports/predictions_test.{csv,parquet}
reports/explainability.json
reports/permutation_importance.csv
reports/learning_curve.csv
reports/tuning.json                   (when tuning is enabled)
figures/eda/*.png                     8 EDA figures
figures/results/*.png                 predictions, residuals, comparison, curves
figures/explainability/*.png          SHAP, importance, error structure
data/processed/manifest.json          exactly what stage 1 did
```

## 11. Repository structure

```
battery-rul-platform/
├── configs/                    default · fast · tuned · chronological · walk_forward · synthetic
├── data/
│   ├── raw/nasa/mat/           NASA .mat files (gitignored)
│   ├── interim/                cached canonical cycle table
│   └── processed/              modelling dataset + manifest
├── docs/
│   ├── architecture.md         design, stage flow, leakage boundaries
│   ├── dataset_card.md         provenance, cohort selection, quality issues
│   ├── model_card.md           intended use, performance, caveats
│   └── limitations.md          known limitations & future work
├── figures/                    eda · results · explainability
├── models/                     champion + feature pipeline + full zoo
├── notebooks/                  01 EDA · 02 features & target · 03 model comparison
├── reports/                    metrics.json, evaluation_report.md, tables
├── scripts/                    thin CLI wrappers, one per stage
├── src/battery_rul/
│   ├── config.py               typed, validated configuration
│   ├── _compat.py              native library load ordering
│   ├── cli.py
│   ├── data/                   schema · base · nasa · synthetic · validation · loader
│   ├── features/               target · engineering · pipeline · splitting · sequences
│   ├── models/                 base · classical · neural · search_spaces
│   ├── evaluation/             metrics · evaluator · reporting
│   ├── explainability/
│   ├── visualization/          style · eda · results
│   ├── pipelines/              prepare_data · tune · train · evaluate · predict · run_pipeline
│   └── utils/                  logging · seed · io · timing
├── tests/                      147 tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 12. Configuration

Everything is configurable and nothing is hardcoded — no dataset paths, no
learning rates, no window sizes buried in code. Configuration is a validated
pydantic model (`src/battery_rul/config.py`) with `extra="forbid"`, so a typo in
YAML raises rather than silently taking a default.

Configs support `extends:` inheritance (chains and all), and every field is
reachable from the command line:

```bash
python scripts/train.py \
  --set models.training.epochs=200 \
  --set features.rolling_windows='[5, 10, 25]' \
  --set split.strategy=walk_forward
```

## 13. Testing

```bash
pytest                    # 147 tests, ~25 s
pytest --cov=battery_rul  # with coverage
ruff check . && black --check .
```

Coverage spans data loading and schema coercion, the validation gate, target
generation and EOL detection, feature engineering, **the causality guarantees**,
all three splitting strategies, metrics, every model's fit/predict/persist cycle,
and the end-to-end pipeline including training/serving consistency.

The suite runs entirely on the synthetic generator, so it needs no dataset
download; the one test that parses real NASA files is marked `slow` and skips
cleanly when they are absent.

## 14. Limitations

Summarised — the full treatment is **[`docs/limitations.md`](docs/limitations.md)**.

* **8 cells, 2 test cells.** Small. Per-cell metrics matter more than aggregates,
  and the bootstrap interval understates true uncertainty (rows within a cell are
  correlated).
* **Right-censored cells are excluded, not modelled.** In a real fleet most cells
  are healthy and censored — this discards exactly the population you would
  monitor. Survival analysis is the fix and the largest methodological gap.
* **No uncertainty quantification.** The model emits a point estimate; a
  maintenance decision wants an interval. Conformal prediction is the obvious
  next step.
* **One chemistry, one format, one rig.** LCO 18650, chamber-cycled, 2008-era
  instrumentation. Transfer to field data is unproven.
* **Early-life RUL is close to unpredictable** and the metrics reflect that.
* **The champion is selected on two validation cells**, so model selection is
  itself high-variance.
* **Not a serving system.** Batch inference only — no API, container, registry or
  drift monitoring.

## 15. Roadmap

Milestone 1 (this repository) covers RUL prediction only. Deliberately **not**
implemented: the Digital Twin, fleet dashboard, failure-risk classification,
maintenance recommendation, and MLOps monitoring.

**Milestone 2 — Battery Digital Twin.** A stateful per-cell object that ingests
cycles as they arrive and maintains a live estimate of state of health, RUL, and
a forward capacity-fade trajectory with uncertainty bands. The groundwork is
already here: `configs/chronological.yaml` frames the forecasting question,
`RULPredictor` scores incrementally, and the causal feature contract means a twin
can be updated cycle by cycle without recomputing history.

---

## Citation

```bibtex
@misc{saha2007battery,
  author       = {Saha, B. and Goebel, K.},
  title        = {Battery Data Set},
  year         = {2007},
  publisher    = {NASA Ames Prognostics Data Repository},
  howpublished = {NASA Ames Research Center, Moffett Field, CA}
}
```

## Licence

MIT for the code. The NASA dataset is US-government work, redistributed by NASA
for research use; please cite the original authors.
