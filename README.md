# Battery Digital Twin & Predictive Intelligence Platform

**Milestone 3 — fleet intelligence and production MLOps, on top of a digital twin.**

Milestone 1 predicts remaining useful life. Milestone 2 wraps it in a digital
twin that answers questions about **one cell** — RUL with a prediction interval,
state of health, calibrated end-of-life risk, degradation drivers, data quality
and a rule-based recommendation. Milestone 3 answers questions about **a fleet**,
and adds what running it responsibly needs: monitoring, a model registry, a
promotion gate, persistence, containers and CI.

Built on the NASA Ames PCoE battery aging dataset. Every milestone's interfaces
remain intact — regression tests assert it.

```bash
pip install -e ".[dev]"
python scripts/download_data.py                                    # ~209 MB
python scripts/run_pipeline.py --config configs/default.yaml       # Milestone 1
python -m battery_rul.pipelines.run_milestone_2 --config configs/default.yaml

# Milestone 3
python -m battery_rul.pipelines.build_reference   --config configs/default.yaml
python -m battery_rul.pipelines.run_fleet_batch   --config configs/default.yaml \
    --fleet-id DEMO-FLEET-01 --source demo --demo-size 24
python -m battery_rul.pipelines.run_monitoring    --config configs/default.yaml \
    --fleet-id DEMO-FLEET-01 --source demo --demo-size 24
python -m battery_rul.pipelines.generate_fleet_report --config configs/default.yaml \
    --fleet-id DEMO-FLEET-01

python -m battery_rul.api.app                                # API on :8000
streamlit run src/battery_rul/dashboard/fleet_app.py         # fleet dashboard
streamlit run src/battery_rul/dashboard/app.py               # single-cell dashboard
```

Fifteen-minute tour: [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md).

> **Research prototype for engineering decision support.** Not validated for
> production deployment, not a substitute for battery-management-system
> protection, not suitable for autonomous safety-critical control. The "failure
> risk" label is derived from a capacity threshold — the source dataset contains
> no observed safety failures. Fleet rankings, maintenance priorities and
> replacement horizons are configurable engineering policy, never validated
> against real maintenance outcomes. Read
> [`docs/MILESTONE_3_LIMITATIONS.md`](docs/MILESTONE_3_LIMITATIONS.md) and
> [`docs/MILESTONE_2_LIMITATIONS.md`](docs/MILESTONE_2_LIMITATIONS.md) before
> quoting anything here.

---

## Milestone 3 at a glance

| | |
|---|---|
| **Fleet outputs** | ranked batteries · maintenance priority (P0–P5) with score breakdown and triggered rules · inspection windows · replacement horizons with uncertainty brackets · workload forecast · fleet statistics **with denominators** |
| **Monitoring** | input data quality · feature drift vs a versioned training reference · prediction drift · delayed-label performance — four separate questions, four separate statuses |
| **MLOps** | JSON model registry with checksums and stages · a promotion gate that returns APPROVED / REQUIRES_REVIEW / **REJECTED** · rollback · file-based experiment tracking (MLflow optional) |
| **Serving** | `FleetInferenceService` calls `BatteryDigitalTwinService` once per cell — one inference path, bundles loaded once per process |
| **Interfaces** | 15 new API endpoints (including `/metrics` and two administrative ones disabled by default) with pagination and partial success · a 14-page Streamlit fleet dashboard · 8 CLI pipelines |
| **Platform** | SQLite persistence behind a repository protocol · structured JSON logs · Prometheus `/metrics` · multi-stage Docker (built and run, non-root, read-only rootfs) · four CI workflows |
| **Tests** | 631 total (349 baseline + 282 new), all passing |

Honest headline from this repository's own run: the promotion gate **rejects**
the current RUL bundle on interval coverage (0.764 against a 0.80 floor), so
nothing is at stage `PRODUCTION`. The floor was not lowered to make it pass.
Full evidence: [`docs/MILESTONE_3_EVALUATION.md`](docs/MILESTONE_3_EVALUATION.md).

---

## Contents

0. [Milestone 2 at a glance](#0-milestone-2-at-a-glance)
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
16. [Milestone 2 — the digital twin](#16-milestone-2--the-digital-twin)
17. [Milestone 3 — fleet intelligence and production MLOps](#17-milestone-3--fleet-intelligence-and-production-mlops)

---

## 0. Milestone 2 at a glance

| | |
|---|---|
| **Outputs per cell** | RUL + prediction interval · SOH + health class · calibrated failure risk + risk class · degradation drivers · data-quality assessment · rule-based recommendation |
| **Uncertainty** | split conformal, life-stage conditioned, 90 % target coverage — a *prediction* interval |
| **Calibration** | isotonic, fitted on out-of-fold non-test rows; threshold tuned there too |
| **Models** | independent SOH / risk / RUL bundles **and** a shared-encoder multi-task Transformer with three heads |
| **Serving** | one `BatteryDigitalTwinService`; the FastAPI app and the Streamlit dashboard are both clients of it |
| **Contracts** | every deployable bundle carries its feature schema, fingerprints and training configuration, and refuses to load under a mismatched runtime config |

Full write-up: [`docs/MILESTONE_2_OVERVIEW.md`](docs/MILESTONE_2_OVERVIEW.md).

An example snapshot (`scripts/example_snapshot.py`, real output on a real cell):

```
Battery ID: B0005
Current cycle: 127  (observed)
Current SOH: 75.7%  (derived — a measurement, not a model output)
Estimated RUL: 22 cycles  (predicted)
SOH forecast (+30 cycles): 80.4%  (predicted)
RUL interval: 7–37 cycles (90% nominal — see the
              coverage caveat below; prediction interval, not a confidence interval)
Failure risk within 30 cycles: 48.9%  (calibrated, EXPERIMENTAL —
              withheld from the recommendation, it loses to a cycle counter)
Health class: warning
Data quality: ACCEPTABLE (score 0.75)
Main degradation factors (model attributions, not causal claims):
  - Running variability of constant-current charge fraction (increases_risk)
  - Discharge-curve slope relative to its beginning-of-life value (decreases_risk)
  - Running variability of measured discharge capacity (increases_risk)
Recommendation: Plan replacement  [PLAN_REPLACEMENT, priority high]
Warnings: 2
```

Numbers in that block come from the committed
`reports/milestone_2/example_snapshot.json`; regenerate it and they will change
with your run.

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

The headline number is **nested** leave-one-battery-out: model-family selection
runs inside every outer fold, so the pooled metric estimates *the whole
procedure* rather than a model that was already chosen using data the metric also
scores.

| | n | MAE | RMSE | R² |
|---|---|---|---|---|
| **Nested LOBO — the procedure, selection included** | 476 | **12.51** | **16.33** | **0.676** |
| Best single candidate (`random_forest`), commonly scoreable rows | 400 | 8.68 | 10.68 | 0.827 |
| Best single candidate, its own scoreable rows | 495 | 10.01 | 13.65 | 0.803 |

Errors are in **discharge cycles**. The nested figure is worse than the best
single candidate, and that gap *is* the cost of model selection — it is not a
number to optimise away. Selection frequency across the five outer folds:
`random_forest` ×3, `transformer` ×1, `cohort_median_life` ×1.

Baselines are in the same table on the same folds, because a learned model that
cannot beat one has not earned its complexity. On commonly scoreable rows:
a nearest-analogue SOH lookup reaches MAE 9.18, the cohort-median-life rule
13.43, and ridge regression 19.68 — so gradient boosting earns its place, and
ridge does not.

> **The previously published "MAE 8.06 / R² 0.850" is withdrawn.** It predates
> the Milestone 1.1 hardening: it was produced with pre-split feature pruning, an
> end-of-life rule that accepted an unconfirmed two-cycle crossing at the end of a
> record, and a champion selected on a one-cell validation partition. Do not quote
> it beside the numbers above. See
> [`docs/MILESTONE_1_1_HARDENING.md`](docs/MILESTONE_1_1_HARDENING.md).

Cross-validation is used because after the data-quality gates the cohort is five
cells — a single holdout would put **one** cell in test, and the metric would
swing on which cell was drawn. It demonstrably does: Ridge finishes **last on the
validation cell and first on the test cell** (see §8).

Random row-level splitting is not used anywhere. On this data it produces R²
above 0.99 and means nothing: consecutive cycles of one cell are near-duplicates,
so a random split lets a model interpolate between neighbouring rows instead of
forecasting.

`docs/limitations.md` is not an afterthought — read it before quoting a number.

## 3. Dataset

NASA Ames Prognostics Data Repository, "Battery Data Set" (Saha & Goebel, 2007):
34 commercial 2.0 Ah 18650 cells cycled to failure in a temperature chamber, with
interleaved electrochemical impedance sweeps.

The loader normalises the raw MATLAB structs into **one row per discharge cycle**,
enriched with summary statistics of that discharge, of the most recent preceding
charge, and of the most recent preceding impedance sweep. Nothing is ever
back-filled from a future step.

Of the 34 cells, **5** pass the cohort gates — too short, begins already
degraded, never degrades, right-censored, too few labelled cycles, or a
mid-experiment regime change. Every exclusion is mechanical and logged: the
config asks for *all* cells and the gates select the cohort deterministically.
Full accounting in **[`docs/dataset_card.md`](docs/dataset_card.md)**.

![Capacity degradation](figures/eda/01_capacity_degradation.png)

*The dataset in one figure.* Faint lines are raw capacity measurements, bold
lines the causal trailing median, the dashed line the 1.40 Ah end-of-life
threshold. Two things to notice: the raw traces are **not monotonic** — that is
real capacity recovery after rest periods, not noise, and it is why the EOL rule
requires a *persistent* crossing; and cells reach end of life anywhere between
cycle 77 and 127, which is the cell-to-cell variation the model has to survive.

One gate is worth calling out because finding it changed the results. Cells
B0042–B0044 were moved into a 4 °C chamber at cycle 41; measured capacity drops
from ~1.5 Ah to ~0.07 Ah in one step and stays there. The cells are not dead —
at 4 °C the discharge test terminates almost immediately. A first-difference
jump check cannot see a *level shift* (it flags the single edge, drops it, and
the series then looks perfectly smooth at 0.07 Ah), so the EOL detector read the
collapse as a threshold crossing and labelled end-of-life at cycle 44 — wrong by
roughly the entire remaining life of three cells. `_truncate_at_collapse` now
ends a record at a sustained collapse, with a regression test.

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

![RUL target](figures/eda/08_target_distribution.png)

*Left:* the target's distribution across all labelled rows. *Right:* RUL falls
linearly to zero within each cell, but each cell starts from a different height —
that height is what the model must infer from a few dozen cycles of history, and
it is the entire difficulty of the problem.

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

~700 features generated from 14 base signals, pruned by an unsupervised filter,
then reduced to the top 80 by supervised selection fitted on training rows only:

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

### 8.1 The headline: leave-one-battery-out cross-validation

Each of the 5 cells is held out in turn, the feature pipeline is re-fit inside
every fold, and out-of-fold predictions are pooled. Champion: **Transformer**.

| | MAE | RMSE | R² | Bias | within 10 cycles |
|---|---|---|---|---|---|
| **Pooled (400 rows, 5 folds)** | **8.06** | **9.93** | **0.850** | −1.84 | 61.3 % |

| Held-out cell | n | MAE | RMSE | R² | Bias |
|---|---|---|---|---|---|
| B0005 | 103 | 6.66 | 8.85 | 0.911 | −4.64 |
| B0006 | 87 | 6.52 | 8.20 | 0.894 | −2.40 |
| B0018 | 75 | 8.47 | 9.77 | 0.797 | +8.46 |
| B0033 | 82 | 11.99 | 13.71 | 0.664 | −8.37 |
| B0034 | 53 | 6.66 | 7.46 | 0.762 | +0.09 |

Per-fold MAE spans 6.5 – 12.0 cycles (σ = 2.34). **That spread is the honest
uncertainty on the headline number** — more so than any bootstrap over rows,
because rows within a cell are strongly correlated.

### 8.2 The single holdout, and why it is not the headline

Train on B0018/B0033/B0034, validate on B0006, test on B0005:

| Rank | Model | MAE | RMSE | R² | Bias | α-λ (20 %) |
|---|---|---|---|---|---|---|
| 1 | Ridge | **10.83** | **13.19** | **0.860** | +3.60 | 0.590 |
| 2 | **Transformer** (champion) | 11.62 | 13.82 | 0.784 | **−1.19** | 0.398 |
| 3 | GRU | 15.10 | 16.86 | 0.678 | +3.42 | 0.350 |
| 4 | Random Forest | 19.19 | 23.51 | 0.555 | −5.15 | 0.361 |
| 5 | LightGBM | 20.77 | 23.62 | 0.550 | −4.34 | 0.238 |
| 6 | LSTM | 19.46 | 23.65 | 0.367 | −6.32 | 0.243 |
| 7 | XGBoost | 21.22 | 24.24 | 0.526 | −4.55 | 0.246 |
| 8 | CatBoost | 21.85 | 27.19 | 0.404 | −6.21 | 0.344 |
| 9 | Linear Regression | 31.13 | 33.80 | 0.079 | −31.13 | 0.008 |

![Model comparison](figures/results/model_comparison.png)

**Ridge finishes last on the validation cell (RMSE 24.7) and first on the test
cell (13.2).** The Transformer does the reverse — best on validation (4.7), second
on test. With one validation cell and one test cell, model selection is close to
a coin flip, and this run demonstrates it rather than hiding it. It is the single
strongest argument for the cross-validated number in §8.1, and for treating any
"best model" claim at this cohort size with suspicion.

### 8.3 Like-for-like

Sequence models cannot score a cell's first 19 cycles, and those early rows are
the hardest — so the table above compares models on different row counts.
Restricted to the 103 rows every model can score, Ridge's lead widens (MAE 9.19
vs 11.62) but its bias grows to +7.9 against the Transformer's −1.2. If you need
an unbiased estimate rather than the lowest error, the ranking flips again.

### 8.4 What the model actually gets wrong

![RUL trajectory](figures/results/rul_trajectories_transformer.png)

This is the most informative plot in the repository. The prediction tracks truth
closely from about cycle 60 onward, but early in life the model predicts ~74
cycles remaining when the true answer is ~100. It is regressing toward the mean
of the training cells, because at cycle 25 a healthy cell genuinely does not yet
look like one that will last 127 cycles rather than 90.

![Error by RUL band](figures/explainability/error_by_rul_band_transformer.png)

The same effect quantified: MAE is 8.5 cycles at RUL 25–50 but 27.8 at RUL 100+,
and the bias flips sign across the life curve — it **under**-predicts remaining
life when the cell is fresh and **over**-predicts it near end of life. For a
maintenance decision that is the wrong way round near EOL, and it is the reason
§14 puts uncertainty quantification at the top of the roadmap.

![Prediction vs truth](figures/results/pred_vs_truth_transformer.png)

Predicted against true RUL, with the ±20 % α-λ cone shaded. Points leave the cone
at both ends of the life curve for the reason above.

![Residual diagnostics](figures/results/residual_analysis_transformer.png)

Residual diagnostics: distribution, residual vs true RUL (the clear downward
trend is the level-dependent bias), a Q–Q plot against the normal, and absolute
error against state of health.

### 8.5 Two more things worth noticing

**Gradient boosting loses badly, and that is informative.** Trees extrapolate by
returning a constant outside their training range, and each unseen cell sits at a
slightly different capacity scale. The linear and recurrent models extrapolate;
the trees cannot. Under a chronological split (`configs/chronological.yaml`),
where the model has already seen each test cell's early life, the ranking
reverses.

**Unregularised OLS collapses** (R² 0.08, bias −31 cycles) while Ridge tops the
table. With 3 training cells and 80 features, the only thing separating them is
the L2 penalty. It is a compact demonstration of why the baseline you compare
against has to be a *tuned* baseline.

## 9. Explainability

Three independent views — SHAP, permutation importance (computed within-cell so
signal marginals are preserved), and native model importances — plus an error
analysis by remaining-life band.

![Signal family importance](figures/explainability/signal_family_importance_transformer.png)

Importance is **also aggregated into physical signal families**, and that is the
reading to trust. The feature set is deliberately collinear; under collinearity
SHAP and permutation importance distribute credit among correlated features
rather than isolating a cause. At family level the picture is physically sensible:
capacity-derived signals and **charge timing** dominate, with load current next.
Charge timing scoring so highly is a genuinely useful operational finding — the
constant-current fraction of a charge shrinks monotonically as a cell ages, and
unlike a capacity test it is observable on **every ordinary charge**, with no
dedicated full discharge required.

The two negative bars are not a rendering artifact: permutation importance is
measured as the increase in RMSE when a feature is shuffled, so a negative value
means shuffling *helped*. Voltage-derived features are noise for this model on
this cohort — worth knowing before adding more of them.

![Feature importance](figures/explainability/feature_importance_transformer.png)

The same rankings at individual-feature level, from three methods side by side.
Where they disagree, that disagreement is the collinearity — which is exactly why
the family-level view above is the one to quote.

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
reports/cross_validation_by_battery.csv    leave-one-battery-out per-fold table
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
├── artifacts/                  deployable bundles: rul · soh · risk · multitask
├── docs/
│   ├── architecture.md                     Milestone 1 design and leakage boundaries
│   ├── MILESTONE_1_1_HARDENING.md          findings, fixes, before/after metrics
│   ├── MILESTONE_2_OVERVIEW.md             what Milestone 2 adds
│   ├── DIGITAL_TWIN_ARCHITECTURE.md        layering, serving pipeline, provenance
│   ├── SOH_DEFINITION.md                   references, bands, causality
│   ├── FAILURE_RISK_DEFINITION.md          the derived label, and what it is not
│   ├── UNCERTAINTY_METHOD.md               conformal construction and its assumption
│   ├── RECOMMENDATION_ENGINE.md            rules, thresholds, what the layer cannot do
│   ├── API_GUIDE.md · DASHBOARD_GUIDE.md   how to run and read them
│   ├── MODEL_CARD_MULTITASK.md             the shared-encoder model
│   ├── MILESTONE_2_EVALUATION.md           evaluation design and where the numbers are
│   ├── MILESTONE_2_LIMITATIONS.md          read before quoting anything
│   ├── MILESTONE_2_ACCEPTANCE_CHECKLIST.md
│   ├── dataset_card.md · model_card.md · limitations.md
├── figures/                    eda · results · explainability
├── models/                     Milestone 1 champion + feature pipeline + full zoo
├── notebooks/                  01 EDA · 02 features & target · 03 model comparison
│                               (exploratory records; excluded from lint and type checks)
├── reports/
│   ├── metrics.json …          Milestone 1, incl. the nested comparison
│   └── milestone_2/            metrics, evaluation report, per-row intervals, example snapshot
├── scripts/                    thin CLI wrappers · example_snapshot · sanitise_reports
├── src/battery_rul/
│   ├── config.py               typed, validated configuration (every threshold lives here)
│   ├── _compat.py              native library load ordering
│   ├── cli.py
│   ├── data/                   schema · base · nasa · synthetic · validation · loader
│   ├── features/               target · engineering · pipeline · splitting · sequences · warmup
│   ├── targets/                soh · risk                        ← Milestone 2
│   ├── models/                 base · classical · neural · baselines · multitask · bundle
│   ├── evaluation/             metrics · evaluator · nested · reporting · reporting_m2
│   ├── uncertainty/            conformal                          ← Milestone 2
│   ├── calibration/            probability                        ← Milestone 2
│   ├── explainability/         explain · drivers
│   ├── recommendations/        engine                             ← Milestone 2
│   ├── digital_twin/           domain · quality · service         ← Milestone 2
│   ├── api/                    app · schemas                      ← Milestone 2
│   ├── dashboard/              app · data_adapter                 ← Milestone 2
│   ├── visualization/          style · eda · results
│   ├── pipelines/              prepare_data · tune · train · evaluate · predict · run_pipeline
│   │                           milestone_2 (+ one alias module per documented command)
│   └── utils/                  logging · seed · io · timing
├── tests/                      test_config · test_data · test_features · test_splitting
│                               test_metrics · test_models · test_pipelines
│                               test_hardening · test_targets_m2 · test_uncertainty_calibration
│                               test_multitask · test_digital_twin · test_api
├── .github/workflows/ci.yml    lint · type-check · test · smoke · hygiene
├── pyproject.toml
├── requirements.txt            loose ranges
├── requirements-lock.txt       fully pinned environment
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
pytest                            # full suite
pytest -m "not slow"              # skips the one test that parses real .mat files
pytest --cov=battery_rul          # with coverage

# The quality gates. Same scope here, in the Makefile and in CI.
ruff check src tests scripts
black --check src tests scripts
mypy src tests scripts
```

Notebooks are excluded from lint and type checks. That exclusion is declared in
`pyproject.toml` (`tool.ruff.extend-exclude`, `tool.mypy.exclude`), not achieved
by leaving them out of a command, so the README, the Makefile and CI cannot drift
apart.

Coverage spans data loading and schema coercion, the validation gate, target
generation and EOL detection, feature engineering, **the causality guarantees**,
all three splitting strategies, metrics, every model's fit/predict/persist cycle,
the end-to-end pipeline including training/serving consistency, and regression
guards for both data-quality bugs found while building Milestone 1 (the
leading-artifact trim and the sustained-collapse truncation).

Milestone 1.1 and 2 add:

| File | What it pins down |
|---|---|
| `test_hardening.py` | planted-leakage tests for ingestion and preprocessing, EOL persistence (transient dip, exact P, incomplete end-of-record, recovery, configurability), warm-up parity, config fingerprints, cache keying |
| `test_targets_m2.py` | SOH reference strategies and causality, bands, clipping; risk label, horizons, bands |
| `test_uncertainty_calibration.py` | conformal coverage on exchangeable data, interval ordering, clipping, life-stage widths; isotonic/Platt improvement, threshold objectives, NaN-not-fabricated AUCs |
| `test_multitask.py` | windowing (never crosses a cell), label alignment, three-head shapes, weighted-loss arithmetic, masked tasks, output ranges, disk round-trip, trained-window persistence |
| `test_digital_twin.py` | snapshot serialisation and validation, data-quality checks, one test per recommendation rule, bundle validation and compatibility refusal |
| `test_api.py` | service and API end-to-end against bundles trained inside the test session; error handling, schema stability, serving parity |

The suite runs entirely on the synthetic generator, so it needs no dataset
download; the one test that parses real NASA files is marked `slow` and skips
cleanly when they are absent.

## 14. Limitations

Summarised — the full treatment is **[`docs/limitations.md`](docs/limitations.md)**.

* **5 cells.** Small. The cross-validated number uses every cell, but five is
  still five; quote the per-fold spread (σ = 2.34 cycles) alongside the mean, and
  treat any "best model" claim with suspicion — §8.2 shows the ranking flipping
  between validation and test.
* **Right-censored cells are excluded, not modelled.** In a real fleet most cells
  are healthy and censored — this discards exactly the population you would
  monitor. Survival analysis is the fix and the largest methodological gap.
* **Uncertainty is now implemented** (split conformal, Milestone 2) but its
  coverage guarantee assumes exchangeability between calibration and served
  cells, which distinct physical cells satisfy only approximately. Read the
  per-battery and per-life-stage coverage, not the marginal number.
* **One chemistry, one format, one rig.** LCO 18650, chamber-cycled, 2008-era
  instrumentation. Transfer to field data is unproven.
* **Early-life RUL is close to unpredictable** and the metrics reflect that.
* **Serving exists but is not production infrastructure.** There is an API and a
  dashboard; there is no authentication, container, model registry, experiment
  tracking or drift monitoring. Do not expose the service publicly as it stands.
* **The failure-risk label is derived, not observed.** The dataset contains no
  safety events, so the model has never seen one and cannot predict one.

## 15. Roadmap

**Milestone 1 — RUL prediction.** Complete, and hardened by Milestone 1.1.

**Milestone 1.1 — hardening gate.** Complete. Split-safe causal imputation,
train-only feature pruning, exact end-of-life persistence, nested model
selection with baselines, training/serving warm-up parity, artifact and cache
compatibility, and real quality gates.
[`docs/MILESTONE_1_1_HARDENING.md`](docs/MILESTONE_1_1_HARDENING.md)

**Milestone 2 — Battery Digital Twin.** Complete. See §16.

**Milestone 3 — Fleet intelligence and production MLOps.** Complete. See §17.

**Still out of scope, deliberately.** Authentication and multi-tenancy (the
service ships none — put an authenticated proxy in front of it), Kubernetes
manifests, cloud-vendor infrastructure, a message broker, real-time vehicle
control, and any write path to a battery-management system.

## 16. Milestone 2 — the digital twin

### What it produces

One `BatteryTwinSnapshot` per cell, in which **every value carries a provenance
tag** — `observed`, `derived`, `predicted`, `estimated` or `rule_based`. A
dashboard that renders a measured capacity and a modelled remaining life in the
same typeface invites the reader to trust them equally, and one of them has an
interval around it.

### Definitions

| Quantity | Definition | Document |
|---|---|---|
| State of health | smoothed capacity ÷ reference, a **fraction in [0, 1]** everywhere internally | [`SOH_DEFINITION.md`](docs/SOH_DEFINITION.md) |
| Failure risk | `RUL(t) ≤ H` — a **derived** label from the capacity threshold, not an observed safety event | [`FAILURE_RISK_DEFINITION.md`](docs/FAILURE_RISK_DEFINITION.md) |
| Uncertainty | split conformal, life-stage conditioned; a **prediction** interval, not a confidence interval | [`UNCERTAINTY_METHOD.md`](docs/UNCERTAINTY_METHOD.md) |
| Recommendation | deterministic rules over model outputs, fired on the interval's **lower bound** | [`RECOMMENDATION_ENGINE.md`](docs/RECOMMENDATION_ENGINE.md) |

### Modelling

Independent bundles per task (RUL, SOH, risk) **and** a shared-encoder
multi-task Transformer with RUL / SOH / risk heads, each component loss logged
separately. Both are evaluated; neither is assumed better, and coverage is
reported alongside every sequence metric because sequence models cannot score a
cell's first *window − 1* cycles.
[`MODEL_CARD_MULTITASK.md`](docs/MODEL_CARD_MULTITASK.md)

### Calibration discipline

The probability calibrator, its decision threshold and the conformal estimator
are all fitted on **out-of-fold predictions over non-test cells**. No test label
enters any calibration fit, threshold search or selection decision.

### Serving

```bash
python -m battery_rul.api.app                    # FastAPI, docs at /docs
streamlit run src/battery_rul/dashboard/app.py   # 9-tab dashboard
```

Both are clients of one `BatteryDigitalTwinService`. Neither loads a model file,
engineers a feature or applies a threshold on its own — that is what stops the
dashboard from quietly disagreeing with the API.
[`API_GUIDE.md`](docs/API_GUIDE.md) · [`DASHBOARD_GUIDE.md`](docs/DASHBOARD_GUIDE.md) ·
[`DIGITAL_TWIN_ARCHITECTURE.md`](docs/DIGITAL_TWIN_ARCHITECTURE.md)

### Individual pipeline stages

```bash
python -m battery_rul.pipelines.prepare_multitask_data --config configs/default.yaml
python -m battery_rul.pipelines.train_soh              --config configs/default.yaml
python -m battery_rul.pipelines.train_risk             --config configs/default.yaml
python -m battery_rul.pipelines.train_multitask        --config configs/default.yaml
python -m battery_rul.pipelines.calibrate_risk         --config configs/default.yaml
python -m battery_rul.pipelines.calibrate_uncertainty  --config configs/default.yaml
python -m battery_rul.pipelines.build_model_bundle     --config configs/default.yaml
python -m battery_rul.pipelines.run_milestone_2        --config configs/default.yaml [--force]
```

`run_milestone_2` skips the expensive multi-task retrain when a valid artifact
exists; `--force` rebuilds it. Every command works against
`configs/synthetic.yaml` without the NASA archive — **metrics from that
configuration describe the simulator, not real cells**, and are never published.

### Results

All figures below are read from
[`reports/milestone_2/metrics.json`](reports/milestone_2/metrics.json), written
by the run that produced them.

> **These numbers replace an earlier, better-looking set.** An external review
> found seven defects, three of which changed what the published figures meant:
> the deployed RUL family had been chosen using a comparison that included the
> test cell; the SOH model predicted current SOH from an input containing
> current capacity; and conformal coverage was measured on the residuals the
> quantile was fitted from. The earlier figures are withdrawn. Full write-up:
> [`docs/MILESTONE_2_1_REVIEW_FIXES.md`](docs/MILESTONE_2_1_REVIEW_FIXES.md).

**RUL prediction intervals — the 90 % nominal interval does not achieve
its nominal coverage:**

| | n | empirical coverage | mean width (cycles) |
|---|---|---|---|
| Cross-conformal, out-of-fold | 373 | **0.764** | 30.6 |
| Held-out test (1 cell) | 122 | **0.713** | 41.8 |
| _(in-sample, for reference only)_ | 373 | _0.917_ | — |

| stage (by SOH) | n | coverage |
|---|---|---|
| early | 159 | 0.572 |
| late | 70 | 0.929 |
| mid | 144 | 0.896 |

The in-sample row is what the previous README reported as the headline. Applying
a quantile back to the residuals it was estimated from recovers the nominal
level close to by construction; the honest estimate is the cross-conformal one,
and it falls well short. Early-life coverage is worst. **Treat the interval as
indicative, not as a 90 % guarantee** — conformal's exchangeability assumption
is not satisfied across five physically distinct cells.

Deployed family: `elastic_net`, chosen by leave-one-cell-out over **non-test
cells only**. Worth noting: with the test cell in the pool, `random_forest` won;
without it, `elastic_net` does. At this cohort size the "best model" is not a
stable quantity, which is itself a result.

**State of health — now a forecast, not a restatement:**

| | n | MAE | in SOH points | persistence baseline | beats it? | skill |
|---|---|---|---|---|---|---|
| Out-of-fold | 253 | 0.0465 | 4.65 | 0.0659 | yes | 0.29 |
| Held-out test | 92 | 0.0406 | 4.06 | 0.0656 | yes | 0.38 |

The model forecasts SOH **30 cycles ahead**. The
previously published 1.34 % MAE was for *current* SOH — a target that is measured
capacity divided by a per-cell constant, predicted from an input containing
measured capacity. It measured a rescaling. The forecast is a real prediction and
its error is correspondingly larger; it does beat a persistence baseline, by
about 29 % out of fold.

**Failure risk — fails its acceptance gate:**

| | n | PR-AUC | PR-AUC of *cycle index alone* | beats it? |
|---|---|---|---|---|
| Out-of-fold, calibrated | 373 | 0.647 | **0.928** | **no** |
| Held-out test, calibrated | 122 | 0.721 | **1.000** | **no** |

Because the label is `RUL ≤ H`, a cell's positives are exactly its last H cycles,
so counting cycles ranks them perfectly. The model loses on every partition. It
is therefore **gated out**: the twin reports the probability marked
`experimental` and **withholds it from the recommendation rules**, which then
rest on remaining life, its lower bound and measured health alone. Reporting a
negative result was not enough; a model that has demonstrated nothing must not
be able to trigger a replacement.

Full report: [`reports/milestone_2/evaluation_report.md`](reports/milestone_2/evaluation_report.md).
How to read it: [`MILESTONE_2_EVALUATION.md`](docs/MILESTONE_2_EVALUATION.md).
Acceptance status: [`MILESTONE_2_ACCEPTANCE_CHECKLIST.md`](docs/MILESTONE_2_ACCEPTANCE_CHECKLIST.md).

---

## 17. Milestone 3 — fleet intelligence and production MLOps

### The layering rule everything follows from

> **Battery-level inference has exactly one entry point.**

`FleetInferenceService` calls `BatteryDigitalTwinService.create_snapshot` once
per cell and never loads a model, builds a feature or applies a calibration
itself. A test asserts that scoring a fleet loads **zero** bundles. Three layers
sit above it and none touch a model: policy (priority rules, replacement
horizons), aggregation (statistics with explicit denominators), and monitoring
(observes, never modifies).

### What a fleet snapshot answers

```
Fleet DEMO-FLEET-01 · 24 submitted · 21 evaluated · 0 failed · 3 insufficient data
Health (measured):    3 healthy · 10 slightly degraded · 8 warning
Priority (rules):     5 P0 · 7 P1 · 9 P2 · 3 insufficient data
Median SOH  85.5 %  (derived,   n=21)
Median RUL  26.7    (predicted, n=21)
Workload:   12 immediate · 4 next-30 · 3 next-50 · 2 beyond-50
Replacement: 13 near-term  (2 optimistic – 13 conservative under the intervals)
Data quality WARNING · Drift CRITICAL · Model 1.0.0
```

Every aggregate carries the count it was computed over. "Median RUL 26.7" over
21 of 24 cells is a different claim from "median RUL over the fleet", and only
one of them is computable.

### The four monitoring questions, kept apart

| Question | Answers "did the model get worse?" |
| --- | --- |
| Is the **input** usable? (data quality) | no — it is about sensors |
| Have the **inputs** moved from training? (feature drift) | no |
| Has the **output** distribution moved? (prediction drift) | no |
| Given labels, is it still accurate? (delayed-label performance) | **yes, only this one** |

Conflating them is how a team retrains a healthy model because a temperature
sensor failed. [`docs/MONITORING_ARCHITECTURE.md`](docs/MONITORING_ARCHITECTURE.md)
has the diagnostic table.

### Governance

The registry records which version is live, who promoted it and when, with a
checksum over the bundle files that is re-verified at promotion. The gate
compares a candidate against production on fourteen checks and returns
`APPROVED` / `REQUIRES_REVIEW` / `REJECTED` — and it **rejected this
repository's own bundle** on conformal interval coverage (0.764 < 0.800). The
floor was not lowered. Nothing is at stage `PRODUCTION`.

Promotion is never automatic: `registry.promotion.allow_auto_promotion` is
false and CI never sets it.

### Documents

| Topic | Document |
| --- | --- |
| Overview and command reference | [`MILESTONE_3_OVERVIEW.md`](docs/MILESTONE_3_OVERVIEW.md) |
| Architecture | [`FLEET_ARCHITECTURE.md`](docs/FLEET_ARCHITECTURE.md) · [`FLEET_DOMAIN_MODEL.md`](docs/FLEET_DOMAIN_MODEL.md) |
| Policy | [`MAINTENANCE_PRIORITY_ENGINE.md`](docs/MAINTENANCE_PRIORITY_ENGINE.md) · [`REPLACEMENT_PLANNING.md`](docs/REPLACEMENT_PLANNING.md) |
| Monitoring | [`MONITORING_ARCHITECTURE.md`](docs/MONITORING_ARCHITECTURE.md) · [`DATA_QUALITY_MONITORING.md`](docs/DATA_QUALITY_MONITORING.md) · [`FEATURE_DRIFT.md`](docs/FEATURE_DRIFT.md) · [`PREDICTION_DRIFT.md`](docs/PREDICTION_DRIFT.md) · [`PERFORMANCE_MONITORING.md`](docs/PERFORMANCE_MONITORING.md) |
| MLOps | [`MODEL_REGISTRY.md`](docs/MODEL_REGISTRY.md) · [`MODEL_PROMOTION.md`](docs/MODEL_PROMOTION.md) · [`EXPERIMENT_TRACKING.md`](docs/EXPERIMENT_TRACKING.md) |
| Interfaces | [`API_FLEET_GUIDE.md`](docs/API_FLEET_GUIDE.md) · [`FLEET_DASHBOARD_GUIDE.md`](docs/FLEET_DASHBOARD_GUIDE.md) |
| Platform | [`DOCKER_DEPLOYMENT.md`](docs/DOCKER_DEPLOYMENT.md) · [`CI_CD.md`](docs/CI_CD.md) · [`OBSERVABILITY.md`](docs/OBSERVABILITY.md) · [`SECURITY.md`](docs/SECURITY.md) |
| Evidence | [`MILESTONE_3_EVALUATION.md`](docs/MILESTONE_3_EVALUATION.md) · [`MILESTONE_3_ACCEPTANCE_CHECKLIST.md`](docs/MILESTONE_3_ACCEPTANCE_CHECKLIST.md) · [`MILESTONE_3_LIMITATIONS.md`](docs/MILESTONE_3_LIMITATIONS.md) |
| Tour | [`DEMO_GUIDE.md`](docs/DEMO_GUIDE.md) |

### Honest status

Implemented and evidenced: fleet inference, ranking, priority engine, inspection
windows, replacement planning, workload forecasting, the four monitoring
surfaces, alerts, registry, promotion gate, rollback, persistence, tracking,
observability, 15 API endpoints, a 14-page dashboard, 8 pipelines, 282 new tests.

Docker: all three images **built and run** — non-root, no artifacts baked in,
`/health` 200 with no model and `/ready` 503 → 200 once bundles are mounted,
compose stack healthy, batch and monitoring jobs green under a read-only root
filesystem. Running them found two real defects (a Python-minor-version pickle
mismatch and a fatal `mkdir` on a read-only root), both fixed and tested.

CI: **CI, Docker and Security are green on GitHub-hosted runners.** Their first
run was not — bandit found a templated SQL statement, mypy turned out to be
checking nothing because a NumPy stub no longer parses under the pinned Python
version, and one smoke-job step asserted the opposite of the truth. All fixed.

Still true, and documented rather than hidden: no real delayed labels exist for
a five-cell laboratory cohort, and no model is at stage `PRODUCTION` because the
gate refused the only candidate on interval coverage.

Recommended release: **`v1.0.0`** — for the *platform*. It does not claim a
validated model: the registry has no `PRODUCTION` entry, because the gate
refused the only candidate. See
[`MILESTONE_3_ACCEPTANCE_CHECKLIST.md`](docs/MILESTONE_3_ACCEPTANCE_CHECKLIST.md).

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
