# Architecture

## Design principles

1. **Notebooks contain no logic.** Every transformation lives in `src/`, so what
   a notebook demonstrates is exactly what runs in the pipeline.
2. **Configuration is typed and validated.** A typo in YAML raises; it does not
   silently take a default.
3. **Leakage is prevented structurally, then verified mechanically.** Causality
   is a property of the code, and `assert_no_leakage` proves it on every run.
4. **Datasets are plug-in.** Adding CALCE or Oxford means one new class.
5. **Artifacts are self-describing.** Every output carries the config, git
   revision and package versions that produced it.

## Stage flow

```
                        configs/*.yaml
                              │
                     ┌────────▼─────────┐
                     │ ExperimentConfig │   pydantic, extra="forbid"
                     └────────┬─────────┘
                              │
  data/raw/nasa/mat/*.mat     │
            │                 │
  ┌─────────▼─────────────────▼──────────────────────────────────┐
  │ STAGE 1  prepare_data                                        │
  │                                                              │
  │  BatterySource.load()      per-cell .mat parsing             │
  │        ▼                                                     │
  │  coerce_schema()           canonical cycle-level table       │
  │        ▼                                                     │
  │  _trim_leading_artifacts() drop aborted opening cycles       │
  │        ▼                                                     │
  │  _derive_health()          trailing-median capacity, SoH     │
  │        ▼                                                     │
  │  validate_cycles()         bounds, jumps, causal imputation  │
  │        ▼                                                     │
  │  _apply_cohort_gates()     beginning-of-life health screen   │
  │        ▼                                                     │
  │  attach_target()           RUL = k_EOL − k                   │
  │        ▼                                                     │
  │  build_features()          ~700 causal features              │
  │        ▼                                                     │
  │  assert_no_leakage()       ◄── truncation proof              │
  │        ▼                                                     │
  │  make_split()              battery-holdout / chrono / WF     │
  └───────────────────────┬──────────────────────────────────────┘
                          │  data/processed/{dataset,cycles}.parquet
                          │  data/processed/manifest.json
  ┌───────────────────────▼──────────────────────────────────────┐
  │ STAGE 1b  tune  (optional)                                   │
  │  Optuna × battery-grouped CV, pipeline re-fit per fold       │
  └───────────────────────┬──────────────────────────────────────┘
                          │  reports/tuning.json
  ┌───────────────────────▼──────────────────────────────────────┐
  │ STAGE 2  train                                               │
  │                                                              │
  │  FeaturePipeline.fit(TRAIN ONLY)   scale + top-K selection   │
  │        ▼                                                     │
  │  for model in zoo:  fit(train, val)                          │
  │        ▼                                                     │
  │  champion = argmin(VALIDATION metric)                        │
  │        ▼                                                     │
  │  score champion + zoo on TEST (once)                         │
  └───────────────────────┬──────────────────────────────────────┘
                          │  models/trained_model.pkl
                          │  models/feature_pipeline.pkl
                          │  models/zoo/*.pkl
                          │  reports/metrics.json
  ┌───────────────────────▼──────────────────────────────────────┐
  │ STAGE 3  evaluate                                            │
  │  EDA figures · prediction plots · residual diagnostics       │
  │  SHAP · permutation importance · error analysis              │
  │  learning curves · evaluation_report.md                      │
  └───────────────────────┬──────────────────────────────────────┘
                          │  figures/**  reports/evaluation_report.md
  ┌───────────────────────▼──────────────────────────────────────┐
  │ STAGE 4  predict                                             │
  │  load persisted pipeline + model → score unseen cycles       │
  │  (build_features with prune=False — see below)               │
  └──────────────────────────────────────────────────────────────┘
```

## Package layout

| Module | Responsibility |
|---|---|
| `config.py` | Typed configuration; the single source of every tunable |
| `_compat.py` | Native-library load ordering (duplicate-OpenMP fix) |
| `data/schema.py` | The canonical cycle-level column contract |
| `data/base.py` | `BatterySource` ABC + source registry |
| `data/nasa.py` | NASA `.mat` parsing and cycle summarisation |
| `data/synthetic.py` | Physics-informed surrogate (tests + fallback) |
| `data/validation.py` | Integrity gate and causal imputation |
| `data/loader.py` | Ingestion orchestration, health derivation, caching |
| `features/target.py` | RUL definition and EOL detection |
| `features/engineering.py` | Causal feature generation + leakage proof |
| `features/pipeline.py` | Fitted scaler + supervised selection (serialisable) |
| `features/splitting.py` | Battery-holdout / chronological / walk-forward |
| `features/sequences.py` | Sliding-window tensor construction |
| `models/base.py` | `BaseModel` ABC, `TrainingData`, registry |
| `models/classical.py` | Linear, RF, XGBoost, LightGBM, CatBoost |
| `models/neural.py` | LSTM, GRU, Transformer + shared training loop |
| `models/search_spaces.py` | Optuna spaces, versioned with the code |
| `evaluation/metrics.py` | MAE/RMSE/MAPE/R² + α-λ, prognostic horizon |
| `evaluation/evaluator.py` | Scoring, comparison tables, learning curves |
| `evaluation/reporting.py` | Markdown report rendering |
| `explainability/explain.py` | SHAP, permutation importance, error analysis |
| `visualization/` | Shared style, EDA figures, result figures |
| `pipelines/` | The five executable stages |
| `cli.py` | `battery-rul` command-line entry point |

## The three leakage boundaries

Leakage is the failure mode this codebase is built around. There are three
distinct boundaries, each enforced in a different place:

**1. Temporal, within a cell** — `features/engineering.py`

Every feature at cycle *k* reads only cycles ≤ *k* of that cell. All windows are
trailing; there is no `shift(-n)` anywhere; capacity smoothing uses a trailing
median rather than a centred filter. Verified by `assert_no_leakage`, which
rebuilds features on a truncated history and requires bit-identical values —
and which the test suite proves *can* fail by feeding it a deliberately
non-causal builder.

**2. Between cells** — `features/engineering.py` + `features/splitting.py`

Features are computed inside `groupby('battery_id')`, so no cell's statistics
touch another's. The default split holds out whole cells.

**3. Train → validation/test** — `features/pipeline.py`

Scaler statistics and supervised feature selection are fit on training rows only.
Inside Optuna, the pipeline is re-fit **within each CV fold** — sharing one
pipeline across folds would leak held-out statistics into every trial's score.

Unsupervised pruning (near-constant and near-duplicate columns) runs before the
split. That is deliberate and safe: it never consults the target, so it cannot
transfer label information.

## Training/serving skew: one specific trap

Correlation and variance pruning are **data-dependent** — the surviving column
set depends on which rows are in hand. A serving batch would therefore prune a
different set than training did, and the fitted pipeline would be handed columns
it has never seen.

The fix is that `predict.py` calls `build_features(..., prune=False)`. Feature
*generation* is deterministic, so the unpruned output is always a superset of the
training columns, and `FeaturePipeline.transform` then selects the exact set and
order it was fitted on — raising loudly if any are missing.
`test_predict_matches_training_time_scores` asserts the two paths agree
numerically.

## Native library load order

PyTorch, LightGBM, XGBoost and scikit-learn each vendor their own OpenMP runtime.
On the reference environment four `libomp` copies are resident, and whichever
loads second binds against the first — LightGBM then dies with SIGSEGV inside
`fit`, with no Python traceback. `_compat.py` imports the boosting libraries
before torch and pins `OMP_NUM_THREADS=1`; it is imported at the top of
`battery_rul/__init__.py` so any consumer gets a working process. The widely
cited `KMP_DUPLICATE_LIB_OK=TRUE` workaround was measured here and does *not*
fix it.

## Extension points

| To add… | Do this |
|---|---|
| A dataset | Subclass `BatterySource`, decorate `@register_source("key")` |
| A model | Subclass `BaseModel`, decorate `@register_model("key")` |
| A metric | Add to `evaluation/metrics.py` and `METRIC_DIRECTION` |
| A search space | Add a function to `models/search_spaces.SEARCH_SPACES` |
| A feature family | Add to `_build_for_battery`; causality is your responsibility, and `assert_no_leakage` will check it |
| A split strategy | Add to the dispatch table in `features/splitting.make_split` |
