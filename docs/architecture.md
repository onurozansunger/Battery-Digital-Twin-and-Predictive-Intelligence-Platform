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
  │  _truncate_at_collapse()   end record at a regime change     │
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
  │        ▼                                                     │
  │  leave-one-battery-out CV  ◄── the headline number           │
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
| `evaluation/evaluator.py` | Scoring, comparison tables, LOBO cross-validation, learning curves |
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

**Every** data-dependent decision is fitted on training rows only, inside
`FeaturePipeline`: the fleet-level imputation fallback, the variance filter, the
correlation pruning, the supervised top-K selection and the scaler statistics.
Inside Optuna and inside every cross-validation fold the pipeline is re-fit —
sharing one pipeline across folds would leak held-out statistics into every score.

> **Changed in Milestone 1.1.** Variance filtering and correlation pruning used to
> run *before* the split, on the argument that they never consult the target so
> they cannot transfer label information. That argument is incomplete: both are
> statistics of the rows they see, so the identity of the surviving columns was a
> function of the held-out cells even though no label crossed the boundary. They
> now live behind the evaluation boundary with everything else. See
> `docs/MILESTONE_1_1_HARDENING.md` §1.2.

## Training/serving skew: two specific traps

**Feature-set drift.** Correlation and variance pruning are data-dependent, so a
serving batch would prune a different set than training did and the fitted
pipeline would be handed columns it has never seen. The fix is that feature
*generation* is now stateless — the column set is a pure function of the
configuration, so a one-cell serving batch produces exactly the training columns —
and `FeaturePipeline.transform` selects the exact set and order it was fitted on,
raising loudly if any are missing.
`test_predict_matches_training_time_scores` asserts the two paths agree
numerically.

**Warm-up drift.** A serving path that pads a short history produces a confident
prediction from a window the model never saw. `features/warmup.py` is the single
source of truth for the first scoreable cycle (warm-up trim plus sequence
window); training, evaluation and the digital-twin service all read it, and the
service refuses to predict below it with an explicit reason rather than guessing.

## Two data-quality traps worth knowing about

Both were found by looking at plots of the loaded data, not by a failing test,
and both silently corrupted labels before they were fixed.

**Aborted opening cycles.** Nine cells begin with one to seven discharges whose
capacity is a fraction of the cell's true beginning-of-life level — the rig was
still being brought up. Left in, they corrupt the beginning-of-life reference and
every `*_ratio_to_initial` feature. `_trim_leading_artifacts` removes a *prefix*
only, stopping at the first healthy cycle.

**Sustained collapse from a regime change.** Cells B0042–B0044 move into a 4 °C
chamber at cycle 41 and their measured capacity drops from ~1.5 Ah to ~0.07 Ah,
staying there. The cells are not dead; the discharge test aborts almost
immediately at that temperature. The end-of-life detector reads the collapse as a
persistent threshold crossing and labels EOL at cycle 44 — wrong by roughly the
entire remaining life of three cells.

The instructive part is *why the existing check missed it*. `max_capacity_jump`
inspects first differences, so it flagged the single transition cycle and dropped
it — after which the series looked perfectly smooth at 0.07 Ah and nothing
further tripped. **A first-difference test can only see the edge of a level
shift, and dropping the edge destroys the evidence.** `_truncate_at_collapse`
tests the *level* instead, with a persistence requirement so one bad reading
truncates nothing.

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
