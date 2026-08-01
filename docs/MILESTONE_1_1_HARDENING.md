# Milestone 1.1 — hardening gate

Milestone 2 was blocked on this pass. An engineering review of Milestone 1
identified defects that would have made every Milestone 2 metric inherit the same
problems, so they were fixed, tested and the experiments re-run **before** any
digital-twin model was trained.

## Findings and fixes

### 1.1 — Full-dataset imputation crossed the evaluation boundary

**Found.** `data/validation.py::_causal_impute` fell back to `df[col].median()`
computed over **every loaded row** — all batteries, before any split existed.
A held-out cell's readings could therefore shift a training cell's imputed
value, and vice versa.

**Fixed.** Imputation is now strictly within-cell and past-only: forward fill,
then the cell's expanding median. Values with no in-cell observation stay NaN
and are resolved by a **fleet fallback learned from training rows only**, held
in `FeaturePipeline.fallback_values`, re-fitted inside every CV fold and
persisted with the bundle so serving replays the training-time value.
Missingness indicators (`<column>_is_missing`) are emitted so the absence of a
reading survives as information rather than being papered over.

**Tests.** `tests/test_hardening.py::test_imputation_never_uses_a_full_dataset_statistic`
(plants a corrupted held-out cell and requires the training cell's imputed values
to be unchanged), `…::test_imputation_never_reads_a_future_cycle`,
`…::test_missingness_indicators_are_emitted`,
`…::test_pipeline_fallback_is_learned_from_training_rows_only`.

### 1.2 — Variance filtering and correlation pruning ran before the split

**Found.** `features/engineering.py::_prune` computed variances and a correlation
matrix over the **whole dataset**, then dropped columns. Both are statistics of
the rows they see, so the identity of the surviving feature schema was a function
of the held-out cells. The claim "test cells were never seen during training,
scaling or feature selection" was not true of pruning.

**Fixed.** Both moved into `FeaturePipeline.fit`, alongside the supervised top-K
selection that was already there. Feature generation is now stateless: the column
set is a pure function of the configuration, so a one-cell serving batch produces
exactly the columns the training table did. `build_features(prune=True)` is
accepted and ignored with a warning.

**Tests.** `…::test_held_out_battery_cannot_change_the_training_schema` (corrupts
a held-out cell with random noise and requires an identical feature schema and
fingerprint), `…::test_feature_generation_does_not_prune`,
`…::test_correlated_features_are_pruned_by_the_pipeline`.

### 1.3 — End-of-life persistence was not enforced at the end of a record

**Found.** `find_eol_cycle` accepted a crossing that held "for every remaining
observation". A cell whose final one or two rows dipped below threshold was
labelled as having reached end of life on the strength of those readings — the
exact transient the three-cycle persistence rule exists to reject, and precisely
where capacity recovery makes a single reading least trustworthy.

**Fixed.** A crossing now requires **P complete consecutive observations**
(`target.eol_persistence`, default 3, now configurable rather than a module
constant). A record that ends before persistence can be confirmed is
**right-censored** — `find_eol_cycle` returns `None` and the configured censoring
policy applies.

**Tests.** `…::test_transient_dip_is_not_end_of_life`,
`…::test_exact_p_cycle_crossing_is_detected`,
`…::test_incomplete_end_of_record_crossing_is_censored`,
`…::test_recovery_after_a_dip_defers_the_crossing`,
`…::test_persistence_is_configurable`,
`…::test_record_shorter_than_persistence_is_censored`.

**Cohort impact.** Recomputed labels and cohort membership; see the before/after
table below.

### 1.4 — Model selection on one cell, then CV of only the winner

**Found.** The champion was chosen on a validation partition containing a
**single** cell, and only that already-chosen model was then cross-validated. The
headline was the cross-validated number — which describes a model selected using
data the cross-validation also scores, so it is conditional on a selection step
the interval does not account for.

**Fixed.** `evaluation/nested.py` runs a nested leave-one-battery-out design:
family selection happens in an inner leave-one-battery-out loop over the outer
fold's **training cells only**; the chosen family is refitted and scores the
held-out cell. Pooling those out-of-fold predictions estimates the whole
procedure. Every candidate is also refitted per outer fold and scored, giving
per-candidate pooled metrics, fold dispersion and **selection frequency**.
Interpretable baselines are mandatory candidates: cohort median life, capacity-
fade extrapolation, an SOH nearest-analogue lookup, elastic net and ridge.
Metrics are reported both on each model's own scoreable rows (with unscored
counts and coverage) and on the intersection every candidate can score; the
ranking uses the intersection.

**Residual bias, stated.** Hyperparameters *within* a family are not searched
inside the inner loop at this cohort size. That is disclosed rather than hidden.

### 1.5 — Warm-up and serving parity was implicit

**Found.** The warm-up trim and the sequence window requirement lived only in
training code. Nothing stopped a serving path from padding a short history and
returning a confident prediction from a window the model had never seen.

**Fixed.** `features/warmup.py` is the single source of truth. `WarmupPolicy`
computes the first scoreable cycle from the warm-up trim plus the sequence
window; training, evaluation and the digital-twin service all read it. The
service marks pre-warm-up cycles unscoreable with an explicit reason rather than
predicting, and the warm-up policy is persisted in every bundle.

**Tests.** `…::test_first_scoreable_cycle_matches_the_warmup_trim`,
`…::test_sequence_first_scoreable_cycle_includes_the_window`,
`…::test_training_and_serving_agree_on_the_first_scoreable_cycle`,
plus `tests/test_api.py::test_prediction_before_the_first_scoreable_cycle_is_refused`
and `…::test_overlapping_predictions_are_stable_as_history_grows`.

### 1.6 — Artifacts and caches could silently mismatch their configuration

**Found.** (a) A pickled model carried no record of the feature schema, target
transform, end-of-life definition or sequence length it was trained under, so a
configuration drift produced confident nonsense rather than an error. (b) The
interim cache was keyed on the **source name alone**, so editing the collapse
threshold, the leading-artifact trim or the smoothing window silently reused a
table built under the old settings. (c) Committed reports contained absolute
developer paths. (d) There was no pinned environment.

**Fixed.**
- `models/bundle.py` — a bundle is model + preprocessing + `metadata.json`
  (schema version, model version, git revision, data fingerprint, preprocessing
  fingerprint, dataset fingerprint, feature schema, warm-up policy, target
  definition, thresholds, metrics, dependency versions). `load_bundle` validates
  all of it and, under `artifacts.strict_compatibility`, refuses a bundle whose
  persisted training configuration differs from the runtime one on any
  data-affecting field — naming the field.
- The interim cache key is now `hash(data-affecting config) + hash(source files)`.
- `ExperimentConfig.data_affecting_dict()` / `data_fingerprint()` define exactly
  which fields count, so cosmetics (figure DPI, bootstrap count) do not
  invalidate a cache and reviewers are not trained to ignore the check.
- Absolute paths are stripped from committed report artifacts by
  `scripts/sanitise_reports.py`, run at the end of the pipeline.
- `requirements-lock.txt` pins the full environment.

**Tests.** `tests/test_digital_twin.py` — bundle round-trip, missing directory,
incomplete bundle, incompatible configuration (message names the field),
unsupported schema version, missing required metadata, missing calibrator when
required, feature-schema mismatch. `tests/test_hardening.py` —
`test_data_fingerprint_changes_with_a_data_affecting_field`,
`test_data_fingerprint_ignores_cosmetic_fields`,
`test_cache_path_embeds_the_fingerprint`.

### 1.7 — Quality gates were configured but not enforced

**Found.** The CI job was *named* "Lint, type-check and test" but ran no type
checker. Mypy was configured and never invoked.

**Fixed.** A separate `type-check` CI job runs `mypy src tests scripts`. Notebooks
are explicitly excluded in `pyproject.toml` (Ruff `extend-exclude`, mypy
`exclude`), and the README, the Makefile and CI use exactly the same scope.
Fixing the 117 initial mypy findings surfaced one real defect: two tests in
`tests/test_uncertainty_calibration.py` shared a name, so the first was never
run. Both are now named distinctly and both run.

## Milestone 1 metrics: before and after

The two sets are **not comparable** and are never mixed. The persistence fix
changes labels; the preprocessing-boundary fix changes which features survive;
and the nested design changes what the headline number is an estimate *of*.

| Quantity | Before (Milestone 1, as published) | After (Milestone 1.1) |
|---|---|---|
| Headline scheme | LOBO CV of the model selected on a one-cell validation partition | **Nested** LOBO: selection inside every outer fold |
| Headline model | `transformer` (selected on validation) | selection varies by fold — see the frequency table |
| Headline MAE | 8.06 cycles | see `reports/metrics.json → nested_evaluation.nested_metrics` |
| What it estimates | the selected model, on the rows it can score | **the whole procedure**, selection included |
| Baselines compared | none | 5, on identical folds |
| Common-row comparison | test partition only | every nested fold |

The current numbers live in `reports/metrics.json`,
`reports/nested_model_comparison.csv`,
`reports/nested_model_comparison_common_rows.csv` and
`reports/nested_per_fold.csv`, all regenerated from the hardened code. **The
pre-hardening 8.06 figure is withdrawn** and must not be quoted alongside the
new numbers.

The post-hardening headline is higher than the withdrawn one. That is the
expected direction: it now includes the cost of model selection, is measured on
labels a transient dip can no longer produce, and uses a feature schema no
held-out cell helped choose.

## Exit criteria

| Criterion | Status |
|---|---|
| No full-dataset fallback imputation crosses an evaluation boundary | ✅ |
| Data-dependent pruning fitted only on training data, inside folds | ✅ |
| EOL persistence requires the configured complete number of observations | ✅ |
| All candidates and baselines compared on identical battery-aware folds | ✅ |
| Model selection nested; residual bias quantified and disclosed | ✅ (hyperparameter search inside the inner loop is out of scope and stated) |
| Training and serving agree on the first scoreable cycle and overlapping predictions | ✅ |
| Artifact/configuration incompatibility fails clearly | ✅ |
| Cache reuse cannot ignore data-affecting configuration changes | ✅ |
| Milestone 1 tests and new regression tests pass | ✅ |
| Lint, format and type-check gates pass | ✅ |
| Smoke pipeline completes | ✅ |
| Published Milestone 1 reports regenerated from hardened code | ✅ |
