# Milestone 2 — acceptance checklist

Status recorded against the run that produced the current contents of `reports/`.
"✅" means implemented, tested, and exercised by a real run — not merely coded.

## Milestone 1.1 gate (blocking)

| Criterion | Status | Evidence |
|---|---|---|
| No full-dataset imputation statistic crosses an evaluation boundary | ✅ | `data/validation.py::_causal_impute`; `tests/test_hardening.py` (planted-leakage test) |
| Data-dependent pruning fitted on training rows only, inside folds | ✅ | `features/pipeline.py`; `test_held_out_battery_cannot_change_the_training_schema` |
| EOL persistence requires P **complete** observations | ✅ | `features/target.py::find_eol_cycle`; six persistence regression tests |
| All candidates + baselines on identical battery-aware folds | ✅ | `evaluation/nested.py`; `reports/nested_model_comparison*.csv` |
| Selection nested; residual bias disclosed | ✅ | nested design; hyperparameter search inside the inner loop explicitly out of scope |
| Training and serving agree on the first scoreable cycle | ✅ | `features/warmup.py`; `tests/test_api.py` parity tests |
| Artifact/config incompatibility fails clearly | ✅ | `models/bundle.py`; eight bundle-validation tests |
| Cache cannot ignore data-affecting config changes | ✅ | `data/loader.py` fingerprinted cache key; `test_cache_path_embeds_the_fingerprint` |
| Milestone 1 tests + new regression tests pass | ✅ | full suite |
| Lint, format, type-check gates pass | ✅ | `ruff` / `black` / `mypy` over `src tests scripts` |
| Smoke pipeline completes | ✅ | `configs/synthetic.yaml`, Milestone 1 and 2 |
| Milestone 1 reports regenerated from hardened code | ✅ | `reports/`; pre-hardening headline withdrawn |

## Milestone 2

| Criterion | Status | Evidence |
|---|---|---|
| SOH target implemented and documented | ✅ | `targets/soh.py`, `docs/SOH_DEFINITION.md` |
| Failure-risk target implemented and documented | ✅ | `targets/risk.py`, `docs/FAILURE_RISK_DEFINITION.md` |
| At least one SOH model trained and evaluated | ✅ | `train_soh`, selection on validation, out-of-fold + test metrics |
| At least one risk model trained, calibrated, evaluated | ✅ | `train_risk`, isotonic calibration, tuned threshold |
| Multi-task sequence model implemented | ✅ | `models/multitask.py`, `docs/MODEL_CARD_MULTITASK.md` |
| RUL, SOH and risk from the shared model | ✅ | three heads, one encoder; per-head metrics reported |
| Prediction uncertainty implemented | ✅ | split conformal, life-stage conditioned; coverage per battery and stage |
| Risk calibration implemented | ✅ | isotonic/Platt, before/after Brier and ECE, reliability curve |
| Digital-twin snapshot serialisable | ✅ | `digital_twin/domain.py`; JSON round-trip test |
| Recommendation engine separate from inference | ✅ | `recommendations/engine.py`; no model import |
| Data-quality assessment included | ✅ | `digital_twin/quality.py`; gates prediction on `INSUFFICIENT` |
| FastAPI endpoints work | ✅ | 10 endpoints; 40 API/service tests |
| Streamlit dashboard works | ✅ | 9 tabs; adapter tested; imports verified in CI |
| Model bundles validate their metadata | ✅ | `load_bundle` refuses incomplete or incompatible bundles |
| API tests pass | ✅ | `tests/test_api.py` |
| No future leakage introduced | ✅ | causality checker + planted-leakage tests, extended to ingestion and preprocessing |
| No results fabricated | ✅ | every metric read from a generated artifact; AUCs are NaN rather than invented on degenerate sets |
| Documentation states limitations | ✅ | `docs/MILESTONE_2_LIMITATIONS.md`, restated in the API description, the dashboard and every snapshot payload |
| Exact reproduction commands provided | ✅ | README, `docs/MILESTONE_2_OVERVIEW.md`, Makefile targets |

## Findings this run surfaced, recorded rather than smoothed over

1. **The failure-risk classifier does not beat a cycle counter.** Because the
   label is `RUL ≤ H`, a cell's positives are exactly its last H cycles, so
   cycle index alone ranks them perfectly within any single cell. Out-of-fold
   PR-AUC is 0.65 against a 0.93 baseline; on the one-cell test partition, 0.72
   against 1.00. Every risk AUC is now reported beside that baseline, with a
   `beats_cycle_index_baseline` flag and a logged warning. This is a negative
   result about the model, not a defect in the metric.
2. **Post-calibration Brier and ECE on out-of-fold rows are in-sample.** The
   calibrator was fitted on those rows, so its ECE there is often exactly zero.
   Flagged in the payload; the test partition carries the out-of-sample evidence.
3. **The held-out cell under-covers.** Conformal intervals reach 0.917 empirical
   coverage out-of-fold against a 0.90 target, but only 0.80 on the single test
   cell — the cross-cell exchangeability caveat showing up in practice.
4. **Conformal stage conditioning originally keyed on life fraction**, which
   needs the end-of-life cycle — a label. It would have worked in evaluation and
   silently degraded to a single global quantile at serving. It now conditions on
   measured state of health, which is observable at every cycle, with a
   regression test pinning that property.

## External review round (Milestone 2.1)

Seven defects found, all reproduced in code before fixing, none disputed. Full
write-up in `docs/MILESTONE_2_1_REVIEW_FIXES.md`.

| # | Finding | Status |
|---|---|---|
| 1 | Test cell influenced the deployed RUL family | ✅ selection restricted to non-test cells |
| 2 | SOH model restated its own input | ✅ current SOH is derived; the model forecasts at a horizon |
| 3 | Risk model lost to its baseline yet drove recommendations | ✅ acceptance gate; withheld from the rules |
| 4 | Conformal coverage measured on the fitting set | ✅ leave-one-cell-out cross-conformal |
| 5 | Collapse filter accepted a one-reading terminal window | ✅ fixed, 3 regression tests |
| 6 | Prediction endpoint returned a self-contradicting payload | ✅ unconditional override removed |
| 7 | `dataset_fingerprint` duplicated the config fingerprint | ✅ derived from the raw files |
| — | SOH reported in RUL metric units | ✅ `soh_metrics` |
| — | SOH selected on validation, then reported OOF including it | ✅ leave-one-cell-out selection |
| — | Multi-task vs independent on identical rows | ⛔ still incomplete, labelled as such |

## Explicitly out of scope at this milestone

Multi-tenant authentication, cloud deployment, Kubernetes, live IoT ingestion,
Kafka, fleet optimisation, automated replacement scheduling, cross-company
Battery Passport, production drift monitoring, alerting, mobile, payments.
