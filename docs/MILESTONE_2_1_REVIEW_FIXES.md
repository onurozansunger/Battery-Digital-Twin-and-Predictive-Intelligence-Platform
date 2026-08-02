# Milestone 2.1 — external review findings and fixes

An external review of the Milestone 2 code identified seven defects. All seven
were reproduced in the code before being fixed. None was disputed.

The common thread is worth naming: Milestone 1.1 established a discipline —
nothing data-dependent may cross an evaluation boundary, and no metric may be
quoted without the trivial baseline that makes it interpretable — and Milestone 2
applied that discipline unevenly. Several findings below are the *same* defect
class that Milestone 1.1 fixed elsewhere and left standing here.

---

## 1. The test cell voted on the deployed RUL family

**Found.** `nested_model_comparison` runs over every cell, including the held-out
test cell B0005. `_preferred_rul_model` then read that run's
`selection_frequency` to choose the production family. B0005 therefore
influenced the choice of the model that was subsequently scored on B0005 and
reported as a held-out test result.

The nested comparison itself is fine — it is a cross-validated estimate of a
procedure, not a held-out claim, and using all cells is correct for that. The
defect was *reusing* it for a deployment decision and then still calling the
B0005 number untouched.

**Fixed.** `_select_family_on_non_test` chooses the family by leave-one-cell-out
over train + validation cells only. `_preferred_rul_model` and its `metrics.json`
read are deleted. The Milestone 1 nested comparison is unchanged and keeps its
role as the procedure-level estimate.

**Test.** The selection function is restricted to `data.non_test` by
construction; `tests/test_api.py` continues to pin the end-to-end path.

---

## 2. The SOH model restated its own input

**Found.** The SOH target is `smoothed_capacity / per-cell reference`, and
`smoothed_capacity` and `soh` are permitted model inputs. The target is
therefore a per-cell rescaling of a feature the model already has. The published
1.34 % test MAE was not evidence of inferring a latent health state; it largely
measured how well a rescaling can be learned. Not future leakage — target-proxy
leakage.

**Fixed**, taking both options the review offered:

* **Current SOH is no longer modelled.** The twin reports it deterministically
  from the measurement, tagged `derived`. `BatteryHealthState.soh` is a
  measurement and says so.
* **The SOH model is redefined as a forecast** at `soh.forecast_horizon_cycles`
  (default 30). At cycle *t* nothing in the input reveals capacity at *t+30*, so
  this is a genuine prediction. Rows within H cycles of a cell's last
  observation have no label and are NaN, never extrapolated. The forecast is a
  separate snapshot field, tagged `predicted`.

Every forecast metric is now reported beside a **persistence baseline** —
predicting that SOH will not change — with a `beats_persistence_baseline` flag
and a logged warning when it loses. Same lesson as the risk model's cycle-index
baseline, applied before rather than after publication.

**Tests.** `test_soh_forecast_target_is_not_the_current_cycle`,
`test_soh_metrics_expose_the_persistence_baseline`.

---

## 3. A model that lost to its baseline still drove recommendations

**Found.** The risk classifier loses to a cycle counter on every partition
(out-of-fold PR-AUC 0.647 against 0.928; test 0.721 against 1.000). Milestone 2
reported this honestly — and then let the same probability trigger
`SCHEDULE_INSPECTION`, `PLAN_REPLACEMENT` and `IMMEDIATE_ENGINEERING_REVIEW`.
Explaining a negative result is not the same as acting on it.

**Fixed.** An acceptance gate. `train_risk` records
`passes_acceptance_gate` in the bundle. When it fails and
`risk.require_beating_baseline` is set, the service marks the assessment
`is_experimental` and `excluded_from_recommendation`, withholds the probability
from the rules, and adds a snapshot warning saying the recommendation rests on
remaining life and measured health alone. The probability is still reported —
hiding it would be its own dishonesty.

**Test.** `test_experimental_risk_is_withheld_from_the_recommendation_rules`.

---

## 4. Conformal coverage was measured on the rows it was fitted from

**Found.** `calibrate_uncertainty` fitted the conformal quantile on the
out-of-fold residuals and then measured coverage on those same residuals. That
recovers the nominal level close to by construction. The reported 91.7 % was
calibration-set coverage, not evidence. The honest number was the test cell's
80.3 %, and 65.6 % in the early-life stage — so the 90 % claim was not being met.

**Fixed.** `_cross_conformal_frame` computes leave-one-cell-out conformal
intervals: for each non-test cell the quantile is refitted on the *other*
non-test cells and only then applied. The in-sample figure is still emitted, as
`in_sample_coverage_for_reference`, so the gap between the two is visible rather
than being the headline.

---

## 5. The collapse filter accepted a one-reading terminal window

**Found.** `_truncate_at_collapse` required
`window.size >= min(persistence, remaining)`, so a single low reading in the
final row truncated the record despite `collapse_persistence = 5`. This is
character-for-character the defect Milestone 1.1 fixed in `find_eol_cycle` — and
documented at length — while its twin in the loader went unexamined.

**Fixed.** `window.size == persistence and window.all()`, with the loop bounded
so a partial terminal window is never considered.

**Tests.** `test_terminal_single_low_reading_does_not_truncate_the_record`,
`test_sustained_collapse_still_truncates`,
`test_collapse_needs_exactly_persistence_observations`.

---

## 6. Prediction endpoints returned a self-contradicting payload

**Found.** With no artifacts present, `/v1/predict/rul` returned HTTP 200 with
`rul_cycles: null` and `is_scoreable: true`. `_assemble_outputs` set the flag
correctly and `create_snapshot` then overwrote it with an unconditional
`prediction.is_scoreable = True`.

**Fixed.** The override is gone. Reaching that branch means the *input* was
scoreable; whether a prediction was produced is what `_assemble_outputs`
already recorded.

---

## 7. `dataset_fingerprint` was a copy of the configuration fingerprint

**Found.** All three bundles passed `cfg.data_fingerprint()` for both fields, so
editing the raw files without touching configuration left the recorded dataset
identity unmoved — which is the one thing the second field existed to catch.

**Fixed.** `_dataset_fingerprint` uses `loader.source_fingerprint`, derived from
the raw files' names, sizes and modification times. The loader's private helper
is now public because two callers need it.

**Test.** `test_dataset_fingerprint_is_not_the_config_fingerprint`.

---

## Also raised, also fixed

* **SOH used RUL metrics.** `compute_metrics` reports `within_10_cycles`,
  `within_25_cycles`, `alpha_lambda` and a cycle-floored MAPE — all defined in
  discharge cycles. Applied to a fraction in [0, 1] they are meaningless, and
  the earlier reports published exactly that. `soh_metrics` replaces them with
  MAE, RMSE, R², median and maximum absolute error, bias, MAE in SOH percentage
  points, and the persistence-baseline comparison.

* **SOH selection bias.** The model was selected on the validation cell and then
  reported out-of-fold results that included it. Selection now runs
  leave-one-cell-out over the non-test cells, and the scheme is recorded in the
  result as `selection_scheme`.

## Raised and acknowledged, not yet resolved

* **Multi-task versus independent models on identical rows.** The report has the
  multi-task numbers and the caveat that the two are only comparable on commonly
  scoreable rows, but not the independent numbers restricted to those rows. The
  comparison is incomplete and is labelled as such rather than left to imply
  more than it shows.

* **The multi-task model is not competitive.** Test RUL MAE 14.43 against the
  independent model's 10.01, SOH MAE ~5.5 SOH points. It is retained as an
  implemented architecture and is not used for any twin output where an
  independent bundle exists; the ranking is stated plainly in the model card.
