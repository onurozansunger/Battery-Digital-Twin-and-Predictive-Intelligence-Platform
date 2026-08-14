# The model-promotion gate

A candidate is compared against the current production model on the axes that
matter operationally. Every check reports its own verdict, and the gate returns
a decision — it does not promote.

---

## Three outcomes

| Decision | When |
| --- | --- |
| `APPROVED` | every required check passed |
| `REQUIRES_REVIEW` | nothing failed, but something could not be checked |
| `REJECTED` | a required check failed |

`REQUIRES_REVIEW` exists because the alternative is worse in both directions.
Treating "we could not measure this" as a pass promotes blind; treating it as a
failure makes the first model of a family unpromotable.

---

## The checks

| Check | Required | Passes when |
| --- | --- | --- |
| `validation_status` | `require_validation_status` | status is VALIDATED or PASSED |
| `artifact_checksum` | `require_artifact_checksum` | on-disk bundle matches the registered checksum |
| `required_metadata` | yes | dataset fingerprint, feature-schema fingerprint and task are present |
| `feature_schema_compatible` | `require_feature_schema_compatible` | the candidate has a schema fingerprint; a *difference* from production is allowed but reported |
| `unit_tests_passed` | `require_tests_passed` | reported true by the caller |
| `contract_tests_passed` | `require_contract_tests` | reported true by the caller |
| `inference_smoke_test` | `require_inference_smoke_test` | reported true by the caller |
| `leakage_check` | `require_leakage_check` | reported true by the caller |
| `rul_mae` | yes | ≤ production × (1 + `max_rul_mae_regression`), or ≤ `max_absolute_rul_mae` when there is no baseline |
| `soh_mae` | no | ≤ production × (1 + `max_soh_mae_regression`) |
| `risk_pr_auc` | no | ≥ production − `max_pr_auc_regression` |
| `risk_brier` | no | ≤ production × (1 + `max_brier_regression`) |
| `interval_coverage` | yes | ≥ `min_interval_coverage` (0.80) |
| `worst_cell_interval_coverage` | yes | worst reported held-out cell ≥ `min_worst_cell_interval_coverage` (0.80) |
| `inference_latency` | no | ≤ production × `max_latency_regression_ratio` (1.5) |

Test evidence is **supplied to the gate**, not assumed by it. A gate that
inferred "tests passed" from the fact that it was running in CI would pass on
any pipeline that forgot to run them.

---

## Worked example — the current repository bundle

```
$ python -m battery_rul.pipelines.evaluate_promotion \
      --model-name battery-rul --model-version 1.0.0 \
      --unit-tests --contract-tests --smoke-test --leakage-check

REJECTED
  validation_status          PASS | validation_status=VALIDATED
  artifact_checksum          PASS | matches the registered checksum
  required_metadata          PASS | complete
  feature_schema_compatible  PASS | no production model to compare against …
  unit_tests_passed          PASS | passed
  contract_tests_passed      PASS | passed
  inference_smoke_test       PASS | passed
  leakage_check              PASS | passed
  rul_mae                    UNKNOWN | MAE 8.561; no production baseline
  interval_coverage          PASS | empirical coverage 0.917, minimum 0.800
  worst_cell_interval_coverage FAIL | worst cell B0033 coverage 0.703, minimum 0.800
  inference_latency          UNKNOWN | no candidate latency measurement was supplied
reasons: ['worst_cell_interval_coverage: worst cell B0033 coverage 0.703, minimum 0.800']
```

This is a real result from the real Milestone 2 RUL bundle. Marginal
battery-block cross-conformal coverage is 0.917, but B0033 covers only 0.703.
The worst-cell check therefore rejects the candidate instead of allowing the
aggregate to hide a cohort-specific miss. The bundle stays at `CANDIDATE`.

---

## Promotion never happens automatically

`registry.promotion.allow_auto_promotion` defaults to **false** and CI never
sets it. An APPROVED verdict is a recommendation; the note on every approved
report says so.

Reasons this is not negotiable:

* a gate that promotes on green turns whichever metric is easiest to move into a
  deploy button;
* an approval is evidence about *metrics*, and a deployment decision also
  involves timing, traffic, rollback readiness and who is on call;
* the gate cannot see the things that make a promotion unwise this afternoon.

The `promote_model` pipeline **re-evaluates the gate** even if CI evaluated it,
because the artifact may have changed since. `--dry-run` reports what would
happen; `--force` overrides a rejection and records `"forced": true` in the
registry history.

---

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | the stage ran; the verdict is APPROVED or REQUIRES_REVIEW |
| 1 | the stage failed (bad arguments, missing registry entry, unreadable artifact) |
| 2 | the stage ran fine and the verdict is REJECTED |

A pipeline can act on 2 without treating a working gate as a broken build. In
this repository's CI the evaluation is run with `|| true`: a rejected gate must
not fail the build, because that creates pressure to loosen the gate.

---

## Reports

Every evaluation writes:

* `artifacts/registry/promotion_reports/<name>_<version>_<timestamp>.json`
* `reports/milestone_3/model_promotion_report.json` (the latest)

Both contain every check with its status, detail, candidate value, production
value and threshold — the argument, not just the verdict.
