# Milestone 3 — acceptance checklist

Evidence for each criterion. "Verified" means a command was run or a test
asserts it; anything else says what is missing.

---

## Regression

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Milestone 1 tests still pass | ✅ | 631/631 pass, exit 0; baseline was 349 before this milestone |
| 2 | Milestone 2 tests still pass | ✅ | same run; `test_milestone_3_regression.py` asserts the M2 service, endpoints, dashboard, schema versions and recommendation engine |
| 3 | `BatteryDigitalTwinService` remains the battery-level entry point | ✅ | `test_the_fleet_service_reuses_the_battery_level_service`, `test_bundles_are_loaded_once_not_once_per_battery` |
| 4 | Existing snapshot schemas remain compatible | ✅ | `SNAPSHOT_SCHEMA_VERSION == "2.0"`, `BUNDLE_SCHEMA_VERSION == "2.0"` asserted |
| 5 | The data fingerprint is unchanged by the new config | ✅ | `test_the_data_fingerprint_is_unchanged_by_the_new_configuration_sections`; existing bundles still load |

## Fleet

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 6 | `FleetInferenceService` implemented | ✅ | `fleet/inference.py`; 24 tests |
| 7 | `FleetSnapshot` is serialisable | ✅ | `test_the_snapshot_is_json_serialisable_and_reloadable` round-trips without loss |
| 8 | Fleet aggregation implemented | ✅ | `fleet/aggregation.py`; denominators asserted |
| 9 | Ranking implemented | ✅ | 12 keys, all parametrised in `test_fleet_ranking.py` |
| 10 | Maintenance priority implemented | ✅ | 7 levels, every rule individually tested |
| 11 | Priority-score breakdown returned | ✅ | `test_the_breakdown_reconstructs_the_score` |
| 12 | Inspection windows implemented | ✅ | cycles always; days only with a measured duty rate |
| 13 | Replacement planning implemented | ✅ | 3 horizons + uncertainty brackets |
| 14 | Workload forecasting implemented | ✅ | `test_the_workload_forecast_covers_every_cell_exactly_once` |
| 15 | Fleet ingestion handles files, directories, frames, records | ✅ | 21 tests incl. path traversal and partial failure |

## Monitoring

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 16 | Fleet data-quality monitoring | ✅ | `monitoring/data_quality.py`; live run reported WARNING at 12.5 % |
| 17 | Feature drift detection | ✅ | PSI/KS/Wasserstein/JS + chi-square/unseen-rate; live run 77/80 flagged |
| 18 | Prediction drift detection | ✅ | live run OK on 21 scored cells |
| 19 | Delayed-label performance monitoring | ⚠️ | implemented and tested with **fixture** labels; no real labels exist, real runs report `NO_LABELS` |
| 20 | Monitoring snapshots persisted | ✅ | SQLite; `GET /v1/monitoring/latest` returns them |
| 21 | Alert policy implemented | ✅ | 11 types; live run raised 3; every alert names a human action |
| 22 | Drift reference from the training partition only | ✅ | `test_the_reference_is_built_from_the_training_partition_only` |

## MLOps

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 23 | Experiment tracking implemented | ✅ | file backend default, MLflow optional with fallback; 6 tests |
| 24 | Model registry implemented | ✅ | `registry/store.py`; 29 tests; `battery-rul:1.0.0` registered |
| 25 | Promotion gate implemented | ✅ | 14 checks; **rejected the real bundle** on interval coverage 0.764 < 0.800 |
| 26 | Rollback implemented | ✅ | restores the previously *live* version; single-production invariant asserted |
| 27 | Reproducible training and inference | ✅ | config-driven, seeded, fingerprinted; `test_running_the_same_fleet_twice_gives_the_same_numbers` |
| 28 | Batch / online separation | ✅ | 413 with the batch command past the online limit |
| 29 | Persistence layer | ✅ | SQLite behind a `Repository` protocol; 30 tests; no SQL in route handlers |
| 30 | Structured observability | ✅ | JSON logs with ambient context, metrics registry, `/metrics` |

## Interfaces

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 31 | Fleet API endpoints work | ✅ | 15 new endpoints (12 fleet/registry/monitoring + `/metrics` + 2 admin); 31 contract tests |
| 32 | Pagination | ✅ | `test_battery_records_are_paged_with_honest_metadata` |
| 33 | Partial-failure behaviour | ✅ | 200 with failures reported; asserted at API and service level |
| 34 | Fleet dashboard works | ✅ | 14 pages; adapter tested; import smoke test in CI |
| 35 | Docker images build | ✅ | all three targets built (3.73 GB each) and run: non-root, no artifacts baked in, `/health` 200, `/ready` 503→200 with mounts, HEALTHCHECK healthy |
| 36 | Docker Compose usable | ✅ | `up api dashboard` → both healthy; `--profile jobs run monitoring` → exit 0 under a read-only root filesystem |
| 37 | CI workflows valid | ⚠️ | all four parse and enumerate jobs; **not executed on a runner** |

## Honesty

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 38 | Security documentation exists | ✅ | `docs/SECURITY.md` — threat model, secrets, incident response |
| 39 | No secrets in the repository | ✅ | `scripts/check_secrets.py` clean over 227 tracked files, in CI |
| 40 | No fabricated metrics | ✅ | every number in `MILESTONE_3_EVALUATION.md` came from a command in that document |
| 41 | Demo data clearly labelled | ✅ | `is_demo_data` through identity → summary → warning → report banner → dashboard banner; asserted |
| 42 | Exact reproduction commands documented | ✅ | `MILESTONE_3_EVALUATION.md` §12, `DEMO_GUIDE.md` |
| 43 | Known limitations explicit | ✅ | `docs/MILESTONE_3_LIMITATIONS.md`, 12 sections |
| 44 | No absolute machine paths | ✅ | `sanitise_reports.py --check` passes; registry and report paths stored relative, asserted |

---

## Not met, and why

**CI execution (#37).** Requires a push to a GitHub remote. To close it: push the
branch and read the Actions tab.

**Real delayed labels (#19).** Requires a deployment that generates post-hoc
outcomes. Until then the metrics are untuned and the statuses untested against
reality.

**No production model.** The gate rejects the only candidate on interval
coverage. Closing it means improving conformal coverage, not lowering the floor.

---

## Recommended release status

**`v0.9.0-rc1`**

Docker is now built, run and evidenced (#35, #36). Two criteria remain open and
neither can be closed from this machine: **CI has not run on a GitHub runner**
(#37), and **no real delayed labels exist** for this cohort (#19). A third — a
model at stage `PRODUCTION` — is open because the promotion gate correctly
refused the current bundle on interval coverage.

`v1.0.0` is appropriate once a runner has executed the workflows green and a
model has cleared the promotion gate on its own merits.
