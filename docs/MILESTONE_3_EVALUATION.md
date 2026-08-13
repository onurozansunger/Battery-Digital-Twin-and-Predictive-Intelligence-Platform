# Milestone 3 — evaluation

What was actually run on this machine, and what it produced. Every number below
came from a command in this document; nothing is estimated or transcribed from
memory.

**Environment.** macOS (darwin 25.5.0), CPython 3.13.5, pandas 3.x, scikit-learn,
LightGBM, torch (CPU). Configuration `configs/default.yaml`, real NASA artifacts
built by Milestone 2.

---

## 1. Quality gates

| Gate | Command | Result |
| --- | --- | --- |
| Lint | `ruff check src tests scripts` | **pass** — all checks passed |
| Format | `black --check src tests scripts` | **pass** |
| Types | `mypy src tests scripts` | **pass** — no issues in 152 source files |
| Tests | `pytest -m "not slow"` | **pass** — 638 collected, exit code 0 |
| Secrets | `python scripts/check_secrets.py` | **pass** — no credential patterns in 227 tracked files |
| Paths | `python scripts/sanitise_reports.py --check` | **pass** — no absolute machine paths |

### Test counts

| | Tests |
| --- | --- |
| Milestone 1 + 2 baseline (before this milestone) | 349 |
| Milestone 3 additions | **282** |
| Registry/conformal/model-selection hardening additions | **7** |
| **Total** | **638** |

Milestone 3 test files:

| File | Tests |
| --- | --- |
| `test_fleet_ingestion.py` | 21 |
| `test_fleet_ranking.py` | 29 |
| `test_fleet_maintenance.py` | 32 |
| `test_fleet_inference.py` | 24 |
| `test_fleet_api.py` | 31 |
| `test_fleet_dashboard.py` | 15 (11 dashboard + 4 Battery Passport) |
| `test_monitoring_drift.py` | 24 |
| `test_monitoring_performance.py` | 25 |
| `test_registry.py` | 31 |
| `test_persistence.py` | 30 |
| `test_milestone_3_regression.py` | 19 |
| `test_digital_twin.py` (bundle interpreter check) | 3 added to an existing file |

The baseline suite was run **before** any Milestone 3 code was written (349
passed) and again after (all 638 pass), so the Milestone 1 and 2 regression
claim is measured, not assumed.

---

## 2. Fleet batch — measured cohort

```bash
python -m battery_rul.pipelines.run_fleet_batch --config configs/default.yaml \
    --fleet-id NASA-COHORT --source processed
```

| Quantity | Value |
| --- | --- |
| Batteries submitted | 5 |
| Successfully evaluated | 5 |
| Failed / insufficient | 0 / 0 |
| Median SOH (measured, n=5) | 75.7 % |
| Median RUL (predicted, n=5) | 3.6 cycles |
| Priorities | 4 × P0_CRITICAL, 1 × P1_URGENT |
| Processing duration | ~1.2 s |
| Active model version | 1.0.0 |

Every cell is at or near end of life because the NASA records run to end of
life and the fleet is scored at each cell's **last** cycle. This is the correct
answer for this input, and it is why the demo fleet exists.

The snapshot carries the warning that the failure-risk model is experimental and
was withheld from the decision rules.

---

## 3. Fleet batch — demonstration fleet

```bash
python -m battery_rul.pipelines.run_fleet_batch --config configs/default.yaml \
    --fleet-id DEMO-FLEET-01 --source demo --demo-size 24
```

Derived construction: 24 demo cells, each a measured cell truncated at a
different point in its life.

| Quantity | Value |
| --- | --- |
| Batteries submitted | 24 |
| Successfully evaluated | 21 |
| Failed | 0 |
| Insufficient data | 3 |
| Health bands | 3 healthy, 10 slightly degraded, 8 warning, 0 critical |
| Priorities | 5 × P0, 7 × P1, 9 × P2, 3 × INSUFFICIENT_DATA |
| Median SOH (measured, n=21) | 85.5 % |
| Median RUL (predicted, n=21) | 26.7 cycles |
| Workload | 12 immediate · 4 next-30 · 3 next-50 · 2 beyond-50 · 3 insufficient |
| Replacement candidates | 13 near-term, 6 medium-term, 2 long-term |
| Replacement bracket (near-term) | 2 optimistic – 13 conservative |
| Data quality | WARNING (11 GOOD, 10 ACCEPTABLE, 3 INSUFFICIENT) |
| Processing duration | ~3.3 s |

The uncertainty bracket is doing visible work: 2 to 13 near-term replacements is
a materially different planning conversation from a flat 13.

**These are demonstration numbers.** The fleet is derived from five measured
cells and is not a description of any operator's fleet.

---

## 4. Monitoring run

```bash
python -m battery_rul.pipelines.run_monitoring --config configs/default.yaml \
    --fleet-id DEMO-FLEET-01 --set-prediction-reference
python -m battery_rul.pipelines.run_monitoring --config configs/default.yaml \
    --fleet-id DEMO-FLEET-01 --source demo --demo-size 24
```

| Section | Result |
| --- | --- |
| Overall status | **CRITICAL** |
| Data quality | WARNING — 12.5 % of the fleet POOR or worse |
| Feature drift | CRITICAL — **77 of 80** tested features flagged |
| Prediction drift | OK — 0 of the monitored quantities shifted, sample 21 |
| Delayed-label performance | NO_LABELS — 0 labels joined, coverage 0.0 |
| Alerts | 3: DATA_QUALITY_WARNING, HIGH_CRITICAL_BATTERY_COUNT, FEATURE_DRIFT_CRITICAL |

**Reading the drift result honestly.** 77 of 80 features flagged is a real
distributional difference, and it is *not* evidence of a broken pipeline: the
reference covers the three **training** cells, and the batch contains cells
derived from all five, including the validation and test cells. On a five-cell
cohort there is no second sample of the same population to compare against. The
detector is behaving correctly on a comparison that a production deployment
would not be making. See `docs/MILESTONE_3_LIMITATIONS.md` §4.

Prediction drift on the same run is OK — because it compares against a reference
batch from the same fleet, which is the comparison it is designed for.

The reference itself: **80 features, 267 rows, train partition only**.

---

## 5. Model registry and the promotion gate

```bash
python -m battery_rul.pipelines.register_model --model-name battery-rul \
    --model-version 1.0.0 --bundle artifacts/rul --validation-status VALIDATED
python -m battery_rul.pipelines.evaluate_promotion --model-name battery-rul \
    --model-version 1.0.0 --unit-tests --contract-tests --smoke-test --leakage-check
```

Registration succeeded: checksum recorded, bundle path stored **relative**,
dataset fingerprint `53df6f08c6c8`, feature-schema fingerprint
`c7c0a875626d83e2`, 80 features.

The gate returned **REQUIRES_REVIEW** (exit code 0):

```
validation_status          PASS
artifact_checksum          PASS   matches the registered checksum
required_metadata          PASS   complete
feature_schema_compatible  PASS   no production model to compare against
unit_tests_passed          PASS
contract_tests_passed      PASS
inference_smoke_test       PASS
leakage_check              PASS
rul_mae                    UNKNOWN  MAE 8.561; no production baseline
interval_coverage          PASS     empirical coverage 0.917, minimum 0.800
inference_latency          UNKNOWN  no candidate latency measurement supplied
```

The current RUL bundle passes every measurable required check. Battery-block
cross-conformal coverage is 0.917; the remaining `UNKNOWN` is RUL MAE because
there is no production baseline or configured absolute first-model floor. It is
left at `CANDIDATE`: `REQUIRES_REVIEW` is not an auto-promotion instruction.

Promotion and rollback are exercised end to end against **fixture bundles** in
`tests/test_registry.py` (31 tests) and
`tests/test_milestone_3_regression.py::test_the_registry_round_trip_promotes_and_rolls_back`,
which registers two versions, promotes both, rolls back, and asserts the
single-production invariant and the restored artifact's checksum.

---

## 6. API smoke test

Run in-process with `fastapi.testclient` against the real artifacts.

| Endpoint | Result |
| --- | --- |
| `GET /health` | 200 |
| `GET /ready` | 200, `ready: true` |
| `GET /metrics` | 200, Prometheus text |
| `GET /v1/models` | 200 |
| `POST /v1/fleet/snapshot` (2 cells) | 200 — 2 submitted, 2 evaluated, pagination `{page 1, size 50, total 2}` |
| `POST /v1/fleet/rank` (`rank_by=rul`) | 200 — order `[B0018, B0005]` |
| `POST /v1/fleet/maintenance-plan` | 200 |
| `POST /v1/fleet/replacement-plan` | 200 |
| `POST /v1/fleet/monitoring/run` | 200 — overall CRITICAL |
| `GET /v1/fleet/NASA-COHORT/summary` | 200 |
| `GET /v1/fleet/NASA-COHORT/critical-batteries` | 200 — 5 critical |
| `GET /v1/fleet/NASA-COHORT/alerts` | 200 — paginated |
| `GET /v1/monitoring/latest` | 200 |
| `POST /v1/admin/models/promote` (disabled) | **403** |
| `POST /v1/admin/models/promote` (enabled, unknown model) | **404**, structured detail |
| `GET /v1/fleet/NOPE/summary` | **404**, detail names the batch command |

Pagination, partial failure, request validation, limit enforcement (413), CORS
defaults and the absence of filesystem paths in error bodies are covered by the
31 contract tests in `tests/test_fleet_api.py`.

---

## 7. Persistence

Verified by 30 tests plus the live runs: fleet snapshots, monitoring snapshots,
alerts, prediction records and outcome labels all round-trip; the latest-snapshot
query returns the newest; acknowledgement survives a repeated identical finding;
read-only mode raises `ReadOnlyStoreError` on every write while reads keep
working. The live database after this session's runs is 620 KB.

---

## 8. Docker

Built and run with `colima` (Docker **29.5.2**, linux/arm64, 4 CPU / 8 GB).

| Check | Result |
| --- | --- |
| `docker compose config --quiet` | pass |
| `bash -n docker/entrypoint.sh` | pass |
| Build `api` / `dashboard` / `jobs` | **all three succeed**, 3.73 GB each (shared layers) |
| Image user | `appuser` — not root |
| Dataset / model artifacts in image | none (`/app/data/raw`, `/app/models/zoo`, `/app/artifacts/rul/model.pkl` absent) |
| Package imports in image | `battery_rul 0.1.0` imports cleanly |
| **No artifacts mounted**: `/health` | **200** |
| **No artifacts mounted**: `/ready` | **503**, reason and four missing bundles named |
| Docker HEALTHCHECK | `healthy` |
| **Artifacts mounted**: `/ready` | **200**, all four bundles loaded |
| `POST /v1/fleet/snapshot` (2 cells) | 200 — `B0018` P0_CRITICAL score 95.0, `B0005` P1_URGENT score 75.5 |
| `POST /v1/fleet/rank` | 200 — `[B0018, B0005]` |
| `POST /v1/admin/models/promote` | **403** (admin disabled by default) |
| `GET /metrics` | 200, 135 lines, `battery_inference_total` present |
| `jobs` image: real fleet batch | exit 0 — 5 cells, 5 evaluated, 680 ms |
| `docker compose up api dashboard` | both **healthy** |
| `docker compose --profile jobs run monitoring` | exit 0 under a read-only root filesystem |
| `read_only: true` enforced | `touch /root-test` → `Read-only file system` |

### Two real defects this validation found

**1. Pickles are not portable across Python minor versions.** The first image
used Python 3.12 against bundles pickled by 3.13 and every bundle failed to load
with `ModuleNotFoundError: No module named 'pathlib._local'` — an error that
names nothing useful. Fixed twice over: the image now tracks the interpreter the
bundles are built with, and `save_bundle` records `python_version` while
`load_bundle` refuses a mismatch with a message naming both versions
(3 new tests in `tests/test_digital_twin.py`).

**2. `paths.ensure()` was fatal on a read-only root filesystem.** It created
*every* configured directory and raised if it could not, so a job with two
writable mounts died on `mkdir /app/data/raw` — a directory it never reads.
Directory creation is now best-effort and logged; using a missing directory
still fails loudly with the path named.

Both were only reachable by actually running the containers, which is the
argument for running them.

## 9. CI

**Executed on GitHub-hosted runners.** All three workflows green on
`milestone-3-fleet-intelligence`:

| Workflow | Result | Run |
| --- | --- | --- |
| CI | **success** | [31686776118](https://github.com/onurozansunger/Battery-Digital-Twin-and-Predictive-Intelligence-Platform/actions/runs/31686776118) |
| Docker | **success** | [31686776108](https://github.com/onurozansunger/Battery-Digital-Twin-and-Predictive-Intelligence-Platform/actions/runs/31686776108) |
| Security | **success** | [31686776102](https://github.com/onurozansunger/Battery-Digital-Twin-and-Predictive-Intelligence-Platform/actions/runs/31686776102) |

CI jobs in the recorded run were successful: Lint and format · type-check ·
Tests (3.11) · Tests (3.12) · End-to-end pipeline (synthetic) · API contract and
model-bundle fixtures · Repository hygiene. The current workflow also adds
Python 3.13 to match the bundle/container interpreter.

### Three defects the runners found

The first run was **not** green, which is the argument for running it.

**1. bandit — `B608`, medium severity.** `SQLiteRepository.prune()` built its
DELETE with an f-string over a table-name loop. The names were a hardcoded
tuple, so it was not injectable — but a templated DELETE is indistinguishable
from an injectable one to a reader and to an analyser, and "the analyser is
wrong here" is how a genuinely injectable query eventually gets waved through.
Replaced with three literal statements.

**2. mypy could not check anything.** NumPy's own stubs now use `type X = ...`,
which mypy refuses to parse under the pinned `python_version = "3.11"`, and a
fatal error in a third-party stub stops the whole run: the job had been
reporting nothing about this codebase. Moved to `3.12`. The 3.11 guarantee that
pin existed for is unaffected — the Tests (3.11) job imports every module and
fails immediately on syntax it cannot parse, which is where the original
regression was caught in the first place.

**3. My own smoke-job step asserted the opposite of the truth.** It ran
`sanitise_reports.py --check` on *freshly generated* artifacts and expected no
absolute paths. The Milestone 1/2 stages write their output locations, and the
sanitiser is the step that strips them before a commit. The step now performs
the cycle a developer actually does — sanitise, then verify — while the
`hygiene` job keeps checking what is already in git.

None of the three was reachable from a green local run.

## 10. Artifacts produced

```
artifacts/
├── fleet/{snapshots,rankings,maintenance_plans,replacement_plans}/   4 batches each
├── monitoring/
│   ├── reference_distributions/training_reference.json    80 features, 267 rows
│   ├── alerts/<batch>.json
│   └── performance_reports/<batch>.json
├── registry/models.json                                   battery-rul:1.0.0 CANDIDATE
│   └── promotion_reports/<name>_<version>_<ts>.json       REQUIRES_REVIEW
└── persistence/platform.db                                620 KB SQLite

reports/milestone_3/
├── fleet_report.md              fleet_summary.json
├── data_quality_report.json     feature_drift_report.json
├── prediction_drift_report.json model_performance_report.json
├── active_alerts.json           model_promotion_report.json
└── <stage>.json                 one per pipeline stage run
```

Generated artifacts under `artifacts/` are gitignored, consistent with the
existing artifact policy (`artifacts/*/*` ignored, metadata kept). The
`reports/milestone_3/` outputs are tracked, like the Milestone 1 and 2 reports,
because they are the reviewable deliverable.

---

## 11. What was not validated

* **No real delayed labels.** The performance monitor reports `NO_LABELS` on
  every real run. The pathway is exercised with fixture labels only.
* **No load or multi-replica testing of the containers.** They were run
  single-instance on one laptop.
* **No production model.** The only candidate clears the interval floor but
  requires human review because no first-model MAE policy is configured.
* **No multi-replica or load testing.** Latency figures here are single-process
  timings on one laptop.
* **No validation against real maintenance outcomes.** The priority score, the
  inspection windows and the replacement horizons are unvalidated policy.

---

## 12. Reproducing this

```bash
pip install -e ".[dev]"
python scripts/download_data.py
python scripts/run_pipeline.py --config configs/default.yaml
python -m battery_rul.pipelines.run_milestone_2 --config configs/default.yaml
python -m battery_rul.pipelines.build_reference --config configs/default.yaml
python -m battery_rul.pipelines.run_fleet_batch --config configs/default.yaml --fleet-id NASA-COHORT --source processed
python -m battery_rul.pipelines.run_fleet_batch --config configs/default.yaml --fleet-id DEMO-FLEET-01 --source demo --demo-size 24
python -m battery_rul.pipelines.run_monitoring --config configs/default.yaml --fleet-id DEMO-FLEET-01 --set-prediction-reference
python -m battery_rul.pipelines.run_monitoring --config configs/default.yaml --fleet-id DEMO-FLEET-01 --source demo --demo-size 24
python -m battery_rul.pipelines.generate_fleet_report --config configs/default.yaml --fleet-id DEMO-FLEET-01
python -m battery_rul.pipelines.register_model --config configs/default.yaml --model-name battery-rul --model-version 1.0.0 --bundle artifacts/rul --validation-status VALIDATED
python -m battery_rul.pipelines.evaluate_promotion --config configs/default.yaml --model-name battery-rul --model-version 1.0.0 --unit-tests --contract-tests --smoke-test --leakage-check
make lint && make type && make test
```

Exact numbers will differ where a stage retrains; the structural results
(counts, statuses, gate verdicts) are deterministic given the same artifacts.
