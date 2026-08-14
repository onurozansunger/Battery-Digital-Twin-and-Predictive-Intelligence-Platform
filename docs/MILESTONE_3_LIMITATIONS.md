# Milestone 3 — limitations

Read this before quoting anything from this milestone as a result.

---

## 1. The cohort is five laboratory cells

The measured data behind everything here is the NASA Ames battery dataset:
**five cells** (B0005, B0006, B0018, B0033, B0034) of one chemistry (LCO 18650),
cycled on one test rig under a small set of duty profiles, in a laboratory.

That is not a fleet. It has:

* no vehicle-level variation (drive cycles, thermal management, pack topology);
* no manufacturing-batch variation;
* no field conditions (vibration, humidity, partial charging, calendar rest);
* no maintenance history to validate a maintenance policy against;
* five points from which to estimate a fleet distribution.

Every fleet-level number computed from it describes those five cells. **Nothing
here is validated for production electric-vehicle deployment**, and no result in
this repository should be read as evidence about a real fleet.

---

## 2. Demo fleets are synthetic and labelled everywhere

Fleet pages need more than five rows to demonstrate ranking, workload bucketing
and drift. The honest way to get them is a labelled simulation, so `--source
demo` generates cells from the physics-informed generator in
`battery_rul/data/synthetic.py`.

Guarantees: ids are prefixed `DEMO-`, `is_demo_data=True` travels into the
identity, the summary and a fleet-level warning, and the dashboard shows a red
banner on every page. There is no code path that produces demo data without that
flag.

**No metric computed on a demo fleet is a research result.**

---

## 3. The priority score has never been validated

The composite score is a **configurable decision-support policy**. Its weights
were chosen to be reasonable, not fitted, because fitting them would need
labelled maintenance outcomes — which cell should have been inspected first,
and what happened when it was not — and this platform has none.

Two operators with different weights will get different orders and both will be
legitimate. The score is reported with its full breakdown so it can be argued
with, and the API says `"not an optimum"` in `methodology_note`.

---

## 4. Feature drift here measures cell-to-cell variation

Running the monitoring pipeline against the NASA cohort reports **73 of 80
features drifted**. That is a real distributional difference and not a bug: the
reference covers the three *training* cells and the batch contains all five,
including the validation and test cells.

In a production fleet the current batch is new data from the same population.
Here it is partly a different population, because a five-cell cohort has no
"same population" to draw a second sample from. The drift machinery is correct;
the comparison it is making in this repository is not the comparison it would
make in production.

---

## 5. Prediction drift is under-powered on this cohort

`monitoring.prediction_drift.min_sample_size` is 20 scored cells. The NASA
cohort has five, so real prediction-drift runs report `UNKNOWN` and publish no
metric. That is the honest behaviour — five points do not support a distribution
comparison — and the populated path is exercised by the 24-cell demo fleet and
by unit tests.

---

## 6. No real delayed labels exist

The delayed-label workflow is fully implemented, and it has never been fed a
real production label, because there are none: these cells' end-of-life cycles
are already known and are the training labels.

The pathway is exercised with **fixture labels** in the test suite, and real
monitoring runs in this repository report `NO_LABELS`. Until a deployment
generates genuine post-hoc outcomes, the performance monitor's thresholds are
untuned and its statuses are untested against reality.

---

## 7. The risk model fails its own acceptance gate

Carried over from Milestone 2 and visible in every fleet snapshot here: the
failure-risk classifier does not beat a cycle-index baseline out of fold. The
platform therefore marks its probability `is_experimental` and **withholds it
from every decision rule** — the priority engine, the replacement planner and
the priority score all treat it as absent.

The consequence for Milestone 3 is that fleet decisions in this repository rest
on remaining life, its interval and measured health only. That is the correct
behaviour, and it is also a reduced evidence base.

---

## 8. The current RUL bundle fails the promotion gate

Battery-block cross-conformal out-of-fold interval coverage is **0.917**, but
the worst held-out cell (`B0033`) covers only **0.703** against a configured
worst-cell floor of **0.80**. The promotion gate therefore returns `REJECTED`;
the aggregate is not allowed to hide this cohort-specific miss. The first model
also has no production MAE baseline or configured absolute MAE floor.

Nothing in this repository is at stage `PRODUCTION`. The serving path still
loads the configured artifacts and works; what is missing is the *explicit*
statement that a reviewed model is live. Promotion remains a human decision.

---

## 9. Statistical caveats in the drift layer

* KS and Wasserstein are computed against a reference sample **reconstructed
  from stored quantiles**, not raw training rows. Both are approximate, and both
  are reported beside their sample sizes.
* PSI, Wasserstein and JS have no p-value and are judged on thresholds alone.
  Only KS and chi-square are multiple-comparison corrected.
* PSI's conventional bands come from credit scoring, not from battery
  prognostics.
* Fleet drift fractions are jumpy on small feature sets: with four tested
  features, one flagged is 25 % and lands on CRITICAL.

---

## 10. Operational gaps

| Gap | Consequence |
| --- | --- |
| No authentication or authorisation | do not expose the service to an untrusted network |
| No rate limiting | a proxy must provide it |
| Single-node SQLite | several API replicas against one database file on a shared volume is untested |
| No tracing | correlation relies on `batch_id` in structured logs |
| No candidate latency measurement | the promotion gate reports `inference_latency: UNKNOWN`; serving performance has not been benchmarked |
| SOH test set is one cell | `n_test_cells = 1`, so its test metric does not establish cross-cell generalisation |
| Measured fleet is five near-EOL cells | median RUL is 3.625 cycles; ranking on a healthy real fleet has not been tested, and the larger demo is synthetic |
| Docker images not built here | see the note at the top of `docs/DOCKER_DEPLOYMENT.md` |
| CI not executed here | workflows are YAML-valid and mirror locally-run commands |
| Model bundles are pickles | mitigated (configured paths only, checksums, JSON reference), not eliminated |

---

## 11. What "advisory" means

Replacement horizons, inspection windows and maintenance priorities are
**engineering decision support**. This platform:

* does not schedule work,
* does not command a battery-management system,
* does not take an asset out of service,
* does not retrain or promote a model on its own,
* attaches no cost model to any count, and refuses to produce savings figures
  without explicit cost assumptions.

Every payload carries that disclaimer, and every alert names a **human** action.

---

## 12. Attribution and drift are not causal

Feature attributions describe how the model's output responds to its inputs.
Drift statistics describe how two distributions differ. Neither is a causal
claim about a cell, and attention weights in particular are a diagnostic, not an
explanation. Carried over unchanged from
`docs/MILESTONE_2_LIMITATIONS.md`.
