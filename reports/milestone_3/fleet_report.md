# Fleet report — DEMO-FLEET-01

*Generated 2026-08-13T08:43:16.448268+00:00 · snapshot `20260813T084312Z-1faadee9`*

> **DEMONSTRATION FLEET DERIVED FROM MEASURED CELLS. Each cell here is one of the measured laboratory cells, truncated at a different point in its own life and given a demo identifier, so that a small cohort can illustrate a fleet at a range of ages. Several demo cells share an underlying physical cell. The measurements are real; the fleet is not, and no number computed from it describes any operator's fleet.**

## Executive summary

| Quantity | Value | Basis |
| --- | --- | --- |
| Batteries submitted | 24 | observed |
| Successfully evaluated | 21 | produced a prediction |
| Failed | 0 | excluded from all aggregates |
| Insufficient data | 3 | input cannot support a prediction |
| Healthy | 3 | measured SOH band |
| Slightly degraded | 10 | measured SOH band |
| Warning | 8 | measured SOH band |
| Critical | 0 | measured SOH band |
| Inspection recommended | 21 | rule-based |
| Replacement planning | 21 | rule-based, advisory |
| Median SOH | 85.5 % | derived, n=21 |
| Median RUL | 26.7 cycles | predicted, n=21 |
| Data-quality status | WARNING | monitoring |
| Drift status | CRITICAL | monitoring |
| Active model version | `1.0.0` | registry/bundle |

Denominators differ by quantity on purpose: 24 cells were submitted, 21 produced a prediction, and only those enter the predicted-quantity statistics.

## Maintenance priority distribution

| Priority | Count |
| --- | --- |
| INSUFFICIENT_DATA | 3 |
| P0_CRITICAL | 5 |
| P1_URGENT | 7 |
| P2_HIGH | 9 |

## Highest-priority cells

| Battery | Priority | Score | RUL (pred.) | RUL lower | SOH (meas.) | Risk | Quality | Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `DEMO-0013` | P0_CRITICAL | 95.0 | 2.8 | 0.0 | 76.7 % | 100.0 % *(experimental)* | GOOD | IMMEDIATE_ENGINEERING_REVIEW |
| `DEMO-0018` | P0_CRITICAL | 95.0 | 11.4 | 0.0 | 78.2 % | 100.0 % *(experimental)* | GOOD | IMMEDIATE_ENGINEERING_REVIEW |
| `DEMO-0019` | P0_CRITICAL | 95.0 | 9.7 | 0.0 | 85.5 % | 100.0 % *(experimental)* | GOOD | IMMEDIATE_ENGINEERING_REVIEW |
| `DEMO-0020` | P0_CRITICAL | 95.0 | 9.5 | 0.0 | 87.8 % | 100.0 % *(experimental)* | GOOD | IMMEDIATE_ENGINEERING_REVIEW |
| `DEMO-0024` | P0_CRITICAL | 95.0 | 0.9 | 0.0 | 83.1 % | 100.0 % *(experimental)* | GOOD | IMMEDIATE_ENGINEERING_REVIEW |
| `DEMO-0001` | P1_URGENT | 75.1 | 23.5 | 8.5 | 76.8 % | 48.9 % *(experimental)* | ACCEPTABLE | PLAN_REPLACEMENT |
| `DEMO-0021` | P1_URGENT | 73.9 | 26.3 | 11.3 | 77.1 % | 48.9 % *(experimental)* | ACCEPTABLE | PLAN_REPLACEMENT |
| `DEMO-0011` | P1_URGENT | 73.4 | 26.7 | 11.7 | 77.4 % | 48.9 % *(experimental)* | ACCEPTABLE | PLAN_REPLACEMENT |
| `DEMO-0017` | P1_URGENT | 71.1 | 24.8 | 9.8 | 72.8 % | 48.9 % *(experimental)* | ACCEPTABLE | PLAN_REPLACEMENT |
| `DEMO-0015` | P1_URGENT | 61.7 | 19.2 | 8.4 | 88.2 % | 100.0 % *(experimental)* | GOOD | PLAN_REPLACEMENT |
| `DEMO-0004` | P1_URGENT | 59.0 | 34.8 | 5.2 | 90.6 % | 0.0 % *(experimental)* | GOOD | PLAN_REPLACEMENT |
| `DEMO-0010` | P1_URGENT | 57.8 | 21.3 | 10.5 | 88.5 % | 100.0 % *(experimental)* | GOOD | PLAN_REPLACEMENT |
| `DEMO-0012` | P2_HIGH | 72.2 | 33.3 | 18.3 | 73.8 % | 48.9 % *(experimental)* | ACCEPTABLE | SCHEDULE_INSPECTION |
| `DEMO-0016` | P2_HIGH | 63.0 | 34.5 | 23.7 | 80.5 % | 48.9 % *(experimental)* | ACCEPTABLE | SCHEDULE_INSPECTION |
| `DEMO-0007` | P2_HIGH | 60.5 | 48.3 | 33.3 | 78.2 % | 48.9 % *(experimental)* | ACCEPTABLE | SCHEDULE_INSPECTION |

## Maintenance workload forecast

| Horizon | Cells | % of evaluated | Lower | Upper |
| --- | --- | --- | --- | --- |
| immediate | 12 | 57.1 % | — | — |
| next_10_cycles | 0 | 0.0 % | 0 | 9 |
| next_30_cycles | 4 | 19.1 % | 5 | 16 |
| next_50_cycles | 3 | 14.3 % | 14 | 19 |
| beyond_50_cycles | 2 | 9.5 % | — | — |
| monitor_only | 0 | 0.0 % | — | — |
| insufficient_data | 3 | 14.3 % | — | — |

Lower and upper counts bracket the forecast under the prediction intervals: lower uses each cell's most optimistic remaining life, upper its most conservative. This is a workload forecast, not a schedule.

## Replacement planning (advisory)

| Horizon | Candidates | Lower | Upper |
| --- | --- | --- | --- |
| near_term | 13 | 2 | 13 |
| medium_term | 6 | 14 | 19 |
| long_term | 2 | 21 | 21 |

- Replacement horizons are advisory planning input derived from model predictions and configurable thresholds, not a maintenance schedule.
- Remaining-life predictions carry interval-width uncertainty; the lower and upper counts bracket the plan under those intervals.
- No cost, downtime or spares-availability assumption is applied, so these counts must not be converted into financial figures without them.

## Monitoring

- Monitoring snapshot: `mon-20260813T084312Z-1faadee9`
- Overall status: **CRITICAL**
- Feature drift: CRITICAL — 77 of 80 tested features flagged (reference `training_reference`)
- Prediction drift: OK — 0 quantity/quantities shifted
- Delayed-label performance: NO_LABELS (0 labels joined, coverage 0.0%)

Feature or prediction drift is not evidence that the model has become less accurate. Only labelled outcomes can show that, and the delayed-label section above states how many are available.

## Active alerts

| Severity | Type | Message | Recommended human action |
| --- | --- | --- | --- |
| CRITICAL | FEATURE_DRIFT_CRITICAL | 77 of 80 tested features have drifted from the training reference. | Establish whether the inputs changed (pipeline, sensors, units) or the population did (an older fleet). Feature drift alone is not evidence the model is less accurate — check the performance report before considering a retrain. |
| WARNING | HIGH_CRITICAL_BATTERY_COUNT | 12 batteries are at a critical maintenance priority. | Review the critical cells individually against their measured capacity before scheduling any work. |
| WARNING | DATA_QUALITY_WARNING | Fleet input data quality is degraded. | Review which cells are affected and why their telemetry is thin. |
| CRITICAL | FEATURE_DRIFT_CRITICAL | 73 of 80 tested features have drifted from the training reference. | Establish whether the inputs changed (pipeline, sensors, units) or the population did (an older fleet). Feature drift alone is not evidence the model is less accurate — check the performance report before considering a retrain. |
| WARNING | HIGH_CRITICAL_BATTERY_COUNT | 5 batteries are at a critical maintenance priority. | Review the critical cells individually against their measured capacity before scheduling any work. |
| CRITICAL | FEATURE_DRIFT_CRITICAL | 77 of 80 tested features have drifted from the training reference. | Establish whether the inputs changed (pipeline, sensors, units) or the population did (an older fleet). Feature drift alone is not evidence the model is less accurate — check the performance report before considering a retrain. |
| WARNING | HIGH_CRITICAL_BATTERY_COUNT | 12 batteries are at a critical maintenance priority. | Review the critical cells individually against their measured capacity before scheduling any work. |
| WARNING | DATA_QUALITY_WARNING | Fleet input data quality is degraded. | Review which cells are affected and why their telemetry is thin. |
| CRITICAL | FEATURE_DRIFT_CRITICAL | 77 of 80 tested features have drifted from the training reference. | Establish whether the inputs changed (pipeline, sensors, units) or the population did (an older fleet). Feature drift alone is not evidence the model is less accurate — check the performance report before considering a retrain. |
| WARNING | HIGH_CRITICAL_BATTERY_COUNT | 12 batteries are at a critical maintenance priority. | Review the critical cells individually against their measured capacity before scheduling any work. |
| WARNING | DATA_QUALITY_WARNING | Fleet input data quality is degraded. | Review which cells are affected and why their telemetry is thin. |

## Warnings

- DEMONSTRATION FLEET DERIVED FROM MEASURED CELLS. Each cell here is one of the measured laboratory cells, truncated at a different point in its own life and given a demo identifier, so that a small cohort can illustrate a fleet at a range of ages. Several demo cells share an underlying physical cell. The measurements are real; the fleet is not, and no number computed from it describes any operator's fleet.
- The failure-risk model is marked experimental (it did not beat the cycle-index baseline out of fold). Risk probabilities are reported but were withheld from the maintenance rules.
- DEMO DATA: this fleet contains synthetic histories from the physics-informed generator. It is not measured data and must not be read as a description of any real fleet.

## Definitions in force

- End of life: smoothed capacity at or below 70% of the nominal reference for 3 consecutive cycles.
- Failure risk: projected end-of-life crossing within 30 cycles — a derived label, not an observed safety event.
- Prediction intervals: split_conformal at 90% target coverage.
- Priority score: configurable weighted policy (weights {'risk': 0.3, 'rul': 0.2, 'rul_lower_bound': 0.2, 'soh': 0.15, 'trend': 0.05, 'uncertainty': 0.05, 'data_quality': 0.05}), normalised to 0–100. Not an optimum.

---

Fleet intelligence from a research prototype. Rankings, maintenance priorities and replacement horizons are configurable engineering policy applied to model outputs, not validated operational decisions, and not a substitute for battery-management-system protection or qualified engineering review.
