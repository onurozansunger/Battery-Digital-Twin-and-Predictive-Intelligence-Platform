# Fleet report — NASA-COHORT

*Generated 2026-08-13T10:55:30.681279+00:00 · snapshot `20260813T105529Z-f48bfe3a`*

## Executive summary

| Quantity | Value | Basis |
| --- | --- | --- |
| Batteries submitted | 5 | observed |
| Successfully evaluated | 5 | produced a prediction |
| Failed | 0 | excluded from all aggregates |
| Insufficient data | 0 | input cannot support a prediction |
| Healthy | 0 | measured SOH band |
| Slightly degraded | 2 | measured SOH band |
| Warning | 2 | measured SOH band |
| Critical | 1 | measured SOH band |
| Inspection recommended | 5 | rule-based |
| Replacement planning | 5 | rule-based, advisory |
| Median SOH | 75.7 % | derived, n=5 |
| Median RUL | 3.6 cycles | predicted, n=5 |
| Data-quality status | OK | monitoring |
| Drift status | CRITICAL | monitoring |
| Active model version | `1.0.0` | registry/bundle |

Denominators differ by quantity on purpose: 5 cells were submitted, 5 produced a prediction, and only those enter the predicted-quantity statistics.

## Maintenance priority distribution

| Priority | Count |
| --- | --- |
| P0_CRITICAL | 5 |

## Highest-priority cells

| Battery | Priority | Score | RUL (pred.) | RUL lower | SOH (meas.) | Risk | Quality | Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `B0005` | P0_CRITICAL | 95.0 | 21.5 | 0.0 | 75.7 % | 56.6 % *(experimental)* | ACCEPTABLE | IMMEDIATE_ENGINEERING_REVIEW |
| `B0006` | P0_CRITICAL | 95.0 | 16.1 | 0.0 | 68.9 % | 56.6 % *(experimental)* | ACCEPTABLE | IMMEDIATE_ENGINEERING_REVIEW |
| `B0018` | P0_CRITICAL | 95.0 | 1.2 | 0.0 | 75.7 % | 100.0 % *(experimental)* | GOOD | IMMEDIATE_ENGINEERING_REVIEW |
| `B0033` | P0_CRITICAL | 95.0 | 0.9 | 0.0 | 83.1 % | 100.0 % *(experimental)* | GOOD | IMMEDIATE_ENGINEERING_REVIEW |
| `B0034` | P0_CRITICAL | 95.0 | 3.6 | 0.0 | 83.8 % | 100.0 % *(experimental)* | GOOD | IMMEDIATE_ENGINEERING_REVIEW |

## Maintenance workload forecast

| Horizon | Cells | % of evaluated | Lower | Upper |
| --- | --- | --- | --- | --- |
| immediate | 5 | 100.0 % | — | — |
| next_10_cycles | 0 | 0.0 % | 0 | 5 |
| next_30_cycles | 0 | 0.0 % | 2 | 5 |
| next_50_cycles | 0 | 0.0 % | 3 | 5 |
| beyond_50_cycles | 0 | 0.0 % | — | — |
| monitor_only | 0 | 0.0 % | — | — |
| insufficient_data | 0 | 0.0 % | — | — |

Lower and upper counts bracket the forecast under the prediction intervals: lower uses each cell's most optimistic remaining life, upper its most conservative. This is a workload forecast, not a schedule.

## Replacement planning (advisory)

| Horizon | Candidates | Lower | Upper |
| --- | --- | --- | --- |
| near_term | 5 | 2 | 5 |
| medium_term | 0 | 3 | 5 |
| long_term | 0 | 5 | 5 |

- Replacement horizons are advisory planning input derived from model predictions and configurable thresholds, not a maintenance schedule.
- Remaining-life predictions carry interval-width uncertainty; the lower and upper counts bracket the plan under those intervals.
- No cost, downtime or spares-availability assumption is applied, so these counts must not be converted into financial figures without them.

## Monitoring

- Monitoring snapshot: `mon-20260813T105529Z-f48bfe3a`
- Overall status: **CRITICAL**
- Feature drift: CRITICAL — 73 of 80 tested features flagged (reference `training_reference`)
- Delayed-label performance: NO_LABELS (0 labels joined, coverage 0.0%)

Feature or prediction drift is not evidence that the model has become less accurate. Only labelled outcomes can show that, and the delayed-label section above states how many are available.

## Active alerts

| Severity | Type | Message | Recommended human action |
| --- | --- | --- | --- |
| CRITICAL | FEATURE_DRIFT_CRITICAL | 73 of 80 tested features have drifted from the training reference. | Establish whether the inputs changed (pipeline, sensors, units) or the population did (an older fleet). Feature drift alone is not evidence the model is less accurate — check the performance report before considering a retrain. |
| WARNING | HIGH_CRITICAL_BATTERY_COUNT | 5 batteries are at a critical maintenance priority. | Review the critical cells individually against their measured capacity before scheduling any work. |

## Warnings

- The failure-risk model is marked experimental (it did not beat the cycle-index baseline out of fold). Risk probabilities are reported but were withheld from the maintenance rules.

## Definitions in force

- End of life: smoothed capacity at or below 70% of the nominal reference for 3 consecutive cycles.
- Failure risk: projected end-of-life crossing within 30 cycles — a derived label, not an observed safety event.
- Prediction intervals: split_conformal at 90% target coverage.
- Priority score: configurable weighted policy (weights {'risk': 0.3, 'rul': 0.2, 'rul_lower_bound': 0.2, 'soh': 0.15, 'trend': 0.05, 'uncertainty': 0.05, 'data_quality': 0.05}), normalised to 0–100. Not an optimum.

---

Fleet intelligence from a research prototype. Rankings, maintenance priorities and replacement horizons are configurable engineering policy applied to model outputs, not validated operational decisions, and not a substitute for battery-management-system protection or qualified engineering review.
