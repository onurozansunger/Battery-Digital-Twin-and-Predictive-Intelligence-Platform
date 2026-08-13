# Fleet data-quality monitoring

Milestone 2 asks "can I score this cell?" before scoring it. Milestone 3
aggregates those answers and asks a different question: **is this fleet's
telemetry healthy?**

---

## Not drift

A sensor that stopped reporting and a population that has genuinely aged both
move the numbers, and the remedies are opposite — one is fixed by an engineer
with a screwdriver, the other by retraining or by accepting the change. Filing
them under one heading guarantees the wrong remedy is applied about half the
time, so this module and `monitoring/drift.py` are separate, produce separate
statuses, and raise separate alert types.

---

## Per battery (Milestone 2, unchanged)

`digital_twin/quality.py` scores each history over independent checks —
history length, duplicate cycles, cycle ordering, cycle gaps, missing features,
value plausibility, distribution range — and classifies it `GOOD`,
`ACCEPTABLE`, `POOR` or `INSUFFICIENT`. `INSUFFICIENT` is a hard stop: no
prediction is produced.

---

## Per fleet

`monitoring/data_quality.py::summarise_fleet_data_quality` produces:

| Field | Meaning |
| --- | --- |
| `status` | OK / WARNING / CRITICAL / UNKNOWN |
| `quality_class_counts` | how many cells in each class |
| `mean_quality_score` | over cells that have one |
| `poor_or_worse_fraction` | POOR + INSUFFICIENT + failed, over the whole fleet |
| `insufficient_fraction` | cells that cannot be scored at all |
| `mean_missing_feature_fraction` | average share of required features absent |
| `per_feature_missing_rate` | which feature is missing, and how often — the actionable one |
| `check_failure_rates` | which check fails across the fleet |
| `batteries_with_schema_mismatch` | cells missing required features |
| `batteries_with_ood_features` | cells outside the training range |
| `denominator` | cells assessed |
| `warnings` | plain-language statements of each finding |

Failed cells count towards `poor_or_worse_fraction`: a cell that could not be
processed is a data-quality problem, and excluding it would make a fleet look
healthier the more of it broke.

---

## Thresholds

`monitoring.data_quality`:

| Setting | Warning | Critical |
| --- | --- | --- |
| `*_poor_fraction` | 0.10 | 0.25 |
| `*_insufficient_fraction` | 0.10 | 0.25 |
| `*_missing_feature_fraction` | 0.10 | 0.30 |

The worst of the three drives the status. An empty fleet is `UNKNOWN`, not
`OK` — nothing was checked.

---

## Out-of-distribution features

Cells with features outside the training reference range are named, with the
warning:

> N cell(s) have at least one feature outside the training reference range; those
> predictions are extrapolations. This is an input-range observation, not a drift
> verdict — see the feature-drift report for that.

The distinction matters. One cell outside the range is an extrapolation for that
cell; a *shifted distribution* across the fleet is drift. The first is a caveat
on a prediction, the second is a question about the model.

---

## Where it surfaces

* `FleetSnapshot.data_quality` — on every snapshot, no extra run needed
* `reports/milestone_3/data_quality_report.json` — written by the monitoring run
* `MonitoringSnapshot.data_quality_summary` — persisted with the run
* Dashboard → Data Quality — class counts, per-feature missingness, check
  failure rates, and the list of cells that could not be processed with their
  errors
* Alerts `DATA_QUALITY_WARNING` / `DATA_QUALITY_CRITICAL`, whose recommended
  action names the telemetry pipeline rather than the model
