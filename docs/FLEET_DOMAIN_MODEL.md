# The fleet domain model

Every object in `battery_rul/fleet/domain.py` is a Pydantic model with
`extra="forbid"`, JSON-serialisable by construction. These objects are
simultaneously the API response body, the persisted document and the dashboard's
input — one shape, so the three cannot disagree.

---

## Two conventions that run through everything

**Provenance.** Inherited from Milestone 2: every value is tagged `observed`,
`derived`, `predicted`, `estimated` or `rule_based`. A page that renders a
measured 84 % SOH beside a predicted 38-cycle RUL in the same typeface invites
the reader to trust both equally, and one of them has a 19-cycle interval around
it.

**Denominators.** Every aggregate carries the count it was computed over.
"Median RUL 94 cycles" is a different claim depending on whether it covers 128
cells or the 103 that could be scored, and only the second is computable.

---

## Enumerations

```python
ProcessingStatus     success | insufficient_data | failed
MaintenancePriority  P0_CRITICAL … P5_MONITOR | INSUFFICIENT_DATA
ReplacementHorizon   near_term | medium_term | long_term | not_flagged | unknown
MonitoringStatus     OK | WARNING | CRITICAL | UNKNOWN
```

`MaintenancePriority.severity` orders the ladder, with `INSUFFICIENT_DATA` last:
it is not a severity, it is an absence of evidence, and ranking it beside real
severities would either hide critical cells or invent urgency.

`MonitoringStatus.worst()` combines statuses by taking the maximum severity,
never an average — averaging a CRITICAL with three OKs produces a number that
means nothing.

---

## Objects

| Object | Holds |
| --- | --- |
| `FleetIdentity` | fleet id, source, `is_demo_data`, data notice |
| `FleetBatteryReference` | a pointer to a cell's history, without the measurements |
| `BatteryIngestionRecord` | per-cell ingestion outcome; a `failed` record must carry an error |
| `FleetIngestionResult` | accepted references, all records, fingerprint, source metadata |
| `ScoreComponent` | one term of the priority score with its transformation |
| `InspectionRecommendation` | cycles, optional days, basis, assumptions |
| `BatteryPriorityRecord` | priority, score, breakdown, triggered rules, evidence, action |
| `ReplacementCandidate` | horizon, confidence, evidence, caveats |
| `FleetBatteryRecord` | one cell's row in a snapshot |
| `FleetHealthDistribution` / `FleetRiskDistribution` | class counts with denominators |
| `FleetMaintenanceSummary` | priority and action counts |
| `FleetReplacementSummary` | candidates by horizon with uncertainty brackets |
| `FleetWorkloadForecast` / `WorkloadBucket` | demand by horizon |
| `FleetStatistics` | medians, means, quantiles, thresholds, missingness |
| `FleetDataQualitySummary` | input-quality status and detail |
| `FleetDriftStatus` | a pointer to the drift verdict and its monitoring snapshot |
| `FleetModelMetadata` | active version, registry stage, fingerprints, definitions |
| `FleetSummary` | the executive numbers |
| `FleetSnapshot` | all of the above, plus per-battery records |
| `FleetTrendPoint` | one point of a trend series, with its denominator |

---

## `FleetBatteryRecord`

A compact projection of the full `BatteryTwinSnapshot` plus the fleet layer's
verdicts. The full snapshot is **not** embedded: a 128-cell fleet would produce
a response measured in megabytes, and the battery-level endpoint already returns
it for the one cell a reader drills into.

Fields group into measured (`measured_soh`, `capacity_fade_percent`,
`health_class`), predicted (`predicted_rul`, bounds, `failure_risk`,
`predicted_soh_forecast`), derived trends, quality, and decisions (`priority`,
`priority_record`, `replacement`, `recommended_action`).

`twin_action_code` keeps the Milestone 2 battery-level recommendation distinct
from the fleet-level action, so the two layers stay separately auditable.

Validation: the interval must bracket the point estimate. A record that violates
it raises at construction rather than propagating into a plan.

---

## `FleetSnapshot`

```python
fleet_id, snapshot_id, generated_at_utc, schema_version
identity, battery_count, successfully_processed_count,
failed_count, insufficient_data_count
batteries: list[FleetBatteryRecord]      # every submitted cell, including failures
summary, health_distribution, risk_distribution
maintenance_summary, replacement_summary, workload_forecast
fleet_statistics, data_quality, drift_status, model_metadata
data_fingerprint, batch_id, processing_duration_ms
warnings, disclaimer
```

Validators: `battery_count` must equal `len(batteries)` when records are
present, and no battery may appear twice.

Helpers: `to_json_dict()`, `battery(id)`, `evaluated()`, `without_batteries()`.

Schema version **3.0**. The Milestone 2 battery snapshot stays at 2.0 — a
regression test asserts it did not move.

---

## Serialisation contract

```python
snapshot.to_json_dict()                 # exactly what the API returns and the store keeps
FleetSnapshot(**json.loads(text))       # round-trips without loss
```

A test asserts `restored.to_json_dict() == original.to_json_dict()`.
