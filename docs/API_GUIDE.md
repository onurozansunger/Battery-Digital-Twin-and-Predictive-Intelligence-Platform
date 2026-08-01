# API guide

## Run it

```bash
python -m battery_rul.api.app
```

or, with explicit control:

```bash
uvicorn "battery_rul.api.app:create_app" --factory --host 127.0.0.1 --port 8000
```

Interactive documentation: `http://127.0.0.1:8000/docs`.
Machine-readable schema: `http://127.0.0.1:8000/openapi.json`.

Host and port come from `service.api_host` / `service.api_port`.

The application never trains anything at startup. It loads and validates
artifacts; if none are usable it still starts, `/health` answers, and `/ready`
returns 503 with the reason — which is what a load balancer needs in order to
keep the instance out of rotation rather than black-holing requests.

## Endpoints

### Operations

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness. 200 whenever the process is up. |
| GET | `/ready` | Readiness. 503 when no artifact can answer a prediction. |
| GET | `/version` | API, package and snapshot-schema versions; git revision. |
| GET | `/model-info` | Loaded bundles, fingerprints, and every definition in force. |

### Prediction

| Method | Path | Returns |
|---|---|---|
| POST | `/v1/predict/rul` | RUL in cycles + conformal prediction interval |
| POST | `/v1/predict/soh` | SOH as a fraction + health band |
| POST | `/v1/predict/risk` | Calibrated end-of-life-within-horizon probability |
| POST | `/v1/predict/full` | The complete `BatteryTwinSnapshot` |
| POST | `/v1/digital-twin/snapshot` | Alias of `/predict/full` |
| POST | `/v1/explain` | Local feature attributions |

## Request

```json
{
  "battery_id": "B0007",
  "include_explanation": true,
  "history": [
    {"cycle_index": 1, "capacity_ah": 1.856, "voltage_mean_v": 3.31,
     "temperature_max_c": 38.7, "discharge_duration_s": 3690.0},
    {"cycle_index": 2, "capacity_ah": 1.846}
  ]
}
```

Only `cycle_index` and `capacity_ah` are required per record. Every other sensor
is optional — a real fleet does not have impedance sweeps on every cell, and
demanding them would make the API unusable rather than making the data better.
Absent signals appear in `measurement_summary.missing_signals` and are counted by
the data-quality assessment.

Unknown fields are rejected (422) rather than silently dropped, so a typo is an
error instead of a quiet omission.

## Response — `/v1/predict/full`

```json
{
  "snapshot": {
    "battery_id": "B0007",
    "generated_at_utc": "2026-08-01T11:20:33.412Z",
    "identity":            { "battery_id": "B0007", "chemistry": "LCO 18650 (NASA cohort)", "provenance": "observed" },
    "measurement_summary": { "latest_cycle": 121, "measured_capacity_ah": 1.42, "provenance": "observed" },
    "health":              { "soh": 0.846, "soh_percent": 84.6, "health_class": "warning",
                             "soh_measured": 0.851, "provenance": "predicted" },
    "prediction": {
      "rul_cycles": 38.0,
      "rul_interval": {
        "point_estimate": 38.0, "lower_bound": 29.1, "upper_bound": 48.7,
        "interval_coverage_target": 0.9, "uncertainty_method": "split_conformal_by_life_stage",
        "calibration_sample_size": 412, "life_stage": "late",
        "interval_type": "prediction_interval", "provenance": "estimated"
      },
      "provenance": "predicted"
    },
    "failure_risk": {
      "horizon_cycles": 30, "probability": 0.71, "probability_raw": 0.83,
      "is_calibrated": true, "calibration_method": "isotonic",
      "decision_threshold": 0.42, "exceeds_threshold": true, "risk_class": "high",
      "label_definition": "Derived label: … Not an observed safety failure …",
      "provenance": "predicted"
    },
    "explanation":    { "method": "shap_tree", "drivers": [ … ], "caveat": "…" },
    "recommendation": { "action_code": "SCHEDULE_INSPECTION", "priority": "medium",
                        "suggested_window_cycles": [10, 20], "evidence": [ … ],
                        "disclaimer": "…", "provenance": "rule_based" },
    "data_quality":   { "quality_score": 0.85, "quality_class": "GOOD", "warnings": [ … ],
                        "provenance": "derived" },
    "metadata":       { "snapshot_schema_version": "2.0", "model_name": "…",
                        "feature_pipeline_fingerprint": "…", "first_scoreable_cycle": 25, … },
    "warnings": [],
    "disclaimer": "Research prototype for engineering decision support. …"
  }
}
```

Every value carries a `provenance` tag: `observed`, `derived`, `predicted`,
`estimated` or `rule_based`. Never render a `predicted` value as though it were
a measurement.

## Errors

Every non-2xx response has the same shape:

```json
{
  "error": "invalid_history",
  "detail": "3 row(s) share a cycle_index. …",
  "request_id": "9f0c…",
  "hint": "See the canonical cycle schema in battery_rul.data.schema."
}
```

| Status | `error` | Cause |
|---|---|---|
| 422 | validation error | schema violation — missing field, unknown field, out-of-range value |
| 422 | `invalid_history` | duplicate cycles, missing required column, empty or oversized history |
| 503 | `models_unavailable` | no usable artifact |

A **thin history is not an error.** It returns 200 with
`data_quality_class: "INSUFFICIENT"`, a null prediction and the
`INSUFFICIENT_DATA` recommendation — the twin's answer is "I cannot tell you",
and that is a result, not a server failure.

## Request tracing

Send `X-Request-ID`; it is echoed on the response and appears in the logs. If you
do not send one, a UUID is generated and returned. Request bodies are never
logged — only method, path, status and duration.

## Client example

```python
import httpx

history = [{"cycle_index": i, "capacity_ah": c} for i, c in enumerate(capacities, start=1)]
response = httpx.post(
    "http://127.0.0.1:8000/v1/digital-twin/snapshot",
    json={"battery_id": "B0007", "history": history},
    timeout=60.0,
)
response.raise_for_status()
snapshot = response.json()["snapshot"]
print(snapshot["prediction"]["rul_cycles"], snapshot["recommendation"]["action_code"])
```

## Not implemented (by design, at this milestone)

Authentication, multi-tenancy, rate limiting, TLS termination, cloud deployment
and drift monitoring are Milestone 3 concerns. Do not expose this service
publicly as it stands.
