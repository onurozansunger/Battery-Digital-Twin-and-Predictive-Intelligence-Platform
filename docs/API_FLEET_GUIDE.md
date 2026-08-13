# Fleet API guide

The Milestone 2 endpoints are unchanged. This document covers what Milestone 3
added.

```bash
python -m battery_rul.api.app          # http://127.0.0.1:8000
open http://127.0.0.1:8000/docs        # OpenAPI
```

---

## Endpoints

### Operations

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | liveness — answers with no model loaded |
| GET | `/ready` | readiness — **503** when no artifact can answer |
| GET | `/version` | API, package and snapshot schema versions |
| GET | `/model-info` | loaded artifacts, fingerprints, definitions in force |
| GET | `/metrics` | Prometheus text exposition (`deployment.metrics_endpoint_enabled`) |

`/health` and `/ready` are deliberately different questions. A load balancer
needs to know whether to keep an instance in rotation (`/ready`); a supervisor
needs to know whether to restart the process (`/health`).

### Fleet — online scoring

| Method | Path | Returns |
| --- | --- | --- |
| POST | `/v1/fleet/snapshot` | full snapshot, battery records paged |
| POST | `/v1/fleet/rank` | ordered fleet by one criterion |
| POST | `/v1/fleet/maintenance-plan` | priorities, actions, workload forecast |
| POST | `/v1/fleet/replacement-plan` | candidates by horizon with caveats |
| POST | `/v1/fleet/monitoring/run` | data-quality and drift over the submitted fleet |

### Fleet — stored reads

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/v1/fleet/{fleet_id}/latest` | last stored snapshot, paged |
| GET | `/v1/fleet/{fleet_id}/summary` | aggregates only — the cheap poll |
| GET | `/v1/fleet/{fleet_id}/critical-batteries` | cells at a critical priority |
| GET | `/v1/fleet/{fleet_id}/alerts` | stored alerts, paged, filterable |
| GET | `/v1/monitoring/latest` | last monitoring snapshot |

### Registry

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/v1/models` | registered versions and stages |
| GET | `/v1/models/production` | the live model, or 404 with the command to promote one |
| POST | `/v1/admin/models/promote` | **403 unless `deployment.admin_endpoints_enabled`** |
| POST | `/v1/admin/models/rollback` | same |

---

## Request shape

```json
{
  "fleet_id": "DEMO-FLEET-01",
  "batteries": [
    {"battery_id": "B0005",
     "history": [{"cycle_index": 1, "capacity_ah": 1.86, "temperature_max_c": 38.2}]}
  ],
  "include_battery_records": true,
  "page": 1,
  "page_size": 50
}
```

Only `cycle_index` and `capacity_ah` are required per record. A real fleet does
not have impedance sweeps on every cell, and demanding them would make the API
unusable rather than making the data better; absent signals are reported in the
data-quality assessment.

---

## Pagination

Battery records are paged; **aggregates never are**, because a summary of a page
is not a summary of a fleet.

```json
{"batteries": {"items": [...],
  "pagination": {"page": 1, "page_size": 50, "total_items": 128,
                 "total_pages": 3, "has_next": true}}}
```

`total_items` is always the true total, so a client can tell a short page from a
short fleet. `page_size` is capped at `fleet.max_page_size`.

---

## Partial success is a 200

A fleet where nine cells failed is a **successful request that returns nine
failures**. Turning it into a 500 would lose the 119 that worked.

```json
{"battery_count": 128, "successfully_processed_count": 116,
 "failed_count": 9, "insufficient_data_count": 3,
 "ingestion_records": [{"battery_id": "B0042", "status": "failed",
                        "errors": ["3 row(s) share a cycle_index. …"]}],
 "warnings": ["9 battery/batteries failed and are excluded from every predicted-quantity aggregate: …"]}
```

Failed cells appear in `batteries` with `status: "failed"` **and** in
`ingestion_records`, and are excluded from every predicted-quantity denominator.

---

## Errors

| Status | When |
| --- | --- |
| 403 | administrative endpoint while disabled |
| 404 | no stored snapshot / no production model — the detail names the command to run |
| 409 | a registry conflict (illegal transition, several production entries) |
| 413 | more batteries than `fleet.max_batteries_per_request` — the detail names the batch command |
| 422 | request validation (duplicate ids, unknown fields, path separators, empty fleet) |
| 503 | no model artifact loaded, or no persistence backend configured |

Error bodies never contain a filesystem path; a test asserts it.

---

## Limits

| Limit | Default | Setting |
| --- | --- | --- |
| Batteries per online request | 100 | `fleet.max_batteries_per_request` |
| Hard schema cap | 500 | `MAX_BATTERIES_PER_REQUEST` (code) |
| Cycles per battery | 5 000 | `service.max_history_cycles` |
| Hard schema cap | 20 000 | `MAX_CYCLES_PER_BATTERY` (code) |
| Page size | 50 / max 500 | `fleet.page_size`, `fleet.max_page_size` |
| Request body | 32 MB | `deployment.max_request_bytes` (enforce at the proxy) |

Configuration may lower the schema caps; it may not raise them. Those caps bound
the memory a single unvalidated request can make the process allocate.

For anything larger, use the batch pipeline — the 413 says so.

---

## Client example

```python
import httpx, pandas as pd
from battery_rul.api.schemas import CycleRecord

allowed = set(CycleRecord.model_fields)
cycles = pd.read_parquet("data/processed/cycles.parquet")

def records(battery_id: str) -> list[dict]:
    frame = cycles.loc[cycles["battery_id"] == battery_id]
    frame = frame[[c for c in frame.columns if c in allowed]].copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = frame["timestamp"].astype(str)
    return [{k: (None if pd.isna(v) else v) for k, v in row.items()}
            for row in frame.to_dict(orient="records")]

payload = {"fleet_id": "NASA-COHORT",
           "batteries": [{"battery_id": b, "history": records(b)}
                         for b in ["B0005", "B0018"]]}

snapshot = httpx.post("http://127.0.0.1:8000/v1/fleet/snapshot",
                      json=payload, timeout=120).json()
print(snapshot["summary"]["median_rul"], snapshot["summary"]["critical_count"])
```

---

## Security defaults

* No authentication ships with this build. Put it in front of the service.
* CORS is an explicit allow-list; empty by default means no cross-origin access.
* Administrative endpoints are disabled by default and refuse in read-only mode.
* A request cannot name a file, a bundle, a model or a directory.

See `docs/SECURITY.md`.
