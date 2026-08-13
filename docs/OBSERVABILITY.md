# Observability

Structured logs and in-process metrics. Two modules, kept separate because they
answer different questions: logs explain *what happened to this request*, metrics
explain *what is happening to the service*.

---

## Structured logging

`battery_rul/observability/logging.py`.

```python
from battery_rul.observability.logging import bind_context, log_event

with bind_context(fleet_id="DEMO-FLEET-01", batch_id=batch, model_version="1.0.0"):
    log_event(logger, "fleet_batch_started", battery_count=128)
    ...
    log_event(logger, "fleet_batch_completed", duration_ms=8421.3, failed_count=9)
```

```json
{"timestamp": "2026-08-13T00:07:44.812+00:00", "level": "INFO",
 "logger": "battery_rul.fleet.inference", "message": "fleet_batch_completed",
 "service": "battery-rul", "fleet_id": "DEMO-FLEET-01",
 "batch_id": "20260813T000743Z-2c3a5863", "model_version": "1.0.0",
 "event": "fleet_batch_completed", "status": "ok", "duration_ms": 8421.3,
 "battery_count": 128, "failed_count": 9}
```

### Why a context variable

A fleet batch calls the twin once per cell, which calls feature generation,
which logs. Threading `batch_id` down that chain would put an operational
concern into six function signatures that have nothing to do with operations. A
`contextvars.ContextVar` binds it once at the boundary and every line emitted
underneath carries it — including from code that knows nothing about this module.

Contexts nest and inherit: a per-battery block inside a batch keeps the batch's
identifiers and adds its own.

### Fields

`timestamp`, `level`, `logger`, `message`, `service`, `request_id`, `batch_id`,
`fleet_id`, `battery_id`, `model_version`, `event`, `status`, `duration_ms`,
`error_code`, plus whatever the call site passes.

`event` is a stable machine-readable name (`fleet_batch_completed`), not a
sentence: dashboards group on it, and a message that changes wording breaks
every saved query built on it.

### What is never logged

Raw cycle histories. The formatter drops any `history`, `records`, `frame`,
`cycles`, `payload` or `raw` field defensively, replacing it with
`"<omitted: raw measurement data is never logged>"`, so an accidental
`extra={"history": frame}` cannot leak a cell's telemetry. A test asserts it.

Request bodies are never logged by the API middleware — path, status and
duration only.

### Enabling JSON output

```bash
export BATTERY_RUL_LOG_FORMAT=json     # containers set this
python -c "from battery_rul.observability import configure_structured_logging; configure_structured_logging()"
```

Interactive runs keep the Rich console handler from
`battery_rul/utils/logging.py`; a log line read by a person and a log line
parsed by a collector want different things.

---

## Metrics

`battery_rul/observability/metrics.py` — counters, gauges and histograms with
string labels, and a Prometheus text rendering.

**Not `prometheus_client`.** The only thing this project needs from it is the
exposition format, which is a dozen lines of string formatting, and its global
default registry fights with test isolation.

| Metric | Type | Labels |
| --- | --- | --- |
| `api_requests_total` | counter | method, path, status |
| `api_request_duration_seconds` | histogram | method, path |
| `battery_inference_total` | counter | — |
| `battery_inference_failures_total` | counter | — |
| `battery_insufficient_data_total` | counter | — |
| `battery_inference_duration_seconds` | histogram | — |
| `fleet_batch_duration_seconds` | histogram | fleet_id |
| `fleet_critical_battery_count` | gauge | fleet_id |
| `monitoring_alerts_total` | counter | type, severity |

Requests are labelled by **route template**, not URL: labelling by path would
create a time series per fleet id and blow up the cardinality.

```bash
curl -s localhost:8000/metrics
```

```
# HELP api_requests_total HTTP requests handled, by route and status.
# TYPE api_requests_total counter
api_requests_total{method="POST",path="/v1/fleet/snapshot",status="200"} 12.0
# TYPE fleet_batch_duration_seconds histogram
fleet_batch_duration_seconds_bucket{fleet_id="DEMO-FLEET-01",le="10.0"} 3
```

Controlled by `deployment.metrics_enabled` and
`deployment.metrics_endpoint_enabled`. No Prometheus server is shipped — the
endpoint is there for one to scrape if you run one.

---

## What to watch

| Signal | Why |
| --- | --- |
| `battery_inference_failures_total` rising | a data-source change or a broken artifact |
| `battery_insufficient_data_total` rising | telemetry thinning out before it becomes visible in a median |
| `fleet_critical_battery_count` jumping | either the fleet aged or a threshold changed — check the config commit |
| `api_request_duration_seconds` p99 | a fleet request approaching a client timeout |
| `monitoring_alerts_total{severity="CRITICAL"}` | anything at all |
| `/ready` returning 503 | the model is not loaded; see `docs/SECURITY.md` incident notes |

---

## Tracing

Not implemented. A single-node batch platform with structured logs carrying a
shared `batch_id` gives most of what a trace would, and OpenTelemetry here would
be a dependency and a collector to run for a correlation the `batch_id` already
provides. If the platform ever spans services, this is the first thing to add.
