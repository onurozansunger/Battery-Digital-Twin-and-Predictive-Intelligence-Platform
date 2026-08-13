# Experiment tracking

One interface, two backends. The default needs no server.

---

## Why a file backend by default

A tracking system that needs infrastructure to record a five-second experiment
does not get used, and an untracked experiment is an experiment nobody can
reproduce. `artifacts/tracking/<experiment>/<run_id>.json` is enough to compare
runs, survives being copied around, and can be read without starting anything.

MLflow is supported and never required.

```yaml
tracking:
  backend: file          # or "mlflow"
  dir: artifacts/tracking
  experiment_name: battery-fleet
  mlflow_tracking_uri: null   # null → MLflow's local file store
```

`build_tracker(cfg)` returns the configured backend and **falls back to the file
tracker with a warning** when `mlflow` is selected but not installed. A missing
optional dependency must not stop a training run: a run tracked to a file is
strictly better than a run that did not happen.

---

## What a run records

```python
from battery_rul.tracking.experiment import tracked_run

with tracked_run(cfg, "rul-elastic-net", stage="milestone_2") as tracker:
    tracker.log_params({"model": "elastic_net", "alpha": 0.5, "features": 80})
    tracker.log_metrics({"mae": 8.561, "coverage": 0.917})
    tracker.log_artifact("bundle_metadata", cfg.artifacts.rul_dir / "metadata.json")
```

Every record carries, without the call site asking:

| Field | Why it is there |
| --- | --- |
| `run_id`, `experiment_name`, `started_at_utc`, `finished_at_utc`, `status` | identity and outcome |
| `git_revision` | which code |
| `data_fingerprint` | which data-affecting configuration |
| `dataset_fingerprint` | which source data |
| `seed` | reproducibility |
| `environment` | Python, platform and library versions |
| `params`, `metrics`, `tags`, `features` | what was run and what came out |
| `artifacts`, `tables`, `figures` | **relative** paths to what it produced |

A metric alone cannot answer "could I reproduce this?". These fields can.

`tracked_run` records a failure as `FAILED` and re-raises. A run that crashed and
left a `RUNNING` record is indistinguishable from one still going.

---

## What is never tracked

Raw cycle measurements. `log_params` and `log_metrics` drop `history`, `cycles`,
`frame`, `records` and `raw_data`, replacing them with an explicit omission
marker. `tracking.log_raw_data` is `False` and frozen. Tracking stores are
widely readable; a battery's telemetry in one is a copy of someone's operational
data in a place nobody is auditing.

Artifacts are **referenced, not copied** — copying would double every model
bundle on disk — and their paths are stored relative to the project root, so a
run record never carries a developer's home directory into a commit.

---

## Comparing runs

```python
from battery_rul.tracking import compare_runs
import pandas as pd

pd.DataFrame(compare_runs(cfg, limit=20))
```

```
run_id                            status    model_type   git_revision  metric.mae  metric.coverage
20260813T001204Z-rul-elastic-net  FINISHED  elastic_net  f476aeb            8.561            0.917
20260813T000731Z-rul-random-fore  FINISHED  random_fore  f476aeb            9.204            0.812
```

---

## Using MLflow

```yaml
tracking:
  backend: mlflow
  mlflow_tracking_uri: null        # local ./artifacts/tracking/mlruns
```

```bash
pip install mlflow
mlflow ui --backend-store-uri artifacts/tracking/mlruns --port 5000
# or, containerised:
docker compose --profile tracking up mlflow      # http://localhost:5000
```

For a remote server, set `mlflow_tracking_uri` to its URL and supply
credentials through the environment — never in a config file (`docs/SECURITY.md`).

The MLflow backend logs the same fields through MLflow's params/metrics/artifact
API and keeps the same `RunRecord` in memory, so switching backends does not
change what is recorded.

---

## Relationship to the registry

Tracking answers *what did I try, and what happened?* The registry answers
*which of them is live, and who decided that?* They are separate stores with
separate lifecycles: an experiment is immutable once finished, a registry entry
moves through stages. See `docs/MODEL_REGISTRY.md`.
