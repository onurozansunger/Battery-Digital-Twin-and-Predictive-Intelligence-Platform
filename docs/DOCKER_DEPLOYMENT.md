# Docker deployment

> **Built and run.** All three images were built with `colima` (Docker 29.5.2,
> linux/arm64) and exercised: the API image serves `/health`, `/ready` and the
> fleet endpoints with artifacts mounted, the dashboard image reports healthy,
> and the jobs image ran a real fleet batch and a real monitoring run under a
> read-only root filesystem. Measured size: **3.73 GB** per image (they share
> every layer below the target, so the three together are not 11 GB on disk).

---

## Images

One multi-stage `docker/Dockerfile` with three targets:

| Target | Runs | Port | Health check |
| --- | --- | --- | --- |
| `api` | uvicorn → FastAPI | 8000 | `GET /health` |
| `dashboard` | Streamlit fleet dashboard | 8501 | `GET /_stcore/health` |
| `jobs` | batch / monitoring / report CLI | — | none (a finite process) |

```bash
docker build -f docker/Dockerfile --target api       -t battery-rul-api .
docker build -f docker/Dockerfile --target dashboard -t battery-rul-dashboard .
docker build -f docker/Dockerfile --target jobs      -t battery-rul-jobs .
```

---

## Decisions worth knowing

**CPU torch.** The build installs from `https://download.pytorch.org/whl/cpu`.
The default wheels pull CUDA runtime libraries worth several gigabytes that a
CPU inference container will never execute.

**No dataset, no artifacts baked in.** `.dockerignore` excludes `data/`,
`models/`, `artifacts/` and `reports/`. Model bundles are **mounted** at run
time. An image carrying a trained model has to be rebuilt to ship a retrain, and
its provenance stops being checkable — the registry records which bundle is
live, and that only means something if the bundle is a mounted artifact rather
than a layer. The Docker workflow asserts no dataset or model file is present in
the built image.

**Non-root.** Everything runs as `appuser` (uid 10001). Verified:
`docker inspect --format '{{.Config.User}}'` returns `appuser`.

**Read-only root filesystem.** The compose file sets `read_only: true` with a
256 MB `/tmp` tmpfs and writable volume mounts. Verified: `touch /root-test`
inside the running API container returns `Read-only file system`, and the
monitoring job completes with exit 0 under it.

Running under a read-only root exposed a real defect during this validation:
`PathsConfig.ensure()` used to create *every* configured directory and raise if
it could not, so a job with only two writable mounts died on `mkdir
/app/data/raw` — a directory it never reads. Directory creation is now
best-effort and logged; the *use* of a missing directory still fails loudly with
the path named.

**Startup never trains.** Every entry point loads artifacts and serves. Verified
with no artifacts mounted: `/health` -> **200**, `/ready` -> **503** with
`"No RUL-capable artifact is loaded"` and the four missing bundles named, while
Docker's own HEALTHCHECK reports `healthy`. That is what a load balancer needs
in order to keep the instance out of rotation instead of black-holing requests.

**The Python minor version is part of the contract.** Model bundles are pickles
and pickles are not portable across Python minor versions. The first build of
this image used Python 3.12 against bundles pickled by 3.13 and failed with
`ModuleNotFoundError: No module named 'pathlib._local'` — an error naming
nothing useful. Two changes came out of that: the image now tracks the
interpreter the bundles are built with, and `load_bundle` records
`python_version` in the bundle metadata and refuses a mismatch with a message
naming both versions. **If you change the base image's Python, rebuild the
bundles.**

**Dependency layer first.** `pyproject.toml` and the package `__init__` are
copied before the source, so editing a module does not reinstall torch.

---

## Compose

```bash
docker compose build
docker compose up api dashboard                 # long-running services
docker compose --profile jobs run --rm fleet-batch
docker compose --profile jobs run --rm monitoring
docker compose --profile tracking up mlflow     # optional MLflow UI on :5000
```

| Service | Profile | Notes |
| --- | --- | --- |
| `api` | default | published on `127.0.0.1:8000` only |
| `dashboard` | default | waits for `api` to be healthy |
| `fleet-batch` | `jobs` | one-shot, `restart: "no"` |
| `monitoring` | `jobs` | one-shot |
| `mlflow` | `tracking` | not started by default — the file tracker is the default backend |

Ports bind to `127.0.0.1`, not `0.0.0.0`: this build ships no authentication, so
the default must not be reachable from the network. See `docs/SECURITY.md`.

### Volumes

| Host | Container | Mode | Why |
| --- | --- | --- | --- |
| `./artifacts` | `/app/artifacts` | rw | bundles, registry, monitoring, database |
| `./reports` | `/app/reports` | rw | generated reports |
| `./data/processed` | `/app/data/processed` | **ro** | the fleet source; the service never writes it |

Published ports are configurable (`API_PORT_HOST`, `DASHBOARD_PORT_HOST`,
defaulting to 8000 and 8501) because a developer machine usually has something
on those ports already, and a hardcoded port turns that into a failed
`compose up` rather than a one-variable fix:

```bash
DASHBOARD_PORT_HOST=8502 docker compose up -d api dashboard
```

---

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `BATTERY_RUL_ROOT` | `/app` | project root inside the container |
| `BATTERY_RUL_CONFIG` | `/app/configs/default.yaml` | configuration file |
| `BATTERY_RUL_LOG_FORMAT` | `json` | structured logs for a collector |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | API bind |
| `DASHBOARD_PORT` | `8501` | dashboard bind |
| `FLEET_ID` | `DEMO-FLEET-01` | fleet for the job images |
| `FLEET_SOURCE` | `processed` | `processed`, `demo` or `file` |
| `LOG_LEVEL` | `info` | uvicorn log level |

No secret is required to run any of this, and none is baked into any image.

---

## First run

```bash
# On the host: build the artifacts the containers will mount
python -m battery_rul.pipelines.run_milestone_2 --config configs/default.yaml
python -m battery_rul.pipelines.build_reference --config configs/default.yaml

docker compose build
docker compose --profile jobs run --rm fleet-batch      # produces a snapshot
docker compose up -d api dashboard

curl -s localhost:8000/health
curl -s localhost:8000/ready | jq
open http://localhost:8501
```

If `/ready` returns 503, the artifacts are not mounted or are incompatible —
the response body says which. Do not "fix" it by disabling
`artifacts.strict_compatibility`; see the incident notes in `docs/SECURITY.md`.

---

## Scheduling the jobs

The job image is designed to be run by whatever scheduler you already have
(cron, systemd timers, a CI schedule, a Kubernetes CronJob):

```bash
docker run --rm \
  -v "$PWD/artifacts:/app/artifacts" -v "$PWD/reports:/app/reports" \
  -v "$PWD/data/processed:/app/data/processed:ro" \
  -e FLEET_ID=DEMO-FLEET-01 \
  battery-rul-jobs:local monitoring
```

Exit code 0 means the run succeeded. No scheduler is shipped: adding one here
would be an opinion about someone else's infrastructure.

---

## What is not provided

* **Kubernetes manifests.** The repository has no cluster to validate them
  against, and unvalidated manifests are a liability.
* **Registry publishing.** Needs credentials this repository does not hold. The
  release workflow builds images and stops.
* **A metrics stack.** `/metrics` is exposed for a Prometheus you already run.
* **Horizontal scaling notes.** The API is stateless apart from the SQLite file;
  running several replicas against one SQLite database on a shared volume works
  for reads but is not a design this repository has tested. Point
  `persistence.database_path` at per-replica storage, or replace the repository
  implementation, before doing it.
