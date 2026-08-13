# syntax=docker/dockerfile:1.7
# (The syntax directive is honoured by BuildKit and ignored by the legacy
#  builder; nothing below requires BuildKit — see the dependency layer.)
#
# Multi-stage, multi-target image for the Battery Digital Twin & Fleet
# Intelligence platform.
#
#   docker build -f docker/Dockerfile --target api       -t battery-rul-api .
#   docker build -f docker/Dockerfile --target dashboard -t battery-rul-dashboard .
#   docker build -f docker/Dockerfile --target jobs      -t battery-rul-jobs .
#
# Decisions worth knowing before you change them
# ----------------------------------------------
# *CPU torch.* The PyTorch CPU index is used deliberately: the default wheels
#   pull CUDA runtime libraries worth several gigabytes that a CPU inference
#   container will never execute.
#
# *No dataset, no artifacts baked in.* `.dockerignore` excludes data/, models/
#   and artifacts/. Model bundles are **mounted** at runtime (see
#   docker-compose.yml) or copied in by an explicit build argument. A container
#   image with a 200 MB dataset in it is an image nobody can rebuild quickly and
#   a distribution channel for data that may not be redistributable.
#
# *Non-root.* Everything runs as `appuser`. The application never needs to write
#   inside the image; the only writable locations are the mounted volumes.
#
# *Startup does not train.* Every entry point loads artifacts and serves. If the
#   artifacts are missing the API still starts and reports not-ready, which is
#   what a load balancer needs in order to keep the instance out of rotation.
#
# *The Python minor version is part of the contract.* Model bundles are pickles,
#   and pickles are not portable across Python minor versions — a bundle written
#   on 3.13 fails on 3.12 with `ModuleNotFoundError: pathlib._local`, which names
#   nothing useful. This image tracks the interpreter the bundles are built with;
#   `load_bundle` also records and checks it, so a mismatch is a sentence rather
#   than a puzzle. Change this line only together with the build environment.

# ---------------------------------------------------------------------------
# Stage 1 — builder: compile wheels into a virtual environment
# ---------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential is needed by a few scientific wheels on this base image; it
# stays in the builder stage and never reaches the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependency layer first, so editing source does not reinstall torch.
#
# No `--mount=type=cache` here on purpose: it is a BuildKit-only feature, and a
# Dockerfile that cannot be built by a plain `docker build` is a Dockerfile that
# fails on whichever machine does not have buildx installed. The layer ordering
# is what actually saves the rebuild time; the pip cache only helps when the
# dependency set itself changes. CI still gets registry-layer caching through
# buildx.
COPY pyproject.toml README.md ./
COPY src/battery_rul/__init__.py src/battery_rul/__init__.py
RUN pip install --upgrade pip setuptools wheel \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu .

# Now the real source.
COPY src/ src/
COPY configs/ configs/
RUN pip install --no-deps .

# ---------------------------------------------------------------------------
# Stage 2 — runtime base
# ---------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    BATTERY_RUL_ROOT=/app \
    BATTERY_RUL_LOG_FORMAT=json

# libgomp is required by LightGBM/XGBoost at runtime. curl is used by the
# container health checks and nothing else.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser configs/ /app/configs/
COPY --chown=appuser:appuser docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Mount points. Owned by appuser so a read-only root filesystem still works with
# writable volumes attached here.
RUN mkdir -p /app/artifacts /app/data /app/reports /app/models \
    && chown -R appuser:appuser /app

USER appuser
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# ---------------------------------------------------------------------------
# Stage 3 — API
# ---------------------------------------------------------------------------
FROM runtime AS api

ENV BATTERY_RUL_CONFIG=/app/configs/default.yaml \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000
# /health answers even with no model loaded — the process being alive is what a
# liveness probe should test. Readiness is a separate endpoint (/ready) and a
# separate decision, made by the orchestrator.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${API_PORT}/health" || exit 1
CMD ["api"]

# ---------------------------------------------------------------------------
# Stage 4 — dashboard
# ---------------------------------------------------------------------------
FROM runtime AS dashboard

ENV BATTERY_RUL_CONFIG=/app/configs/default.yaml \
    DASHBOARD_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${DASHBOARD_PORT}/_stcore/health" || exit 1
CMD ["dashboard"]

# ---------------------------------------------------------------------------
# Stage 5 — batch and monitoring jobs
# ---------------------------------------------------------------------------
FROM runtime AS jobs

ENV BATTERY_RUL_CONFIG=/app/configs/default.yaml

# No HEALTHCHECK: a batch job that exits 0 has succeeded, and a health check on
# a finite process reports unhealthy the moment it finishes.
CMD ["fleet-batch"]
