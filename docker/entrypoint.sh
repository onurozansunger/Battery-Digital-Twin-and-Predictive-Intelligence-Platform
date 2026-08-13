#!/usr/bin/env bash
# Container entry point: one script, one named command per role.
#
# Named commands rather than raw shell so the compose file, the documentation and
# the image all refer to the same thing, and so a typo in a long uvicorn
# invocation cannot silently change what the API image serves.
set -euo pipefail

CONFIG="${BATTERY_RUL_CONFIG:-/app/configs/default.yaml}"
FLEET_ID="${FLEET_ID:-DEMO-FLEET-01}"
FLEET_SOURCE="${FLEET_SOURCE:-processed}"

usage() {
    cat <<'EOF'
Usage: <command> [args...]

  api             Serve the FastAPI application (uvicorn).
  dashboard       Serve the Streamlit fleet dashboard.
  fleet-batch     Score a fleet offline and persist the snapshot.
  monitoring      Run the monitoring suite over a fresh fleet batch.
  build-reference Build the drift reference from the training partition.
  report          Render the Markdown fleet report.
  shell           Drop into a shell (debugging only).

Environment:
  BATTERY_RUL_CONFIG  configuration file            (default /app/configs/default.yaml)
  API_HOST/API_PORT   API bind address              (default 0.0.0.0:8000)
  DASHBOARD_PORT      dashboard port                (default 8501)
  FLEET_ID            fleet identifier for jobs     (default DEMO-FLEET-01)
  FLEET_SOURCE        processed | demo | file       (default processed)
EOF
}

command="${1:-api}"
shift || true

case "${command}" in
    api)
        exec uvicorn "battery_rul.api.app:create_app" \
            --factory \
            --host "${API_HOST:-0.0.0.0}" \
            --port "${API_PORT:-8000}" \
            --log-level "${LOG_LEVEL:-info}" \
            "$@"
        ;;
    dashboard)
        exec streamlit run /opt/venv/lib/python3.13/site-packages/battery_rul/dashboard/fleet_app.py \
            --server.port "${DASHBOARD_PORT:-8501}" \
            --server.address "${DASHBOARD_HOST:-0.0.0.0}" \
            --server.headless true \
            "$@"
        ;;
    fleet-batch)
        exec python -m battery_rul.pipelines.run_fleet_batch \
            --config "${CONFIG}" --fleet-id "${FLEET_ID}" --source "${FLEET_SOURCE}" "$@"
        ;;
    monitoring)
        exec python -m battery_rul.pipelines.run_monitoring \
            --config "${CONFIG}" --fleet-id "${FLEET_ID}" --source "${FLEET_SOURCE}" "$@"
        ;;
    build-reference)
        exec python -m battery_rul.pipelines.build_reference --config "${CONFIG}" "$@"
        ;;
    report)
        exec python -m battery_rul.pipelines.generate_fleet_report \
            --config "${CONFIG}" --fleet-id "${FLEET_ID}" "$@"
        ;;
    shell)
        exec /bin/bash "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        # Anything else is executed verbatim, so `docker run image python -c ...`
        # still works without a special case per tool.
        exec "${command}" "$@"
        ;;
esac
