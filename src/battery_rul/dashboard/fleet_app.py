"""Streamlit fleet intelligence dashboard — layout only, no model logic.

Run it with::

    streamlit run src/battery_rul/dashboard/fleet_app.py

Every number on every page comes from a :class:`FleetSnapshot` produced by the
fleet service, or from a stored monitoring snapshot. Nothing is recomputed here:
a dashboard that recalculates a priority is a second implementation of the
policy, and the two will disagree.

Honesty rules this dashboard follows
------------------------------------
* measured and predicted quantities are labelled as such, everywhere;
* denominators are shown beside aggregates ("median over 103 of 128 cells");
* demo fleets carry a banner on every page, not a footnote on one;
* the risk probability is shown greyed and marked when the model is
  experimental, because that model is excluded from the decision rules;
* no page offers a one-click promotion unless administrative actions are
  explicitly enabled in configuration.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from battery_rul.config import ExperimentConfig, load_config
from battery_rul.dashboard.fleet_adapter import (
    FleetDashboardAdapter,
    battery_table,
    score_breakdown_table,
    workload_table,
)
from battery_rul.fleet.analytics import fleet_trend_series
from battery_rul.fleet.demo import DEMO_NOTICE
from battery_rul.fleet.domain import FleetSnapshot

PAGES = [
    "Executive Fleet Overview",
    "Battery Ranking",
    "Critical Batteries",
    "Maintenance Planning",
    "Replacement Planning",
    "Fleet Trends",
    "Data Quality",
    "Feature Drift",
    "Prediction Drift",
    "Model Performance",
    "Model Registry",
    "Monitoring Alerts",
    "Battery Digital Twin",
    "Architecture & Limitations",
]

DISCLAIMER = (
    "Fleet intelligence from a research prototype. Rankings, maintenance priorities "
    "and replacement horizons are configurable engineering policy applied to model "
    "outputs — decision support, not validated operational decisions, and not a "
    "substitute for battery-management-system protection or qualified engineering "
    "review."
)

_STATUS_BADGE = {
    "OK": "● OK",
    "WARNING": "◑ WARNING",
    "CRITICAL": "▲ CRITICAL",
    "UNKNOWN": "? UNKNOWN",
}
_PRIORITY_BADGE = {
    "P0_CRITICAL": "▲ P0 CRITICAL",
    "P1_URGENT": "▲ P1 URGENT",
    "P2_HIGH": "◑ P2 HIGH",
    "P3_MEDIUM": "◐ P3 MEDIUM",
    "P4_LOW": "● P4 LOW",
    "P5_MONITOR": "◌ P5 MONITOR",
    "INSUFFICIENT_DATA": "? INSUFFICIENT DATA",
}


@st.cache_resource(show_spinner=False)
def _config(path: str) -> ExperimentConfig:
    from pathlib import Path

    candidate = Path(path)
    return load_config(candidate if candidate.is_file() else None)


@st.cache_resource(show_spinner="Loading model artifacts…")
def _adapter(path: str) -> FleetDashboardAdapter:
    return FleetDashboardAdapter.build(_config(path))


def main() -> None:  # pragma: no cover - exercised by the import smoke test
    st.set_page_config(page_title="Battery Fleet Intelligence", page_icon="🔋", layout="wide")
    config_path = st.sidebar.text_input("Configuration", value="configs/default.yaml")
    cfg = _config(config_path)
    adapter = _adapter(config_path)

    st.sidebar.markdown("### Service status")
    readiness = adapter.readiness()
    if readiness.get("ready"):
        st.sidebar.success("● Models loaded and ready")
    else:
        st.sidebar.error("▲ Models unavailable — predicted fields will be empty")
        for name, message in (readiness.get("errors") or {}).items():
            st.sidebar.caption(f"{name}: {message}")

    fleet_id = st.sidebar.text_input("Fleet id", value=cfg.fleet.default_fleet_id)
    source = st.sidebar.selectbox(
        "Fleet source",
        ["stored snapshot", "processed cycles (measured)", "demo fleet (synthetic)"],
        help="'Stored snapshot' reads the last batch. The other two score a fleet now.",
    )
    demo_size = st.sidebar.number_input("Demo fleet size", 4, 200, 24, step=4)

    snapshot = _resolve_snapshot(adapter, fleet_id, source, int(demo_size))
    page = st.sidebar.radio("Page", PAGES, index=0)

    st.title("🔋 Battery Fleet Intelligence")
    st.caption(DISCLAIMER)

    if snapshot is None:
        st.warning(
            "No fleet snapshot is available. Either run a batch "
            "(`python -m battery_rul.pipelines.run_fleet_batch`) or choose a source "
            "above that scores a fleet now. No sample fleet is invented."
        )
        if page == "Architecture & Limitations":
            _architecture(cfg)
        return

    if snapshot.identity.is_demo_data:
        st.error(f"**DEMO FLEET** — {DEMO_NOTICE}")

    dispatch = {
        "Executive Fleet Overview": lambda: _overview(snapshot),
        "Battery Ranking": lambda: _ranking(snapshot),
        "Critical Batteries": lambda: _critical(snapshot, cfg),
        "Maintenance Planning": lambda: _maintenance(snapshot, cfg),
        "Replacement Planning": lambda: _replacement(snapshot),
        "Fleet Trends": lambda: _trends(adapter, fleet_id),
        "Data Quality": lambda: _quality(snapshot),
        "Feature Drift": lambda: _feature_drift(adapter, fleet_id),
        "Prediction Drift": lambda: _prediction_drift(adapter, fleet_id),
        "Model Performance": lambda: _performance(adapter, fleet_id),
        "Model Registry": lambda: _registry(adapter, cfg),
        "Monitoring Alerts": lambda: _alerts(adapter, fleet_id),
        "Battery Digital Twin": lambda: _battery(snapshot),
        "Architecture & Limitations": lambda: _architecture(cfg),
    }
    dispatch[page]()


def _resolve_snapshot(
    adapter: FleetDashboardAdapter, fleet_id: str, source: str, demo_size: int
) -> FleetSnapshot | None:
    if source == "stored snapshot":
        return adapter.latest_snapshot(fleet_id)
    key = f"{source}:{fleet_id}:{demo_size}"
    stale = st.session_state.get("_fleet_key") != key or "_fleet_snapshot" not in st.session_state
    if stale and st.button(f"Score fleet from {source}", type="primary"):
        with st.spinner("Scoring the fleet…"):
            st.session_state["_fleet_snapshot"] = adapter.run_fleet(
                source="demo" if "demo" in source else "processed",
                fleet_id=fleet_id,
                demo_size=demo_size,
            )
            st.session_state["_fleet_key"] = key
    return st.session_state.get("_fleet_snapshot")


# ---------------------------------------------------------------------------
def _overview(snapshot: FleetSnapshot) -> None:
    summary, statistics = snapshot.summary, snapshot.fleet_statistics
    st.subheader(f"Fleet {snapshot.fleet_id}")
    st.caption(
        f"Snapshot `{snapshot.snapshot_id}` · generated {snapshot.generated_at_utc} · "
        f"model `{summary.active_model_version or 'none'}`"
    )

    columns = st.columns(4)
    columns[0].metric("Batteries submitted", summary.battery_count)
    columns[1].metric("Evaluated", summary.successfully_processed_count)
    columns[2].metric("Failed", summary.failed_count)
    columns[3].metric("Insufficient data", summary.insufficient_data_count)

    columns = st.columns(4)
    columns[0].metric("Healthy (measured)", summary.healthy_count)
    columns[1].metric("Slightly degraded", summary.slightly_degraded_count)
    columns[2].metric("Warning", summary.warning_count)
    columns[3].metric("Critical", summary.critical_count)

    columns = st.columns(4)
    columns[0].metric(
        "Median SOH (measured)",
        "—" if statistics.soh_median is None else f"{100 * statistics.soh_median:.1f} %",
        help=f"Over {statistics.soh_denominator} cell(s) that have a measured SOH.",
    )
    columns[1].metric(
        "Median RUL (predicted)",
        "—" if statistics.rul_median is None else f"{statistics.rul_median:.0f} cycles",
        help=f"Over {statistics.rul_denominator} cell(s) that produced a prediction.",
    )
    columns[2].metric("Inspection recommended", summary.inspection_recommended_count)
    columns[3].metric("Replacement candidates", summary.replacement_planning_count)

    columns = st.columns(3)
    columns[0].markdown(
        f"**Data quality** — {_STATUS_BADGE.get(summary.data_quality_status.value, '?')}"
    )
    columns[1].markdown(f"**Drift** — {_STATUS_BADGE.get(summary.drift_status.value, '?')}")
    columns[2].markdown(f"**Active model** — `{summary.active_model_version or 'none'}`")

    st.info(
        f"Denominators differ by quantity: {summary.battery_count} cells were submitted, "
        f"{summary.successfully_processed_count} produced a prediction, and only those "
        "enter the predicted-quantity statistics. Failed and insufficient-data cells "
        "are listed on the Data Quality page rather than dropped."
    )

    st.markdown("#### Maintenance priority distribution")
    st.bar_chart(
        pd.DataFrame(
            sorted(snapshot.maintenance_summary.priority_counts.items()),
            columns=["priority", "count"],
        ).set_index("priority"),
        height=260,
    )

    if summary.high_priority_battery_ids:
        st.markdown("#### Highest-priority cells")
        st.write(", ".join(f"`{b}`" for b in summary.high_priority_battery_ids))

    if snapshot.warnings:
        with st.expander(f"Fleet warnings ({len(snapshot.warnings)})", expanded=True):
            for warning in snapshot.warnings:
                st.warning(warning)


def _ranking(snapshot: FleetSnapshot) -> None:
    from battery_rul.fleet.ranking import RANKING_KEYS, rank_batteries

    st.subheader("Battery ranking")
    left, right = st.columns([2, 1])
    key = left.selectbox("Rank by", RANKING_KEYS, index=0)
    include = right.checkbox("Include unevaluated cells", value=False)
    ordered = rank_batteries(snapshot.batteries, by=key, include_unevaluated=include)

    st.caption(
        "The composite priority score is a configurable policy, not an optimum. "
        "Open a cell on the Critical Batteries page to see its full score breakdown."
    )
    frame = battery_table(snapshot.model_copy(update={"batteries": ordered}))
    priorities = st.multiselect(
        "Filter by priority", sorted(frame["priority"].unique().tolist()), default=[]
    )
    if priorities:
        frame = frame.loc[frame["priority"].isin(priorities)]
    st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button(
        "Download this ranking (CSV)",
        frame.to_csv(index=False),
        file_name=f"{snapshot.fleet_id}_ranking.csv",
        mime="text/csv",
    )


def _critical(snapshot: FleetSnapshot, cfg: ExperimentConfig) -> None:
    st.subheader("Critical batteries")
    critical = [r for r in snapshot.batteries if r.priority.value in cfg.fleet.critical_priorities]
    critical.sort(key=lambda r: (r.priority.severity, -r.priority_score))
    if not critical:
        st.success("No cell is at a critical priority in this snapshot.")
        return

    st.write(f"{len(critical)} cell(s) at priority {', '.join(cfg.fleet.critical_priorities)}.")
    selected = st.selectbox("Inspect a cell", [r.battery_id for r in critical])
    record = snapshot.battery(selected)
    if record is None or record.priority_record is None:
        return

    columns = st.columns(4)
    columns[0].markdown(f"**Priority**\n\n{_PRIORITY_BADGE.get(record.priority.value, '?')}")
    columns[1].metric("Priority score", f"{record.priority_score:.1f}")
    columns[2].metric(
        "RUL lower bound (predicted)",
        "—" if record.rul_lower_bound is None else f"{record.rul_lower_bound:.0f}",
    )
    columns[3].metric(
        "SOH (measured)",
        "—" if record.measured_soh is None else f"{100 * record.measured_soh:.1f} %",
    )

    st.markdown("#### Score breakdown")
    st.dataframe(score_breakdown_table(snapshot, selected), width="stretch", hide_index=True)

    st.markdown("#### Triggered rules")
    for rule in record.priority_record.triggered_rules:
        st.write(f"- `{rule}`")

    st.markdown("#### Evidence")
    for item in record.priority_record.evidence:
        st.write(f"- {item}")

    inspection = record.priority_record.inspection
    if inspection:
        st.markdown("#### Inspection window")
        st.write(
            f"**{inspection.recommended_label}**"
            + (
                f" — within {inspection.recommended_cycles} cycles"
                if inspection.recommended_cycles is not None
                else ""
            )
            + (
                f" (≈ {inspection.estimated_days:.1f} days at the recent duty rate)"
                if inspection.estimated_days is not None
                else ""
            )
        )
        st.caption(inspection.basis)
        for assumption in inspection.assumptions:
            st.caption(f"· {assumption}")
    st.caption(record.priority_record.disclaimer)


def _maintenance(snapshot: FleetSnapshot, cfg: ExperimentConfig) -> None:
    st.subheader("Maintenance planning")
    summary = snapshot.maintenance_summary
    columns = st.columns(3)
    columns[0].metric("Critical cells", summary.critical_count)
    columns[1].metric("Inspection recommended", summary.inspection_recommended_count)
    columns[2].metric("Insufficient data", summary.insufficient_data_count)

    st.markdown("#### Workload forecast by horizon")
    frame = workload_table(snapshot)
    st.dataframe(frame, width="stretch", hide_index=True)
    st.bar_chart(frame.set_index("horizon")[["batteries"]], height=260)
    st.caption(snapshot.workload_forecast.basis)
    for caveat in snapshot.workload_forecast.caveats:
        st.caption(f"· {caveat}")

    st.markdown("#### Recommended actions")
    st.dataframe(
        pd.DataFrame(sorted(summary.action_counts.items()), columns=["action", "count"]),
        width="stretch",
        hide_index=True,
    )
    st.caption(cfg.fleet.maintenance.disclaimer)


def _replacement(snapshot: FleetSnapshot) -> None:
    st.subheader("Replacement planning (advisory)")
    summary = snapshot.replacement_summary
    frame = pd.DataFrame(
        [
            {
                "horizon": horizon,
                "candidates": summary.counts_by_horizon.get(horizon, 0),
                "lower (optimistic)": summary.lower_counts_by_horizon.get(horizon, 0),
                "upper (conservative)": summary.upper_counts_by_horizon.get(horizon, 0),
            }
            for horizon in ("near_term", "medium_term", "long_term")
        ]
    )
    st.dataframe(frame, width="stretch", hide_index=True)
    st.caption(
        "Lower and upper counts bracket the plan under the prediction intervals. A "
        "single number would assert more than the intervals support."
    )

    candidates = [
        (r.battery_id, r.replacement)
        for r in snapshot.batteries
        if r.replacement is not None and r.replacement.replacement_candidate
    ]
    if candidates:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "battery_id": battery_id,
                        "horizon": candidate.replacement_horizon.value,
                        "confidence": candidate.confidence,
                        "RUL (predicted)": candidate.rul_point,
                        "RUL lower": candidate.rul_lower_bound,
                        "category": candidate.planning_category,
                    }
                    for battery_id, candidate in candidates
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        selected = st.selectbox("Evidence for", [battery_id for battery_id, _ in candidates])
        record = snapshot.battery(selected)
        if record and record.replacement:
            for item in record.replacement.evidence:
                st.write(f"- {item}")
            for caveat in record.replacement.caveats:
                st.caption(f"· {caveat}")
    else:
        st.success("No replacement candidates in this snapshot.")


def _trends(adapter: FleetDashboardAdapter, fleet_id: str) -> None:
    st.subheader("Fleet trends")
    history = adapter.snapshot_history(fleet_id, limit=50)
    if len(history) < 2:
        st.info(
            f"{len(history)} stored snapshot(s) for this fleet. Trends need at least "
            "two: run the batch pipeline repeatedly to build a history."
        )
        return
    metric = st.selectbox("Metric", ["median_soh", "median_rul", "mean_risk", "critical_count"])
    points = fleet_trend_series(history, metric=metric)
    frame = pd.DataFrame(
        [
            {
                "generated_at_utc": p.generated_at_utc,
                metric: p.value,
                "denominator": p.denominator,
            }
            for p in points
        ]
    )
    st.line_chart(frame.set_index("generated_at_utc")[[metric]], height=300)
    st.dataframe(frame, width="stretch", hide_index=True)
    st.caption(
        "The denominator column matters: a median that moves because cells stopped "
        "reporting is a different event from one that moves because the fleet aged."
    )


def _quality(snapshot: FleetSnapshot) -> None:
    st.subheader("Data quality")
    quality = snapshot.data_quality
    columns = st.columns(3)
    columns[0].markdown(f"**Status**\n\n{_STATUS_BADGE.get(quality.status.value, '?')}")
    columns[1].metric(
        "Mean quality score",
        "—" if quality.mean_quality_score is None else f"{quality.mean_quality_score:.2f}",
    )
    columns[2].metric("Cells assessed", quality.denominator)

    st.dataframe(
        pd.DataFrame(sorted(quality.quality_class_counts.items()), columns=["class", "count"]),
        width="stretch",
        hide_index=True,
    )
    for warning in quality.warnings:
        st.warning(warning)

    if quality.per_feature_missing_rate:
        st.markdown("#### Per-feature missing rate")
        st.dataframe(
            pd.DataFrame(
                sorted(quality.per_feature_missing_rate.items(), key=lambda kv: -kv[1]),
                columns=["feature", "missing rate"],
            ),
            width="stretch",
            hide_index=True,
        )
    if quality.check_failure_rates:
        st.markdown("#### Check failure rates")
        st.dataframe(
            pd.DataFrame(
                sorted(quality.check_failure_rates.items()), columns=["check", "failure rate"]
            ),
            width="stretch",
            hide_index=True,
        )
    failed = [r for r in snapshot.batteries if r.errors]
    if failed:
        st.markdown("#### Cells that could not be processed")
        st.dataframe(
            pd.DataFrame(
                [{"battery_id": r.battery_id, "errors": " | ".join(r.errors)} for r in failed]
            ),
            width="stretch",
            hide_index=True,
        )
    st.info(
        "Input data quality is not model drift. A sensor that stopped reporting and a "
        "population that has aged move different numbers and need different remedies."
    )


def _feature_drift(adapter: FleetDashboardAdapter, fleet_id: str) -> None:
    st.subheader("Feature drift")
    monitoring = adapter.latest_monitoring(fleet_id)
    if monitoring is None or not monitoring.feature_drift_summary:
        st.info(
            "No feature-drift report is stored. Run "
            "`python -m battery_rul.pipelines.run_monitoring`."
        )
        return
    payload: dict[str, Any] = monitoring.feature_drift_summary
    columns = st.columns(4)
    columns[0].markdown(
        f"**Status**\n\n{_STATUS_BADGE.get(str(payload.get('status', 'UNKNOWN')), '?')}"
    )
    columns[1].metric("Features tested", payload.get("n_features_tested", 0))
    columns[2].metric("Flagged", payload.get("n_features_drifted", 0))
    columns[3].metric("Skipped", payload.get("n_features_skipped", 0))
    st.caption(f"Reference: `{payload.get('reference_id')}` ({payload.get('reference_window')})")

    results = payload.get("results") or []
    if results:
        st.dataframe(
            pd.DataFrame(results)[
                [
                    c
                    for c in (
                        "feature_name",
                        "drift_metric",
                        "drift_value",
                        "p_value",
                        "adjusted_p_value",
                        "threshold",
                        "severity",
                        "sample_size",
                        "reliable",
                    )
                    if c in pd.DataFrame(results).columns
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    for note in payload.get("method_notes", []):
        st.caption(f"· {note}")
    st.info(
        "Feature drift means the inputs have moved away from what training saw. It is "
        "not evidence that the model has become less accurate — see Model Performance."
    )


def _prediction_drift(adapter: FleetDashboardAdapter, fleet_id: str) -> None:
    st.subheader("Prediction drift")
    monitoring = adapter.latest_monitoring(fleet_id)
    payload = (monitoring.prediction_drift_summary if monitoring else {}) or {}
    if not payload:
        st.info(
            "No prediction-drift report is stored. Seed a prediction reference with "
            "`python -m battery_rul.pipelines.run_monitoring --set-prediction-reference`, "
            "then run monitoring again."
        )
        return
    st.markdown(f"**Status** — {_STATUS_BADGE.get(str(payload.get('status', 'UNKNOWN')), '?')}")
    results = payload.get("results") or []
    if results:
        st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)
    st.warning(
        payload.get("interpretation")
        or "Prediction drift indicates changed model behaviour or a changed population, "
        "not proven model degradation."
    )


def _performance(adapter: FleetDashboardAdapter, fleet_id: str) -> None:
    st.subheader("Model performance (delayed labels)")
    monitoring = adapter.latest_monitoring(fleet_id)
    payload = (monitoring.performance_summary if monitoring else {}) or {}
    if not payload:
        st.info("No performance report is stored yet.")
        return

    status = payload.get("status", "NO_LABELS")
    columns = st.columns(3)
    columns[0].markdown(f"**Status**\n\n{status}")
    columns[1].metric("Labels joined", payload.get("n_labels_joined", 0))
    columns[2].metric("Label coverage", f"{100 * payload.get('label_coverage', 0):.1f} %")

    if status in ("NO_LABELS", "INSUFFICIENT_LABELS"):
        st.info(
            "Prognostic labels arrive only once a cell reaches end of life, so an empty "
            "report early in a deployment is expected. No metric is published until the "
            "configured minimum number of labels has been joined."
        )
        for warning in payload.get("warnings", []):
            st.caption(f"· {warning}")
        return

    for name, title in (
        ("rul_metrics", "RUL"),
        ("soh_metrics", "SOH"),
        ("risk_metrics", "Failure risk"),
        ("interval_coverage", "Prediction-interval coverage"),
    ):
        block = payload.get(name) or {}
        if block:
            st.markdown(f"#### {title}")
            st.json(block, expanded=False)
    if payload.get("rul_error_by_life_stage"):
        st.markdown("#### Error by life stage")
        st.dataframe(
            pd.DataFrame(payload["rul_error_by_life_stage"]), width="stretch", hide_index=True
        )
    for breach in payload.get("breaches", []):
        st.warning(breach)
    st.caption(payload.get("comparison_note", ""))


def _registry(adapter: FleetDashboardAdapter, cfg: ExperimentConfig) -> None:
    st.subheader("Model registry")
    payload = adapter.models()
    if payload.get("error"):
        st.error(payload["error"])

    production = payload.get("production")
    if production:
        st.success(
            f"Production: **{production['model_name']}:{production['model_version']}** "
            f"(promoted {production.get('promoted_at_utc')} by "
            f"{production.get('promoted_by')})"
        )
    else:
        st.warning(
            "No model is at stage PRODUCTION. The serving path falls back to the "
            "configured artifact directory; promote a version to make the live model "
            "explicit."
        )

    models = payload.get("models") or []
    if models:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "model": f"{m['model_name']}:{m['model_version']}",
                        "stage": m["stage"],
                        "task": m.get("task"),
                        "validation": m.get("validation_status"),
                        "features": m.get("n_features"),
                        "created": m.get("created_at_utc"),
                        "promoted": m.get("promoted_at_utc"),
                        "checksum": str(m.get("artifact_checksum", ""))[:12],
                    }
                    for m in models
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    if payload.get("history"):
        with st.expander("Transition history"):
            st.dataframe(pd.DataFrame(payload["history"]), width="stretch", hide_index=True)

    if cfg.deployment.admin_endpoints_enabled:
        st.warning(
            "Administrative endpoints are enabled in this configuration. Promotion is "
            "still performed from the CLI; this page does not offer a promote button."
        )
    else:
        st.caption(
            "Promotion and rollback are CLI operations "
            "(`python -m battery_rul.pipelines.promote_model`). There is no "
            "one-click promotion here: a model going live should require a command and "
            "an author, not a mis-click."
        )


def _alerts(adapter: FleetDashboardAdapter, fleet_id: str) -> None:
    st.subheader("Monitoring alerts")
    alerts = adapter.alerts(fleet_id)
    if not alerts:
        st.success("No stored alerts for this fleet.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "severity": a["severity"],
                    "type": a["type"],
                    "generated": a["generated_at_utc"],
                    "message": a["message"],
                    "acknowledged": a.get("acknowledged", False),
                }
                for a in alerts
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    selected = st.selectbox("Detail for", [a["alert_id"] for a in alerts])
    alert = next(a for a in alerts if a["alert_id"] == selected)
    st.markdown(f"**{alert['type']}** — {alert['message']}")
    st.markdown(f"**Recommended human action:** {alert['recommended_human_action']}")
    for item in alert.get("evidence", []):
        st.write(f"- {item}")
    st.info(
        "Alerts require human review. Nothing in this platform retrains, promotes or "
        "removes an asset from service on an alert."
    )


def _battery(snapshot: FleetSnapshot) -> None:
    st.subheader("Battery digital twin")
    battery_id = st.selectbox("Battery", [r.battery_id for r in snapshot.batteries])
    record = snapshot.battery(battery_id)
    if record is None:
        return

    columns = st.columns(4)
    columns[0].metric("Latest cycle (measured)", record.latest_cycle or 0)
    columns[1].metric(
        "SOH (measured)",
        "—" if record.measured_soh is None else f"{100 * record.measured_soh:.1f} %",
    )
    columns[2].metric(
        "RUL (predicted)",
        "—" if record.predicted_rul is None else f"{record.predicted_rul:.0f} cycles",
    )
    columns[3].metric(
        "Risk (predicted)",
        "—" if record.failure_risk is None else f"{100 * record.failure_risk:.0f} %",
    )
    if record.risk_is_experimental:
        st.warning(
            "The failure-risk model is marked experimental: it did not beat the "
            "cycle-index baseline out of fold, so its probability is reported but was "
            "withheld from the maintenance rules."
        )
    if record.rul_lower_bound is not None and record.rul_upper_bound is not None:
        st.info(
            f"RUL prediction interval **{record.rul_lower_bound:.0f} – "
            f"{record.rul_upper_bound:.0f} cycles** at a "
            f"{100 * (record.interval_coverage_target or 0):.0f} % target coverage. "
            "A prediction interval, not a confidence interval."
        )
    for warning in record.warnings:
        st.caption(f"· {warning}")
    with st.expander("Fleet record JSON"):
        st.code(json.dumps(record.model_dump(mode="json"), indent=2), language="json")
    st.caption(
        "For the full battery-level twin — attributions, trajectory replay, quality "
        "checks — run the Milestone 2 dashboard: "
        "`streamlit run src/battery_rul/dashboard/app.py`."
    )


def _architecture(cfg: ExperimentConfig) -> None:
    st.subheader("Architecture & limitations")
    st.error(DISCLAIMER)
    st.markdown(f"""
**How a number reaches this page**

1. `FleetIngestor` validates each cell's history and reports every rejection.
2. `FleetInferenceService` calls `BatteryDigitalTwinService` once per cell —
   the same inference path as the battery-level API, with the model bundles
   loaded once per process.
3. The maintenance-priority engine applies configurable rules to model outputs.
4. Aggregation computes fleet statistics with explicit denominators.
5. Monitoring compares this batch against a versioned training reference.

**Definitions in force**

- End of life: smoothed capacity at or below {cfg.data.eol_threshold:.0%} of the
  {cfg.data.eol_reference} reference for {cfg.target.eol_persistence} consecutive cycles.
- Failure risk: projected end-of-life crossing within {cfg.risk.horizon_cycles}
  cycles — a derived label, not an observed safety event.
- Prediction intervals: {cfg.uncertainty.method} at {cfg.uncertainty.coverage:.0%}
  target coverage.
- Priority score: weights {cfg.fleet.ranking.weights()}, normalised to
  0–{cfg.fleet.ranking.score_scale:.0f}.

**Known limitations**

- The measured cohort is a handful of laboratory cells of one chemistry on one
  duty cycle. Nothing here is validated for production electric-vehicle use.
- Demo fleets are synthetic and labelled as such on every page.
- The composite priority score has never been validated against real maintenance
  outcomes, because this platform has none. It orders a fleet under a stated
  policy; it does not claim that order is optimal.
- Replacement horizons are advisory planning input, not a schedule, and carry no
  cost model.
- Feature attributions and drift statistics describe the model and its inputs.
  They are not causal claims about the cells.

See `docs/MILESTONE_3_LIMITATIONS.md` for the full statement.
        """)


if __name__ == "__main__":  # pragma: no cover
    main()
