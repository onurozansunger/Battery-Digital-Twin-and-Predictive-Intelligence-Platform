"""Streamlit dashboard — layout only, no model logic.

Run it with::

    streamlit run src/battery_rul/dashboard/app.py

Accessibility
-------------
Health and risk categories are never signalled by colour alone: every badge
carries its text label and a shape/symbol prefix, and the trajectory charts
label their series. A reviewer with a monochrome display or a colour-vision
difference gets the same information.

Honesty
-------
Predicted quantities are labelled "predicted" and shown with their interval;
measured quantities are labelled "measured". Demo mode reads the processed
cycles produced by the real pipeline; there is no synthetic fleet, and when no
processed data exists the dashboard says so instead of inventing one.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from battery_rul.config import ExperimentConfig, load_config
from battery_rul.dashboard.data_adapter import (
    DashboardData,
    TwinClient,
    load_demo_cycles,
    trajectory_frame,
)
from battery_rul.digital_twin.domain import BatteryTwinSnapshot

_HEALTH_BADGE: dict[str, str] = {
    "healthy": "● Healthy",
    "slightly_degraded": "◐ Slightly degraded",
    "warning": "◑ Warning",
    "critical": "▲ Critical",
    "unknown": "? Unknown",
}
_RISK_BADGE: dict[str, str] = {
    "low": "● Low",
    "medium": "◐ Medium",
    "high": "◑ High",
    "very_high": "▲ Very high",
    "unknown": "? Unknown",
}
_QUALITY_BADGE: dict[str, str] = {
    "GOOD": "● GOOD",
    "ACCEPTABLE": "◐ ACCEPTABLE",
    "POOR": "◑ POOR",
    "INSUFFICIENT": "▲ INSUFFICIENT",
}
_PRIORITY_BADGE: dict[str, str] = {
    "none": "● none",
    "low": "● low",
    "medium": "◐ medium",
    "high": "◑ high",
    "urgent": "▲ urgent",
}

DISCLAIMER = (
    "Research prototype for engineering decision support. Not validated for "
    "production deployment, not a substitute for battery-management-system "
    "protection, and not suitable for autonomous safety-critical control. "
    "Every predicted value carries model error; review the prediction interval."
)


@st.cache_resource(show_spinner=False)
def _config(path: str) -> ExperimentConfig:
    from pathlib import Path

    candidate = Path(path)
    return load_config(candidate if candidate.is_file() else None)


@st.cache_resource(show_spinner="Loading model artifacts…")
def _client(path: str) -> TwinClient:
    return TwinClient.build(_config(path))


@st.cache_data(show_spinner="Loading processed cycles…")
def _demo(path: str) -> DashboardData | None:
    return load_demo_cycles(_config(path))


def main() -> None:  # pragma: no cover - exercised by the smoke test, not unit tests
    st.set_page_config(page_title="Battery Digital Twin", page_icon="🔋", layout="wide")
    config_path = st.sidebar.text_input("Configuration", value="configs/default.yaml")
    cfg = _config(config_path)
    client = _client(config_path)
    demo = _demo(config_path)

    st.sidebar.markdown("### Service status")
    readiness = client.readiness()
    if readiness.get("ready"):
        st.sidebar.success("● Models loaded and ready")
    else:
        st.sidebar.error("▲ Models unavailable")
        for name, message in (readiness.get("errors") or {}).items():
            st.sidebar.caption(f"{name}: {message}")

    st.title("🔋 Battery Digital Twin & Predictive Intelligence")
    st.caption(DISCLAIMER)

    if demo is None:
        st.warning(
            "No processed cycle table was found. Run "
            "`python scripts/run_pipeline.py --config configs/default.yaml` first. "
            "No sample fleet is generated — a demo built on invented data would "
            "misrepresent what this platform can do."
        )
        return

    battery_id = st.sidebar.selectbox("Battery", demo.batteries)
    full = demo.history(battery_id)
    max_cycle = int(full["cycle_index"].max())
    min_cycle = int(full["cycle_index"].min())
    as_of = st.sidebar.slider(
        "Evaluate as of cycle",
        min_value=min_cycle,
        max_value=max_cycle,
        value=max_cycle,
        help="Only cycles at or before this point are shown to the model — the "
        "twin sees exactly what would have been available at that time.",
    )
    history = demo.history(battery_id, up_to_cycle=as_of)

    tabs = st.tabs(
        [
            "Overview",
            "Digital Twin",
            "RUL",
            "SOH",
            "Failure Risk",
            "Explainability",
            "Model Performance",
            "Data Quality",
            "About & Limitations",
        ]
    )

    snapshot: BatteryTwinSnapshot | None = None
    if readiness.get("ready"):
        try:
            snapshot = client.snapshot(battery_id, history)
        except Exception as exc:  # noqa: BLE001 - surfaced, not raised, in a UI
            st.error(f"Snapshot failed: {type(exc).__name__}: {exc}")

    with tabs[0]:
        _overview(demo, battery_id, history)
    with tabs[1]:
        _twin(snapshot)
    with tabs[2]:
        _rul(client, battery_id, full, as_of, snapshot)
    with tabs[3]:
        _soh(history, snapshot, cfg)
    with tabs[4]:
        _risk(snapshot, cfg)
    with tabs[5]:
        _explain(snapshot)
    with tabs[6]:
        _performance(cfg)
    with tabs[7]:
        _quality(snapshot)
    with tabs[8]:
        _about(cfg, client)


# ---------------------------------------------------------------------------
def _overview(demo: DashboardData, battery_id: str, history: pd.DataFrame) -> None:
    st.subheader("Cohort overview")
    st.write(
        f"{len(demo.batteries)} cell(s) in the processed dataset, loaded from "
        f"`{demo.source}`. All values on this page are **measured** or directly "
        "**derived** — nothing here is a model output."
    )
    summary = (
        demo.cycles.groupby("battery_id")
        .agg(
            cycles=("cycle_index", "max"),
            capacity_start_ah=("capacity_ah", "first"),
            capacity_end_ah=("capacity_ah", "last"),
        )
        .reset_index()
    )
    summary["fade_percent"] = (
        100
        * (summary["capacity_start_ah"] - summary["capacity_end_ah"])
        / summary["capacity_start_ah"]
    ).round(2)
    st.dataframe(summary, width="stretch")

    st.subheader(f"Capacity degradation — {battery_id} (measured)")
    chart = history.set_index("cycle_index")[["capacity_ah"]]
    st.line_chart(chart, height=280)


def _twin(snapshot: BatteryTwinSnapshot | None) -> None:
    st.subheader("Digital twin snapshot")
    if snapshot is None:
        st.info("No snapshot available — see the service status in the sidebar.")
        return

    health, prediction, risk = snapshot.health, snapshot.prediction, snapshot.failure_risk
    columns = st.columns(4)
    columns[0].metric("Latest cycle (measured)", snapshot.measurement_summary.latest_cycle)
    columns[1].metric(
        "State of health (predicted)",
        "—" if health.soh_percent is None else f"{health.soh_percent:.1f} %",
        help="Fraction of the reference capacity. Provenance: " f"{health.provenance.value}.",
    )
    columns[2].metric(
        "Remaining useful life (predicted)",
        "—" if prediction.rul_cycles is None else f"{prediction.rul_cycles:.0f} cycles",
    )
    columns[3].metric(
        f"Risk within {risk.horizon_cycles} cycles",
        "—" if risk.probability is None else f"{100 * risk.probability:.0f} %",
    )

    badges = st.columns(3)
    badges[0].markdown(f"**Health class** — {_HEALTH_BADGE.get(health.health_class, '?')}")
    badges[1].markdown(f"**Risk class** — {_RISK_BADGE.get(risk.risk_class, '?')}")
    badges[2].markdown(
        f"**Data quality** — {_QUALITY_BADGE.get(snapshot.data_quality.quality_class, '?')}"
    )

    interval = prediction.rul_interval
    if interval:
        st.info(
            f"**RUL prediction interval** {interval.lower_bound:.0f} – "
            f"{interval.upper_bound:.0f} cycles at a "
            f"{100 * interval.interval_coverage_target:.0f} % target coverage "
            f"({interval.uncertainty_method}, {interval.calibration_sample_size} "
            "calibration rows). This is a prediction interval, not a confidence "
            "interval."
        )
    else:
        st.warning("No prediction interval is available for this cell.")

    recommendation = snapshot.recommendation
    st.markdown("### Recommendation (rule-based, not a model output)")
    st.markdown(
        f"**{recommendation.title}** &nbsp;·&nbsp; priority "
        f"{_PRIORITY_BADGE.get(recommendation.priority, recommendation.priority)} "
        f"&nbsp;·&nbsp; `{recommendation.action_code}`"
    )
    st.write(recommendation.explanation)
    if recommendation.suggested_window_cycles:
        low, high = recommendation.suggested_window_cycles
        st.write(f"Suggested window: **{low}–{high} cycles**.")
    with st.expander("Evidence the rules used"):
        for item in recommendation.evidence:
            st.write(f"- {item}")
    st.caption(recommendation.disclaimer)

    if snapshot.warnings:
        with st.expander(f"Warnings ({len(snapshot.warnings)})"):
            for warning in snapshot.warnings:
                st.write(f"- {warning}")

    with st.expander("Full snapshot JSON"):
        st.code(json.dumps(snapshot.to_json_dict(), indent=2), language="json")


def _rul(
    client: TwinClient,
    battery_id: str,
    full: pd.DataFrame,
    as_of: int,
    snapshot: BatteryTwinSnapshot | None,
) -> None:
    st.subheader("Remaining useful life")
    if snapshot is None:
        st.info("No snapshot available.")
        return
    st.write(
        "The trajectory below replays the cell: each point is what the twin would "
        "have predicted using only the cycles available at that time."
    )
    if st.button("Compute RUL trajectory", key="rul_traj"):
        with st.spinner("Replaying the cell…"):
            frame = trajectory_frame(client, battery_id, full.loc[full["cycle_index"] <= as_of])
        if frame.empty:
            st.warning("No scoreable points in this range.")
        else:
            st.session_state["trajectory"] = frame

    frame = st.session_state.get("trajectory")
    if frame is not None and not frame.empty:
        st.line_chart(frame.set_index("cycle_index")[["rul", "rul_lower", "rul_upper"]], height=320)
        st.caption(
            "Solid centre: point estimate. The other two series are the lower and "
            "upper bounds of the conformal prediction interval."
        )
        st.dataframe(frame, width="stretch")


def _soh(
    history: pd.DataFrame, snapshot: BatteryTwinSnapshot | None, cfg: ExperimentConfig
) -> None:
    st.subheader("State of health")
    st.write(
        f"SOH is reported as a fraction of the reference capacity "
        f"(strategy `{cfg.soh.reference_strategy}`, "
        f"{cfg.soh.reference_cycles} cycle(s)). Bands: healthy ≥ "
        f"{cfg.soh.healthy_min:.0%}, slightly degraded ≥ "
        f"{cfg.soh.slightly_degraded_min:.0%}, warning ≥ {cfg.soh.warning_min:.0%}, "
        "critical below that. These are configurable demonstration thresholds, not "
        "a standard."
    )
    if snapshot is not None:
        left, right = st.columns(2)
        left.metric(
            "SOH — measured (derived)",
            (
                "—"
                if snapshot.health.soh_measured is None
                else f"{100 * snapshot.health.soh_measured:.2f} %"
            ),
        )
        right.metric(
            "SOH — model estimate (predicted)",
            "—" if snapshot.health.soh_percent is None else f"{snapshot.health.soh_percent:.2f} %",
        )
    if "capacity_ah" in history.columns:
        st.line_chart(history.set_index("cycle_index")[["capacity_ah"]], height=280)
        st.caption("Measured discharge capacity over cycles.")
    for column, label in (
        ("temperature_max_c", "Peak cell temperature (measured)"),
        ("internal_resistance_ohm", "Internal resistance (measured)"),
    ):
        if column in history.columns and history[column].notna().any():
            st.line_chart(history.set_index("cycle_index")[[column]], height=220)
            st.caption(label)


def _risk(snapshot: BatteryTwinSnapshot | None, cfg: ExperimentConfig) -> None:
    st.subheader("Failure risk")
    st.warning(
        f"The risk label is **derived**: it means the cell's capacity is projected "
        f"to cross the {cfg.data.eol_threshold:.0%} end-of-life threshold within "
        f"{cfg.risk.horizon_cycles} cycles. The source dataset contains no observed "
        "safety failures, so the model has never seen one and cannot predict one."
    )
    if snapshot is None:
        return
    risk = snapshot.failure_risk
    columns = st.columns(4)
    columns[0].metric(
        "Calibrated probability",
        "—" if risk.probability is None else f"{100 * risk.probability:.1f} %",
    )
    columns[1].metric(
        "Raw score",
        "—" if risk.probability_raw is None else f"{100 * risk.probability_raw:.1f} %",
    )
    columns[2].metric(
        "Decision threshold",
        "—" if risk.decision_threshold is None else f"{risk.decision_threshold:.3f}",
    )
    columns[3].markdown(f"**Risk class**\n\n{_RISK_BADGE.get(risk.risk_class, '?')}")
    st.caption(
        f"Calibration: {risk.calibration_method or 'none'} "
        f"({'calibrated' if risk.is_calibrated else 'uncalibrated'}). "
        f"{risk.label_definition}"
    )

    frame = st.session_state.get("trajectory")
    if frame is not None and not frame.empty and frame["risk"].notna().any():
        st.line_chart(frame.set_index("cycle_index")[["risk"]], height=280)
        st.caption("Calibrated risk probability over the replayed trajectory.")


def _explain(snapshot: BatteryTwinSnapshot | None) -> None:
    st.subheader("Explainability")
    if snapshot is None or snapshot.explanation is None or not snapshot.explanation.drivers:
        st.info("No attributions are available for this snapshot.")
        return
    explanation = snapshot.explanation
    st.caption(f"Method: `{explanation.method}`. {explanation.caveat}")
    frame = pd.DataFrame(
        [
            {
                "feature": driver.display_name,
                "direction": driver.contribution_direction.replace("_", " "),
                "magnitude": driver.contribution_magnitude,
                "current_value": driver.current_value,
                "reference_value": driver.reference_value,
            }
            for driver in explanation.drivers
        ]
    )
    st.bar_chart(frame.set_index("feature")[["magnitude"]], height=300)
    st.dataframe(frame, width="stretch")
    for driver in explanation.drivers:
        st.write(f"- **{driver.display_name}** — {driver.explanation_text}")


def _performance(cfg: ExperimentConfig) -> None:
    st.subheader("Model performance")
    st.write(
        "These tables are read directly from the report artifacts the pipelines "
        "wrote. Nothing on this page is recomputed in the dashboard."
    )
    for label, path in (
        ("Milestone 2 metrics", cfg.paths.reports_dir / "milestone_2" / "metrics.json"),
        ("Nested model comparison", cfg.paths.reports_dir / "nested_model_comparison.csv"),
        (
            "Nested comparison, common rows",
            cfg.paths.reports_dir / "nested_model_comparison_common_rows.csv",
        ),
        ("Per-fold results", cfg.paths.reports_dir / "nested_per_fold.csv"),
    ):
        st.markdown(f"**{label}** — `{path.name}`")
        if not path.is_file():
            st.caption("Not generated yet.")
            continue
        if path.suffix == ".csv":
            st.dataframe(pd.read_csv(path), width="stretch")
        else:
            payload: dict[str, Any] = json.loads(path.read_text())
            st.json(
                {k: v for k, v in payload.items() if k not in {"config", "environment"}},
                expanded=False,
            )


def _quality(snapshot: BatteryTwinSnapshot | None) -> None:
    st.subheader("Data quality")
    if snapshot is None:
        st.info("No snapshot available.")
        return
    quality = snapshot.data_quality
    left, right = st.columns(2)
    left.metric("Quality score", f"{quality.quality_score:.2f}")
    right.markdown(f"**Class**\n\n{_QUALITY_BADGE.get(quality.quality_class, '?')}")
    if quality.warnings:
        for warning in quality.warnings:
            st.warning(warning)
    st.dataframe(
        pd.DataFrame([{"check": name, **values} for name, values in quality.checks.items()]),
        width="stretch",
    )
    if quality.missing_features:
        with st.expander(f"Missing features ({len(quality.missing_features)})"):
            st.write(", ".join(quality.missing_features))
    if quality.out_of_distribution_flags:
        with st.expander(f"Outside the training range ({len(quality.out_of_distribution_flags)})"):
            st.write(", ".join(quality.out_of_distribution_flags))


def _about(cfg: ExperimentConfig, client: TwinClient) -> None:
    st.subheader("About & limitations")
    st.error(DISCLAIMER)
    st.markdown(f"""
**Definitions in force**

- End of life: smoothed capacity at or below {cfg.data.eol_threshold:.0%} of the
  {cfg.data.eol_reference} reference for {cfg.target.eol_persistence} complete
  consecutive cycles.
- State of health: fraction in [0, 1]; reference `{cfg.soh.reference_strategy}`.
- Failure risk: projected end-of-life crossing within
  {cfg.risk.horizon_cycles} cycles — a derived label, not an observed safety event.
- Uncertainty: {cfg.uncertainty.method} at {cfg.uncertainty.coverage:.0%} target
  coverage. Prediction intervals, not confidence intervals.

**Known limitations**

- The cohort is a handful of laboratory cells of one chemistry on one duty cycle.
  Nothing here has been validated for production electric-vehicle deployment.
- Conformal coverage assumes exchangeability between calibration and served
  cells; distinct physical cells satisfy that only approximately.
- Feature attributions describe the model's response to its inputs. They are not
  causal claims about the cell, and attention weights are a diagnostic only.
- Recommendations are deterministic rules over model outputs and require review
  by a qualified engineer before any action.

See `docs/MILESTONE_2_LIMITATIONS.md` for the full statement.
        """)
    with st.expander("Loaded model metadata"):
        st.json(client.model_metadata(), expanded=False)


if __name__ == "__main__":  # pragma: no cover
    main()
