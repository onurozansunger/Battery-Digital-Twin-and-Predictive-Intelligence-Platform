# Fleet dashboard guide

```bash
streamlit run src/battery_rul/dashboard/fleet_app.py
```

The Milestone 2 battery-level dashboard (`dashboard/app.py`) is unchanged and
still the place to drill into one cell's trajectory, attributions and quality
checks.

---

## How it gets its data

Every number comes from a `FleetSnapshot` produced by the fleet service or from
a stored monitoring snapshot. **Nothing is recomputed in the dashboard.** A
dashboard that recalculates a priority is a second implementation of the policy,
and the two will disagree.

All data access lives in `dashboard/fleet_adapter.py` as plain functions over
plain objects — which is what makes it testable — and the Streamlit script does
layout only.

Two back-ends, chosen by `service.dashboard_mode`:

* `service` — calls the fleet service in-process (default)
* `api` — calls the running FastAPI service over HTTP

Both go through the same service layer; one just goes through a socket first.

---

## Sidebar

| Control | Effect |
| --- | --- |
| Configuration | path to the YAML config |
| Service status | ready / not ready, with the per-bundle errors |
| Fleet id | which fleet to read or score |
| Fleet source | `stored snapshot` (reads the last batch), `processed cycles (measured)`, or `demo fleet (synthetic)` |
| Demo fleet size | cells to generate when using the demo source |
| Page | the 14 pages below |

When no snapshot is available the dashboard says so and names the command to
run. It does not invent a fleet.

---

## The pages

**1. Executive Fleet Overview** — submitted / evaluated / failed / insufficient
counts, the health bands, median SOH and RUL **with their denominators in the
tooltip**, inspection and replacement counts, data-quality and drift status, the
active model version, the priority histogram, and the fleet warnings expanded by
default.

**2. Battery Ranking** — sortable table over any of the twelve ranking keys, with
a priority filter and a CSV download. Columns are labelled `(measured)` or
`(predicted)`.

**3. Critical Batteries** — the cells at a critical priority, and for a selected
one: the **full score breakdown** (component, raw value, normalisation, weight,
contribution), the triggered rules, the evidence list, and the inspection window
with its basis and assumptions.

**4. Maintenance Planning** — workload by horizon with the uncertainty bracket,
the action counts, and the policy disclaimer.

**5. Replacement Planning** — candidates by horizon with lower/upper counts, per
candidate evidence and caveats.

**6. Fleet Trends** — median SOH, median RUL, mean risk or critical count across
stored snapshots. Needs at least two; says so when there is one. Every point
carries its denominator, because a median that moves because cells stopped
reporting is a different event from one that moves because the fleet aged.

**7. Data Quality** — class counts, mean score, per-feature missing rate, check
failure rates, and the cells that could not be processed with their errors. Ends
with the reminder that input quality is not model drift.

**8. Feature Drift** — status, features tested / flagged / skipped, the reference
id and window, the per-feature results with p-values and adjusted p-values, and
the method notes. Ends with "this is not evidence the model became less
accurate".

**9. Prediction Drift** — status and per-quantity results, with the
interpretation warning displayed prominently rather than in a footnote.

**10. Model Performance** — the delayed-label report. When the status is
`NO_LABELS` or `INSUFFICIENT_LABELS` it explains why that is expected rather
than showing an empty chart.

**11. Model Registry** — the production model (or an explicit warning that none
is promoted), every registered version with stage, validation status and
checksum prefix, and the transition history. **There is no promote button**: a
model going live should require a command and an author, not a mis-click.

**12. Monitoring Alerts** — stored alerts with severity, type, message and
acknowledgement state, and for a selected alert its evidence and its
**recommended human action**.

**13. Battery Digital Twin** — one cell's fleet record, its interval, the
experimental-risk warning where it applies, and the raw JSON. Points to the
Milestone 2 dashboard for the full twin view.

**14. Architecture & Limitations** — how a number reaches the page, the
definitions in force, and the known limitations.

---

## Honesty rules the layout follows

* measured and predicted quantities are labelled as such, on every page;
* denominators appear beside aggregates;
* demo fleets carry a banner on **every** page, not a footnote on one;
* an experimental risk model is marked wherever its probability appears, with
  the note that it was excluded from the decision rules;
* health, risk and status are never signalled by colour alone — every badge
  carries its text label and a shape prefix (carried over from Milestone 2).

---

## Demo mode

Selecting `demo fleet (synthetic)` scores a clearly-labelled demonstration
fleet. Where a processed cycle table exists, the demo is **derived** from the
measured cells (each truncated at a different point in its life); otherwise it
is generated by the physics-informed simulator. Both are marked
`is_demo_data=True` everywhere. See `docs/DEMO_GUIDE.md`.
