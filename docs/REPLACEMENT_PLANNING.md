# Replacement planning

Advisory. The word carries weight: this module answers "which cells might need
replacing within 20 / 50 / 100 cycles, and how confident is that?" It does not
schedule anything, raise a purchase order, or convert cells into money.

---

## Horizons

| Horizon | Default | Meaning |
| --- | --- | --- |
| `near_term` | ≤ 20 cycles | plan now |
| `medium_term` | ≤ 50 cycles | plan this period |
| `long_term` | ≤ 100 cycles | keep in view |
| `not_flagged` | beyond | no action |
| `unknown` | not evaluated | no assessment |

All in `fleet.replacement`.

---

## What makes a candidate

The planning quantity is the **RUL interval lower bound** when one exists
(`use_lower_bound`, default true), falling back to the point estimate with a
caveat recorded on the result.

A cell also becomes a candidate on:

* calibrated risk ≥ `risk_candidate_threshold` (0.60) — medium term,
* measured SOH ≤ `soh_candidate_threshold` (0.72) — near term,
* maintenance priority P0 or P1 — near term.

An **experimental** risk model contributes nothing, exactly as in the priority
engine, and the exclusion is recorded in the caveats.

---

## Confidence

Not the model's confidence in itself, which it does not report — a statement
about how much weight the horizon can bear.

| Confidence | When |
| --- | --- |
| `high` | interval width < half `wide_interval_ratio` × point estimate, and quality GOOD |
| `medium` | moderate width, or quality ACCEPTABLE |
| `low` | width ≥ `wide_interval_ratio` × point estimate, no interval at all, or quality POOR/INSUFFICIENT |
| `unknown` | no remaining-life estimate |

---

## Uncertainty-aware counts

Every horizon reports three numbers:

| Count | Computed from | Reading |
| --- | --- | --- |
| `count` | the planning quantity | the plan under the configured policy |
| `lower_count` | each cell's **upper** RUL bound | optimistic — the smallest defensible number |
| `upper_count` | each cell's **lower** RUL bound | conservative — the largest defensible number |

A single number would assert more than the prediction intervals support. Three
numbers make the width of those intervals visible in the plan itself, and a
spread of 4–17 replacements in the next 20 cycles is a materially different
planning conversation from a flat "9".

---

## Workload forecast

A separate view, in `fleet/replacement.py::workload_forecast`:

| Bucket | Contains |
| --- | --- |
| `immediate` | every P0 and P1 cell, whatever their RUL says |
| `next_10_cycles`, `next_30_cycles`, `next_50_cycles` | by planning RUL (configurable in `fleet.workload.horizons_cycles`) |
| `beyond_50_cycles` | the rest |
| `monitor_only` | evaluated, but no remaining-life estimate |
| `insufficient_data` | not evaluated |

Every submitted cell lands in exactly one bucket — a test asserts the counts sum
to the fleet size. Percentages are **of the evaluated fleet**, and the excluded
count is reported beside them.

---

## No cost model

Cost per replacement, downtime, spare availability and fleet utilisation are
inputs this platform does not have. A savings figure computed without them would
be fiction with a currency symbol on it, so nothing here produces one, and the
caveats on every candidate say so:

> No cost, downtime or spares-availability assumption is applied, so these counts
> must not be converted into financial figures without them.

If an operator supplies those assumptions, the counts and their uncertainty
brackets are the right input to that calculation — done explicitly, outside this
module.

---

## API

`POST /v1/fleet/replacement-plan` returns the summary, the paged candidates and
the caveats. The dashboard's Replacement Planning page renders the same objects.
