# Recommendation engine

## What it is

A deterministic, side-effect-free rule set that reads model *outputs* plus
deterministic trend statistics and emits exactly one primary action, with
evidence and a disclaimer.

## Why it is not part of the model

Three practical reasons:

**Auditability.** An engineer asked to act on "schedule inspection within 10–20
cycles" is entitled to know exactly why. A rule set answers that in one line; a
learned policy answers it with an attribution plot.

**Changeability.** Maintenance thresholds are a business decision and they move.
A fleet operator tightening its inspection policy should edit configuration, not
retrain a network.

**Separation of failure modes.** When the model degrades, the rules keep behaving
predictably. When the policy changes, the model is untouched and its evaluation
stays valid. Fusing them means every policy change invalidates the model's
measured performance.

## Inputs

| Input | Source |
|---|---|
| `rul_point`, `rul_lower_bound` | RUL model + conformal interval |
| `soh`, `health_class` | SOH model / measured derivation |
| `risk_probability`, `risk_class` | calibrated risk classifier |
| `quality_class` | data-quality assessment |
| `temperature_trend_c_per_10` | trailing OLS over the last 20 cycles — no model |
| `resistance_trend_pct_per_10` | trailing OLS, relative — no model |
| `fade_trend_pct_per_10` | trailing OLS on smoothed capacity — no model |
| `is_scoreable` | warm-up policy |

## The conservative bound

**Rules fire on the lower bound of the RUL interval, not the point estimate.**
A point estimate of 45 cycles with a lower bound of 12 is not a 45-cycle
situation, and planning against the middle of a wide interval is how prognostics
gets someone stranded. `tests/test_digital_twin.py::test_rules_use_the_lower_bound_not_the_point_estimate`
pins this behaviour.

## Actions

| Code | Priority | Fires when |
|---|---|---|
| `INSUFFICIENT_DATA` | low | quality is `INSUFFICIENT`, or the cell is before its first scoreable cycle |
| `IMMEDIATE_ENGINEERING_REVIEW` | urgent | effective RUL ≤ `urgent_rul_cycles` (5) or risk ≥ `urgent_risk` (0.85) |
| `PLAN_REPLACEMENT` | high | effective RUL ≤ `replacement_rul_cycles` (15), risk ≥ `replacement_risk` (0.60), or health class `critical` |
| `SCHEDULE_INSPECTION` | medium | effective RUL ≤ `inspection_rul_cycles` (40), risk ≥ `inspection_risk` (0.30), or health class `warning` |
| `REDUCE_HIGH_TEMPERATURE_OPERATION` | medium | temperature trend ≥ `temperature_trend_c_per_10_cycles` |
| `REDUCE_AGGRESSIVE_CHARGING` | medium | resistance or fade trend above its configured limit |
| `MONITOR_MORE_FREQUENTLY` | low | health class `slightly_degraded` |
| `NORMAL_OPERATION` | none | no threshold met |

The ladder is evaluated most-severe-first. Trend advisories that do not win the
primary slot are attached to the evidence list rather than discarded.

Every threshold is in `recommendations` configuration.

## The hard stop

`INSUFFICIENT_DATA` is not a fallback, it is the point. A confident maintenance
action derived from four cycles of a cell is worse than no answer, because it
looks like an answer. The engine returns it — with no RUL, no risk and no
window — whenever data quality is `INSUFFICIENT` or the cell has not reached the
first scoreable cycle under the training warm-up policy.

## What this layer cannot do

- It does not schedule work, dispatch technicians or create tickets.
- It does not command or influence a battery management system.
- It does not replace a qualified engineer's judgement.
- It has no authority over safety: the BMS protections are the safety layer, and
  this is decision support sitting well above them.

Every recommendation payload carries `recommendations.disclaimer` stating this —
in the response body, not only in the documentation.

## Where it is implemented

- `src/battery_rul/recommendations/engine.py`
- `src/battery_rul/config.py::RecommendationConfig`
- `tests/test_digital_twin.py` — one test per rule, plus determinism and the
  disclaimer invariant
