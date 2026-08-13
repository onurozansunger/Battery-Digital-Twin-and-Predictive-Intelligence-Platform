# The maintenance-priority engine

Deterministic rules over model outputs. No model runs inside this layer, and no
threshold in it is learned.

---

## Why the policy is not part of the model

Four reasons, all practical.

**Auditability.** An engineer told to inspect cell `B0042` first is entitled to
know exactly why. A rule set answers that in one line; a learned policy answers
it with an attribution plot.

**Changeability.** Maintenance thresholds are a business decision and they move.
A fleet operator tightening its inspection policy should edit configuration, not
retrain a network.

**Separate failure modes.** When the model degrades, the rules keep behaving
predictably. When the policy changes, the model's evaluation is untouched.

**Stability at fleet scale.** An operations team comparing 128 cells needs the
comparison to be stable. A learned policy that re-ranks the fleet on every
retrain gives them no way to ask "what changed since yesterday".

---

## The ladder

Evaluated top to bottom; the first match wins.

| Priority | Fires when | Action |
| --- | --- | --- |
| `P0_CRITICAL` | measured SOH ≤ `critical_soh` (0.70) | `IMMEDIATE_ENGINEERING_REVIEW` |
| | or risk ≥ `critical_risk` (0.85) **and** planning RUL ≤ `critical_rul_lower_cycles` (5) | |
| | or planning RUL ≤ 5 with no usable risk probability | |
| | or a critical rule-based data warning | |
| `P1_URGENT` | risk ≥ `urgent_risk` (0.60) **and** planning RUL ≤ `urgent_rul_lower_cycles` (15) | `PLAN_REPLACEMENT` |
| | or planning RUL ≤ 15 with no usable risk probability | |
| `P2_HIGH` | health class is `warning` | `SCHEDULE_INSPECTION` |
| | or planning RUL ≤ `high_rul_lower_cycles` (40) | |
| | or fade trend ≥ `high_fade_trend_pct_per_10` (1.0 %/10 cycles) | |
| `P3_MEDIUM` | slightly degraded **and** fade trend ≥ `medium_fade_trend_pct_per_10` (0.5) | `MONITOR_MORE_FREQUENTLY` |
| `P5_MONITOR` | healthy but poorly characterised: interval width ≥ 60 cycles, quality below GOOD, or out-of-distribution features | `MONITOR_UNCERTAIN_ESTIMATE` |
| `P4_LOW` | nothing above fires | `NORMAL_OPERATION` |
| `INSUFFICIENT_DATA` | quality is INSUFFICIENT, or the cell was not evaluated | `INSUFFICIENT_DATA` |

Every threshold lives in `fleet.maintenance` and every one is configurable. The
defaults are demonstration values, not an industry standard.

`P5_MONITOR` sits *below* `P4_LOW` in the ladder order but is a different kind
of statement: P4 says "this cell is fine", P5 says "this cell looks fine and I
am not confident about that". Collapsing them would hide the cells where the
model is least trustworthy behind the label it uses for the cells it is most
sure about.

---

## The conservative planning quantity

Rules fire on the **lower bound** of the RUL prediction interval wherever one
exists, falling back to the point estimate otherwise (and saying so in the
evidence).

A point estimate of 45 cycles with a lower bound of 12 is not a 45-cycle
situation. Planning a fleet against the middle of wide intervals is how
prognostics strands assets. A test asserts that two cells with the same point
estimate and different lower bounds get different priorities.

---

## The experimental-risk rule

Milestone 2 gates the failure-risk model on beating a cycle-index baseline out
of fold. When it does not, the twin marks the probability `is_experimental` and
the fleet engine treats it as **absent** — not as weak evidence, not as
discounted evidence, absent.

The probability is still reported (hiding it would be its own dishonesty), the
withholding is recorded as a `risk_withheld` triggered rule, and the evidence
list says why. In the current repository state the risk model **does** fail this
gate, so every real fleet snapshot here carries that warning.

---

## What every result contains

```json
{
  "battery_id": "B0042",
  "priority": "P1_URGENT",
  "priority_score": 75.9,
  "score_breakdown": [
    {"name": "risk", "raw_value": null, "normalised": 0.0, "weight": 0.30,
     "contribution": 0.0, "available": false,
     "transformation": "…; withheld because the risk model failed its acceptance gate"},
    {"name": "rul_lower_bound", "raw_value": 6.5, "normalised": 0.935, "weight": 0.20,
     "contribution": 0.187, "available": true,
     "transformation": "1 - RUL_lower/100, clipped to [0, 1]; the conservative planning quantity"}
  ],
  "triggered_rules": ["risk_withheld: …", "urgent_rul_only: planning RUL 6.5 <= 15 cycles"],
  "evidence": ["Measured state of health 75.7 % (warning) — derived from measured capacity.", "…"],
  "recommended_action": "PLAN_REPLACEMENT",
  "inspection": {"recommended_cycles": 5, "recommended_label": "within_5_cycles", "…": "…"},
  "disclaimer": "Fleet maintenance priority is deterministic decision support …"
}
```

The breakdown is not decoration. A ranking nobody can interrogate is a ranking
nobody should act on.

---

## The composite priority score

```
score = score_scale x  Σ(wᵢ · nᵢ) / Σ(wᵢ)      over available components only
```

| Component | Raw value | Normalisation | Default weight |
| --- | --- | --- | --- |
| `risk` | calibrated probability | used directly (already in [0,1]) | 0.30 |
| `rul` | RUL point estimate | `1 − RUL/100`, clipped | 0.20 |
| `rul_lower_bound` | RUL interval lower bound | `1 − RUL_low/100`, clipped | 0.20 |
| `soh` | measured SOH | `(1.00 − SOH)/(1.00 − 0.70)`, clipped | 0.15 |
| `trend` | capacity fade, %/10 cycles | `fade/3.0`, clipped | 0.05 |
| `uncertainty` | interval width, cycles | `width/100`, clipped | 0.05 |
| `data_quality` | quality score | `1 − score`, clipped | 0.05 |

**Missing components are excluded from the denominator, not scored as zero.** A
cell with no risk probability is not a cell with zero risk, and treating it as
one would make poorly-instrumented cells look safe. The number of available
components is reported, so a score built from two components is visibly
different from one built from seven.

**A P0 rule raises the score to at least `critical_override_score` (95).** The
rule is the decision; the score only orders cells within it. Without the
override a smoothly-scored cell could out-rank one the rules called critical.

### What this score is not

It is a **configurable decision-support policy**. It has never been validated
against real maintenance outcomes, because this platform has none. Two operators
with different weights will get different orders and both will be legitimate.
Nothing in the codebase claims it is optimal, and the API response says so in
`methodology_note`.

---

## Inspection windows

Cycle-based, because cycles are what the model reasons in.

| Priority | Window | Label |
| --- | --- | --- |
| `P0_CRITICAL` | 0 | `immediate_engineering_review` |
| `P1_URGENT` | 5 | `within_5_cycles` |
| `P2_HIGH` | 10 | `within_10_cycles` |
| `P3_MEDIUM` | 20 | `within_20_cycles` |
| `P4_LOW` / `P5_MONITOR` | — | `next_scheduled_inspection` |
| `INSUFFICIENT_DATA` | — | `insufficient_data` |

A window is shortened when the planning remaining-life estimate is shorter than
the policy window: recommending an inspection after the cell is expected to be
gone is worse than useless.

**A calendar estimate is produced only when a duty rate could be measured** from
at least `min_cycles_for_rate_estimate` timestamped cycles:

```
estimated_days = recommended_cycles / recent_cycles_per_day
```

Without timestamps there is no rate, and the recommendation says so in its
assumptions rather than inventing one. A date an operator plans around must not
come from a guessed duty cycle.
