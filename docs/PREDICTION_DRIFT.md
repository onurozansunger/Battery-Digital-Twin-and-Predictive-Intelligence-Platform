# Prediction drift

Has the model's **output** distribution moved?

> **Prediction drift does not prove model degradation.** It says the model is
> saying different things. That happens when the population changes (a fleet
> that has simply aged), when the input pipeline changes, or when the model
> changes. Distinguishing those needs labels — see
> `docs/PERFORMANCE_MONITORING.md`.

That sentence is in the code (`PredictionDriftReport.interpretation`), in every
report, on the dashboard page and here, because it is the one thing about this
signal that is most often got wrong. A fleet whose cells are a hundred cycles
older than last month *should* show prediction drift; treating that as an
incident trains an operations team to ignore the alert that matters.

---

## What is monitored

| Quantity | Metrics |
| --- | --- |
| Predicted RUL | standardised mean shift, standardised quantile shift, PSI |
| Predicted SOH forecast | same |
| Calibrated failure risk | same |
| Prediction-interval width | same, plus a median width **ratio** |
| Health-class frequencies | total-variation distance |
| Risk-class frequencies | total-variation distance |
| Maintenance-priority frequencies | total-variation distance |

---

## Metrics and thresholds

All in `monitoring.prediction_drift`.

| Metric | Definition | Default (warn, critical) |
| --- | --- | --- |
| `standardised_mean_shift` | \|mean_now − mean_ref\| ÷ σ_ref | 0.25, 0.50 |
| `standardised_quantile_shift` | same, per configured quantile | 0.25, 0.50 |
| `psi` | over the reference's own histogram bins | 0.10, 0.25 |
| `width_ratio` | median interval width now ÷ reference | 1.25, 1.75 |
| `total_variation_distance` | ½·Σ\|freq_now − freq_ref\| over classes | 0.15, 0.30 |

Total-variation distance was chosen over a chi-square test for the class
frequencies because the reading is direct: 0.30 means 30 % of the fleet's mass
moved between classes, which is a sentence an operations team can act on.

Standardisation by the reference σ (falling back to its range when σ is zero)
makes one threshold usable across quantities measured in cycles, fractions and
probabilities.

**Uncertainty inflation is separated from location shift.** A widening interval
means the model is less certain about this population, not that it is wrong
about it, and `width_ratio` carries that note in its payload.

---

## The reference

Prediction drift needs a reference *batch*, not a training distribution: the
question is "compared with how this model normally behaves on this fleet".

```bash
# 1. Produce a batch you are willing to call normal
python -m battery_rul.pipelines.run_fleet_batch --fleet-id DEMO-FLEET-01 --source processed

# 2. Adopt it as the prediction reference (explicit, manual)
python -m battery_rul.pipelines.run_monitoring --fleet-id DEMO-FLEET-01 --set-prediction-reference

# 3. From now on, monitoring runs compare against it
python -m battery_rul.pipelines.run_monitoring --fleet-id DEMO-FLEET-01 --source processed
```

Step 2 is manual on purpose. A reference that updated itself on every run would
compare each batch with the previous one, detect a step change once, and then
treat the new behaviour as normal for ever — the failure mode where a model
degrades slowly and monitoring never notices.

The prediction reference is stored inside the same versioned reference artifact
as the feature statistics, so a report cites one `reference_id` for both.

---

## Small samples

Below `monitoring.prediction_drift.min_sample_size` (default 20 scored cells)
the status is `UNKNOWN` and no metric is published, with the reason in
`warnings`. The five-cell NASA cohort in this repository is below that threshold,
so real prediction-drift runs here report `UNKNOWN` rather than a number
computed from five points. The 24-cell demo fleet exercises the populated path.

---

## Reading a report

```json
{
  "status": "WARNING",
  "n_drifted": 2,
  "sample_size": 24,
  "results": [
    {"output_name": "predicted_rul", "metric": "standardised_mean_shift",
     "reference_value": 84.2, "current_value": 61.7, "drift_value": 0.41,
     "threshold": 0.25, "drift_detected": true, "severity": "WARNING"},
    {"output_name": "priority_frequencies", "metric": "total_variation_distance",
     "drift_value": 0.19, "threshold": 0.15, "drift_detected": true,
     "detail": {"reference": {"P4_LOW": 0.6, "…": "…"}, "current": {"…": "…"}}}
  ],
  "interpretation": "Prediction drift does not prove model degradation. …"
}
```

Read it beside the feature-drift report and the performance report. On its own
it establishes that something changed, not what.
