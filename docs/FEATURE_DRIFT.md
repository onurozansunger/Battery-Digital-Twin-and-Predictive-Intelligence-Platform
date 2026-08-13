# Feature drift

Batch comparison of a fleet's engineered features against a versioned reference
fitted on the training partition.

---

## The features that are tested

The **serving** features — the ones the model actually sees. The monitoring
pipeline calls `BatteryDigitalTwinService.prepare_features`, which runs the same
health derivation, the same causal feature generation and the same warm-up trim
as inference. A drift detector that rebuilt features itself would eventually
watch a different feature space from the one the model reads, and its verdicts
would stop meaning anything about the model.

Feature names come from the deployed bundle's metadata, capped at
`monitoring.drift.max_features` (default 200).

---

## Metrics

### Numerical (`monitoring.drift.numerical_metrics`, default `["psi", "ks"]`)

| Metric | What it measures | p-value | Default thresholds (warn, critical) |
| --- | --- | --- | --- |
| **PSI** | Σ(c−r)·ln(c/r) over the reference's own bins | no | 0.10, 0.25 |
| **KS** | max CDF distance, two-sample | yes | 0.10, 0.20 |
| **Wasserstein** | earth-mover distance ÷ reference σ | no | 0.10, 0.25 |
| **JS** | Jensen–Shannon divergence over binned distributions | no | 0.10, 0.25 |

PSI's conventional bands (< 0.1 stable, 0.1–0.25 moderate, > 0.25 significant)
are a rule of thumb from credit scoring, not a law. They are configurable for
exactly that reason.

Wasserstein is standardised by the reference's spread. An unnormalised distance
is in the feature's own units, so one threshold cannot serve a voltage and a
duration in the same report.

**KS and Wasserstein are approximate here.** The reference artifact stores
quantiles, not raw rows — it is JSON, and storing rows would make it the training
set in a file. The reference sample is reconstructed by inverting the stored
empirical CDF, which is exact at the stored quantiles and linear between them.
Both statistics are reported beside their sample sizes rather than on their own,
and the report's `method_notes` says this.

### Categorical (`monitoring.drift.categorical_metrics`, default `["chi2", "unseen_rate"]`)

| Metric | What it measures | Thresholds |
| --- | --- | --- |
| **chi-square** | two-sample 2×K contingency test | `alpha` (0.05) |
| **unseen rate** | fraction of rows in categories absent from training | 0.01, 0.05 |
| **JS** | divergence between category frequencies | 0.10, 0.25 |

The chi-square is a **two-sample** test, not a goodness-of-fit test against the
stored proportions. Treating the reference proportions as exact ignores the
reference's own sampling error and roughly doubles the statistic: a fair coin
measured twice at 500 rows each then reports "drift" several times more often
than the nominal alpha claims. This was found by a test and fixed.

The unseen rate is reported separately because it is the operationally important
finding: a category the model never trained on is not a shift in proportions, it
is an input the model has no basis for.

---

## Cases that are reported without a verdict

| Case | `reliable` | `severity` | Why |
| --- | --- | --- | --- |
| Constant reference feature | `false` | `UNKNOWN` | no distribution to compare against |
| Fewer than `min_sample_size` (50) current rows | `false` | `UNKNOWN` | the statistic would be sampling noise |
| Feature absent from the batch | `false` | `UNKNOWN` | a data-collection failure, reported as one |
| Feature absent from the reference | not tested | — | listed in `warnings` |

Each of these has a dedicated test. A drift detector that raises on a constant
column is a drift detector that gets wrapped in a bare `except` within a month.

---

## Multiple comparisons

Testing 200 features at α = 0.05 produces about ten "significant" results from
noise alone. The p-value-bearing tests (KS, chi-square) are corrected with
**Benjamini–Hochberg** by default (`monitoring.drift.multiple_comparison`;
`bonferroni` and `none` are also available).

For a corrected test, **both** conditions must hold before drift is flagged:

1. the effect exceeds the configured magnitude threshold, and
2. the adjusted p-value is at or below α.

The magnitude metrics (PSI, Wasserstein, JS) have no p-value and are judged on
thresholds alone. That limitation is stated in every report's `method_notes`
rather than papered over.

---

## Fleet status

```
fraction = distinct drifted features / distinct reliably tested features

CRITICAL  fraction >= fleet_critical_fraction (0.25)  or any CRITICAL feature
WARNING   fraction >= fleet_warning_fraction  (0.10)  or any flagged feature
OK        otherwise
UNKNOWN   nothing could be tested reliably
```

Note the small-fleet artifact: with four tested features, one flagged is 25 %
and lands on CRITICAL. With the 80-feature reference this repository builds, the
fractions behave as intended.

---

## Reading a report

```json
{
  "reference_id": "training_reference",
  "reference_fingerprint": "1a2b3c…",
  "reference_partition": "train",
  "reference_window": "train partition, 267 rows",
  "current_window": "495 rows from batch 20260813T0007Z-…",
  "status": "CRITICAL",
  "n_features_tested": 80,
  "n_features_drifted": 73,
  "results": [
    {"feature_name": "capacity_ah", "drift_metric": "psi", "drift_value": 0.183,
     "threshold": 0.10, "drift_detected": true, "severity": "WARNING",
     "reference_sample_size": 267, "sample_size": 495, "reliable": true}
  ],
  "multiple_comparison": "benjamini_hochberg"
}
```

**A worked example of a misleading-looking result.** Running the monitoring
pipeline against this repository's NASA cohort reports 73 of 80 features
drifted. That is a real distributional difference and not a bug: the reference
covers the three *training* cells, and the batch contains all five, including
the validation and test cells. The comparison is measuring cell-to-cell
variation in a five-cell cohort. In a production fleet the current batch is new
data from the same population; here it is partly a different population. See
`docs/MILESTONE_3_LIMITATIONS.md`.

---

## Commands

```bash
python -m battery_rul.pipelines.build_reference --config configs/default.yaml
python -m battery_rul.pipelines.run_monitoring --config configs/default.yaml --fleet-id DEMO-FLEET-01
```

The report is written to `reports/milestone_3/feature_drift_report.json` and
embedded in the monitoring snapshot.
