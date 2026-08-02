# Milestone 2 — limitations

Read this before quoting any number from this repository.

## The data

**Five cells after quality gating.** The NASA randomised-usage set contains more
cells, but the cohort gates (beginning-of-life health, minimum fade, confirmed
end-of-life crossing, minimum labelled cycles) leave a handful. Every metric in
this repository is a five-cell result. Cross-validation makes the estimate
honest; it does not make the cohort large.

**One chemistry, one format, one rig.** LCO 18650 cells, cycled in a laboratory
at a small number of ambient set points. Nothing here has been evaluated on NMC,
LFP, NCA, pouch or prismatic cells, on a different form factor, or outside a
temperature-controlled chamber.

**Laboratory duty cycles, not automotive ones.** Constant-current discharge to a
fixed cut-off is not a drive cycle. Real automotive loads are transient, partial,
and interleaved with rest, fast charge and regenerative braking. The features
this platform relies on — discharge duration, voltage-curve shape, temperature
rise — behave differently under those conditions.

**No pack-level effects.** Cell-to-cell imbalance, interconnect resistance,
thermal gradients across a module and BMS balancing strategy are all absent, and
all of them dominate real fleet degradation.

**Therefore: this platform is not validated for production electric-vehicle
deployment, and its numbers do not transfer to one.**

## The failure label

**There are no observed failures in this dataset.** The "failure risk" target is
derived arithmetically: capacity is projected to cross a threshold this project
chose (70 % of reference, persisting 3 cycles) within a horizon this project
chose (30 cycles). No thermal event, venting incident, internal short or
pack-level fault appears anywhere in the source data.

A model cannot predict a category of event it has never seen. Treating this
output as a safety-hazard probability would be a serious misreading. See
`docs/FAILURE_RISK_DEFINITION.md`.

## An external review found seven defects

They are listed with their fixes in `docs/MILESTONE_2_1_REVIEW_FIXES.md`. Three
changed what the published numbers mean, and the earlier ones should not be
quoted:

* the deployed RUL family was chosen using a comparison that included the test
  cell, so the Milestone 2 "held-out test" figure was not untouched;
* the SOH model predicted current SOH from an input containing current
  capacity, so its 1.34 % MAE measured a rescaling, not health inference;
* conformal coverage was measured on the residuals the quantile was fitted
  from, so 91.7 % was calibration-set coverage rather than evidence.

## Risk metrics are easy to over-read

The label is "RUL ≤ H", so a cell's positives are exactly its last H cycles and
**cycle index alone scores a perfect AUC within any single cell**. The
battery-holdout puts one cell in test, so the headline test AUC is degenerate: a
model at 0.93 is doing worse than counting cycles. Every AUC in this repository is
reported next to a `*_cycle_index_baseline` computed on the same rows, with a
`beats_cycle_index_baseline` flag. Read the comparison, not the absolute number.

The same applies to the multi-task model's near-perfect risk PR-AUC on
single-cell partitions. It is not evidence of a good classifier.

**The risk model loses to that baseline on every partition**, so it now fails an
acceptance gate: the twin reports its probability as experimental and withholds
it from the recommendation rules entirely.

Post-calibration Brier and ECE on out-of-fold rows are **in-sample** — the
calibrator was fitted on them. Only the test-partition calibration figures are
out-of-sample.

## The uncertainty

Conformal coverage holds under **exchangeability between calibration and served
rows**, which is only approximately satisfied here:

- rows within a cell are strongly autocorrelated, so the effective sample size is
  well below the nominal one and realised coverage is noisier than 90 % suggests;
- calibration cells and the served cell are different physical cells — a
  cross-cell assumption, not an i.i.d.-rows one.

Coverage is therefore reported per battery and per life stage, not only
marginally. Check those tables before relying on an interval near end of life.

## The explanations

Feature attributions describe **how the model's output responds to its inputs**.
They are not causal claims about the cell. "Recent operating temperature is above
the training reference range and contributed to the model's elevated risk
estimate" is supportable; "high temperature caused this battery to fail" is not,
and nothing in this codebase produces the second sentence.

**Attention weights are not explanations.** The multi-task encoder exposes its
pooling weights as a diagnostic. "Looked at" is not "was influenced by" — a
high-attention timestep can have no effect on the output at all. They are never
presented as attributions.

## The recommendations

Deterministic rules over model outputs, with configurable thresholds chosen for
demonstration. They are decision *support*: they do not schedule work, do not
command a battery management system, and do not replace a qualified engineer's
judgement. The BMS protections are the safety layer; this sits well above them.

## Health and risk bands

`healthy` / `slightly_degraded` / `warning` / `critical` and `low` / `medium` /
`high` / `very_high` are project-level engineering categories with configurable
demonstration thresholds. They are not a standard and not a regulatory
classification.

## Modelling

- **Selection bias is reduced, not eliminated.** The nested leave-one-battery-out
  design re-runs family selection inside every outer fold, so its pooled metric
  estimates the whole procedure. Hyperparameters within a family are *not*
  searched inside the inner loop at this cohort size; that residual bias is
  stated rather than hidden.
- **Sequence models score fewer rows.** They cannot score a cell's first
  *window − 1* scoreable cycles, and those are the hardest early-life rows.
  Metrics are reported both on each model's own rows (with the unscored count)
  and on the intersection every candidate can score; the ranking uses the
  intersection.
- **The multi-task model and the independent models are comparable only on
  commonly scoreable rows.** Coverage is reported alongside every multi-task
  metric.
- **Right-censored cells are excluded, not modelled.** Proper handling needs
  survival methods. Excluding them biases the cohort toward cells that failed
  within the experiment's duration.

## Engineering

- Interim caches are keyed by the data-affecting configuration and a source-file
  fingerprint (size and mtime, not content hashes — a 209 MB re-hash per run
  would not be worth it). A file edited in place with a preserved mtime would not
  invalidate the cache.
- The environment is pinned by `requirements-lock.txt`; the lock is generated on
  macOS/Python 3.13 and is not a multi-platform resolution.
- There is no drift monitoring, no model registry, no experiment tracking and no
  containerisation. Those are Milestone 3.

## Milestone 1 metrics

Milestone 1's published figures were regenerated after the Milestone 1.1
hardening. Pre-hardening and post-hardening numbers are reported separately and
are **not comparable**: the end-of-life persistence fix changes labels, and the
preprocessing-boundary fix changes which features survive. See
`docs/MILESTONE_1_1_HARDENING.md` for the before/after and the explanation of
each change.
