# Digital twin — architecture

## Layering

```
                       ┌──────────────────────────────────────┐
   FastAPI  ───────►   │                                      │
   Streamlit ──────►   │   BatteryDigitalTwinService          │
   scripts/  ──────►   │   (the ONLY place inference happens) │
                       └──────────────┬───────────────────────┘
                                      │
        ┌───────────────┬─────────────┼─────────────┬──────────────────┐
        ▼               ▼             ▼             ▼                  ▼
   data.loader     features.     models.bundle   uncertainty.    recommendations.
   derive_health   engineering   load_bundle     conformal       engine
   (TRAINING       build_features (model +        (interval)      (rules, no model)
    derivation)    (TRAINING      preprocessing
                    generator)    + metadata)
                                      │
                                      ▼
                              calibration.probability
                                (risk probability)
```

Nothing above the service line loads a model file, engineers a feature, or
applies a threshold. That is not tidiness — duplicated inference logic is how a
dashboard ends up disagreeing with the API it is supposed to be a view of, and
how a serving path drifts away from the training path one small edit at a time.

## The serving pipeline

`BatteryDigitalTwinService.create_snapshot` runs, in order:

1. **Validate** the supplied history — schema, ordering, duplicates, size cap.
2. **Derive health** with `data.loader.derive_health` — the *training* function,
   not a lookalike. Trailing-median smoothing, same reference, same SoH.
3. **Generate features** with `features.engineering.build_features` — the
   *training* generator. Stateless and causal, so the column set is a pure
   function of configuration and a one-cell batch produces the same columns as
   the training table.
4. **Assess data quality** — and stop here if `INSUFFICIENT`.
5. **Transform** with the *fitted* `FeaturePipeline` from the bundle: the
   persisted fallback imputation values, pruning decisions, column order and
   scaler statistics learned at training time.
6. **Check the warm-up policy** — is this cycle scoreable at all under the rule
   training used? (`features.warmup`)
7. **Predict** RUL, SOH, risk (independent bundles and/or the multi-task model).
8. **Calibrate** the risk probability with the persisted calibrator.
9. **Attach a conformal interval** to the RUL point estimate.
10. **Attribute** the prediction to features (SHAP for trees, ablation otherwise).
11. **Apply the deterministic recommendation rules.**
12. **Assemble the snapshot**, tagging every value's provenance.

Steps 2, 3 and 5 are literally the training code paths. Step 6 reads the same
policy object the training pipeline reads.

## Provenance tagging

Every quantity in a snapshot carries one of:

| Tag | Meaning |
|---|---|
| `observed` | measured by the rig, passed through unchanged |
| `derived` | computed deterministically from observations |
| `predicted` | a model output — a claim, not a fact |
| `estimated` | an uncertainty statement about a prediction |
| `rule_based` | produced by the recommendation rules, no model involved |

A dashboard that renders "SOH 84.6 %" beside "RUL 38 cycles" in the same
typeface invites the reader to treat both as measurements, and one of them has a
19-cycle interval around it. The tag is a required field, not an optional
annotation, so every surface can render the difference.

## Domain model

`src/battery_rul/digital_twin/domain.py`, all Pydantic, all `extra="forbid"`:

```
BatteryTwinSnapshot
├── battery_id, generated_at_utc, disclaimer, warnings[]
├── identity            : BatteryIdentity
├── measurement_summary : BatteryMeasurements     (observed)
├── health              : BatteryHealthState      (predicted + measured)
├── prediction          : BatteryPrediction
│   └── rul_interval    : BatteryUncertainty      (estimated)
├── failure_risk        : BatteryRiskAssessment   (predicted, calibrated)
├── explanation         : BatteryExplanation
│   └── drivers[]       : DegradationDriver       (derived)
├── recommendation      : BatteryRecommendation   (rule_based)
├── data_quality        : DataQualityAssessment   (derived)
└── metadata            : TwinMetadata
```

`TwinMetadata` carries the model version and name, the preprocessing
fingerprint, the data fingerprint, the git revision, the end-of-life definition,
the risk horizon, the uncertainty method, the calibration method, the SOH
definition, the input cycle range and the warm-up policy — so a snapshot can be
interpreted a year later without the code that produced it.

The wire format is deliberately separate from the internal model classes. An
internal refactor must not be a breaking API change.

## Artifact contract

A bundle directory is complete or it is not loadable:

```
artifacts/<task>/
├── model.pkl           fitted estimator
├── preprocessing.pkl   fitted FeaturePipeline
├── metadata.json       BundleMetadata
├── calibration.pkl     optional probability calibrator
└── uncertainty.pkl     optional conformal estimator
```

`load_bundle` validates the schema version, the required metadata fields, the
agreement between the persisted feature schema and the preprocessing artifact,
and — when `artifacts.strict_compatibility` is on — every data-affecting
configuration field against the persisted training configuration. A mismatch
raises `ArtifactCompatibilityError` naming the field. The alternative, warning
and continuing, means the warning is read once and ignored forever.

## Failure behaviour

| Situation | Behaviour |
|---|---|
| No bundle present | `/health` 200, `/ready` 503 with the reason; snapshot returns measured/derived values only, predictions null |
| Bundle incompatible with runtime config | `ArtifactCompatibilityError` at load, recorded in `readiness().errors` |
| History shorter than the warm-up requirement | 200 with `data_quality_class: INSUFFICIENT`, null prediction, `INSUFFICIENT_DATA` recommendation |
| Cell before its first scoreable cycle | `prediction.is_scoreable = false` and an explicit `unscoreable_reason` |
| Duplicate or malformed cycles | 422 with a structured error and a request id |
| Explanation fails | prediction still returned; explanation method `unavailable` |

## Security

- Model paths come from configuration only. No request input names a file, a
  model or a pickle.
- Request bodies carry measurements and are never logged; only method, path,
  status and duration are.
- `battery_id` rejects path separators.
- Request size is capped twice: a hard schema cap (20 000 records) and a
  configurable service cap (`service.max_history_cycles`).
- Bundles are joblib pickles loaded only from trusted local paths this process
  configured.
