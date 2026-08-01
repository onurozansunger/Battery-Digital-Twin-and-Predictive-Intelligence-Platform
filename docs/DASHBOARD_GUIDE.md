# Dashboard guide

## Run it

```bash
streamlit run src/battery_rul/dashboard/app.py
```

Two back-ends, selected by `service.dashboard_mode`:

- `service` (default) — calls `BatteryDigitalTwinService` in-process.
- `api` — calls the running FastAPI service at `service.dashboard_api_url`.

Both return the same `BatteryTwinSnapshot`, because both are the same code path;
one of them just goes through a socket first. The dashboard holds **no** model
logic, feature engineering or thresholds of its own.

## Prerequisites

The dashboard reads `data/processed/cycles.parquet`, produced by the Milestone 1
pipeline. If it is absent the dashboard says so and stops. **No sample fleet is
generated** — inventing plausible-looking battery data for a demo is how a
prototype gets mistaken for a working product.

```bash
python scripts/run_pipeline.py --config configs/default.yaml
python -m battery_rul.pipelines.run_milestone_2 --config configs/default.yaml
```

## Pages

| Tab | Contents |
|---|---|
| **Overview** | Cohort table and measured capacity degradation. Everything on this page is measured or directly derived. |
| **Digital Twin** | Latest cycle, SOH, health class, RUL with its prediction interval, calibrated risk, risk class, data quality, the rule-based recommendation with its evidence, and the full snapshot JSON. |
| **RUL** | Trajectory replay: each point is what the twin *would have predicted* using only the cycles available at that time, with interval bounds. |
| **SOH** | Measured versus model-estimated SOH side by side, the band definitions, and the capacity, temperature and internal-resistance trends. |
| **Failure Risk** | Calibrated probability, raw score, decision threshold, risk band, and the risk trajectory. Opens with the derived-label warning. |
| **Explainability** | Top degradation drivers with direction, magnitude, current value and training reference, plus the method and its caveat. |
| **Model Performance** | Metric tables read directly from the report artifacts. Nothing is recomputed here. |
| **Data Quality** | Per-check results, warnings, missing features, out-of-distribution flags. |
| **About & Limitations** | Definitions in force, known limitations, loaded model metadata. |

## The "as of cycle" control

The sidebar slider truncates the history handed to the twin. Only cycles at or
before the selected point are shown to the model, so the panel answers "what
would we have known at cycle *k*?" rather than "what do we know now, displayed
next to cycle *k*". Sliding it back is the quickest way to see the prediction
interval widen as evidence is removed.

## Accessibility

Health, risk and quality categories are **never signalled by colour alone**.
Every badge carries its text label and a shape prefix (`●` `◐` `◑` `▲`), so the
information survives a monochrome display or a colour-vision difference. Chart
series are labelled rather than distinguished only by hue.

## Honesty conventions

- Predicted quantities are labelled "(predicted)"; measured ones "(measured)".
- The RUL interval is always shown with its point estimate, never alone.
- The recommendation panel is headed "rule-based, not a model output".
- The demo loader strips every label column (`rul_cycles`, `eol_cycle`,
  `life_fraction`, `soh_target`, …) before the frame reaches the UI, so nothing
  on screen can accidentally display a target as a measurement.

## Screenshots

Not committed to the repository. Generate them from your own run — a screenshot
of someone else's numbers is not evidence about yours.
