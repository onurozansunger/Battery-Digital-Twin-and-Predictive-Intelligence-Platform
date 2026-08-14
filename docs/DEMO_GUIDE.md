# Demo guide

A fifteen-minute tour of the platform, from a clean checkout.

---

## 0. Install

```bash
pip install -e ".[all,dev]"
```

## 1. Build the models (once, ~10 minutes)

With the NASA dataset:

```bash
python scripts/download_data.py                                   # ~209 MB
python scripts/run_pipeline.py --config configs/default.yaml      # Milestone 1
python -m battery_rul.pipelines.run_milestone_2 --config configs/default.yaml
```

Without it — everything below still works, on synthetic cells:

```bash
python scripts/run_pipeline.py --config configs/synthetic.yaml
python -m battery_rul.pipelines.run_milestone_2 --config configs/synthetic.yaml
```

## 2. Build the drift reference

```bash
python -m battery_rul.pipelines.build_reference --config configs/default.yaml
```

Fitted on the training partition only. It prints the feature count, the row
count and a content fingerprint.

## 3. Score a fleet

The measured cohort:

```bash
python -m battery_rul.pipelines.run_fleet_batch --config configs/default.yaml \
    --fleet-id NASA-COHORT --source processed
```

Or a demonstration fleet of 24 cells at a range of ages:

```bash
python -m battery_rul.pipelines.run_fleet_batch --config configs/default.yaml \
    --fleet-id DEMO-FLEET-01 --source demo --demo-size 24
```

Both write `artifacts/fleet/snapshots/<id>.json`, a ranking CSV, a maintenance
plan, a replacement plan, and `reports/milestone_3/fleet_summary.json`, and
persist the snapshot and its prediction records to SQLite.

## 4. Monitor it

```bash
# Seed a prediction-drift reference from the batch you just ran (once)
python -m battery_rul.pipelines.run_monitoring --config configs/default.yaml \
    --fleet-id DEMO-FLEET-01 --set-prediction-reference

# Then run the full suite
python -m battery_rul.pipelines.run_monitoring --config configs/default.yaml \
    --fleet-id DEMO-FLEET-01 --source demo --demo-size 24
```

Reports land in `reports/milestone_3/`: data quality, feature drift, prediction
drift, model performance, active alerts.

## 5. Read the report

```bash
python -m battery_rul.pipelines.generate_fleet_report --config configs/default.yaml \
    --fleet-id DEMO-FLEET-01
open reports/milestone_3/fleet_report.md
```

## 6. Registry lifecycle

```bash
python -m battery_rul.pipelines.register_model --config configs/default.yaml \
    --model-name battery-rul --model-version 1.0.0 --bundle artifacts/rul \
    --validation-status VALIDATED

python -m battery_rul.pipelines.evaluate_promotion --config configs/default.yaml \
    --model-name battery-rul --model-version 1.0.0 \
    --unit-tests --contract-tests --smoke-test --leakage-check
```

On this repository's real bundle the gate returns **REJECTED**: marginal
interval coverage passes (0.917 against a 0.80 floor), but the worst held-out
cell B0033 covers only 0.703. It remains a candidate; see
`docs/MODEL_PROMOTION.md`.

## 7. Serve it

```bash
python -m battery_rul.api.app          # http://127.0.0.1:8000/docs
```

```bash
curl -s localhost:8000/health
curl -s localhost:8000/ready | jq
curl -s "localhost:8000/v1/fleet/DEMO-FLEET-01/summary" | jq '.summary'
curl -s "localhost:8000/v1/fleet/DEMO-FLEET-01/critical-batteries?page_size=5" | jq '.total_critical'
curl -s localhost:8000/metrics | head -20
```

## 8. Look at it

```bash
streamlit run src/battery_rul/dashboard/fleet_app.py    # fleet, 14 pages
streamlit run src/battery_rul/dashboard/app.py          # one cell, Milestone 2
```

---

## About the demo fleet

Two constructions, both clearly labelled, chosen automatically:

**Derived** (default when a processed cycle table exists). Each demo cell is one
of the **measured** cells, truncated at a different point in its own life, under
a `DEMO-` identifier. The measurements are real; the fleet is not. Several demo
cells share an underlying physical cell, so the fleet size is not a count of
distinct cells.

**Synthetic** (fallback). Cells from the physics-informed generator. Worth
running once to see what the platform does with out-of-distribution input: a
model trained on the NASA cohort scoring cells from a different generator flags
dozens of out-of-distribution features per cell and its remaining-life estimates
collapse to the clip floor. That is the data-quality machinery doing its job.

```bash
# force the synthetic construction
python -c "
from battery_rul.config import load_config
from battery_rul.fleet.demo import DemoFleetSpec, ingest_demo_fleet
cfg = load_config('configs/default.yaml')
result, histories = ingest_demo_fleet(cfg, DemoFleetSpec(n_batteries=8, construction='synthetic'))
print(result.accepted_count, 'cells;', result.warnings[-1][:80])
"
```

Both carry `is_demo_data=True` into the ingestion result, the snapshot identity,
the fleet summary, a fleet-level warning, the report banner and a red banner on
every dashboard page. **No metric computed on a demo fleet is a research
result.**

---

## Optional: a Battery Passport export

```python
from battery_rul.config import load_config
from battery_rul.fleet.passport import SuppliedBatteryMetadata, build_passport
from battery_rul.persistence import build_repository

cfg = load_config("configs/default.yaml")
snapshot = build_repository(cfg).latest_fleet_snapshot("DEMO-FLEET-01")
passport = build_passport(
    snapshot.batteries[0], cfg,
    SuppliedBatteryMetadata(chemistry="LCO 18650", manufacturer="<operator-supplied>"),
)
print(passport.to_json_dict())
```

Not regulatory compliance, and it says so on the document. Fields the platform
cannot source stay empty and are listed as `unavailable`; no manufacturer or
carbon-footprint value is ever generated. See the module docstring in
`src/battery_rul/fleet/passport.py`.

---

## What to read next

| Question | Document |
| --- | --- |
| How does this fit together? | `docs/FLEET_ARCHITECTURE.md` |
| Why is this cell P1? | `docs/MAINTENANCE_PRIORITY_ENGINE.md` |
| Is the drift real? | `docs/FEATURE_DRIFT.md`, `docs/MONITORING_ARCHITECTURE.md` |
| Which model is live? | `docs/MODEL_REGISTRY.md` |
| What should I not believe? | `docs/MILESTONE_3_LIMITATIONS.md` |
