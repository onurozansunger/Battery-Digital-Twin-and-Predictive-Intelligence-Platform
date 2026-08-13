# Battery RUL platform — common tasks.
.PHONY: help install data prepare train tune evaluate predict all fast smoke test test-fast lint type format clean clean-data \
	milestone2 snapshot api dashboard lock sanitise \
	reference fleet monitoring fleet-report milestone3 fleet-dashboard docker-build secrets

# Fleet defaults for the Milestone 3 targets; override on the command line:
#   make fleet FLEET_ID=NASA-COHORT FLEET_SOURCE=processed
FLEET_ID ?= DEMO-FLEET-01
FLEET_SOURCE ?= demo

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev extras
	pip install -e ".[dev]"

data:  ## Download and unpack the NASA dataset (~209 MB)
	python scripts/download_data.py

prepare:  ## Stage 1 — build the modelling dataset
	python scripts/prepare_data.py --config configs/default.yaml

tune:  ## Stage 1b — Optuna hyperparameter search
	python scripts/tune.py --config configs/tuned.yaml

train:  ## Stage 2 — fit the model zoo
	python scripts/train.py --config configs/default.yaml

evaluate:  ## Stage 3 — figures, SHAP and the evaluation report
	python scripts/evaluate.py --config configs/default.yaml

predict:  ## Stage 4 — score the held-out cells
	python scripts/predict.py --config configs/default.yaml

all:  ## Full pipeline with the default config
	python scripts/run_pipeline.py --config configs/default.yaml

fast:  ## Full pipeline, reduced zoo (~1 minute)
	python scripts/run_pipeline.py --config configs/fast.yaml

smoke:  ## Full pipeline on synthetic data — no dataset needed
	python scripts/run_pipeline.py --config configs/synthetic.yaml

test:  ## Run the test suite
	pytest

test-fast:  ## Tests, skipping the real-data parse test
	pytest -m "not slow"

lint:  ## Static checks (same scope as CI)
	ruff check src tests scripts
	black --check src tests scripts

type:  ## Static type check (same scope as CI)
	mypy src tests scripts

format:  ## Auto-format
	ruff check --fix src tests scripts
	black src tests scripts

milestone2:  ## Milestone 2 — targets, models, calibration, bundles, report
	python -m battery_rul.pipelines.run_milestone_2 --config configs/default.yaml

snapshot:  ## Write an example digital-twin snapshot
	python scripts/example_snapshot.py --config configs/default.yaml

api:  ## Serve the FastAPI application
	python -m battery_rul.api.app

dashboard:  ## Serve the Streamlit single-cell dashboard
	streamlit run src/battery_rul/dashboard/app.py

# ---------------------------------------------------------------- milestone 3
reference:  ## Milestone 3 — build the drift reference from the training partition
	python -m battery_rul.pipelines.build_reference --config configs/default.yaml

fleet:  ## Milestone 3 — score a fleet and persist the snapshot, ranking and plans
	python -m battery_rul.pipelines.run_fleet_batch --config configs/default.yaml \
		--fleet-id $(FLEET_ID) --source $(FLEET_SOURCE)

monitoring:  ## Milestone 3 — data quality, drift and delayed-label performance
	python -m battery_rul.pipelines.run_monitoring --config configs/default.yaml \
		--fleet-id $(FLEET_ID) --source $(FLEET_SOURCE)

fleet-report:  ## Milestone 3 — render the Markdown fleet report
	python -m battery_rul.pipelines.generate_fleet_report --config configs/default.yaml \
		--fleet-id $(FLEET_ID)

milestone3:  ## Milestone 3 — reference, fleet batch, monitoring and report
	$(MAKE) reference fleet monitoring fleet-report

fleet-dashboard:  ## Serve the Streamlit fleet dashboard
	streamlit run src/battery_rul/dashboard/fleet_app.py

docker-build:  ## Build the API, dashboard and jobs images
	docker build -f docker/Dockerfile --target api       -t battery-rul-api:local .
	docker build -f docker/Dockerfile --target dashboard -t battery-rul-dashboard:local .
	docker build -f docker/Dockerfile --target jobs      -t battery-rul-jobs:local .

secrets:  ## Fail if a credential pattern appears in a tracked file
	python scripts/check_secrets.py

sanitise:  ## Strip absolute machine paths from committed artifacts
	python scripts/sanitise_reports.py

lock:  ## Regenerate the pinned environment
	uv pip compile pyproject.toml --extra dev --universal --python-version 3.13 \
		--output-file requirements-lock.txt

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info htmlcov .coverage

clean-data:  ## Remove derived data and run artifacts (keeps data/raw)
	rm -rf data/interim/*.parquet data/processed/* models/zoo models/*.pkl
	rm -rf reports/* figures/* artifacts/*
