# CI/CD

Four workflows. None of them holds a secret, and none publishes anything.

| Workflow | Trigger | Jobs |
| --- | --- | --- |
| `ci.yml` | push to main, PR, manual | lint, type-check, test (3.11 + 3.12), smoke, contract, hygiene |
| `docker.yml` | PR/push touching Docker or source, manual | compose-validate, build × 3 targets |
| `security.yml` | push, PR, weekly cron, manual | dependency-audit, bandit, secrets, path hygiene |
| `release.yml` | version tag or manual | quality-gate, artifacts, images |

> **Not executed here.** These files were written and their YAML validated on
> this machine; they have not run on a GitHub runner from this session. The test,
> lint and type-check steps mirror commands that *were* run locally — see
> `docs/MILESTONE_3_EVALUATION.md` for what was actually executed.

---

## `ci.yml`

**lint** — `ruff check src tests scripts`, `black --check src tests scripts`.

**type-check** — `mypy src tests scripts`, pinned to `python_version = 3.11`
(the lowest supported interpreter) so a 3.12-only construct is caught here
rather than in the 3.11 test job.

**test** — matrix over Python 3.11 and 3.12, `pytest -m "not slow"` with
coverage. The one test that parses real `.mat` files is deselected: the 209 MB
NASA archive is not downloaded in CI, and everything else runs against the
synthetic generator.

**smoke** — the end-to-end proof, on synthetic cells:

1. Milestone 1 full pipeline
2. Milestone 2 digital-twin pipeline
3. example snapshot
4. **Milestone 3**: build reference → fleet batch → monitoring → fleet report
5. **Milestone 3**: register a bundle and evaluate the promotion gate
6. assert every expected artifact exists
7. both dashboards import and compile
8. no absolute machine paths in the generated artifacts

The promotion-gate step ends in `|| true` on purpose. A REJECTED verdict exits
2, and a gate that fails the build creates pressure to loosen the gate.
Promotion itself is never automated.

**contract** — the API contract, regression, registry and persistence tests,
plus import smoke tests for every new package. These run against fixture bundles
built inside the tests, so they need neither the dataset nor a trained model,
and they are the gate that catches a breaking change to a published response
shape.

**hygiene** — `scripts/sanitise_reports.py --check` (no absolute machine paths
in committed artifacts) and `scripts/check_secrets.py`.

### Training does not run on every PR

The smoke job trains *tiny* models on synthetic cells (a few minutes). Real
training on the NASA cohort runs on a developer's machine or a dedicated
runner, deliberately: a CI pipeline that retrains on every push burns runner
time and, worse, makes "the model changed" invisible in a diff.

---

## `docker.yml`

`compose-validate` runs `docker compose config --quiet` and `bash -n` on the
entry point — both cheap, and both catch the mistakes that otherwise surface
only in a deployment.

`build` builds each target with layer caching, then asserts:

* the image's configured user is neither empty nor `root`;
* no dataset or model artifact is baked in (`/app/data/raw`, `/app/models/zoo`,
  `/app/artifacts/rul/model.pkl` must not exist);
* the package imports inside the image;
* for the API target: `/health` answers **and** `/ready` returns **503** with no
  artifacts mounted. A service that claims to be ready without a model is worse
  than one that admits it is not, so the workflow asserts the 503 rather than
  tolerating it.

Nothing is pushed to a registry.

---

## `security.yml`

| Job | Tool | Blocking? |
| --- | --- | --- |
| dependency-audit | `pip-audit` over everything | no — reported |
| dependency-audit | `pip-audit` over the serving surface (fastapi, starlette, uvicorn, pydantic, httpx, jinja2, urllib3, requests) | **yes** |
| static-analysis | `bandit -r src/battery_rul -ll` | yes |
| secrets | `scripts/check_secrets.py` | yes |
| filesystem-hygiene | `scripts/sanitise_reports.py --check` | yes |

The split audit is the point. This project's dependency tree includes scientific
libraries whose advisories are frequently unfixable in the short term; a
blocking full audit trains reviewers to skip it. The packages that actually face
untrusted input do block.

Bandit's known-and-accepted findings are the joblib loads, documented in
`docs/SECURITY.md` — they only ever read paths this process configured.

The weekly cron matters: a vulnerability disclosed after a merge is still found.

---

## `release.yml`

Triggered by a `v*.*.*` tag or manually.

1. **quality-gate** — lint, format, types, tests. Nothing is built if this fails.
2. **artifacts** — `python -m build`, `sha256sum`, and a `RELEASE_MANIFEST.json`
   recording the version, git revision, build time and per-artifact checksums,
   so a downloaded file can be tied back to a commit without trusting its name.
3. **images** — builds all three targets and reports digests and sizes.

**Nothing is published.** Pushing to a container registry or to PyPI needs
credentials this repository does not hold. Add the publish step when the secrets
exist, not before.

---

## Required repository secrets

**None.** Every workflow runs on the default `GITHUB_TOKEN` with `contents:
read`.

If you later add publishing, you will need:

| Secret | Used by | Purpose |
| --- | --- | --- |
| `REGISTRY_USERNAME` / `REGISTRY_TOKEN` | a new publish step in `release.yml` | container registry push |
| `PYPI_API_TOKEN` | a new publish step | package publishing |
| `MLFLOW_TRACKING_URI` | a remote-tracking step | remote MLflow |

Names are documented; values are not invented here.

---

## Running the same checks locally

```bash
make lint        # ruff + black, same scope as CI
make type        # mypy, same scope
make test        # pytest
make smoke       # the synthetic end-to-end pipeline
python scripts/check_secrets.py
python scripts/sanitise_reports.py --check
```

The scope string `src tests scripts` is identical in the Makefile, the README
and every workflow. Notebooks are excluded in `pyproject.toml`, not by omitting
them from a command.
