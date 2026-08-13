# Security

What this build defends against, what it does not, and what you must add before
exposing it to anything.

> **This service ships no authentication and no authorisation.** It is a research
> prototype. Do not expose it to a network you do not control. Everything below
> assumes it sits behind something that authenticates callers.

---

## Threat model

Assets: the model bundles (intellectual property and the integrity of every
decision), the cell telemetry (operational data belonging to whoever runs the
fleet), the registry (which model is live), and the availability of the service.

| # | Threat | Control | Residual risk |
| --- | --- | --- | --- |
| T1 | Arbitrary code execution via a malicious pickle | Bundles load **only** from configured local paths. No request names a path. The drift reference is JSON, never a pickle. | An attacker with filesystem write access can replace a bundle. Registry checksums detect it at promotion, not at load. |
| T2 | Path traversal through a fleet file or reference id | `resolve_within()` refuses anything outside the permitted directory; reference ids are validated against a character allow-list; battery and fleet ids reject separators and `..`. | None known for the HTTP surface, which accepts no paths at all. |
| T3 | Memory exhaustion via a large request | Hard schema caps (500 batteries, 20 000 cycles) that configuration cannot raise; configurable lower limits; 413 with the batch command. | A proxy-level body limit is still recommended (`deployment.max_request_bytes` documents the number to set). |
| T4 | Unauthorised model promotion | Admin endpoints disabled by default (403), refused in read-only mode, and promotion re-runs the gate. | With admin endpoints enabled and no authentication in front, anyone who can reach the port can promote. **Do not enable without authentication.** |
| T5 | Telemetry leaking into logs | The structured formatter drops `history`/`records`/`frame`/`cycles`/`payload`/`raw` fields; the tracker drops the same; request bodies are never logged. | Application code could still format a value into a message string by hand. |
| T6 | Filesystem layout leaking to clients | Registry responses omit `bundle_path`; report paths are stored relative; error bodies are tested for absolute paths. | None known. |
| T7 | Supply-chain vulnerability | `pip-audit` weekly and per PR; the serving-surface subset blocks the build. | The scientific stack often has unfixable-in-the-short-term advisories; those are reported, not blocking, and triaged. |
| T8 | Credential committed to the repository | `scripts/check_secrets.py` in CI and in the hygiene job. | Pattern-based; it will not catch a high-entropy string with no recognisable prefix. |
| T9 | Cross-origin browser access | CORS is an explicit allow-list, empty by default. | None while empty. |
| T10 | Container escape / privilege escalation | Non-root user (uid 10001), `no-new-privileges`, read-only root filesystem, writable volumes only. | Standard container isolation; no seccomp profile is shipped. |

---

## Secrets

**There are no secrets in this repository, and none are required to run it.**

The optional integrations and what they would need:

| Integration | Secret | Where it would go |
| --- | --- | --- |
| Remote MLflow tracking | `MLFLOW_TRACKING_URI`, credentials | environment variable, never committed |
| Container registry publishing | registry token | GitHub Actions secret, used only by a publish step that does not currently exist |
| External alerting | webhook URL / API token | not implemented; `monitoring.alerts.external_notifications` is `False` and frozen |

If you add one:

* environment variables or a mounted secrets file — never a config file in git;
* never in a Docker image layer (`.dockerignore` excludes `.env*`, `*.pem`, `*.key`, `secrets/`);
* never in a log line, a tracking parameter or an artifact;
* rotate on exposure, and treat any value that reached a build log as exposed.

`python scripts/check_secrets.py` fails the build on AWS keys, GitHub/Slack/
Google/Anthropic/OpenAI tokens, private-key blocks, JWTs, hardcoded password
assignments and database URLs with embedded credentials. If a finding is a false
positive, make the value obviously a placeholder (`<your-token>`, `${TOKEN}`)
rather than adding an exception — an exception list is where a real key
eventually hides.

---

## Serialisation

Model bundles are joblib pickles, which are executable content. The controls:

1. bundles load only from directories named in configuration — no request, no
   environment variable and no filename from outside the process selects one;
2. `load_bundle` validates metadata, schema version and feature-schema
   consistency before use;
3. registry entries carry a SHA-256 over the bundle files, re-verified at
   promotion;
4. the monitoring reference — the artifact a long-running service reads most
   often — is **JSON**, so no code path deserialises executable content from it.

Replacing the pickle format entirely (ONNX, skops) is the right long-term fix
and is not done here; the compensating controls are listed above and are what
the current design relies on.

---

## Deployment checklist

Before exposing this service to anything:

- [ ] authentication in front of it (reverse proxy, API gateway, or a mesh)
- [ ] TLS terminated in front of it
- [ ] `deployment.admin_endpoints_enabled` left `false`, or authenticated
- [ ] `deployment.cors_allow_origins` set to the exact origins that need it
- [ ] request body limit set at the proxy
- [ ] rate limiting at the proxy (none is implemented here)
- [ ] containers run non-root with a read-only root filesystem (the compose file does)
- [ ] model artifacts mounted read-only where the service does not write them
- [ ] logs shipped somewhere with access control — they contain fleet and cell ids
- [ ] `deployment.read_only=true` on any replica that must not write

---

## Incident response: the model is unavailable

**Symptom.** `/ready` returns 503; `/health` still returns 200; prediction
endpoints return 503 with `models_unavailable`; a `MODEL_UNAVAILABLE` alert is
raised on the next monitoring run.

1. `curl -s localhost:8000/ready | jq` — the `errors` map names each bundle and
   why it failed to load.
2. Common causes, in order of frequency:
   * artifacts not mounted (compose volume missing) — fix the mount;
   * `ArtifactCompatibilityError` — the runtime configuration disagrees with the
     training configuration on a data-affecting field. Do **not** set
     `artifacts.strict_compatibility=false` to make it start; find the field.
   * bundle incomplete (a partial copy) — re-copy or rebuild.
3. Keep the instance out of rotation: `/ready` already reports 503, which is what
   a load balancer needs.
4. If a recent promotion caused it:
   `python -m battery_rul.pipelines.rollback_model --model-name <name> --by <you> --reason "…"`,
   then restart the service so it loads the restored artifacts.
5. The batch and monitoring jobs fail loudly rather than producing empty
   snapshots; a fleet snapshot produced with no model loaded carries an explicit
   warning and null predicted fields.

**What never happens automatically:** no fallback to a previous model, no dummy
predictions, no silent degradation. A service that cannot load a model says so.

---

## Reporting a vulnerability

This is a research prototype with no support commitment. Open an issue for
anything non-sensitive; for something you would rather not post publicly,
contact the repository owner directly.
