# Serverless v2 — design sketch

The current homework runs everything locally via `docker-compose`. This document sketches what the v2 deployment looks like on Nebius Cloud. It's intentionally a design doc, not running code.

## Component map

| Local (v1)                | Production (v2)                                                                       |
|---------------------------|---------------------------------------------------------------------------------------|
| `uvicorn` on host         | Nebius Serverless Containers                                                          |
| `prometheus` container    | Self-hosted Prometheus on a Nebius VM, *or* Nebius Monitoring (push model — see below)|
| `grafana` container       | Self-hosted Grafana on the same VM, or Grafana Cloud                                  |
| `postgres` container      | Nebius Managed PostgreSQL                                                             |
| `minio` container         | Nebius Object Storage (S3-compatible)                                                 |
| `mlflow` container        | MLflow stateless container on Nebius Serverless Containers, backed by managed PG + Object Storage |

The MLflow tracking server is stateless if you externalize the backend store (Postgres) and artifact store (S3) — v1 was built with that shape on purpose, so v2 is a config change, not a redesign.

## Secrets

The `.env` pattern from local dev does *not* ship to production.

**Recommended starting point (Option A).** Set `NEBIUS_API_KEY` as an environment variable in the Nebius Serverless Containers deployment configuration. Nebius stores it encrypted at rest; at runtime it's injected into the container's `os.environ`. `pydantic-settings` picks it up unchanged — zero code delta from local dev.

**Stronger (Option B).** Fetch the key from a managed secrets store at container startup, using the container's workload identity (instance-metadata token) to authenticate. Rotation is centralized — change the value once, no redeploy.

**Strongest (Option C).** Short-lived workload-identity tokens for calling Token Factory directly — no long-lived API key exists at all. Verify against current Nebius product details before promising students this path.

## Monitoring in serverless

Serverless containers are ephemeral, so the *scrape* model (Prometheus pulling `/metrics` from a stable address) doesn't work directly. Two options:

1. **Pushgateway.** Each container instance pushes metrics to a Prometheus Pushgateway every N seconds. Prometheus scrapes the Pushgateway. Pushgateway becomes a single point of failure but the model is straightforward.
2. **Nebius Monitoring or OpenTelemetry.** Send metrics directly to a managed observability backend via OTLP. More integration work, but eliminates the Pushgateway and gives you traces and logs in the same pane.

## Cron'd golden-set replay

Beyond live monitoring, run the eval dataset against the *deployed* service on a schedule — daily or weekly — and log results to MLflow as a new run each time. Catches upstream drift: Nebius rolls a new model snapshot, refusal behavior shifts overnight, the cheap signal stays green but the judge starts flagging leakage.

Implementation sketch:

- A Nebius Cloud Functions / cron-style trigger fires a small container that runs `python -m src.eval`.
- The script targets the deployed service URL over HTTP instead of building the pipeline in-process. (Small CLI-flag change in `src/eval.py`: currently in-process; for a fair production replay it should hit `/chat`.)
- The MLflow run gets a tag like `mlflow.set_tag("source", "scheduled_replay")`.

If a scheduled replay's `accuracy_overall` drops below a threshold, alert. This is the production-grade circuit breaker that catches what live cheap signals can't.

## Out of scope

- Multi-region.
- Canary deployments / auto-rollback on regression.
- LLM caching layer (some prompts repeat; an embedding-based cache would meaningfully cut cost).

These are v3 territory. For v2, the goal is just: *the homework works on Nebius Cloud, with proper secret management, observable, and self-checking via scheduled replay.*
