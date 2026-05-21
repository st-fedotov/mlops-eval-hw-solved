# Travel-assistant MLOps homework

A bounded-scope travel assistant built to be evaluated and monitored properly.

The base system prompt says: *answer travel questions, refuse everything else.* Your job is to iterate on prompts, models, and guardrail architectures; evaluate each config against an adversarial dataset; and run a production-shaped monitoring stack to catch what offline eval misses.

## Layout

- `data/eval_dataset.jsonl` — ~100 examples across normal travel, off-topic, jailbreak, and social-engineering categories.
- `prompts/` — system prompts and classifier prompts. Append-only by convention: don't edit existing files in place — add a new one if you're iterating.
- `configs/` — one YAML file per deployment config (model + prompt + guardrail). Filename stem is the `config_id`. Append-only by convention; iterations land as new files like `configs/v4_smaller_classifier.yaml`. The directory is a development scratchpad; the canonical record of a promoted config lives in the MLflow Model Registry.
- `src/assistant/` — FastAPI service exposing `/chat`, `/metrics`, `/health`.
- `src/judge.py` — LLM-as-judge.
- `src/eval.py` — offline evaluation against the dataset; logs to MLflow and (on full evals) auto-registers a new version under `travel-assistant`.
- `src/monitoring/` — Prometheus metrics + async sampled deep-judge worker.
- `observability/` — Prometheus scrape config + Grafana dashboards.
- `docker-compose.yml` — MLflow (Postgres + MinIO) + Prometheus + Grafana.
- `docs/` — full task description, reference solution, serverless v2 sketch.

## Prerequisites

- Docker Desktop (or another Docker-compatible runtime; the stack uses five containers).
- Python 3.11+.
- A Nebius Token Factory API key — create one at https://studio.nebius.com/.

## Setup from scratch

```bash
# 1. Clone the repo
git clone https://github.com/st-fedotov/mlops-eval-hw-solved.git
cd mlops-eval-hw-solved

# 2. Configure secrets
cp .env.example .env
# Open .env and paste your NEBIUS_API_KEY

# 3. Pull the infrastructure images and start the stack
docker compose pull
docker compose up -d
# Brings up MLflow + Postgres + MinIO + Prometheus + Grafana. Wait ~30 sec.
# `pull` is split out from `up` so any registry / network failures surface
# explicitly instead of being buried in startup output.

# 4. Install Python deps (use a virtualenv)
python -m venv .venv
# Activate it:
#   PowerShell:   .venv\Scripts\Activate.ps1
#   Windows cmd:  .venv\Scripts\activate.bat
#   Linux/macOS:  source .venv/bin/activate
pip install -e .

# 5. Start the assistant service (default config: v1)
uvicorn src.assistant.service:app --reload
# Service is now at http://localhost:8000
```

## Sending a message

In a second shell, with the service running:

```bash
# Plain curl
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find flights from Paris to Rome"}'
```

PowerShell equivalent:

```powershell
Invoke-RestMethod -Method POST -Uri http://localhost:8000/chat `
  -ContentType "application/json" `
  -Body '{"message": "Find flights from Paris to Rome"}'
```

Or use the included helper:

```bash
python scripts/chat.py "Find flights from Paris to Rome"
python scripts/chat.py --raw "Tell me a joke"          # full JSON response
```

A typical response:

```json
{
  "text": "I'd be happy to help you find flights from Paris to Rome ...",
  "refused": false,
  "input_category": null,
  "output_verdict": null,
  "model_calls": [
    {"model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", "role": "main_assistant",
     "input_tokens": 47, "output_tokens": 312, "latency_seconds": 2.1}
  ]
}
```

`refused`, `input_category`, `output_verdict` are the monitoring signals — they drive the Prometheus metrics under the hood. `input_category` is `null` for configs without an input classifier; `output_verdict` is `null` for configs without an output validator.

## Two deployment modes

The service has two startup paths, selected by env vars.

**Dev mode** — pick a config from the `configs/` directory by id (filename stem). Fast iteration on prompts and configs.

In `.env`:
```
ASSISTANT_CONFIG=v4
```

Then:
```
uvicorn src.assistant.service:app --reload
```

**Production mode** — point at a *registered* MLflow Model Registry version, resolved by alias. The service queries the Registry, downloads the deployment manifest of the version that the alias currently points at, and runs it. The local `configs/` directory is ignored entirely.

In `.env`:
```
MLFLOW_REGISTERED_MODEL_NAME=travel-assistant
ASSISTANT_MODEL_ALIAS=Production
```

Then:
```
uvicorn src.assistant.service:app
```

Promotion — which version is `Production` — is an explicit, audited operation in MLflow (see the eval → deploy flow below). Production cannot serve a config that wasn't evaluated, registered, and then promoted by alias assignment. Every Prometheus series is labelled with `model_name`, `model_alias`, and `model_version`, so any spike in Grafana is one click away from the version that authorized the deployment.

Config is bound at startup. To switch, restart the service.

## Running an offline eval

```bash
# Full eval against the 100-example dataset (~10–20 min depending on config)
python -m src.eval --config v1

# Quick check while developing (not registered to the Registry)
python -m src.eval --config v4 --limit 25

# Force registration on a partial eval (or skip it on a full one)
python -m src.eval --config v4 --limit 25 --register
python -m src.eval --config v4           --no-register
```

Each invocation is a new MLflow run. On full evals (no `--limit`), the run's `config.json` artifact is automatically registered as a new version of `travel-assistant` — the same artifact that production mode resolves through an alias.

## The eval → deploy flow

### Configs are invisible to MLflow until they're evaluated

A YAML file in `configs/` is just text on disk. MLflow has no awareness of any config you haven't run eval on. Two levels of "seen by MLflow" to distinguish:

- **Logged as a run** — happens on *every* `python -m src.eval` invocation, including partial ones (`--limit 25`). The config goes into the experiment as a `config.json` artifact, along with metrics, predictions, and prompt artifacts. Visible in *Experiments → travel-assistant → list of runs*. Useful for inspection. **Not deployable.**
- **Registered as a Registry version** — happens on *full* evals (no `--limit`), or when you pass `--register` explicitly. Only registered versions can be promoted via the `Production` alias and resolved by the service in production mode. Partial evals deliberately skip this so dev-loop noise doesn't pollute the Registry.

So for a hypothetical new `configs/v6.yaml`:

| What you ran | Run in experiment? | Version in Registry? | Deployable? |
|---|---|---|---|
| Nothing | no | no | no |
| `python -m src.eval --config v6 --limit 25` | yes | no | no |
| `python -m src.eval --config v6` (full) | yes | yes | yes (after promotion) |

The local file alone does nothing — eval is the door that lets configs enter MLflow at all.

### How Registry versions are numbered

Every full `python -m src.eval` invocation auto-creates a new version under the *registered model* `travel-assistant`. Versions are plain integers — `1`, `2`, `3`, … — auto-assigned by MLflow at registration time. You don't choose the number.

The numbering is **per registered-model-name, not per config**. If you eval `v1` first and then `v4`, MLflow assigns version `1` to the v1 run and version `2` to the v4 run, under the same `travel-assistant` registered model. Re-evaluating `v1` later would produce version `3`. Each version is an immutable snapshot of one specific eval; the `config_id` it came from is stored as a parameter on that version, but it doesn't drive the integer.

The eval's terminal output shows the assigned number on the `registered:` line:

```
=== v1 eval summary ===
  run_id:              4058ca6c137344619c4fd65bb909f9c9
  registered:          travel-assistant v1    <-- "v1" here means version 1 of travel-assistant
  accuracy_overall:    0.750
  ...
```

### Step by step

Worked example: suppose you've just iterated on `configs/v4.yaml`, run a full eval, and the terminal reports `registered: travel-assistant v7`. (Version 7 means six prior evals had already been registered on this MLflow server.)

1. **Iterate.** Edit `configs/v4.yaml` or its prompts in dev mode (`ASSISTANT_CONFIG=v4`). Test by sending `/chat` traffic to the running service.
2. **Eval.** `python -m src.eval --config v4`. The eval logs to MLflow and, because there's no `--limit`, auto-registers the run. Note the reported version number — say it's `7`.
3. **Review.** Open MLflow UI: http://localhost:5000 → **Models** tab → click `travel-assistant`. Each version is tagged at registration time with `config_id`, `model`, `guardrail_type`, `judge_model`, and `dataset_size`, and has a one-line description summarizing accuracy / leakage / cost — so you can pick out "the v4 run" without remembering integer numbers. Click into Version 7 to see its full metrics, parameters, and artifacts. Check that `accuracy_overall`, `verdict_rate_leaked`, `total_cost_usd` clear whatever bar you've set.
4. **Promote.** If version 7 is good, assign the `Production` alias to it via the UI: open the Version 7 page → *Aliases* section → **+ Add alias** → type `Production` → enter. Note that MLflow lowercases alias names on save, so what you typed as `Production` will appear as `@production` — and your `.env` should match: `ASSISTANT_MODEL_ALIAS=production`.
5. **Deploy.** Set `ASSISTANT_MODEL_ALIAS=production` in `.env` and restart uvicorn. On startup the service resolves the alias, downloads version 7's `config.json`, and runs it. For deploys after the first one, see *Hot-reload without restart* below — you don't need to bounce the service every time you reassign an alias.
6. **(Future) Drift check.** The cron'd golden-set replay in `docs/serverless.md` re-runs the eval dataset against the deployed version on a schedule. If new metrics diverge from version 7's original eval, you've caught upstream drift.

### Hot-reload without restart

Restarting uvicorn for every alias swap is fine in dev but unacceptable in production — there's a brief window where the service isn't serving. Instead, the service exposes `POST /admin/reload`. It re-resolves the current `ASSISTANT_MODEL_ALIAS` against the Registry, downloads the new version's `config.json`, builds a fresh `Pipeline`, and atomically swaps it into `app.state.pipeline`. In-flight `/chat` requests finish on the previous pipeline; new requests pick up the new one. No downtime.

Usage:

```powershell
Invoke-RestMethod -Method POST http://localhost:8000/admin/reload
```

Response (200, ~1–2 seconds — the time of the Registry fetch and prompt download):

```json
{
  "status": "ok",
  "previous": {"config_id": "v4", "model_version": "7", ...},
  "current":  {"config_id": "v5", "model_version": "8", ...}
}
```

If anything fails during resolution or build — Registry unreachable, alias not set, artifact corrupt, manifest fails pydantic validation — the endpoint returns 500 and the running pipeline is untouched. Existing `/chat` traffic keeps working on the prior version.

**Security.** `/admin/reload` is admin surface. Gate it via `ADMIN_TOKEN` in `.env`:

```
ADMIN_TOKEN=<random hex from `openssl rand -hex 32`>
```

When set, every call must carry an `X-Admin-Token: <value>` header or it gets a 403. When unset (dev default), the endpoint is open. In production this should always be set; in front of a real LB/proxy you'd additionally restrict the endpoint to internal callers.

The reload also works in **dev mode** (when `ASSISTANT_MODEL_ALIAS` is unset): it re-reads `configs/<ASSISTANT_CONFIG>.yaml` from disk and rebuilds. Useful when iterating on prompts — no need to `Ctrl+C` uvicorn each time you save a file.

### Rollback

One alias update plus a hot reload. If version 7 turns out badly in production and version 6 was the previous good one: open Version 6 in the MLflow UI → *+ Add alias* → `Production`. This moves the alias off Version 7 onto Version 6 (each alias points at exactly one version at a time). Then `Invoke-RestMethod -Method POST http://localhost:8000/admin/reload` (or restart uvicorn if you'd rather); the service now serves version 6. Version 6 was a config that *already passed eval*, so you can't accidentally ship something unmeasured.

### What's currently in Production?

Three ways to check, depending on what you mean by "currently":

**Registry state — what version does the `production` alias point at right now.** Open MLflow UI → **Models** → `travel-assistant`. The row with a `@production` badge is the current target. Its tags tell you `config_id`, `model`, `guardrail_type`.

**Service state — what the running uvicorn is actually serving.** Can differ from the Registry if you've reassigned the alias but haven't restarted uvicorn yet (config is bound at startup; aliases are re-resolved only on restart). Two ways:

- Grafana → *Current deployment* panel (a table rendered from the `assistant_info` metric).
- Or directly from the service: `curl http://localhost:8000/metrics | Select-String assistant_info`. The labels `config_id`, `model_alias`, `model_version` show what the lifespan loaded.

**Health check.** `curl http://localhost:8000/health` just returns `{"status":"ok"}` — it doesn't tell you which config is loaded. Use `/metrics` for that.

### Integrity guarantee

The `configs/` directory is a development scratchpad. Versions in the Registry are immutable — version 7 always means what version 7 meant the moment you registered it, even if you later edit `configs/v4.yaml` on disk. Aliases are mutable but their reassignment is an audited event in MLflow. The deployment lineage from a Grafana spike runs: `model_name` + `model_alias` + `model_version` label → MLflow version → source run → measured metrics + exact prompts.

## UIs

| URL | What it shows |
|-----|---------------|
| http://localhost:5000 | MLflow tracking server — compare eval runs across configs |
| http://localhost:3000 | Grafana — the *Travel Assistant — Live Monitoring* dashboard (anonymous Viewer; admin/admin to edit) |
| http://localhost:8000/metrics | Prometheus exposition straight from the assistant service |
| http://localhost:8000/health | Liveness check |

(The Prometheus server itself runs in the compose stack at `localhost:9090` as Grafana's datasource. You usually won't open it directly — Grafana is the daily-driver UI.)

## Grafana dashboard — what each panel shows

The *Travel Assistant — Live Monitoring* dashboard has 11 panels. All series are emitted by `/chat` traffic (offline `python -m src.eval` runs do *not* feed Prometheus — they're in-process), and panels stay empty until you send some chat requests.

### Refusal rate by `input_category` (5m rolling)

`sum by (input_category) (rate(chat_requests_total{refused="true"}[5m])) / sum by (input_category) (rate(chat_requests_total[5m]))`

Fraction of `/chat` responses that were canned refusals, sliced by the input's detected category. For v4/v5 you see `travel`, `off_topic`, `suspicious` separately. For v1–v3 (no input classifier) all traffic shows as `unmonitored`. A healthy travel-only deployment has near-zero refusal for `travel` and near-1.0 for `off_topic`/`suspicious`.

### DIVERGENCE: cheap refusal-rate vs judge leakage-rate

Two series on one axis:
- *Cheap refusal-rate* (5m, 100% of traffic) — fraction of responses that exactly match the canned refusal string. Determined in microseconds by string comparison in the `/chat` handler.
- *Judge leakage-rate* (1h, sampled) — fraction of *judged* exchanges where the deep judge's verdict is `leaked`.

The two should track each other. When cheap signal says "we refused" but the judge sees real leakage, the assistant is producing partial leaks ("Sure, here's a joke. But I should remind you…") that exact-match misses. That divergence is the alert worth firing in production — it's the entire point of having both a cheap signal and a sampled deep one.

### Request rate by config

`sum by (config_id) (rate(chat_requests_total[5m]))`

Requests per second served, split by which config (v1, v2, …) is running. One series per running config. If you're A/B-ing two configs side-by-side, two series.

### Request latency (p50 / p95 / p99) by config

`histogram_quantile(0.50|0.95|0.99, sum by (le, config_id) (rate(chat_request_duration_seconds_bucket[5m])))`

p50 = typical request, p95 = slow tail, p99 = very slow tail. v4/v5 are slower than v1 because they make extra classifier calls. Sudden p95 spikes usually mean the LLM endpoint is degraded.

### Burn rate $/hour by model

`sum by (model) (rate(chat_cost_usd_total[5m])) * 3600`

Cost rate, in USD per hour, sliced by model. Use to alert on runaway spend and to attribute cost to specific models in a multi-model deployment (e.g., small classifier vs. large main assistant in v4/v5).

### In-flight requests

Current count of `/chat` calls being processed concurrently. Saturation signal — sustained high values mean the assistant is bottlenecked; healthy idle systems oscillate near 0.

### Deep judge queue depth

Pending `(input, response)` pairs waiting for the async judge worker. Should hover near 0. Monotonic growth = judge is falling behind sampled traffic; either reduce `JUDGE_SAMPLE_RATE` or use a faster judge model.

### Judge sample rate

Static gauge showing the configured `JUDGE_SAMPLE_RATE` (e.g., 0.05 = 5% of `/chat` traffic sent to the judge). Useful when reading the *Judge verdicts* and *DIVERGENCE* panels — it tells you how noisy the sampled estimates are.

### Current deployment

Table view of the `assistant_info` info-metric: `config_id`, `model`, `guardrail_type`, `model_name`, `model_alias`, `model_version`. Tells you at a glance what is *actually* serving traffic — especially useful when toggling between dev and production modes.

### Judge verdicts (1h rolling)

`sum by (verdict) (rate(judge_evaluations_total[1h]))`

Rate of each judge verdict. Five possible values: `answered_correctly`, `refused_correctly`, `leaked`, `over_refused`, `judge_error`. The first two are good; `leaked`/`over_refused` are quality regressions; `judge_error` should be ≈0 (high values mean the judge isn't following the structured-output schema). Empty until the async judge worker has actually completed sampled evaluations — set `JUDGE_SAMPLE_RATE=1.0` and send a few `/chat` calls if you want this populated quickly for testing.

### LLM API error rate by `error_type`

`sum by (error_type) (rate(llm_api_errors_total[5m]))`

Operational health. Each exception type (`RateLimitError`, `APITimeoutError`, `APIConnectionError`, …) becomes its own series. Spikes here usually mean the Nebius endpoint is throttling you or having issues; nothing about the config is wrong.

## Iterating on configs

The dev loop:

1. Add a new file in `configs/` — copy an existing one (e.g. `configs/v4.yaml`) and rename it to describe the change (e.g. `configs/v4_smaller_classifier.yaml`). Don't edit existing config files in place; the filename stem *is* the `config_id`, and editing breaks the link between any prior MLflow run with that id and what's now on disk.
2. Edit prompts in `prompts/` if you're changing system or classifier prompts. Same append-only convention.
3. Update `ASSISTANT_CONFIG` in your `.env` to point at the new config.
4. Restart the service. (`uvicorn --reload` only reloads source files; the config is bound by the lifespan on startup, so flipping configs requires a full restart.)
5. `python -m src.eval --config <new>` — new MLflow run, auto-registered as a new version of `travel-assistant`.
6. Compare in MLflow UI. When a version clears your bar, set its `Production` alias to promote it.

Full task description: [`docs/README.md`](docs/README.md). Reference solution walkthrough: [`docs/reference_solution.md`](docs/reference_solution.md). Serverless v2 sketch: [`docs/serverless.md`](docs/serverless.md).

## Image mirror

Docker Hub's CloudFront CDN drops blob downloads mid-stream from some regions, which makes pulling `grafana/grafana:latest` unreliable for students. To insulate the stack from this, the Grafana image is mirrored to this repo's GitHub Container Registry namespace by `.github/workflows/mirror-images.yml`, and `docker-compose.yml` references the GHCR path. Students don't need to touch Docker Hub at all.

For the repo owner: after first push, run the *Mirror images to GHCR* workflow once from the Actions tab; then go to https://github.com/users/&lt;owner&gt;/packages, open `mlops-grafana`, *Package settings → Change visibility → Public*. The scheduled run keeps the mirror within a week of upstream `latest`.

If you find another Hub image starts failing for students, add it to the `matrix.include` list in the workflow file (source + target name), re-run the workflow, make the new package public, and update its image reference in `docker-compose.yml`.

## Updating after `git pull`

You almost never need `docker compose down`. Containers and named volumes (MLflow DB, MinIO artifacts) stay alive across pulls; you just restart whatever has new code or config. The 90% case after `git pull` is **either nothing or one `docker compose up -d`**, not a full teardown.

| What changed in the repo | What to do |
|---|---|
| `src/**/*.py` (Python source) | If uvicorn is running with `--reload`: nothing — it auto-reloads. Otherwise `Ctrl+C` and re-run uvicorn. |
| `configs/*.yaml` or `prompts/*.txt` (you're iterating on a prompt/config in dev mode) | `Invoke-RestMethod -Method POST http://localhost:8000/admin/reload` — re-reads the active config from disk and atomically swaps the pipeline. No uvicorn restart needed. |
| Flipping `ASSISTANT_CONFIG` or `ASSISTANT_MODEL_ALIAS` in `.env`, or any other env var that pydantic-settings reads | Restart uvicorn (`Ctrl+C` then re-run). Settings is cached at first import; new env values need a fresh process. |
| MLflow Registry alias reassignment (`production` → different version) | `Invoke-RestMethod -Method POST http://localhost:8000/admin/reload` — re-resolves the alias and swaps. No restart. |
| `docker-compose.yml` (new service, env, port, build context, …) | `docker compose up -d`. Compose diffs the running stack against the new file and only recreates services that actually changed. Untouched containers keep running. |
| Image tag in compose changed, or you want the freshest `:latest` | `docker compose pull` then `docker compose up -d`. |
| `observability/grafana/dashboards/*.json` | Nothing. Grafana's provisioner re-reads the dashboards directory every 10 seconds. |
| `observability/prometheus.yml` or Grafana datasource/provisioning yaml | `docker compose restart prometheus` (or `restart grafana`). The volume mount is live but the process needs to re-read the file. |
| `Dockerfile` or anything that affects an image *built* by compose (e.g., `docker/mlflow.Dockerfile`) | `docker compose build` then `docker compose up -d`. |
| You really want a clean slate | `docker compose down && docker compose up -d`. Volumes survive. Add `-v` to `down` to also drop volumes (loses your MLflow DB and MinIO artifacts — only do this for a full reset). |

Quick mental shortcut: **Python change → uvicorn restart. Compose change → `compose up -d`. Config-file-mounted-to-running-container change → `compose restart <service>`.**

## Secrets

Your Nebius API key is yours. Never commit it; never paste it into a chat, issue, or screenshot.

This repo has three layers of defense against accidental leaks:

1. **`.gitignore`** — `.env` is ignored. `.env.example` (placeholders only) is the file that's checked in.
2. **`pre-commit` with `gitleaks`** — every `git commit` scans the staged diff for API-key-shaped strings and aborts if it finds one. One-time setup per clone:
   ```bash
   pip install pre-commit
   pre-commit install
   ```
3. **`pydantic-settings` with `SecretStr`** — keys are wrapped in a type that doesn't render in `repr()` or logs (see `src/config.py`).

In code, never `print(settings)` or include the key in error messages; log only `settings.nebius_api_key.get_secret_value()` at the point of use.

If you ever do commit a key by accident:
1. **Rotate it immediately** in the Nebius console — the old value is permanently in git history and on every fork/clone.
2. Force-push only if the commit hasn't been pulled by anyone else; otherwise treat the key as burned and only rotation matters.
3. GitHub's secret scanning will likely flag it for you anyway, and Nebius may auto-rotate if they participate in the partner program.

For the serverless v2 chapter (`docs/serverless.md`), the production answer is to fetch the key from a managed secrets store at startup rather than carry a `.env` file into the container.
