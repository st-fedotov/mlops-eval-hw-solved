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

# 3. Start the infrastructure stack
docker compose up -d
# Brings up MLflow + Postgres + MinIO + Prometheus + Grafana. Wait ~30 sec.

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
    {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct", "role": "main_assistant",
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

1. Iterate on `configs/<your_config>.yaml` and the prompts it references in dev mode (`ASSISTANT_CONFIG=your_config`).
2. When a config looks good, run a full eval: `python -m src.eval --config your_config`.
3. The eval auto-registers the result as a new version of `travel-assistant` and prints `registered: travel-assistant vN`. The same version is visible in the MLflow UI under Models.
4. Review version N's metrics. If it meets your bar (e.g. `accuracy_overall >= 0.9`, `verdict_rate_leaked <= 0.02`, `total_cost_usd <= $X`), **promote** it by assigning the `Production` alias:
   ```
   mlflow models set-alias travel-assistant Production N
   ```
   Or via UI: open version N → *Set alias* → `Production`.
5. Restart the service. It re-resolves the alias and picks up the new version.
6. (Future) The cron'd golden-set replay in `docs/serverless.md` re-runs the same dataset against the deployed version on a schedule. If metrics diverge from the original eval, you've caught upstream drift.

**Rollback** is one alias update plus a restart: `mlflow models set-alias travel-assistant Production N-1`. The previous version is a config that *already passed eval*; no risk of shipping something unmeasured.

This is the integrity guarantee. The `configs/` directory is a development scratchpad. Versions in the Registry are immutable; aliases are mutable but their reassignment is an audited event. The deployment lineage from a Grafana spike runs: `model_name` + `model_alias` + `model_version` label → MLflow version → source run → measured metrics + exact prompts.

## UIs

| URL | What it shows |
|-----|---------------|
| http://localhost:5000 | MLflow tracking server — compare eval runs across configs |
| http://localhost:3000 | Grafana — the *Travel Assistant — Live Monitoring* dashboard (anonymous Viewer; admin/admin to edit) |
| http://localhost:9090 | Prometheus — raw metrics + PromQL query UI |
| http://localhost:8000/metrics | Prometheus exposition straight from the assistant service |
| http://localhost:8000/health | Liveness check |

## Iterating on configs

The dev loop:

1. Add a new file in `configs/` — copy an existing one (e.g. `configs/v4.yaml`) and rename it to describe the change (e.g. `configs/v4_smaller_classifier.yaml`). Don't edit existing config files in place; the filename stem *is* the `config_id`, and editing breaks the link between any prior MLflow run with that id and what's now on disk.
2. Edit prompts in `prompts/` if you're changing system or classifier prompts. Same append-only convention.
3. Update `ASSISTANT_CONFIG` in your `.env` to point at the new config.
4. Restart the service. (`uvicorn --reload` only reloads source files; the config is bound by the lifespan on startup, so flipping configs requires a full restart.)
5. `python -m src.eval --config <new>` — new MLflow run, auto-registered as a new version of `travel-assistant`.
6. Compare in MLflow UI. When a version clears your bar, set its `Production` alias to promote it.

Full task description: [`docs/README.md`](docs/README.md). Reference solution walkthrough: [`docs/reference_solution.md`](docs/reference_solution.md). Serverless v2 sketch: [`docs/serverless.md`](docs/serverless.md).

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
