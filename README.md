# Travel-assistant MLOps homework

A bounded-scope travel assistant built to be evaluated and monitored properly.

The base system prompt says: *answer travel questions, refuse everything else.* Your job is to iterate on prompts, models, and guardrail architectures; evaluate each variant against an adversarial dataset; and run a production-shaped monitoring stack to catch what offline eval misses.

## Layout

- `data/eval_dataset.jsonl` — ~100 examples across normal travel, off-topic, jailbreak, and social-engineering categories.
- `prompts/` — system prompts and classifier prompts.
- `variants.yaml` — full configuration of each assistant deployment (model + prompt + guardrail). Adding a new variant is one YAML block.
- `src/assistant/` — FastAPI service exposing `/chat` and `/metrics`.
- `src/judge.py` — LLM-as-judge.
- `src/eval.py` — offline evaluation against the dataset; logs to MLflow.
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

# 5. Start the assistant service (default variant: v1)
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

`refused`, `input_category`, `output_verdict` are the monitoring signals — they drive the Prometheus metrics under the hood. `input_category` is `null` for variants without an input classifier; `output_verdict` is `null` for variants without an output validator.

## Choosing a variant at deployment

Which variant the service runs is chosen at startup time via the `VARIANT` env var (default: `v1`). Each variant in `variants.yaml` fully describes one deployment: main model, system prompt, guardrail architecture.

```bash
# v2 — positive-list scope + canned refusal string
VARIANT=v2 uvicorn src.assistant.service:app --reload

# v4 — input classifier in front of the main assistant
VARIANT=v4 uvicorn src.assistant.service:app --reload

# v5 — input classifier + output validator (sandwich)
VARIANT=v5 uvicorn src.assistant.service:app --reload
```

Or set it persistently in `.env`:

```
VARIANT=v4
```

Variant is bound at startup. To switch, stop the service (`Ctrl+C`) and re-run with a different `VARIANT`. There is no hot-swap — intentional, because Prometheus labels (and so the Grafana dashboards) carry `variant_id` as a dimension and a series should belong to one deployment.

## Running an offline eval

```bash
# Full eval against the 100-example dataset (~10–20 min depending on variant)
python -m src.eval --variant v1

# Quick check while developing
python -m src.eval --variant v4 --limit 25
```

Each invocation is a new MLflow run. Open the MLflow UI and compare runs side-by-side: per-category accuracy, refusal rates, total cost, latency, full per-example predictions as a downloadable artifact.

## UIs

| URL | What it shows |
|-----|---------------|
| http://localhost:5000 | MLflow tracking server — compare eval runs across variants |
| http://localhost:3000 | Grafana — the *Travel Assistant — Live Monitoring* dashboard (anonymous Viewer; admin/admin to edit) |
| http://localhost:9090 | Prometheus — raw metrics + PromQL query UI |
| http://localhost:8000/metrics | Prometheus exposition straight from the assistant service |
| http://localhost:8000/health | Liveness check |

## Iterating on variants

The dev loop:

1. Edit `variants.yaml` — add a new variant block, change a model, swap a guardrail config.
2. Edit prompts in `prompts/` if you're changing system or classifier prompts.
3. Stop the service (`Ctrl+C`) and re-run `uvicorn` with the new `VARIANT`.
4. `python -m src.eval --variant <new>` — new MLflow run.
5. Compare in MLflow UI; iterate.

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
3. **`pydantic-settings` with `SecretStr`** — keys are wrapped in a type that doesn't render in `repr()` or logs (see `src/config.py` once that lands).

In code, never `print(settings)` or include the key in error messages; log only `settings.nebius_api_key.get_secret_value()` at the point of use.

If you ever do commit a key by accident:
1. **Rotate it immediately** in the Nebius console — the old value is permanently in git history and on every fork/clone.
2. Force-push only if the commit hasn't been pulled by anyone else; otherwise treat the key as burned and only rotation matters.
3. GitHub's secret scanning will likely flag it for you anyway, and Nebius may auto-rotate if they participate in the partner program.

For the serverless v2 chapter (`docs/serverless.md`), the production answer is to fetch the key from a managed secrets store at startup rather than carry a `.env` file into the container.

