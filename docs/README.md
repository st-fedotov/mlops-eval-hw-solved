# Travel-assistant MLOps homework

You're building, evaluating, and operating a *bounded-scope* assistant that should only answer travel questions and refuse everything else. The goal is to learn the full MLOps loop for an LLM product: prompt iteration, eval against an adversarial dataset, real-time monitoring with Prometheus/Grafana, and reproducible runs in MLflow with Registry-rooted deployments.

## What's given

- A FastAPI service (`src/assistant/service.py`) that loads one **config** at startup and exposes `/chat`, `/metrics`, `/health`.
- Five reference configs in `configs/` (one YAML file each):
  - `v1` — minimal baseline.
  - `v2` — positive-list scope + canned refusal string.
  - `v3` — prompt-hardening with explicit anti-jailbreak rules (kept as a *foil* — don't ship this in prod).
  - `v4` — input classifier guardrail.
  - `v5` — sandwich: input classifier + output validator.
- An eval CLI (`python -m src.eval`) that runs the whole dataset through the configured assistant and logs everything to MLflow. On full evals it auto-registers a new version of `travel-assistant` in the MLflow Model Registry.
- An adversarial dataset of 100 examples (`data/eval_dataset.jsonl`), balanced across four categories: normal travel, off-topic, jailbreak, social-engineering.
- An MLflow tracking server (with Model Registry), Prometheus, and Grafana — all defined in `docker-compose.yml`.

## What you do

1. `cp .env.example .env`; put your Nebius API key in `.env`.
2. `docker compose pull` then `docker compose up -d` to start MLflow + Postgres + MinIO + Prometheus + Grafana.
3. `pip install -e .` to install the project.
4. `uvicorn src.assistant.service:app --reload` to start the assistant.
5. `python -m src.eval --config v1` to run the baseline eval. Open MLflow UI at http://localhost:5000 and look at per-category accuracy and cost. On a full eval, the run is auto-registered under Models → `travel-assistant`.
6. **Iterate.** Add new files to `configs/` (different system prompts, different main models, different guardrail configurations). For each new config, run the eval. Each invocation is a new MLflow run; each *full* eval is a new registered version.
7. **Promote.** When a version's metrics clear your bar, assign the `Production` alias to it. The eval prints `registered: travel-assistant v<NUMBER>` — use that integer. From the MLflow UI: Models → travel-assistant → click the version → *+ Add alias* → `Production`. Or a one-liner (replace `7` with your version): `python -c "from mlflow.tracking import MlflowClient; MlflowClient().set_registered_model_alias('travel-assistant', 'Production', 7)"`. Set `ASSISTANT_MODEL_ALIAS=Production` in your `.env` and restart the service.
8. **Watch live monitoring.** With the service running, generate traffic via `curl` or `scripts/chat.py`; open Grafana at http://localhost:3000 and look at the *Travel Assistant — Live Monitoring* dashboard. The key panel is **DIVERGENCE: cheap refusal-rate vs judge leakage-rate** — when they disagree, you've shipped a regression.

## Metrics you should care about

From MLflow runs:

| metric | what it tells you |
|---|---|
| `accuracy_overall` | overall correctness vs expected behavior |
| `accuracy_travel` | the bot answered legitimate travel questions |
| `accuracy_off_topic`, `accuracy_jailbreak`, `accuracy_social_engineering` | the bot refused things it should refuse |
| `verdict_rate_leaked` | fraction of exchanges where the judge says you leaked |
| `verdict_rate_over_refused` | fraction where you refused a legitimate question |
| `total_cost_usd` | how much one eval run costs |
| `avg_latency_seconds` | how slow your config is |
| `avg_calls_per_request` | how many model calls per response (higher with guardrails) |

The trade-off you're looking for: pushing leakage down without crushing `accuracy_travel` or blowing up cost.

## What to submit

- The contents of `configs/` corresponding to your best config (the file itself plus the registered MLflow version number — your `Production` alias should point at it).
- MLflow run IDs for at least three of your iterations (baseline + improvements), with a short note (~3 sentences) each explaining what you changed and why.
- A screenshot of the live-monitoring Grafana dashboard taken while your best config is running and being queried, plus a screenshot of the `travel-assistant` model in MLflow's Models tab showing the version history.

## Hints

- Prompt-only fixes plateau. After a couple of iterations, structural changes (input classifier, output validator, ensembles) generally beat longer system prompts. Try them.
- The judge has a per-call cost. During development, use `--limit 25` (partial evals are *not* auto-registered, so they don't pollute the Registry); run the full 100 only before promoting.
- The `judge_error` rate should always be near 0. If it spikes, the judge model isn't following the structured-output schema — check that the model supports `json_schema` on Nebius.
- Configs are append-only by convention. To iterate v4, copy `configs/v4.yaml` to `configs/v4_my_change.yaml` and edit *that*. The filename stem is the `config_id`, and editing in place silently breaks the link between any prior MLflow run with that id and what's now on disk.
