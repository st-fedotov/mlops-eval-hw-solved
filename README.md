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

## Quick start

```bash
cp .env.example .env
# Fill in NEBIUS_API_KEY in .env

docker compose up -d                                # MLflow, Postgres, MinIO, Prometheus, Grafana

pip install -e .
uvicorn src.assistant.service:app --reload         # the assistant on :8000
python -m src.eval --variant v1                    # offline eval -> MLflow
```

Open:
- MLflow UI: http://localhost:5000
- Grafana:   http://localhost:3000 (anonymous Viewer; admin/admin for edit)
- Prometheus: http://localhost:9090

Full task description: [`docs/README.md`](docs/README.md). Reference solution walkthrough: [`docs/reference_solution.md`](docs/reference_solution.md).

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

