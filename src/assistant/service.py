"""FastAPI service for the travel assistant.

Endpoints:
- POST /chat     — main path; returns the response and updates Prometheus.
- GET  /metrics  — Prometheus exposition (mounted ASGI app from prometheus_client).
- GET  /health   — liveness check.

Lifecycle: on startup, build the pipeline for the configured variant and
spawn the async judge worker. On shutdown, cancel the worker.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from prometheus_client import make_asgi_app
from pydantic import BaseModel, ConfigDict, Field

from src.assistant import AssistantResponse, build_pipeline
from src.config import get_settings
from src.constants import cost_usd
from src.monitoring.judge_worker import JudgeWorker
from src.monitoring.metrics import (
    assistant_info,
    chat_cost_usd_total,
    chat_input_tokens,
    chat_output_tokens,
    chat_request_duration_seconds,
    chat_requests_total,
    deep_judge_queue_depth,
    in_flight_requests,
    judge_sample_rate,
    llm_api_errors_total,
)
from src.variants import load_variant, load_variant_from_mlflow

log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)


class ModelCallSchema(BaseModel):
    # `model` clashes with pydantic's protected namespace; opt out for this schema.
    model_config = ConfigDict(protected_namespaces=())

    model: str
    role: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


class ChatResponse(BaseModel):
    text: str
    refused: bool
    input_category: str | None
    output_verdict: str | None
    model_calls: list[ModelCallSchema]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    # Two paths for resolving the deployment config:
    # - production: settings.mlflow_run_id is set; fetch the manifest from MLflow.
    # - dev: load from local variants.yaml.
    if settings.mlflow_run_id:
        variant = load_variant_from_mlflow(settings.mlflow_run_id)
        variant_id = variant.variant_id or settings.mlflow_run_id
        log.info(
            "Loaded variant from MLflow run_id=%s (variant_id=%s)",
            settings.mlflow_run_id,
            variant_id,
        )
    else:
        variant = load_variant(settings.variant, settings.variants_file)
        variant_id = settings.variant
        log.info(
            "Loaded variant %s from %s (dev mode)",
            settings.variant,
            settings.variants_file,
        )

    pipeline = build_pipeline(variant)

    assistant_info.labels(
        variant_id=variant_id,
        model=variant.model.name,
        guardrail_type=variant.guardrail.type,
        mlflow_run_id=settings.mlflow_run_id or "local",
    ).set(1)
    judge_sample_rate.set(settings.judge_sample_rate)

    queue: asyncio.Queue = asyncio.Queue(maxsize=settings.judge_queue_maxsize)
    deep_judge_queue_depth.set(0)

    worker = JudgeWorker(queue)
    worker_task = asyncio.create_task(worker.run())

    app.state.pipeline = pipeline
    app.state.variant_id = variant_id
    app.state.judge_queue = queue
    app.state.judge_sample_rate = settings.judge_sample_rate

    log.info(
        "startup: variant_id=%s judge_sample_rate=%.3f model=%s guardrail=%s",
        variant_id,
        settings.judge_sample_rate,
        variant.model.name,
        variant.guardrail.type,
    )
    try:
        yield
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan, title="Travel assistant")
app.mount("/metrics", make_asgi_app())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    variant_id = app.state.variant_id
    pipeline = app.state.pipeline
    judge_queue: asyncio.Queue = app.state.judge_queue
    sample_rate: float = app.state.judge_sample_rate

    in_flight_requests.inc()
    start = time.perf_counter()
    try:
        try:
            response: AssistantResponse = await asyncio.to_thread(
                pipeline.respond, req.message
            )
        except Exception as exc:  # noqa: BLE001
            llm_api_errors_total.labels(
                variant_id=variant_id, error_type=type(exc).__name__
            ).inc()
            raise

        for call in response.model_calls:
            chat_input_tokens.labels(
                variant_id=variant_id, model=call.model
            ).observe(call.input_tokens)
            chat_output_tokens.labels(
                variant_id=variant_id, model=call.model
            ).observe(call.output_tokens)
            chat_cost_usd_total.labels(
                variant_id=variant_id, model=call.model
            ).inc(cost_usd(call.model, call.input_tokens, call.output_tokens))

        chat_requests_total.labels(
            variant_id=variant_id,
            refused=str(response.refused).lower(),
            input_category=response.input_category or "unmonitored",
        ).inc()

        # Sample into the deep-judge queue (non-blocking).
        if random.random() < sample_rate:
            try:
                judge_queue.put_nowait((variant_id, req.message, response.text))
                deep_judge_queue_depth.set(judge_queue.qsize())
            except asyncio.QueueFull:
                # Drop on overflow. Add a dedicated counter if drops become
                # operationally meaningful.
                pass

        return ChatResponse(
            text=response.text,
            refused=response.refused,
            input_category=response.input_category,
            output_verdict=response.output_verdict,
            model_calls=[
                ModelCallSchema(
                    model=c.model,
                    role=c.role,
                    input_tokens=c.input_tokens,
                    output_tokens=c.output_tokens,
                    latency_seconds=c.latency_seconds,
                )
                for c in response.model_calls
            ],
        )
    finally:
        chat_request_duration_seconds.labels(variant_id=variant_id).observe(
            time.perf_counter() - start
        )
        in_flight_requests.dec()
