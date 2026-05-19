"""Prometheus metric definitions.

All metrics live on the default registry, defined at module-import time.
Importing this module is idempotent; instantiation happens once.

Three signal classes (see docs/reference_solution.md):
- Cheap signals: emitted from /chat on every request.
- State signals (gauges): point-in-time view of the service.
- Sampled signals: emitted by the async judge worker.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- Cheap signals -----------------------------------------------------------

chat_requests_total = Counter(
    "chat_requests_total",
    "Total /chat invocations that returned a response (not counting LLM errors).",
    ["variant_id", "refused", "input_category"],
)

llm_api_errors_total = Counter(
    "llm_api_errors_total",
    "Errors raised by the LLM client during a /chat invocation.",
    ["variant_id", "error_type"],
)

chat_cost_usd_total = Counter(
    "chat_cost_usd_total",
    "Cumulative USD cost of /chat completions, sliced by model.",
    ["variant_id", "model"],
)

chat_request_duration_seconds = Histogram(
    "chat_request_duration_seconds",
    "End-to-end duration of one /chat request, in seconds.",
    ["variant_id"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
)

chat_input_tokens = Histogram(
    "chat_input_tokens",
    "Per-completion input token count.",
    ["variant_id", "model"],
    buckets=(16, 64, 256, 1024, 4096, 16384),
)

chat_output_tokens = Histogram(
    "chat_output_tokens",
    "Per-completion output token count.",
    ["variant_id", "model"],
    buckets=(8, 32, 128, 512, 2048),
)

# --- State signals (gauges) --------------------------------------------------

in_flight_requests = Gauge(
    "in_flight_requests",
    "Number of /chat requests currently being processed.",
)

deep_judge_queue_depth = Gauge(
    "deep_judge_queue_depth",
    "Pending (input, response) pairs waiting for sampled judge evaluation.",
)

judge_sample_rate = Gauge(
    "judge_sample_rate",
    "Configured fraction of /chat traffic forwarded to the deep judge.",
)

assistant_info = Gauge(
    "assistant_info",
    "Info metric carrying deployment identity in labels. Always 1.",
    ["variant_id", "model", "guardrail_type", "mlflow_run_id"],
)

# --- Sampled signals (async worker emits) ------------------------------------

judge_evaluations_total = Counter(
    "judge_evaluations_total",
    "Total deep-judge evaluations completed, sliced by verdict.",
    ["variant_id", "verdict"],
)

judge_latency_seconds = Histogram(
    "judge_latency_seconds",
    "Latency of one deep-judge evaluation (the judge API call).",
    ["variant_id"],
    buckets=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
)
