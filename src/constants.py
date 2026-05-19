"""Shared constants across assistant, judge, monitoring, and eval."""

from __future__ import annotations

# v2..v5 system prompts instruct the assistant to emit this string verbatim
# when refusing. Cheap refusal-rate monitoring is an exact equality check.
# MUST stay in sync with prompts/v2_explicit_refusal.txt and v3_prompt_hardening.txt.
CANNED_REFUSAL = (
    "I can only help with travel-related questions "
    "(flights, hotels, visas, baggage, travel planning). "
    "I can't help with that."
)

# USD per 1M tokens (input, output). VERIFY against the Nebius pricing page
# before treating absolute numbers as accurate; for the homework, what matters
# is the relative ordering across variants.
MODEL_PRICES_USD_PER_1M: dict[str, tuple[float, float]] = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": (0.02, 0.06),
    "meta-llama/Llama-3.3-70B-Instruct": (0.13, 0.40),
    "Qwen/Qwen3-32B": (0.10, 0.30),
    "Qwen/Qwen3-235B-A22B-Instruct-2507": (0.20, 0.60),
    "google/gemma-3-27b-it": (0.10, 0.30),
    "deepseek-ai/DeepSeek-V3.2": (0.27, 1.10),
    "openai/gpt-oss-120b": (0.20, 0.60),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for one completion. Returns 0.0 if model is not in the table."""
    prices = MODEL_PRICES_USD_PER_1M.get(model)
    if prices is None:
        return 0.0
    in_price, out_price = prices
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
