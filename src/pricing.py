"""Per-model token pricing.

Fetches per-token rates from Nebius Token Factory's verbose models endpoint
(`GET /v1/models?verbose=true`). The response includes a `pricing` object
with `prompt` and `completion` rates expressed as USD per single token; this
module multiplies token counts directly without unit conversion.

The fetch is lazy and cached for the lifetime of the process via lru_cache.
The first call to `cost_usd(...)` triggers one HTTP round-trip; subsequent
calls hit the in-process cache. If the fetch fails (network, bad key, etc.),
the module logs an error and returns 0 for every model — the same degraded
behavior as the previous YAML-backed implementation when prices.yaml was
missing.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import httpx

from src.config import get_settings

log = logging.getLogger(__name__)


# Models we've already warned about, to avoid one log line per request.
_warned_models: set[str] = set()


@lru_cache(maxsize=1)
def _get_prices() -> dict[str, tuple[float, float]]:
    """Fetch per-model (prompt, completion) per-token USD rates from Nebius.

    Returns an empty dict on any fetch or parse failure. Cached per process;
    call `_get_prices.cache_clear()` to force a refresh.
    """
    settings = get_settings()
    url = settings.nebius_base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {settings.nebius_api_key.get_secret_value()}"}
    try:
        resp = httpx.get(
            url,
            params={"verbose": "true"},
            headers=headers,
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.error(
            "Failed to fetch model pricing from Nebius (%s); cost will be 0 "
            "for all models this process.",
            exc,
        )
        return {}

    out: dict[str, tuple[float, float]] = {}
    for m in data.get("data", []):
        pricing = m.get("pricing") or {}
        try:
            prompt = float(pricing.get("prompt"))
            completion = float(pricing.get("completion"))
        except (TypeError, ValueError):
            continue
        out[m["id"]] = (prompt, completion)
    return out


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for one completion. Returns 0.0 (with a one-shot warning) if
    Nebius didn't return pricing for this model."""
    prices = _get_prices()
    entry = prices.get(model)
    if entry is None:
        if model not in _warned_models:
            log.warning(
                "No pricing returned by Nebius for model %r. Cost will be 0 "
                "until the model is added to the Nebius catalog.",
                model,
            )
            _warned_models.add(model)
        return 0.0
    in_price, out_price = entry
    return input_tokens * in_price + output_tokens * out_price
