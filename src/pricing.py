"""Per-model token pricing.

Loads the price table from `prices.yaml` at the repo root and exposes
`cost_usd(model, input_tokens, output_tokens)` for computing per-completion
cost.

Nebius Token Factory does not expose per-token rates via a programmatic API
(no public pricing endpoint; operator-facing rates UI is auth-gated), so the
table is hand-maintained. See `prices.yaml` for verification procedure and
last-verified date.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


_PRICES_FILE = Path(__file__).resolve().parent.parent / "prices.yaml"


def _load_prices() -> dict[str, tuple[float, float]]:
    """Load per-model USD prices from prices.yaml. Returns a mapping from
    model id to (input_per_1m_usd, output_per_1m_usd). Empty dict if the
    file is missing, unreadable, or malformed — never raises at import."""
    if not _PRICES_FILE.exists():
        log.warning(
            "prices.yaml not found at %s; cost will be 0 for all models.",
            _PRICES_FILE,
        )
        return {}
    try:
        with open(_PRICES_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        log.error(
            "Failed to parse prices.yaml: %s. Cost will be 0 for all models.",
            exc,
        )
        return {}
    out: dict[str, tuple[float, float]] = {}
    for row in data.get("prices") or []:
        try:
            out[row["model"]] = (
                float(row["input_per_1m_usd"]),
                float(row["output_per_1m_usd"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.error(
                "Skipping malformed entry in prices.yaml: %r (%s)", row, exc
            )
    return out


MODEL_PRICES_USD_PER_1M: dict[str, tuple[float, float]] = _load_prices()


# Models we've already warned about, to avoid one log line per request.
_warned_models: set[str] = set()


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for one completion. Returns 0.0 (with a one-shot warning) if
    the model is not priced in prices.yaml — add a row to fix."""
    prices = MODEL_PRICES_USD_PER_1M.get(model)
    if prices is None:
        if model not in _warned_models:
            log.warning(
                "No price entry for model %r in prices.yaml. "
                "Cost will be 0 for this model; add a row to fix.",
                model,
            )
            _warned_models.add(model)
        return 0.0
    in_price, out_price = prices
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
