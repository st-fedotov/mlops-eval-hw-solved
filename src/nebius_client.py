"""Nebius Token Factory client (OpenAI-compatible).

The unwrapped API key only appears at the OpenAI client boundary; elsewhere
it lives inside a SecretStr.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from src.config import get_settings


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=settings.nebius_api_key.get_secret_value(),
        base_url=settings.nebius_base_url,
    )
