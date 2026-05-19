"""Application settings loaded from .env / environment.

The Nebius API key is wrapped in SecretStr — it does not appear in repr,
str, or default JSON. Unwrap only at the OpenAI client boundary via
.get_secret_value().
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nebius Token Factory
    nebius_api_key: SecretStr = Field(...)
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1/"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "travel-assistant"

    # Assistant service
    variant: str = "v1"
    variants_file: Path = Path("variants.yaml")
    # If set, the service loads its variant from this MLflow run's `variant.json`
    # artifact instead of from variants_file. Used in deployed environments;
    # leave unset for local-dev mode.
    mlflow_run_id: str | None = None
    assistant_host: str = "0.0.0.0"
    assistant_port: int = 8000

    # Judge — orthogonal to assistant variants
    judge_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    judge_temperature: float = 0.0
    judge_sample_rate: float = Field(0.05, ge=0.0, le=1.0)
    judge_queue_maxsize: int = 1000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
