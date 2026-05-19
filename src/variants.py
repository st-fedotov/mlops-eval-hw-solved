"""Variant configuration — the unit of deployment.

A variant fully describes one assistant: main model + sampling, system prompt,
and (optional) guardrail architecture. Loaded from variants.yaml; the parsed
Variant object is the source of truth for both the service and MLflow logging.

Run from the repo root so the relative prompt paths resolve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    temperature: float = 0.0
    max_tokens: int = 512


class ClassifierSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: ModelSpec
    prompt: Path


class GuardrailNone(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["none"]


class GuardrailInputClassifier(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["input_classifier"]
    classifier: ClassifierSpec


class GuardrailSandwich(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["sandwich"]
    input_classifier: ClassifierSpec
    output_validator: ClassifierSpec


Guardrail = Annotated[
    Union[GuardrailNone, GuardrailInputClassifier, GuardrailSandwich],
    Field(discriminator="type"),
]


class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str
    description: str
    model: ModelSpec
    system_prompt: Path
    guardrail: Guardrail


def load_variants(path: Path | str = "variants.yaml") -> dict[str, Variant]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {key: Variant.model_validate(v) for key, v in data.items()}


def load_variant(variant_id: str, path: Path | str = "variants.yaml") -> Variant:
    variants = load_variants(path)
    if variant_id not in variants:
        raise KeyError(
            f"Variant {variant_id!r} not found in {path}. "
            f"Available: {sorted(variants)}"
        )
    return variants[variant_id]
