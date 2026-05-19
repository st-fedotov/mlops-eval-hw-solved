"""Variant configuration — the unit of deployment.

A Variant fully describes one assistant: main model, system prompt (inline
content, not a file path), and (optional) guardrail architecture.

Loaded from either:
- `variants.yaml` (dev): file paths in YAML are resolved to content at load time.
- An MLflow run artifact (production): the manifest is already self-contained.

The runtime Variant always carries prompts as strings, never as paths.
"""

from __future__ import annotations

import json
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
    prompt: str  # inline prompt content (NOT a file path)


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
    variant_id: str | None = None       # populated by loaders
    display_name: str
    description: str
    model: ModelSpec
    system_prompt: str                  # inline content
    guardrail: Guardrail


def _read_prompt(path_str: str, base: Path) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = base / p
    return p.read_text(encoding="utf-8")


def _inline_prompts(raw: dict, base: Path) -> dict:
    """Replace prompt file-path references in a raw variant dict with content."""
    raw = dict(raw)
    if "system_prompt" in raw:
        raw["system_prompt"] = _read_prompt(raw["system_prompt"], base)
    g = dict(raw.get("guardrail", {}) or {})
    if g.get("type") == "input_classifier":
        cls = dict(g["classifier"])
        cls["prompt"] = _read_prompt(cls["prompt"], base)
        g["classifier"] = cls
    elif g.get("type") == "sandwich":
        ic = dict(g["input_classifier"])
        ic["prompt"] = _read_prompt(ic["prompt"], base)
        g["input_classifier"] = ic
        ov = dict(g["output_validator"])
        ov["prompt"] = _read_prompt(ov["prompt"], base)
        g["output_validator"] = ov
    raw["guardrail"] = g
    return raw


def load_variants(path: Path | str = "variants.yaml") -> dict[str, Variant]:
    """Load all variants from YAML. Resolves prompt file paths to inline content."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    base = path.parent.resolve()
    result: dict[str, Variant] = {}
    for vid, raw in data.items():
        inlined = _inline_prompts(raw, base)
        inlined.setdefault("variant_id", vid)
        result[vid] = Variant.model_validate(inlined)
    return result


def load_variant(
    variant_id: str, path: Path | str = "variants.yaml"
) -> Variant:
    variants = load_variants(path)
    if variant_id not in variants:
        raise KeyError(
            f"Variant {variant_id!r} not found in {path}. "
            f"Available: {sorted(variants)}"
        )
    return variants[variant_id]


def load_variant_from_mlflow(run_id: str) -> Variant:
    """Fetch the deployment manifest artifact from a specific MLflow run.

    Direct-by-run-id loader, used as a debug utility. Production loads should
    go through the Model Registry via `load_variant_from_registry`.
    """
    import mlflow  # local import: mlflow is heavy and dev-mode service shouldn't pay for it

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="variant.json"
    )
    with open(local_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Variant.model_validate(data)


def load_variant_from_registry(name: str, alias: str) -> tuple[Variant, int]:
    """Resolve a Model Registry alias to a Variant.

    Looks up the version that `name`@`alias` currently points at, downloads
    its `variant.json` artifact (the self-contained deployment manifest), and
    parses it. Returns the Variant and the resolved version number.

    Raises if the alias is not set on any version of the registered model.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    mv = client.get_model_version_by_alias(name=name, alias=alias)
    local_path = mlflow.artifacts.download_artifacts(mv.source)
    with open(local_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Variant.model_validate(data), int(mv.version)
