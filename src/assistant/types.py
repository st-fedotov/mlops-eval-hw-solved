"""Response types returned by the assistant pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelCall:
    """One call to a Nebius model. Used downstream for cost/latency tracking."""

    model: str
    role: str  # "main_assistant" | "input_classifier" | "output_validator"
    input_tokens: int
    output_tokens: int
    latency_seconds: float


@dataclass
class AssistantResponse:
    """The full result of one /chat invocation."""

    text: str
    refused: bool
    input_category: str | None       # set if an input classifier ran
    output_verdict: str | None       # set if an output validator ran
    model_calls: list[ModelCall] = field(default_factory=list)
