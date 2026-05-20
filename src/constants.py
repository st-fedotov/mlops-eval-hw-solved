"""Constants shared across the codebase."""

from __future__ import annotations

# v2..v5 system prompts instruct the assistant to emit this string verbatim
# when refusing. Cheap refusal-rate monitoring is an exact equality check.
# MUST stay in sync with prompts/v2_explicit_refusal.txt and v3_prompt_hardening.txt.
CANNED_REFUSAL = (
    "I can only help with travel-related questions "
    "(flights, hotels, visas, baggage, travel planning). "
    "I can't help with that."
)
