"""Probe the Nebius Token Factory model catalog.

Reads the API key from either the NEBIUS_API_KEY env var or a local
`nebius_api_key` file (in that order). Prints model IDs to stdout; the
key value is never logged.

Usage:
    python scripts/list_models.py
"""

from __future__ import annotations

import os
import pathlib
import sys


def load_key() -> str:
    env_key = os.environ.get("NEBIUS_API_KEY")
    if env_key:
        return env_key.strip()
    candidate = pathlib.Path("mlops-hw-tf-api-key")
    if not candidate.exists():
        print(
            "No API key found. Set NEBIUS_API_KEY env var or create a `nebius_api_key` file.",
            file=sys.stderr,
        )
        sys.exit(1)
    raw = candidate.read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    text = candidate.read_text(encoding="utf-8-sig")
    key = text.strip().strip("'\"")
    # Safe diagnostics: never prints any character of the key.
    print(
        f"diag: file_bytes={len(raw)} has_bom={has_bom} "
        f"key_len={len(key)} all_ascii={key.isascii()} "
        f"has_inner_ws={any(c.isspace() for c in key)}",
        file=sys.stderr,
    )
    return key


def main() -> None:
    key = load_key()
    base_url = os.environ.get(
        "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"
    )

    try:
        from openai import OpenAI
    except ImportError:
        print("Install the openai package: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=key, base_url=base_url)
    try:
        page = client.models.list()
    except Exception as exc:
        # openai-python masks the api key in exception strings, but stay defensive.
        print(f"models.list failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    for m in page.data:
        print(m.id)


if __name__ == "__main__":
    main()
