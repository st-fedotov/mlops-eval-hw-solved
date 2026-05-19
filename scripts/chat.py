"""Send a message to the running assistant service and print the response.

Uses urllib so it has no runtime dependencies beyond the standard library.

Usage:
    python scripts/chat.py "Find flights from Paris to Rome"
    python scripts/chat.py --raw "Tell me a joke about programming"
    python scripts/chat.py --url http://localhost:8000 "What is Lufthansa's baggage policy?"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", help="The message to send to /chat")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Assistant service base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the full JSON response instead of just the text",
    )
    args = parser.parse_args()

    payload = json.dumps({"message": args.message}).encode("utf-8")
    req = urllib.request.Request(
        f"{args.url}/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.raw:
        print(json.dumps(data, indent=2))
    else:
        print(data["text"])
        if data.get("refused"):
            print("(refused)", file=sys.stderr)


if __name__ == "__main__":
    main()
