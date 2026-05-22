#!/usr/bin/env python3
"""Ask a Neolink Gemini video model about one local video file."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request


def _extract_text(result: dict) -> str:
    try:
        parts = result["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return json.dumps(result, ensure_ascii=False, indent=2)

    texts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
    return "\n".join(texts) if texts else json.dumps(result, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to a local mp4 video.")
    parser.add_argument("--prompt", required=True, help="Question/instruction for the video model.")
    parser.add_argument("--model", default="gemini-3.5-flash", help="Neolink model id.")
    parser.add_argument("--base-url", default="https://neolink.com/api", help="Neolink API base URL.")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--raw", action="store_true", help="Print the full JSON response instead of extracted text.")
    args = parser.parse_args()

    api_key = os.environ.get("NEOLINK_API_KEY")
    if not api_key:
        raise RuntimeError("NEOLINK_API_KEY is not set in this shell.")

    with open(args.video, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "video/mp4",
                            "data": video_b64,
                        }
                    },
                    {"text": args.prompt},
                ]
            }
        ]
    }

    url = f"{args.base_url.rstrip('/')}/v1beta/models/{args.model}:generateContent"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}", file=sys.stderr)
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1

    if args.raw:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(_extract_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
