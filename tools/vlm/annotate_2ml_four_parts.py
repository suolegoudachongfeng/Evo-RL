#!/usr/bin/env python3
"""Batch annotate 2 mL right-wrist videos into four manipulation stages.

The script calls a Gemini-compatible Neolink endpoint and writes one JSON file
per episode plus a JSONL manifest. It does not modify any dataset.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_API_BASE = "https://neolink.com/api/v1beta"
DEFAULT_PROMPT = Path(__file__).resolve().parent / "prompts" / "segment_2ml_four_parts_prompt_zh.txt"


def _extract_episode_index(path: Path) -> int | None:
    match = re.search(r"episode_(\d+)", path.name)
    return None if match is None else int(match.group(1))


def _load_prompt(path: Path) -> str:
    return path.expanduser().read_text()


def _extract_text(response: dict[str, Any]) -> str:
    parts = response["candidates"][0]["content"]["parts"]
    texts = [part["text"] for part in parts if "text" in part]
    if not texts:
        raise ValueError("No text part found in model response")
    return "\n".join(texts)


def _parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Model response does not contain a JSON object: {text[:200]!r}")
    return json.loads(cleaned[start : end + 1])


def _request_annotation(
    *,
    video_path: Path,
    prompt: str,
    api_key: str,
    api_base: str,
    model: str,
    timeout_s: int,
) -> dict[str, Any]:
    video_b64 = base64.b64encode(video_path.read_bytes()).decode("utf-8")
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
                    {"text": prompt},
                ]
            }
        ]
    }
    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/models/{model}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    text = _extract_text(raw)
    annotation = _parse_json_text(text)
    return {
        "annotation": annotation,
        "raw_text": text,
        "usage": raw.get("usageMetadata") or raw.get("usage"),
        "model": raw.get("modelVersion") or model,
    }


def _iter_videos(clips_dir: Path) -> list[Path]:
    return sorted(p for p in clips_dir.glob("episode_*_right_wrist_image.mp4") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key-env", default="NEOLINK_API_KEY")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key and args.api_key_file is not None:
        api_key = args.api_key_file.expanduser().read_text().strip()
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}; alternatively pass --api-key-file")

    clips_dir = args.clips_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    per_episode_dir = output_dir / "per_episode"
    per_episode_dir.mkdir(parents=True, exist_ok=True)

    prompt = _load_prompt(args.prompt_file)
    videos = []
    for path in _iter_videos(clips_dir):
        episode_index = _extract_episode_index(path)
        if episode_index is None or episode_index < args.start_episode:
            continue
        videos.append(path)
    if args.limit is not None:
        videos = videos[: args.limit]

    manifest_path = output_dir / "annotations.jsonl"
    summary = {
        "clips_dir": str(clips_dir),
        "output_dir": str(output_dir),
        "prompt_file": str(args.prompt_file.expanduser().resolve()),
        "model": args.model,
        "total_requested": len(videos),
        "completed": 0,
        "failed": 0,
        "failures": [],
    }

    with manifest_path.open("a" if args.resume else "w") as manifest:
        for video_path in videos:
            episode_index = _extract_episode_index(video_path)
            out_path = per_episode_dir / f"episode_{episode_index:03d}_segments.json"
            if args.resume and out_path.exists():
                continue

            record: dict[str, Any] = {
                "episode_index": episode_index,
                "video_path": str(video_path),
                "output_path": str(out_path),
            }
            try:
                result = _request_annotation(
                    video_path=video_path,
                    prompt=prompt,
                    api_key=api_key,
                    api_base=args.api_base,
                    model=args.model,
                    timeout_s=args.timeout_s,
                )
                record.update(
                    {
                        "annotation": result["annotation"],
                        "model": result["model"],
                        "usage": result["usage"],
                        "status": "ok",
                    }
                )
                if args.keep_raw:
                    record["raw_text"] = result["raw_text"]
                out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
                summary["completed"] += 1
            except (urllib.error.HTTPError, urllib.error.URLError, ValueError, KeyError, json.JSONDecodeError) as exc:
                record.update({"status": "failed", "error": repr(exc)})
                summary["failed"] += 1
                summary["failures"].append(record)
                out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))

            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
