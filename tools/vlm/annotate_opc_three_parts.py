#!/usr/bin/env python3
"""Batch annotate OPC drawer/rack videos into three manipulation stages."""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_API_BASE = "https://neolink.com/api/v1beta"
DEFAULT_PROMPT = Path(__file__).resolve().parent / "prompts" / "segment_opc_three_parts_prompt_zh.txt"


def _extract_episode_index(path: Path) -> int | None:
    match = re.search(r"episode_(\d+)", path.name)
    return None if match is None else int(match.group(1))


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
    payload = cleaned[start : end + 1]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        repaired = re.sub(r"\\u(?![0-9a-fA-F]{4})", r"\\\\u", payload)
        if repaired == payload:
            raise
        return json.loads(repaired)


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
                    {"inline_data": {"mime_type": "video/mp4", "data": video_b64}},
                    {"text": prompt},
                ]
            }
        ]
    }
    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/models/{model}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    text = _extract_text(raw)
    return {
        "annotation": _parse_json_text(text),
        "raw_text": text,
        "usage": raw.get("usageMetadata") or raw.get("usage"),
        "model": raw.get("modelVersion") or model,
    }


def _resolve_api_key(api_key_env: str, api_key_file: Path | None, no_prompt: bool) -> str:
    api_key = os.environ.get(api_key_env)
    if api_key:
        return api_key.strip()
    if api_key_file is not None:
        return api_key_file.expanduser().read_text().strip()
    if not no_prompt and sys.stdin.isatty():
        return getpass.getpass("Neolink API key (input hidden): ").strip()
    raise RuntimeError(
        f"Missing API key env var: {api_key_env}; alternatively pass --api-key-file "
        "or run from an interactive terminal to enter it securely."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--video-glob", default="episode_*_head_image.mp4")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key-env", default="NEOLINK_API_KEY")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--no-api-key-prompt", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    api_key = _resolve_api_key(args.api_key_env, args.api_key_file, args.no_api_key_prompt)
    clips_dir = args.clips_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    per_episode_dir = output_dir / "per_episode"
    per_episode_dir.mkdir(parents=True, exist_ok=True)

    prompt = args.prompt_file.expanduser().read_text()
    videos = []
    for path in sorted(clips_dir.glob(args.video_glob)):
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
        "video_glob": args.video_glob,
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
