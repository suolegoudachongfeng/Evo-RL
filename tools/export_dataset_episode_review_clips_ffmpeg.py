#!/usr/bin/env python3
"""Export one H.264 review clip per LeRobot episode and camera.

The output is intended for VLM-based segmentation annotation. It does not
modify the source dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pandas as pd


DEFAULT_CAMERAS = (
    "observation.images.head_image",
    "observation.images.left_wrist_image",
    "observation.images.right_wrist_image",
)


def _load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _video_path(dataset_root: Path, video_key: str, file_index: int) -> Path:
    return dataset_root / "videos" / video_key / "chunk-000" / f"file-{file_index:03d}.mp4"


def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )


def _export_episode_camera_clip(
    *,
    ffmpeg: str,
    dataset_root: Path,
    row: pd.Series,
    episode_index: int,
    camera: str,
    out_path: Path,
    fps: float,
    height: int,
    crf: int,
) -> dict:
    file_idx = int(row[f"videos/{camera}/file_index"])
    start_ts = float(row[f"videos/{camera}/from_timestamp"])
    to_ts = float(row[f"videos/{camera}/to_timestamp"])
    duration = max(0.001, to_ts - start_ts)
    src = _video_path(dataset_root, camera, file_idx)
    if not src.exists():
        raise FileNotFoundError(src)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_ts:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(src),
        "-vf",
        f"scale=-2:{height},setsar=1",
        "-r",
        f"{fps:g}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    _run_ffmpeg(cmd)

    return {
        "episode_index": episode_index,
        "camera": camera,
        "camera_name": camera.rsplit(".", 1)[-1],
        "source_video": str(src),
        "source_file_index": file_idx,
        "from_timestamp": start_ts,
        "to_timestamp": to_ts,
        "duration_s": duration,
        "length_frames": int(row["length"]),
        "episode_success": row.get("episode_success", None),
        "output": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--cameras", nargs="*", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.replace and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = _load_json(dataset_root / "meta" / "info.json")
    fps = float(info.get("fps", 30))
    episodes = pd.read_parquet(dataset_root / "meta/episodes/chunk-000/file-000.parquet")
    end = None if args.num_episodes is None else args.start_episode + args.num_episodes
    selected = episodes.iloc[args.start_episode:end]

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    manifest = []
    cameras = tuple(args.cameras)

    for _, row in selected.iterrows():
        episode_index = int(row["episode_index"])
        for camera in cameras:
            camera_name = camera.rsplit(".", 1)[-1]
            out_path = output_dir / "clips" / camera_name / f"episode_{episode_index:03d}_{camera_name}.mp4"
            manifest.append(
                _export_episode_camera_clip(
                    ffmpeg=ffmpeg,
                    dataset_root=dataset_root,
                    row=row,
                    episode_index=episode_index,
                    camera=camera,
                    out_path=out_path,
                    fps=fps,
                    height=args.height,
                    crf=args.crf,
                )
            )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with (output_dir / "manifest.jsonl").open("w") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "episodes": int(selected["episode_index"].nunique()),
        "clips": len(manifest),
        "cameras": list(cameras),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
