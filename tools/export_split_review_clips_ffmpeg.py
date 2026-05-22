#!/usr/bin/env python3
"""Export H.264 review clips for split LeRobot datasets using ffmpeg.

This avoids OpenCV's fragile AV1 decoding path and writes broadly compatible
H.264/yuv420p mp4 clips for manual review.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pandas as pd


DEFAULT_PARTS = (
    "part1_first_vial_bottom_right",
    "part2_second_vial_top_right",
)

DEFAULT_CAMERAS = (
    "observation.images.head_image",
    "observation.images.left_wrist_image",
    "observation.images.right_wrist_image",
)


def _load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _video_path(part_root: Path, video_key: str, file_index: int) -> Path:
    return part_root / "videos" / video_key / "chunk-000" / f"file-{file_index:03d}.mp4"


def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )


def _export_episode_clip(
    *,
    ffmpeg: str,
    part_root: Path,
    part_name: str,
    episode_row: pd.Series,
    episode_index: int,
    cameras: tuple[str, ...],
    out_path: Path,
    fps: float,
    height: int,
    crf: int,
) -> dict:
    duration = float(episode_row["length"]) / fps
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    input_summaries = []

    for camera in cameras:
        file_idx = int(episode_row[f"videos/{camera}/file_index"])
        start_ts = float(episode_row[f"videos/{camera}/from_timestamp"])
        src = _video_path(part_root, camera, file_idx)
        cmd.extend(["-ss", f"{start_ts:.6f}", "-t", f"{duration:.6f}", "-i", str(src)])
        input_summaries.append({"camera": camera, "file_index": file_idx, "start": start_ts})

    scaled = []
    for idx, _camera in enumerate(cameras):
        # Some static ffmpeg builds do not include drawtext; keep the filter
        # minimal and rely on output filenames for labels.
        vf = f"[{idx}:v]scale=-2:{height},setsar=1[v{idx}]"
        scaled.append(vf)

    hstack_inputs = "".join(f"[v{idx}]" for idx in range(len(cameras)))
    filter_complex = ";".join(scaled + [f"{hstack_inputs}hstack=inputs={len(cameras)}[out]"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
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
    )
    _run_ffmpeg(cmd)

    return {
        "part": part_name,
        "episode_index": episode_index,
        "length": int(episode_row["length"]),
        "duration_s": duration,
        "output": str(out_path),
        "inputs": input_summaries,
    }


def _export_episode_camera_clip(
    *,
    ffmpeg: str,
    part_root: Path,
    part_name: str,
    episode_row: pd.Series,
    episode_index: int,
    camera: str,
    out_path: Path,
    fps: float,
    height: int,
    crf: int,
) -> dict:
    duration = float(episode_row["length"]) / fps
    file_idx = int(episode_row[f"videos/{camera}/file_index"])
    start_ts = float(episode_row[f"videos/{camera}/from_timestamp"])
    src = _video_path(part_root, camera, file_idx)

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
        "part": part_name,
        "episode_index": episode_index,
        "camera": camera,
        "length": int(episode_row["length"]),
        "duration_s": duration,
        "output": str(out_path),
        "input": {"camera": camera, "file_index": file_idx, "start": start_ts},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--layout", choices=("triptych", "separate"), default="separate")
    parser.add_argument("--parts", nargs="*", default=list(DEFAULT_PARTS))
    parser.add_argument("--cameras", nargs="*", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    split_root = args.split_root.expanduser().resolve()
    output_dir = (args.output_dir or split_root / "review" / "clips").expanduser().resolve()
    if args.replace and output_dir.exists():
        shutil.rmtree(output_dir)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cameras = tuple(args.cameras)
    manifest = []

    for part_name in args.parts:
        part_root = split_root / part_name
        info = _load_json(part_root / "meta" / "info.json")
        fps = float(info.get("fps", 30))
        episodes = pd.read_parquet(part_root / "meta/episodes/chunk-000/file-000.parquet")
        selected = episodes.iloc[args.start_episode : args.start_episode + args.num_episodes]

        for _, row in selected.iterrows():
            episode_index = int(row["episode_index"])
            if args.layout == "triptych":
                out_path = output_dir / part_name / f"episode_{episode_index:03d}_{part_name}_triptych.mp4"
                manifest.append(
                    _export_episode_clip(
                        ffmpeg=ffmpeg,
                        part_root=part_root,
                        part_name=part_name,
                        episode_row=row,
                        episode_index=episode_index,
                        cameras=cameras,
                        out_path=out_path,
                        fps=fps,
                        height=args.height,
                        crf=args.crf,
                    )
                )
            else:
                for camera in cameras:
                    camera_name = camera.rsplit(".", 1)[-1]
                    out_path = (
                        output_dir
                        / part_name
                        / camera_name
                        / f"episode_{episode_index:03d}_{part_name}_{camera_name}.mp4"
                    )
                    manifest.append(
                        _export_episode_camera_clip(
                            ffmpeg=ffmpeg,
                            part_root=part_root,
                            part_name=part_name,
                            episode_row=row,
                            episode_index=episode_index,
                            camera=camera,
                            out_path=out_path,
                            fps=fps,
                            height=args.height,
                            crf=args.crf,
                        )
                    )

    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"output_dir": str(output_dir), "clips": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
