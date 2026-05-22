#!/usr/bin/env python3
"""Export short multiview review clips from split LeRobot datasets.

The split datasets keep the original video shards and use episode metadata to
point at the relevant time range. This tool materializes those ranges as small
mp4 files so the split boundary can be checked visually.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
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


def _open_capture(path: Path, start_timestamp_s: float, dataset_fps: float) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or dataset_fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start_timestamp_s * video_fps)))
    return cap


def _read_frame(cap: cv2.VideoCapture, height: int) -> np.ndarray:
    ok, frame = cap.read()
    if not ok or frame is None:
        return np.zeros((height, int(height * 16 / 9), 3), dtype=np.uint8)
    if frame.shape[0] != height:
        width = int(round(frame.shape[1] * height / frame.shape[0]))
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def _annotate(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _export_episode_clip(
    *,
    part_root: Path,
    part_name: str,
    episode_row: pd.Series,
    episode_index: int,
    cameras: tuple[str, ...],
    out_path: Path,
    fps: float,
    height: int,
) -> dict:
    caps: list[cv2.VideoCapture] = []
    try:
        for camera in cameras:
            file_idx = int(episode_row[f"videos/{camera}/file_index"])
            start_ts = float(episode_row[f"videos/{camera}/from_timestamp"])
            caps.append(_open_capture(_video_path(part_root, camera, file_idx), start_ts, fps))

        length = int(episode_row["length"])
        first_frames = [_read_frame(cap, height) for cap in caps]
        width = sum(frame.shape[1] for frame in first_frames)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open writer: {out_path}")

        for frame_idx in range(length):
            if frame_idx == 0:
                frames = first_frames
            else:
                frames = [_read_frame(cap, height) for cap in caps]

            annotated = []
            for camera, frame in zip(cameras, frames, strict=True):
                label = f"{part_name} | ep {episode_index:03d} | {camera.rsplit('.', 1)[-1]} | f {frame_idx:04d}"
                annotated.append(_annotate(frame, label))
            writer.write(np.concatenate(annotated, axis=1))
        writer.release()
    finally:
        for cap in caps:
            cap.release()

    return {
        "part": part_name,
        "episode_index": episode_index,
        "length": int(episode_row["length"]),
        "output": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--parts", nargs="*", default=list(DEFAULT_PARTS))
    parser.add_argument("--cameras", nargs="*", default=list(DEFAULT_CAMERAS))
    args = parser.parse_args()

    split_root = args.split_root.expanduser().resolve()
    output_dir = (args.output_dir or split_root / "review" / "clips").expanduser().resolve()
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
            out_path = output_dir / part_name / f"episode_{episode_index:03d}_{part_name}_triptych.mp4"
            manifest.append(
                _export_episode_clip(
                    part_root=part_root,
                    part_name=part_name,
                    episode_row=row,
                    episode_index=episode_index,
                    cameras=cameras,
                    out_path=out_path,
                    fps=fps,
                    height=args.height,
                )
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"output_dir": str(output_dir), "clips": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
