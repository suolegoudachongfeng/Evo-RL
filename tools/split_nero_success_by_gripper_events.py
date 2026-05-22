#!/usr/bin/env python3
"""Split successful NERO two-vial episodes into first-vial and second-vial segments.

This is intentionally conservative: it produces review-ready LeRobot-style
datasets plus a detailed CSV/JSON report, but it does not attempt to validate
task success visually. The split point is the first stable right-gripper
release after the first stable close event.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PART1_TASK = (
    "2mL_first_vial_bottom_right. Object: one small 2 mL vial. Goal: pick up "
    "the first small vial and place it upright into the bottom-right corner "
    "hole of the empty rack. Constraint: use the right column only; do not "
    "place the vial into the left column, center holes, or any other hole."
)

PART2_TASK = (
    "2mL_second_vial_top_right. Object: one small 2 mL vial. Goal: pick up "
    "the second small vial and place it upright into the top-right corner "
    "hole of the empty rack. Constraint: use the right column only; do not "
    "place the vial into the left column, center holes, or any other hole."
)


@dataclass
class SegmentSpec:
    original_episode_index: int
    new_episode_index: int
    start_frame: int
    end_frame: int
    split_frame: int
    first_close_frame: int
    first_release_frame: int
    second_close_frame: int | None
    second_release_frame: int | None
    status: str

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=2)


def _normalize_success(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode()
    text = str(value).strip().lower()
    if text in {"success", "true", "1", "s"}:
        return "success"
    if text in {"failure", "false", "0", "f"}:
        return "failure"
    return text


def _majority(values: np.ndarray) -> bool:
    return bool(np.mean(values.astype(np.float32)) >= 0.5)


def _debounce_binary(signal: np.ndarray, min_run_frames: int) -> np.ndarray:
    """Remove short binary runs that are usually gripper jitter."""
    if signal.size == 0 or min_run_frames <= 1:
        return signal.astype(bool)

    clean = signal.astype(bool).copy()
    changed = True
    while changed:
        changed = False
        starts = np.r_[0, np.flatnonzero(clean[1:] != clean[:-1]) + 1]
        ends = np.r_[starts[1:], clean.size]
        for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
            run_len = end - start
            if run_len >= min_run_frames:
                continue
            prev_value = clean[starts[i - 1]] if i > 0 else None
            next_value = clean[ends[i]] if i + 1 < len(starts) else None
            if prev_value is not None and next_value is not None and prev_value == next_value:
                clean[start:end] = prev_value
                changed = True
            elif prev_value is not None:
                clean[start:end] = prev_value
                changed = True
            elif next_value is not None:
                clean[start:end] = next_value
                changed = True
    return clean


def _find_split(
    gripper_values: np.ndarray,
    threshold: float,
    min_run_frames: int,
    post_release_buffer_frames: int,
) -> tuple[int | None, dict[str, Any]]:
    """Return split frame after the first release, plus diagnostic metadata.

    In the current NERO datasets, right gripper values above threshold correspond
    to the open state; below threshold corresponds to closed.
    """
    open_state = _debounce_binary(gripper_values > threshold, min_run_frames)
    transition_frames = np.flatnonzero(open_state[1:] != open_state[:-1]) + 1
    close_frames: list[int] = []
    release_frames: list[int] = []
    for frame in transition_frames.tolist():
        before = bool(open_state[frame - 1])
        after = bool(open_state[frame])
        if before and not after:
            close_frames.append(frame)
        elif not before and after:
            release_frames.append(frame)

    first_close = close_frames[0] if close_frames else None
    first_release = None
    if first_close is not None:
        first_release = next((f for f in release_frames if f > first_close), None)

    split_frame = None
    if first_release is not None:
        split_frame = min(len(gripper_values), first_release + post_release_buffer_frames)

    return split_frame, {
        "raw_transition_frames": transition_frames.tolist(),
        "close_frames": close_frames,
        "release_frames": release_frames,
        "first_close_frame": first_close,
        "first_release_frame": first_release,
        "second_close_frame": next((f for f in close_frames if first_release is not None and f > first_release), None),
        "second_release_frame": next((f for f in release_frames if first_release is not None and f > first_release), None),
    }


def _copy_or_link_videos(src_root: Path, dst_root: Path, mode: str) -> None:
    src_videos = src_root / "videos"
    dst_videos = dst_root / "videos"
    if dst_videos.exists() or dst_videos.is_symlink():
        if dst_videos.is_symlink() or dst_videos.is_file():
            dst_videos.unlink()
        else:
            shutil.rmtree(dst_videos)
    dst_videos.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst_videos.symlink_to(src_videos, target_is_directory=True)
    elif mode == "copy":
        shutil.copytree(src_videos, dst_videos)
    else:
        raise ValueError(f"Unsupported video mode: {mode}")


def _task_table(task: str) -> pd.DataFrame:
    return pd.DataFrame({"task_index": [0]}, index=pd.Index([task], name=None))


def _build_segment_dataset(
    *,
    src_root: Path,
    dst_root: Path,
    src_data: pd.DataFrame,
    src_episodes: pd.DataFrame,
    src_info: dict[str, Any],
    specs: list[SegmentSpec],
    task: str,
    video_mode: str,
    fps: int,
) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    (dst_root / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (dst_root / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)

    _copy_or_link_videos(src_root, dst_root, video_mode)

    data_parts: list[pd.DataFrame] = []
    episode_rows: list[dict[str, Any]] = []
    global_index = 0
    video_cols = [c for c in src_episodes.columns if c.startswith("videos/")]

    for spec in specs:
        src_ep_row = src_episodes.loc[src_episodes["episode_index"] == spec.original_episode_index].iloc[0]
        ep_data = src_data.loc[src_data["episode_index"] == spec.original_episode_index].iloc[
            spec.start_frame : spec.end_frame
        ].copy()
        ep_len = len(ep_data)
        if ep_len <= 0:
            continue

        new_frame_index = np.arange(ep_len, dtype=np.int64)
        ep_data["episode_index"] = spec.new_episode_index
        ep_data["frame_index"] = new_frame_index
        ep_data["timestamp"] = new_frame_index.astype(np.float32) / float(fps)
        ep_data["index"] = np.arange(global_index, global_index + ep_len, dtype=np.int64)
        ep_data["task_index"] = 0

        data_parts.append(ep_data)

        episode_meta: dict[str, Any] = {
            "episode_index": spec.new_episode_index,
            "tasks": np.array([task], dtype=object),
            "length": ep_len,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": global_index,
            "dataset_to_index": global_index + ep_len,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "episode_success": "success",
            "original_episode_index": spec.original_episode_index,
            "original_start_frame": spec.start_frame,
            "original_end_frame": spec.end_frame,
            "split_frame": spec.split_frame,
        }
        for col in video_cols:
            episode_meta[col] = src_ep_row[col]
        for col in video_cols:
            if col.endswith("/from_timestamp"):
                episode_meta[col] = float(src_ep_row[col]) + spec.start_frame / float(fps)
            elif col.endswith("/to_timestamp"):
                from_col = col[: -len("/to_timestamp")] + "/from_timestamp"
                episode_meta[col] = float(src_ep_row[from_col]) + spec.end_frame / float(fps)

        episode_rows.append(episode_meta)
        global_index += ep_len

    if not data_parts:
        raise RuntimeError(f"No segment data generated for {dst_root}")

    data_df = pd.concat(data_parts, ignore_index=True)
    data_df.to_parquet(dst_root / "data/chunk-000/file-000.parquet", index=False)
    pd.DataFrame(episode_rows).to_parquet(dst_root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    _task_table(task).to_parquet(dst_root / "meta/tasks.parquet")

    info = dict(src_info)
    info.update(
        {
            "total_episodes": len(episode_rows),
            "total_frames": len(data_df),
            "total_tasks": 1,
            "splits": {"train": f"0:{len(episode_rows)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        }
    )
    _write_json(dst_root / "meta/info.json", info)

    # These stats are approximate for review. If these split datasets are later
    # used for training, recompute stats with the official LeRobot tooling first.
    src_stats = src_root / "meta/stats.json"
    if src_stats.exists():
        shutil.copy2(src_stats, dst_root / "meta/stats.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-run-frames", type=int, default=5)
    parser.add_argument("--post-release-buffer-frames", type=int, default=30)
    parser.add_argument("--right-gripper-action-index", type=int, default=13)
    parser.add_argument("--video-mode", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()

    src_root = args.dataset_root.expanduser().resolve()
    out_root = args.output_root.expanduser().resolve()
    fps = int(_load_json(src_root / "meta/info.json").get("fps", 30))
    src_info = _load_json(src_root / "meta/info.json")

    data_path = src_root / "data/chunk-000/file-000.parquet"
    episodes_path = src_root / "meta/episodes/chunk-000/file-000.parquet"
    src_data = pd.read_parquet(data_path)
    src_episodes = pd.read_parquet(episodes_path)

    success_episode_ids = src_episodes.loc[
        src_episodes["episode_success"].map(_normalize_success) == "success", "episode_index"
    ].astype(int)

    part1_specs: list[SegmentSpec] = []
    part2_specs: list[SegmentSpec] = []
    report_rows: list[dict[str, Any]] = []

    for original_episode_index in success_episode_ids.tolist():
        ep_data = src_data.loc[src_data["episode_index"] == original_episode_index].reset_index(drop=True)
        action_values = np.stack(ep_data["action"].to_numpy())
        gripper = action_values[:, args.right_gripper_action_index].astype(float)
        split_frame, diag = _find_split(
            gripper,
            threshold=args.threshold,
            min_run_frames=args.min_run_frames,
            post_release_buffer_frames=args.post_release_buffer_frames,
        )

        status = "ok"
        if split_frame is None:
            status = "missing_first_release"
        elif split_frame <= 5 or split_frame >= len(ep_data) - 5:
            status = "bad_split_boundary"
        elif diag["second_close_frame"] is None or diag["second_release_frame"] is None:
            status = "missing_second_gripper_cycle"

        row = {
            "episode_index": original_episode_index,
            "episode_length": len(ep_data),
            "status": status,
            "split_frame": split_frame,
            "split_time_s": None if split_frame is None else split_frame / float(fps),
            **diag,
        }
        report_rows.append(row)

        if status != "ok":
            continue

        part1_specs.append(
            SegmentSpec(
                original_episode_index=original_episode_index,
                new_episode_index=len(part1_specs),
                start_frame=0,
                end_frame=int(split_frame),
                split_frame=int(split_frame),
                first_close_frame=int(diag["first_close_frame"]),
                first_release_frame=int(diag["first_release_frame"]),
                second_close_frame=diag["second_close_frame"],
                second_release_frame=diag["second_release_frame"],
                status=status,
            )
        )
        part2_specs.append(
            SegmentSpec(
                original_episode_index=original_episode_index,
                new_episode_index=len(part2_specs),
                start_frame=int(split_frame),
                end_frame=len(ep_data),
                split_frame=int(split_frame),
                first_close_frame=int(diag["first_close_frame"]),
                first_release_frame=int(diag["first_release_frame"]),
                second_close_frame=diag["second_close_frame"],
                second_release_frame=diag["second_release_frame"],
                status=status,
            )
        )

    review_dir = out_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(review_dir / "split_report.csv", index=False)
    _write_json(review_dir / "split_report.json", report_rows)
    _write_json(
        review_dir / "summary.json",
        {
            "source_dataset": str(src_root),
            "output_root": str(out_root),
            "success_episodes": int(len(success_episode_ids)),
            "part1_episodes": int(len(part1_specs)),
            "part2_episodes": int(len(part2_specs)),
            "status_counts": report_df["status"].value_counts(dropna=False).to_dict(),
            "threshold": args.threshold,
            "min_run_frames": args.min_run_frames,
            "post_release_buffer_frames": args.post_release_buffer_frames,
            "part1_task": PART1_TASK,
            "part2_task": PART2_TASK,
            "note": "Stats are copied from the source dataset for review only; recompute stats before training.",
        },
    )

    _build_segment_dataset(
        src_root=src_root,
        dst_root=out_root / "part1_first_vial_bottom_right",
        src_data=src_data,
        src_episodes=src_episodes,
        src_info=src_info,
        specs=part1_specs,
        task=PART1_TASK,
        video_mode=args.video_mode,
        fps=fps,
    )
    _build_segment_dataset(
        src_root=src_root,
        dst_root=out_root / "part2_second_vial_top_right",
        src_data=src_data,
        src_episodes=src_episodes,
        src_info=src_info,
        specs=part2_specs,
        task=PART2_TASK,
        video_mode=args.video_mode,
        fps=fps,
    )

    print(json.dumps(_load_json(review_dir / "summary.json"), indent=2))


if __name__ == "__main__":
    main()
