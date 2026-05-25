#!/usr/bin/env python3
"""Combine four strict 2 mL segment datasets into one multi-task dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from split_nero_dataset_by_vlm_segments import (
    SEGMENT_TASKS,
    _compute_stats,
    _copy_or_link_videos,
    _copy_static_metadata,
    _load_json,
    _write_json,
)


SEGMENT_ORDER = [
    "grasp_first_vial",
    "insert_first_vial",
    "grasp_second_vial",
    "insert_second_vial",
]


def _read_single_parquet(root: Path, pattern: str) -> pd.DataFrame:
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matched {root / pattern}")
    frames = [pd.read_parquet(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def _copy_video_tree(src_root: Path, dst_root: Path, mode: str) -> None:
    if mode == "skip":
        return
    _copy_or_link_videos(src_root, dst_root, mode)


def _task_table(prompts: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"task_index": np.arange(len(prompts), dtype=np.int64)},
        index=pd.Index(prompts, name=None),
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def combine_segments(
    *,
    split_root: Path,
    output_root: Path,
    video_mode: str,
    replace: bool,
) -> dict[str, Any]:
    segment_roots = {
        key: split_root / SEGMENT_TASKS[key]["dir"]
        for key in SEGMENT_ORDER
    }
    missing = [str(path) for path in segment_roots.values() if not (path / "data/chunk-000/file-000.parquet").is_file()]
    if missing:
        raise FileNotFoundError("Missing segment datasets:\n" + "\n".join(missing))

    if output_root.exists():
        if not replace:
            raise FileExistsError(f"{output_root} exists; pass --replace")
        shutil.rmtree(output_root)

    first_root = segment_roots[SEGMENT_ORDER[0]]
    first_info = _load_json(first_root / "meta/info.json")
    first_stats = _load_json(first_root / "meta/stats.json")

    _copy_static_metadata(first_root, output_root)
    (output_root / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (output_root / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    _copy_video_tree(first_root, output_root, video_mode)

    prompts = [SEGMENT_TASKS[key]["prompt"] for key in SEGMENT_ORDER]
    data_parts: list[pd.DataFrame] = []
    episode_rows: list[dict[str, Any]] = []
    segment_summary: dict[str, dict[str, Any]] = {}
    global_index = 0

    for task_index, segment_key in enumerate(SEGMENT_ORDER):
        segment_root = segment_roots[segment_key]
        segment_name = SEGMENT_TASKS[segment_key]["dir"]
        data_df = _read_single_parquet(segment_root / "data", "chunk-*/file-*.parquet")
        episodes_df = _read_single_parquet(segment_root / "meta/episodes", "chunk-*/file-*.parquet")
        episodes_by_id = {int(row["episode_index"]): row for _, row in episodes_df.iterrows()}

        segment_frames = 0
        segment_episode_count = 0
        for old_episode_index in sorted(episodes_by_id):
            old_episode = episodes_by_id[old_episode_index]
            ep_data = data_df[data_df["episode_index"].astype(int) == old_episode_index].copy()
            if ep_data.empty:
                continue

            new_episode_index = len(episode_rows)
            length = int(len(ep_data))
            ep_data["episode_index"] = new_episode_index
            ep_data["task_index"] = task_index
            ep_data["index"] = np.arange(global_index, global_index + length, dtype=np.int64)
            data_parts.append(ep_data)

            row = dict(old_episode)
            row.update(
                {
                    "episode_index": new_episode_index,
                    "tasks": np.array([prompts[task_index]], dtype=object),
                    "length": length,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": global_index,
                    "dataset_to_index": global_index + length,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                    "task_index": task_index,
                    "segment_key": segment_key,
                    "segment_dir": segment_name,
                    "source_segment_episode_index": old_episode_index,
                }
            )
            episode_rows.append(row)
            segment_frames += length
            segment_episode_count += 1
            global_index += length

        segment_summary[segment_key] = {
            "segment_dir": segment_name,
            "task_index": task_index,
            "prompt": prompts[task_index],
            "episodes": segment_episode_count,
            "frames": segment_frames,
        }

    if not data_parts:
        raise RuntimeError("No data generated for combined multi-task dataset")

    combined_data = pd.concat(data_parts, ignore_index=True)
    combined_episodes = pd.DataFrame(episode_rows)
    combined_data.to_parquet(output_root / "data/chunk-000/file-000.parquet", index=False)
    combined_episodes.to_parquet(output_root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    _task_table(prompts).to_parquet(output_root / "meta/tasks.parquet")

    info = dict(first_info)
    info.update(
        {
            "total_episodes": int(len(combined_episodes)),
            "total_frames": int(len(combined_data)),
            "total_tasks": len(prompts),
            "splits": {"train": f"0:{len(combined_episodes)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        }
    )
    _write_json(output_root / "meta/info.json", info)
    _write_json(output_root / "meta/stats.json", _compute_stats(combined_data, first_stats))

    report = {
        "split_root": str(split_root),
        "output_root": str(output_root),
        "video_mode": video_mode,
        "segments": segment_summary,
        "total_episodes": int(len(combined_episodes)),
        "total_frames": int(len(combined_data)),
        "total_tasks": len(prompts),
        "note": "Each segment becomes one task_index in a single LeRobot dataset; source timestamps and frame_index values are preserved.",
    }
    _write_json(output_root / "combine_summary.json", report)
    pd.DataFrame(
        [
            {
                "segment_key": key,
                **value,
            }
            for key, value in segment_summary.items()
        ]
    ).to_csv(output_root / "combine_summary.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video-mode", choices=("copy", "symlink", "skip"), default="copy")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    report = combine_segments(
        split_root=args.split_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        video_mode=args.video_mode,
        replace=args.replace,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
