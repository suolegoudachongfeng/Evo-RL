#!/usr/bin/env python3
"""Create NERO dataset copies with idle action frames removed.

This script is intentionally conservative:
- the source dataset is never modified;
- videos are symlinked or copied from the source dataset;
- original per-frame timestamps are preserved so symlinked videos remain aligned;
- global frame indices and episode metadata are rebuilt for LeRobot loading.

For the current 2 mL NERO task the right arm performs the work, while gripper
state is encoded in the last action dimensions. A strictly all-zero 14-D action
does not occur in the collected data, so the default idle criterion removes
frames where the right arm motion slice action[6:12] is exactly zero.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


QUANTILE_KEYS = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def _normalize_success(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode()
    text = str(value).strip().lower()
    if text in {"success", "true", "1", "s"}:
        return "success"
    if text in {"failure", "false", "0", "f"}:
        return "failure"
    return text


def _read_single_parquet(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _parse_slice(text: str) -> slice:
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected slice in start:end form, got {text!r}")
    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if parts[1] else None
    return slice(start, end)


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


def _copy_static_metadata(src_root: Path, dst_root: Path) -> None:
    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"
    if dst_meta.exists():
        shutil.rmtree(dst_meta)
    dst_meta.mkdir(parents=True, exist_ok=True)

    for item in src_meta.iterdir():
        if item.name in {"info.json", "stats.json", "episodes"}:
            continue
        dst = dst_meta / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    for name in ("README.md", ".gitattributes"):
        src = src_root / name
        if src.exists():
            shutil.copy2(src, dst_root / name)


def _stack_feature(series: pd.Series) -> np.ndarray:
    values = series.to_numpy()
    first = values[0]
    if isinstance(first, np.ndarray):
        return np.stack(values)
    return np.asarray(values)


def _feature_stats(array: np.ndarray) -> dict[str, list[Any]]:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
        squeeze_scalar = True
    else:
        squeeze_scalar = False

    stats: dict[str, Any] = {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
        "count": np.asarray([arr.shape[0]], dtype=np.int64),
    }
    for key, q in QUANTILE_KEYS.items():
        stats[key] = np.quantile(arr, q, axis=0)

    if squeeze_scalar:
        return {key: np.asarray(value).reshape(-1).tolist() for key, value in stats.items()}
    return {key: np.asarray(value).tolist() for key, value in stats.items()}


def _compute_stats(data_df: pd.DataFrame, src_stats: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key, value in src_stats.items():
        if key.startswith("observation.images."):
            # Visual normalization is identity in current PI05 configs. Reusing
            # image stats keeps metadata complete while avoiding video decoding.
            stats[key] = value
            continue
        if key not in data_df.columns or len(data_df) == 0:
            continue
        try:
            stats[key] = _feature_stats(_stack_feature(data_df[key]))
        except Exception:
            # Keep non-standard stats out rather than writing malformed metadata.
            continue
    return stats


def _build_filtered_dataset(
    *,
    src_root: Path,
    dst_root: Path,
    src_data: pd.DataFrame,
    src_episodes: pd.DataFrame,
    src_info: dict[str, Any],
    src_stats: dict[str, Any],
    action_zero_mask: np.ndarray,
    selected_original_episodes: list[int],
    video_mode: str,
    replace: bool,
    label: str,
    action_slice_text: str,
) -> dict[str, Any]:
    if dst_root.exists():
        if not replace:
            raise FileExistsError(f"{dst_root} already exists; pass --replace to overwrite")
        shutil.rmtree(dst_root)

    _copy_static_metadata(src_root, dst_root)
    (dst_root / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (dst_root / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    _copy_or_link_videos(src_root, dst_root, video_mode)

    data_parts: list[pd.DataFrame] = []
    episode_rows: list[dict[str, Any]] = []
    video_cols = [c for c in src_episodes.columns if c.startswith("videos/")]
    global_index = 0
    dropped_zero_frames = 0
    dropped_empty_episodes: list[int] = []

    data_by_episode = {int(k): v for k, v in src_data.groupby("episode_index", sort=False)}
    episode_by_id = {
        int(row["episode_index"]): row for _, row in src_episodes.iterrows()
    }

    for new_ep_idx, original_ep_idx in enumerate(selected_original_episodes):
        ep_data = data_by_episode[int(original_ep_idx)].copy()
        keep = ~action_zero_mask[ep_data.index.to_numpy()]
        removed = int((~keep).sum())
        dropped_zero_frames += removed
        ep_data = ep_data.loc[keep].copy()
        if ep_data.empty:
            dropped_empty_episodes.append(int(original_ep_idx))
            continue

        new_ep_idx = len(episode_rows)
        ep_len = int(len(ep_data))
        ep_data["episode_index"] = new_ep_idx
        ep_data["index"] = np.arange(global_index, global_index + ep_len, dtype=np.int64)
        # Keep original frame_index and timestamp. They are needed to query the
        # original videos after idle frames are removed.
        data_parts.append(ep_data)

        src_row = episode_by_id[int(original_ep_idx)]
        row: dict[str, Any] = {
            "episode_index": new_ep_idx,
            "tasks": src_row["tasks"],
            "length": ep_len,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": global_index,
            "dataset_to_index": global_index + ep_len,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "episode_success": src_row.get("episode_success", None),
            "original_episode_index": int(original_ep_idx),
            "removed_zero_right_motion_frames": removed,
        }
        for col in video_cols:
            row[col] = src_row[col]
        episode_rows.append(row)
        global_index += ep_len

    if not data_parts:
        raise RuntimeError(f"No frames remain after filtering for {dst_root}")

    data_df = pd.concat(data_parts, ignore_index=True)
    episodes_df = pd.DataFrame(episode_rows)
    data_df.to_parquet(dst_root / "data/chunk-000/file-000.parquet", index=False)
    episodes_df.to_parquet(dst_root / "meta/episodes/chunk-000/file-000.parquet", index=False)

    info = dict(src_info)
    info.update(
        {
            "total_episodes": int(len(episode_rows)),
            "total_frames": int(len(data_df)),
            "splits": {"train": f"0:{len(episode_rows)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        }
    )
    _write_json(dst_root / "meta/info.json", info)
    _write_json(dst_root / "meta/stats.json", _compute_stats(data_df, src_stats))

    summary = {
        "label": label,
        "source_dataset": str(src_root),
        "output_dataset": str(dst_root),
        "video_mode": video_mode,
        "action_zero_definition": f"all(action[{action_slice_text}] == 0)",
        "selected_original_episodes": int(len(selected_original_episodes)),
        "output_episodes": int(len(episode_rows)),
        "source_selected_frames": int(
            src_data.loc[src_data["episode_index"].isin(selected_original_episodes)].shape[0]
        ),
        "output_frames": int(len(data_df)),
        "dropped_zero_right_motion_frames": int(dropped_zero_frames),
        "dropped_empty_original_episodes": dropped_empty_episodes,
        "note": (
            "Per-frame timestamp and frame_index values are preserved from the source "
            "dataset so symlinked videos remain aligned after row filtering."
        ),
    }
    _write_json(dst_root / "cleaning_report.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-all", type=Path, required=True)
    parser.add_argument("--output-success", type=Path, required=True)
    parser.add_argument("--action-slice", default="6:12")
    parser.add_argument("--video-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    src_root = args.dataset_root.expanduser().resolve()
    action_slice = _parse_slice(args.action_slice)
    src_data = _read_single_parquet(src_root, "data/chunk-000/file-000.parquet")
    src_episodes = _read_single_parquet(src_root, "meta/episodes/chunk-000/file-000.parquet")
    src_info = _load_json(src_root / "meta/info.json")
    src_stats = _load_json(src_root / "meta/stats.json")

    action = np.stack(src_data["action"].to_numpy())
    zero_motion = np.all(action[:, action_slice] == 0, axis=1)

    all_episode_ids = src_episodes["episode_index"].astype(int).tolist()
    success_episode_ids = (
        src_episodes.loc[
            src_episodes["episode_success"].map(_normalize_success) == "success", "episode_index"
        ]
        .astype(int)
        .tolist()
    )

    all_summary = _build_filtered_dataset(
        src_root=src_root,
        dst_root=args.output_all.expanduser().resolve(),
        src_data=src_data,
        src_episodes=src_episodes,
        src_info=src_info,
        src_stats=src_stats,
        action_zero_mask=zero_motion,
        selected_original_episodes=all_episode_ids,
        video_mode=args.video_mode,
        replace=args.replace,
        label="all_episodes_zero_right_motion_removed",
        action_slice_text=args.action_slice,
    )
    success_summary = _build_filtered_dataset(
        src_root=src_root,
        dst_root=args.output_success.expanduser().resolve(),
        src_data=src_data,
        src_episodes=src_episodes,
        src_info=src_info,
        src_stats=src_stats,
        action_zero_mask=zero_motion,
        selected_original_episodes=success_episode_ids,
        video_mode=args.video_mode,
        replace=args.replace,
        label="success_only_zero_right_motion_removed",
        action_slice_text=args.action_slice,
    )

    summary = {
        "source_dataset": str(src_root),
        "source_episodes": int(src_info["total_episodes"]),
        "source_frames": int(src_info["total_frames"]),
        "source_success_episodes": int(len(success_episode_ids)),
        "strict_all_14d_zero_action_frames": int(np.all(action == 0, axis=1).sum()),
        "zero_right_motion_frames": int(zero_motion.sum()),
        "outputs": {
            "all": all_summary,
            "success_only": success_summary,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
