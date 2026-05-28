#!/usr/bin/env python3
"""Merge single-chunk LeRobot v3 datasets with sequential episode/frame indices.

This is intentionally narrow: it is for locally converted datasets that use one
data parquet file and video files referenced from episode metadata. It avoids the
generic aggregate path when parquet episode-stat schemas drift between datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
COMPLEMENTARY_PREFIX = "complementary_info."


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _json_dump(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False))


def _as_matrix(series: pd.Series) -> np.ndarray:
    first = series.iloc[0]
    if isinstance(first, np.ndarray):
        return np.stack(series.to_numpy()).astype(np.float64)
    if isinstance(first, (list, tuple)):
        return np.asarray(series.to_list(), dtype=np.float64)
    return series.to_numpy(dtype=np.float64).reshape(-1, 1)


def _feature_stats(values: np.ndarray) -> dict[str, list[Any]]:
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    stats: dict[str, Any] = {
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "count": np.array([values.shape[0]], dtype=np.int64),
    }
    for q in QUANTILES:
        stats[f"q{int(q * 100):02d}"] = np.quantile(values, q, axis=0)
    return {name: np.asarray(value).tolist() for name, value in stats.items()}


def _strip_complementary_features(info: dict[str, Any]) -> dict[str, Any]:
    out = dict(info)
    out["features"] = {
        name: spec for name, spec in info["features"].items() if not name.startswith(COMPLEMENTARY_PREFIX)
    }
    return out


def _numeric_feature_names(info: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for name, spec in info["features"].items():
        if spec.get("dtype") in {"image", "video", "string"}:
            continue
        names.append(name)
    return names


def _video_keys(info: dict[str, Any]) -> list[str]:
    return [name for name, spec in info["features"].items() if spec.get("dtype") == "video"]


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copy_video_files(src_root: Path, dst_root: Path, video_key: str, next_file_idx: int) -> dict[int, int]:
    src_dir = src_root / "videos" / video_key / "chunk-000"
    mapping: dict[int, int] = {}
    if not src_dir.exists():
        return mapping
    for src_file in sorted(src_dir.glob("file-*.mp4")):
        old_idx = int(src_file.stem.split("-")[-1])
        new_idx = next_file_idx + len(mapping)
        dst_file = dst_root / "videos" / video_key / "chunk-000" / f"file-{new_idx:03d}.mp4"
        _link_or_copy(src_file, dst_file)
        mapping[old_idx] = new_idx
    return mapping


def _minimal_episode_row(
    src_row: pd.Series,
    new_episode_index: int,
    dataset_from_index: int,
    dataset_to_index: int,
    video_keys: list[str],
    video_file_maps: dict[str, dict[int, int]],
    task: str | None,
) -> dict[str, Any]:
    length = int(dataset_to_index - dataset_from_index)
    tasks_value: Any
    if task is not None:
        tasks_value = np.asarray([task], dtype=object)
    else:
        tasks_value = src_row.get("tasks", np.asarray([], dtype=object))

    out: dict[str, Any] = {
        "episode_index": int(new_episode_index),
        "tasks": tasks_value,
        "length": length,
        "data/chunk_index": 0,
        "data/file_index": 0,
        "dataset_from_index": int(dataset_from_index),
        "dataset_to_index": int(dataset_to_index),
        "meta/episodes/chunk_index": 0,
        "meta/episodes/file_index": 0,
        "episode_success": "success",
    }
    for key in video_keys:
        base = f"videos/{key}"
        old_file_idx = int(src_row.get(f"{base}/file_index", 0))
        out[f"{base}/chunk_index"] = 0
        out[f"{base}/file_index"] = int(video_file_maps[key][old_file_idx])
        out[f"{base}/from_timestamp"] = float(src_row.get(f"{base}/from_timestamp", 0.0))
        out[f"{base}/to_timestamp"] = float(src_row.get(f"{base}/to_timestamp", 0.0))
    return out


def merge_datasets(
    src_roots: list[Path],
    dst_root: Path,
    repo_id: str,
    task: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(dst_root)
        shutil.rmtree(dst_root)

    first_info = _strip_complementary_features(_json_load(src_roots[0] / "meta/info.json"))
    video_keys = _video_keys(first_info)
    numeric_features = _numeric_feature_names(first_info)

    all_frames: list[pd.DataFrame] = []
    episode_rows: list[dict[str, Any]] = []
    video_next_idx = {key: 0 for key in video_keys}
    next_episode_index = 0
    next_frame_index = 0

    dst_root.mkdir(parents=True, exist_ok=True)

    for src_root in src_roots:
        info = _strip_complementary_features(_json_load(src_root / "meta/info.json"))
        if list(info["features"].keys()) != list(first_info["features"].keys()):
            raise ValueError(f"Feature mismatch for {src_root}")

        data_path = src_root / "data/chunk-000/file-000.parquet"
        episodes_path = src_root / "meta/episodes/chunk-000/file-000.parquet"
        if not data_path.exists() or not episodes_path.exists():
            raise FileNotFoundError(f"Missing data or episodes parquet in {src_root}")

        src_data = pd.read_parquet(data_path)
        src_episodes = pd.read_parquet(episodes_path)
        src_data = src_data[[col for col in src_data.columns if not col.startswith(COMPLEMENTARY_PREFIX)]].copy()

        video_maps: dict[str, dict[int, int]] = {}
        for key in video_keys:
            mapping = _copy_video_files(src_root, dst_root, key, video_next_idx[key])
            video_next_idx[key] += len(mapping)
            video_maps[key] = mapping

        episode_map: dict[int, int] = {}
        for old_episode_index in sorted(src_data["episode_index"].unique()):
            old_episode_index = int(old_episode_index)
            episode_map[old_episode_index] = next_episode_index
            rows = src_data[src_data["episode_index"] == old_episode_index]
            length = int(len(rows))
            src_ep_matches = src_episodes[src_episodes["episode_index"] == old_episode_index]
            if len(src_ep_matches) != 1:
                raise ValueError(f"Expected one episode row for {src_root} episode {old_episode_index}")
            episode_rows.append(
                _minimal_episode_row(
                    src_ep_matches.iloc[0],
                    new_episode_index=next_episode_index,
                    dataset_from_index=next_frame_index,
                    dataset_to_index=next_frame_index + length,
                    video_keys=video_keys,
                    video_file_maps=video_maps,
                    task=task,
                )
            )
            next_episode_index += 1
            next_frame_index += length

        src_data["episode_index"] = src_data["episode_index"].map(episode_map).astype("int64")
        src_data["index"] = np.arange(len(src_data), dtype=np.int64) + sum(len(x) for x in all_frames)
        src_data["task_index"] = 0
        all_frames.append(src_data)

    merged = pd.concat(all_frames, ignore_index=True)
    merged["index"] = np.arange(len(merged), dtype=np.int64)

    data_dir = dst_root / "data/chunk-000"
    meta_dir = dst_root / "meta"
    episodes_dir = meta_dir / "episodes/chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    merged.to_parquet(data_dir / "file-000.parquet", index=False)
    pd.DataFrame(episode_rows).to_parquet(episodes_dir / "file-000.parquet", index=False)

    # Keep the existing task table shape used by these NERO datasets.
    pd.DataFrame({"task_index": [0]}).to_parquet(meta_dir / "tasks.parquet", index=False)

    stats: dict[str, Any] = {}
    for feature in numeric_features:
        if feature in merged.columns:
            stats[feature] = _feature_stats(_as_matrix(merged[feature]))

    # Image stats are not used for VISUAL normalization in this setup, but keeping
    # template values preserves compatibility with existing dataset readers.
    first_stats = _json_load(src_roots[0] / "meta/stats.json")
    for key in video_keys:
        if key in first_stats:
            stats[key] = first_stats[key]

    info = first_info
    info["repo_id"] = repo_id
    info["total_episodes"] = int(len(episode_rows))
    info["total_frames"] = int(len(merged))
    info["total_tasks"] = 1
    info["splits"] = {"train": f"0:{len(episode_rows)}"}
    info["data_files_size_in_mb"] = float((data_dir / "file-000.parquet").stat().st_size / 1024**2)
    info["video_files_size_in_mb"] = float(
        sum(path.stat().st_size for path in (dst_root / "videos").rglob("*.mp4")) / 1024**2
    )

    _json_dump(info, meta_dir / "info.json")
    _json_dump(stats, meta_dir / "stats.json")

    return {
        "dst_root": str(dst_root),
        "repo_id": repo_id,
        "total_episodes": int(len(episode_rows)),
        "total_frames": int(len(merged)),
        "sources": [str(path) for path in src_roots],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", action="append", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = merge_datasets(args.src_root, args.dst_root, args.repo_id, args.task, args.overwrite)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
