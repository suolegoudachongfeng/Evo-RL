#!/usr/bin/env python3
"""Create a LeRobot dataset variant with zero end-effector action frames removed.

The filter only inspects action dimensions whose names contain ``delta_ee_pose``.
Gripper command dimensions are intentionally ignored, so a frame with unchanged
end-effector action is removable even if the gripper command value is non-zero.
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


def _json_dump(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False))


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _as_matrix(series: pd.Series) -> np.ndarray:
    first = series.iloc[0]
    if isinstance(first, np.ndarray):
        return np.stack(series.to_numpy()).astype(np.float64)
    if isinstance(first, (list, tuple)):
        return np.asarray(series.to_list(), dtype=np.float64)
    return series.to_numpy(dtype=np.float64).reshape(-1, 1)


def _feature_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    stats = {
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "count": np.array([values.shape[0]], dtype=np.int64),
    }
    for q in QUANTILES:
        stats[f"q{int(q * 100):02d}"] = np.quantile(values, q, axis=0)
    return stats


def _serialize_stats(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, list[float]]]:
    return {
        feature: {name: np.asarray(value).tolist() for name, value in feature_stats.items()}
        for feature, feature_stats in stats.items()
    }


def _copy_tree_with_hardlinks(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        out = dst / rel
        if path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, out)
        except OSError:
            shutil.copy2(path, out)


def _action_ee_indices(info: dict[str, Any]) -> list[int]:
    action = info["features"]["action"]
    names = action.get("names") or []
    indices = [idx for idx, name in enumerate(names) if "delta_ee_pose" in str(name)]
    if not indices:
        raise ValueError("No action dimensions containing 'delta_ee_pose' were found.")
    return indices


def _numeric_stat_features(info: dict[str, Any], stats_template: dict[str, Any]) -> list[str]:
    features = []
    for name, spec in info["features"].items():
        if spec.get("dtype") in {"video", "image", "string"}:
            continue
        if name in stats_template:
            features.append(name)
    return features


def _update_episode_stats_cell(episodes: pd.DataFrame, row_idx: int, feature: str, stats: dict[str, np.ndarray]) -> None:
    for stat_name, value in stats.items():
        col = f"stats/{feature}/{stat_name}"
        if col in episodes.columns:
            # Pandas can coerce a shape-(1,) ndarray into a 0-d array when assigning
            # into an object cell, which PyArrow then refuses to serialize. Lists
            # preserve the expected 1-D list<...> parquet representation.
            episodes.at[row_idx, col] = np.asarray(value).tolist()


def create_filtered_dataset(
    src_root: Path,
    dst_root: Path,
    repo_id: str,
    threshold: float,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    info_path = src_root / "meta/info.json"
    stats_path = src_root / "meta/stats.json"
    episodes_path = src_root / "meta/episodes/chunk-000/file-000.parquet"
    data_path = src_root / "data/chunk-000/file-000.parquet"
    tasks_path = src_root / "meta/tasks.parquet"

    for path in (info_path, stats_path, episodes_path, data_path, tasks_path):
        if not path.exists():
            raise FileNotFoundError(path)

    info = _json_load(info_path)
    stats_template = _json_load(stats_path)
    ee_indices = _action_ee_indices(info)
    df = pd.read_parquet(data_path)
    action = np.stack(df["action"].to_numpy())
    zero_mask = np.max(np.abs(action[:, ee_indices]), axis=1) <= threshold
    keep_mask = ~zero_mask

    per_episode = (
        pd.DataFrame({"episode_index": df["episode_index"], "zero": zero_mask})
        .groupby("episode_index")["zero"]
        .agg(total="size", removed="sum")
    )
    per_episode["kept"] = per_episode["total"] - per_episode["removed"]
    empty_episodes = per_episode.index[per_episode["kept"] == 0].tolist()
    if empty_episodes:
        raise ValueError(f"Filtering would create empty episodes: {empty_episodes}")

    summary = {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "repo_id": repo_id,
        "threshold": threshold,
        "ee_action_indices": ee_indices,
        "total_frames": int(len(df)),
        "removed_frames": int(zero_mask.sum()),
        "kept_frames": int(keep_mask.sum()),
        "removed_percent": float(zero_mask.mean() * 100.0),
        "episodes": int(per_episode.shape[0]),
    }
    if dry_run:
        return summary

    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {dst_root}")
        shutil.rmtree(dst_root)

    (dst_root / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (dst_root / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    shutil.copy2(tasks_path, dst_root / "meta/tasks.parquet")
    _copy_tree_with_hardlinks(src_root / "videos", dst_root / "videos")

    filtered = df.loc[keep_mask].copy().reset_index(drop=True)
    filtered["index"] = np.arange(len(filtered), dtype=np.int64)

    numeric_features = _numeric_stat_features(info, stats_template)
    global_stats: dict[str, dict[str, np.ndarray]] = dict(stats_template)
    for feature in numeric_features:
        if feature in filtered.columns:
            global_stats[feature] = _feature_stats(_as_matrix(filtered[feature]))

    episodes = pd.read_parquet(episodes_path).copy()
    for ep_idx, ep_rows in filtered.groupby("episode_index", sort=True):
        ep_idx = int(ep_idx)
        row_idx = episodes.index[episodes["episode_index"] == ep_idx]
        if len(row_idx) != 1:
            raise ValueError(f"Expected one metadata row for episode {ep_idx}, found {len(row_idx)}")
        row_idx = int(row_idx[0])
        episodes.at[row_idx, "length"] = int(len(ep_rows))
        episodes.at[row_idx, "data/chunk_index"] = 0
        episodes.at[row_idx, "data/file_index"] = 0
        episodes.at[row_idx, "dataset_from_index"] = int(ep_rows["index"].min())
        episodes.at[row_idx, "dataset_to_index"] = int(ep_rows["index"].max() + 1)
        for feature in numeric_features:
            if feature in ep_rows.columns:
                _update_episode_stats_cell(episodes, row_idx, feature, _feature_stats(_as_matrix(ep_rows[feature])))

    filtered.to_parquet(dst_root / "data/chunk-000/file-000.parquet", index=False)
    episodes.to_parquet(dst_root / "meta/episodes/chunk-000/file-000.parquet", index=False)

    info["repo_id"] = repo_id
    info["total_frames"] = int(len(filtered))
    info["total_episodes"] = int(episodes.shape[0])
    info["total_tasks"] = 1
    info["splits"] = {"train": f"0:{episodes.shape[0]}"}
    info["data_files_size_in_mb"] = float((dst_root / "data/chunk-000/file-000.parquet").stat().st_size / 1024**2)
    _json_dump(info, dst_root / "meta/info.json")
    _json_dump(_serialize_stats(global_stats), dst_root / "meta/stats.json")
    _json_dump(summary, dst_root / "meta/zero_ee_action_filter_report.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = create_filtered_dataset(
        src_root=args.src_root,
        dst_root=args.dst_root,
        repo_id=args.repo_id,
        threshold=args.threshold,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
