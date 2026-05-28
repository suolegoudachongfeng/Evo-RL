#!/usr/bin/env python
"""Prepare the empty-rack E113 baseline dataset for EvoRL training.

This script keeps the original camera/video geometry unchanged, removes the
single bad short episode, and adds the metadata fields expected by EvoRL.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset


PROMPT = (
    "Place small vials right-bottom then right-top with right gripper; "
    "place large vials left-bottom then left-top with left gripper."
)


def _patch_metadata(root: Path, repo_id: str) -> None:
    tasks_path = root / "meta" / "tasks.parquet"
    pd.DataFrame({"task_index": [0]}, index=[PROMPT]).to_parquet(tasks_path)

    for episodes_path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        episodes = pd.read_parquet(episodes_path)
        episodes["tasks"] = [[PROMPT] for _ in range(len(episodes))]
        episodes["episode_success"] = "success"
        episodes.to_parquet(episodes_path, index=False)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["repo_id"] = repo_id
    info["total_tasks"] = 1
    info["splits"] = {"train": f"0:{info['total_episodes']}"}
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--drop-episode", type=int, default=58)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.src.exists():
        raise FileNotFoundError(args.src)
    if args.dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.dst} exists; pass --overwrite to replace it")
        shutil.rmtree(args.dst)

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.src)
    delete_episodes(
        dataset=dataset,
        episode_indices=[args.drop_episode],
        output_dir=args.dst,
        repo_id=args.repo_id,
    )
    _patch_metadata(args.dst, args.repo_id)

    info = json.loads((args.dst / "meta" / "info.json").read_text())
    episodes = pd.concat(
        [pd.read_parquet(p) for p in sorted((args.dst / "meta" / "episodes").glob("chunk-*/file-*.parquet"))],
        ignore_index=True,
    )
    print("prepared", args.dst)
    print("repo_id", info.get("repo_id"))
    print("episodes", info.get("total_episodes"), "frames", info.get("total_frames"), "fps", info.get("fps"))
    print("episode_success_counts", episodes["episode_success"].value_counts(dropna=False).to_dict())
    print("shortest_after_drop", episodes.sort_values("length").head(5)[["episode_index", "length"]].to_dict("records"))
    print("prompt", PROMPT)


if __name__ == "__main__":
    main()
