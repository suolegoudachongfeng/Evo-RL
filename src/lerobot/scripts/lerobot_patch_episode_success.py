#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Patch episode-level success/failure labels into a local LeRobot dataset."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.recording_annotations import (
    EPISODE_FAILURE,
    EPISODE_SUCCESS,
    normalize_episode_success_label,
)
from lerobot.utils.utils import init_logging


def resolve_dataset_root(dataset: str, root: str | None) -> Path:
    dataset_path = Path(dataset).expanduser()
    if dataset_path.exists():
        return dataset_path.resolve()

    base_root = Path(root).expanduser().resolve() if root is not None else HF_LEROBOT_HOME.resolve()
    candidate = base_root / dataset
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(f"Dataset not found as path or under {base_root}: {dataset}")


def parse_episode_indices(raw: str | None, label_name: str) -> set[int]:
    if raw is None or raw.strip() == "":
        return set()

    indices: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"{label_name} range '{token}' has end < start.")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(token))
    return indices


def resolve_label(default_success: str, episode_index: int, success_eps: set[int], failure_eps: set[int]) -> str:
    in_success = episode_index in success_eps
    in_failure = episode_index in failure_eps
    if in_success and in_failure:
        raise ValueError(f"Episode {episode_index} is listed as both success and failure.")
    if in_success:
        return EPISODE_SUCCESS
    if in_failure:
        return EPISODE_FAILURE
    return default_success


def patch_episode_success(
    dataset_root: Path,
    field: str,
    default_success: str,
    success_episodes: set[int],
    failure_episodes: set[int],
    overwrite: bool,
    dry_run: bool,
) -> dict[str, int]:
    episodes_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episodes_files:
        raise FileNotFoundError(f"No episode metadata parquet files found under {dataset_root / 'meta/episodes'}")

    total_rows = 0
    success_count = 0
    failure_count = 0
    changed_files = 0

    for path in episodes_files:
        df = pd.read_parquet(path)
        if "episode_index" not in df.columns:
            raise KeyError(f"{path} is missing required column 'episode_index'.")
        if field in df.columns and not overwrite:
            raise ValueError(f"{path} already has '{field}'. Pass --overwrite to replace it.")

        labels = [
            resolve_label(default_success, int(episode_index), success_episodes, failure_episodes)
            for episode_index in df["episode_index"].to_list()
        ]
        success_count += sum(label == EPISODE_SUCCESS for label in labels)
        failure_count += sum(label == EPISODE_FAILURE for label in labels)
        total_rows += len(labels)

        if not dry_run:
            df[field] = labels
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, path, compression="snappy", use_dictionary=True)
            changed_files += 1

    return {
        "episodes": total_rows,
        "success": success_count,
        "failure": failure_count,
        "changed_files": changed_files,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset repo_id or local dataset path.")
    parser.add_argument("--root", default=None, help="Base LeRobot cache root. Defaults to HF_LEROBOT_HOME.")
    parser.add_argument("--field", default="episode_success", help="Episode metadata field to write.")
    parser.add_argument(
        "--default-success",
        required=True,
        choices=[EPISODE_SUCCESS, EPISODE_FAILURE],
        help="Default label for every episode unless overridden.",
    )
    parser.add_argument(
        "--success-episodes",
        default=None,
        help="Comma-separated episode indices/ranges to force success, e.g. '0,2,5-8'.",
    )
    parser.add_argument(
        "--failure-episodes",
        default=None,
        help="Comma-separated episode indices/ranges to force failure, e.g. '1,3,9-12'.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing labels if present.")
    parser.add_argument("--dry-run", action="store_true", help="Print counts without writing files.")
    return parser


def main() -> None:
    init_logging()
    args = build_parser().parse_args()

    default_success = normalize_episode_success_label(args.default_success)
    if default_success is None:
        raise ValueError("--default-success must be success or failure.")

    dataset_root = resolve_dataset_root(args.dataset, args.root)
    success_episodes = parse_episode_indices(args.success_episodes, "--success-episodes")
    failure_episodes = parse_episode_indices(args.failure_episodes, "--failure-episodes")

    result = patch_episode_success(
        dataset_root=dataset_root,
        field=args.field,
        default_success=default_success,
        success_episodes=success_episodes,
        failure_episodes=failure_episodes,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    logging.info(
        "Patched %s | episodes=%d success=%d failure=%d changed_files=%d dry_run=%s",
        dataset_root,
        result["episodes"],
        result["success"],
        result["failure"],
        result["changed_files"],
        args.dry_run,
    )


if __name__ == "__main__":
    main()
