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

"""Convert legacy NERO LeRobot datasets to the Evo-RL human-in-loop schema.

The converter keeps the NERO 28D observation schema used by the ACT/NERO project:
left joints, right joints, left EE pose, right EE pose, then left/right grippers.
It adds Evo-RL's complementary_info fields with zero/false defaults so purely
human-collected demonstrations can be used by the Evo-RL training pipeline.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.utils import write_stats
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging

LEGACY_ACTION_NAMES = [
    "left_delta_ee_pose.x",
    "left_delta_ee_pose.y",
    "left_delta_ee_pose.z",
    "left_delta_ee_pose.rx",
    "left_delta_ee_pose.ry",
    "left_delta_ee_pose.rz",
    "right_delta_ee_pose.x",
    "right_delta_ee_pose.y",
    "right_delta_ee_pose.z",
    "right_delta_ee_pose.rx",
    "right_delta_ee_pose.ry",
    "right_delta_ee_pose.rz",
    "left_gripper_cmd",
    "right_gripper_cmd",
]

EVORL_NERO_ACTION_NAMES = [
    *LEGACY_ACTION_NAMES[:-2],
    "left_gripper_cmd_bin",
    "right_gripper_cmd_bin",
]

EVORL_NERO_STATE_NAMES = [
    *[f"left_joint_{i}.pos" for i in range(1, 8)],
    *[f"right_joint_{i}.pos" for i in range(1, 8)],
    "left_ee_pose.x",
    "left_ee_pose.y",
    "left_ee_pose.z",
    "left_ee_pose.rx",
    "left_ee_pose.ry",
    "left_ee_pose.rz",
    "right_ee_pose.x",
    "right_ee_pose.y",
    "right_ee_pose.z",
    "right_ee_pose.rx",
    "right_ee_pose.ry",
    "right_ee_pose.rz",
    "left_gripper_cmd_bin",
    "right_gripper_cmd_bin",
]

EVORL_COMPLEMENTARY_FEATURES = {
    "complementary_info.policy_action": {
        "dtype": "float32",
        "shape": [len(EVORL_NERO_ACTION_NAMES)],
        "names": EVORL_NERO_ACTION_NAMES,
    },
    "complementary_info.is_intervention": {
        "dtype": "float32",
        "shape": [1],
        "names": ["is_intervention"],
    },
    "complementary_info.state": {
        "dtype": "float32",
        "shape": [1],
        "names": ["state"],
    },
}

LEGACY_DAGGER_FEATURES = {"action_source", "is_expert"}


def resolve_dataset_root(dataset: str, root: str | None) -> Path:
    dataset_path = Path(dataset).expanduser()
    if dataset_path.exists():
        return dataset_path.resolve()

    base_root = Path(root).expanduser().resolve() if root is not None else HF_LEROBOT_HOME.resolve()
    candidate = base_root / dataset
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(f"Dataset not found as path or under {base_root}: {dataset}")


def infer_output_repo_id(src_root: Path, dataset_arg: str, output_repo_id: str | None) -> str:
    if output_repo_id is not None:
        return output_repo_id
    dataset_path = Path(dataset_arg)
    if dataset_path.exists():
        return f"{src_root.parent.name}/{src_root.name}_evorl"
    return f"{dataset_arg}_evorl"


def resolve_output_root(output_repo_id: str, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return (HF_LEROBOT_HOME / output_repo_id).resolve()


def load_info(dataset_root: Path) -> dict[str, Any]:
    with (dataset_root / "meta" / "info.json").open() as f:
        return json.load(f)


def write_info(dataset_root: Path, info: dict[str, Any]) -> None:
    with (dataset_root / "meta" / "info.json").open("w") as f:
        json.dump(info, f, indent=4)


def feature_shape(info: dict[str, Any], key: str) -> tuple[int, ...]:
    shape = info["features"][key]["shape"]
    return tuple(int(v) for v in shape)


def validate_source_schema(src_root: Path, info: dict[str, Any]) -> None:
    features = info.get("features", {})
    missing = [key for key in ("action", "observation.state") if key not in features]
    if missing:
        raise ValueError(f"{src_root} is missing required features: {missing}")

    action_shape = feature_shape(info, "action")
    if action_shape != (len(EVORL_NERO_ACTION_NAMES),):
        raise ValueError(f"Expected 14D NERO action, got shape {action_shape}.")

    state_shape = feature_shape(info, "observation.state")
    if state_shape != (len(EVORL_NERO_STATE_NAMES),):
        raise ValueError(
            "Expected the expanded 28D NERO observation.state. "
            f"Got shape {state_shape}. The older 14D NERO datasets should be left out of this conversion."
        )


def normalize_features(info: dict[str, Any], keep_legacy_dagger_fields: bool) -> dict[str, Any]:
    features = dict(info["features"])
    if not keep_legacy_dagger_fields:
        for name in LEGACY_DAGGER_FEATURES:
            features.pop(name, None)

    features["action"] = {
        "dtype": "float32",
        "shape": [len(EVORL_NERO_ACTION_NAMES)],
        "names": EVORL_NERO_ACTION_NAMES,
    }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [len(EVORL_NERO_STATE_NAMES)],
        "names": EVORL_NERO_STATE_NAMES,
    }
    features.update(EVORL_COMPLEMENTARY_FEATURES)
    return features


def stack_feature(series: pd.Series) -> np.ndarray:
    values = [np.asarray(value, dtype=np.float32) for value in series.to_list()]
    return np.stack(values, axis=0)


def assign_vector_feature(df: pd.DataFrame, column: str, values: np.ndarray) -> None:
    df[column] = [row.astype(np.float32, copy=False) for row in values]


def transform_data_file(path: Path, keep_legacy_dagger_fields: bool) -> None:
    df = pd.read_parquet(path)

    actions = stack_feature(df["action"])
    if actions.shape[1] != len(EVORL_NERO_ACTION_NAMES):
        raise ValueError(f"{path}: expected action width 14, got {actions.shape[1]}")
    assign_vector_feature(df, "action", actions)

    states = stack_feature(df["observation.state"])
    if states.shape[1] != len(EVORL_NERO_STATE_NAMES):
        raise ValueError(f"{path}: expected observation.state width 28, got {states.shape[1]}")
    assign_vector_feature(df, "observation.state", states)

    num_frames = len(df)
    assign_vector_feature(
        df,
        "complementary_info.policy_action",
        np.zeros((num_frames, len(EVORL_NERO_ACTION_NAMES)), dtype=np.float32),
    )
    # LeRobot stores shape-[1] features as scalar parquet columns.
    df["complementary_info.is_intervention"] = np.zeros(num_frames, dtype=np.float32)
    df["complementary_info.state"] = np.zeros(num_frames, dtype=np.float32)

    if not keep_legacy_dagger_fields:
        df = df.drop(columns=[col for col in LEGACY_DAGGER_FEATURES if col in df.columns])

    pq.write_table(
        pa_table_from_pandas(df),
        path,
        compression="snappy",
        use_dictionary=True,
    )


def pa_table_from_pandas(df: pd.DataFrame):
    # Delay importing pyarrow until write time to keep the top-level namespace small.
    import pyarrow as pa

    return pa.Table.from_pandas(df, preserve_index=False)


def strip_legacy_episode_stats(episodes_path: Path, keep_legacy_dagger_fields: bool) -> None:
    if not episodes_path.exists():
        return

    df = pd.read_parquet(episodes_path)
    if not keep_legacy_dagger_fields:
        prefixes = tuple(f"stats/{name}/" for name in LEGACY_DAGGER_FEATURES)
        df = df.drop(columns=[col for col in df.columns if col.startswith(prefixes)])

    for feature_name, width in [
        ("complementary_info.policy_action", len(EVORL_NERO_ACTION_NAMES)),
        ("complementary_info.is_intervention", 1),
        ("complementary_info.state", 1),
    ]:
        for stat_name in ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]:
            df[f"stats/{feature_name}/{stat_name}"] = [
                np.zeros(width, dtype=np.float32) for _ in range(len(df))
            ]
        df[f"stats/{feature_name}/count"] = [
            np.array([int(length)], dtype=np.int64) for length in df["length"].to_list()
        ]

    pq.write_table(pa_table_from_pandas(df), episodes_path, compression="snappy", use_dictionary=True)


def compute_numeric_stats(dataset_root: Path, features: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    columns: dict[str, list[np.ndarray]] = {}
    for data_path in sorted((dataset_root / "data").glob("chunk-*/file-*.parquet")):
        df = pd.read_parquet(data_path)
        for key, feature in features.items():
            if feature["dtype"] in {"video", "image", "string"} or key not in df.columns:
                continue
            is_scalar_feature = feature_shape({"features": features}, key) == (1,)
            if is_scalar_feature and not isinstance(df[key].iloc[0], np.ndarray):
                values = df[key].to_numpy(dtype=np.float32).reshape(-1, 1)
            else:
                values = stack_feature(df[key])
            columns.setdefault(key, []).append(values)

    stats = {}
    for key, chunks in columns.items():
        values = np.concatenate(chunks, axis=0)
        stats[key] = get_feature_stats(values, axis=0, keepdims=False)
    return stats


def merge_video_stats(
    src_root: Path,
    stats: dict[str, dict[str, np.ndarray]],
    features: dict[str, Any],
) -> None:
    old_stats_path = src_root / "meta" / "stats.json"
    if not old_stats_path.exists():
        return

    with old_stats_path.open() as f:
        old_stats = json.load(f)
    for key, feature in features.items():
        if feature["dtype"] in {"video", "image"} and key in old_stats:
            stats[key] = {stat_name: np.asarray(value) for stat_name, value in old_stats[key].items()}


def convert_dataset(args: argparse.Namespace) -> Path:
    src_root = resolve_dataset_root(args.dataset, args.root)
    output_repo_id = infer_output_repo_id(src_root, args.dataset, args.output_repo_id)
    dst_root = resolve_output_root(output_repo_id, args.output_dir)

    src_info = load_info(src_root)
    validate_source_schema(src_root, src_info)

    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {dst_root}. Pass --overwrite to replace it.")
        shutil.rmtree(dst_root)

    logging.info("Copying %s -> %s", src_root, dst_root)
    shutil.copytree(src_root, dst_root)

    new_info = dict(src_info)
    new_info["features"] = normalize_features(src_info, args.keep_legacy_dagger_fields)
    write_info(dst_root, new_info)

    for data_path in sorted((dst_root / "data").glob("chunk-*/file-*.parquet")):
        logging.info("Converting data file %s", data_path)
        transform_data_file(data_path, args.keep_legacy_dagger_fields)

    for episodes_path in sorted((dst_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        logging.info("Patching episode metadata %s", episodes_path)
        strip_legacy_episode_stats(episodes_path, args.keep_legacy_dagger_fields)

    logging.info("Recomputing numeric stats")
    stats = compute_numeric_stats(dst_root, new_info["features"])
    merge_video_stats(src_root, stats, new_info["features"])
    write_stats(stats, dst_root)

    logging.info("Converted dataset written to %s", dst_root)
    return dst_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Source dataset repo_id or local dataset path.")
    parser.add_argument("--root", default=None, help="Base LeRobot cache root. Defaults to HF_LEROBOT_HOME.")
    parser.add_argument("--output-repo-id", default=None, help="Output repo_id. Defaults to <dataset>_evorl.")
    parser.add_argument("--output-dir", default=None, help="Explicit output directory.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory if it already exists.",
    )
    parser.add_argument(
        "--keep-legacy-dagger-fields",
        action="store_true",
        help="Keep legacy action_source/is_expert columns instead of dropping them.",
    )
    return parser


def main() -> None:
    init_logging()
    args = build_parser().parse_args()
    convert_dataset(args)


if __name__ == "__main__":
    main()
