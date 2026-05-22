#!/usr/bin/env python
"""Evaluate a NERO PI05 policy against dataset actions on one task.

This is a single-task variant of ``lerobot_policy_action_probe``. It samples
frames from a LeRobot dataset, runs the policy on the recorded observations,
and compares the first predicted action with the dataset action at the same
frame.

For NERO dual-arm actions the assumed layout is:
  0:6   left end-effector delta
  6:12  right end-effector delta
  12    left gripper
  13    right gripper
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.scripts.lerobot_policy_action_probe import (
    _load_predictions,
    _parse_motion_dims,
    _read_frame_index,
    _read_tasks,
    _sample_indices,
    _summarize,
    _summarize_arm_usage,
)


LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--pretrained-path", required=True, type=Path)
    parser.add_argument(
        "--task-substring",
        default=None,
        help="Substring used to choose the task from meta/tasks.parquet. Defaults to the first task.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt override. Defaults to the dataset prompt for the selected task.",
    )
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-active-frames", action="store_true")
    parser.add_argument(
        "--active-selection",
        choices=("top", "first-per-episode", "random"),
        default="first-per-episode",
    )
    parser.add_argument("--frames-per-episode", type=int, default=2)
    parser.add_argument(
        "--motion-dims",
        default="6:12",
        help="Dims used to rank active frames. For small-vial right-arm tasks use 6:12.",
    )
    parser.add_argument("--min-motion-norm", type=float, default=0.001)
    parser.add_argument("--episode-min", type=int, default=None)
    parser.add_argument("--episode-max", type=int, default=None)
    parser.add_argument(
        "--expected-arm",
        choices=("left", "right"),
        default="right",
        help="Expected active arm for arm-usage diagnostics.",
    )
    parser.add_argument("--arm-active-threshold", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--n-action-steps", type=int, default=30)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _task_from_table(tasks: pd.DataFrame, substring: str | None) -> tuple[int, str]:
    if "task_index" not in tasks.columns:
        raise ValueError(f"Unexpected tasks.parquet columns: {list(tasks.columns)}")

    def row_task(row: pd.Series, fallback_index) -> str:
        if "task" in tasks.columns:
            return str(row["task"])
        return str(fallback_index)

    if substring is None:
        row = tasks.iloc[0]
        return int(row["task_index"]), row_task(row, tasks.index[0])

    needle = substring.lower()
    for index, row in tasks.iterrows():
        task = row_task(row, index)
        if needle in task.lower():
            return int(row["task_index"]), task

    available = "\n".join(row_task(row, index) for index, row in tasks.iterrows())
    raise ValueError(f"Could not find task containing {substring!r}. Available tasks:\n{available}")


def _filter_episodes(frame_table: pd.DataFrame, episode_min: int | None, episode_max: int | None) -> pd.DataFrame:
    filtered = frame_table
    if episode_min is not None:
        filtered = filtered.loc[filtered["episode_index"] >= episode_min]
    if episode_max is not None:
        filtered = filtered.loc[filtered["episode_index"] <= episode_max]
    if len(filtered) == 0:
        raise ValueError(f"No frames left after episode filter: min={episode_min}, max={episode_max}")
    return filtered


def _part_metrics(pred: np.ndarray, target: np.ndarray, dims: slice) -> dict:
    pred_part = pred[:, dims]
    target_part = target[:, dims]
    diff = pred_part - target_part
    error_l2 = np.linalg.norm(diff, axis=1)
    target_l2 = np.linalg.norm(target_part, axis=1)
    pred_l2 = np.linalg.norm(pred_part, axis=1)
    return {
        "target_l2_mean": float(target_l2.mean()),
        "target_l2_median": float(np.median(target_l2)),
        "pred_l2_mean": float(pred_l2.mean()),
        "pred_l2_median": float(np.median(pred_l2)),
        "error_l2_mean": float(error_l2.mean()),
        "error_l2_median": float(np.median(error_l2)),
        "error_l2_p90": float(np.percentile(error_l2, 90)),
        "relative_error_mean": float(error_l2.mean() / (target_l2.mean() + 1e-8)),
        "mae_per_dim": [float(x) for x in np.abs(diff).mean(axis=0)],
    }


def _gripper_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    if pred.shape[1] < 14:
        return {}
    left_diff = pred[:, 12] - target[:, 12]
    right_diff = pred[:, 13] - target[:, 13]
    return {
        "left_gripper_target_mean": float(target[:, 12].mean()),
        "left_gripper_pred_mean": float(pred[:, 12].mean()),
        "left_gripper_mae": float(np.abs(left_diff).mean()),
        "right_gripper_target_mean": float(target[:, 13].mean()),
        "right_gripper_pred_mean": float(pred[:, 13].mean()),
        "right_gripper_mae": float(np.abs(right_diff).mean()),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    tasks = _read_tasks(args.dataset_root)
    task_index, dataset_prompt = _task_from_table(tasks, args.task_substring)
    prompt = args.prompt if args.prompt is not None else dataset_prompt
    motion_dims = _parse_motion_dims(args.motion_dims)

    frame_table = _read_frame_index(args.dataset_root, include_action=args.sample_active_frames)
    frame_table = _filter_episodes(frame_table, args.episode_min, args.episode_max)
    indices = _sample_indices(
        frame_table=frame_table,
        task_index=task_index,
        limit=args.limit,
        seed=args.seed,
        sample_episode_starts=False,
        start_frames_per_episode=1,
        start_frame_offset=0,
        sample_active_frames=args.sample_active_frames,
        active_selection=args.active_selection,
        frames_per_episode=args.frames_per_episode,
        motion_dims=motion_dims,
        min_motion_norm=args.min_motion_norm,
    )
    LOG.info("Selected task_index=%s prompt=%r", task_index, dataset_prompt)
    LOG.info("Sampled %s frames from episodes [%s, %s]", len(indices), args.episode_min, args.episode_max)

    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        video_backend=args.video_backend,
    )
    policy_cfg = PI05Config(
        pretrained_path=args.pretrained_path,
        device=args.device,
        dtype=args.dtype,
        num_inference_steps=args.num_inference_steps,
        n_action_steps=args.n_action_steps,
    )
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.pretrained_path,
        dataset_stats=dataset.meta.stats,
    )

    outputs = _load_predictions(
        dataset=dataset,
        indices=indices,
        batch_size=args.batch_size,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        prompts={"dataset_prompt": prompt},
    )
    target = outputs["target"]
    pred = outputs["predictions"]["dataset_prompt"]

    metrics = {
        "all_action": _summarize(pred, target),
        "left_delta": _part_metrics(pred, target, slice(0, 6)),
        "right_delta": _part_metrics(pred, target, slice(6, 12)),
        "gripper": _gripper_metrics(pred, target),
        "target_arm_usage": _summarize_arm_usage(
            target, expected_arm=args.expected_arm, threshold=args.arm_active_threshold
        ),
        "pred_arm_usage": _summarize_arm_usage(
            pred, expected_arm=args.expected_arm, threshold=args.arm_active_threshold
        ),
    }

    result = {
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "pretrained_path": str(args.pretrained_path),
        "task_index": int(task_index),
        "dataset_prompt": dataset_prompt,
        "used_prompt": prompt,
        "num_sampled_frames": int(len(indices)),
        "sampled_indices_first10": [int(x) for x in indices[:10]],
        "episode_min": args.episode_min,
        "episode_max": args.episode_max,
        "sample_active_frames": bool(args.sample_active_frames),
        "active_selection": args.active_selection,
        "frames_per_episode": int(args.frames_per_episode),
        "motion_dims": motion_dims,
        "min_motion_norm": float(args.min_motion_norm),
        "expected_arm": args.expected_arm,
        "num_inference_steps": int(args.num_inference_steps),
        "n_action_steps": int(args.n_action_steps),
        "metrics": metrics,
    }

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
        LOG.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()
