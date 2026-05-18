#!/usr/bin/env python
"""Offline policy-vs-dataset action probe.

This diagnostic samples frames from a LeRobot dataset, runs a policy on the
observations, and compares the predicted first action against the dataset
action at the same frame. It is intended for quick policy sanity checks, e.g.
testing whether a multi-task checkpoint follows the requested task prompt.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config


LOG = logging.getLogger(__name__)

ACTION_KEY = "action"
TASK_KEY = "task"


PROMPT_PRESETS = {
    "short": {
        "2ml": "pick up small vials and place them in the empty rack",
        "8ml": "pick up big vials and place them in the empty rack",
    },
    "prompt_v1": {
        "2ml": (
            "Task type: 2mL_two_vials_right_column. From the robot front-camera view, "
            "pick up two 2 mL small vials one by one and place them upright into the "
            "strict corner holes of the rightmost column of the empty rack: one vial "
            "in the bottom-right hole and one vial in the top-right hole."
        ),
        "8ml": (
            "Task type: 8mL_two_vials_left_column. From the robot front-camera view, "
            "pick up two 8 mL large vials one by one and place them upright into the "
            "strict corner holes of the leftmost column of the empty rack: one vial "
            "in the bottom-left hole and one vial in the top-left hole."
        ),
    },
    "prompt_v2": {
        "2ml": (
            "Task: 2mL_two_vials_right_column.\n"
            "Object: two small 2 mL vials.\n"
            "Goal: place the two vials upright into the rightmost column of the empty rack.\n"
            "Targets: bottom-right corner hole and top-right corner hole only.\n"
            "Do not place either vial into any other hole, row, or column."
        ),
        "8ml": (
            "Task: 8mL_two_vials_left_column.\n"
            "Object: two large 8 mL vials.\n"
            "Goal: place the two vials upright into the leftmost column of the empty rack.\n"
            "Targets: bottom-left corner hole and top-left corner hole only.\n"
            "Do not place either vial into any other hole, row, or column."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo_id.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="LeRobot dataset root directory.")
    parser.add_argument("--pretrained-path", required=True, type=Path, help="Policy pretrained_model directory.")
    parser.add_argument("--small-task-substring", default="2mL_two_vials_right_column")
    parser.add_argument("--big-task-substring", default="8mL_two_vials_left_column")
    parser.add_argument(
        "--prompt-styles",
        default="prompt_v1,prompt_v2,dataset",
        help="Comma-separated prompt styles: short,prompt_v1,prompt_v2,dataset.",
    )
    parser.add_argument("--limit", type=int, default=32, help="Frames per task to evaluate.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--sample-active-frames",
        action="store_true",
        help="Sample frames with the largest non-gripper action norm instead of random frames.",
    )
    parser.add_argument(
        "--sample-episode-starts",
        action="store_true",
        help="Sample the first frames of each episode. This is useful for checking rollout task selection.",
    )
    parser.add_argument("--start-frames-per-episode", type=int, default=1)
    parser.add_argument("--start-frame-offset", type=int, default=0)
    parser.add_argument(
        "--active-selection",
        choices=("top", "first-per-episode", "random"),
        default="top",
        help="How to choose frames after --sample-active-frames filtering.",
    )
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=1,
        help="Frames selected per episode when --active-selection=first-per-episode.",
    )
    parser.add_argument(
        "--motion-dims",
        default="0:12",
        help="Action dimensions used to rank active frames. NERO uses 0:6 left delta, 6:12 right delta.",
    )
    parser.add_argument(
        "--min-motion-norm",
        type=float,
        default=0.0,
        help="Minimum non-gripper action norm when --sample-active-frames is enabled.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--arm-active-threshold", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _read_tasks(root: Path) -> pd.DataFrame:
    path = root / "meta" / "tasks.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing tasks parquet: {path}")
    return pd.read_parquet(path)


def _find_task(tasks: pd.DataFrame, substring: str) -> tuple[int, str]:
    substring_lower = substring.lower()
    task_column = "task" if "task" in tasks.columns else None
    for index, row in tasks.iterrows():
        task = str(row[task_column]) if task_column is not None else str(index)
        if substring_lower in task.lower():
            return int(row["task_index"]), task
    if task_column is not None:
        available_tasks = tasks[task_column].tolist()
    else:
        available_tasks = tasks.index.tolist()
    available = "\n".join(str(task) for task in available_tasks)
    raise ValueError(f"Could not find task containing {substring!r}. Available tasks:\n{available}")


def _parse_motion_dims(spec: str) -> list[int]:
    dims: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start, end = item.split(":", maxsplit=1)
            dims.extend(range(int(start), int(end)))
        else:
            dims.append(int(item))
    if not dims:
        raise ValueError("--motion-dims must select at least one action dimension")
    return dims


def _read_frame_index(root: Path, include_action: bool) -> pd.DataFrame:
    data_dir = root / "data"
    files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not files:
        files = sorted(data_dir.glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquet files found under {data_dir}")

    columns = ["index", "episode_index", "frame_index", "task_index"]
    if include_action:
        columns.append("action")

    frames = []
    for file in files:
        frames.append(pd.read_parquet(file, columns=columns))
    return pd.concat(frames, ignore_index=True)


def _action_motion_norm(action, dims: list[int]) -> float:
    array = np.asarray(action, dtype=np.float32)
    valid_dims = [dim for dim in dims if dim < array.shape[0]]
    if not valid_dims:
        return 0.0
    return float(np.linalg.norm(array[valid_dims]))


def _sample_indices(
    frame_table: pd.DataFrame,
    task_index: int,
    limit: int,
    seed: int,
    sample_episode_starts: bool,
    start_frames_per_episode: int,
    start_frame_offset: int,
    sample_active_frames: bool,
    active_selection: str,
    frames_per_episode: int,
    motion_dims: list[int],
    min_motion_norm: float,
) -> np.ndarray:
    task_rows = frame_table.loc[frame_table["task_index"] == task_index].copy()
    if len(task_rows) == 0:
        raise ValueError(f"No frames found for task_index={task_index}")

    if sample_episode_starts:
        selected = (
            task_rows.loc[task_rows["frame_index"] >= start_frame_offset]
            .sort_values(["episode_index", "frame_index"])
            .groupby("episode_index", sort=False)
            .head(start_frames_per_episode)
        )
        if len(selected) > limit:
            selected = selected.head(limit)
        return np.sort(selected["index"].to_numpy().astype(int))

    if sample_active_frames:
        if "action" not in task_rows.columns:
            raise ValueError("--sample-active-frames requires the frame table to include the action column")
        task_rows["_motion_norm"] = task_rows["action"].map(lambda action: _action_motion_norm(action, motion_dims))
        task_rows = task_rows.loc[task_rows["_motion_norm"] >= min_motion_norm]
        if len(task_rows) == 0:
            raise ValueError(
                f"No active frames left for task_index={task_index} with min_motion_norm={min_motion_norm}"
            )

        if active_selection == "top":
            selected = task_rows.sort_values("_motion_norm", ascending=False).head(limit)
        elif active_selection == "first-per-episode":
            selected = (
                task_rows.sort_values(["episode_index", "frame_index"])
                .groupby("episode_index", sort=False)
                .head(frames_per_episode)
            )
            if len(selected) > limit:
                selected = selected.head(limit)
        else:
            rng = np.random.default_rng(seed)
            selected = task_rows.sample(n=min(limit, len(task_rows)), random_state=rng)
        return np.sort(selected["index"].to_numpy().astype(int))

    task_frames = task_rows["index"].to_numpy()
    if len(task_frames) <= limit:
        return task_frames.astype(int)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(task_frames, size=limit, replace=False)).astype(int)


def _batched(values: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _extract_action(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict) and ACTION_KEY in value:
        return value[ACTION_KEY]
    raise TypeError(f"Unsupported postprocessor output type: {type(value)!r}")


def _with_prompt(batch: dict, prompt: str) -> dict:
    copied = dict(batch)
    batch_len = int(batch[ACTION_KEY].shape[0])
    copied[TASK_KEY] = [prompt] * batch_len
    return copied


@torch.inference_mode()
def _predict_first_action(policy, preprocessor, postprocessor, raw_batch: dict, prompt: str) -> torch.Tensor:
    processed = preprocessor(_with_prompt(raw_batch, prompt))
    action_chunk = policy.predict_action_chunk(processed)
    first_action = action_chunk[:, 0]
    return _extract_action(postprocessor(first_action))


def _summarize(pred: np.ndarray, target: np.ndarray) -> dict:
    diff = pred - target
    l2 = np.linalg.norm(diff, axis=1)
    target_l2 = np.linalg.norm(target, axis=1)
    return {
        "num_frames": int(pred.shape[0]),
        "action_dim": int(pred.shape[1]),
        "mae_all_dims": float(np.abs(diff).mean()),
        "rmse_all_dims": float(np.sqrt(np.square(diff).mean())),
        "l2_mean": float(l2.mean()),
        "l2_median": float(np.median(l2)),
        "l2_p90": float(np.percentile(l2, 90)),
        "target_l2_mean": float(target_l2.mean()),
        "relative_l2_mean": float(l2.mean() / (target_l2.mean() + 1e-8)),
        "per_dim_mae": [float(x) for x in np.abs(diff).mean(axis=0)],
        "pred_mean": [float(x) for x in pred.mean(axis=0)],
        "target_mean": [float(x) for x in target.mean(axis=0)],
    }


def _summarize_delta(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    l2 = np.linalg.norm(diff, axis=1)
    return {
        "num_frames": int(a.shape[0]),
        "mae_all_dims": float(np.abs(diff).mean()),
        "l2_mean": float(l2.mean()),
        "l2_median": float(np.median(l2)),
        "l2_p90": float(np.percentile(l2, 90)),
        "per_dim_mae": [float(x) for x in np.abs(diff).mean(axis=0)],
    }


def _summarize_arm_usage(actions: np.ndarray, expected_arm: str, threshold: float) -> dict:
    if actions.shape[1] < 14:
        raise ValueError(f"Arm usage summary expects at least 14 action dims, got {actions.shape[1]}")

    # NERO action layout:
    # 0:6 left_delta_ee_pose, 6:12 right_delta_ee_pose, 12 left gripper, 13 right gripper.
    left_norm = np.linalg.norm(actions[:, 0:6], axis=1)
    right_norm = np.linalg.norm(actions[:, 6:12], axis=1)
    if expected_arm == "left":
        correct_norm = left_norm
        wrong_norm = right_norm
    elif expected_arm == "right":
        correct_norm = right_norm
        wrong_norm = left_norm
    else:
        raise ValueError(f"expected_arm must be 'left' or 'right', got {expected_arm!r}")

    return {
        "expected_arm": expected_arm,
        "left_motion_l2_mean": float(left_norm.mean()),
        "left_motion_l2_p90": float(np.percentile(left_norm, 90)),
        "right_motion_l2_mean": float(right_norm.mean()),
        "right_motion_l2_p90": float(np.percentile(right_norm, 90)),
        "correct_arm_l2_mean": float(correct_norm.mean()),
        "wrong_arm_l2_mean": float(wrong_norm.mean()),
        "wrong_over_correct_l2_mean": float(np.mean(wrong_norm / (correct_norm + 1e-8))),
        "correct_dominant_rate": float(np.mean(correct_norm > wrong_norm)),
        "wrong_dominant_rate": float(np.mean(wrong_norm > correct_norm)),
        "correct_active_rate": float(np.mean(correct_norm > threshold)),
        "wrong_active_rate": float(np.mean(wrong_norm > threshold)),
        "left_gripper_mean": float(actions[:, 12].mean()),
        "right_gripper_mean": float(actions[:, 13].mean()),
    }


def _load_predictions(dataset, indices: np.ndarray, batch_size: int, policy, preprocessor, postprocessor, prompts: dict) -> dict:
    all_targets = []
    pred_by_prompt = {name: [] for name in prompts}

    for batch_indices in _batched(indices, batch_size):
        batch = default_collate([dataset[int(index)] for index in batch_indices])
        target = batch[ACTION_KEY]
        all_targets.append(_to_numpy(target))

        for name, prompt in prompts.items():
            pred = _predict_first_action(policy, preprocessor, postprocessor, batch, prompt)
            pred_by_prompt[name].append(_to_numpy(pred))

    return {
        "target": np.concatenate(all_targets, axis=0),
        "predictions": {name: np.concatenate(chunks, axis=0) for name, chunks in pred_by_prompt.items()},
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    prompt_styles = [style.strip() for style in args.prompt_styles.split(",") if style.strip()]
    unknown = [style for style in prompt_styles if style not in {"dataset", *PROMPT_PRESETS.keys()}]
    if unknown:
        raise ValueError(f"Unknown prompt style(s): {unknown}")

    tasks = _read_tasks(args.dataset_root)
    small_task_index, small_dataset_prompt = _find_task(tasks, args.small_task_substring)
    big_task_index, big_dataset_prompt = _find_task(tasks, args.big_task_substring)
    motion_dims = _parse_motion_dims(args.motion_dims)
    frame_table = _read_frame_index(args.dataset_root, include_action=args.sample_active_frames)

    small_indices = _sample_indices(
        frame_table,
        small_task_index,
        args.limit,
        args.seed,
        args.sample_episode_starts,
        args.start_frames_per_episode,
        args.start_frame_offset,
        args.sample_active_frames,
        args.active_selection,
        args.frames_per_episode,
        motion_dims,
        args.min_motion_norm,
    )
    big_indices = _sample_indices(
        frame_table,
        big_task_index,
        args.limit,
        args.seed + 1,
        args.sample_episode_starts,
        args.start_frames_per_episode,
        args.start_frame_offset,
        args.sample_active_frames,
        args.active_selection,
        args.frames_per_episode,
        motion_dims,
        args.min_motion_norm,
    )

    LOG.info("Small task_index=%s prompt=%r sampled_frames=%s", small_task_index, small_dataset_prompt, len(small_indices))
    LOG.info("Big task_index=%s prompt=%r sampled_frames=%s", big_task_index, big_dataset_prompt, len(big_indices))

    dataset = LeRobotDataset(repo_id=args.dataset_repo_id, root=args.dataset_root)
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

    results = {
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "pretrained_path": str(args.pretrained_path),
        "limit_per_task": int(args.limit),
        "batch_size": int(args.batch_size),
        "num_inference_steps": int(args.num_inference_steps),
        "sample_episode_starts": bool(args.sample_episode_starts),
        "start_frames_per_episode": int(args.start_frames_per_episode),
        "start_frame_offset": int(args.start_frame_offset),
        "sample_active_frames": bool(args.sample_active_frames),
        "active_selection": args.active_selection,
        "frames_per_episode": int(args.frames_per_episode),
        "motion_dims": motion_dims,
        "min_motion_norm": float(args.min_motion_norm),
        "arm_active_threshold": float(args.arm_active_threshold),
        "tasks": {
            "small": {"task_index": small_task_index, "dataset_prompt": small_dataset_prompt},
            "big": {"task_index": big_task_index, "dataset_prompt": big_dataset_prompt},
        },
        "metrics": {},
    }

    for style in prompt_styles:
        if style == "dataset":
            prompts = {"small_prompt": small_dataset_prompt, "big_prompt": big_dataset_prompt}
        else:
            prompts = {"small_prompt": PROMPT_PRESETS[style]["2ml"], "big_prompt": PROMPT_PRESETS[style]["8ml"]}

        LOG.info("Evaluating prompt style: %s", style)
        small_eval = _load_predictions(
            dataset, small_indices, args.batch_size, policy, preprocessor, postprocessor, prompts
        )
        big_eval = _load_predictions(
            dataset, big_indices, args.batch_size, policy, preprocessor, postprocessor, prompts
        )

        style_metrics = {
            "small_frames_with_small_prompt_vs_dataset_action": _summarize(
                small_eval["predictions"]["small_prompt"], small_eval["target"]
            ),
            "small_frames_with_big_prompt_vs_dataset_action": _summarize(
                small_eval["predictions"]["big_prompt"], small_eval["target"]
            ),
            "small_frames_prompt_sensitivity": _summarize_delta(
                small_eval["predictions"]["small_prompt"], small_eval["predictions"]["big_prompt"]
            ),
            "big_frames_with_big_prompt_vs_dataset_action": _summarize(
                big_eval["predictions"]["big_prompt"], big_eval["target"]
            ),
            "big_frames_with_small_prompt_vs_dataset_action": _summarize(
                big_eval["predictions"]["small_prompt"], big_eval["target"]
            ),
            "big_frames_prompt_sensitivity": _summarize_delta(
                big_eval["predictions"]["small_prompt"], big_eval["predictions"]["big_prompt"]
            ),
            "small_frames_dataset_action_arm_usage": _summarize_arm_usage(
                small_eval["target"], expected_arm="right", threshold=args.arm_active_threshold
            ),
            "small_frames_small_prompt_pred_arm_usage": _summarize_arm_usage(
                small_eval["predictions"]["small_prompt"], expected_arm="right", threshold=args.arm_active_threshold
            ),
            "small_frames_big_prompt_pred_arm_usage": _summarize_arm_usage(
                small_eval["predictions"]["big_prompt"], expected_arm="right", threshold=args.arm_active_threshold
            ),
            "big_frames_dataset_action_arm_usage": _summarize_arm_usage(
                big_eval["target"], expected_arm="left", threshold=args.arm_active_threshold
            ),
            "big_frames_big_prompt_pred_arm_usage": _summarize_arm_usage(
                big_eval["predictions"]["big_prompt"], expected_arm="left", threshold=args.arm_active_threshold
            ),
            "big_frames_small_prompt_pred_arm_usage": _summarize_arm_usage(
                big_eval["predictions"]["small_prompt"], expected_arm="left", threshold=args.arm_active_threshold
            ),
        }
        results["metrics"][style] = style_metrics

        LOG.info(
            "[%s] small prompt error l2=%.4f, big prompt on small frames l2=%.4f, prompt delta l2=%.4f",
            style,
            style_metrics["small_frames_with_small_prompt_vs_dataset_action"]["l2_mean"],
            style_metrics["small_frames_with_big_prompt_vs_dataset_action"]["l2_mean"],
            style_metrics["small_frames_prompt_sensitivity"]["l2_mean"],
        )
        LOG.info(
            "[%s] big prompt error on big frames l2=%.4f, small prompt on big frames l2=%.4f, prompt delta l2=%.4f",
            style,
            style_metrics["big_frames_with_big_prompt_vs_dataset_action"]["l2_mean"],
            style_metrics["big_frames_with_small_prompt_vs_dataset_action"]["l2_mean"],
            style_metrics["big_frames_prompt_sensitivity"]["l2_mean"],
        )

    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
        LOG.info("Wrote results to %s", args.output_json)


if __name__ == "__main__":
    main()
