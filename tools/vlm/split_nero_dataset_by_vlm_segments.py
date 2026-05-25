#!/usr/bin/env python3
"""Split a NERO LeRobot dataset using VLM-produced segment timestamps.

The VLM annotations are expected to contain seconds relative to each review
clip. This script preserves the source timestamps/frame indices so copied
source videos remain aligned after zero-motion frame removal.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEGMENT_TASKS = {
    "grasp_first_vial": {
        "dir": "seg1_grasp_first_vial",
        "prompt": (
            "Use the right gripper to grasp and lift the first 2 mL vial from the table. "
            "Stop once the vial is securely held above the table."
        ),
    },
    "insert_first_vial": {
        "dir": "seg2_insert_first_vial_bottom_right",
        "prompt": (
            "Starting with the first 2 mL vial already held by the right gripper, "
            "place it upright into the bottom-right hole of the rack. "
            "Stop once the vial is released and stable in the hole."
        ),
    },
    "grasp_second_vial": {
        "dir": "seg3_grasp_remaining_vial",
        "prompt": (
            "Starting with the first vial already placed in the bottom-right hole, "
            "use the right gripper to grasp and lift the remaining 2 mL vial from the table. "
            "Stop once the remaining vial is securely held above the table."
        ),
    },
    "insert_second_vial": {
        "dir": "seg4_insert_remaining_vial_top_right",
        "prompt": (
            "Starting with the remaining 2 mL vial already held by the right gripper, "
            "place it upright into the top-right hole of the rack. "
            "Stop once the vial is released and stable in the hole."
        ),
    },
}

QUANTILE_KEYS = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


@dataclass(frozen=True)
class SegmentRow:
    segment_key: str
    new_episode_index: int
    source_episode_index: int
    original_episode_index: int
    start_time: float
    end_time: float
    confidence: float
    evidence: str
    frames: int


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def _copy_or_link_videos(src_root: Path, dst_root: Path, mode: str) -> None:
    src = src_root / "videos"
    dst = dst_root / "videos"
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=True)
    elif mode == "copy":
        shutil.copytree(src, dst)
    else:
        raise ValueError(f"Unsupported video mode: {mode}")


def _copy_static_metadata(src_root: Path, dst_root: Path) -> None:
    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"
    if dst_meta.exists():
        shutil.rmtree(dst_meta)
    dst_meta.mkdir(parents=True, exist_ok=True)
    for item in src_meta.iterdir():
        if item.name in {"info.json", "stats.json", "episodes", "tasks.parquet"}:
            continue
        dst = dst_meta / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)


def _task_table(prompt: str) -> pd.DataFrame:
    return pd.DataFrame({"task_index": [0]}, index=pd.Index([prompt], name=None))


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
    stats: dict[str, Any] = {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
        "count": np.asarray([arr.shape[0]], dtype=np.int64),
    }
    for key, q in QUANTILE_KEYS.items():
        stats[key] = np.quantile(arr, q, axis=0)
    return {key: np.asarray(value).tolist() for key, value in stats.items()}


def _compute_stats(data_df: pd.DataFrame, src_stats: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key, value in src_stats.items():
        if key.startswith("observation.images."):
            stats[key] = value
            continue
        if key not in data_df.columns or len(data_df) == 0:
            continue
        try:
            stats[key] = _feature_stats(_stack_feature(data_df[key]))
        except Exception:
            continue
    return stats


def _load_annotation(path: Path) -> dict[str, Any] | None:
    try:
        record = _load_json(path)
    except Exception:
        return None
    if record.get("status") != "ok":
        return None
    annotation = record.get("annotation")
    return annotation if isinstance(annotation, dict) else None


def _extract_segments(
    annotations_dir: Path,
    episodes_df: pd.DataFrame,
    *,
    pre_margin_s: float,
    post_margin_s: float,
    min_confidence: float,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_key: dict[str, list[dict[str, Any]]] = {key: [] for key in SEGMENT_TASKS}
    failures: list[dict[str, Any]] = []
    episode_original = {
        int(row["episode_index"]): int(row.get("original_episode_index", row["episode_index"]))
        for _, row in episodes_df.iterrows()
    }
    per_episode = annotations_dir / "per_episode"
    for ep_idx in sorted(episode_original):
        path = per_episode / f"episode_{ep_idx:03d}_segments.json"
        annotation = _load_annotation(path)
        if annotation is None:
            failures.append({"episode_index": ep_idx, "reason": "missing_or_failed_annotation", "path": str(path)})
            continue
        segments = annotation.get("segments", {})
        for segment_key in SEGMENT_TASKS:
            segment = segments.get(segment_key, {})
            start = segment.get("start_time")
            end = segment.get("end_time")
            confidence = float(segment.get("confidence", 0.0) or 0.0)
            if start is None or end is None or confidence < min_confidence or float(end) <= float(start):
                failures.append(
                    {
                        "episode_index": ep_idx,
                        "segment_key": segment_key,
                        "reason": "invalid_or_low_confidence_segment",
                        "start_time": start,
                        "end_time": end,
                        "confidence": confidence,
                    }
                )
                continue
            by_key[segment_key].append(
                {
                    "source_episode_index": ep_idx,
                    "original_episode_index": episode_original[ep_idx],
                    "start_time": max(0.0, float(start) - pre_margin_s),
                    "end_time": float(end) + post_margin_s,
                    "confidence": confidence,
                    "evidence": str(segment.get("evidence", "")),
                }
            )
    return by_key, failures


def _build_one_dataset(
    *,
    src_root: Path,
    output_root: Path,
    segment_key: str,
    specs: list[dict[str, Any]],
    src_data: pd.DataFrame,
    src_episodes: pd.DataFrame,
    src_info: dict[str, Any],
    src_stats: dict[str, Any],
    video_mode: str,
    replace: bool,
    min_frames: int,
) -> dict[str, Any]:
    cfg = SEGMENT_TASKS[segment_key]
    dst_root = output_root / cfg["dir"]
    if dst_root.exists():
        if not replace:
            raise FileExistsError(f"{dst_root} exists; pass --replace")
        shutil.rmtree(dst_root)

    _copy_static_metadata(src_root, dst_root)
    (dst_root / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (dst_root / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    _copy_or_link_videos(src_root, dst_root, video_mode)

    src_by_ep = {int(k): v for k, v in src_data.groupby("episode_index", sort=False)}
    ep_by_id = {int(row["episode_index"]): row for _, row in src_episodes.iterrows()}
    video_cols = [c for c in src_episodes.columns if c.startswith("videos/")]

    data_parts: list[pd.DataFrame] = []
    episode_rows: list[dict[str, Any]] = []
    segment_rows: list[SegmentRow] = []
    skipped: list[dict[str, Any]] = []
    global_index = 0

    for spec in specs:
        ep_idx = int(spec["source_episode_index"])
        ep_data = src_by_ep[ep_idx]
        mask = (ep_data["timestamp"].astype(float) >= float(spec["start_time"])) & (
            ep_data["timestamp"].astype(float) <= float(spec["end_time"])
        )
        seg_data = ep_data.loc[mask].copy()
        if len(seg_data) < min_frames:
            skipped.append({**spec, "reason": "too_few_frames", "frames": int(len(seg_data))})
            continue

        new_ep = len(episode_rows)
        seg_len = int(len(seg_data))
        seg_data["episode_index"] = new_ep
        seg_data["index"] = np.arange(global_index, global_index + seg_len, dtype=np.int64)
        seg_data["task_index"] = 0
        data_parts.append(seg_data)

        src_row = ep_by_id[ep_idx]
        ep_row: dict[str, Any] = {
            "episode_index": new_ep,
            "tasks": np.array([cfg["prompt"]], dtype=object),
            "length": seg_len,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": global_index,
            "dataset_to_index": global_index + seg_len,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "episode_success": "success",
            "source_episode_index": ep_idx,
            "original_episode_index": int(spec["original_episode_index"]),
            "vlm_start_time": float(spec["start_time"]),
            "vlm_end_time": float(spec["end_time"]),
            "vlm_confidence": float(spec["confidence"]),
            "vlm_evidence": spec["evidence"],
        }
        for col in video_cols:
            ep_row[col] = src_row[col]
        episode_rows.append(ep_row)
        segment_rows.append(
            SegmentRow(
                segment_key=segment_key,
                new_episode_index=new_ep,
                source_episode_index=ep_idx,
                original_episode_index=int(spec["original_episode_index"]),
                start_time=float(spec["start_time"]),
                end_time=float(spec["end_time"]),
                confidence=float(spec["confidence"]),
                evidence=spec["evidence"],
                frames=seg_len,
            )
        )
        global_index += seg_len

    if not data_parts:
        raise RuntimeError(f"No segment data generated for {segment_key}")

    data_df = pd.concat(data_parts, ignore_index=True)
    episodes_out = pd.DataFrame(episode_rows)
    data_df.to_parquet(dst_root / "data/chunk-000/file-000.parquet", index=False)
    episodes_out.to_parquet(dst_root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    _task_table(cfg["prompt"]).to_parquet(dst_root / "meta/tasks.parquet")

    info = dict(src_info)
    info.update(
        {
            "total_episodes": int(len(episodes_out)),
            "total_frames": int(len(data_df)),
            "total_tasks": 1,
            "splits": {"train": f"0:{len(episodes_out)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        }
    )
    _write_json(dst_root / "meta/info.json", info)
    _write_json(dst_root / "meta/stats.json", _compute_stats(data_df, src_stats))

    report = {
        "source_dataset": str(src_root),
        "output_dataset": str(dst_root),
        "segment_key": segment_key,
        "prompt": cfg["prompt"],
        "episodes": int(len(episodes_out)),
        "frames": int(len(data_df)),
        "skipped": skipped,
        "video_mode": video_mode,
        "note": "Timestamps and frame_index values are preserved from the zero-motion-cleaned source dataset.",
    }
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row.__dict__ for row in segment_rows]).to_csv(
        report_dir / f"{segment_key}_segments.csv", index=False
    )
    _write_json(report_dir / f"{segment_key}_summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video-mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--pre-margin-s", type=float, default=0.2)
    parser.add_argument("--post-margin-s", type=float, default=0.2)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--min-frames", type=int, default=5)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    src_root = args.dataset_root.expanduser().resolve()
    annotations_dir = args.annotations_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    src_data = pd.read_parquet(src_root / "data/chunk-000/file-000.parquet")
    src_episodes = pd.read_parquet(src_root / "meta/episodes/chunk-000/file-000.parquet")
    src_info = _load_json(src_root / "meta/info.json")
    src_stats = _load_json(src_root / "meta/stats.json")

    specs_by_key, failures = _extract_segments(
        annotations_dir,
        src_episodes,
        pre_margin_s=args.pre_margin_s,
        post_margin_s=args.post_margin_s,
        min_confidence=args.min_confidence,
    )
    reports = {}
    for segment_key, specs in specs_by_key.items():
        reports[segment_key] = _build_one_dataset(
            src_root=src_root,
            output_root=output_root,
            segment_key=segment_key,
            specs=specs,
            src_data=src_data,
            src_episodes=src_episodes,
            src_info=src_info,
            src_stats=src_stats,
            video_mode=args.video_mode,
            replace=args.replace,
            min_frames=args.min_frames,
        )

    summary = {
        "dataset_root": str(src_root),
        "annotations_dir": str(annotations_dir),
        "output_root": str(output_root),
        "reports": reports,
        "annotation_failures": failures,
        "pre_margin_s": args.pre_margin_s,
        "post_margin_s": args.post_margin_s,
        "min_confidence": args.min_confidence,
    }
    _write_json(output_root / "split_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
