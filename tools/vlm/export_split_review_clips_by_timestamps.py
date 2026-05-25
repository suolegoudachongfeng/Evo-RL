#!/usr/bin/env python3
"""Export per-episode review clips from split LeRobot datasets.

The VLM split datasets preserve source timestamps in data parquet files and
point each episode at the original video shard in metadata. This exporter uses
the data timestamp range for each split episode, avoiding the common mistake of
cutting from the full source episode start instead of the split segment start.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


DEFAULT_SEGMENTS = (
    "seg1_grasp_first_vial",
    "seg2_insert_first_vial_bottom_right",
    "seg3_grasp_remaining_vial",
    "seg4_insert_remaining_vial_top_right",
)


def _run_ffmpeg(job: tuple[list[str], Path]) -> tuple[str, Path, str]:
    cmd, output_path = job
    tmp_path = output_path.with_suffix(".tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()
    cmd = [*cmd[:-1], str(tmp_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        if tmp_path.exists():
            tmp_path.unlink()
        return "failed", output_path, proc.stderr.strip()
    tmp_path.replace(output_path)
    return "done", output_path, ""


def _build_jobs(
    *,
    split_root: Path,
    output_root: Path,
    camera: str,
    segments: tuple[str, ...],
    ffmpeg: str,
    fps: float,
    crf: int,
    preset: str,
    replace: bool,
) -> tuple[list[tuple[list[str], Path]], list[dict[str, object]]]:
    jobs: list[tuple[list[str], Path]] = []
    rows: list[dict[str, object]] = []
    for segment_name in segments:
        segment_root = split_root / segment_name
        episodes_path = segment_root / "meta/episodes/chunk-000/file-000.parquet"
        data_path = segment_root / "data/chunk-000/file-000.parquet"
        if not episodes_path.exists() or not data_path.exists():
            raise FileNotFoundError(f"Missing LeRobot parquet files for {segment_root}")

        episodes = pd.read_parquet(episodes_path)
        data = pd.read_parquet(data_path, columns=["episode_index", "timestamp"])
        ranges = data.groupby("episode_index")["timestamp"].agg(["min", "max", "count"]).reset_index()
        merged = episodes.merge(ranges, on="episode_index", how="inner")
        output_dir = output_root / camera.rsplit(".", 1)[-1] / segment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        file_col = f"videos/{camera}/file_index"
        chunk_col = f"videos/{camera}/chunk_index"
        from_col = f"videos/{camera}/from_timestamp"
        for _, row in merged.iterrows():
            episode_index = int(row["episode_index"])
            source_episode_index = int(row.get("source_episode_index", episode_index))
            original_episode_index = int(row.get("original_episode_index", source_episode_index))
            file_index = int(row[file_col])
            chunk_index = int(row[chunk_col])
            source_video_from = float(row[from_col])
            timestamp_min = float(row["min"])
            timestamp_max = float(row["max"])
            start_s = max(0.0, source_video_from + timestamp_min)
            duration_s = max(1.0 / fps, (timestamp_max - timestamp_min) + 1.0 / fps)
            input_video = (
                segment_root
                / "videos"
                / camera
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            if not input_video.exists():
                raise FileNotFoundError(input_video)

            output_path = (
                output_dir
                / f"episode_{episode_index:03d}_source_{source_episode_index:03d}_{segment_name}_{camera.rsplit('.', 1)[-1]}.mp4"
            )
            if output_path.exists() and output_path.stat().st_size > 1024 and not replace:
                continue

            cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start_s:.6f}",
                "-t",
                f"{duration_s:.6f}",
                "-i",
                str(input_video),
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            jobs.append((cmd, output_path))
            rows.append(
                {
                    "segment": segment_name,
                    "episode_index": episode_index,
                    "source_episode_index": source_episode_index,
                    "original_episode_index": original_episode_index,
                    "camera": camera,
                    "input_video": str(input_video),
                    "input_start_s": f"{start_s:.6f}",
                    "duration_s": f"{duration_s:.6f}",
                    "frame_count": int(row["count"]),
                    "output_video": str(output_path),
                    "vlm_start_time": row.get("vlm_start_time", ""),
                    "vlm_end_time": row.get("vlm_end_time", ""),
                    "vlm_confidence": row.get("vlm_confidence", ""),
                    "vlm_evidence": row.get("vlm_evidence", ""),
                }
            )
    return jobs, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--camera", default="observation.images.right_wrist_image")
    parser.add_argument("--segments", nargs="*", default=list(DEFAULT_SEGMENTS))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="ultrafast")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    split_root = args.split_root.expanduser().resolve()
    output_root = (args.output_root or split_root / "review_clips").expanduser().resolve()
    if args.replace and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError("ffmpeg is not available on PATH and imageio_ffmpeg fallback failed") from exc

    jobs, rows = _build_jobs(
        split_root=split_root,
        output_root=output_root,
        camera=args.camera,
        segments=tuple(args.segments),
        ffmpeg=ffmpeg,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        replace=args.replace,
    )
    print(f"Prepared {len(jobs)} clip exports under {output_root}")

    failures: list[tuple[Path, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_ffmpeg, job) for job in jobs]
        for future in as_completed(futures):
            status, output_path, error = future.result()
            if status == "failed":
                failures.append((output_path, error))
            else:
                completed += 1
            total = completed + len(failures)
            if total % 50 == 0 or total == len(jobs):
                print(f"progress {total}/{len(jobs)} completed={completed} failed={len(failures)}")
                sys.stdout.flush()

    manifest_path = output_root / args.camera.rsplit(".", 1)[-1] / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if failures:
        failure_path = output_root / args.camera.rsplit(".", 1)[-1] / "failed.txt"
        with failure_path.open("w", encoding="utf-8") as f:
            for output_path, error in failures:
                f.write(f"{output_path}\n{error}\n\n")
        raise RuntimeError(f"{len(failures)} clip exports failed; see {failure_path}")

    print(f"completed={completed}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
