#!/usr/bin/env python
"""Copy a video-backed LeRobot dataset while center-cropping and resizing videos.

This keeps all parquet/action/task metadata unchanged and rewrites only the MP4
files plus image feature metadata. It is intended for controlled visual
preprocessing ablations where the control/action data should stay identical.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import av
import cv2


def _video_feature_keys(info: dict) -> list[str]:
    return [key for key, ft in info["features"].items() if ft.get("dtype") == "video"]


def _transcode_one(task: tuple[str, str, int, int, int, int, int, int, str, int]) -> dict:
    src_s, dst_s, top, left, crop_h, crop_w, out_h, out_w, codec, fps = task
    src = Path(src_s)
    dst = Path(dst_s)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)

    in_container = av.open(str(src))
    if not in_container.streams.video:
        raise RuntimeError(f"No video stream found: {src}")

    in_stream = in_container.streams.video[0]
    fps_fraction = Fraction(fps, 1)
    out_container = av.open(str(dst), mode="w")
    out_stream = out_container.add_stream(codec, rate=fps_fraction)
    out_stream.width = out_w
    out_stream.height = out_h
    out_stream.pix_fmt = "yuv420p"
    out_stream.time_base = Fraction(1, fps)

    frame_count = 0
    started = time.time()
    try:
        for packet in in_container.demux(in_stream):
            for frame in packet.decode():
                arr = frame.to_ndarray(format="rgb24")
                crop = arr[top : top + crop_h, left : left + crop_w]
                if crop.shape[0] != crop_h or crop.shape[1] != crop_w:
                    raise RuntimeError(
                        f"Invalid crop for {src}: got {crop.shape[:2]}, expected {(crop_h, crop_w)}"
                    )
                resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
                out_frame = av.VideoFrame.from_ndarray(resized, format="rgb24")
                out_frame.pts = frame_count
                out_frame.time_base = Fraction(1, fps)
                for out_packet in out_stream.encode(out_frame):
                    out_container.mux(out_packet)
                frame_count += 1

        for out_packet in out_stream.encode():
            out_container.mux(out_packet)
    finally:
        out_container.close()
        in_container.close()

    return {
        "src": str(src),
        "dst": str(dst),
        "frames": frame_count,
        "seconds": round(time.time() - started, 3),
        "size_mb": round(dst.stat().st_size / (1024 * 1024), 3),
    }


def _verify_video(path: Path, expected_h: int, expected_w: int) -> dict:
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        codec = stream.codec_context.name
    finally:
        container.close()
    if (height, width) != (expected_h, expected_w):
        raise RuntimeError(f"Unexpected video shape for {path}: {(height, width)}")
    return {"height": height, "width": width, "codec": codec}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--crop-size", type=int, default=240)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    info_path = src / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)

    info = json.loads(info_path.read_text())
    video_keys = _video_feature_keys(info)
    if not video_keys:
        raise RuntimeError(f"No video feature keys found in {info_path}")

    shapes = {key: info["features"][key]["shape"] for key in video_keys}
    heights = {shape[0] for shape in shapes.values()}
    widths = {shape[1] for shape in shapes.values()}
    if len(heights) != 1 or len(widths) != 1:
        raise RuntimeError(f"Expected all cameras to share image shape, got {shapes}")
    in_h = heights.pop()
    in_w = widths.pop()
    if args.crop_size > in_h or args.crop_size > in_w:
        raise RuntimeError(f"crop-size={args.crop_size} does not fit source shape {(in_h, in_w)}")

    top = (in_h - args.crop_size) // 2
    left = (in_w - args.crop_size) // 2

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"Destination exists: {dst}")
        shutil.rmtree(dst)

    print(f"Copying metadata/data: {src} -> {dst}")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("videos"))

    video_files = sorted((src / "videos").rglob("*.mp4"))
    if not video_files:
        raise RuntimeError(f"No mp4 files under {src / 'videos'}")

    tasks = []
    for src_video in video_files:
        rel = src_video.relative_to(src)
        dst_video = dst / rel
        tasks.append(
            (
                str(src_video),
                str(dst_video),
                top,
                left,
                args.crop_size,
                args.crop_size,
                args.output_size,
                args.output_size,
                args.codec,
                args.fps,
            )
        )

    print(
        f"Transcoding {len(tasks)} videos: crop top={top}, left={left}, "
        f"size={args.crop_size} -> {args.output_size}, codec={args.codec}"
    )
    results = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_transcode_one, task) for task in tasks]
        for i, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            results.append(result)
            print(
                f"[{i:03d}/{len(futures):03d}] {Path(result['dst']).name} "
                f"frames={result['frames']} size={result['size_mb']}MB time={result['seconds']}s",
                flush=True,
            )

    for key in video_keys:
        ft = info["features"][key]
        ft["shape"] = [args.output_size, args.output_size, 3]
        ft.setdefault("info", {})
        ft["info"].update(
            {
                "video.height": args.output_size,
                "video.width": args.output_size,
                "video.codec": "h264" if args.codec in {"libx264", "h264"} else args.codec,
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": args.fps,
                "video.channels": 3,
                "has_audio": False,
            }
        )
    info["repo_id"] = args.repo_id
    info["video_files_size_in_mb"] = sum(p.stat().st_size for p in (dst / "videos").rglob("*.mp4")) / (
        1024 * 1024
    )
    (dst / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=4) + "\n")

    crop_meta = {
        "source_dataset": str(src),
        "repo_id": args.repo_id,
        "source_shape_hwc": [in_h, in_w, 3],
        "crop_top_left_hw": [top, left, args.crop_size, args.crop_size],
        "output_shape_hwc": [args.output_size, args.output_size, 3],
        "codec": args.codec,
        "fps": args.fps,
        "num_videos": len(results),
        "elapsed_s": round(time.time() - started, 3),
    }
    (dst / "meta" / "image_preprocessing_centercrop240_r224.json").write_text(
        json.dumps(crop_meta, ensure_ascii=False, indent=4) + "\n"
    )

    first_video = sorted((dst / "videos").rglob("*.mp4"))[0]
    verified = _verify_video(first_video, args.output_size, args.output_size)
    print(f"Verified first video: {first_video} {verified}")
    print(f"Wrote cropped dataset: {dst}")


if __name__ == "__main__":
    main()
