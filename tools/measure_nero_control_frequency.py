#!/usr/bin/env python3
"""Measure observable NERO RPC/HAL control frequencies.

This script intentionally separates read-only measurements from command-path
measurements. The default mode is read-only and does not send motion commands.

Examples:
  # Read-only RPC feedback throughput for 5 seconds.
  python tools/measure_nero_control_frequency.py --mode read-pose --duration 5

  # Measure whether the command path can sustain 60 Hz using zero delta commands.
  # This still sends commands to the robot server; only run when the robot is safe.
  python tools/measure_nero_control_frequency.py --mode noop-servo-p-ol --target-hz 60 --duration 5
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

try:
    import zerorpc
except ImportError as exc:  # pragma: no cover - depends on robot env
    raise SystemExit("zerorpc is required. Run in the same env as Le-nero/Evo-RL.") from exc


ARM_TO_PREFIX = {
    "left": "left_robot",
    "right": "right_robot",
}


@dataclass
class Sample:
    start: float
    end: float
    ok: bool
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        return (self.end - self.start) * 1000.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    values_sorted = sorted(values)
    idx = (len(values_sorted) - 1) * pct
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values_sorted[lo]
    return values_sorted[lo] * (hi - idx) + values_sorted[hi] * (idx - lo)


def summarize(samples: list[Sample], wall_start: float, wall_end: float) -> None:
    duration = wall_end - wall_start
    ok_samples = [s for s in samples if s.ok]
    errors = [s for s in samples if not s.ok]
    latencies = [s.latency_ms for s in ok_samples]
    starts = [s.start for s in ok_samples]
    intervals_ms = [(b - a) * 1000.0 for a, b in zip(starts, starts[1:])]

    achieved_hz = len(ok_samples) / duration if duration > 0 else float("nan")
    attempted_hz = len(samples) / duration if duration > 0 else float("nan")

    print("\n=== Summary ===")
    print(f"wall_duration_s: {duration:.3f}")
    print(f"attempted_calls: {len(samples)} ({attempted_hz:.2f} Hz)")
    print(f"successful_calls: {len(ok_samples)} ({achieved_hz:.2f} Hz)")
    print(f"errors: {len(errors)}")

    if latencies:
        print("\nRPC latency, successful calls:")
        print(f"  mean_ms: {statistics.fmean(latencies):.3f}")
        print(f"  p50_ms:  {percentile(latencies, 0.50):.3f}")
        print(f"  p95_ms:  {percentile(latencies, 0.95):.3f}")
        print(f"  max_ms:  {max(latencies):.3f}")

    if intervals_ms:
        print("\nInter-call interval, successful calls:")
        print(f"  mean_ms: {statistics.fmean(intervals_ms):.3f}")
        print(f"  p50_ms:  {percentile(intervals_ms, 0.50):.3f}")
        print(f"  p95_ms:  {percentile(intervals_ms, 0.95):.3f}")
        print(f"  max_ms:  {max(intervals_ms):.3f}")

    if errors:
        print("\nFirst errors:")
        for sample in errors[:5]:
            print(f"  {sample.error}")


def sleep_until(target_time: float) -> None:
    while True:
        remaining = target_time - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.002:
            time.sleep(remaining - 0.001)


def make_call(client: Any, mode: str, arm: str) -> Callable[[], Any]:
    prefix = ARM_TO_PREFIX[arm]
    robot_arm = f"{arm}_robot"

    if mode == "read-pose":
        return getattr(client, f"{prefix}_get_ee_pose")
    if mode == "read-joints":
        return getattr(client, f"{prefix}_get_joint_positions")
    if mode == "read-status":
        return getattr(client, f"{prefix}_get_arm_status")
    if mode == "noop-servo-p-ol":
        return lambda: client.servo_p_OL(robot_arm, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], True)
    if mode == "noop-servo-p":
        return lambda: client.servo_p(robot_arm, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], True)
    raise ValueError(f"Unsupported mode: {mode}")


def run(args: argparse.Namespace) -> int:
    if args.mode.startswith("noop") and not args.i_understand_this_sends_commands:
        print(
            "Refusing to run command-path mode without "
            "--i-understand-this-sends-commands.",
            file=sys.stderr,
        )
        return 2

    client = zerorpc.Client(heartbeat=args.heartbeat_s, timeout=args.rpc_timeout_s)
    client.connect(f"tcp://{args.host}:{args.port}")
    call = make_call(client, args.mode, args.arm)

    period = None if args.target_hz <= 0 else 1.0 / args.target_hz
    wall_start = time.perf_counter()
    next_start = wall_start
    samples: list[Sample] = []

    print(
        f"mode={args.mode} arm={args.arm} target_hz="
        f"{'max' if period is None else args.target_hz} duration_s={args.duration}"
    )

    while True:
        now = time.perf_counter()
        if now - wall_start >= args.duration:
            break
        if period is not None:
            sleep_until(next_start)
            next_start += period

        start = time.perf_counter()
        try:
            call()
            end = time.perf_counter()
            samples.append(Sample(start=start, end=end, ok=True))
        except Exception as exc:  # pragma: no cover - hardware dependent
            end = time.perf_counter()
            samples.append(Sample(start=start, end=end, ok=False, error=repr(exc)))
            if not args.keep_going_on_error:
                break

    wall_end = time.perf_counter()
    summarize(samples, wall_start, wall_end)
    client.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="10.10.10.1", help="NERO server host/IP.")
    parser.add_argument("--port", type=int, default=4242, help="NERO server port.")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument(
        "--mode",
        choices=("read-pose", "read-joints", "read-status", "noop-servo-p-ol", "noop-servo-p"),
        default="read-pose",
        help="Measurement mode. read-* modes do not send motion commands.",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--target-hz",
        type=float,
        default=0.0,
        help="Target call frequency. Use 0 for max-throughput mode.",
    )
    parser.add_argument("--rpc-timeout-s", type=float, default=5.0)
    parser.add_argument("--heartbeat-s", type=float, default=20.0)
    parser.add_argument("--keep-going-on-error", action="store_true")
    parser.add_argument(
        "--i-understand-this-sends-commands",
        action="store_true",
        help="Required for noop-servo-* modes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
