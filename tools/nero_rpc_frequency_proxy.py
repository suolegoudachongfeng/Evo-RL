#!/usr/bin/env python3
"""Zerorpc proxy for measuring NERO control RPC frequency during robot-record.

Run this proxy between Le-nero robot-record and the real NERO server:

  python tools/nero_rpc_frequency_proxy.py \
    --listen tcp://0.0.0.0:4243 \
    --target tcp://10.10.10.1:4242

Then point the robot client at the proxy port instead of the real server
temporarily. The proxy forwards every call unchanged and prints per-method
frequency/latency statistics once per second.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_METHODS = (
    "left_robot_get_joint_positions",
    "left_robot_get_joint_velocities",
    "left_robot_get_ee_pose",
    "left_robot_get_arm_status",
    "right_robot_get_joint_positions",
    "right_robot_get_joint_velocities",
    "right_robot_get_ee_pose",
    "right_robot_get_arm_status",
    "left_robot_move_to_joint_positions",
    "left_robot_move_to_ee_pose",
    "right_robot_move_to_joint_positions",
    "right_robot_move_to_ee_pose",
    "dual_robot_move_to_ee_pose",
    "left_robot_go_home",
    "right_robot_go_home",
    "robot_go_home",
    "servo_j",
    "servo_p_OL",
    "servo_p",
    "left_gripper_goto",
    "left_gripper_grasp",
    "left_gripper_get_state",
    "right_gripper_goto",
    "right_gripper_grasp",
    "right_gripper_get_state",
    "robot_stop",
)

CONTROL_METHODS = {"servo_j", "servo_p_OL", "servo_p"}


@dataclass(frozen=True)
class CallRecord:
    method: str
    arm: str
    start: float
    end: float
    ok: bool
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        return (self.end - self.start) * 1000.0


def infer_arm(method_name: str, args: tuple[Any, ...]) -> str:
    if args and isinstance(args[0], str) and args[0] in {"left_robot", "right_robot"}:
        return args[0]
    return "-"


class FrequencyStats:
    def __init__(self, window_s: float = 1.0):
        self.window_s = window_s
        self._records: deque[CallRecord] = deque()
        self._lock = threading.Lock()

    def record(
        self,
        method_name: str,
        start: float,
        end: float,
        args: tuple[Any, ...],
        *,
        ok: bool = True,
        error: str | None = None,
    ) -> None:
        record = CallRecord(
            method=method_name,
            arm=infer_arm(method_name, args),
            start=start,
            end=end,
            ok=ok,
            error=error,
        )
        with self._lock:
            self._records.append(record)
            self._trim_locked(end)

    def snapshot(self, now: float | None = None) -> dict[tuple[str, str], dict[str, Any]]:
        now = time.perf_counter() if now is None else now
        with self._lock:
            self._trim_locked(now)
            records = list(self._records)

        grouped: dict[tuple[str, str], list[CallRecord]] = defaultdict(list)
        for record in records:
            grouped[(record.method, record.arm)].append(record)

        rows: dict[tuple[str, str], dict[str, Any]] = {}
        for key, group in grouped.items():
            ok_records = [r for r in group if r.ok]
            latencies = [r.latency_ms for r in ok_records]
            rows[key] = {
                "count": len(group),
                "ok": len(ok_records),
                "errors": len(group) - len(ok_records),
                "hz": round(len(group) / self.window_s, 3),
                "mean_latency_ms": round(statistics.fmean(latencies), 3) if latencies else None,
                "max_latency_ms": round(max(latencies), 3) if latencies else None,
            }
        return rows

    def _trim_locked(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._records and self._records[0].end < cutoff:
            self._records.popleft()


def make_forwarder(
    *,
    target: Any,
    method_name: str,
    stats: FrequencyStats,
    clock: Callable[[], float] = time.perf_counter,
) -> Callable[..., Any]:
    def forwarder(*args: Any, **kwargs: Any) -> Any:
        start = clock()
        try:
            result = getattr(target, method_name)(*args, **kwargs)
        except Exception as exc:
            end = clock()
            stats.record(method_name, start, end, args, ok=False, error=repr(exc))
            raise
        end = clock()
        stats.record(method_name, start, end, args, ok=True)
        return result

    forwarder.__name__ = method_name
    forwarder.__doc__ = f"Forwarded NERO RPC method {method_name}."
    return forwarder


def format_rows(rows: dict[tuple[str, str], dict[str, Any]], *, control_only: bool) -> str:
    visible = []
    for (method, arm), values in sorted(rows.items()):
        if control_only and method not in CONTROL_METHODS:
            continue
        visible.append((method, arm, values))

    if not visible:
        return "no calls in current window"

    lines = []
    for method, arm, values in visible:
        lines.append(
            f"{method:14s} {arm:11s} "
            f"hz={values['hz']:7.2f} count={values['count']:4d} "
            f"ok={values['ok']:4d} err={values['errors']:3d} "
            f"mean={values['mean_latency_ms']}ms max={values['max_latency_ms']}ms"
        )
    return "\n".join(lines)


def stats_printer(
    *,
    stats: FrequencyStats,
    interval_s: float,
    control_only: bool,
    jsonl_path: str | None,
    stop_event: threading.Event,
) -> None:
    jsonl_file = open(jsonl_path, "a", encoding="utf-8") if jsonl_path else None
    try:
        while not stop_event.wait(interval_s):
            now = time.time()
            rows = stats.snapshot()
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] last {stats.window_s:.1f}s")
            print(format_rows(rows, control_only=control_only), flush=True)
            if jsonl_file is not None:
                payload = {
                    "wall_time": now,
                    "window_s": stats.window_s,
                    "rows": [
                        {"method": method, "arm": arm, **values}
                        for (method, arm), values in sorted(rows.items())
                    ],
                }
                jsonl_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                jsonl_file.flush()
    finally:
        if jsonl_file is not None:
            jsonl_file.close()


def build_methods(target: Any, stats: FrequencyStats, methods: tuple[str, ...]) -> dict[str, Callable[..., Any]]:
    return {
        method_name: make_forwarder(target=target, method_name=method_name, stats=stats)
        for method_name in methods
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="tcp://0.0.0.0:4243", help="Proxy bind address.")
    parser.add_argument("--target", default="tcp://10.10.10.1:4242", help="Real NERO server address.")
    parser.add_argument("--window-s", type=float, default=1.0, help="Rolling frequency window.")
    parser.add_argument("--print-interval-s", type=float, default=1.0)
    parser.add_argument("--heartbeat-s", type=float, default=20.0)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--jsonl", default=None, help="Optional JSONL stats log path.")
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Print read/status/gripper calls too. Default prints control calls only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import zerorpc

    target = zerorpc.Client(heartbeat=args.heartbeat_s)
    target.connect(args.target)

    stats = FrequencyStats(window_s=args.window_s)
    methods = build_methods(target, stats, DEFAULT_METHODS)
    server = zerorpc.Server(methods, heartbeat=args.heartbeat_s, pool_size=args.pool_size)

    stop_event = threading.Event()
    printer = threading.Thread(
        target=stats_printer,
        kwargs={
            "stats": stats,
            "interval_s": args.print_interval_s,
            "control_only": not args.all_methods,
            "jsonl_path": args.jsonl,
            "stop_event": stop_event,
        },
        daemon=True,
    )
    printer.start()

    print(f"[proxy] listening on {args.listen}")
    print(f"[proxy] forwarding to {args.target}")
    print("[proxy] point robot-record robot_ip/robot_port to this proxy while measuring")
    try:
        server.bind(args.listen)
        server.run()
    finally:
        stop_event.set()
        target.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
