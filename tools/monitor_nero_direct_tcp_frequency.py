#!/usr/bin/env python3
"""Monitor an already-running direct robot-record -> NERO server connection.

This is a non-invasive fallback for cases where robot-record is already
connected directly to the NERO server and cannot be restarted to use an RPC
proxy. It reads Linux TCP counters through `ss -tinp`; no packet capture or
ptrace permissions are required.

Limitations:
  - This measures TCP data segment rates, not decoded ZeroRPC method names.
  - For the current Le-nero robot, each record frame normally issues four
    observation RPCs:
      left_joint_positions, left_ee_pose, right_joint_positions, right_ee_pose
    So at 30 Hz, an observation-only baseline is about 120 data segments/s.
  - Extra request segments above that baseline are usually servo/gripper/reset
    RPCs. Treat this as an estimate, not a method-level trace.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from dataclasses import dataclass


TCP_RE = re.compile(
    r"bytes_sent:(?P<bytes_sent>\d+).*?"
    r"bytes_acked:(?P<bytes_acked>\d+).*?"
    r"bytes_received:(?P<bytes_received>\d+).*?"
    r"segs_out:(?P<segs_out>\d+).*?"
    r"segs_in:(?P<segs_in>\d+).*?"
    r"data_segs_out:(?P<data_segs_out>\d+).*?"
    r"data_segs_in:(?P<data_segs_in>\d+)"
)


@dataclass(frozen=True)
class TcpSample:
    t: float
    counters: dict[str, int]


def read_connection_counters(local: str, remote: str | None) -> dict[str, int]:
    out = subprocess.check_output(["ss", "-tinp"], text=True, stderr=subprocess.DEVNULL)
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if local not in line:
            continue
        if remote and remote not in line:
            continue
        detail = lines[i + 1] if i + 1 < len(lines) else ""
        match = TCP_RE.search(detail)
        if match:
            return {key: int(value) for key, value in match.groupdict().items()}
    raise RuntimeError(f"No matching TCP connection found for local={local!r} remote={remote!r}")


def diff_rate(a: TcpSample, b: TcpSample, key: str) -> float:
    dt = b.t - a.t
    if dt <= 0:
        return 0.0
    return (b.counters[key] - a.counters[key]) / dt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", default="10.10.10.1:4242", help="Local server endpoint shown by ss.")
    parser.add_argument("--remote", default="10.10.10.2:", help="Remote client prefix shown by ss.")
    parser.add_argument("--fps", type=float, default=30.0, help="Expected robot-record loop FPS.")
    parser.add_argument(
        "--obs-rpcs-per-frame",
        type=float,
        default=4.0,
        help="Expected observation RPC calls per frame for Le-nero NERO robot.",
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=0, help="0 means run forever.")
    args = parser.parse_args()

    prev = TcpSample(time.time(), read_connection_counters(args.local, args.remote))
    expected_obs_rpc_hz = args.fps * args.obs_rpcs_per_frame

    print(
        "Monitoring existing direct NERO TCP connection. "
        "Press Ctrl-C to stop.\n"
        f"local={args.local} remote_prefix={args.remote} "
        f"expected_observation_baseline={expected_obs_rpc_hz:.1f} rpc/s"
    )

    count = 0
    while args.samples <= 0 or count < args.samples:
        time.sleep(args.interval)
        cur = TcpSample(time.time(), read_connection_counters(args.local, args.remote))
        count += 1

        req_hz = diff_rate(prev, cur, "data_segs_in")
        resp_hz = diff_rate(prev, cur, "data_segs_out")
        recv_bps = diff_rate(prev, cur, "bytes_received")
        sent_bps = diff_rate(prev, cur, "bytes_sent")
        obs_only_loop_hz = req_hz / args.obs_rpcs_per_frame if args.obs_rpcs_per_frame else 0.0
        extra_rpc_hz = max(0.0, req_hz - expected_obs_rpc_hz)

        print(
            time.strftime("[%Y-%m-%d %H:%M:%S]"),
            f"tcp_req_segments={req_hz:7.2f}/s",
            f"tcp_resp_segments={resp_hz:7.2f}/s",
            f"obs_only_loop_est={obs_only_loop_hz:6.2f}Hz",
            f"extra_rpc_over_obs={extra_rpc_hz:6.2f}/s",
            f"client_to_server={recv_bps:8.1f}B/s",
            f"server_to_client={sent_bps:8.1f}B/s",
        )
        prev = cur

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
