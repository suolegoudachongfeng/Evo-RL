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

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def _normalize_side(side: str) -> str:
    normalized = side.strip().lower()
    aliases = {
        "l": "left_arm",
        "left": "left_arm",
        "left_arm": "left_arm",
        "r": "right_arm",
        "right": "right_arm",
        "right_arm": "right_arm",
    }
    if normalized not in aliases:
        raise ValueError("side must be one of: left, left_arm, right, right_arm")
    return aliases[normalized]


def _normalize_side_or_both(side: str) -> str:
    normalized = side.strip().lower()
    if normalized in {"both", "all"}:
        return "both"
    return _normalize_side(side)


class FrankaDualArmClient:
    """ZeroRPC client for the external dual-Franka + Robotiq server."""

    def __init__(self, ip: str = "127.0.0.1", port: int = 4242, timeout: float = 30.0):
        import zerorpc

        self.ip = ip
        self.port = int(port)
        self.timeout = float(timeout)
        self.server_url = f"tcp://{ip}:{self.port}"
        self.server = zerorpc.Client(timeout=self.timeout)
        self.server.connect(self.server_url)
        log.info("[FRANKA CLIENT] Connected to %s", self.server_url)

    def close(self) -> None:
        if self.server is None:
            return
        try:
            self.server.close()
        finally:
            self.server = None

    def _call(self, name: str, *args):
        if self.server is None:
            raise RuntimeError("Franka RPC client is closed.")
        return getattr(self.server, name)(*args)

    def ping(self):
        return self._call("ping")

    def reset(self):
        return self._call("reset")

    def step(self, action: dict[str, Any] | None = None):
        return self._call("step", action)

    def get_observation(self):
        return self._call("get_observation")

    def get_full_state(self):
        return self.get_observation()

    def get_home(self):
        return self._call("get_home")

    def set_home_current(self, side: str = "both"):
        return self._call("set_home_current", _normalize_side_or_both(side))

    def save_home_current(self, side: str = "both"):
        return self._call("save_home_current", _normalize_side_or_both(side))

    def go_home(self, side: str = "both", duration_sec: float | None = None, rate_hz: float | None = None):
        return self._call("go_home", _normalize_side_or_both(side), duration_sec, rate_hz)

    def recover_robot(self, side: str = "both"):
        return self._call("recover_robot", _normalize_side_or_both(side))

    def command_gripper(self, side: str = "left_arm", command: dict[str, Any] | None = None):
        return self._call("command_gripper", _normalize_side(side), command or {})

    def reactivate_gripper(self, side: str = "left_arm"):
        return self._call("reactivate_gripper", _normalize_side(side))

    def open_gripper(self, side: str = "left_arm"):
        return self._call("open_gripper", _normalize_side(side))

    def close_gripper(self, side: str = "left_arm"):
        return self._call("close_gripper", _normalize_side(side))

    def left_gripper_initialize(self):
        return self.reactivate_gripper("left_arm")

    def right_gripper_initialize(self):
        return self.reactivate_gripper("right_arm")

    def gripper_initialize(self):
        return {
            "left": self.left_gripper_initialize(),
            "right": self.right_gripper_initialize(),
        }

    def left_gripper_goto(
        self,
        width: float,
        speed: float = 0.1,
        force: float = 10.0,
        blocking: bool = True,
    ):
        del blocking
        return self.command_gripper(
            "left_arm",
            {"width": float(width), "max_velocity": float(speed), "max_effort": float(force)},
        )

    def right_gripper_goto(
        self,
        width: float,
        speed: float = 0.1,
        force: float = 10.0,
        blocking: bool = True,
    ):
        del blocking
        return self.command_gripper(
            "right_arm",
            {"width": float(width), "max_velocity": float(speed), "max_effort": float(force)},
        )

    def left_gripper_get_state(self) -> dict[str, Any]:
        obs = self.get_observation()
        side_obs = obs.get("left_arm", {}) if isinstance(obs, dict) else {}
        return self._gripper_state_from_observation(side_obs)

    def right_gripper_get_state(self) -> dict[str, Any]:
        obs = self.get_observation()
        side_obs = obs.get("right_arm", {}) if isinstance(obs, dict) else {}
        return self._gripper_state_from_observation(side_obs)

    @staticmethod
    def _gripper_state_from_observation(side_obs: dict[str, Any]) -> dict[str, Any]:
        grip = side_obs.get("gripper", {}) if isinstance(side_obs, dict) else {}
        if not isinstance(grip, dict):
            grip = {"position": grip}
        return grip

    def set_left_gripper(self, normalized_close: float):
        return self.command_gripper("left_arm", {"normalized": float(normalized_close)})

    def set_right_gripper(self, normalized_close: float):
        return self.command_gripper("right_arm", {"normalized": float(normalized_close)})

    def dual_robot_move_to_ee_pose(self, left_delta, right_delta, delta: bool = True, wait: bool = False):
        del wait
        if not delta:
            raise NotImplementedError("Only delta=True is supported by this RPC server adapter.")
        action = {
            "left_arm": {
                "motion": {
                    "translation": list(left_delta[:3]),
                    "rotation_rotvec": list(left_delta[3:]),
                }
            },
            "right_arm": {
                "motion": {
                    "translation": list(right_delta[:3]),
                    "rotation_rotvec": list(right_delta[3:]),
                }
            },
        }
        return self.step(action)
