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
from collections.abc import Mapping
from functools import cached_property
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_franka_dual_arm import FrankaDualArmConfig
from .franka_dual_arm_client import FrankaDualArmClient

logger = logging.getLogger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _as_np(values: Any, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=float)
    if values is None:
        return out
    arr = np.asarray(values, dtype=float).reshape(-1)
    n = min(length, arr.size)
    if n:
        out[:n] = arr[:n]
    return out


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _robot_state_from_side(side_state: Mapping[str, Any]) -> Mapping[str, Any]:
    robot_state = side_state.get("robot_state")
    return _as_mapping(robot_state) if robot_state is not None else side_state


def _ee_pose_from_side(side_state: Mapping[str, Any]) -> np.ndarray:
    robot_state = _robot_state_from_side(side_state)
    if "end_pose" in side_state:
        return _as_np(side_state.get("end_pose"), 6)
    if "end_pose" in robot_state:
        return _as_np(robot_state.get("end_pose"), 6)

    eef_pose = _as_mapping(robot_state.get("eef_pose"))
    if not eef_pose:
        return np.zeros(6, dtype=float)
    position = _as_np(eef_pose.get("position"), 3)
    quat = _as_np(eef_pose.get("orientation_xyzw"), 4)
    rotvec = np.zeros(3, dtype=float)
    if np.linalg.norm(quat) > 1e-12:
        try:
            rotvec = R.from_quat(quat).as_rotvec()
        except ValueError:
            rotvec = np.zeros(3, dtype=float)
    return np.concatenate([position, rotvec])


def _gripper_open_fraction(gripper_state: Any, fallback: float) -> float:
    if isinstance(gripper_state, Mapping):
        if "open_fraction" in gripper_state:
            return _clamp(_safe_float(gripper_state.get("open_fraction"), fallback), 0.0, 1.0)
        if "width" in gripper_state:
            width = _safe_float(gripper_state.get("width"), fallback)
            max_width = _safe_float(gripper_state.get("open_width"), 1.0)
            if abs(max_width) > 1e-9:
                return _clamp(width / max_width, 0.0, 1.0)
            return _clamp(width, 0.0, 1.0)
        if "position" in gripper_state:
            position = _safe_float(gripper_state.get("position"), fallback)
            open_position = _safe_float(gripper_state.get("open_position"), 0.0)
            closed_position = _safe_float(gripper_state.get("closed_position"), 1.0)
            span = closed_position - open_position
            if abs(span) > 1e-9:
                closed_fraction = (position - open_position) / span
                return _clamp(1.0 - closed_fraction, 0.0, 1.0)
            return _clamp(position, 0.0, 1.0)
        return fallback
    return _clamp(_safe_float(gripper_state, fallback), 0.0, 1.0)


def _gripper_open_fraction_from_side(side_state: Mapping[str, Any], fallback: float) -> float:
    if "gripper" in side_state:
        return _gripper_open_fraction(side_state.get("gripper"), fallback)
    sensors = _as_mapping(side_state.get("sensors"))
    if "robotiq" in sensors:
        return _gripper_open_fraction(sensors.get("robotiq"), fallback)
    return fallback


class FrankaDualArm(Robot):
    """Dual Franka adapter that talks to the external ROS2 ZeroRPC bridge."""

    config_class = FrankaDualArmConfig
    name = "franka_dual_arm"

    def __init__(self, config: FrankaDualArmConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._robot: FrankaDualArmClient | None = None
        self._is_connected = False
        self._teleop_send_only_mode = False
        self._prev_observation: RobotObservation | None = None
        self._cached_rpc_state: dict[str, Any] | None = None
        self._left_gripper_state = 1.0
        self._right_gripper_state = 1.0
        self._last_left_gripper_open: float | None = None
        self._last_right_gripper_open: float | None = None
        self._delta_clip_warn_count = 0
        self._nonfinite_action_warn_count = 0

    @property
    def _motors_ft(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for i in range(self.config.num_joints_per_arm):
            features[f"left_joint_{i + 1}.pos"] = float
        for i in range(self.config.num_joints_per_arm):
            features[f"right_joint_{i + 1}.pos"] = float
        for axis in ("x", "y", "z", "rx", "ry", "rz"):
            features[f"left_ee_pose.{axis}"] = float
        for axis in ("x", "y", "z", "rx", "ry", "rz"):
            features[f"right_ee_pose.{axis}"] = float
        if self.config.use_gripper:
            features["left_gripper_cmd_bin"] = float
            features["right_gripper_cmd_bin"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for axis in ("x", "y", "z", "rx", "ry", "rz"):
            features[f"left_delta_ee_pose.{axis}"] = float
        for axis in ("x", "y", "z", "rx", "ry", "rz"):
            features[f"right_delta_ee_pose.{axis}"] = float
        if self.config.use_gripper:
            features["left_gripper_cmd_bin"] = float
            features["right_gripper_cmd_bin"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self._is_connected and (
            self._teleop_send_only_mode or all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def set_teleop_send_only_mode(self, enabled: bool) -> None:
        if self._is_connected:
            raise RuntimeError("teleop send-only mode must be configured before connecting the robot.")
        self._teleop_send_only_mode = enabled

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self._robot = FrankaDualArmClient(
            ip=self.config.robot_ip,
            port=self.config.robot_port,
            timeout=self.config.rpc_timeout_sec,
        )
        logger.info("[FRANKA] Server ping: %s", self._robot.ping())
        if self.config.use_gripper:
            try:
                self._robot.gripper_initialize()
            except Exception:
                logger.warning("[FRANKA] Gripper initialize failed.", exc_info=True)
            if self.config.open_grippers_on_connect:
                self._open_both_grippers(blocking=True)
        if not self._teleop_send_only_mode:
            for cam in self.cameras.values():
                cam.connect()
        self._is_connected = True
        logger.info("%s connected to %s:%s.", self, self.config.robot_ip, self.config.robot_port)

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self._robot is None:
            raise RuntimeError("Franka robot client is not initialized.")

        sent_action = dict(action)
        if bool(action.get("reset_requested", False)):
            self.reset()
            return sent_action

        server_action: dict[str, Any] = {}
        gripper_updates: list[tuple[str, float]] = []
        has_cartesian_action = "left_delta_ee_pose.x" in action or "right_delta_ee_pose.x" in action

        if has_cartesian_action:
            left_delta, right_delta = self._cartesian_deltas_from_action(action)
            if self.config.debug:
                left_delta = np.zeros(6, dtype=float)
                right_delta = np.zeros(6, dtype=float)
            self._update_sent_cartesian_action(sent_action, left_delta, right_delta, action)
            self._add_cartesian_step_action(server_action, left_delta, right_delta)

        if self.config.use_gripper:
            if "left_gripper_cmd_bin" in action:
                sent_action["left_gripper_cmd_bin"] = self._add_gripper_step_action(
                    server_action,
                    "left",
                    float(action["left_gripper_cmd_bin"]),
                    gripper_updates,
                )
            if "right_gripper_cmd_bin" in action:
                sent_action["right_gripper_cmd_bin"] = self._add_gripper_step_action(
                    server_action,
                    "right",
                    float(action["right_gripper_cmd_bin"]),
                    gripper_updates,
                )

        if server_action:
            step_result = self._robot.step(server_action)
            self._update_cached_rpc_state_from_step(step_result)
            for side, open_fraction in gripper_updates:
                if side == "left":
                    self._last_left_gripper_open = open_fraction
                else:
                    self._last_right_gripper_open = open_fraction
        return sent_action

    def reset(self) -> None:
        if self._robot is None:
            raise RuntimeError("Franka robot client is not initialized.")
        if self.config.reset_go_home:
            self._robot.go_home("both", self.config.go_home_duration_sec, self.config.go_home_rate_hz)
        else:
            self._robot.reset()
        self._cached_rpc_state = None
        if self.config.use_gripper and self.config.reset_opens_grippers:
            self._open_both_grippers(blocking=True)

    def _open_both_grippers(self, blocking: bool = True) -> None:
        if self._robot is None:
            return
        self._robot.left_gripper_goto(
            width=self.config.gripper_max_open,
            speed=self.config.gripper_speed,
            force=self.config.gripper_force,
            blocking=blocking,
        )
        self._robot.right_gripper_goto(
            width=self.config.gripper_max_open,
            speed=self.config.gripper_speed,
            force=self.config.gripper_force,
            blocking=blocking,
        )
        self._last_left_gripper_open = 1.0
        self._last_right_gripper_open = 1.0
        self._cached_rpc_state = None

    def _cartesian_deltas_from_action(self, action: RobotAction) -> tuple[np.ndarray, np.ndarray]:
        axes = ("x", "y", "z", "rx", "ry", "rz")
        left_delta = np.array([action.get(f"left_delta_ee_pose.{axis}", 0.0) for axis in axes], dtype=float)
        right_delta = np.array([action.get(f"right_delta_ee_pose.{axis}", 0.0) for axis in axes], dtype=float)
        if not np.all(np.isfinite(left_delta)) or not np.all(np.isfinite(right_delta)):
            self._nonfinite_action_warn_count += 1
            if self._nonfinite_action_warn_count <= 5 or self._nonfinite_action_warn_count % 100 == 0:
                logger.warning(
                    "[FRANKA] Non-finite cartesian action received; replacing NaN/Inf with 0 "
                    "(left=%s right=%s)",
                    left_delta.tolist(),
                    right_delta.tolist(),
                )
            left_delta = np.nan_to_num(left_delta, nan=0.0, posinf=0.0, neginf=0.0)
            right_delta = np.nan_to_num(right_delta, nan=0.0, posinf=0.0, neginf=0.0)
        return self._clip_cartesian_deltas(left_delta, right_delta)

    def _clip_cartesian_deltas(
        self, left_delta: np.ndarray, right_delta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.config.max_cartesian_delta is None and self.config.max_rotation_delta is None:
            return left_delta, right_delta

        raw_left = left_delta.copy()
        raw_right = right_delta.copy()
        if self.config.max_cartesian_delta is not None and self.config.max_cartesian_delta > 0.0:
            left_delta[:3] = np.clip(left_delta[:3], -self.config.max_cartesian_delta, self.config.max_cartesian_delta)
            right_delta[:3] = np.clip(
                right_delta[:3], -self.config.max_cartesian_delta, self.config.max_cartesian_delta
            )
        if self.config.max_rotation_delta is not None and self.config.max_rotation_delta > 0.0:
            left_delta[3:] = np.clip(left_delta[3:], -self.config.max_rotation_delta, self.config.max_rotation_delta)
            right_delta[3:] = np.clip(
                right_delta[3:], -self.config.max_rotation_delta, self.config.max_rotation_delta
            )
        if not np.allclose(raw_left, left_delta) or not np.allclose(raw_right, right_delta):
            self._delta_clip_warn_count += 1
            if self._delta_clip_warn_count <= 5 or self._delta_clip_warn_count % 100 == 0:
                logger.warning(
                    "[FRANKA] Cartesian action clipped to per-step limits "
                    "(max_translation=%s max_rotation=%s); raw_left=%s raw_right=%s",
                    self.config.max_cartesian_delta,
                    self.config.max_rotation_delta,
                    raw_left.tolist(),
                    raw_right.tolist(),
                )
        return left_delta, right_delta

    def _add_cartesian_step_action(
        self, server_action: dict[str, Any], left_delta: np.ndarray, right_delta: np.ndarray
    ) -> None:
        if np.linalg.norm(left_delta) >= self.config.min_motion_norm:
            server_action.setdefault("left_arm", {})["motion"] = {
                "translation": left_delta[:3].tolist(),
                "rotation_rotvec": left_delta[3:].tolist(),
            }
        if np.linalg.norm(right_delta) >= self.config.min_motion_norm:
            server_action.setdefault("right_arm", {})["motion"] = {
                "translation": right_delta[:3].tolist(),
                "rotation_rotvec": right_delta[3:].tolist(),
            }

    @staticmethod
    def _update_sent_cartesian_action(
        sent_action: RobotAction,
        left_delta: np.ndarray,
        right_delta: np.ndarray,
        source_action: RobotAction,
    ) -> None:
        axes = ("x", "y", "z", "rx", "ry", "rz")
        for index, axis in enumerate(axes):
            left_key = f"left_delta_ee_pose.{axis}"
            if left_key in source_action:
                sent_action[left_key] = float(left_delta[index])
            right_key = f"right_delta_ee_pose.{axis}"
            if right_key in source_action:
                sent_action[right_key] = float(right_delta[index])

    def _add_gripper_step_action(
        self,
        server_action: dict[str, Any],
        side: str,
        value: float,
        gripper_updates: list[tuple[str, float]],
    ) -> float:
        commanded_open_fraction = _clamp(value, 0.0, 1.0)
        open_fraction = commanded_open_fraction
        if self.config.gripper_reverse:
            open_fraction = 1.0 - open_fraction
        width = open_fraction * self.config.gripper_max_open

        if side == "left":
            if self._last_left_gripper_open is not None and abs(open_fraction - self._last_left_gripper_open) < 1e-4:
                return commanded_open_fraction
            side_key = "left_arm"
        else:
            if self._last_right_gripper_open is not None and abs(open_fraction - self._last_right_gripper_open) < 1e-4:
                return commanded_open_fraction
            side_key = "right_arm"
        server_action.setdefault(side_key, {})["gripper"] = {
            "width": width,
            "max_velocity": self.config.gripper_speed,
            "max_effort": self.config.gripper_force,
        }
        gripper_updates.append((side, open_fraction))
        return commanded_open_fraction

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if self._robot is None:
            raise RuntimeError("Franka robot client is not initialized.")

        try:
            state = self._get_cached_or_live_state()
        except Exception:
            logger.warning("[FRANKA] Failed to query observation, using previous observation.", exc_info=True)
            if self._prev_observation is not None:
                return self._prev_observation
            raise

        obs_dict: RobotObservation = {}
        left_side = _as_mapping(state.get("left_arm", {}))
        right_side = _as_mapping(state.get("right_arm", {}))
        left_robot_state = _robot_state_from_side(left_side)
        right_robot_state = _robot_state_from_side(right_side)
        left_joints = _as_np(left_robot_state.get("joint_positions"), self.config.num_joints_per_arm)
        right_joints = _as_np(right_robot_state.get("joint_positions"), self.config.num_joints_per_arm)
        left_pose = _ee_pose_from_side(left_side)
        right_pose = _ee_pose_from_side(right_side)

        for i in range(self.config.num_joints_per_arm):
            obs_dict[f"left_joint_{i + 1}.pos"] = float(left_joints[i])
            obs_dict[f"right_joint_{i + 1}.pos"] = float(right_joints[i])
        for i, axis in enumerate(("x", "y", "z", "rx", "ry", "rz")):
            obs_dict[f"left_ee_pose.{axis}"] = float(left_pose[i])
            obs_dict[f"right_ee_pose.{axis}"] = float(right_pose[i])

        if self.config.use_gripper:
            left_grip = _gripper_open_fraction_from_side(left_side, self._left_gripper_state)
            right_grip = _gripper_open_fraction_from_side(right_side, self._right_gripper_state)
            if self.config.gripper_reverse:
                left_grip = 1.0 - left_grip
                right_grip = 1.0 - right_grip
            self._left_gripper_state = _clamp(left_grip, 0.0, 1.0)
            self._right_gripper_state = _clamp(right_grip, 0.0, 1.0)
            obs_dict["left_gripper_cmd_bin"] = (
                self._last_left_gripper_open if self._last_left_gripper_open is not None else self._left_gripper_state
            )
            obs_dict["right_gripper_cmd_bin"] = (
                self._last_right_gripper_open
                if self._last_right_gripper_open is not None
                else self._right_gripper_state
            )

        if not self._teleop_send_only_mode:
            for cam_key, cam in self.cameras.items():
                obs_dict[cam_key] = cam.async_read()

        self._prev_observation = obs_dict
        return obs_dict

    def _update_cached_rpc_state_from_step(self, step_result: Any) -> None:
        if not isinstance(step_result, Mapping):
            return
        observation = step_result.get("observation")
        if isinstance(observation, Mapping):
            self._cached_rpc_state = dict(observation)

    def _get_cached_or_live_state(self) -> dict[str, Any]:
        if self._cached_rpc_state is not None:
            state = self._cached_rpc_state
            self._cached_rpc_state = None
            return state
        if self._robot is None:
            raise RuntimeError("Franka robot client is not initialized.")
        state = self._robot.get_full_state()
        if not isinstance(state, Mapping):
            raise RuntimeError(f"Unexpected state payload from RPC server: {type(state)!r}")
        return dict(state)

    @check_if_not_connected
    def disconnect(self) -> None:
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        if self._robot is not None:
            self._robot.close()
            self._robot = None
        self._is_connected = False
        self._cached_rpc_state = None
        self._prev_observation = None
        logger.info("%s disconnected.", self)
