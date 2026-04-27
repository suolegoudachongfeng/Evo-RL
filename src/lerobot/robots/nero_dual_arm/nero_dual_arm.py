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

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_nero_dual_arm import NeroDualArmConfig
from .nero_interface_client import NeroDualArmClient

logger = logging.getLogger(__name__)


class NeroDualArm(Robot):
    """Dual-arm NERO robot adapter that reuses the external zerorpc control server."""

    config_class = NeroDualArmConfig
    name = "nero_dual_arm"

    def __init__(self, config: NeroDualArmConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._robot: NeroDualArmClient | None = None
        self._is_connected = False
        self._prev_observation: RobotObservation | None = None
        self._teleop_send_only_mode = False
        self._left_gripper_cmd_bin = 1.0
        self._right_gripper_cmd_bin = 1.0
        self._last_action_send_time = 0.0

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
        self._robot = NeroDualArmClient(ip=self.config.robot_ip, port=self.config.robot_port)
        if self.config.use_gripper:
            self._initialize_grippers()
        if not self._teleop_send_only_mode:
            for cam in self.cameras.values():
                cam.connect()
        self._is_connected = True
        logger.info("%s connected to %s:%s.", self, self.config.robot_ip, self.config.robot_port)

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def _initialize_grippers(self) -> None:
        if self._robot is None:
            return
        self._robot.left_gripper_goto(self.config.gripper_max_open, self.config.gripper_force)
        self._robot.right_gripper_goto(self.config.gripper_max_open, self.config.gripper_force)
        self._left_gripper_cmd_bin = 1.0
        self._right_gripper_cmd_bin = 1.0

    def _should_send_action(self) -> bool:
        interval = 1.0 / max(self.config.action_send_freq_hz, 1e-6)
        now = time.monotonic()
        if now - self._last_action_send_time < interval:
            return False
        self._last_action_send_time = now
        return True

    def _handle_gripper(self, arm_side: str, gripper_value: float) -> None:
        if not self.config.use_gripper or self._robot is None:
            return

        gripper_cmd = float(gripper_value)
        if self.config.gripper_reverse:
            gripper_cmd = self.config.gripper_max_open - gripper_cmd

        width = gripper_cmd * self.config.gripper_max_open
        if arm_side == "left":
            self._robot.left_gripper_goto(width=width, force=self.config.gripper_force)
            self._left_gripper_cmd_bin = gripper_cmd
        else:
            self._robot.right_gripper_goto(width=width, force=self.config.gripper_force)
            self._right_gripper_cmd_bin = gripper_cmd

    def _send_action_cartesian(self, action: RobotAction) -> None:
        if self._robot is None or not self._should_send_action():
            return

        left_delta = np.array(
            [action[f"left_delta_ee_pose.{axis}"] for axis in ("x", "y", "z", "rx", "ry", "rz")],
            dtype=float,
        )
        right_delta = np.array(
            [action[f"right_delta_ee_pose.{axis}"] for axis in ("x", "y", "z", "rx", "ry", "rz")],
            dtype=float,
        )

        if self.config.debug:
            return

        if np.linalg.norm(left_delta) >= 1e-3:
            self._robot.servo_p_ol("left_robot", left_delta, delta=True)
        if np.linalg.norm(right_delta) >= 1e-3:
            self._robot.servo_p_ol("right_robot", right_delta, delta=True)

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self._robot is None:
            raise RuntimeError("NERO robot client is not initialized.")

        if bool(action.get("reset_requested", False)):
            self._robot.robot_go_home()
            if self.config.use_gripper:
                self._initialize_grippers()
            return action

        self._send_action_cartesian(action)

        if "left_gripper_cmd_bin" in action:
            self._handle_gripper("left", float(action["left_gripper_cmd_bin"]))
        if "right_gripper_cmd_bin" in action:
            self._handle_gripper("right", float(action["right_gripper_cmd_bin"]))
        return action

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if self._robot is None:
            raise RuntimeError("NERO robot client is not initialized.")

        try:
            left_joint_pos = self._robot.left_robot_get_joint_positions()
            right_joint_pos = self._robot.right_robot_get_joint_positions()
            left_ee_pose = self._robot.left_robot_get_ee_pose()
            right_ee_pose = self._robot.right_robot_get_ee_pose()
        except Exception:
            logger.warning("[NERO] Failed to query observation, using previous observation.", exc_info=True)
            if self._prev_observation is not None:
                return self._prev_observation
            raise

        obs_dict: RobotObservation = {}
        for i in range(self.config.num_joints_per_arm):
            obs_dict[f"left_joint_{i + 1}.pos"] = float(left_joint_pos[i])
        for i in range(self.config.num_joints_per_arm):
            obs_dict[f"right_joint_{i + 1}.pos"] = float(right_joint_pos[i])

        # Keep the axis mapping aligned with the existing ACT/NERO project for compatibility.
        for i, axis in enumerate(("x", "y", "z", "rz", "ry", "rx")):
            obs_dict[f"left_ee_pose.{axis}"] = float(left_ee_pose[i])
        for i, axis in enumerate(("x", "y", "z", "rz", "ry", "rx")):
            obs_dict[f"right_ee_pose.{axis}"] = float(right_ee_pose[i])

        if self.config.use_gripper:
            obs_dict["left_gripper_cmd_bin"] = float(self._left_gripper_cmd_bin)
            obs_dict["right_gripper_cmd_bin"] = float(self._right_gripper_cmd_bin)

        if not self._teleop_send_only_mode:
            for cam_key, cam in self.cameras.items():
                obs_dict[cam_key] = cam.async_read()

        self._prev_observation = obs_dict
        return obs_dict

    @check_if_not_connected
    def disconnect(self) -> None:
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        if self._robot is not None:
            self._robot.close()
            self._robot = None
        self._is_connected = False
        logger.info("%s disconnected.", self)
