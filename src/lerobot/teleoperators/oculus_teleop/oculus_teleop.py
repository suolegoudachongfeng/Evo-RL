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

from typing import Any

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from .config_oculus_teleop import OculusTeleopConfig
from .oculus_dual_arm_robot import OculusDualArmRobot


class OculusTeleop(Teleoperator):
    """Dual-arm teleoperator that reads both Quest controllers and outputs bimanual delta EE actions."""

    config_class = OculusTeleopConfig
    name = "oculus_teleop"

    def __init__(self, config: OculusTeleopConfig):
        super().__init__(config)
        self.config = config
        self.oculus_robot: OculusDualArmRobot | None = None
        self._is_connected = False

    @property
    def action_features(self) -> dict:
        features: dict[str, type] = {}
        for arm in ("left", "right"):
            for axis in ("x", "y", "z", "rx", "ry", "rz"):
                features[f"{arm}_delta_ee_pose.{axis}"] = float
        if self.config.use_gripper:
            features["left_gripper_cmd_bin"] = float
            features["right_gripper_cmd_bin"] = float
        features["reset_requested"] = bool
        return features

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.oculus_robot = OculusDualArmRobot(
            ip=self.config.ip,
            use_gripper=self.config.use_gripper,
            left_pose_scaler=self.config.left_pose_scaler,
            left_channel_signs=self.config.left_channel_signs,
            right_pose_scaler=self.config.right_pose_scaler,
            right_channel_signs=self.config.right_channel_signs,
        )
        self._is_connected = True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        if self.oculus_robot is None:
            raise RuntimeError("Oculus teleoperator is not initialized.")
        return self.oculus_robot.get_observations()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    def disconnect(self) -> None:
        if self.oculus_robot is not None:
            self.oculus_robot.close()
            self.oculus_robot = None
        self._is_connected = False
