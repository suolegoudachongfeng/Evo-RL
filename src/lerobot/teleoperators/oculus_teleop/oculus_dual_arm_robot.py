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

from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from .oculus_reader.reader import OculusReader


class OculusDualArmRobot:
    """Helper that converts Quest controller poses into bimanual robot delta poses."""

    _T_OCULUS_TO_ROBOT = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    def __init__(
        self,
        ip: str | None,
        use_gripper: bool,
        left_pose_scaler: Sequence[float],
        left_channel_signs: Sequence[int],
        right_pose_scaler: Sequence[float],
        right_channel_signs: Sequence[int],
    ):
        self._oculus_reader = OculusReader(ip_address=ip)
        self._use_gripper = use_gripper
        self._left_pose_scaler = tuple(left_pose_scaler)
        self._left_channel_signs = tuple(left_channel_signs)
        self._right_pose_scaler = tuple(right_pose_scaler)
        self._right_channel_signs = tuple(right_channel_signs)
        self._left_prev_transform: np.ndarray | None = None
        self._right_prev_transform: np.ndarray | None = None
        self._reset_requested = False

    def close(self) -> None:
        self._oculus_reader.stop()

    def is_reset_requested(self) -> bool:
        return self._reset_requested

    def _compute_delta_pose(
        self,
        current_transform: np.ndarray,
        prev_transform: np.ndarray | None,
    ) -> np.ndarray:
        if prev_transform is None:
            return np.zeros(6, dtype=float)

        oculus_delta_pos = current_transform[:3, 3] - prev_transform[:3, 3]
        robot_delta_pos = self._T_OCULUS_TO_ROBOT @ oculus_delta_pos

        current_rot = current_transform[:3, :3]
        prev_rot = prev_transform[:3, :3]
        delta_rot_oculus = current_rot @ prev_rot.T
        oculus_delta_rotvec = R.from_matrix(delta_rot_oculus).as_rotvec()

        robot_delta_rotvec = np.array(
            [
                oculus_delta_rotvec[2],
                oculus_delta_rotvec[0],
                oculus_delta_rotvec[1],
            ],
            dtype=float,
        )

        return np.concatenate([robot_delta_pos, robot_delta_rotvec])

    @staticmethod
    def _apply_scaling(
        delta_pose: np.ndarray,
        pose_scaler: Sequence[float],
        channel_signs: Sequence[int],
    ) -> np.ndarray:
        scaled = np.zeros(6, dtype=float)
        position_scale, orientation_scale = pose_scaler[0], pose_scaler[1]
        scaled[0] = delta_pose[0] * position_scale * channel_signs[0]
        scaled[1] = delta_pose[1] * position_scale * channel_signs[1]
        scaled[2] = delta_pose[2] * position_scale * channel_signs[2]
        scaled[3] = delta_pose[3] * orientation_scale * channel_signs[3]
        scaled[4] = delta_pose[4] * orientation_scale * channel_signs[4]
        scaled[5] = delta_pose[5] * orientation_scale * channel_signs[5]
        return scaled

    def get_observations(self) -> dict[str, float | bool | None]:
        transforms, buttons = self._oculus_reader.get_transformations_and_buttons()

        lg_pressed = bool(buttons.get("LG", False))
        rg_pressed = bool(buttons.get("RG", False))
        self._reset_requested = bool(buttons.get("A", False))

        left_delta = np.zeros(6, dtype=float)
        right_delta = np.zeros(6, dtype=float)

        left_transform = transforms.get("l")
        if left_transform is not None and lg_pressed:
            left_delta = self._apply_scaling(
                self._compute_delta_pose(left_transform, self._left_prev_transform),
                self._left_pose_scaler,
                self._left_channel_signs,
            )
            self._left_prev_transform = left_transform.copy()
        else:
            self._left_prev_transform = None

        right_transform = transforms.get("r")
        if right_transform is not None and rg_pressed:
            right_delta = self._apply_scaling(
                self._compute_delta_pose(right_transform, self._right_prev_transform),
                self._right_pose_scaler,
                self._right_channel_signs,
            )
            self._right_prev_transform = right_transform.copy()
        else:
            self._right_prev_transform = None

        obs: dict[str, float | bool | None] = {}
        for i, axis in enumerate(("x", "y", "z", "rx", "ry", "rz")):
            obs[f"left_delta_ee_pose.{axis}"] = float(left_delta[i])
            obs[f"right_delta_ee_pose.{axis}"] = float(right_delta[i])

        if self._use_gripper:
            left_trigger = buttons.get("leftTrig", (0.0,))
            right_trigger = buttons.get("rightTrig", (0.0,))
            lt_value = left_trigger[0] if isinstance(left_trigger, tuple) and left_trigger else 0.0
            rt_value = right_trigger[0] if isinstance(right_trigger, tuple) and right_trigger else 0.0
            obs["left_gripper_cmd_bin"] = float(1.0 - lt_value)
            obs["right_gripper_cmd_bin"] = float(1.0 - rt_value)
        else:
            obs["left_gripper_cmd_bin"] = None
            obs["right_gripper_cmd_bin"] = None

        obs["reset_requested"] = self._reset_requested
        return obs
