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

import numpy as np

log = logging.getLogger(__name__)


class NeroDualArmClient:
    """Zerorpc client used to communicate with the existing NERO control server."""

    def __init__(self, ip: str = "127.0.0.1", port: int = 4242):
        import zerorpc

        self.ip = ip
        self.port = port
        self._zerorpc = zerorpc
        try:
            self.server = zerorpc.Client(heartbeat=20)
            self.server.connect(f"tcp://{ip}:{port}")
            log.info("[NERO CLIENT] Connected to %s:%s", ip, port)
        except Exception:
            log.exception("[NERO CLIENT] Connection failed to %s:%s", ip, port)
            self.server = None
            raise

    def left_robot_get_joint_positions(self) -> np.ndarray:
        if self.server is None:
            return np.zeros(7)
        return np.array(self.server.left_robot_get_joint_positions(), dtype=float)

    def left_robot_get_joint_velocities(self) -> np.ndarray:
        if self.server is None:
            return np.zeros(7)
        return np.array(self.server.left_robot_get_joint_velocities(), dtype=float)

    def left_robot_get_arm_status(self) -> dict:
        if self.server is None:
            return {"ctrl_mode": 0, "arm_status": 0, "motion_status": 0}
        return self.server.left_robot_get_arm_status()

    def right_robot_get_joint_positions(self) -> np.ndarray:
        if self.server is None:
            return np.zeros(7)
        return np.array(self.server.right_robot_get_joint_positions(), dtype=float)

    def right_robot_get_joint_velocities(self) -> np.ndarray:
        if self.server is None:
            return np.zeros(7)
        return np.array(self.server.right_robot_get_joint_velocities(), dtype=float)

    def right_robot_get_arm_status(self) -> dict:
        if self.server is None:
            return {"ctrl_mode": 0, "arm_status": 0, "motion_status": 0}
        return self.server.right_robot_get_arm_status()

    def left_robot_get_ee_pose(self) -> np.ndarray:
        if self.server is None:
            return np.zeros(6)
        return np.array(self.server.left_robot_get_ee_pose(), dtype=float)

    def right_robot_get_ee_pose(self) -> np.ndarray:
        if self.server is None:
            return np.zeros(6)
        return np.array(self.server.right_robot_get_ee_pose(), dtype=float)

    def left_robot_move_to_ee_pose(self, pose: np.ndarray, delta: bool = False) -> None:
        if self.server is None:
            return
        self.server.left_robot_move_to_ee_pose(pose.tolist(), delta)

    def right_robot_move_to_ee_pose(self, pose: np.ndarray, delta: bool = False) -> None:
        if self.server is None:
            return
        self.server.right_robot_move_to_ee_pose(pose.tolist(), delta)

    def dual_robot_move_to_ee_pose(self, left_pose: np.ndarray, right_pose: np.ndarray, delta: bool = False) -> None:
        if self.server is None:
            return
        self.server.dual_robot_move_to_ee_pose(left_pose.tolist(), right_pose.tolist(), delta)

    def robot_go_home(self) -> None:
        if self.server is None:
            return
        self.server.robot_go_home()

    def servo_p(self, robot_arm: str, pose: np.ndarray, delta: bool = False) -> bool:
        if self.server is None:
            return True
        return bool(self.server.servo_p(robot_arm, pose.tolist(), delta))

    def servo_p_ol(self, robot_arm: str, pose: np.ndarray, delta: bool = False) -> bool:
        if self.server is None:
            return True
        return bool(self.server.servo_p_OL(robot_arm, pose.tolist(), delta))

    def left_gripper_goto(self, width: float, force: float) -> None:
        if self.server is None:
            return
        self.server.left_gripper_goto(width, force)

    def left_gripper_get_state(self) -> dict:
        if self.server is None:
            return {"width": 0.0, "is_moving": False, "is_grasped": False}
        return self.server.left_gripper_get_state()

    def right_gripper_goto(self, width: float, force: float) -> None:
        if self.server is None:
            return
        self.server.right_gripper_goto(width, force)

    def right_gripper_get_state(self) -> dict:
        if self.server is None:
            return {"width": 0.0, "is_moving": False, "is_grasped": False}
        return self.server.right_gripper_get_state()

    def stop(self, arm_name: str) -> None:
        if self.server is None:
            return
        self.server.robot_stop(arm_name)

    def close(self) -> None:
        if self.server is None:
            return
        try:
            self.server.robot_stop("left_robot")
            self.server.robot_stop("right_robot")
        except Exception:
            log.warning("[NERO CLIENT] Failed to stop one or more arms during close.", exc_info=True)
        try:
            self.server.close()
        finally:
            self.server = None
