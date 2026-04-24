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

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("nero_dual_arm")
@dataclass(kw_only=True)
class NeroDualArmConfig(RobotConfig):
    """Configuration for the dual-arm NERO follower controlled through an external RPC server."""

    robot_ip: str = "127.0.0.1"
    robot_port: int = 4242

    use_gripper: bool = True
    gripper_max_open: float = 0.1
    gripper_force: float = 2.0
    gripper_reverse: bool = False
    close_threshold: float = 0.05

    control_mode: str = "oculus"
    debug: bool = False

    num_joints_per_arm: int = 7
    action_send_freq_hz: float = 50.0

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
