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

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("oculus_teleop")
@dataclass(kw_only=True)
class OculusTeleopConfig(TeleoperatorConfig):
    """Configuration for dual-arm Oculus Quest teleoperation."""

    ip: str | None = None
    use_gripper: bool = True

    left_pose_scaler: list[float] = field(default_factory=lambda: [1.0, 1.0])
    left_channel_signs: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    right_pose_scaler: list[float] = field(default_factory=lambda: [1.0, 1.0])
    right_channel_signs: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
