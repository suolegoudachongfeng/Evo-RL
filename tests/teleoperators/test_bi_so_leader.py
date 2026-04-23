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

from unittest.mock import MagicMock, patch

import pytest

from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.errors import DeviceNotConnectedError


def _make_arm_mock(name: str) -> MagicMock:
    arm = MagicMock(name=name)
    arm.is_connected = False
    arm.is_calibrated = True
    arm.action_features = {"joint.pos": float}

    def _connect(*_args, **_kwargs):
        arm.is_connected = True

    def _disconnect():
        arm.is_connected = False

    arm.connect.side_effect = _connect
    arm.disconnect.side_effect = _disconnect
    arm.get_action.return_value = {"joint.pos": 0.0}
    return arm


@pytest.fixture
def bi_leader():
    left_arm = _make_arm_mock("left_arm")
    right_arm = _make_arm_mock("right_arm")

    with patch(
        "lerobot.teleoperators.bi_so_leader.bi_so_leader.SOLeader",
        side_effect=[left_arm, right_arm],
    ):
        teleop = BiSOLeader(
            BiSOLeaderConfig(
                left_arm_config=SOLeaderConfig(port="/dev/left"),
                right_arm_config=SOLeaderConfig(port="/dev/right"),
            )
        )
        yield teleop, left_arm, right_arm


def test_set_manual_control_requires_connection(bi_leader):
    teleop, _, _ = bi_leader

    with pytest.raises(DeviceNotConnectedError):
        teleop.set_manual_control(True)


def test_set_manual_control_forwards_to_both_arms(bi_leader):
    teleop, left_arm, right_arm = bi_leader
    teleop.connect()

    teleop.set_manual_control(True)
    teleop.set_manual_control(False)

    left_arm.set_manual_control.assert_any_call(True)
    left_arm.set_manual_control.assert_any_call(False)
    right_arm.set_manual_control.assert_any_call(True)
    right_arm.set_manual_control.assert_any_call(False)
