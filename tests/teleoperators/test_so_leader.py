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

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig


def _make_bus_mock() -> MagicMock:
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect():
        bus.is_connected = False

    @contextmanager
    def _dummy_cm():
        yield

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    bus.torque_disabled.side_effect = _dummy_cm
    return bus


@pytest.fixture
def leader():
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        motors_order: list[str] = list(bus_mock.motors)
        bus_mock.sync_read.return_value = {motor: idx for idx, motor in enumerate(motors_order, 1)}
        bus_mock.sync_write.return_value = None
        bus_mock.write.return_value = None
        bus_mock.disable_torque.return_value = None
        bus_mock.is_calibrated = True
        return bus_mock

    with (
        patch(
            "lerobot.teleoperators.so_leader.so_leader.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(SO100Leader, "configure", lambda self: None),
    ):
        cfg = SO100LeaderConfig(port="/dev/null")
        teleop = SO100Leader(cfg)
        yield teleop
        if teleop.is_connected:
            teleop.disconnect()


def test_connect_disconnect(leader):
    assert not leader.is_connected
    leader.connect()
    assert leader.is_connected
    leader.disconnect()
    assert not leader.is_connected


def test_get_action(leader):
    leader.connect()
    action = leader.get_action()
    expected_keys = {f"{m}.pos" for m in leader.bus.motors}
    assert set(action.keys()) == expected_keys


def test_send_feedback(leader):
    leader.connect()
    feedback = {f"{m}.pos": i * 10 for i, m in enumerate(leader.bus.motors, 1)}
    leader.send_feedback(feedback)

    goal_pos = {m: (i + 1) * 10 for i, m in enumerate(leader.bus.motors)}
    leader.bus.sync_write.assert_called_once_with("Goal_Position", goal_pos)
