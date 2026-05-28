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

import pytest

from lerobot.robots.franka_dual_arm import franka_dual_arm as franka_mod
from lerobot.robots.franka_dual_arm.config_franka_dual_arm import FrankaDualArmConfig
from lerobot.robots.franka_dual_arm.franka_dual_arm import FrankaDualArm
from lerobot.robots.franka_dual_arm.franka_dual_arm_client import FrankaDualArmClient


class FakeFrankaClient:
    instances: list["FakeFrankaClient"] = []

    def __init__(self, ip: str, port: int, timeout: float):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.calls = []
        FakeFrankaClient.instances.append(self)

    def _state(self) -> dict:
        return {
            "left_arm": {
                "robot_state": {
                    "joint_positions": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    "eef_pose": {
                        "position": [0.1, 0.2, 0.3],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                "gripper": 0.25,
            },
            "right_arm": {
                "robot_state": {
                    "joint_positions": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                    "eef_pose": {
                        "position": [0.4, 0.5, 0.6],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                "gripper": {
                    "position": 0.4,
                    "open_position": 0.0,
                    "closed_position": 0.8,
                },
            },
        }

    def ping(self) -> dict:
        self.calls.append(("ping",))
        return {"ok": True}

    def gripper_initialize(self) -> dict:
        self.calls.append(("gripper_initialize",))
        return {"left": {"ok": True}, "right": {"ok": True}}

    def left_gripper_goto(self, **kwargs) -> dict:
        self.calls.append(("left_gripper_goto", kwargs))
        return {"ok": True}

    def right_gripper_goto(self, **kwargs) -> dict:
        self.calls.append(("right_gripper_goto", kwargs))
        return {"ok": True}

    def step(self, action: dict) -> dict:
        self.calls.append(("step", action))
        return {"observation": self._state(), "reward": 0.0, "done": False, "info": {}}

    def get_full_state(self) -> dict:
        self.calls.append(("get_full_state",))
        return self._state()

    def go_home(self, side: str, duration_sec: float | None, rate_hz: float | None) -> dict:
        self.calls.append(("go_home", side, duration_sec, rate_hz))
        return {"ok": True}

    def reset(self) -> dict:
        self.calls.append(("reset",))
        return self._state()

    def close(self) -> None:
        self.calls.append(("close",))


@pytest.fixture
def fake_client(monkeypatch):
    FakeFrankaClient.instances.clear()
    monkeypatch.setattr(franka_mod, "FrankaDualArmClient", FakeFrankaClient)
    return FakeFrankaClient


def make_robot(**kwargs) -> FrankaDualArm:
    config = FrankaDualArmConfig(
        id="test_franka",
        robot_ip="10.0.0.2",
        robot_port=4242,
        rpc_timeout_sec=3.0,
        cameras={},
        **kwargs,
    )
    return FrankaDualArm(config)


def connected_robot(fake_client, **kwargs) -> tuple[FrankaDualArm, FakeFrankaClient]:
    robot = make_robot(**kwargs)
    robot.connect()
    return robot, fake_client.instances[-1]


def test_send_action_builds_single_rpc_step(fake_client):
    robot, fake = connected_robot(
        fake_client,
        debug=False,
        use_gripper=True,
        max_cartesian_delta=0.06,
        max_rotation_delta=0.08,
    )

    sent_action = robot.send_action(
        {
            "left_delta_ee_pose.x": 0.08,
            "left_delta_ee_pose.rx": 0.2,
            "right_delta_ee_pose.y": -0.03,
            "left_gripper_cmd_bin": 0.5,
            "right_gripper_cmd_bin": 1.2,
        }
    )

    step_call = [call for call in fake.calls if call[0] == "step"][-1]
    server_action = step_call[1]
    assert server_action["left_arm"]["motion"]["translation"] == [0.06, 0.0, 0.0]
    assert server_action["left_arm"]["motion"]["rotation_rotvec"] == [0.08, 0.0, 0.0]
    assert server_action["right_arm"]["motion"]["translation"] == [0.0, -0.03, 0.0]
    assert server_action["right_arm"]["gripper"]["width"] == pytest.approx(0.085)
    assert server_action["left_arm"]["gripper"]["width"] == pytest.approx(0.0425)
    assert sent_action["left_delta_ee_pose.x"] == 0.06
    assert sent_action["left_delta_ee_pose.rx"] == 0.08
    assert sent_action["right_gripper_cmd_bin"] == 1.0


def test_observation_flattens_nested_rpc_state(fake_client):
    robot, _fake = connected_robot(fake_client, use_gripper=True)

    obs = robot.get_observation()

    assert obs["left_joint_2.pos"] == 0.1
    assert obs["right_joint_1.pos"] == 1.0
    assert obs["left_ee_pose.x"] == 0.1
    assert obs["right_ee_pose.z"] == 0.6
    assert obs["left_gripper_cmd_bin"] == 0.25
    assert obs["right_gripper_cmd_bin"] == 0.5


def test_step_observation_cache_is_consumed_once(fake_client):
    robot, fake = connected_robot(fake_client, debug=False, use_gripper=False)
    robot.send_action({"left_delta_ee_pose.x": 0.01})
    fake.calls.clear()

    robot.get_observation()
    assert not any(call[0] == "get_full_state" for call in fake.calls)

    robot.get_observation()
    assert any(call[0] == "get_full_state" for call in fake.calls)


class FakeRpcServer:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _method(*args):
            self.calls.append((name, args))
            if name == "get_observation":
                return {
                    "left_arm": {"gripper": 0.25},
                    "right_arm": {"gripper": {"position": 0.4}},
                }
            return {"ok": True}

        return _method


def make_rpc_client_with_fake_server() -> tuple[FrankaDualArmClient, FakeRpcServer]:
    client = FrankaDualArmClient.__new__(FrankaDualArmClient)
    fake_server = FakeRpcServer()
    client.ip = "127.0.0.1"
    client.port = 4242
    client.timeout = 30.0
    client.server_url = "tcp://127.0.0.1:4242"
    client.server = fake_server
    return client, fake_server


def test_rpc_client_methods_match_dual_franka_server_contract():
    client, fake_server = make_rpc_client_with_fake_server()

    client.ping()
    client.reset()
    client.step({"left_arm": {"motion": {"translation": [0, 0, 0], "rotation_rotvec": [0, 0, 0]}}})
    client.get_observation()
    client.get_home()
    client.set_home_current("left")
    client.save_home_current("right")
    client.go_home("both", 5.0, 50.0)
    client.recover_robot("left_arm")
    client.command_gripper("right", {"width": 0.02})
    client.open_gripper("left")
    client.close_gripper("right")
    client.reactivate_gripper("left_arm")
    client.left_gripper_goto(width=0.085, speed=0.1, force=10.0)
    client.right_gripper_goto(width=0.0, speed=0.1, force=10.0)
    client.set_left_gripper(0.5)
    client.set_right_gripper(1.0)
    client.dual_robot_move_to_ee_pose([0.01, 0, 0, 0, 0, 0], [0, 0.01, 0, 0, 0, 0])

    assert ("ping", ()) in fake_server.calls
    assert ("reset", ()) in fake_server.calls
    assert ("get_observation", ()) in fake_server.calls
    assert ("get_home", ()) in fake_server.calls
    assert ("set_home_current", ("left_arm",)) in fake_server.calls
    assert ("save_home_current", ("right_arm",)) in fake_server.calls
    assert ("go_home", ("both", 5.0, 50.0)) in fake_server.calls
    assert ("recover_robot", ("left_arm",)) in fake_server.calls
    assert ("command_gripper", ("right_arm", {"width": 0.02})) in fake_server.calls
    assert ("open_gripper", ("left_arm",)) in fake_server.calls
    assert ("close_gripper", ("right_arm",)) in fake_server.calls
    assert ("reactivate_gripper", ("left_arm",)) in fake_server.calls
    assert (
        "command_gripper",
        ("left_arm", {"width": 0.085, "max_velocity": 0.1, "max_effort": 10.0}),
    ) in fake_server.calls
    assert (
        "command_gripper",
        ("right_arm", {"width": 0.0, "max_velocity": 0.1, "max_effort": 10.0}),
    ) in fake_server.calls
    assert ("command_gripper", ("left_arm", {"normalized": 0.5})) in fake_server.calls
    assert ("command_gripper", ("right_arm", {"normalized": 1.0})) in fake_server.calls
    assert any(
        call
        == (
            "step",
            (
                {
                    "left_arm": {
                        "motion": {
                            "translation": [0.01, 0, 0],
                            "rotation_rotvec": [0, 0, 0],
                        }
                    },
                    "right_arm": {
                        "motion": {
                            "translation": [0, 0.01, 0],
                            "rotation_rotvec": [0, 0, 0],
                        }
                    },
                },
            ),
        )
        for call in fake_server.calls
    )
