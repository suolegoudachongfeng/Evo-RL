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

from types import SimpleNamespace

import pytest

from lerobot.utils.recording_annotations import (
    EPISODE_FAILURE,
    EPISODE_SUCCESS,
    infer_collector_policy_id,
    infer_collector_policy_version,
    normalize_episode_success_label,
    resolve_collector_policy_id,
    resolve_episode_success_label,
)


def test_normalize_episode_success_label():
    assert normalize_episode_success_label("SUCCESS") == EPISODE_SUCCESS
    assert normalize_episode_success_label("failure") == EPISODE_FAILURE
    assert normalize_episode_success_label(None) is None

    with pytest.raises(ValueError):
        normalize_episode_success_label("maybe")


def test_resolve_episode_success_label():
    assert resolve_episode_success_label("success", default_label="failure") == EPISODE_SUCCESS
    assert resolve_episode_success_label(None, default_label="failure") == EPISODE_FAILURE
    assert resolve_episode_success_label(None, default_label=None, require_label=False) is None

    with pytest.raises(ValueError):
        resolve_episode_success_label(None, default_label=None, require_label=True)


def test_infer_collector_policy_id_and_version():
    policy_cfg = SimpleNamespace(pretrained_path="org/act_v1", type="act")
    assert infer_collector_policy_id(policy_cfg) == "org/act_v1"
    assert infer_collector_policy_version(policy_cfg) == "act_v1"
    assert infer_collector_policy_id(None) == "human"
    assert infer_collector_policy_version(None) == "human"


def test_resolve_collector_policy_id():
    assert (
        resolve_collector_policy_id(
            intervention_enabled=True,
            is_intervention=True,
            selected_from_policy=True,
            policy_id="act_v1",
            human_id="human",
        )
        == "human"
    )
    assert (
        resolve_collector_policy_id(
            intervention_enabled=True,
            is_intervention=False,
            selected_from_policy=True,
            policy_id="act_v1",
            human_id="human",
        )
        == "act_v1"
    )
    assert (
        resolve_collector_policy_id(
            intervention_enabled=False,
            is_intervention=False,
            selected_from_policy=False,
            policy_id="act_v1",
            human_id="human",
        )
        == "human"
    )
