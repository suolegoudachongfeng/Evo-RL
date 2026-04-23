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

from lerobot.scripts.lerobot_dataset_report import (
    _build_episode_length_histogram,
    _format_ascii_histogram,
)


def test_build_episode_length_histogram_uses_20_bins_and_preserves_counts():
    lengths = [float(v) for v in range(1, 101)]
    histogram = _build_episode_length_histogram(lengths, bins=20)

    assert len(histogram) == 20
    assert sum(bin_info["count"] for bin_info in histogram) == len(lengths)
    assert histogram[0]["start"] == 1.0
    assert histogram[-1]["end"] >= 100


def test_build_episode_length_histogram_handles_constant_length():
    lengths = [42.0, 42.0, 42.0, 42.0]
    histogram = _build_episode_length_histogram(lengths, bins=20)

    assert len(histogram) == 20
    assert sum(bin_info["count"] for bin_info in histogram) == 4
    assert sum(1 for bin_info in histogram if bin_info["count"] > 0) == 1


def test_format_ascii_histogram_renders_one_line_per_bin():
    histogram = _build_episode_length_histogram([10.0, 20.0, 30.0, 40.0], bins=20)
    lines = _format_ascii_histogram(histogram)

    assert len(lines) == 20
    assert all("|" in line for line in lines)
    assert all("s]" in line for line in lines)
