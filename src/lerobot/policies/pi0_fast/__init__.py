#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from .configuration_pi0_fast import PI0FastConfig

__all__ = ["PI0FastConfig", "PI0FastPolicy", "make_pi0_fast_pre_post_processors"]


def __getattr__(name: str):
    if name == "PI0FastPolicy":
        from .modeling_pi0_fast import PI0FastPolicy

        return PI0FastPolicy
    if name == "make_pi0_fast_pre_post_processors":
        from .processor_pi0_fast import make_pi0_fast_pre_post_processors

        return make_pi0_fast_pre_post_processors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
