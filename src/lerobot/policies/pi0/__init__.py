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

from .configuration_pi0 import PI0Config

__all__ = ["PI0Config", "PI0Policy", "make_pi0_pre_post_processors"]


def __getattr__(name: str):
    if name == "PI0Policy":
        from .modeling_pi0 import PI0Policy

        return PI0Policy
    if name == "make_pi0_pre_post_processors":
        from .processor_pi0 import make_pi0_pre_post_processors

        return make_pi0_pre_post_processors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
