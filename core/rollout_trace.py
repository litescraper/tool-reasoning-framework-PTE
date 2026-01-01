# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import functools
import inspect
from typing import Optional


def rollout_trace_op(func):
    """Decorator for rollout tracing. Simplified version without backend dependencies."""
    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        return await func(self, *args, **kwargs)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else wrapper

