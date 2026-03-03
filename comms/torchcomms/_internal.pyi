#!/usr/bin/env python3
# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.

from datetime import timedelta
from typing import Any

class _BackendWrapper:
    def __init__(self, comm: Any) -> None: ...
    def get_comm(self) -> Any: ...

def _get_store(backend_name: str, name_str: str, timeout: timedelta = ...) -> Any: ...
