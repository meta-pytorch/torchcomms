#!/usr/bin/env python3
# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Any

class _BackendWrapper:
    def __init__(self, comm: Any) -> None: ...
    def get_comm(self) -> Any: ...
