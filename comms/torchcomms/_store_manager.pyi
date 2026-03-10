#!/usr/bin/env python3
# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.

from datetime import timedelta
from typing import Any

def _create_prefix_store(prefix: str, timeout: timedelta = ...) -> Any: ...
def _dup_prefix_store(
    prefix: str, bootstrap_store: Any, timeout: timedelta = ...
) -> Any: ...
