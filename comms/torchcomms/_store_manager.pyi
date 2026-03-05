#!/usr/bin/env python3
# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.

from datetime import timedelta
from typing import Any

def _create_prefixed_store(
    backend_name: str, name_str: str, timeout: timedelta = ...
) -> Any: ...
