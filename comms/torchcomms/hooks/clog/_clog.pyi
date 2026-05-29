#!/usr/bin/env python3
# pyre-strict
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Any

class clog:
    """Hook for logging collective operation signatures."""

    def __init__(
        self,
        output: str,
        events: list[str],
        verbose: list[str] = ...,
    ) -> None: ...
    def register_with_comm(self, comm: Any) -> None: ...
