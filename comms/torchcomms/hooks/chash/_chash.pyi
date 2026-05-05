#!/usr/bin/env python3
# pyre-strict
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Any

class chash:
    """Hook for computing and logging communication buffer hashes."""

    def __init__(
        self,
        output: str,
        ring_size: int = ...,
        num_blocks: int = ...,
    ) -> None: ...
    def register_with_comm(self, comm: Any) -> None: ...
