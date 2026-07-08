# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""CuTe send/recv — minimal interface stubs.

Reserves the CuTe backend's device API so the contract is visible now; the real
CuTe kernels (mirroring the Triton ``send``/``recv`` against the same transport +
ops seam) land in the CuTe stack. Kept as plain stubs to avoid pulling the CuTe
DSL dependency into this interface diff.
"""

from __future__ import annotations

from typing import Any

_RESERVED = "framework CuTe send/recv is reserved; implemented in the CuTe stack"


def send(transport: Any, send_buf: Any, peer: int, **kwargs: Any) -> None:
    raise NotImplementedError(_RESERVED)


def recv(transport: Any, recv_buf: Any, peer: int, **kwargs: Any) -> None:
    raise NotImplementedError(_RESERVED)
