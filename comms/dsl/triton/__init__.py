# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Triton backend: device send/recv kernels + NVLink ops + demo hooks + launchers."""

from . import ib_ops, nvl_ops
from .hooks import copy_consume, copy_produce
from .launch import recv, send, sendrecv  # transport-agnostic host launchers
from .send_recv import recv_tiles, send_tiles  # device primitives (power users)

__all__ = [
    # host launchers (what most users call)
    "send",
    "recv",
    "sendrecv",
    # device primitives (custom schedules)
    "send_tiles",
    "recv_tiles",
    # ops + demo hooks
    "nvl_ops",
    "ib_ops",
    "copy_produce",
    "copy_consume",
]
