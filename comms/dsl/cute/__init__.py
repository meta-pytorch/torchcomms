# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""CuTe backend: minimal real send/recv over the shared transport contract."""

from . import ib_ops, nvl_ops
from .hooks import copy_consume, copy_produce
from .launch import recv, send, sendrecv
from .send_recv import recv_tiles, send_tiles  # device-transfer impl (power users)

__all__ = [
    "send",
    "recv",
    "sendrecv",
    "send_tiles",
    "recv_tiles",
    "nvl_ops",
    "ib_ops",
    "copy_produce",
    "copy_consume",
]
