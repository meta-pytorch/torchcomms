# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""comms/dsl — NVLink all_to_all over PyTorch symmetric memory.

Exposes the user-owned transport contract: :class:`NvlTransport` +
:func:`nvl_rendezvous` (staging buffer, signal pad, and graph-safe step counters
created once via a single collective rendezvous), the device-side
:class:`PeerTable` that a fused schedule indexes by peer in-kernel, and the pre-launch
:func:`check_transfer` validator.
"""

from __future__ import annotations

from .transport import check_transfer, nvl_rendezvous, NvlTransport, PeerTable

__all__ = [
    "NvlTransport",
    "PeerTable",
    "nvl_rendezvous",
    "check_transfer",
]
