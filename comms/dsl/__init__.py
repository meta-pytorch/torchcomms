# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Composable send/recv framework — DSL-agnostic contract.

This package exposes the stable, backend-independent transport contract: the
user-owned p2p transport objects (:class:`NvlTransport` now; IB / Mesh
transports reserved as future intent, not implemented here) and
:func:`nvl_rendezvous`. A device schedule binds to a transport's per-peer
:class:`PeerEndpoint` and stays fabric-agnostic.
"""

from __future__ import annotations

from .transport import (
    check_transfer,
    LinkKind,
    nvl_rendezvous,
    NvlTransport,
    P2pTransport,
    PeerEndpoint,
    PeerTable,
)

__all__ = [
    "LinkKind",
    "P2pTransport",
    "PeerEndpoint",
    "PeerTable",
    "NvlTransport",
    "nvl_rendezvous",
    "check_transfer",
]
