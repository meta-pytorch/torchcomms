# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Composable send/recv framework — DSL-agnostic contract.

This package exposes the stable, backend-independent contract:

* the hook contract :class:`Ctx` (+ ``ProduceFn`` / ``ConsumeFn`` aliases),
* the user-owned p2p transport objects (:class:`NvlTransport` now,
  :class:`IbTransport` / :class:`MeshTransport` reserved) and
  :func:`nvl_rendezvous`.

Device ``send``/``recv`` live in the per-DSL subpackages
(``comms.dsl.triton``, ``comms.dsl.cute``).
"""

from __future__ import annotations

from .ctx import ConsumeFn, Ctx, ProduceFn
from .transport import (
    check_transfer,
    ib_rendezvous,
    IbTransport,
    LinkKind,
    MeshTransport,
    nvl_rendezvous,
    NvlTransport,
    P2pTransport,
    PeerEndpoint,
)

__all__ = [
    "Ctx",
    "ProduceFn",
    "ConsumeFn",
    "LinkKind",
    "P2pTransport",
    "PeerEndpoint",
    "NvlTransport",
    "nvl_rendezvous",
    "check_transfer",
    "IbTransport",
    "ib_rendezvous",
    "MeshTransport",
]
