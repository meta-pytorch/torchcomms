# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Host launchers for the minimal CuTe backend.

Same shape as ``framework/triton/launch.py`` (``send`` / ``recv`` /
``sendrecv`` taking a user-owned ``P2pTransport``), proving the host contract
is DSL-agnostic — only the device kernel differs.

.. warning::
    Same cross-rank contract as the Triton backend: both peers must use the same
    ``num_blocks``. Not validated here.
"""

from __future__ import annotations

import torch
from comms.dsl.transport import check_transfer, P2pTransport

from . import nvl_ops
from .hooks import copy_consume, copy_produce
from .send_recv import _NUM_THREADS, recv_tiles, send_tiles

_DEFAULT_NUM_BLOCKS: int = 8


def _check_tileable(numel: int) -> None:
    assert numel % _NUM_THREADS == 0, (
        f"cute minimal backend requires numel % {_NUM_THREADS} == 0 "
        f"(no tail handling); got numel={numel}"
    )


def send(
    transport: P2pTransport,
    send_buf: torch.Tensor,
    peer: int,
    *,
    produce=copy_produce,
    num_blocks: int = _DEFAULT_NUM_BLOCKS,
) -> None:
    assert send_buf.is_cuda and send_buf.is_contiguous()
    check_transfer(transport, send_buf.numel(), send_buf.dtype, num_blocks)
    _check_tileable(send_buf.numel())
    ep = transport.endpoint(peer, dtype=send_buf.dtype)
    staging = ep.send_dst[: send_buf.numel()]
    send_tiles(
        send_buf,
        staging,
        ep.signal_dst,
        num_blocks=num_blocks,
        hook=produce,
        put=nvl_ops.put,
        signal=nvl_ops.signal,
    )


def recv(
    transport: P2pTransport,
    recv_buf: torch.Tensor,
    peer: int,
    *,
    consume=copy_consume,
    num_blocks: int = _DEFAULT_NUM_BLOCKS,
) -> None:
    assert recv_buf.is_cuda and recv_buf.is_contiguous()
    check_transfer(transport, recv_buf.numel(), recv_buf.dtype, num_blocks)
    _check_tileable(recv_buf.numel())
    ep = transport.endpoint(peer, dtype=recv_buf.dtype)
    staging = ep.recv_src[: recv_buf.numel()]
    recv_tiles(
        recv_buf,
        staging,
        ep.signal_src,
        num_blocks=num_blocks,
        hook=consume,
        get=nvl_ops.get,
        wait=nvl_ops.wait,
    )


def sendrecv(
    transport: P2pTransport,
    send_buf: torch.Tensor,
    recv_buf: torch.Tensor,
    send_peer: int,
    recv_peer: int | None = None,
    *,
    produce=copy_produce,
    consume=copy_consume,
    num_blocks: int = _DEFAULT_NUM_BLOCKS,
) -> None:
    # Route the disjoint hooks explicitly: ``produce`` only to ``send`` and
    # ``consume`` only to ``recv`` (forwarding a shared **kwargs to both would
    # raise TypeError since each accepts only its own hook).
    recv_peer = send_peer if recv_peer is None else recv_peer
    send(transport, send_buf, send_peer, produce=produce, num_blocks=num_blocks)
    recv(transport, recv_buf, recv_peer, consume=consume, num_blocks=num_blocks)
