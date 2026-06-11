# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Host launchers wiring the NVLink ops + demo hooks into the minimal kernels.

These exist to make the minimal path runnable (and to drive the 2-rank smoke
test). The user owns the transport (one rendezvous) and passes it in.

A commented mixed-transport sketch at the bottom shows the interface guarantee:
the *same* ``send``/``recv`` serve different transports per peer by passing the
matching ops at each call site.

.. warning::
    ``num_blocks`` is a cross-rank contract: both peers of an exchange MUST use
    the same ``num_blocks`` (it determines the chunk partition and the per-block
    signal slot). A mismatch silently misaligns chunks or hangs a receiver block
    waiting on a slot the sender never signals. There is no cross-rank
    validation. (Whether to bind ``num_blocks`` into the transport rendezvous is
    an open question for the pipelined follow-up.)
"""

import torch
from comms.dsl.transport import check_transfer, P2pTransport

from . import nvl_ops
from .hooks import copy_consume, copy_produce
from .send_recv import recv_tiles as _recv, send_tiles as _send

_DEFAULT_NUM_BLOCKS: int = 8
_DEFAULT_NUM_WARPS: int = 4
_DEFAULT_BLOCK: int = 2048


def send(
    transport: P2pTransport,
    send_buf: torch.Tensor,
    peer: int,
    *,
    produce=copy_produce,
    num_blocks: int = _DEFAULT_NUM_BLOCKS,
    num_warps: int = _DEFAULT_NUM_WARPS,
    block: int = _DEFAULT_BLOCK,
) -> None:
    assert send_buf.is_cuda and send_buf.is_contiguous()
    check_transfer(transport, send_buf.numel(), send_buf.dtype, num_blocks)
    ep = transport.endpoint(peer, dtype=send_buf.dtype)
    # pyre-ignore[28]: triton launcher accepts JIT-special kwargs (num_warps).
    _send[(num_blocks,)](
        send_buf,
        ep.send_dst,
        ep.signal_dst,
        send_buf.numel(),
        produce,
        nvl_ops.put,
        nvl_ops.signal,
        BLOCK=block,
        num_warps=num_warps,
    )


def recv(
    transport: P2pTransport,
    recv_buf: torch.Tensor,
    peer: int,
    *,
    consume=copy_consume,
    num_blocks: int = _DEFAULT_NUM_BLOCKS,
    num_warps: int = _DEFAULT_NUM_WARPS,
    block: int = _DEFAULT_BLOCK,
) -> None:
    assert recv_buf.is_cuda and recv_buf.is_contiguous()
    check_transfer(transport, recv_buf.numel(), recv_buf.dtype, num_blocks)
    ep = transport.endpoint(peer, dtype=recv_buf.dtype)
    # pyre-ignore[28]: triton launcher accepts JIT-special kwargs (num_warps).
    _recv[(num_blocks,)](
        recv_buf,
        ep.recv_src,
        ep.signal_src,
        recv_buf.numel(),
        consume,
        nvl_ops.get,
        nvl_ops.wait,
        BLOCK=block,
        num_warps=num_warps,
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
    **kwargs: int,
) -> None:
    """Bidirectional minimal exchange. ``send`` is non-blocking, so issuing it
    before ``recv`` does not deadlock.

    ``produce`` routes to the send leg and ``consume`` to the recv leg; the
    remaining launch kwargs (``num_blocks``/``num_warps``/``block``) go to both.
    """
    recv_peer = send_peer if recv_peer is None else recv_peer
    send(transport, send_buf, send_peer, produce=produce, **kwargs)
    recv(transport, recv_buf, recv_peer, consume=consume, **kwargs)


# ---------------------------------------------------------------------------
# Mixed-transport sketch (interface guarantee; IB impl is a follow-up).
#
# The same `send`/`recv` serve different transports per peer — the user's
# schedule branches on `link_kind` and passes the matching ops + buffers:
#
#   from comms.dsl.transport import LinkKind
#   from . import ib_ops
#   for peer in peers:
#       ep = mesh.endpoint(peer, dtype=buf.dtype)
#       if mesh.link_kind(peer) is LinkKind.NVLINK:
#           _send[grid](buf, ep.send_dst, ep.signal_dst, buf.numel(),
#                       copy_produce, nvl_ops.put, nvl_ops.signal, BLOCK=...)
#       else:  # LinkKind.IB
#           _send[grid](buf, ep.send_dst, ep.signal_dst, buf.numel(),
#                       copy_produce, ib_ops.put, ib_ops.signal, BLOCK=...)
# ---------------------------------------------------------------------------
