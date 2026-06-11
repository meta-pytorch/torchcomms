# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""User-owned p2p transport objects for the composable send/recv framework.

A transport owns all transport state (staging buffer + signal pads for the
symmetric-memory case) behind an abstraction. The device ``send``/``recv``
primitives consume a per-peer :class:`PeerEndpoint` resolved from a transport and
never see whether it is NVLink or IB.

Design points realized here:

* **User-owned, one rendezvous.** :func:`nvl_rendezvous` does a single collective
  rendezvous for the whole group and returns an object the user holds (reused
  across CUDA-graph replays). There is no hidden module-level cache.
* **Mixed transport.** :class:`MeshTransport` composes an intra-domain NVLink
  transport with a (future) inter-domain IB transport and routes per peer via
  :meth:`P2pTransport.link_kind`, so a single collective can mix transports.

This interface diff implements NVLink for real (minimal) and reserves IB.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch._C._distributed_c10d import _SymmetricMemory

logger: logging.Logger = logging.getLogger(__name__)

# Shape-independent default; the staging region per peer must hold the message.
_DEFAULT_MAX_BLOCKS_PER_PEER: int = 32


class LinkKind(Enum):
    """How a given peer is reached."""

    NVLINK = "nvlink"
    IB = "ib"


@dataclass(frozen=True)
class PeerEndpoint:
    """Host-resolved per-peer device state, consumed by ``send``/``recv``.

    The four handles are passed directly to the kernel (typed tensors, not
    pointer-of-pointer indirection). Today each is a ``torch.Tensor`` (NVLink:
    symm-mem-mapped); an IB transport will carry a ``(remote_addr, rkey)``
    descriptor instead — a field-type change behind this same struct. ``send_dst``/``signal_dst`` are used by ``send`` (this rank ->
    ``peer``); ``recv_src``/``signal_src`` by ``recv`` (``peer`` -> this rank).
    """

    send_dst: torch.Tensor  # WHERE this rank writes for the peer (remote)
    recv_src: torch.Tensor  # WHERE this rank reads what the peer wrote (local)
    signal_dst: torch.Tensor  # signal this rank raises at the peer
    signal_src: torch.Tensor  # signal this rank waits on (the peer raises it)


@runtime_checkable
class P2pTransport(Protocol):
    """The abstraction ``send``/``recv`` depend on (NVLink now, IB later)."""

    world_size: int
    per_peer_bytes: int

    def link_kind(self, peer: int) -> LinkKind: ...

    def endpoint(self, peer: int, *, dtype: torch.dtype) -> PeerEndpoint: ...


@dataclass
class NvlTransport:
    """Symmetric-memory NVLink transport (real, minimal).

    Owns the ``_SymmetricMemory`` handle. The buffer is ``per_peer_bytes`` per
    peer (``per_peer_bytes * world_size`` total); the signal pad holds
    ``world_size * max_blocks_per_peer`` int64 entries laid out by
    ``[sender_rank * max_blocks_per_peer + block]``.
    """

    handle: _SymmetricMemory
    world_size: int
    local_rank: int
    per_peer_bytes: int
    max_blocks_per_peer: int = _DEFAULT_MAX_BLOCKS_PER_PEER

    def link_kind(self, peer: int) -> LinkKind:
        return LinkKind.NVLINK

    def endpoint(self, peer: int, *, dtype: torch.dtype) -> PeerEndpoint:
        assert 0 <= peer < self.world_size and peer != self.local_rank
        elem = torch.tensor([], dtype=dtype).element_size()
        assert self.per_peer_bytes % elem == 0, (
            f"per_peer_bytes={self.per_peer_bytes} not a multiple of {elem}"
        )
        cap_elems = self.per_peer_bytes // elem
        mbp = self.max_blocks_per_peer

        # Staging regions are indexed by SENDER rank within each rank's buffer.
        send_dst = self.handle.get_buffer(
            peer,
            sizes=[cap_elems],
            dtype=dtype,
            storage_offset=self.local_rank * cap_elems,
        )
        recv_src = self.handle.get_buffer(
            self.local_rank,
            sizes=[cap_elems],
            dtype=dtype,
            storage_offset=peer * cap_elems,
        )
        peer_sig = self.handle.get_signal_pad(peer).view(torch.int64)
        local_sig = self.handle.get_signal_pad(self.local_rank).view(torch.int64)
        signal_dst = peer_sig[self.local_rank * mbp : self.local_rank * mbp + mbp]
        signal_src = local_sig[peer * mbp : peer * mbp + mbp]
        return PeerEndpoint(send_dst, recv_src, signal_dst, signal_src)


def nvl_rendezvous(
    group: dist.ProcessGroup,
    device: torch.device,
    per_peer_bytes: int,
    max_blocks_per_peer: int = _DEFAULT_MAX_BLOCKS_PER_PEER,
) -> NvlTransport:
    """One collective rendezvous; returns a user-owned :class:`NvlTransport`.

    Allocates ``per_peer_bytes * world_size`` of symmetric memory (one per-peer
    staging region per peer) and zeroes the signal pad so the first
    ``wait(_, 1)`` is race-free after the barrier.
    """
    world_size = dist.get_world_size(group)
    local_rank = dist.get_rank(group)
    total = per_peer_bytes * world_size
    logger.info(
        "nvl_rendezvous on PG %s: allocating %d bytes (%d peers x %d)",
        group.group_desc,
        total,
        world_size,
        per_peer_bytes,
    )
    raw = symm_mem.empty(total, dtype=torch.uint8, device=device)
    handle = symm_mem.rendezvous(raw, group=group)

    need = world_size * max_blocks_per_peer
    sig = handle.get_signal_pad(handle.rank).view(torch.int64)
    assert sig.numel() >= need, (
        f"signal pad too small: {sig.numel()} int64 < required {need} "
        f"(world_size={world_size} x max_blocks_per_peer={max_blocks_per_peer})"
    )
    sig.zero_()
    dist.barrier(group)
    return NvlTransport(
        handle=handle,
        world_size=world_size,
        local_rank=local_rank,
        per_peer_bytes=per_peer_bytes,
        max_blocks_per_peer=max_blocks_per_peer,
    )


def check_transfer(
    transport: P2pTransport,
    numel: int,
    dtype: torch.dtype,
    num_blocks: int,
) -> None:
    """Validate a transfer against the transport before launch (fail loud).

    Two invariants whose violation would otherwise cause **silent remote
    corruption** (a kernel writing past a per-peer region overruns the next
    peer's staging or signal-pad region on the remote rank):

    * ``numel`` must fit the per-peer staging region (``per_peer_bytes``).
    * ``num_blocks`` must fit the per-peer signal-pad slots
      (``max_blocks_per_peer``), since each block signals slot ``block_id``.
    """
    elem = torch.tensor([], dtype=dtype).element_size()
    cap_elems = transport.per_peer_bytes // elem
    if numel > cap_elems:
        raise ValueError(
            f"transfer numel={numel} exceeds per-peer capacity={cap_elems} elems "
            f"(per_peer_bytes={transport.per_peer_bytes}, dtype={dtype}); "
            f"increase per_peer_bytes at rendezvous"
        )
    mbp = getattr(transport, "max_blocks_per_peer", None)
    if mbp is not None and not (1 <= num_blocks <= mbp):
        raise ValueError(
            f"num_blocks={num_blocks} must be in [1, {mbp}] "
            f"(one signal-pad slot per block per peer)"
        )


# ---------------------------------------------------------------------------
# Reserved: IB transport + mixed-transport mesh (interface only this diff)
# ---------------------------------------------------------------------------


@dataclass
class IbTransport:
    """Reserved IB (torchcomms window) transport. Implemented in the IB stack."""

    world_size: int
    per_peer_bytes: int

    def link_kind(self, peer: int) -> LinkKind:
        return LinkKind.IB

    def endpoint(self, peer: int, *, dtype: torch.dtype) -> PeerEndpoint:
        raise NotImplementedError(
            "IbTransport.endpoint is reserved; implemented in the IB stack"
        )


def ib_rendezvous(
    group: dist.ProcessGroup,
    device: torch.device,
    per_peer_bytes: int,
) -> IbTransport:
    raise NotImplementedError("ib_rendezvous is reserved; implemented in the IB stack")


@dataclass
class MeshTransport:
    """Composes per-link transports and routes each peer to the right one.

    Enables a single collective to mix NVLink (intra-domain) and IB
    (inter-domain). Constructible today from a single :class:`NvlTransport`
    (all peers intra); the ``inter`` IB slot and inter-domain routing are filled
    in by the IB stack.
    """

    intra: NvlTransport
    inter: IbTransport | None = None
    local_domain_size: int | None = None

    @property
    def world_size(self) -> int:
        return self.intra.world_size

    @property
    def per_peer_bytes(self) -> int:
        return self.intra.per_peer_bytes

    @property
    def max_blocks_per_peer(self) -> int:
        # Delegate so check_transfer's num_blocks guard applies to mesh routing
        # too (a mesh routes NVLINK peers through ``intra``).
        return self.intra.max_blocks_per_peer

    def _domain_size(self) -> int:
        return self.local_domain_size or self.intra.world_size

    def link_kind(self, peer: int) -> LinkKind:
        if self.inter is None:
            return LinkKind.NVLINK
        lds = self._domain_size()
        same_domain = (peer // lds) == (self.intra.local_rank // lds)
        return LinkKind.NVLINK if same_domain else LinkKind.IB

    def endpoint(self, peer: int, *, dtype: torch.dtype) -> PeerEndpoint:
        if self.link_kind(peer) is LinkKind.NVLINK:
            return self.intra.endpoint(peer, dtype=dtype)
        assert self.inter is not None
        return self.inter.endpoint(peer, dtype=dtype)
