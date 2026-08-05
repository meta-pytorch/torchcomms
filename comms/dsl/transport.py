# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""User-owned p2p transport objects for the composable send/recv framework.

A transport owns all transport state (staging buffer + signal pads for the
symmetric-memory case) behind an abstraction. A device schedule consumes a
per-peer :class:`PeerEndpoint` resolved from a transport and never sees whether
it is NVLink or IB.

Design points realized here:

* **User-owned, one rendezvous.** :func:`nvl_rendezvous` does a single collective
  rendezvous for the whole group and returns an object the user holds (reused
  across CUDA-graph replays). There is no hidden module-level cache.
* **Mixed transport (future).** A ``MeshTransport`` would compose an intra-domain NVLink
  transport with a (future) inter-domain IB transport and route per peer via
  :meth:`P2pTransport.link_kind`, so a single collective could mix transports. Reserved
  intent only -- not implemented here (see the future-note at the end of this module).

Device transport-ops seam: a transport's only transport-specific device code is four
primitives -- ``put`` (produced tile -> peer region), ``get`` (received tile <- region),
``signal`` (publish "data ready"), ``wait`` (await a peer's data). A device schedule is
written once against this seam and treats the per-peer buffers as opaque, so one schedule
serves NVLink today and IB later, and a mixed-transport kernel can bind different ops per
call site. There is no runtime dispatch -- the concrete ops are passed as compile-time
constants; implementations live per transport + DSL.

NVLink is implemented for real (minimal) here; IB and the mixed-transport mesh are reserved
seams (the future-fabric interface contract, raising until their stack lands).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch._C._distributed_c10d import _SymmetricMemory

logger: logging.Logger = logging.getLogger(__name__)

_SR_DEBUG: bool | None = None


def _sr_debug() -> bool:
    global _SR_DEBUG
    if _SR_DEBUG is None:
        _SR_DEBUG = os.environ.get("SENDRECV_DEBUG", "0") == "1"
    return _SR_DEBUG


def _rdv_dbg(group: dist.ProcessGroup, msg: str) -> None:
    if _sr_debug():
        logger.debug("[r%d] rdv: %s", dist.get_rank(group), msg)


# Shape-independent default; the staging region per peer must hold the message.
_DEFAULT_MAX_BLOCKS_PER_PEER: int = 32

# Floor for the total symmetric-memory allocation (see nvl_rendezvous): GB300's NVL72
# fabric rendezvous deadlocks on sub-MB allocations, so always allocate at least this much.
_MIN_SYMM_TOTAL_BYTES: int = 4 * 1024 * 1024


class LinkKind(Enum):
    """How a given peer is reached. NVLINK is the only implemented kind today; IB is reserved
    for the future IB (torchcomms window) transport and has no backing implementation yet."""

    NVLINK = "nvlink"
    IB = "ib"


@dataclass(frozen=True)
class PeerEndpoint:
    """Host-resolved per-peer device state for one peer.

    The four handles are passed directly to a device schedule (typed tensors, not
    pointer-of-pointer indirection). Today each is a ``torch.Tensor`` (NVLink:
    symm-mem-mapped); an IB transport will carry a ``(remote_addr, rkey)``
    descriptor instead — a field-type change behind this same struct. The
    ``send_dst``/``signal_dst`` pair drives this rank's writes to ``peer``; the
    ``recv_src``/``signal_src`` pair drives this rank's reads of what ``peer`` wrote.
    """

    send_dst: torch.Tensor  # WHERE this rank writes for the peer (remote)
    recv_src: torch.Tensor  # WHERE this rank reads what the peer wrote (local)
    signal_dst: torch.Tensor  # signal this rank raises at the peer
    signal_src: torch.Tensor  # signal this rank waits on (the peer raises it)


@dataclass(frozen=True)
class PeerTable:
    """Device-side addressing for fused multi-peer schedules.

    The multi-peer analogue of :class:`PeerEndpoint` (host-resolved, single
    peer): every peer's staging-buffer and signal-pad base pointer as an int64
    device tensor, indexed by peer id (cast int->ptr) inside the kernel. Used by
    a fused multi-peer schedule that picks the peer on device via ``program_id``
    instead of pre-slicing one ``PeerEndpoint`` per peer on the host.
    """

    buffer_ptrs: torch.Tensor  # int64[world_size]: peer -> staging-buffer base
    signal_pad_ptrs: torch.Tensor  # int64[world_size]: peer -> signal-pad base


@runtime_checkable
class P2pTransport(Protocol):
    """The transport abstraction a device schedule depends on (NVLink now, IB later)."""

    world_size: int
    per_peer_bytes: int
    max_blocks_per_peer: int

    def link_kind(self, peer: int) -> LinkKind: ...

    def endpoint(self, peer: int, *, dtype: torch.dtype) -> PeerEndpoint: ...

    def endpoints_device(self) -> PeerTable: ...


@dataclass
class NvlTransport:
    """Symmetric-memory NVLink transport (real, minimal).

    Owns the ``_SymmetricMemory`` handle. The buffer is ``per_peer_bytes`` per
    peer (``per_peer_bytes * world_size`` total); the signal pad holds two int64
    regions of ``world_size * max_blocks_per_peer`` entries each (tail then head),
    laid out by ``[sender_rank * max_blocks_per_peer + block]``. Tail = sender
    "data ready"; head = receiver "slot consumed" (so a sender never overwrites a
    slot a receiver has not drained).
    """

    handle: _SymmetricMemory
    world_size: int
    # Rank within the NVLink rendezvous group (``dist.get_rank(group)``), i.e. the
    # index used to address this rank's slot in the symmetric-memory buffer. This is
    # NOT the node-local rank (``LOCAL_RANK`` env / ``get_node_local_rank``): on an
    # NVL fabric the rendezvous group may span trays, so it is a group-relative rank.
    local_rank: int
    per_peer_bytes: int
    max_blocks_per_peer: int = _DEFAULT_MAX_BLOCKS_PER_PEER
    _endpoints_device_cache: PeerTable | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _step_state_cache: tuple[torch.Tensor, torch.Tensor] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def link_kind(self, peer: int) -> LinkKind:
        return LinkKind.NVLINK

    def endpoint(self, peer: int, *, dtype: torch.dtype) -> PeerEndpoint:
        if not (0 <= peer < self.world_size) or peer == self.local_rank:
            raise ValueError(
                f"peer={peer} must be in [0, {self.world_size}) and "
                f"!= local_rank={self.local_rank}"
            )
        elem = dtype.itemsize
        if self.per_peer_bytes % elem != 0:
            raise ValueError(
                f"per_peer_bytes={self.per_peer_bytes} not a multiple of {elem} "
                f"(dtype={dtype})"
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

    def endpoints_device(self) -> PeerTable:
        """All peers' buffer + signal-pad base pointers as a device :class:`PeerTable`.

        The device-side counterpart of :meth:`endpoint` for a single fused
        multi-peer kernel: index by peer on device (cast int->ptr) instead of
        pre-slicing one :class:`PeerEndpoint` per peer. The per-peer staging
        region for sender ``s`` inside a rank's buffer is at
        ``s * (per_peer_bytes // elem)`` (same convention as :meth:`endpoint`),
        and signal-pad slots for sender ``s`` start at ``s * max_blocks_per_peer``.
        """
        if self._endpoints_device_cache is None:
            # Symm-mem base addresses are fixed after rendezvous, so build once and
            # cache: repeated fused-schedule calls and CUDA-graph capture must not
            # re-allocate / re-copy. Device is the transport's own symm-mem device,
            # not the caller's current device.
            dev = self.handle.get_buffer(
                self.local_rank, sizes=[1], dtype=torch.uint8
            ).device
            self._endpoints_device_cache = PeerTable(
                buffer_ptrs=torch.tensor(
                    self.handle.buffer_ptrs, dtype=torch.int64, device=dev
                ),
                signal_pad_ptrs=torch.tensor(
                    self.handle.signal_pad_ptrs, dtype=torch.int64, device=dev
                ),
            )
        return self._endpoints_device_cache

    def step_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Persistent monotonic step counters for graph-safe signalling.

        Returns ``(sender_step, recver_step)``, each an int64 device tensor of
        ``world_size * max_blocks_per_peer`` entries indexed by
        ``peer * max_blocks_per_peer + block``. Each slot has exactly one writer
        (the local send leg writes ``sender_step``; the local recv leg writes
        ``recver_step``), so no atomics are needed.

        The schedule signals an absolute, monotonically increasing sequence
        number per call and the kernel advances these counters itself, so a
        reused transport / CUDA-graph replay never reads a stale signal slot.
        Allocated once and cached (zero-initialized; the signal pad is zeroed at
        rendezvous, so the first ``wait_ge(_, 1)`` is race-free).
        """
        if self._step_state_cache is None:
            dev = self.handle.get_buffer(
                self.local_rank, sizes=[1], dtype=torch.uint8
            ).device
            n = self.world_size * self.max_blocks_per_peer
            self._step_state_cache = (
                torch.zeros(n, dtype=torch.int64, device=dev),
                torch.zeros(n, dtype=torch.int64, device=dev),
            )
        return self._step_state_cache


def nvl_rendezvous(
    group: dist.ProcessGroup,
    device: torch.device,
    per_peer_bytes: int,
    max_blocks_per_peer: int = _DEFAULT_MAX_BLOCKS_PER_PEER,
) -> NvlTransport:
    """One collective rendezvous; returns a user-owned :class:`NvlTransport`.

    Allocates ``per_peer_bytes * world_size`` of symmetric memory (one per-peer
    staging region per peer), floored at ``_MIN_SYMM_TOTAL_BYTES`` (4 MiB) because
    the GB300 NVL72 fabric rendezvous deadlocks on sub-MB allocations -- so a small
    transfer allocates more than the raw product. Zeroes the signal pad so the first
    ``wait(_, 1)`` is race-free after the barrier.
    """
    world_size = dist.get_world_size(group)
    local_rank = dist.get_rank(group)
    total = per_peer_bytes * world_size
    # GB300 NVL72 symm-mem fabric rendezvous DEADLOCKS on a tiny allocation: a 32B-per-peer
    # (64B total) transfer wedges the receiver inside symm_mem.rendezvous() while the sender
    # returns (pinpointed via SENDRECV_DEBUG -- rank1 freezes at symm_mem.rendezvous start).
    # Floor the symm-mem allocation to a fabric-safe size. The per-peer staging math
    # (cap_elems = per_peer_bytes // elem) is unchanged, so the extra bytes are unused
    # headroom and the measured transfer size is unaffected; the signal pad is sized
    # separately. H100 does not need this (it handles tiny allocations fine).
    total = max(total, _MIN_SYMM_TOTAL_BYTES)
    logger.info(
        "nvl_rendezvous on PG %s: allocating %d bytes (%d peers x %d)",
        group.group_desc,
        total,
        world_size,
        per_peer_bytes,
    )
    _rdv_dbg(group, f"symm_mem.empty({total}B) start")
    raw = symm_mem.empty(total, dtype=torch.uint8, device=device)
    _rdv_dbg(group, "symm_mem.empty done; symm_mem.rendezvous start")
    handle = symm_mem.rendezvous(raw, group=group)
    _rdv_dbg(group, "symm_mem.rendezvous done")

    # Signal pad holds two int64 regions per (sender, block) slot:
    #   tail [0 .. ws*MBP): sender publishes "data ready" (polled by receiver)
    #   head [ws*MBP .. 2*ws*MBP): receiver publishes "slot consumed" (polled by
    #       sender, so a sender never overwrites a slot the receiver has not yet
    #       drained -> back-to-back / reuse is race-free).
    need = 2 * world_size * max_blocks_per_peer
    sig = handle.get_signal_pad(handle.rank).view(torch.int64)
    if sig.numel() < need:
        raise RuntimeError(
            f"signal pad too small: {sig.numel()} int64 < required {need} "
            f"(2 x world_size={world_size} x max_blocks_per_peer={max_blocks_per_peer})"
        )
    sig.zero_()
    _rdv_dbg(group, "sig.zero_ done; barrier start")
    dist.barrier(group)
    _rdv_dbg(group, "barrier done")
    transport = NvlTransport(
        handle=handle,
        world_size=world_size,
        local_rank=local_rank,
        per_peer_bytes=per_peer_bytes,
        max_blocks_per_peer=max_blocks_per_peer,
    )
    # Eagerly materialize the device-side caches (peer table + step counters) now,
    # so the first collective can be issued inside a CUDA-graph capture without the
    # lazy allocation happening during capture.
    transport.endpoints_device()
    _rdv_dbg(group, "endpoints_device done")
    transport.step_state()
    _rdv_dbg(group, "step_state done")
    return transport


def check_transfer(
    transport: P2pTransport,
    numel: int,
    dtype: torch.dtype,
    num_blocks: int,
) -> None:
    """Validate a transfer against the transport before launch (fail loud).

    Three invariants whose violation would otherwise cause **silent remote
    corruption** (a kernel writing past a per-peer region overruns the next
    peer's staging or signal-pad region on the remote rank):

    * ``per_peer_bytes`` must be a whole multiple of the dtype itemsize (mirrors
      ``endpoint()``), so the per-peer region holds whole elements.
    * ``numel`` must fit the per-peer staging region (``per_peer_bytes``).
    * ``num_blocks`` must fit the per-peer signal-pad slots
      (``max_blocks_per_peer``), since each block signals slot ``block_id``.
    """
    elem = dtype.itemsize
    # Mirror endpoint()'s divisibility guard so pre-launch validation is complete: a per-peer
    # region that is not a whole number of elements would otherwise pass here and fail later
    # inside endpoint() (or silently mis-slice the symm-mem buffer).
    if transport.per_peer_bytes % elem != 0:
        raise ValueError(
            f"per_peer_bytes={transport.per_peer_bytes} not a multiple of dtype itemsize "
            f"{elem} ({dtype}); the per-peer region cannot hold whole elements"
        )
    cap_elems = transport.per_peer_bytes // elem
    if numel > cap_elems:
        raise ValueError(
            f"transfer numel={numel} exceeds per-peer capacity={cap_elems} elems "
            f"(per_peer_bytes={transport.per_peer_bytes}, dtype={dtype}); "
            f"increase per_peer_bytes at rendezvous"
        )
    mbp = transport.max_blocks_per_peer
    if not (1 <= num_blocks <= mbp):
        raise ValueError(
            f"num_blocks={num_blocks} must be in [1, {mbp}] "
            f"(one signal-pad slot per block per peer)"
        )


# ---------------------------------------------------------------------------
# Future: IB / Mesh transports.
#
# The intent is retained but not implemented here: an IB (torchcomms window)
# transport and a mixed-transport ``MeshTransport`` (intra-domain NVLink composed
# with inter-domain IB, routed per peer via ``P2pTransport.link_kind``) would slot
# onto the same ``P2pTransport`` protocol + four-op device seam above, so device
# schedules stay fabric-agnostic. Not shipped in this framework (no kernel uses it);
# ``LinkKind.IB`` is the reserved kind for the IB transport, and a mesh kind would be
# added to ``LinkKind`` when ``MeshTransport`` lands (only NVLINK/IB exist today).
# ---------------------------------------------------------------------------
