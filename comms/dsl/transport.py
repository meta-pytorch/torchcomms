# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""User-owned NVLink transport for the comms/dsl all_to_all kernels.

:class:`NvlTransport` owns all transport state behind one object: the
symmetric-memory staging buffer, the signal pad, and the persistent per-(peer,
block) step counters. A device schedule consumes the device-side
:class:`PeerTable` (every peer's staging-buffer + signal-pad base pointer,
indexed by peer id in-kernel). The transport contains no collective algorithm.

Design points realized here:

* **User-owned, one rendezvous.** :func:`nvl_rendezvous` does a single collective
  rendezvous for the whole group and returns an object the user holds (reused
  across CUDA-graph replays). There is no hidden module-level cache.
* **Graph-safe signalling.** The signal pad plus persistent monotonic step
  counters let a reused transport / captured graph replay without ever reading a
  stale signal slot (see :meth:`NvlTransport.step_state`).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

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


@dataclass(frozen=True)
class PeerTable:
    """Device-side addressing for the fused multi-peer schedule.

    Every peer's staging-buffer and signal-pad base pointer as an int64 device
    tensor, indexed by peer id (cast int->ptr) inside the kernel, so a fused
    multi-peer schedule picks the peer on device via ``program_id`` instead of
    pre-slicing per-peer state on the host. The per-peer staging region for
    sender ``s`` inside a rank's buffer is at ``s * (per_peer_bytes // elem)``,
    and signal-pad slots for sender ``s`` start at ``s * max_blocks_per_peer``.
    """

    buffer_ptrs: torch.Tensor  # int64[world_size]: peer -> staging-buffer base
    signal_pad_ptrs: torch.Tensor  # int64[world_size]: peer -> signal-pad base


@dataclass
class NvlTransport:
    """Symmetric-memory NVLink transport (real, minimal).

    Owns the ``_SymmetricMemory`` handle. The buffer is ``per_peer_bytes`` per
    peer (``per_peer_bytes * world_size`` total); the signal pad holds two int64
    regions of ``world_size * max_blocks_per_peer`` entries each (tail then head),
    laid out by ``[sender_rank * max_blocks_per_peer + block]``. Tail = sender
    "data ready" (polled by the receiver); head = receiver "slot free" credit
    (polled by the sender before overwriting reusable staging).
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
    _last_a2a_launch_metadata: dict[str, Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def _record_a2a_launch_metadata(self, metadata: dict[str, Any]) -> None:
        self._last_a2a_launch_metadata = dict(metadata)

    def _get_a2a_launch_metadata(self) -> dict[str, Any]:
        if self._last_a2a_launch_metadata is None:
            raise RuntimeError("transport has no resolved a2a launch metadata")
        return dict(self._last_a2a_launch_metadata)

    def endpoints_device(self) -> PeerTable:
        """All peers' buffer + signal-pad base pointers as a device :class:`PeerTable`.

        Index by peer on device (cast int->ptr) instead of pre-slicing per-peer
        state on the host. Symm-mem base addresses are fixed after rendezvous, so
        build once and cache: repeated fused-schedule calls and CUDA-graph capture
        must not re-allocate / re-copy.
        """
        if self._endpoints_device_cache is None:
            # Device is the transport's own symm-mem device, not the caller's
            # current device.
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
        rendezvous, so the first wait for sequence 1 is race-free).
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


def _require_signal_pad(
    sig_numel: int, world_size: int, max_blocks_per_peer: int
) -> None:
    """Raise if the symm-mem signal pad cannot hold the tail+head regions.

    The pad needs two int64 regions of ``world_size * max_blocks_per_peer`` entries
    each (tail then head). Extracted so the invariant is unit-testable host-only.
    """
    need = 2 * world_size * max_blocks_per_peer
    if sig_numel < need:
        raise RuntimeError(
            f"signal pad too small: {sig_numel} int64 < required {need} "
            f"(2 x world_size={world_size} x max_blocks_per_peer={max_blocks_per_peer})"
        )


def nvl_rendezvous(
    group: dist.ProcessGroup,
    device: torch.device,
    per_peer_bytes: int,
    max_blocks_per_peer: int = _DEFAULT_MAX_BLOCKS_PER_PEER,
) -> NvlTransport:
    """One collective rendezvous; returns a user-owned :class:`NvlTransport`.

    Allocates ``per_peer_bytes * world_size`` of symmetric memory (one per-peer
    staging region per peer). Zeroes the signal pad so the first wait for sequence 1
    is race-free after the barrier.
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
    _rdv_dbg(group, f"symm_mem.empty({total}B) start")
    raw = symm_mem.empty(total, dtype=torch.uint8, device=device)
    _rdv_dbg(group, "symm_mem.empty done; symm_mem.rendezvous start")
    handle = symm_mem.rendezvous(raw, group=group)
    _rdv_dbg(group, "symm_mem.rendezvous done")

    # The transport addresses every symm-mem slot by local_rank (= group rank), so the
    # handle's own rank convention must agree, else this rank would zero / index a peer's
    # region. Enforce it (raise, not assert -- survives -O) rather than assume equality.
    if handle.rank != local_rank:
        raise RuntimeError(
            f"symm-mem handle rank {handle.rank} != group rank {local_rank}"
        )

    # Signal pad holds two int64 regions per (sender, block) slot:
    #   tail [0 .. ws*MBP): sender publishes "data ready" (polled by receiver)
    #   head [ws*MBP .. 2*ws*MBP): receiver publishes "slot free" before reuse.
    sig = handle.get_signal_pad(local_rank).view(torch.int64)
    _require_signal_pad(sig.numel(), world_size, max_blocks_per_peer)
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
    transport: NvlTransport,
    numel: int,
    dtype: torch.dtype,
    num_blocks: int,
) -> None:
    """Validate a transfer against the transport before launch (fail loud).

    Three invariants whose violation would otherwise cause **silent remote
    corruption** (a kernel writing past a per-peer region overruns the next
    peer's staging or signal-pad region on the remote rank):

    * ``per_peer_bytes`` must be a whole multiple of the dtype itemsize, so the
      per-peer region holds whole elements.
    * ``numel`` must fit the per-peer staging region (``per_peer_bytes``).
    * ``num_blocks`` must fit the per-peer signal-pad slots
      (``max_blocks_per_peer``), since each block signals slot ``block_id``.

    ``numel`` is the **per-peer** element count (the elements that must fit one peer's
    staging region), not the whole-tensor total. ``numel == 0`` (empty transfer) is
    allowed; it corrupts nothing.
    """
    elem = dtype.itemsize
    # A per-peer region that is not a whole number of elements would otherwise pass here
    # and later silently mis-slice the symm-mem buffer.
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
