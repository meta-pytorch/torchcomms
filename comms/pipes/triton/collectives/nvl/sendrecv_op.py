# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Host-side wrapper for the NVLink copy-based Triton send/recv kernel.

Allocates (and caches) the symmetric memory staging buffer plus the
persistent step-state tensors, computes a kernel config, and launches.

Supports **N-rank groups** with separate ``send_peer`` and ``recv_peer``
arguments. Each rank's staging buffer is partitioned by sender rank,
matching the layout used by ``all_to_all_single.py``. The signal pad
uses a per-(peer, block) layout: TAIL at ``[peer * MBPP + block_id]``
and HEAD at ``[HEAD_OFFSET + peer * MBPP + block_id]`` where
``HEAD_OFFSET = world_size * MAX_BLOCKS_PER_PEER``.

Since ``send_peer`` and ``recv_peer`` are known at host launch time,
the host pre-resolves all staging buffer, signal pad, and step state
pointers for the specific peer pair. This preserves the direct typed
tensor argument pattern in the kernel, avoiding the store scalarization
(``STG.E.U16``) that pointer-of-pointer indirection causes.

Composing collectives from this primitive:

  * **Ring AllGather**: call ``triton_nvl_sendrecv`` with
    ``send_peer = (rank + 1) % N``, ``recv_peer = (rank - 1) % N``
    in a host loop over N-1 ring steps.
  * **Bidirectional exchange**: ``send_peer == recv_peer`` (the v1 API).

Usage patterns:

  * **2-rank P2P group**: ``triton_nvl_send``/``triton_nvl_recv`` may
    initialize lazily because the matching peers are also all ranks in the
    group.
  * **N-rank collective schedule**: ``triton_nvl_sendrecv`` may initialize
    lazily when every rank in ``group`` enters the same collective step, such
    as Ring AllGather.
  * **N-rank sparse P2P**: call ``triton_nvl_sendrecv_prepare`` on all ranks
    in ``group`` before only a subset calls ``triton_nvl_send``,
    ``triton_nvl_recv``, or ``triton_nvl_sendrecv``. The symmetric-memory
    arena is collective over ``group`` even when a later operation only uses a
    peer pair.
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch._C._distributed_c10d import _SymmetricMemory

from .sendrecv import triton_nvl_sendrecv_kernel
from .sendrecv_ws import triton_nvl_sendrecv_bidir_ws_kernel

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables (powers of 2). These match the analogous constants in
# all_to_all_single_non_contig.py and are deliberately shape-independent
# so back-to-back launches with different block counts cannot race.
# ---------------------------------------------------------------------------

_BLOCK_STRIDE_BYTES: int = int(
    os.environ.get("TRITON_NVL_BLOCK_STRIDE_BYTES", str(256 * 1024))
)
_PIPELINE_DEPTH: int = int(os.environ.get("TRITON_NVL_PIPELINE_DEPTH", "2"))
_MAX_BLOCKS_PER_PEER: int = 32
_DEFAULT_NUM_SEND_BLOCKS: int = 16
_DEFAULT_NUM_WARPS: int = 8
_DEFAULT_WS_SENDER_WARPS: int = int(os.environ.get("TRITON_NVL_WS_SENDER_WARPS", "4"))
_DEFAULT_WS_RECEIVER_WARPS: int = int(
    os.environ.get("TRITON_NVL_WS_RECEIVER_WARPS", "4")
)
_MODE_BIDIRECTIONAL: int = 0
_MODE_SEND_ONLY: int = 1
_MODE_RECV_ONLY: int = 2


class _Config(NamedTuple):
    tile_rows: int
    tile_cols: int
    num_stages: int


def _compute_config(element_size: int) -> _Config:
    return _Config(
        tile_rows=32,
        tile_cols=2048 // element_size,
        num_stages=2,
    )


def _compute_ws_config(element_size: int) -> _Config:
    tile_bytes = int(os.environ.get("TRITON_NVL_WS_TILE_BYTES", "2048"))
    return _Config(
        tile_rows=32,
        tile_cols=tile_bytes // element_size,
        num_stages=2,
    )


def _per_peer_bytes() -> int:
    return _BLOCK_STRIDE_BYTES * _PIPELINE_DEPTH * _MAX_BLOCKS_PER_PEER


# ---------------------------------------------------------------------------
# Per-group caches
#
# These dicts are not thread-safe: a multi-threaded caller racing on the
# first use of a given ``group`` could double-allocate the staging buffer
# or step state. PyTorch's torch.distributed call sites are single-threaded
# per rank in practice, so we accept the simpler dict here. If a future
# caller multiplexes threads onto a single PG, wrap these in a lock.
# ---------------------------------------------------------------------------

_HANDLE_CACHE: dict[tuple[dist.ProcessGroup, torch.device], _SymmetricMemory] = {}

_STEP_STATE_CACHE: dict[
    tuple[dist.ProcessGroup, torch.device],
    tuple[torch.Tensor, torch.Tensor],
] = {}


def _required_signal_pad_bytes(world_size: int) -> int:
    return 2 * world_size * _MAX_BLOCKS_PER_PEER * 8  # int64 = 8 bytes


def _is_prepared(group: dist.ProcessGroup, device: torch.device) -> bool:
    return (group, device) in _HANDLE_CACHE


def _get_staging_buffer(
    group: dist.ProcessGroup,
    device: torch.device,
    world_size: int,
) -> _SymmetricMemory:
    """Get or create the symmetric memory handle for ``group``.

    Staging is sized for N-rank: ``per_peer_bytes * world_size``.

    The signal-pad sufficiency check runs only on the first allocation
    for a given ``(group, device)``. Callers must call
    ``symm_mem.set_signal_pad_size()`` before the first launch on any
    process group; lazy reconfiguration after the first launch will not
    be re-validated for already-cached groups.
    """
    cache_key = (group, device)
    cached = _HANDLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    num_bytes = _per_peer_bytes() * world_size
    logger.info(
        "triton_nvl_sendrecv staging cache miss on PG %s: allocating %d bytes (%d ranks)",
        group.group_desc,
        num_bytes,
        world_size,
    )

    required_sig = _required_signal_pad_bytes(world_size)
    current_sig = symm_mem.get_signal_pad_size()
    if current_sig < required_sig:
        raise RuntimeError(
            f"Signal pad too small for {world_size}-rank group with "
            f"{_MAX_BLOCKS_PER_PEER} blocks per peer: need {required_sig} bytes, "
            f"have {current_sig}. Call symm_mem.set_signal_pad_size() before "
            f"first allocation."
        )

    raw = symm_mem.empty(num_bytes, dtype=torch.uint8, device=device)
    handle = symm_mem.rendezvous(raw, group=group)
    # Defensive: explicitly zero the signal pad before any block can poll it.
    # Both ranks must complete the zero before either issues a wait_ge — the
    # rendezvous-then-barrier sequence below guarantees that.
    handle.get_signal_pad(handle.rank).zero_()
    dist.barrier(group)
    _HANDLE_CACHE[cache_key] = handle
    return handle


def _get_step_state(
    group: dist.ProcessGroup,
    device: torch.device,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get or create persistent monotonic step counters for ``group``.

    Shape: ``(world_size, MAX_BLOCKS_PER_PEER)`` — one counter per
    (peer, block) pair.
    """
    cache_key = (group, device)
    cached = _STEP_STATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    shape = (world_size, _MAX_BLOCKS_PER_PEER)
    sender_step = torch.zeros(shape, dtype=torch.int64, device=device)
    recver_step = torch.zeros(shape, dtype=torch.int64, device=device)
    _STEP_STATE_CACHE[cache_key] = (sender_step, recver_step)
    return _STEP_STATE_CACHE[cache_key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def triton_nvl_sendrecv_prepare(
    *,
    group: dist.ProcessGroup,
    device: torch.device,
) -> None:
    """Collectively initialize NVLink sendrecv state for ``group``.

    All ranks in ``group`` must call this before sparse N-rank
    ``triton_nvl_send``/``triton_nvl_recv`` first use. Ring-style
    ``triton_nvl_sendrecv`` schedules where every rank participates may still
    initialize lazily on the first collective step.
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return
    _get_staging_buffer(group, device, world_size)
    _get_step_state(group, device, world_size)


def triton_nvl_sendrecv(
    send_buf: torch.Tensor,
    recv_buf: torch.Tensor,
    send_peer: int,
    recv_peer: int | None = None,
    *,
    group: dist.ProcessGroup,
    num_blocks: int = _DEFAULT_NUM_SEND_BLOCKS,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> None:
    """Bidirectional NVLink copy-based send/recv.

    Sends ``send_buf`` to ``send_peer`` and receives into ``recv_buf``
    from ``recv_peer`` (defaults to ``send_peer`` if not given).

    Both peers must issue matching operations simultaneously:
    ``send_peer`` must recv from this rank, and ``recv_peer`` must
    send to this rank.

    This API is intended as an N-rank collective step primitive. If only a
    sparse peer pair will call into this module on first use, call
    ``triton_nvl_sendrecv_prepare`` on all ranks in ``group`` first.

    .. warning::
        On a ``world_size > 2`` group, the first call lazily performs a
        symmetric-memory rendezvous that is **collective over** ``group``.
        Unlike the send-only / recv-only paths, this bidirectional entry is
        **not** guarded against sparse first use: if only a subset of ranks
        call it before any ``triton_nvl_sendrecv_prepare``, the absent ranks
        never reach the rendezvous and the participating ranks **hang** (no
        error is raised). Ring-style schedules where every rank enters the
        same step are safe; for any sparse pattern, call
        ``triton_nvl_sendrecv_prepare`` on all ranks first.

    Args:
        send_buf: Source tensor to send. Contiguous, CUDA.
        recv_buf: Destination tensor to receive into. Contiguous, CUDA.
        send_peer: Rank in ``group`` to send to.
        recv_peer: Rank in ``group`` to receive from. Defaults to
            ``send_peer`` for bidirectional exchange with a single peer.
        group: Process group. Supports any world_size >= 2.
        num_blocks: Thread blocks per direction.
        num_warps: Triton warps per block.

    Note: the lower-level launch path specializes ``send_buf.numel()`` and
    ``recv_buf.numel()`` independently, but this public v1 API intentionally
    releases only symmetric bidirectional exchange. If that constraint is
    lifted later, each distinct ``(send_numel, recv_numel)`` pair will trigger
    a kernel recompile. Ring AllGather (single shard size per process) is
    unaffected; callers driving many distinct sizes per process should consider
    reusing fixed-size scratch buffers.
    """
    if recv_peer is None:
        recv_peer = send_peer
    assert send_buf.is_cuda and recv_buf.is_cuda
    assert send_buf.is_contiguous() and recv_buf.is_contiguous()
    assert send_buf.dtype == recv_buf.dtype
    # The kernel/launch path is shaped for independent send/recv numel, but
    # asymmetric bidirectional exchange is not part of the released v1 API.
    # Lifting this requires explicit asymmetric matching, graph, and
    # back-to-back slot-reuse tests.
    assert send_buf.numel() == recv_buf.numel()
    _launch(
        send_buf,
        recv_buf,
        send_peer,
        recv_peer,
        group=group,
        num_blocks=num_blocks,
        num_warps=num_warps,
        mode=_MODE_BIDIRECTIONAL,
    )


def triton_nvl_sendrecv_ws(
    send_buf: torch.Tensor,
    recv_buf: torch.Tensor,
    send_peer: int,
    recv_peer: int | None = None,
    *,
    group: dist.ProcessGroup,
    num_blocks: int = _DEFAULT_NUM_SEND_BLOCKS,
    sender_warps: int = _DEFAULT_WS_SENDER_WARPS,
    receiver_warps: int = _DEFAULT_WS_RECEIVER_WARPS,
) -> None:
    """Warp-specialized bidirectional NVLink send/recv.

    Each CTA runs sender and receiver as concurrent TLX async tasks,
    allowing both NVLink directions to share the same SM. Default
    configuration: 4 sender warps + 4 receiver warps = 8 total.

    Notes:
        * ``sender_warps`` and ``receiver_warps`` must each be a power of
          two and sum to a power of two in ``[2, 16]`` (TLX warp-spec +
          Triton ``num_warps`` requirement, which forces
          ``sender_warps == receiver_warps``).
        * The host launcher raises ``BLOCK_STRIDE_ELEMS`` adaptively to
          lower the while-loop iteration count. This is a perf
          optimization only (fewer ``fence.acq_rel.sys`` ops, a few % at
          large messages); the kernel is correct at any iteration count.
    """
    if recv_peer is None:
        recv_peer = send_peer
    assert send_buf.is_cuda and recv_buf.is_cuda
    assert send_buf.is_contiguous() and recv_buf.is_contiguous()
    assert send_buf.dtype == recv_buf.dtype
    assert send_buf.numel() == recv_buf.numel()
    _launch(
        send_buf,
        recv_buf,
        send_peer,
        recv_peer,
        group=group,
        num_blocks=num_blocks,
        num_warps=sender_warps + receiver_warps,
        mode=_MODE_BIDIRECTIONAL,
        warp_specialized=True,
        sender_warps=sender_warps,
        receiver_warps=receiver_warps,
    )


def triton_nvl_send(
    send_buf: torch.Tensor,
    send_peer: int,
    *,
    group: dist.ProcessGroup,
    num_blocks: int = _DEFAULT_NUM_SEND_BLOCKS,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> None:
    """Send ``send_buf`` to ``send_peer`` over NVLink using copy-based staging.

    Caller is responsible for ensuring the peer rank issues a matching
    ``triton_nvl_recv`` with the same ``dtype`` and ``numel``. For sparse
    N-rank first use, all ranks in ``group`` must first call
    ``triton_nvl_sendrecv_prepare``.
    """
    assert send_buf.is_cuda and send_buf.is_contiguous()
    _launch(
        send_buf,
        send_buf,
        send_peer,
        send_peer,
        group=group,
        num_blocks=num_blocks,
        num_warps=num_warps,
        mode=_MODE_SEND_ONLY,
    )


def triton_nvl_recv(
    recv_buf: torch.Tensor,
    recv_peer: int,
    *,
    group: dist.ProcessGroup,
    num_blocks: int = _DEFAULT_NUM_SEND_BLOCKS,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> None:
    """Receive into ``recv_buf`` from ``recv_peer`` over NVLink.

    Caller is responsible for ensuring the peer rank issues a matching
    ``triton_nvl_send`` with the same ``dtype`` and ``numel``. For sparse
    N-rank first use, all ranks in ``group`` must first call
    ``triton_nvl_sendrecv_prepare``.
    """
    assert recv_buf.is_cuda and recv_buf.is_contiguous()
    _launch(
        recv_buf,
        recv_buf,
        recv_peer,
        recv_peer,
        group=group,
        num_blocks=num_blocks,
        num_warps=num_warps,
        mode=_MODE_RECV_ONLY,
    )


def _launch(
    send_buf: torch.Tensor,
    recv_buf: torch.Tensor,
    send_peer: int,
    recv_peer: int,
    *,
    group: dist.ProcessGroup,
    num_blocks: int,
    num_warps: int,
    mode: int,
    warp_specialized: bool = False,
    sender_warps: int = 4,
    receiver_warps: int = 4,
) -> None:
    assert send_buf.dtype == recv_buf.dtype, (
        f"send/recv dtype mismatch: {send_buf.dtype} vs {recv_buf.dtype}"
    )
    assert 0 < num_blocks <= _MAX_BLOCKS_PER_PEER, (
        f"num_blocks must be in (0, {_MAX_BLOCKS_PER_PEER}], got {num_blocks}"
    )
    if warp_specialized:
        # TLX warp specialization requires EACH partition's warp count to be a
        # power of two (verified by `ttg.warp_specialize`: "number of warps
        # must be a power of 2"), and Triton requires the kernel's total
        # `num_warps` to be a power of two as well. Two powers of two sum to a
        # power of two only when they are equal, so this effectively requires
        # `sender_warps == receiver_warps` drawn from {1, 2, 4, 8}. The old
        # `total % 4 == 0` check wrongly accepted splits like (3, 5), which
        # then crashed deep in the TritonTLXFixup MLIR pass.
        def _is_pow2(x: int) -> bool:
            return x >= 1 and (x & (x - 1)) == 0

        total = sender_warps + receiver_warps
        assert _is_pow2(sender_warps) and sender_warps <= 8, (
            f"sender_warps must be a power of 2 in [1, 8], got {sender_warps}"
        )
        assert _is_pow2(receiver_warps) and receiver_warps <= 8, (
            f"receiver_warps must be a power of 2 in [1, 8], got {receiver_warps}"
        )
        assert _is_pow2(total) and total <= 16, (
            f"sender_warps + receiver_warps must be a power of 2 in [2, 16] "
            f"(TLX warp-spec + Triton num_warps requirement, which forces "
            f"sender_warps == receiver_warps); got {sender_warps} + "
            f"{receiver_warps} = {total}"
        )
    else:
        assert num_warps in (4, 8), (
            f"num_warps must be 4 or 8 (only configs covered by tests); got {num_warps}"
        )

    world_size = dist.get_world_size(group)
    if world_size == 1:
        recv_buf.copy_(send_buf)
        return

    if (
        mode != _MODE_BIDIRECTIONAL
        and world_size > 2
        and not _is_prepared(group, send_buf.device)
    ):
        raise RuntimeError(
            "Sparse triton_nvl_send/triton_nvl_recv first use on an N-rank "
            "group requires triton_nvl_sendrecv_prepare(group=..., "
            "device=...) to be called by all ranks first."
        )

    local_rank = dist.get_rank(group)
    assert send_peer != local_rank, "send_peer must differ from local rank"
    assert recv_peer != local_rank, "recv_peer must differ from local rank"
    assert 0 <= send_peer < world_size
    assert 0 <= recv_peer < world_size

    handle = _get_staging_buffer(group, send_buf.device, world_size)
    sender_step_all, recver_step_all = _get_step_state(
        group, send_buf.device, world_size
    )

    config = (
        _compute_ws_config(send_buf.element_size())
        if warp_specialized
        else _compute_config(send_buf.element_size())
    )
    tile_numel = config.tile_rows * config.tile_cols
    tile_bytes = tile_numel * send_buf.element_size()
    per_peer_elems = _per_peer_bytes() // send_buf.element_size()

    orig_block_stride_elems = _BLOCK_STRIDE_BYTES // send_buf.element_size()
    if warp_specialized:
        # Adaptive stride (PERF optimization only). The WS kernel is now
        # correct at any loop-iteration count (per-task compiler-sized
        # barriers via tl.debug_barrier(); see sendrecv_ws.py). Raising the
        # per-block stride lowers the iteration count, which reduces the
        # number of `fence.acq_rel.sys` operations (one per step) and buys a
        # few % at large messages (e.g. 64 MiB: ~1.07x at default 256 KiB
        # stride vs ~1.12x at 2 MiB stride). It is NOT required for
        # correctness, so there is no fail-loud guard here anymore.
        # `max_ws_stride` caps per-block staging so the sum across all
        # `num_blocks` blocks fits within the per-peer staging quota
        # `_BLOCK_STRIDE_BYTES * _MAX_BLOCKS_PER_PEER` (and thus within
        # `staging_elems` below); enforces no buffer overflow even when
        # the requested stride to land a low iter count is large.
        max_ws_stride = _BLOCK_STRIDE_BYTES * _MAX_BLOCKS_PER_PEER // num_blocks
        num_tiles = -(-send_buf.numel() // tile_numel)
        tiles_per_block = -(-num_tiles // num_blocks)
        target_tiles_per_step = -(-tiles_per_block // 2)
        needed_stride = target_tiles_per_step * tile_bytes
        ws_stride = max(min(needed_stride, max_ws_stride), _BLOCK_STRIDE_BYTES)
        block_stride_elems = ws_stride // send_buf.element_size()
        max_tiles_per_block_per_slot = max(ws_stride // tile_bytes, 1)
    else:
        block_stride_elems = orig_block_stride_elems
        max_tiles_per_block_per_slot = max(_BLOCK_STRIDE_BYTES // tile_bytes, 1)

    staging_elems = orig_block_stride_elems * _MAX_BLOCKS_PER_PEER * _PIPELINE_DEPTH

    # Buffer-overflow guard: the kernel writes at most
    # `num_blocks * block_stride_elems * PIPELINE_DEPTH` elements into the
    # per-peer staging region. The WS adaptive stride can raise
    # `block_stride_elems` above the default, so make the (otherwise implicit)
    # bound explicit instead of relying on the `max_ws_stride` floor-division
    # arithmetic alone.
    assert num_blocks * block_stride_elems * _PIPELINE_DEPTH <= staging_elems, (
        f"staging overflow: num_blocks={num_blocks} * "
        f"block_stride_elems={block_stride_elems} * depth={_PIPELINE_DEPTH} "
        f"> staging_elems={staging_elems}"
    )

    # --- Pre-resolve staging buffers for the specific peer pair ---
    # Sender writes to send_peer's staging area at local_rank's partition.
    # Receiver reads from local staging area at recv_peer's partition.
    send_staging_buf = handle.get_buffer(
        send_peer,
        sizes=[staging_elems],
        dtype=send_buf.dtype,
        storage_offset=local_rank * per_peer_elems,
    )
    recv_staging_buf = handle.get_buffer(
        local_rank,
        sizes=[staging_elems],
        dtype=recv_buf.dtype,
        storage_offset=recv_peer * per_peer_elems,
    )

    # --- Pre-resolve signal pad pointers for the specific peer pair ---
    # Signal layout per rank (int64 entries):
    #   [0 .. W*MBPP):       TAIL[peer * MBPP + block_id]
    #   [W*MBPP .. 2*W*MBPP): HEAD[peer * MBPP + block_id]
    head_offset = world_size * _MAX_BLOCKS_PER_PEER

    send_peer_sig = handle.get_signal_pad(send_peer).view(torch.int64)
    recv_peer_sig = handle.get_signal_pad(recv_peer).view(torch.int64)
    local_sig = handle.get_signal_pad(local_rank).view(torch.int64)

    # Each slice is bounded to exactly MBPP int64 entries (one (peer, block)
    # row). The kernel only indexes ``block_id < num_blocks <= MBPP``, so an
    # unbounded ``[start:]`` slice would silently overlap the neighbouring
    # peer's row if a future kernel change indexed past MBPP. The explicit
    # upper bound makes that a hard error instead.
    def _sig_row(pad: torch.Tensor, start: int) -> torch.Tensor:
        return pad[start : start + _MAX_BLOCKS_PER_PEER]

    # Sender: tail on send_peer written by local_rank, head on local polled for send_peer
    send_tail_sig = _sig_row(send_peer_sig, local_rank * _MAX_BLOCKS_PER_PEER)
    send_head_sig = _sig_row(local_sig, head_offset + send_peer * _MAX_BLOCKS_PER_PEER)

    # Receiver: tail on local polled for recv_peer, head on recv_peer written by local_rank
    recv_tail_sig = _sig_row(local_sig, recv_peer * _MAX_BLOCKS_PER_PEER)
    recv_head_sig = _sig_row(
        recv_peer_sig, head_offset + local_rank * _MAX_BLOCKS_PER_PEER
    )

    # --- Pre-slice step state to the peer rows ---
    sender_step_ptr = sender_step_all[send_peer]
    recver_step_ptr = recver_step_all[recv_peer]

    if warp_specialized:
        grid = (num_blocks,)
        # pyre-ignore[28]: triton's `kernel[grid](...)` launcher
        triton_nvl_sendrecv_bidir_ws_kernel[grid](
            send_ptr=send_buf,
            recv_ptr=recv_buf,
            send_staging_buf=send_staging_buf,
            recv_staging_buf=recv_staging_buf,
            send_tail_sig=send_tail_sig,
            send_head_sig=send_head_sig,
            recv_tail_sig=recv_tail_sig,
            recv_head_sig=recv_head_sig,
            sender_step_ptr=sender_step_ptr,
            recver_step_ptr=recver_step_ptr,
            send_numel=send_buf.numel(),
            recv_numel=recv_buf.numel(),
            TILE_ROWS=config.tile_rows,
            TILE_COLS=config.tile_cols,
            BLOCK_STRIDE_ELEMS=block_stride_elems,
            PIPELINE_DEPTH=_PIPELINE_DEPTH,
            MAX_TILES_PER_BLOCK_PER_SLOT=max_tiles_per_block_per_slot,
            NUM_SEND_BLOCKS=num_blocks,
            NUM_SENDER_WARPS=sender_warps,
            NUM_RECEIVER_WARPS=receiver_warps,
            num_warps=num_warps,
            # WS uses TLX async_task for pipelining; disable Triton's
            # software pipelining to avoid stage interleaving on top of
            # the explicit sender/receiver task split.
            num_stages=0,
        )
        return

    grid = (2 * num_blocks,) if mode == _MODE_BIDIRECTIONAL else (num_blocks,)

    # Buffer and signal pad pointers are passed as direct typed tensors
    # (not pointer-of-pointer int64 tensors) so Triton emits 128-bit
    # vectorized stores instead of 16-bit scalar stores. See the kernel
    # docstring in sendrecv.py for the codegen rationale.
    # pyre-ignore[28]: triton's `kernel[grid](...)` launcher accepts JIT-special
    # kwargs (`num_warps`, `num_stages`) that pyre can't see in the kernel
    # function's own signature. Not a real type error.
    triton_nvl_sendrecv_kernel[grid](
        send_ptr=send_buf,
        recv_ptr=recv_buf,
        send_staging_buf=send_staging_buf,
        recv_staging_buf=recv_staging_buf,
        send_tail_sig=send_tail_sig,
        send_head_sig=send_head_sig,
        recv_tail_sig=recv_tail_sig,
        recv_head_sig=recv_head_sig,
        sender_step_ptr=sender_step_ptr,
        recver_step_ptr=recver_step_ptr,
        send_numel=send_buf.numel(),
        recv_numel=recv_buf.numel(),
        TILE_ROWS=config.tile_rows,
        TILE_COLS=config.tile_cols,
        BLOCK_STRIDE_ELEMS=block_stride_elems,
        PIPELINE_DEPTH=_PIPELINE_DEPTH,
        NUM_SEND_BLOCKS=num_blocks,
        MAX_TILES_PER_BLOCK_PER_SLOT=max_tiles_per_block_per_slot,
        MODE=mode,
        num_warps=num_warps,
        num_stages=config.num_stages,
    )
