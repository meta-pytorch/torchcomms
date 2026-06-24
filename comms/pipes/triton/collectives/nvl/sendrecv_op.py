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

Tuning parameters are loaded from a per-kernel JSON config file
(``tuning_sendrecv.json``) keyed by ``(hardware, num_peers, msg_size)``.
Environment variables override JSON config for debugging.

Two kernel variants:

- **Stable kernel** (``sendrecv.py``): supports sub-slot signaling via
  ``signal_bytes`` for latency vs throughput trade-off.
- **WS kernel** (``sendrecv_ws.py``): warp-specialized bidirectional with
  concurrent sender/receiver TLX async tasks. No sub-slot signaling
  (TLX barrier limitation). Adaptive stride as safety clamp.

Composing collectives from this primitive:

  * **Ring AllGather**: call ``triton_nvl_sendrecv`` with
    ``send_peer = (rank + 1) % N``, ``recv_peer = (rank - 1) % N``
    in a host loop over N-1 ring steps.
  * **Bidirectional exchange**: ``send_peer == recv_peer`` (the v1 API).
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
from .tuning_config import _detect_hardware, get_sendrecv_config

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables (powers of 2). These match the analogous constants in
# all_to_all_single_non_contig.py and are deliberately shape-independent
# so back-to-back launches with different block counts cannot race.
# ---------------------------------------------------------------------------

_BLOCK_STRIDE_BYTES: int = int(
    os.environ.get("TRITON_NVL_BLOCK_STRIDE_BYTES", str(256 * 1024))
)
_PIPELINE_DEPTH: int = 2
_MAX_BLOCKS_PER_PEER: int = 32
_MODE_BIDIRECTIONAL: int = 0
_MODE_SEND_ONLY: int = 1
_MODE_RECV_ONLY: int = 2

# Sentinel defaults — when these are passed, JSON config is used instead.
_SENTINEL_NUM_BLOCKS: int = -1
_SENTINEL_NUM_WARPS: int = -1


class _Config(NamedTuple):
    tile_rows: int
    tile_cols: int
    num_stages: int


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
    return 2 * world_size * _MAX_BLOCKS_PER_PEER * 8


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
        "triton_nvl_sendrecv staging cache miss on PG %s: "
        "allocating %d bytes (%d ranks)",
        group.group_desc,
        num_bytes,
        world_size,
    )

    required_sig = _required_signal_pad_bytes(world_size)
    current_sig = symm_mem.get_signal_pad_size()
    if current_sig < required_sig:
        raise RuntimeError(
            f"Signal pad too small for {world_size}-rank group with "
            f"{_MAX_BLOCKS_PER_PEER} blocks per peer: need {required_sig} "
            f"bytes, have {current_sig}. Call symm_mem.set_signal_pad_size() "
            f"before first allocation."
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


def triton_nvl_sendrecv(
    send_buf: torch.Tensor,
    recv_buf: torch.Tensor,
    send_peer: int,
    recv_peer: int | None = None,
    *,
    group: dist.ProcessGroup,
    num_blocks: int = _SENTINEL_NUM_BLOCKS,
    num_warps: int = _SENTINEL_NUM_WARPS,
) -> None:
    """Bidirectional NVLink copy-based send/recv.

    Sends ``send_buf`` to ``send_peer`` and receives into ``recv_buf``
    from ``recv_peer`` (defaults to ``send_peer`` if not given).

    Both peers must issue matching operations simultaneously:
    ``send_peer`` must recv from this rank, and ``recv_peer`` must
    send to this rank.

    Args:
        send_buf: Source tensor to send. Contiguous, CUDA.
        recv_buf: Destination tensor to receive into. Contiguous, CUDA.
        send_peer: Rank in ``group`` to send to.
        recv_peer: Rank in ``group`` to receive from. Defaults to
            ``send_peer`` for bidirectional exchange with a single peer.
        group: Process group. Supports any world_size >= 2.
        num_blocks: Thread blocks per direction.
        num_warps: Triton warps per block.

    Note: ``send_buf.numel()`` and ``recv_buf.numel()`` are passed as
    Triton specialization constants. Each distinct ``(send_numel,
    recv_numel)`` pair triggers a kernel recompile, which is the
    intentional cost of preserving ~8.8 ms launch perf at 1 GiB. Ring
    AllGather (single shard size per process) is unaffected; callers
    driving many distinct sizes per process should consider reusing
    fixed-size scratch buffers.
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
    num_blocks: int = _SENTINEL_NUM_BLOCKS,
    sender_warps: int = -1,
    receiver_warps: int = -1,
) -> None:
    """Warp-specialized bidirectional NVLink send/recv.

    Each CTA runs sender and receiver as concurrent TLX async tasks,
    allowing both NVLink directions to share the same SM. Default
    configuration: 4 sender warps + 4 receiver warps = 8 total.

    Limitations:
        * TLX ``sync_threads()`` has an iteration-count-dependent
          barrier bug — while-loops with >2 iterations hang. The host
          launcher computes ``BLOCK_STRIDE_ELEMS`` adaptively to keep
          the iter count <=2 and raises ``RuntimeError`` if that is not
          achievable for the given (message size, ``num_blocks``,
          ``TRITON_NVL_BLOCK_STRIDE_BYTES``) combination.
          Empirically safe ranges (cap=4) with default 256KB stride:
          messages up to ~16 MiB; with ``TRITON_NVL_BLOCK_STRIDE_BYTES=2 MiB``
          up to ~64 MiB. For larger messages, fall back to
          :func:`triton_nvl_sendrecv` (the stable kernel).
        * ``sender_warps + receiver_warps`` must be a multiple of 4
          (TLX warp-specialization codegen requires it).
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
        num_warps=0,
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
    num_blocks: int = _SENTINEL_NUM_BLOCKS,
    num_warps: int = _SENTINEL_NUM_WARPS,
) -> None:
    """Send ``send_buf`` to ``send_peer`` over NVLink using copy-based staging.

    Caller is responsible for ensuring the peer rank issues a matching
    ``triton_nvl_recv`` with the same ``dtype`` and ``numel``.
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
    num_blocks: int = _SENTINEL_NUM_BLOCKS,
    num_warps: int = _SENTINEL_NUM_WARPS,
) -> None:
    """Receive into ``recv_buf`` from ``recv_peer`` over NVLink.

    Caller is responsible for ensuring the peer rank issues a matching
    ``triton_nvl_send`` with the same ``dtype`` and ``numel``.
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
    sender_warps: int = -1,
    receiver_warps: int = -1,
) -> None:
    assert send_buf.dtype == recv_buf.dtype

    world_size = dist.get_world_size(group)
    if world_size == 1:
        recv_buf.copy_(send_buf)
        return

    local_rank = dist.get_rank(group)
    assert send_peer != local_rank
    assert recv_peer != local_rank
    assert 0 <= send_peer < world_size
    assert 0 <= recv_peer < world_size

    # --- JSON config lookup ---
    msg_bytes = send_buf.numel() * send_buf.element_size()
    config = get_sendrecv_config(
        msg_bytes=msg_bytes,
        element_size=send_buf.element_size(),
        num_peers=1,
    )

    if warp_specialized:
        if config.ws is None:
            raise RuntimeError(
                f"Warp-specialized variant requested but tuning JSON has no "
                f"`ws` config for (hardware={_detect_hardware()!r}, "
                f"num_peers=1, msg_bytes={msg_bytes}). Either populate "
                f"`tuning_sendrecv.json` with a matching entry, set "
                f"`default_ws`, or call triton_nvl_sendrecv() (stable kernel) "
                f"instead."
            )
        ws_cfg = config.ws
        if num_blocks == _SENTINEL_NUM_BLOCKS:
            num_blocks = ws_cfg.num_blocks
        if sender_warps < 0:
            sender_warps = ws_cfg.sender_warps
        if receiver_warps < 0:
            receiver_warps = ws_cfg.receiver_warps

    if not warp_specialized:
        stable_cfg = config.stable
        if num_blocks == _SENTINEL_NUM_BLOCKS:
            num_blocks = stable_cfg.num_blocks
        if num_warps == _SENTINEL_NUM_WARPS:
            num_warps = stable_cfg.num_warps

    assert 0 < num_blocks <= _MAX_BLOCKS_PER_PEER, (
        f"num_blocks must be in (0, {_MAX_BLOCKS_PER_PEER}], got {num_blocks}"
    )
    if warp_specialized:
        assert 1 <= sender_warps <= 8, (
            f"sender_warps must be in [1, 8], got {sender_warps}"
        )
        assert 1 <= receiver_warps <= 8, (
            f"receiver_warps must be in [1, 8], got {receiver_warps}"
        )
        total_warps = sender_warps + receiver_warps
        # TLX requires `num_warps` to be a multiple of 4 for warp
        # specialization. (1+1=2 and similar small splits are rejected
        # downstream by the TritonTLXFixup MLIR pass.)
        assert total_warps % 4 == 0 and total_warps <= 16, (
            f"sender_warps + receiver_warps must be a multiple of 4 "
            f"in [4, 16] (TLX warp-spec requirement), "
            f"got {sender_warps} + {receiver_warps} = {total_warps}"
        )
        num_warps = total_warps
    else:
        assert num_warps in (4, 8), (
            f"num_warps must be 4 or 8 (only configs covered by tests); got {num_warps}"
        )

    handle = _get_staging_buffer(group, send_buf.device, world_size)
    sender_step_all, recver_step_all = _get_step_state(
        group, send_buf.device, world_size
    )

    element_size = send_buf.element_size()
    per_peer_elems = _per_peer_bytes() // element_size
    block_stride_elems = _BLOCK_STRIDE_BYTES // element_size

    if warp_specialized:
        assert config.ws is not None
        tile_rows = config.ws.tile_rows
        tile_cols = config.ws.tile_cols
        tile_numel = tile_rows * tile_cols
        tile_bytes = tile_numel * element_size

        # Adaptive stride: pick the smallest stride that keeps the
        # while-loop iter count <= 2 (TLX `sync_threads()` bug).
        # `max_ws_stride` caps per-block staging so the sum across all
        # `num_blocks` blocks fits within the per-peer staging quota
        # `_BLOCK_STRIDE_BYTES * _MAX_BLOCKS_PER_PEER` (and thus within
        # `staging_elems` below); enforces no buffer overflow even when
        # the requested stride to land iters<=2 is large.
        max_ws_stride = _BLOCK_STRIDE_BYTES * _MAX_BLOCKS_PER_PEER // num_blocks
        num_tiles = -(-send_buf.numel() // tile_numel)
        tiles_per_block = -(-num_tiles // num_blocks)
        target_tiles_per_step = -(-tiles_per_block // 2)
        needed_stride = target_tiles_per_step * tile_bytes
        ws_stride = max(min(needed_stride, max_ws_stride), _BLOCK_STRIDE_BYTES)
        ws_block_stride_elems = ws_stride // element_size
        max_tiles_per_block_per_slot = max(ws_stride // tile_bytes, 1)
        # Fail-loud on the TLX iteration bug: if the requested stride
        # was capped below `needed_stride`, the achievable iters/block
        # may exceed 2 and hang the kernel. Compute the realized iter
        # count and refuse to launch if it exceeds the safe bound.
        ws_iters_per_block = -(-tiles_per_block // max_tiles_per_block_per_slot)
        if ws_iters_per_block > 2:
            raise RuntimeError(
                f"triton_nvl_sendrecv_ws would require "
                f"{ws_iters_per_block} loop iterations per block (>2), "
                f"which triggers a TLX `sync_threads()` barrier bug and "
                f"hangs the kernel "
                f"(numel={send_buf.numel()}, num_blocks={num_blocks}, "
                f"BLOCK_STRIDE_BYTES={_BLOCK_STRIDE_BYTES}). Either reduce "
                f"the message size, increase num_blocks, raise "
                f"TRITON_NVL_BLOCK_STRIDE_BYTES, or fall back to "
                f"triton_nvl_sendrecv() for large messages."
            )
    else:
        tile_rows = config.stable.tile_rows
        tile_cols = config.stable.tile_cols
        tile_numel = tile_rows * tile_cols
        tile_bytes = tile_numel * element_size

        signal_bytes = config.stable.signal_bytes
        if signal_bytes > _BLOCK_STRIDE_BYTES:
            # Sweep tooling that picks `signal_bytes > _BLOCK_STRIDE_BYTES`
            # would silently lose its choice if we did not surface this.
            # Clamp + warn-once so the sweep entry can be re-tuned (or the
            # block stride raised) without producing wrong perf attribution.
            logger.warning(
                "tuning_config requested signal_bytes=%d but "
                "_BLOCK_STRIDE_BYTES=%d; clamping signal_bytes to the "
                "stride. The sweep entry under "
                "(hardware, num_peers, msg_bytes) is effectively useless "
                "until the block stride is raised.",
                signal_bytes,
                _BLOCK_STRIDE_BYTES,
            )
        signal_bytes = min(signal_bytes, _BLOCK_STRIDE_BYTES)
        # Sub-slot signaling invariants (avoid silent staging-buffer
        # corruption). The kernel writes one full tile per chunk into
        # `slot_base + chunk_in_slot * SIGNAL_STRIDE_ELEMS`; if a tile
        # is larger than a chunk, the write spills into the next chunk
        # region while that chunk may still be in-flight on the wire.
        assert signal_bytes >= tile_bytes, (
            f"signal_bytes ({signal_bytes}) must be >= tile_bytes "
            f"({tile_bytes}); a sub-slot chunk smaller than one tile "
            f"causes staging-buffer overrun across chunks."
        )
        # `chunks_per_slot` is computed as floor(stride / signal_bytes);
        # a non-divisor leaves a wasted tail in each slot (functionally
        # safe but wasteful and an indicator of bad sweep input).
        assert _BLOCK_STRIDE_BYTES % signal_bytes == 0, (
            f"_BLOCK_STRIDE_BYTES ({_BLOCK_STRIDE_BYTES}) must be an "
            f"integer multiple of signal_bytes ({signal_bytes})."
        )

    staging_elems = block_stride_elems * _MAX_BLOCKS_PER_PEER * _PIPELINE_DEPTH

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

    # Sender: tail on send_peer written by local_rank, head on local polled for send_peer
    send_tail_sig = send_peer_sig[local_rank * _MAX_BLOCKS_PER_PEER :]
    send_head_sig = local_sig[head_offset + send_peer * _MAX_BLOCKS_PER_PEER :]

    # Receiver: tail on local polled for recv_peer, head on recv_peer written by local_rank
    recv_tail_sig = local_sig[recv_peer * _MAX_BLOCKS_PER_PEER :]
    recv_head_sig = recv_peer_sig[head_offset + local_rank * _MAX_BLOCKS_PER_PEER :]

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
            TILE_ROWS=tile_rows,
            TILE_COLS=tile_cols,
            BLOCK_STRIDE_ELEMS=ws_block_stride_elems,
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

    # --- Stable kernel with sub-slot signaling ---
    tiles_per_signal = max(signal_bytes // tile_bytes, 1)
    chunks_per_slot = max(_BLOCK_STRIDE_BYTES // signal_bytes, 1)
    signal_stride_elems = signal_bytes // element_size

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
        TILE_ROWS=tile_rows,
        TILE_COLS=tile_cols,
        BLOCK_STRIDE_ELEMS=block_stride_elems,
        SIGNAL_STRIDE_ELEMS=signal_stride_elems,
        PIPELINE_DEPTH=_PIPELINE_DEPTH,
        MAX_BLOCKS_PER_PEER=_MAX_BLOCKS_PER_PEER,
        NUM_SEND_BLOCKS=num_blocks,
        TILES_PER_SIGNAL=tiles_per_signal,
        CHUNKS_PER_SLOT=chunks_per_slot,
        MODE=mode,
        num_warps=num_warps,
        num_stages=2,
    )
