# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Host-side wrapper for the NVLink copy-based Triton send/recv kernel.

Allocates (and caches) the symmetric memory staging buffer plus the
persistent step-state tensors, computes a kernel config, and launches.

v1 scope (intentionally narrow):

  * **2-rank groups only**. The staging buffer holds a single peer pair's
    slot and the API takes a single ``peer_rank`` used for both directions
    (sender → ``peer_rank``, receiver ← ``peer_rank``).
  * Copy-based send-only, recv-only, and bidirectional sendrecv between
    this rank and ``peer_rank``. Peer ranks must launch matching operations
    simultaneously.

The bidirectional single-peer API is **not** directly composable into Ring AllGather on N>2 ranks —
ring needs ``send_peer = (rank+1)%N`` paired with ``recv_peer =
(rank-1)%N`` (different peers per direction), and the staging buffer
needs per-sender slots like the MSL ``all_to_all_single_non_contig``
kernel uses. The follow-up to add ring AllGather will introduce a
``triton_nvl_sendrecv(send, recv, send_peer, recv_peer, ...)`` variant
plus per-peer staging slots indexed by sender rank.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch._C._distributed_c10d import _SymmetricMemory

from .sendrecv import triton_nvl_sendrecv_kernel

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables (powers of 2). These match the analogous constants in
# all_to_all_single_non_contig.py and are deliberately shape-independent
# so back-to-back launches with different block counts cannot race.
# ---------------------------------------------------------------------------

_BLOCK_STRIDE_BYTES: int = 256 * 1024
_PIPELINE_DEPTH: int = 2
_MAX_BLOCKS_PER_DIR: int = 32
_DEFAULT_NUM_SEND_BLOCKS: int = 16
_DEFAULT_NUM_WARPS: int = 8
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


def _per_pair_bytes() -> int:
    return _BLOCK_STRIDE_BYTES * _PIPELINE_DEPTH * _MAX_BLOCKS_PER_DIR


# ---------------------------------------------------------------------------
# Per-group caches
#
# These dicts are not thread-safe: a multi-threaded caller racing on the
# first use of a given ``group`` could double-allocate the staging buffer
# or step state. PyTorch's torch.distributed call sites are single-threaded
# per rank in practice, so we accept the simpler dict here. If a future
# caller multiplexes threads onto a single PG, wrap these in a lock.
# ---------------------------------------------------------------------------

_HANDLE_CACHE: dict[dist.ProcessGroup, _SymmetricMemory] = {}

_STEP_STATE_CACHE: dict[
    dist.ProcessGroup,
    tuple[torch.Tensor, torch.Tensor],
] = {}


def _get_staging_buffer(
    group: dist.ProcessGroup,
    device: torch.device,
) -> _SymmetricMemory:
    """Get or create the symmetric memory handle for ``group``.

    The signal pad must be zero on entry to the first ``wait_ge(_, 0)`` call
    on each block. ``symm_mem.empty`` zeroes the entire allocation including
    the signal pad (via ``cudaMemset`` in CUDASymmetricMemory.cu's
    ``allocate``), so the first launch is safe. We additionally zero the
    signal pad explicitly + barrier so the invariant holds even if the
    underlying allocator behaviour changes in a future torch release.
    """
    cached = _HANDLE_CACHE.get(group)
    if cached is not None:
        return cached

    num_bytes = _per_pair_bytes()
    logger.info(
        "triton_nvl_sendrecv staging cache miss on PG %s: allocating %d bytes",
        group.group_desc,
        num_bytes,
    )
    raw = symm_mem.empty(num_bytes, dtype=torch.uint8, device=device)
    handle = symm_mem.rendezvous(raw, group=group)
    # Defensive: explicitly zero the signal pad before any block can poll it.
    # Both ranks must complete the zero before either issues a wait_ge — the
    # rendezvous-then-barrier sequence below guarantees that.
    handle.get_signal_pad(handle.rank).zero_()
    dist.barrier(group)
    _HANDLE_CACHE[group] = handle
    return handle


def _get_step_state(
    group: dist.ProcessGroup,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get or create persistent monotonic step counters for ``group``."""
    cached = _STEP_STATE_CACHE.get(group)
    if cached is not None:
        return cached

    sender_step = torch.zeros(_MAX_BLOCKS_PER_DIR, dtype=torch.int64, device=device)
    recver_step = torch.zeros(_MAX_BLOCKS_PER_DIR, dtype=torch.int64, device=device)
    _STEP_STATE_CACHE[group] = (sender_step, recver_step)
    return _STEP_STATE_CACHE[group]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def triton_nvl_sendrecv(
    send_buf: torch.Tensor,
    recv_buf: torch.Tensor,
    peer_rank: int,
    *,
    group: dist.ProcessGroup,
    num_blocks: int = _DEFAULT_NUM_SEND_BLOCKS,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> None:
    """Bidirectional NVLink copy-based send/recv between this rank and ``peer_rank``.

    Both ranks in the (sender, receiver) pair must call this function
    simultaneously. ``send_buf`` and ``recv_buf`` must have the same shape
    and dtype on both ranks.

    .. warning::
        ``dtype`` and ``numel`` of ``send_buf`` / ``recv_buf`` MUST match
        between the two ranks. There is no cross-rank validation; a
        mismatch silently produces wrong staging slot offsets and
        corrupts data.

    Args:
        send_buf: Source tensor to send to ``peer_rank``. Contiguous, CUDA.
        recv_buf: Destination tensor to receive from ``peer_rank``. Contiguous, CUDA.
        peer_rank: Rank in ``group`` to exchange with. ``send_peer`` and
            ``recv_peer`` are the same rank in v1; see the module docstring
            for why this is **not** directly extensible to Ring AllGather.
        group: Process group used for symm-mem rendezvous. **v1 requires
            size 2** — the assertion below is the enforcement point.
        num_blocks: Thread blocks per direction. Defaults to 16. The actual
            grid is ``2 * num_blocks`` for bidirectional mode (so default
            gives a 32-block grid, apple-to-apple with NCCL's 32 channels)
            and ``num_blocks`` for send-only / recv-only.
        num_warps: Triton warps per block. Defaults to 8.
    """
    assert send_buf.is_cuda and recv_buf.is_cuda
    assert send_buf.is_contiguous() and recv_buf.is_contiguous()
    assert send_buf.dtype == recv_buf.dtype
    assert send_buf.numel() == recv_buf.numel()
    _launch(
        send_buf,
        recv_buf,
        peer_rank,
        group=group,
        num_blocks=num_blocks,
        num_warps=num_warps,
        mode=_MODE_BIDIRECTIONAL,
    )


def triton_nvl_send(
    send_buf: torch.Tensor,
    peer_rank: int,
    *,
    group: dist.ProcessGroup,
    num_blocks: int = _DEFAULT_NUM_SEND_BLOCKS,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> None:
    """Send ``send_buf`` to ``peer_rank`` over NVLink using copy-based staging.

    Caller is responsible for ensuring the peer rank issues a matching
    ``triton_nvl_recv`` with the same ``dtype`` and ``numel`` — there is
    no cross-rank validation.
    """
    assert send_buf.is_cuda and send_buf.is_contiguous()
    _launch(
        send_buf,
        send_buf,
        peer_rank,
        group=group,
        num_blocks=num_blocks,
        num_warps=num_warps,
        mode=_MODE_SEND_ONLY,
    )


def triton_nvl_recv(
    recv_buf: torch.Tensor,
    peer_rank: int,
    *,
    group: dist.ProcessGroup,
    num_blocks: int = _DEFAULT_NUM_SEND_BLOCKS,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> None:
    """Receive into ``recv_buf`` from ``peer_rank`` over NVLink.

    Caller is responsible for ensuring the peer rank issues a matching
    ``triton_nvl_send`` with the same ``dtype`` and ``numel`` — there is
    no cross-rank validation.
    """
    assert recv_buf.is_cuda and recv_buf.is_contiguous()
    _launch(
        recv_buf,
        recv_buf,
        peer_rank,
        group=group,
        num_blocks=num_blocks,
        num_warps=num_warps,
        mode=_MODE_RECV_ONLY,
    )


def _launch(
    send_buf: torch.Tensor,
    recv_buf: torch.Tensor,
    peer_rank: int,
    *,
    group: dist.ProcessGroup,
    num_blocks: int,
    num_warps: int,
    mode: int,
) -> None:
    assert send_buf.dtype == recv_buf.dtype, (
        f"send/recv dtype mismatch: {send_buf.dtype} vs {recv_buf.dtype}"
    )
    assert 0 < num_blocks <= _MAX_BLOCKS_PER_DIR, (
        f"num_blocks must be in (0, {_MAX_BLOCKS_PER_DIR}], got {num_blocks}"
    )
    assert num_warps in (4, 8), (
        f"num_warps must be 4 or 8 (only configs covered by tests); got {num_warps}"
    )

    world_size = dist.get_world_size(group)
    if world_size == 1:
        recv_buf.copy_(send_buf)
        return

    # v1 limit: single peer-pair staging buffer means we cannot service more
    # than one (sender, receiver) pair per group. Lift this restriction in
    # the follow-up that adds per-peer staging slots for Ring AllGather.
    assert world_size == 2, (
        f"triton_nvl_sendrecv v1 supports only 2-rank groups, got world_size={world_size}. "
        "See module docstring for the planned per-peer slot API."
    )

    local_rank = dist.get_rank(group)
    assert peer_rank != local_rank
    assert 0 <= peer_rank < world_size

    # Buffer and signal pad pointers are passed as direct typed tensors
    # (not pointer-of-pointer int64 tensors) so Triton emits 128-bit
    # vectorized stores instead of 16-bit scalar stores. See the kernel
    # docstring in sendrecv.py for the codegen rationale.
    handle = _get_staging_buffer(group, send_buf.device)
    sender_step, recver_step = _get_step_state(group, send_buf.device)

    config = _compute_config(send_buf.element_size())
    block_stride_elems = _BLOCK_STRIDE_BYTES // send_buf.element_size()
    tile_numel = config.tile_rows * config.tile_cols
    tile_bytes = tile_numel * send_buf.element_size()
    max_tiles_per_block_per_slot = max(_BLOCK_STRIDE_BYTES // tile_bytes, 1)

    staging_elems = block_stride_elems * _MAX_BLOCKS_PER_DIR * _PIPELINE_DEPTH
    remote_buf = handle.get_buffer(
        peer_rank, sizes=[staging_elems], dtype=send_buf.dtype
    )
    local_buf = handle.get_buffer(
        local_rank, sizes=[staging_elems], dtype=recv_buf.dtype
    )
    remote_sig = handle.get_signal_pad(peer_rank).view(torch.int64)
    local_sig = handle.get_signal_pad(local_rank).view(torch.int64)

    grid = (2 * num_blocks,) if mode == _MODE_BIDIRECTIONAL else (num_blocks,)

    # pyre-ignore[28]: triton's `kernel[grid](...)` launcher accepts JIT-special
    # kwargs (`num_warps`, `num_stages`) that pyre can't see in the kernel
    # function's own signature. Not a real type error.
    triton_nvl_sendrecv_kernel[grid](
        send_ptr=send_buf,
        recv_ptr=recv_buf,
        remote_buf=remote_buf,
        local_buf=local_buf,
        remote_sig=remote_sig,
        local_sig=local_sig,
        sender_step_ptr=sender_step,
        recver_step_ptr=recver_step,
        local_rank=local_rank,
        peer_rank=peer_rank,
        send_numel=send_buf.numel(),
        recv_numel=recv_buf.numel(),
        # pyrefly: ignore [bad-argument-type]
        TILE_ROWS=config.tile_rows,
        # pyrefly: ignore [bad-argument-type]
        TILE_COLS=config.tile_cols,
        # pyrefly: ignore [bad-argument-type]
        BLOCK_STRIDE_ELEMS=block_stride_elems,
        # pyrefly: ignore [bad-argument-type]
        PIPELINE_DEPTH=_PIPELINE_DEPTH,
        # pyrefly: ignore [bad-argument-type]
        MAX_BLOCKS_PER_DIR=_MAX_BLOCKS_PER_DIR,
        # pyrefly: ignore [bad-argument-type]
        NUM_SEND_BLOCKS=num_blocks,
        # pyrefly: ignore [bad-argument-type]
        MAX_TILES_PER_BLOCK_PER_SLOT=max_tiles_per_block_per_slot,
        # pyrefly: ignore [bad-argument-type]
        MODE=mode,
        # pyrefly: ignore [unexpected-keyword]
        num_warps=num_warps,
        # pyrefly: ignore [unexpected-keyword]
        num_stages=config.num_stages,
    )
