# Copyright (c) Meta Platforms, Inc. and affiliates.
# Confidential and proprietary.
# pyre-unsafe
"""
Triton LL protocol implementation for Pipes.

Provides low-latency NVLink P2P communication using the LL (Low-Latency)
protocol with dual-flag atomicity detection. Each LL line is 16 bytes:
{data1: u32, flag1: u32, data2: u32, flag2: u32}, carrying 8 bytes of
payload per line.

Core components:
  - Inline PTX primitives for acquire/release loads/stores
  - ll_send_kernel: Send data to peer's LL buffer via NVLink
  - ll_recv_kernel: Receive data from local LL buffer
  - ll_forward_kernel: Forward data through an intermediate rank
  - GIN-safe utility kernels for buffer initialization and iteration tracking
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton  # @manual
import triton.language as tl  # @manual

try:
    from torchcomms.triton.fb import get_nvlink_address, requires_torchcomms
except ImportError:
    get_nvlink_address = None

    def requires_torchcomms(fn):  # type: ignore[no-redef]
        return fn


if TYPE_CHECKING:
    pass

# =============================================================================
# Constants
# =============================================================================

LL_LINE_SIZE = 16
LL_PAYLOAD_PER_LINE = 8
LL_READY_TO_WRITE = 0xFFFFFFFF
LL_MEMSET_INIT_BYTE = 0xFF
# Masks for extracting fields from uint64 words in the v2.u64 representation.
# word = {data (low 32b), flag (high 32b)}
LL_FLAG_MASK = 0xFFFFFFFF00000000
LL_DATA_MASK = 0x00000000FFFFFFFF

# =============================================================================
# Utility Functions (pure Python, no torch/triton deps at call time)
# =============================================================================


def ll_num_lines(nbytes: int) -> int:
    """Number of LL lines needed for a payload of nbytes."""
    if nbytes == 0:
        return 0
    return (nbytes + LL_PAYLOAD_PER_LINE - 1) // LL_PAYLOAD_PER_LINE


def ll_buffer_size(max_message_size: int) -> int:
    """LL buffer size in bytes for a single peer's region."""
    return ll_num_lines(max_message_size) * LL_LINE_SIZE


def ll_total_buffer_size(max_message_size: int, world_size: int) -> int:
    """Total LL buffer size across all peers (world_size - 1 regions)."""
    return ll_buffer_size(max_message_size) * (world_size - 1)


def can_use_ll(nbytes: int) -> bool:
    """Check if payload size is valid for LL (must be 8-byte multiple)."""
    if nbytes == 0:
        return True
    return nbytes % LL_PAYLOAD_PER_LINE == 0


def ll_peer_index(target_rank: int, self_rank: int) -> int:
    """Index of target_rank in self_rank's peer list (skipping self).

    Maps a rank to its 0-based position among (world_size - 1) peers,
    as seen from self_rank's perspective. Matches the C++ pattern in
    DeviceWindow.cuh:rank_to_peer_index().

    Example (world_size=4, self_rank=1):
      target_rank=0 -> index 0
      target_rank=2 -> index 1
      target_rank=3 -> index 2
    """
    return target_rank if target_rank < self_rank else target_rank - 1


def ll_peer_buffer_offset(target_rank: int, self_rank: int, per_peer_size: int) -> int:
    """Byte offset for target_rank's region within self_rank's LL buffer.

    Each rank's LL buffer is partitioned into (world_size - 1) regions,
    one per peer. This returns the byte offset to the region where
    target_rank's data lives in self_rank's buffer.

    For send: ll_peer_buffer_offset(my_rank, peer, size) -- "my region in peer's buf"
    For recv: ll_peer_buffer_offset(peer, my_rank, size) -- "peer's region in my buf"
    """
    return ll_peer_index(target_rank, self_rank) * per_peer_size


# =============================================================================
# Inline PTX Primitives
# =============================================================================


@triton.jit
def _acquire_load_v2_u64(addr):
    """Load an LL line (16B) as two uint64 words with system-scope acquire.

    Emits ld.acquire.sys.v2.u64 (without .global -- PTX infers the state
    space from the pointer, matching NCCL LL's validated pattern:
    ld.acquire.sys.v4.u32). Acquire ordering ensures subsequent reads see
    the sender's writes that were released before the flag store.
    """
    return tl.inline_asm_elementwise(
        "ld.acquire.sys.v2.u64 {$0, $1}, [$2];",
        "=l,=l,l",
        args=[addr],
        dtype=(tl.uint64, tl.uint64),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _release_store_v2_u64(addr, word0, word1):
    """Store an LL line (16B) from two uint64 words with system-scope release.

    Emits st.release.sys.v2.u64 (without .global -- matching CCCL's
    validated pattern: st.release.gpu.v2.u64). Release ordering ensures
    all prior writes (data packing) are visible before the flag becomes
    visible to the receiver.
    """
    tl.inline_asm_elementwise(
        "st.release.sys.v2.u64 [$1], {$2, $3};",
        "=r,l,l,l",
        args=[addr, word0, word1],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _read_globaltimer():
    """Read the GPU global timer in nanoseconds for timeout detection.

    Uses %globaltimer (not %clock64) because it returns wall-clock
    nanoseconds, making timeout values directly interpretable without
    per-device clock rate conversion.
    """
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;",
        "=l",
        args=[],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


# =============================================================================
# Kernel Helpers
# =============================================================================


@triton.jit
def _compute_line_params(
    line_idx, buffer_num_lines, aligned_buf_lines, BLOCK_SIZE: tl.constexpr
):
    """Compute buffer index and flag value for a set of LL lines."""
    if buffer_num_lines > 0:
        buf_idx = line_idx % aligned_buf_lines
        flag_value = tl.cast(1 + (line_idx // aligned_buf_lines), tl.int32)
    else:
        buf_idx = line_idx
        flag_value = tl.full([BLOCK_SIZE], 1, dtype=tl.int32)
    return buf_idx, flag_value


@triton.jit
def _poll_ll_flags(addr, expected_flag, mask):
    """Poll an LL line until both flags match expected_flag.

    @return (word0, word1) from the final successful load.
    """
    word0, word1 = _acquire_load_v2_u64(addr)
    flag1 = (word0 >> 32).to(tl.uint32)
    flag2 = (word1 >> 32).to(tl.uint32)
    ready = (flag1 == expected_flag) & (flag2 == expected_flag)
    ready = tl.where(mask, ready, True)
    all_ready = tl.min(ready.to(tl.int32))

    while all_ready == 0:
        word0, word1 = _acquire_load_v2_u64(addr)
        flag1 = (word0 >> 32).to(tl.uint32)
        flag2 = (word1 >> 32).to(tl.uint32)
        ready = (flag1 == expected_flag) & (flag2 == expected_flag)
        ready = tl.where(mask, ready, True)
        all_ready = tl.min(ready.to(tl.int32))

    return word0, word1


# =============================================================================
# GIN-Safe Utility Kernels
# =============================================================================


@requires_torchcomms
@triton.jit
def _fill_ll_buffer_kernel(buf_ptr_i32, fill_value_i32, N: tl.constexpr):
    """GIN-safe kernel to fill LL buffer with a 32-bit pattern.

    buf_ptr_i32 is the LL buffer pointer cast to int32*. N is the number
    of int32 elements to fill (total_bytes // 4). Each program fills one
    int32 element. Launch with grid=(N,).

    To initialize all flags to LL_READY_TO_WRITE (0xFFFFFFFF), pass
    fill_value_i32 = -1 (which is 0xFFFFFFFF in two's complement int32).
    """
    idx = tl.program_id(0)
    if idx < N:
        tl.store(buf_ptr_i32 + idx, fill_value_i32)


# =============================================================================
# Iteration Tracking (CUDA graph compatible)
# =============================================================================

_ITERATION_TENSOR_CACHE: dict = {}


def _get_iteration_tensor(device: torch.device, world_size: int) -> torch.Tensor:
    """Return a cached int64 scalar tensor for iteration counting.

    Keyed by (device, world_size) to prevent corruption when multiple
    communicators with different world_size values share the same GPU.

    MUST be called BEFORE GIN activation.
    """
    key = (device, world_size)
    if key not in _ITERATION_TENSOR_CACHE:
        _ITERATION_TENSOR_CACHE[key] = torch.zeros(1, dtype=torch.int64, device=device)
    return _ITERATION_TENSOR_CACHE[key]


@requires_torchcomms
@triton.jit
def _increment_iteration_kernel(iteration_ptr):
    """Increment the iteration counter by 1. GIN-safe, CUDA graph capturable."""
    if tl.program_id(0) == 0:
        old_val = tl.load(iteration_ptr)
        tl.store(iteration_ptr, old_val + 1)


# =============================================================================
# LL Send/Recv Loop Helpers (shared by standalone and fused kernels)
# =============================================================================


@triton.jit
def _ll_send_loop(
    pid,
    num_blocks,
    src_ptr,
    remote_base,
    nbytes,
    buffer_num_lines,
    BLOCK_SIZE: tl.constexpr,
):
    """Send loop body: poll for READY_TO_WRITE, pack data, release-store.

    @param pid: Block index within the send direction (0-based).
    @param num_blocks: Number of blocks in the send direction.
    """
    total_lines = nbytes // 8

    if buffer_num_lines > 0:
        active_blocks = tl.minimum(buffer_num_lines // BLOCK_SIZE, num_blocks)
    else:
        active_blocks = num_blocks

    if pid >= active_blocks:
        return

    if buffer_num_lines > 0:
        stride = active_blocks * BLOCK_SIZE
        aligned_buf_lines = (buffer_num_lines // stride) * stride
    else:
        aligned_buf_lines = 0

    line_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ready_flag = tl.full([BLOCK_SIZE], LL_READY_TO_WRITE, dtype=tl.uint32)

    num_steps = (total_lines + active_blocks * BLOCK_SIZE - 1) // (
        active_blocks * BLOCK_SIZE
    )
    step = 0
    while step < num_steps:
        line_idx = line_offsets + step * active_blocks * BLOCK_SIZE
        mask = line_idx < total_lines

        buf_idx, flag_value = _compute_line_params(
            line_idx, buffer_num_lines, aligned_buf_lines, BLOCK_SIZE
        )

        remote_addr = remote_base + buf_idx * 16

        _poll_ll_flags(remote_addr, ready_flag, mask)

        src_offset = line_idx * 8
        data1 = (
            tl.load(src_ptr + src_offset // 4, mask=mask, other=0)
            .to(tl.uint32)
            .to(tl.uint64)
        )
        data2 = (
            tl.load(src_ptr + src_offset // 4 + 1, mask=mask, other=0)
            .to(tl.uint32)
            .to(tl.uint64)
        )

        flag_u64 = flag_value.to(tl.uint64)
        out_word0 = data1 | (flag_u64 << 32)
        out_word1 = data2 | (flag_u64 << 32)

        _release_store_v2_u64(remote_addr, out_word0, out_word1)

        step += 1


@triton.jit
def _ll_recv_loop(
    pid,
    num_blocks,
    dst_ptr,
    local_base,
    nbytes,
    buffer_num_lines,
    BLOCK_SIZE: tl.constexpr,
):
    """Recv loop body: poll for flag, extract data, ACK with READY_TO_WRITE.

    @param pid: Block index within the recv direction (0-based).
    @param num_blocks: Number of blocks in the recv direction.
    """
    total_lines = nbytes // 8

    if buffer_num_lines > 0:
        active_blocks = tl.minimum(buffer_num_lines // BLOCK_SIZE, num_blocks)
    else:
        active_blocks = num_blocks

    if pid >= active_blocks:
        return

    if buffer_num_lines > 0:
        stride = active_blocks * BLOCK_SIZE
        aligned_buf_lines = (buffer_num_lines // stride) * stride
    else:
        aligned_buf_lines = 0

    line_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    ack_word = tl.full([BLOCK_SIZE], LL_READY_TO_WRITE, dtype=tl.uint64) << 32

    num_steps = (total_lines + active_blocks * BLOCK_SIZE - 1) // (
        active_blocks * BLOCK_SIZE
    )
    step = 0
    while step < num_steps:
        line_idx = line_offsets + step * active_blocks * BLOCK_SIZE
        mask = line_idx < total_lines

        buf_idx, flag_value = _compute_line_params(
            line_idx, buffer_num_lines, aligned_buf_lines, BLOCK_SIZE
        )

        local_addr = local_base + buf_idx * 16
        expected_flag = flag_value.to(tl.uint32)

        word0, word1 = _poll_ll_flags(local_addr, expected_flag, mask)

        data1 = (word0 & LL_DATA_MASK).to(tl.uint32)
        data2 = (word1 & LL_DATA_MASK).to(tl.uint32)

        dst_offset = line_idx * 8
        tl.store(dst_ptr + dst_offset // 4, data1, mask=mask)
        tl.store(dst_ptr + dst_offset // 4 + 1, data2, mask=mask)

        _release_store_v2_u64(local_addr, ack_word, ack_word)

        step += 1


# =============================================================================
# LL Send Kernel (standalone)
# =============================================================================


@requires_torchcomms
@triton.jit
def ll_send_kernel(
    src_ptr,
    win,
    peer: tl.constexpr,
    peer_buf_offset,
    nbytes,
    buffer_num_lines,
    BLOCK_SIZE: tl.constexpr,
):
    """Send data to peer's LL buffer via NVLink."""
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)
    remote_base = get_nvlink_address(win, peer) + peer_buf_offset
    _ll_send_loop(
        pid,
        num_blocks,
        src_ptr,
        remote_base,
        nbytes,
        buffer_num_lines,
        BLOCK_SIZE,
    )


# =============================================================================
# LL Recv Kernel (standalone)
# =============================================================================


@requires_torchcomms
@triton.jit
def ll_recv_kernel(
    dst_ptr,
    ll_buf_ptr,
    peer_buf_offset,
    nbytes,
    buffer_num_lines,
    BLOCK_SIZE: tl.constexpr,
):
    """Receive data from local LL buffer."""
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)
    local_base = ll_buf_ptr + peer_buf_offset
    _ll_recv_loop(
        pid,
        num_blocks,
        dst_ptr,
        local_base,
        nbytes,
        buffer_num_lines,
        BLOCK_SIZE,
    )


# =============================================================================
# LL Fused SendRecv Kernel
# =============================================================================


@requires_torchcomms
@triton.jit
def ll_sendrecv_kernel(
    src_ptr,
    dst_ptr,
    win,
    peer: tl.constexpr,
    send_peer_buf_offset,
    recv_peer_buf_offset,
    ll_buf_ptr,
    nbytes,
    buffer_num_lines,
    num_blocks_per_dir,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused send+recv in a single kernel launch.

    Grid: (2 * num_blocks_per_dir,). The first num_blocks_per_dir blocks
    execute the send path; the remaining execute the recv path. Both ranks
    must call this simultaneously.

    num_blocks_per_dir is a runtime parameter (not constexpr) to avoid
    Triton recompiling the kernel for every unique block count. Only
    BLOCK_SIZE is constexpr (required for tl.arange/tl.full).
    """
    pid = tl.program_id(0)

    if pid < num_blocks_per_dir:
        remote_base = get_nvlink_address(win, peer) + send_peer_buf_offset
        _ll_send_loop(
            pid,
            num_blocks_per_dir,
            src_ptr,
            remote_base,
            nbytes,
            buffer_num_lines,
            BLOCK_SIZE,
        )
    else:
        recv_pid = pid - num_blocks_per_dir
        local_base = ll_buf_ptr + recv_peer_buf_offset
        _ll_recv_loop(
            recv_pid,
            num_blocks_per_dir,
            dst_ptr,
            local_base,
            nbytes,
            buffer_num_lines,
            BLOCK_SIZE,
        )


# =============================================================================
# LL Forward Kernel
# =============================================================================


@requires_torchcomms
@triton.jit
def ll_forward_kernel(
    dst_ptr,
    ll_buf_ptr,
    predecessor_buf_offset,
    win,
    successor_peer: tl.constexpr,
    successor_buf_offset,
    nbytes,
    buffer_num_lines,
    BLOCK_SIZE: tl.constexpr,
):
    """Forward data from predecessor to successor, copying to local output.

    @param dst_ptr: Local output buffer
    @param ll_buf_ptr: Local LL buffer base pointer
    @param predecessor_buf_offset: Offset to predecessor's region in local LL buf
    @param win: TorchComms device window handle
    @param successor_peer: Successor rank to forward to (constexpr)
    @param successor_buf_offset: Offset to this rank's region in successor's LL buf
    @param nbytes: Total payload bytes (must be multiple of 8)
    @param buffer_num_lines: LL buffer size in lines per peer (0 = unbounded)
    @param BLOCK_SIZE: Lines per block (constexpr)

    """
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)
    total_lines = nbytes // 8

    if buffer_num_lines > 0:
        active_blocks = tl.minimum(buffer_num_lines // BLOCK_SIZE, num_blocks)
    else:
        active_blocks = num_blocks

    if pid >= active_blocks:
        return

    local_base = ll_buf_ptr + predecessor_buf_offset
    remote_base = get_nvlink_address(win, successor_peer) + successor_buf_offset

    if buffer_num_lines > 0:
        stride = active_blocks * BLOCK_SIZE
        aligned_buf_lines = (buffer_num_lines // stride) * stride
    else:
        stride = 0
        aligned_buf_lines = 0

    line_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    ack_word = tl.full([BLOCK_SIZE], LL_READY_TO_WRITE, dtype=tl.uint64) << 32
    ready_flag = tl.full([BLOCK_SIZE], LL_READY_TO_WRITE, dtype=tl.uint32)

    num_steps = (total_lines + active_blocks * BLOCK_SIZE - 1) // (
        active_blocks * BLOCK_SIZE
    )
    step = 0
    while step < num_steps:
        line_idx = line_offsets + step * active_blocks * BLOCK_SIZE
        mask = line_idx < total_lines

        buf_idx, flag_value = _compute_line_params(
            line_idx, buffer_num_lines, aligned_buf_lines, BLOCK_SIZE
        )

        local_addr = local_base + buf_idx * 16
        remote_addr = remote_base + buf_idx * 16
        expected_flag = flag_value.to(tl.uint32)

        # Phase 1: Poll local LL buffer for data from predecessor
        word0, word1 = _poll_ll_flags(local_addr, expected_flag, mask)

        # Extract data from predecessor's line
        data1 = (word0 & LL_DATA_MASK).to(tl.uint64)
        data2 = (word1 & LL_DATA_MASK).to(tl.uint64)

        # Phase 2: Poll successor's remote LL buffer for READY_TO_WRITE
        _poll_ll_flags(remote_addr, ready_flag, mask)

        # Phase 3: Forward to successor, copy to local output, ACK predecessor
        flag_u64 = flag_value.to(tl.uint64)
        fwd_word0 = data1 | (flag_u64 << 32)
        fwd_word1 = data2 | (flag_u64 << 32)

        _release_store_v2_u64(remote_addr, fwd_word0, fwd_word1)

        dst_offset = line_idx * 8
        tl.store(dst_ptr + dst_offset // 4, data1.to(tl.uint32), mask=mask)
        tl.store(dst_ptr + dst_offset // 4 + 1, data2.to(tl.uint32), mask=mask)

        _release_store_v2_u64(local_addr, ack_word, ack_word)

        step += 1
