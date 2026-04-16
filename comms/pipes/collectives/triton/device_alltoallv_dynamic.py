# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
"""
Device-initiated AlltoAllv Dynamic implementation using TorchComms Triton APIs.

This module provides GPU-initiated alltoallv collectives with DYNAMIC per-rank
counts that perform all-to-all communication with variable message sizes
directly from Triton kernels without CPU involvement.

Simplified Wrapper (Recommended)
--------------------------------
For most users, the ``AlltoallvOp`` wrapper provides a much simpler,
token-level API that handles buffer registration and iteration tracking
automatically (no offset exchange needed due to fixed-slot layout)::

    from comms.pipes.collectives.triton import AlltoallvOp

    op = AlltoallvOp(comm, max_input_tokens=1024, D=4096,
                     dtype=torch.bfloat16, device="cuda:0")
    with op:
        output = op.alltoallv(input_tensor, output_split_sizes,
                              input_split_sizes)

See ``AlltoallvOp`` for full documentation.

Key Feature: GPU-Resident Counts
---------------------------------
Unlike traditional CPU-initiated alltoallv where send counts must be known on
the host before launch, this implementation reads send/receive counts directly
from GPU memory. This enables:

1. Fused compute + communication: A preceding kernel can compute how much data
   to send to each peer, store counts in GPU memory, and this kernel reads them
   directly - no CPU roundtrip needed.

2. Dynamic workloads: Supports workloads where message sizes are data-dependent
   (e.g., sparse operations, filtering, load balancing, MoE token routing).

3. Zero CPU involvement: The entire compute→count→communicate pipeline stays
   on GPU, eliminating synchronization overhead.

Raw API Usage (Advanced):
    # 1. Compute phase (preceding kernel computes variable counts)
    compute_kernel[grid](data, send_counts_gpu, send_offsets_gpu)

    # 2. Communication phase (reads counts from GPU, no CPU copy)
    device_alltoallv_dynamic(
        send_buf, recv_buf,
        send_counts_gpu,    # GPU tensor - read directly by kernel
        send_offsets_gpu,   # GPU tensor - read directly by kernel
        recv_counts_gpu,
        recv_offsets_gpu,
        comm, window
    )
"""

from typing import TYPE_CHECKING

import torch
import triton  # @manual
import triton.language as tl  # @manual
from torchcomms.triton.fb import (
    flush_block,
    put_block_direct,
    put_warp_chunked_direct,
    requires_torchcomms,
    self_copy_block,
    signal_block,
    wait_signal_from,
)


if TYPE_CHECKING:
    from torchcomms import TorchComm


__all__ = [
    "device_alltoallv_dynamic",
    "auto_tune_alltoallv_params",
    "compute_offsets_from_sizes",
    "exchange_offsets",
    "prewarm_completion_counters",
]


# =============================================================================
# Internal Iteration Tensor Cache
# =============================================================================


# Pre-allocated iteration tensor cache to track iteration count per
# (device, world_size) pair. This is managed internally - users don't need
# to create or manage iteration tensors.
_ITERATION_TENSOR_CACHE: dict = {}


def _get_iteration_tensor(world_size: int, device: torch.device) -> torch.Tensor:
    """Return a cached int32 scalar tensor for iteration counting.

    The tensor is allocated once on first use and reused across all
    subsequent calls. The iteration value grows monotonically.

    IMPORTANT: This must be called BEFORE GIN (GPU-Initiated NCCL) is
    activated via get_device_window(). Once GIN is active, regular CUDA
    allocations fail with cudaErrorHostMemoryAlreadyRegistered.
    """
    key = (device, world_size)
    if key not in _ITERATION_TENSOR_CACHE:
        _ITERATION_TENSOR_CACHE[key] = torch.zeros(1, dtype=torch.int32, device=device)
    return _ITERATION_TENSOR_CACHE[key]


def _reset_iteration_counter(world_size: int, device: torch.device) -> None:
    """Reset the iteration counter to zero for a given (device, world_size) pair.

    This is an internal function called automatically by AlltoallvOp.setup().
    It MUST be called when creating a new window (after get_device_window())
    to ensure the iteration counter matches the fresh signal memory state.

    Without this reset, cross-session hangs occur:
    1. Session1 runs with iteration counter 0, signals BUFFER_READY(1)
    2. Session1 teardown destroys window (signal memory freed)
    3. Session2 creates new window (fresh signal memory = 0)
    4. Iteration counter is still 1 (global, not reset)
    5. Session2's send block waits for BUFFER_READY(1) from signal memory that's 0
    6. HANG!

    The iteration counter and signal memory must be synchronized:
    - When signal memory is fresh (new window), iteration counter must be 0
    - This ensures send blocks don't wait for signals from "previous iterations"
      that were actually in a different window's signal memory

    Note: This function uses a GIN-safe Triton kernel to reset the counter,
    so it's safe to call after GIN is activated.

    Args:
        world_size: Number of ranks in the communicator.
        device: CUDA device where the iteration tensor is stored.
    """
    key = (device, world_size)
    if key in _ITERATION_TENSOR_CACHE:
        # Use a GIN-safe kernel to reset the counter
        _fill_int32_kernel[(1,)](_ITERATION_TENSOR_CACHE[key], 0, N=1)


@requires_torchcomms
@triton.jit
def _increment_iteration_kernel(iteration_ptr):
    """Increment the iteration counter tensor by 1.

    This kernel is called after each alltoallv operation. It gets captured in
    CUDA graphs, allowing multiple alltoallv calls within a single graph
    capture to use correct iteration values.
    """
    old_val = tl.load(iteration_ptr)
    tl.store(iteration_ptr, old_val + 1)


# =============================================================================
# Helper: cached completion counters
# =============================================================================


# Pre-allocated completion counters cache to avoid per-call CUDA allocator
# overhead.  Keyed by (device, world_size) → torch.Tensor(int32).
_COMPLETION_COUNTERS_CACHE: dict = {}


@requires_torchcomms
@triton.jit
def _fill_int32_kernel(ptr, value, N: tl.constexpr):
    """GIN-safe kernel to fill an int32 tensor with a scalar value."""
    idx = tl.program_id(0)
    if idx < N:
        tl.store(ptr + idx, value)


@requires_torchcomms
@triton.jit
def _fill_completion_counters_from_iteration_kernel(
    counters_ptr,
    iteration_ptr,
    BLOCKS_PER_PEER: tl.constexpr,
    N: tl.constexpr,
    per_peer_blocks_ptr,
    HAS_PER_PEER_BLOCKS: tl.constexpr,
):
    """GIN-safe kernel to reset completion counters based on iteration.

    Reads iteration from GPU memory and computes the expected base counter
    value for each peer. When HAS_PER_PEER_BLOCKS is True, uses per-peer
    block counts instead of the uniform BLOCKS_PER_PEER constexpr.
    This is CUDA graph compatible since all values are read from GPU.
    """
    idx = tl.program_id(0)
    if idx < N:
        iteration = tl.load(iteration_ptr)
        if HAS_PER_PEER_BLOCKS:
            bpp = tl.load(per_peer_blocks_ptr + idx)
        else:
            bpp = BLOCKS_PER_PEER
        expected_base = bpp * iteration
        tl.store(counters_ptr + idx, expected_base)


def _fill_completion_counters_gin_safe(
    counters: torch.Tensor, value: int, world_size: int
) -> None:
    """Fill completion counters using a GIN-safe Triton kernel.

    Regular CUDA operations like tensor.fill_() fail when GIN is active
    (cudaErrorHostMemoryAlreadyRegistered).  This function uses a Triton
    kernel which only does device-side stores and is GIN-safe.
    """
    _fill_int32_kernel[(world_size,)](counters, value, N=world_size)


def _fill_completion_counters_from_iteration(
    counters: torch.Tensor,
    iteration_tensor: torch.Tensor,
    blocks_per_peer: int,
    world_size: int,
    per_peer_blocks: "torch.Tensor | None" = None,
) -> None:
    """Reset completion counters based on iteration value from GPU tensor.

    This is CUDA graph compatible - reads iteration from GPU memory and
    computes expected_base entirely on GPU.

    When per_peer_blocks is provided, each counter is reset to
    per_peer_blocks[i] * iteration (per-peer rate). Otherwise, all
    counters are reset uniformly to blocks_per_peer * iteration.

    Args:
        counters: GPU tensor [world_size] of int32 counters.
        iteration_tensor: GPU scalar tensor containing iteration count.
        blocks_per_peer: Number of blocks per peer (constexpr upper bound).
        world_size: Number of ranks.
        per_peer_blocks: Optional GPU tensor [world_size] of int32 per-peer
            block counts. When provided, counters are reset per-peer.
    """
    _fill_completion_counters_from_iteration_kernel[(world_size,)](
        counters,
        iteration_tensor,
        BLOCKS_PER_PEER=blocks_per_peer,
        N=world_size,
        per_peer_blocks_ptr=per_peer_blocks,
        HAS_PER_PEER_BLOCKS=per_peer_blocks is not None,
    )


def _get_completion_counters(world_size: int, device: torch.device) -> torch.Tensor:
    """Return a cached int32 tensor of shape [world_size] on device.

    The tensor is allocated once (via torch.zeros) on first use and reused
    across all subsequent calls.  Counters grow monotonically with each
    iteration (no in-kernel reset) to avoid race conditions with CUDA
    block scheduling.

    IMPORTANT: This must be called BEFORE GIN (GPU-Initiated NCCL) is
    activated via get_device_window().  Once GIN is active, regular CUDA
    allocations fail with cudaErrorHostMemoryAlreadyRegistered.
    """
    key = (device, world_size)
    if key not in _COMPLETION_COUNTERS_CACHE:
        _COMPLETION_COUNTERS_CACHE[key] = torch.zeros(
            world_size, dtype=torch.int32, device=device
        )
    return _COMPLETION_COUNTERS_CACHE[key]


def prewarm_completion_counters(world_size: int, device: torch.device) -> None:
    """Pre-allocate completion counters and iteration tensor before GIN is activated.

    This function MUST be called BEFORE get_device_window() to avoid CUDA
    allocation failures when GIN is active.  GIN (GPU-Initiated NCCL)
    registers GPU memory which blocks subsequent regular CUDA allocations.

    Args:
        world_size: Number of ranks in the communicator.
        device: CUDA device to allocate on.

    Example:
        >>> # Before GIN activation
        >>> prewarm_completion_counters(world_size, device)
        >>> # Now activate GIN
        >>> dev_win_ptr = window.get_device_window(signal_count=world_size)
        >>> # Now device_alltoallv_dynamic can be called safely
    """
    _get_completion_counters(world_size, device)
    _get_iteration_tensor(world_size, device)


# =============================================================================
# Non-Pipelined Implementation
# =============================================================================


@requires_torchcomms
@triton.jit
def _device_alltoallv_dynamic_kernel(
    # Window handles
    dst_win_ptr,
    src_registered_buf,  # device ptr to RegisteredBuffer (from register_local_buffer)
    # Buffer pointers for self-copy (typed by Triton from tensors)
    send_buf_ptr,
    recv_buf_ptr,
    # Offset/size arrays (pass tensors directly, Triton handles pointer conversion)
    send_offsets_ptr,
    send_sizes_ptr,
    recv_offsets_ptr,
    recv_sizes_ptr,
    dst_offsets_ptr,
    # Constants
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    signal_id: tl.constexpr,
    # Iteration counter - pointer to GPU scalar (always read from GPU memory)
    iteration_ptr,
    ELEM_BYTES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PEER: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    completion_counters_ptr,
    # Sync buffer mode for buffer-ready synchronization
    SYNC_BUFFER: tl.constexpr,
    BUFFER_READY_SIGNAL_ID: tl.constexpr,
    # Per-peer block counts for topology-aware scheduling (optional)
    per_peer_blocks_ptr,
    HAS_PER_PEER_BLOCKS: tl.constexpr,
):
    """
    Device-initiated AlltoAllv Dynamic kernel.

    Key Feature: Reads send counts and offsets directly from GPU memory,
    enabling fused compute + communication without CPU involvement.

    Algorithm:
    1. Load phase: Read per-peer counts/offsets from GPU memory
    2. Send phase: Put data to all peers using per-peer dst_offsets
    3. Flush phase: Wait for all outgoing transfers to complete
    4. Completion: Coordinate multi-block signaling (if BLOCKS_PER_PEER > 1)
    5. Receive phase: Wait for signals from all peers

    Grid layout: (world_size * BLOCKS_PER_PEER + world_size,) total blocks.
    The first (world_size * BLOCKS_PER_PEER) blocks are SEND blocks.
    The last world_size blocks are RECV blocks (one per peer).

    Send blocks (pid < world_size * BLOCKS_PER_PEER):
        Each peer is served by BLOCKS_PER_PEER consecutive blocks.  When
        BLOCKS_PER_PEER == 1, the kernel behaves identically to the
        single-block-per-peer version.  With BLOCKS_PER_PEER > 1, each
        peer's send data is split into BLOCKS_PER_PEER contiguous chunks
        and each block handles its portion.  Send blocks NEVER call
        wait_signal_from, so they run uninterrupted.

    Recv blocks (pid >= world_size * BLOCKS_PER_PEER):
        Each recv block waits for data from one peer via wait_signal_from.
        These run concurrently with send blocks on separate SMs, fully
        overlapping send and recv.

    Chunked transfer: All transfers use put_block_chunked, which
    distributes CHUNK_SIZE-byte chunks across warps within the block
    using CoopScope::WARP.  This creates multiple concurrent NVLink
    write streams from a single block (one per warp), matching the
    pipes/collectives warp-level architecture.  For messages that fit
    in a single chunk, put_block_chunked falls back to CoopScope::BLOCK
    (equivalent to put_block).  No per-chunk flush is needed — a single
    __syncthreads at the end of put_block_chunked ensures all warps
    complete before the caller signals.

    Signaling:
    - BLOCKS_PER_PEER == 1: The single block calls signal_block after
      put_block_chunked + flush_block.  Release semantics on the signal
      guarantee all prior stores (including all warp-parallel chunks)
      are visible.
    - BLOCKS_PER_PEER > 1: No block signals during puts.  After each
      block completes put_block_chunked + flush_block, it atomically
      increments completion_counters[peer].  The last block to complete
      (atomic result == BLOCKS_PER_PEER - 1) calls signal_block.
      Because flush_block guarantees each block's NVLink writes are
      globally visible before incrementing the counter, the signal is
      only sent after ALL blocks have completed their transfers.

    Args:
        dst_win_ptr: TorchComms device window handle for destination
        src_registered_buf: Device pointer to RegisteredBuffer struct
                           (from register_local_buffer()). Passed directly to
                           put_block_direct / put_warp_chunked_direct which
                           read base_ptr from the struct internally.
        send_offsets_ptr: GPU tensor [world_size] - byte offsets into src
        send_sizes_ptr: GPU tensor [world_size] - bytes to send per peer
        recv_offsets_ptr: GPU tensor [world_size] - byte offsets into local dst
        recv_sizes_ptr: GPU tensor [world_size] - bytes expected from each peer
        dst_offsets_ptr: GPU tensor [world_size] - byte offsets on each peer's
                        recv_buf (obtained via all_to_all_single exchange)
        my_rank: This rank's ID (constexpr)
        world_size: Total number of ranks (constexpr)
        signal_id: Signal ID to use for notifications (constexpr)
        iteration: 0-based iteration counter (non-constexpr runtime arg).
                   Since put_block increments signals via SignalOp::ADD,
                   the kernel waits for signal >= iteration + 1.
        ELEM_BYTES: Size of each element in bytes (constexpr), e.g. 4 for
                    float32, 2 for float16/bfloat16, 8 for float64.
        BLOCKS_PER_PEER: Number of send blocks assigned to each peer
                         (constexpr). Default: 1 (original behavior).
        CHUNK_SIZE: Chunk size for warp-level distribution in
                    put_block_chunked (constexpr).  Each warp processes
                    chunks of this size in round-robin.  Larger values
                    reduce per-warp overhead; smaller values increase
                    warp-level parallelism.
                    Default: 262144 (256KB).
        completion_counters_ptr: GPU tensor [world_size] of int32 counters
                    (zeroed before each call).  Used for multi-block
                    completion coordination when BLOCKS_PER_PEER > 1.
                    Unused (but still passed) when BLOCKS_PER_PEER == 1.
    """
    pid = tl.program_id(axis=0)

    # Load iteration counter from GPU memory.
    # Using a GPU tensor ensures correct behavior for both CUDA graph and
    # non-graph modes. The tensor is incremented via add_(1) at the end of
    # each alltoallv call, which is capturable in CUDA graphs.
    iteration = tl.load(iteration_ptr)

    total_send_blocks: tl.constexpr = world_size * BLOCKS_PER_PEER

    # ── Recv block path ──
    # The last world_size blocks are dedicated recv blocks.
    # Each recv block waits for data from one peer.  These run on
    # separate SMs from send blocks, fully overlapping send and recv.
    #
    # For sync_buffer mode (SYNC_BUFFER=True):
    #   Signal BUFFER_READY at the START of the kernel (before waiting for
    #   new data) to tell the sender "I've consumed the previous iteration's
    #   data, you can send new data now". This ensures proper cross-rank
    #   synchronization: the sender in iteration N cannot proceed until the
    #   receiver has started iteration N (meaning it finished processing
    #   iteration N-1's data, including any clone operations in user code).
    #
    #   Signaling flow:
    #     iter0: recv waits for DATA_COMPLETE(0), no BUFFER_READY signal (skip)
    #     iter1: recv signals BUFFER_READY(1), then waits for DATA_COMPLETE(1)
    #     iterN: recv signals BUFFER_READY(N), then waits for DATA_COMPLETE(N)
    #
    #   The sender waits for BUFFER_READY(N) before sending iteration N's data.
    #   Since BUFFER_READY(N) is signaled at the start of recv's iteration N,
    #   the receiver has necessarily finished with iteration N-1's data.
    if pid >= total_send_blocks:
        recv_peer = pid - total_send_blocks
        if recv_peer < world_size and recv_peer != my_rank:
            if SYNC_BUFFER:
                # Signal BUFFER_READY at START of iteration N.
                # This tells the sender that we've entered iteration N, meaning
                # we've completed iteration N-1 (including any operations like
                # spin loops that run AFTER the alltoallv kernel in the graph).
                #
                # Signal value: We ADD 1 each iteration (not iteration number) because
                # signal_block uses SignalOp::ADD semantics. The cumulative signal
                # value after N iterations is N, which matches wait_signal_from's
                # expected value of iteration (>= N).
                if iteration > 0:
                    # Add 1 to signal, making cumulative value = iteration
                    signal_block(dst_win_ptr, recv_peer, BUFFER_READY_SIGNAL_ID, 1)

                peer_recv_size = tl.load(recv_sizes_ptr + recv_peer)
                if peer_recv_size > 0:
                    # Wait for monotonically increasing signal value.
                    # The sender signals actual_bpp * (iteration + 1).
                    # In symmetric topology, per_peer_blocks[recv_peer] equals
                    # the sender's bpp for this rank.
                    if HAS_PER_PEER_BLOCKS:
                        sender_bpp = tl.load(per_peer_blocks_ptr + recv_peer)
                    else:
                        sender_bpp = BLOCKS_PER_PEER
                    expected_signal = sender_bpp * (iteration + 1)
                    wait_signal_from(dst_win_ptr, recv_peer, signal_id, expected_signal)

        return

    # ── Send block path ──
    # Map program_id → (peer, block_idx within that peer)
    peer = pid // BLOCKS_PER_PEER
    block_idx = pid % BLOCKS_PER_PEER

    # Skip if peer is out of range (grid may be larger than world_size)
    if peer >= world_size:
        return

    # Per-peer block masking: IB peers may use fewer blocks than NVL peers.
    # BLOCKS_PER_PEER is the constexpr upper bound; actual_bpp is the real
    # count for this peer. Excess blocks early-return.
    if HAS_PER_PEER_BLOCKS:
        actual_bpp = tl.load(per_peer_blocks_ptr + peer)
        if block_idx >= actual_bpp:
            return
    else:
        actual_bpp = BLOCKS_PER_PEER

    # NOTE: No counter reset here. Counters grow monotonically with each
    # iteration to avoid race conditions from non-deterministic CUDA block
    # scheduling. With reset, block1 could execute before block0's reset,
    # causing incorrect signal values. The monotonic approach ensures
    # correctness regardless of block execution order.

    # Load send parameters for this peer from GPU memory
    send_offset = tl.load(send_offsets_ptr + peer)
    send_size = tl.load(send_sizes_ptr + peer)

    # SYNC_BUFFER: Wait for BUFFER_READY signal from receiver before sending.
    # The receiver signals BUFFER_READY(N) at the START of iteration N to indicate
    # it has entered iteration N (meaning iteration N-1 is fully complete, including
    # any operations like spin loops that run after the alltoallv kernel).
    # The sender at iteration N must wait for BUFFER_READY(N) to ensure the receiver
    # has entered iteration N before the sender overwrites the recv_buf with new data.
    # Only block 0 for each peer waits; other blocks proceed after unblock.
    # Skip on iteration 0 since there's no previous data to wait for.
    if SYNC_BUFFER and iteration > 0 and peer != my_rank and send_size > 0:
        if block_idx == 0:
            # Wait for receiver to signal that it has entered iteration N.
            # This means the receiver has completed iteration N-1.
            expected_signal = iteration
            wait_signal_from(dst_win_ptr, peer, BUFFER_READY_SIGNAL_ID, expected_signal)

    # Phase 0: Self-copy (peer == my_rank) using self_copy_block extern.
    # Uses a CUDA extern with memcpy_vectorized instead of Triton's
    # tl.load/tl.store with masks.  The Triton masked-copy pattern generates
    # ~85 comparison/mask SSA values in the LLVM IR (tl.arange, icmp, select,
    # predicated ld/st), adding register pressure to the main kernel even
    # though self-copy only executes for 1 of 8 peers.  The extern approach
    # keeps the self-copy IR in a separate function, reducing register
    # pressure on the hot memcpy path.
    if peer == my_rank and send_size > 0:
        dst_offset = tl.load(recv_offsets_ptr + my_rank)
        block_bytes = send_size // actual_bpp
        block_start_bytes = block_bytes * block_idx
        if block_idx == actual_bpp - 1:
            block_bytes = send_size - block_start_bytes
        if block_bytes > 0:
            self_copy_block(
                recv_buf_ptr,
                dst_offset + block_start_bytes,
                send_buf_ptr,
                send_offset + block_start_bytes,
                block_bytes,
            )

    # Phase 1: Send data to peer (skip self and zero-length messages).
    # Send blocks NEVER call wait_signal_from — they run uninterrupted.
    if peer != my_rank and send_size > 0:
        # Split send_size across actual_bpp blocks for this peer.
        block_portion = send_size // actual_bpp
        block_start = block_portion * block_idx
        # Last block absorbs any remainder bytes from integer division
        if block_idx == actual_bpp - 1:
            block_portion = send_size - block_start

        if block_portion > 0:
            # Per-peer destination offset on the peer's receive buffer.
            dst_offset = tl.load(dst_offsets_ptr + peer)

            # Split NVLink put into two separate externs to avoid
            # dual-alloca register pressure.  Each extern has ONE
            # memcpy_vectorized instantiation = ONE alloca [8 x uint4].
            # The single put_block_chunked had TWO allocas (BLOCK + WARP
            # paths) causing 42 register spills in the hot loop.
            if block_portion <= CHUNK_SIZE:
                put_block_direct(
                    dst_win_ptr,
                    dst_offset + block_start,
                    src_registered_buf,
                    send_offset + block_start,
                    peer,
                    block_portion,
                )
            else:
                put_warp_chunked_direct(
                    dst_win_ptr,
                    dst_offset + block_start,
                    src_registered_buf,
                    send_offset + block_start,
                    peer,
                    block_portion,
                    CHUNK_SIZE,
                )

            # Flush + signal AFTER all data is sent.
            # Signal value: We ADD BLOCKS_PER_PEER each iteration because signal_block
            # uses SignalOp::ADD semantics. The cumulative signal value after N+1
            # iterations is BLOCKS_PER_PEER * (N+1), which matches wait_signal_from's
            # expected value of BLOCKS_PER_PEER * (iteration + 1).
            flush_block(dst_win_ptr)
            if SYNC_BUFFER:
                if actual_bpp == 1:
                    # Add actual_bpp to signal, making cumulative value = actual_bpp * (iteration + 1)
                    signal_block(dst_win_ptr, peer, signal_id, actual_bpp)
                else:
                    # Atomically increment counter. Last block to complete signals.
                    # Counter grows monotonically: iteration 0 ends at actual_bpp,
                    # iteration 1 ends at 2*actual_bpp, etc.
                    old = tl.atomic_add(completion_counters_ptr + peer, 1)
                    expected_count = actual_bpp * (iteration + 1)
                    if old + 1 == expected_count:
                        # Add actual_bpp to signal, making cumulative value = actual_bpp * (iteration + 1)
                        signal_block(dst_win_ptr, peer, signal_id, actual_bpp)


def exchange_offsets(
    local_recv_slot_offsets: torch.Tensor,
    comm: "TorchComm",
) -> torch.Tensor:
    """
    Exchange local_recv_slot_offsets across all ranks to compute remote_write_offsets.

    Each rank's local_recv_slot_offsets[peer] tells where peer's data should
    land in this rank's recv_buf.  After the exchange, remote_write_offsets[peer]
    contains where THIS rank's data should land on peer's recv_buf.

    This is a collective operation (all_to_all_single) and should be called
    ONCE during setup, not per-iteration.

    Args:
        local_recv_slot_offsets: GPU tensor of shape [world_size] with offsets
            within "my local" recv buffer where each peer's data lands.
        comm: TorchComm communicator.

    Returns:
        remote_write_offsets: GPU tensor of shape [world_size] with offsets on
            "remote peers" recv buffers where I should write.
    """
    remote_write_offsets = torch.empty_like(local_recv_slot_offsets)
    comm.all_to_all_single(remote_write_offsets, local_recv_slot_offsets, False)
    return remote_write_offsets


def _tune_for_nvl(max_msg_size_bytes: int) -> dict:
    """NVLink-optimized tuning parameters.

    NVLink scales with block parallelism (each block independently
    saturates a portion of the NVLink bandwidth via cooperative memcpy).
    """
    if max_msg_size_bytes <= 1 * 1024:
        return {"blocks_per_peer": 1, "num_warps": 4, "chunk_size": 64 * 1024}
    if max_msg_size_bytes <= 2 * 1024:
        return {"blocks_per_peer": 1, "num_warps": 8, "chunk_size": 64 * 1024}
    elif max_msg_size_bytes <= 4 * 1024:
        return {"blocks_per_peer": 1, "num_warps": 8, "chunk_size": 64 * 1024}
    elif max_msg_size_bytes <= 8 * 1024:
        return {"blocks_per_peer": 1, "num_warps": 16, "chunk_size": 64 * 1024}
    elif max_msg_size_bytes <= 16 * 1024:
        return {"blocks_per_peer": 1, "num_warps": 16, "chunk_size": 64 * 1024}
    elif max_msg_size_bytes <= 32 * 1024:
        return {"blocks_per_peer": 1, "num_warps": 16, "chunk_size": 64 * 1024}
    elif max_msg_size_bytes <= 64 * 1024:
        return {"blocks_per_peer": 1, "num_warps": 32, "chunk_size": 64 * 1024}
    elif max_msg_size_bytes <= 128 * 1024:
        return {"blocks_per_peer": 8, "num_warps": 16, "chunk_size": 64 * 1024}
    elif max_msg_size_bytes <= 256 * 1024:
        return {"blocks_per_peer": 8, "num_warps": 16, "chunk_size": 64 * 1024}
    else:
        return {"blocks_per_peer": 16, "num_warps": 16, "chunk_size": 64 * 1024}


def _tune_for_ib(max_msg_size_bytes: int) -> dict:
    """IB/RDMA-optimized tuning parameters.

    Empirical data from exhaustive sweep on 2x8 H100 (64 param combos
    per message size, CUDA graph mode). Key findings:
    - 1 bpp optimal up to 1MB (RDMA NIC pipelines internally)
    - 4 bpp for 2-4MB (some block parallelism helps)
    - 8 bpp for >=8MB (more blocks to saturate bandwidth)
    - 512KB chunks dominate for medium/large messages
    """
    if max_msg_size_bytes <= 1 * 1024:  # 1KB: 2.35x vs NCCL
        return {"blocks_per_peer": 1, "num_warps": 4, "chunk_size": 256 * 1024}
    elif max_msg_size_bytes <= 4 * 1024:  # 4KB: 2.05x
        return {"blocks_per_peer": 1, "num_warps": 16, "chunk_size": 128 * 1024}
    elif max_msg_size_bytes <= 16 * 1024:  # 16KB: 2.35x
        return {"blocks_per_peer": 1, "num_warps": 8, "chunk_size": 256 * 1024}
    elif max_msg_size_bytes <= 64 * 1024:  # 64KB: 2.76x
        return {"blocks_per_peer": 1, "num_warps": 4, "chunk_size": 64 * 1024}
    elif max_msg_size_bytes <= 256 * 1024:  # 256KB: 1.91x
        return {"blocks_per_peer": 1, "num_warps": 8, "chunk_size": 512 * 1024}
    elif max_msg_size_bytes <= 512 * 1024:  # 512KB: 1.92x
        return {"blocks_per_peer": 1, "num_warps": 4, "chunk_size": 512 * 1024}
    elif max_msg_size_bytes <= 1 * 1024 * 1024:  # 1MB: 1.76x
        return {"blocks_per_peer": 1, "num_warps": 16, "chunk_size": 256 * 1024}
    elif max_msg_size_bytes <= 4 * 1024 * 1024:  # 2-4MB: 1.51-1.60x
        return {"blocks_per_peer": 4, "num_warps": 16, "chunk_size": 512 * 1024}
    else:  # >=8MB: 1.42-1.47x
        return {"blocks_per_peer": 8, "num_warps": 16, "chunk_size": 512 * 1024}


def auto_tune_alltoallv_params(
    max_msg_size_bytes: int,
    peer_is_nvl: "list[bool] | None" = None,
) -> dict:
    """
    Select optimal kernel parameters based on maximum per-peer message size.

    When peer_is_nvl is provided, returns topology-aware config with
    per-peer block counts. NVL peers get NVLink-optimized parameters,
    IB peers get RDMA-optimized parameters. The kernel launches with
    the max blocks_per_peer as a constexpr upper bound and masks excess
    blocks at runtime.

    Args:
        max_msg_size_bytes: Maximum per-peer message size in bytes.
        peer_is_nvl: Optional list of booleans per peer. True = NVLink,
            False = IB. None = all NVLink (backward compatible).

    Returns:
        dict with keys: blocks_per_peer, num_warps, chunk_size,
            per_peer_blocks (list[int] or None)
    """
    nvl_config = _tune_for_nvl(max_msg_size_bytes)

    if peer_is_nvl is None or all(peer_is_nvl):
        return {
            "blocks_per_peer": nvl_config["blocks_per_peer"],
            "num_warps": nvl_config["num_warps"],
            "chunk_size": nvl_config["chunk_size"],
            "per_peer_blocks": None,
        }

    ib_config = _tune_for_ib(max_msg_size_bytes)

    max_bpp = max(nvl_config["blocks_per_peer"], ib_config["blocks_per_peer"])

    per_peer_blocks = [
        nvl_config["blocks_per_peer"] if is_nvl else ib_config["blocks_per_peer"]
        for is_nvl in peer_is_nvl
    ]

    return {
        "blocks_per_peer": max_bpp,
        "num_warps": max(nvl_config["num_warps"], ib_config["num_warps"]),
        "chunk_size": nvl_config["chunk_size"],
        "per_peer_blocks": per_peer_blocks,
    }


def device_alltoallv_dynamic(
    send_buf: torch.Tensor,
    recv_buf: torch.Tensor,
    send_sizes: torch.Tensor,
    send_offsets: torch.Tensor,
    recv_sizes: torch.Tensor,
    local_recv_slot_offsets: torch.Tensor,
    remote_write_offsets: torch.Tensor,
    dev_win_ptr: int,
    src_info: int,
    my_rank: int,
    world_size: int,
    num_warps: int = 16,
    blocks_per_peer: int = 1,
    chunk_size: int = 64 * 1024,
    auto_tune: bool = False,
    sync_buffer: bool = True,
    per_peer_blocks: "torch.Tensor | None" = None,
) -> None:
    """
    Perform device-initiated AlltoAllv with dynamic per-rank counts.

    This is a lean collective with zero host-side overhead — no barriers,
    no torch.cuda.synchronize(), no CUDA allocations, no offset exchange.
    It simply launches the Triton kernel.  All setup (buffer registration,
    offset exchange, get_device_window) is done once by the caller.

    This matches the pipes/collectives and ctran patterns where the
    collective call is just a kernel launch with no per-call overhead.

    Iteration tracking is managed internally — users don't need to create
    or manage iteration tensors. The function auto-increments the iteration
    counter after each call, enabling correct CUDA graph replay behavior.

    Args:
        send_buf: Source tensor containing data to send.
        recv_buf: Destination tensor to receive data into.
        send_sizes: GPU tensor [world_size] with byte counts per peer.
        send_offsets: GPU tensor [world_size] with send buffer offsets.
        recv_sizes: GPU tensor [world_size] with byte counts expected from
                    each peer (used to gate wait_signal_from).
        local_recv_slot_offsets: GPU tensor [world_size] with offsets within
                     "my local" recv buffer where each peer's data lands.
        remote_write_offsets: GPU tensor [world_size] with offsets on "remote
                     peers" recv buffers where I should write (from
                     exchange_offsets()).
        dev_win_ptr: Device window handle (from get_device_window()).
        src_info: Pre-registered send buffer info (from register_local_buffer()).
        my_rank: This rank's ID.
        world_size: Total number of ranks.
        num_warps: Number of warps per block (default: 16, i.e. 512 threads).
                   Controls thread-level parallelism for NVLink memcpy in
                   put_block. Higher values increase copy bandwidth but may
                   reduce occupancy due to register pressure. Valid range:
                   1-32 (CUDA max 1024 threads/block). The pipes AllToAllv
                   benchmarks use 16 warps/block (512 threads) for comparison.
        blocks_per_peer: Number of blocks per peer (default: 1).  With
                         blocks_per_peer > 1, each peer's send data is split
                         into contiguous chunks and each block issues its own
                         put_block — increasing NVLink parallelism for large
                         messages.  Only the last block signals the peer, so
                         the signaling protocol is unchanged.
        sync_buffer: If True (default), enables BUFFER_READY cross-rank
                     synchronization for safe buffer reuse across iterations.
                     This is REQUIRED when using CUDA graphs or replaying
                     multiple iterations without explicit host sync. Adds
                     ~1-3us per-peer latency overhead. Set to False only for
                     microbenchmarking raw kernel throughput.

    Example:
        >>> # One-time setup (call prewarm_completion_counters BEFORE GIN activation)
        >>> prewarm_completion_counters(world_size, device)
        >>> window = comm.new_window()
        >>> window.tensor_register(recv_buf)
        >>> dev_win_ptr = get_device_window(window, signal_count=world_size)
        >>> src_info = register_local_buffer(send_buf, window)
        >>> remote_write_offsets = exchange_offsets(local_recv_slot_offsets, comm)
        >>>
        >>> # Hot loop — zero per-call overhead, iteration managed internally
        >>> for i in range(iterations):
        ...     device_alltoallv_dynamic(
        ...         send_buf, recv_buf,
        ...         send_sizes, send_offsets,
        ...         recv_sizes, local_recv_slot_offsets, remote_write_offsets,
        ...         dev_win_ptr, src_info, my_rank, world_size,
        ...     )
        >>>
        >>> # Cleanup
        >>> deregister_local_buffer(src_info, window)
    """
    src_registered_buf = src_info

    # Get internal iteration tensor (cached per device/world_size).
    # This is managed internally - users don't need to create or track it.
    iteration_tensor = _get_iteration_tensor(world_size, send_buf.device)

    # Auto-tune: select optimal parameters based on max message size.
    # Reads send_sizes.max() from GPU (~5us overhead, negligible for
    # messages >16KB where auto-tuning matters).
    if auto_tune:
        max_msg_size_bytes = int(send_sizes.max().item())
        params = auto_tune_alltoallv_params(max_msg_size_bytes)
        blocks_per_peer = params["blocks_per_peer"]
        num_warps = params["num_warps"]
        chunk_size = params["chunk_size"]

    # Completion counters for multi-block coordination.
    # Pre-allocated via _get_completion_counters() to avoid per-call
    # CUDA allocator overhead.  The cache is keyed by (device, world_size)
    # and lazily created on first use.
    # For BLOCKS_PER_PEER == 1 the kernel never accesses these, but we
    # still need a valid pointer for the uniform launch signature.
    #
    # IMPORTANT: Reset counters to the expected base value for this iteration.
    # The kernel expects counters to start at BLOCKS_PER_PEER * iteration
    # (the value after the previous iteration).  Without this reset,
    # counters would accumulate across independent kernel launches,
    # causing the signal condition (old + 1 == expected_count) to fail.
    #
    # We use _fill_completion_counters_from_iteration() which reads the
    # iteration value from GPU memory, making it CUDA graph compatible.
    completion_counters = _get_completion_counters(world_size, send_buf.device)
    if blocks_per_peer > 1:
        # Reset each peer's counter using a GIN-safe Triton kernel that
        # reads iteration from GPU memory.  This is CUDA graph compatible.
        # When per_peer_blocks is provided, each counter is reset to its
        # own per_peer_blocks[i] * iteration (not the uniform max).
        _fill_completion_counters_from_iteration(
            completion_counters,
            iteration_tensor,
            blocks_per_peer,
            world_size,
            per_peer_blocks=per_peer_blocks,
        )

    grid = (world_size * blocks_per_peer + world_size,)

    _device_alltoallv_dynamic_kernel[grid](
        dev_win_ptr,
        src_registered_buf,
        send_buf,
        recv_buf,
        send_offsets,
        send_sizes,
        local_recv_slot_offsets,
        recv_sizes,
        remote_write_offsets,
        my_rank=my_rank,
        world_size=world_size,
        signal_id=0,
        # Read iteration from internal GPU tensor - works for both CUDA graph
        # and non-graph modes. The tensor is auto-incremented at the end of
        # this function.
        iteration_ptr=iteration_tensor,
        ELEM_BYTES=send_buf.element_size(),
        BLOCK_SIZE=1024,
        BLOCKS_PER_PEER=blocks_per_peer,
        CHUNK_SIZE=chunk_size,
        num_warps=num_warps,
        completion_counters_ptr=completion_counters,
        # Sync buffer mode for buffer-ready synchronization
        SYNC_BUFFER=sync_buffer,
        BUFFER_READY_SIGNAL_ID=1,  # Use signal_id=1 for BUFFER_READY (signal_id=0 is DATA_COMPLETE)
        # Per-peer block counts for topology-aware scheduling
        per_peer_blocks_ptr=per_peer_blocks,
        HAS_PER_PEER_BLOCKS=per_peer_blocks is not None,
    )

    # Auto-increment iteration counter on GPU.
    # Using a Triton kernel is a pure GPU operation that can be captured in
    # CUDA graphs, enabling correct cross-rank synchronization across
    # multiple graph replays.
    _increment_iteration_kernel[(1,)](iteration_tensor)


# =============================================================================
# Helper Functions
# =============================================================================


def compute_offsets_from_sizes(
    sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> None:
    """
    Compute exclusive prefix sum of sizes to get offsets.

    This utility enables the dynamic workflow:
    1. Kernel computes per-peer sizes
    2. This function computes offsets from sizes (stays on GPU)
    3. device_alltoallv_dynamic uses both

    Uses PyTorch's torch.cumsum which runs efficiently on GPU when
    tensors are on GPU.

    Args:
        sizes: GPU tensor of shape [N] with per-peer byte counts
        offsets: GPU tensor of shape [N] to store computed offsets
                 (must be pre-allocated)

    Example:
        >>> sizes = torch.tensor([100, 200, 150, 250], dtype=torch.int64, device='cuda')
        >>> offsets = torch.zeros_like(sizes)
        >>> compute_offsets_from_sizes(sizes, offsets)
        >>> print(offsets)  # tensor([0, 100, 300, 450])
    """
    N = sizes.shape[0]
    if N == 0:
        return
    elif N == 1:
        offsets[0] = 0
    else:
        # Compute exclusive prefix sum using torch.cumsum
        # exclusive_prefix_sum[i] = sum(sizes[0:i])
        cumsum = torch.cumsum(sizes, dim=0)
        offsets[0] = 0
        offsets[1:] = cumsum[:-1]
