# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Triton NVLink-only copy-based send/recv kernel.

Single peer-pair copy-based send, receive, or bidirectional sendrecv,
double-buffered staging, monotonic int64 head/tail signaling. CUDA-graph
replayable via persistent step-state counters that advance across replays.

Scope: 2-rank groups only. ``send_peer`` and ``recv_peer`` are the same
rank (single ``peer_rank`` arg). This kernel is **not** directly usable
for Ring AllGather on world_size > 2 — ring needs ``(send_peer,
recv_peer) = ((rank+1)%N, (rank-1)%N)`` plus per-sender staging slots.
See ``sendrecv_op.py`` module docstring for the planned API/protocol
extension.

Architecture
------------

::

    Rank A (sender)                        Rank B (receiver)
    ┌─────────────┐                        ┌─────────────┐
    │ src tensor  │──(1) read──→           │             │
    │ (local HBM) │            │           │             │
    └─────────────┘            │           │             │
                               ▼           │             │
                   ┌─────────────────────┐ │             │
                   │ staging buffer      │ │             │
                   │ (Rank B symm mem)   │─(2) read───→  │
                   │ [slot 0 | slot 1]   │ │       ▼     │
                   └─────────────────────┘ │  ┌─────────────┐
                                           │  │ dst tensor  │
                                           │  │ (local HBM) │
                                           │  └─────────────┘
                                           └────────────────┘

    (1) Sender reads local src, REMOTE-writes to peer's staging buffer
    (2) Receiver LOCAL-reads its staging buffer, writes to local dst

All remote operations are NVLink fire-and-forget writes (zero NVLink reads).

Block / signal / buffer mapping
-------------------------------

Each block ``k`` owns a fixed staging-buffer region
``[k * BLOCK_STRIDE * PIPELINE_DEPTH, (k+1) * BLOCK_STRIDE * PIPELINE_DEPTH)``
and a fixed ``(head[k], tail[k])`` signal pair. Sender block ``k`` and
receiver block ``k`` use the SAME signal slot ``k`` and the SAME buffer
region. The block→region mapping is determined by ``MAX_BLOCKS_PER_DIR``
(constant) and is independent of the runtime ``NUM_SEND_BLOCKS``, so
back-to-back kernel launches with different block counts cannot race.

Signal-pad layout per rank (int64 entries; ``MBPD = MAX_BLOCKS_PER_DIR``)::

    [0 .. MBPD):       TAIL — written by remote sender, polled by local receiver
    [MBPD .. 2*MBPD):  HEAD — written by remote receiver, polled by local sender

Memory ordering:

  * **TAIL** (sender→receiver, publishes payload data): producer uses
    :func:`fence_and_remote_store_i64` (release fence + relaxed store);
    consumer uses :func:`wait_ge` (acquire poll). The fence guarantees the
    receiver sees the staging-buffer data once it sees the new TAIL.

  * **HEAD** (receiver→sender, only transfers slot ownership): producer
    uses :func:`remote_store_i64_relaxed` (no fence) preceded by a local
    ``sync_threads()`` execution barrier. Consumer uses
    :func:`wait_ge_volatile` (no acquire). The sender does not read any
    payload written by the receiver, so the fence is unnecessary.
"""

from __future__ import annotations

import triton
import triton.language as tl

from .utils import (
    fence_and_remote_store_i64,
    get_flat_tid,
    remote_store_i64_relaxed,
    sync_threads,
    wait_ge,
    wait_ge_volatile,
)

_MODE_BIDIRECTIONAL = 0
_MODE_SEND_ONLY = 1
_MODE_RECV_ONLY = 2


@triton.jit(do_not_specialize=["local_rank", "peer_rank"])
def triton_nvl_sendrecv_kernel(  # noqa: C901
    send_ptr,
    recv_ptr,
    buffer_ptrs,  # int64 device tensor, shape (world_size,)
    signal_pad_ptrs,  # int64 device tensor, shape (world_size,)
    sender_step_ptr,  # int64 device tensor, shape (MAX_BLOCKS_PER_DIR,)
    recver_step_ptr,  # int64 device tensor, shape (MAX_BLOCKS_PER_DIR,)
    local_rank,
    peer_rank,
    send_numel,
    recv_numel,
    TILE_ROWS: tl.constexpr,
    TILE_COLS: tl.constexpr,
    BLOCK_STRIDE_ELEMS: tl.constexpr,
    PIPELINE_DEPTH: tl.constexpr,
    MAX_BLOCKS_PER_DIR: tl.constexpr,
    NUM_SEND_BLOCKS: tl.constexpr,
    MAX_TILES_PER_BLOCK_PER_SLOT: tl.constexpr,
    MODE: tl.constexpr,
):
    """NVLink copy-based send/recv kernel.

    Bidirectional mode launches grid ``(2 * NUM_SEND_BLOCKS,)``. The first
    ``NUM_SEND_BLOCKS`` programs run the sender path, and the rest run the
    receiver path. Send-only and recv-only modes launch grid
    ``(NUM_SEND_BLOCKS,)`` and run exactly one path.
    """
    TILE_NUMEL: tl.constexpr = TILE_ROWS * TILE_COLS

    pid = tl.program_id(0)
    if MODE == _MODE_SEND_ONLY:
        is_sender = True
        block_id = pid
    elif MODE == _MODE_RECV_ONLY:
        is_sender = False
        block_id = pid
    else:
        is_sender = pid < NUM_SEND_BLOCKS
        block_id = pid % NUM_SEND_BLOCKS

    # ---- Buffer pointers ----
    # Sender writes to peer rank's staging buffer; receiver reads its own.
    buffer_ptrs_u64 = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    remote_buf = tl.load(buffer_ptrs_u64 + peer_rank).to(
        tl.pointer_type(send_ptr.dtype.element_ty)
    )
    local_buf = tl.load(buffer_ptrs_u64 + local_rank).to(
        tl.pointer_type(recv_ptr.dtype.element_ty)
    )

    # ---- Signal pad pointers ----
    signal_pad_ptrs_u64 = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    remote_sig = tl.load(signal_pad_ptrs_u64 + peer_rank).to(tl.pointer_type(tl.int64))
    local_sig = tl.load(signal_pad_ptrs_u64 + local_rank).to(tl.pointer_type(tl.int64))

    HEAD_OFFSET: tl.constexpr = MAX_BLOCKS_PER_DIR

    flat_tid = get_flat_tid()

    if is_sender:
        # ===== SENDER =====
        sender_id = block_id

        # tail: REMOTE — written here, polled by remote receiver
        # head: LOCAL — polled here, written by remote receiver
        tail_remote_ptr = remote_sig + sender_id
        head_local_ptr = local_sig + HEAD_OFFSET + sender_id

        start_step = tl.load(sender_step_ptr + sender_id).to(tl.int64)

        local_rows = tl.arange(0, TILE_ROWS)
        local_cols = tl.arange(0, TILE_COLS)
        num_send_tiles = tl.cdiv(send_numel, TILE_NUMEL)

        # int64 tile index so `send_tile_idx * TILE_NUMEL` cannot overflow
        # int32 for multi-GiB tensors.
        send_tile_idx = sender_id.to(tl.int64)
        send_step = start_step

        while send_tile_idx < num_send_tiles:
            slot = (send_step - start_step) % PIPELINE_DEPTH
            slot_base = (
                sender_id * BLOCK_STRIDE_ELEMS * PIPELINE_DEPTH
                + slot * BLOCK_STRIDE_ELEMS
            )

            # Wait for slot to be free. For the first PIPELINE_DEPTH steps
            # of a non-first call, the slot may still hold data the remote
            # receiver has not yet consumed — gate on start_step.
            if send_step < start_step + PIPELINE_DEPTH:
                wait_target = start_step
            else:
                wait_target = send_step - PIPELINE_DEPTH + 1
            if flat_tid == 0:
                wait_ge_volatile(head_local_ptr, wait_target.to(tl.int64))
            sync_threads()

            buf_tile_idx = 0
            tiles_to_send_this_step = min(
                MAX_TILES_PER_BLOCK_PER_SLOT,
                tl.cdiv(num_send_tiles - send_tile_idx, NUM_SEND_BLOCKS),
            )
            for _i in range(tiles_to_send_this_step):
                flat_idx = (
                    send_tile_idx * TILE_NUMEL
                    + local_rows[:, None] * TILE_COLS
                    + local_cols[None, :]
                )
                mask = flat_idx < send_numel
                data = tl.load(send_ptr + flat_idx.to(tl.int64), mask=mask)

                buf_t = buf_tile_idx * TILE_ROWS + local_rows
                buf_off = (buf_t[:, None] * TILE_COLS + local_cols[None, :]).to(
                    tl.int64
                )
                tl.store(remote_buf + slot_base + buf_off, data, mask=mask)

                send_tile_idx += NUM_SEND_BLOCKS
                buf_tile_idx += 1

            sync_threads()
            # ``bar.sync 0`` ensures all sender threads have finished issuing
            # their staging stores before thread 0 issues the system fence.
            # Without this barrier, thread 0's fence would only drain its own
            # in-flight stores, leaving other threads' stores unsynchronized
            # with the TAIL publish.
            if flat_tid == 0:
                fence_and_remote_store_i64(
                    tail_remote_ptr, (send_step + 1).to(tl.int64)
                )
            send_step += 1

        # Persist for next launch's start_step.
        tl.store(sender_step_ptr + sender_id, send_step.to(tl.int64))

    else:
        # ===== RECEIVER =====
        receiver_id = block_id

        # tail: LOCAL — polled here, written by remote sender
        # head: REMOTE — written here, polled by remote sender
        tail_local_ptr = local_sig + receiver_id
        head_remote_ptr = remote_sig + HEAD_OFFSET + receiver_id

        start_step = tl.load(recver_step_ptr + receiver_id).to(tl.int64)

        local_rows = tl.arange(0, TILE_ROWS)
        local_cols = tl.arange(0, TILE_COLS)
        num_recv_tiles = tl.cdiv(recv_numel, TILE_NUMEL)

        # int64 tile index so `recv_tile_idx * TILE_NUMEL` cannot overflow
        # int32 for multi-GiB tensors.
        recv_tile_idx = receiver_id.to(tl.int64)
        recv_step = start_step

        while recv_tile_idx < num_recv_tiles:
            slot = (recv_step - start_step) % PIPELINE_DEPTH
            slot_base = (
                receiver_id * BLOCK_STRIDE_ELEMS * PIPELINE_DEPTH
                + slot * BLOCK_STRIDE_ELEMS
            )

            if flat_tid == 0:
                wait_ge(tail_local_ptr, (recv_step + 1).to(tl.int64))
            sync_threads()

            buf_tile_idx = 0
            tiles_to_recv_this_step = min(
                MAX_TILES_PER_BLOCK_PER_SLOT,
                tl.cdiv(num_recv_tiles - recv_tile_idx, NUM_SEND_BLOCKS),
            )
            for _i in range(tiles_to_recv_this_step):
                flat_idx = (
                    recv_tile_idx * TILE_NUMEL
                    + local_rows[:, None] * TILE_COLS
                    + local_cols[None, :]
                )
                mask = flat_idx < recv_numel

                buf_t = buf_tile_idx * TILE_ROWS + local_rows
                buf_off = (buf_t[:, None] * TILE_COLS + local_cols[None, :]).to(
                    tl.int64
                )
                data = tl.load(local_buf + slot_base + buf_off, mask=mask)
                tl.store(recv_ptr + flat_idx.to(tl.int64), data, mask=mask)

                recv_tile_idx += NUM_SEND_BLOCKS
                buf_tile_idx += 1

            sync_threads()
            if flat_tid == 0:
                remote_store_i64_relaxed(head_remote_ptr, (recv_step + 1).to(tl.int64))
            recv_step += 1

        tl.store(recver_step_ptr + receiver_id, recv_step.to(tl.int64))
