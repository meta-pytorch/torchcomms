# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Triton NVLink-only copy-based send/recv kernel.

N-rank group-aware copy-based send, receive, or bidirectional sendrecv,
double-buffered staging, monotonic int64 head/tail signaling. CUDA-graph
replayable via persistent step-state counters that advance across replays.

Supports separate ``send_peer`` and ``recv_peer`` (e.g., for ring
topology: ``send_peer = (rank+1) % N``, ``recv_peer = (rank-1) % N``).
All pointer resolution is done on the host side — the kernel receives
pre-sliced staging buffer, signal pad, and step state pointers for the
specific ``(send_peer, recv_peer)`` pair. This preserves direct typed
tensor arguments, avoiding the store scalarization (``STG.E.U16``) that
pointer-of-pointer indirection causes.

Architecture
------------

::

    Rank A (sender → send_peer)          send_peer (receiver)
    ┌─────────────┐                      ┌─────────────┐
    │ src tensor  │──(1) read──→         │             │
    │ (local HBM) │            │         │             │
    └─────────────┘            │         │             │
                               ▼         │             │
                   ┌─────────────────────┐             │
                   │ staging buffer      │─(2) read──→ │
                   │ (send_peer symm mem │       ▼     │
                   │  local_rank region) │  ┌─────────────┐
                   └─────────────────────┘  │ dst tensor  │
                                            └─────────────┘

    recv_peer (sender)                   Rank A (receiver ← recv_peer)
    ┌─────────────┐                      ┌─────────────┐
    │ src tensor  │──(1) read──→         │             │
    └─────────────┘            │         │             │
                               ▼         │             │
                   ┌─────────────────────┐             │
                   │ staging buffer      │─(2) read──→ │
                   │ (local_rank symm mem│       ▼     │
                   │  recv_peer region)  │  ┌─────────────┐
                   └─────────────────────┘  │ dst tensor  │
                                            └─────────────┘

All remote operations are NVLink fire-and-forget writes (zero NVLink reads).

Block / signal / buffer mapping
-------------------------------

Each block ``k`` owns a fixed staging-buffer region
``[k * BLOCK_STRIDE * PIPELINE_DEPTH, (k+1) * BLOCK_STRIDE * PIPELINE_DEPTH)``
and a fixed ``(head[k], tail[k])`` signal pair. Sender block ``k`` and
receiver block ``k`` use the SAME signal slot ``k`` and the SAME buffer
region. The block→region mapping is determined by ``MAX_BLOCKS_PER_PEER``
(constant) and is independent of the runtime ``NUM_SEND_BLOCKS``, so
back-to-back kernel launches with different block counts cannot race.

Signal pointers are pre-resolved by the host into the N-rank per-(peer,
block) layout. The kernel sees them as simple base pointers and indexes
by ``block_id`` only.

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

Intra-block synchronization pattern (used 4× below):

  * ``if flat_tid == 0: <wait>`` + ``sync_threads()`` — only thread 0
    polls the signal; the barrier publishes that result to the rest of
    the block before any data thread observes the slot as ready.
  * ``sync_threads()`` + ``if flat_tid == 0: <publish>`` — all data
    threads must have committed their stores (or completed their loads)
    before thread 0 advances the counter and releases the slot.

Removing or reordering either ``sync_threads()`` call is a silent
correctness bug — TLS / per-warp scheduling will break the producer/
consumer ordering invariant.

Specialization note: ``send_numel`` and ``recv_numel`` are intentionally
specialized (no ``do_not_specialize``). Distinct ``(send_numel,
recv_numel)`` pairs trigger Triton recompilation. For ring AllGather
(single shard size per call) this is fine; callers driving many
distinct sizes per process should reuse buffers.

Sub-slot signaling
------------------

The ``CHUNKS_PER_SLOT`` constexpr controls how many DATA_READY signals
are issued per pipeline slot fill. When ``CHUNKS_PER_SLOT == 1``
(the default, ``signal_bytes == BLOCK_STRIDE_BYTES``), behavior is
identical to signaling once per slot — full backward compatibility.

When ``CHUNKS_PER_SLOT > 1`` (``signal_bytes < BLOCK_STRIDE_BYTES``),
each slot is divided into chunks of ``SIGNAL_STRIDE_ELEMS`` elements.
The sender signals DATA_READY after each chunk. Backpressure (SLOT_FREE)
fires only at slot boundaries.
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


@triton.jit
def triton_nvl_sendrecv_kernel(  # noqa: C901
    send_ptr,
    recv_ptr,
    send_staging_buf,
    recv_staging_buf,
    send_tail_sig,
    send_head_sig,
    recv_tail_sig,
    recv_head_sig,
    sender_step_ptr,
    recver_step_ptr,
    send_numel,
    recv_numel,
    TILE_ROWS: tl.constexpr,
    TILE_COLS: tl.constexpr,
    BLOCK_STRIDE_ELEMS: tl.constexpr,
    SIGNAL_STRIDE_ELEMS: tl.constexpr,
    PIPELINE_DEPTH: tl.constexpr,
    MAX_BLOCKS_PER_PEER: tl.constexpr,
    NUM_SEND_BLOCKS: tl.constexpr,
    TILES_PER_SIGNAL: tl.constexpr,
    CHUNKS_PER_SLOT: tl.constexpr,
    MODE: tl.constexpr,
):
    """NVLink copy-based send/recv kernel with pre-resolved N-rank pointers.

    The host passes peer-specific staging buffers, signal pads, and step-state
    rows directly. That keeps the kernel simple and avoids pointer-of-pointer
    loads that can pessimize Triton store codegen.
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
        is_sender = (pid % 2) == 0
        block_id = pid // 2

    flat_tid = get_flat_tid()

    if is_sender:
        # ===== SENDER =====
        sender_id = block_id

        # Signal pointers: host pre-resolved to the correct per-(peer, block) base.
        # Buffer and signal pointers are already typed kernel arguments —
        # no pointer-of-pointer load needed (preserves the STG.E.U16-free
        # vectorized-store codegen pattern).
        # tail: REMOTE — written here, polled by remote receiver
        # head: LOCAL  — polled here, written by remote receiver
        tail_remote_ptr = send_tail_sig + sender_id
        head_local_ptr = send_head_sig + sender_id

        start_step = tl.load(sender_step_ptr + sender_id).to(tl.int64)

        local_rows = tl.arange(0, TILE_ROWS)
        local_cols = tl.arange(0, TILE_COLS)
        num_send_tiles = tl.cdiv(send_numel, TILE_NUMEL)

        # int64 tile index so `send_tile_idx * TILE_NUMEL` cannot overflow
        # int32 for multi-GiB tensors.
        send_tile_idx = sender_id.to(tl.int64)
        send_step = start_step

        while send_tile_idx < num_send_tiles:
            if CHUNKS_PER_SLOT == 1:
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
            else:
                chunk_in_slot = (send_step - start_step) % CHUNKS_PER_SLOT
                slot = ((send_step - start_step) // CHUNKS_PER_SLOT) % PIPELINE_DEPTH
                slot_base = (
                    sender_id * BLOCK_STRIDE_ELEMS * PIPELINE_DEPTH
                    + slot * BLOCK_STRIDE_ELEMS
                    + chunk_in_slot * SIGNAL_STRIDE_ELEMS
                )
                if chunk_in_slot == 0:
                    if send_step < start_step + CHUNKS_PER_SLOT * PIPELINE_DEPTH:
                        wait_target = start_step
                    else:
                        wait_target = send_step - CHUNKS_PER_SLOT * PIPELINE_DEPTH + 1
                    if flat_tid == 0:
                        wait_ge_volatile(head_local_ptr, wait_target.to(tl.int64))
                sync_threads()

            buf_tile_idx = 0
            tiles_to_send_this_step = min(
                TILES_PER_SIGNAL,
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
                tl.store(send_staging_buf + slot_base + buf_off, data, mask=mask)

                send_tile_idx += NUM_SEND_BLOCKS
                buf_tile_idx += 1

            sync_threads()
            # ``bar.sync 0`` (sync_threads) above ensures all sender threads have
            # finished issuing their staging stores before thread 0 issues the
            # system fence. Without this barrier, thread 0's fence would only
            # drain its own in-flight stores, leaving other threads' stores
            # unsynchronized with the TAIL publish.
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

        # Signal pointers: host pre-resolved to the correct per-(peer, block) base.
        # tail: LOCAL  — polled here, written by remote sender
        # head: REMOTE — written here, polled by remote sender
        tail_local_ptr = recv_tail_sig + receiver_id
        head_remote_ptr = recv_head_sig + receiver_id

        start_step = tl.load(recver_step_ptr + receiver_id).to(tl.int64)

        local_rows = tl.arange(0, TILE_ROWS)
        local_cols = tl.arange(0, TILE_COLS)
        num_recv_tiles = tl.cdiv(recv_numel, TILE_NUMEL)

        # int64 tile index so `recv_tile_idx * TILE_NUMEL` cannot overflow
        # int32 for multi-GiB tensors.
        recv_tile_idx = receiver_id.to(tl.int64)
        recv_step = start_step

        while recv_tile_idx < num_recv_tiles:
            if CHUNKS_PER_SLOT == 1:
                slot = (recv_step - start_step) % PIPELINE_DEPTH
                slot_base = (
                    receiver_id * BLOCK_STRIDE_ELEMS * PIPELINE_DEPTH
                    + slot * BLOCK_STRIDE_ELEMS
                )
            else:
                chunk_in_slot = (recv_step - start_step) % CHUNKS_PER_SLOT
                slot = ((recv_step - start_step) // CHUNKS_PER_SLOT) % PIPELINE_DEPTH
                slot_base = (
                    receiver_id * BLOCK_STRIDE_ELEMS * PIPELINE_DEPTH
                    + slot * BLOCK_STRIDE_ELEMS
                    + chunk_in_slot * SIGNAL_STRIDE_ELEMS
                )

            if flat_tid == 0:
                wait_ge(tail_local_ptr, (recv_step + 1).to(tl.int64))
            sync_threads()

            buf_tile_idx = 0
            tiles_to_recv_this_step = min(
                TILES_PER_SIGNAL,
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
                data = tl.load(recv_staging_buf + slot_base + buf_off, mask=mask)
                tl.store(recv_ptr + flat_idx.to(tl.int64), data, mask=mask)

                recv_tile_idx += NUM_SEND_BLOCKS
                buf_tile_idx += 1

            sync_threads()
            if CHUNKS_PER_SLOT == 1:
                if flat_tid == 0:
                    remote_store_i64_relaxed(
                        head_remote_ptr, (recv_step + 1).to(tl.int64)
                    )
            else:
                last_in_slot = (chunk_in_slot == CHUNKS_PER_SLOT - 1) or (
                    recv_tile_idx >= num_recv_tiles
                )
                if last_in_slot:
                    if flat_tid == 0:
                        remote_store_i64_relaxed(
                            head_remote_ptr, (recv_step + 1).to(tl.int64)
                        )
            recv_step += 1

        tl.store(recver_step_ptr + receiver_id, recv_step.to(tl.int64))
