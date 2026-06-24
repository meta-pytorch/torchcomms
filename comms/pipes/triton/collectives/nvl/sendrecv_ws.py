# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Warp-specialized bidirectional Triton NVLink send/recv kernel.

Single CTA runs sender and receiver as concurrent TLX async tasks
(default: 4 sender warps + 4 receiver warps). Both NVLink directions
share the SM, mirroring NCCL's per-channel send/recv split.

Staging-buffer / signal layout / monotonic step-state protocol are
inherited from :mod:`sendrecv` — see that module's docstring for the
pre-resolved per-(peer, block) signal layout and the
``fence_and_remote_store_i64`` / ``wait_ge`` ordering invariants.

Known limitation — TLX `sync_threads()` iteration bug
-----------------------------------------------------

``tlx.sync_threads()`` inside a multi-warp ``async_task`` has an
iteration-count-dependent barrier bug: while-loops that execute more
than 2 iterations hang on the second-pass barrier. This kernel relies
on the host launcher (:func:`sendrecv_op._launch`) computing
``BLOCK_STRIDE_ELEMS`` adaptively so that
``ceil(tiles_per_block / max_tiles_per_block_per_slot) <= 2``. The
launcher fail-loud raises if that bound cannot be satisfied — DO NOT
remove that guard. See ``TLX_BARRIER_BUG.md`` for upstream tracking.

Each ``sync_threads()`` call below is annotated with
``TLX-BARRIER-BUG`` to flag the trigger sites for future maintainers.
"""

from __future__ import annotations

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx  # @manual=//triton:triton

from .utils import (
    fence_and_remote_store_i64,
    remote_store_i64_relaxed,
    sync_threads,
    wait_ge,
    wait_ge_volatile,
)


@triton.jit
def triton_nvl_sendrecv_bidir_ws_kernel(  # noqa: C901
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
    PIPELINE_DEPTH: tl.constexpr,
    MAX_TILES_PER_BLOCK_PER_SLOT: tl.constexpr,
    NUM_SEND_BLOCKS: tl.constexpr,
    NUM_SENDER_WARPS: tl.constexpr,
    NUM_RECEIVER_WARPS: tl.constexpr,
):
    TILE_NUMEL: tl.constexpr = TILE_ROWS * TILE_COLS
    block_id = tl.program_id(0)

    with tlx.async_tasks():
        with tlx.async_task("default", num_warps=NUM_SENDER_WARPS):
            task_tid = tlx.thread_id(axis=0)
            sender_id = block_id
            tail_remote_ptr = send_tail_sig + sender_id
            head_local_ptr = send_head_sig + sender_id
            start_step = tl.load(sender_step_ptr + sender_id).to(tl.int64)
            local_rows = tl.arange(0, TILE_ROWS)
            local_cols = tl.arange(0, TILE_COLS)
            num_send_tiles = tl.cdiv(send_numel, TILE_NUMEL)
            send_tile_idx = sender_id.to(tl.int64)
            send_step = start_step

            while send_tile_idx < num_send_tiles:
                slot = (send_step - start_step) % PIPELINE_DEPTH
                slot_base = (
                    sender_id * BLOCK_STRIDE_ELEMS * PIPELINE_DEPTH
                    + slot * BLOCK_STRIDE_ELEMS
                )
                if send_step < start_step + PIPELINE_DEPTH:
                    wait_target = start_step
                else:
                    wait_target = send_step - PIPELINE_DEPTH + 1
                if task_tid == 0:
                    wait_ge_volatile(head_local_ptr, wait_target.to(tl.int64))

                # TLX-BARRIER-BUG: hangs after >2 while-loop iters; host
                # adaptive-stride math caps the iter count to <=2.
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
                    tl.store(send_staging_buf + slot_base + buf_off, data, mask=mask)
                    send_tile_idx += NUM_SEND_BLOCKS
                    buf_tile_idx += 1

                # TLX-BARRIER-BUG: see file docstring.
                sync_threads()
                fence_and_remote_store_i64(
                    tail_remote_ptr, (send_step + 1).to(tl.int64)
                )
                send_step += 1

            tl.store(sender_step_ptr + sender_id, send_step.to(tl.int64))

        with tlx.async_task(num_warps=NUM_RECEIVER_WARPS):
            task_tid = tlx.thread_id(axis=0)
            receiver_id = block_id
            tail_local_ptr = recv_tail_sig + receiver_id
            head_remote_ptr = recv_head_sig + receiver_id
            start_step = tl.load(recver_step_ptr + receiver_id).to(tl.int64)
            local_rows = tl.arange(0, TILE_ROWS)
            local_cols = tl.arange(0, TILE_COLS)
            num_recv_tiles = tl.cdiv(recv_numel, TILE_NUMEL)
            recv_tile_idx = receiver_id.to(tl.int64)
            recv_step = start_step

            while recv_tile_idx < num_recv_tiles:
                slot = (recv_step - start_step) % PIPELINE_DEPTH
                slot_base = (
                    receiver_id * BLOCK_STRIDE_ELEMS * PIPELINE_DEPTH
                    + slot * BLOCK_STRIDE_ELEMS
                )
                if task_tid == 0:
                    wait_ge(tail_local_ptr, (recv_step + 1).to(tl.int64))

                # TLX-BARRIER-BUG: see file docstring.
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
                    data = tl.load(recv_staging_buf + slot_base + buf_off, mask=mask)
                    tl.store(recv_ptr + flat_idx.to(tl.int64), data, mask=mask)
                    recv_tile_idx += NUM_SEND_BLOCKS
                    buf_tile_idx += 1

                # TLX-BARRIER-BUG: see file docstring.
                sync_threads()
                remote_store_i64_relaxed(head_remote_ptr, (recv_step + 1).to(tl.int64))
                recv_step += 1

            tl.store(recver_step_ptr + receiver_id, recv_step.to(tl.int64))
