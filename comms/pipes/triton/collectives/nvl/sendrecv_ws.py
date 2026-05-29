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

Intra-task synchronization — compiler-rewritten warp-group barriers
------------------------------------------------------------------

The sender and receiver run as two separate ``tlx.async_task`` regions,
each owning a DISJOINT subset of the CTA's warps. A full-CTA barrier is
therefore the WRONG primitive inside either task: it blocks until *all*
CTA threads arrive, but the two tasks run independent while-loops, so
their arrivals only pair up by luck — they stay aligned for a couple of
iterations and then desync and hang. The "iteration-count-dependent TLX
codegen bug" originally blamed for this was actually a self-inflicted
full-CTA barrier: the kernel emitted a raw inline-asm ``bar.sync 0;``
(``utils.sync_threads``) which the TLX warp-spec lowering does NOT see
(it only rewrites compiler-visible barrier ops), so it stayed full-CTA.

The fix: use Triton's compiler-visible barrier ``tl.debug_barrier()``
(an ``nvvm.barrier0``). Inside a warp-specialized region the TLX lowering
(``ConvertWarpSpecializeToLLVM``) automatically rewrites each such
barrier into a per-warp-group NAMED barrier, sized to *that* region's
warp count (default region -> barrier 0; each partition -> barrier 2+idx).
This keeps each task synchronizing ONLY its own warps, with the compiler
choosing both the barrier id and the participant count (no hand-rolled
ids/sizes to get wrong). It fixes both the deterministic deadlock AND the
data race / high-iteration hang that the wrong barrier caused: validated
deterministically correct and hang-free across the full size sweep up to
1 GiB (~128 while-loop iterations) at the default stride.
"""

from __future__ import annotations

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx  # @manual=//triton:triton

from .utils import (
    fence_and_remote_store_i64,
    remote_store_i64_relaxed,
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
                # Every sender warp polls the HEAD credit (idempotent
                # acquire spin). ``tlx.thread_id()`` returns the CTA-GLOBAL
                # tid, not a task-local one, so an ``if tid == 0`` elect would
                # never fire for whichever task is not placed at tid 0 (it
                # silently skipped the wait and raced on the staging buffer).
                # Letting every thread poll is correct and layout-independent.
                wait_ge_volatile(head_local_ptr, wait_target.to(tl.int64))

                # Sender-task barrier: order the credit poll before all sender
                # warps fill the staging slot. tl.debug_barrier() is rewritten
                # by the TLX warp-spec lowering into a named barrier scoped to
                # this task's warps (NOT a full-CTA barrier).
                tl.debug_barrier()

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

                # Sender-task barrier: ensure every sender warp has finished
                # writing the staging slot before the TAIL credit is posted
                # (release publish). Rewritten to a task-scoped named barrier.
                tl.debug_barrier()
                fence_and_remote_store_i64(
                    tail_remote_ptr, (send_step + 1).to(tl.int64)
                )
                send_step += 1

            tl.store(sender_step_ptr + sender_id, send_step.to(tl.int64))

        with tlx.async_task(num_warps=NUM_RECEIVER_WARPS):
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
                # Every receiver warp polls the TAIL credit (idempotent
                # acquire spin) for the layout-independence reason documented
                # in the sender task above.
                wait_ge(tail_local_ptr, (recv_step + 1).to(tl.int64))

                # Receiver-task barrier: order the TAIL acquire-poll before
                # all receiver warps read the staging slot. Rewritten to a
                # task-scoped named barrier.
                tl.debug_barrier()

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

                # Receiver-task barrier: ensure every receiver warp has
                # finished draining the staging slot before the HEAD credit
                # is posted. Rewritten to a task-scoped named barrier.
                tl.debug_barrier()
                remote_store_i64_relaxed(head_remote_ptr, (recv_step + 1).to(tl.int64))
                recv_step += 1

            tl.store(recver_step_ptr + receiver_id, recv_step.to(tl.int64))
