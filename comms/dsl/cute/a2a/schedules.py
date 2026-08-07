# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# pyre-ignore-all-errors[6, 35, 58]: @cute.kernel / @cute.jit constexpr params are
# annotated cutlass.Constexpr; pyre models that as Constexpr[Any] and rejects the
# arithmetic / range() / dataclass-field uses the kernel bodies do on them -- the values
# are real compile-time ints at trace time.

"""Device kernels for the fused all_to_all in the CuTe DSL (symmetric-memory NVLink).

The on-device half of the CuTe all_to_all: the fused ``@cute.kernel`` copy schedule
(``_a2a_kernel``), its ``@cute.jit`` launcher (``_launch_a2a``), and the host-side
launch-constant bundle ``_A2ACfg``. The tile/slot sizing (``_pick_tile`` /
``_pick_slots``) and the copy leaf / slot primitives are shared from
``cute/send_recv.py``. The public host entry that resolves a config and
``cute.compile`` / launches this kernel lives in :mod:`comms.dsl.cute.a2a.host`.

Each ``(peer, block)`` program streams its sub-chunk to the peer's staging over NVLink,
publishes a data-ready signal, then drains the peer's matching staging slot into the
output. Device-side peer addressing uses ``cute.make_ptr`` over the transport's int64
peer table, so a single fused launch selects the peer on device via ``block_idx``
instead of one launch per peer.

Graph-safe: signalling uses the transport's persistent monotonic per-(peer, block) step
counters, so a transport is reusable across calls / CUDA-graph replays (with a cross-rank
sync between reused calls; see :func:`comms.dsl.cute.a2a.host.all_to_all`). The slot
pipeline overlaps send/drain on top of a single-shot stage-then-drain.

Scope: equal-split ``all_to_all_single``, identity copy (bf16/fp32), ``numel`` divisible
by ``world_size`` and the per-(peer, block) tile by the thread count.
"""

# Annotations are evaluated eagerly (no ``from __future__ import annotations``): the
# @cute.kernel / @cute.jit launchers classify their compile-time params via the real
# ``cutlass.Constexpr`` annotation object, which inspect.getfullargspec only exposes when
# annotations are NOT stringized -- stringized annotations leave cute unable to tell a
# constexpr from a dynamic arg and it mis-marshals the dtype.
import os
from dataclasses import dataclass
from typing import Any

# Importing send_recv runs its one-time setup (CUTE_DSL_ARCH detection +
# cuda-bindings shim) AND imports cutlass; the os.environ statement below is a
# barrier so the formatter cannot hoist the cutlass imports above this setup.
from ..send_recv import _resolve_cute_dsl_arch

os.environ.setdefault("CUTE_DSL_ARCH", _resolve_cute_dsl_arch())

import cutlass
import cutlass.cute as cute


@dataclass
class _A2ACfg:
    """Host-side bundle of the fused all_to_all launch constants, for readability at the
    call site. It is NEVER passed as a single ``cutlass.Constexpr`` -- cute's tree-walk
    would call ``__extract_mlir_values__`` on the ``dtype`` (a cutlass NumericMeta) and
    crash on the tiny/odd-chunk paths. Always UNPACK it into individual positional args at
    the ``cute.compile`` / ``.launch`` boundary."""

    num_blocks: int
    num_threads: int
    vec: int
    dtype: Any
    dbits: int
    world_size: int
    local_rank: int
    chunk: int
    cap_elems: int
    mbp: int
    num_slots: int
    tiles_per_slot: int
    unroll: int = 1
    cluster: int = 1
    cluster_y: int = 1


from .. import nvl_ops  # noqa: E402

# The shared CuTe send/recv substrate (copy leaves, per-slot credit primitives, and
# tile/slot sizing) lives in ``cute/send_recv.py``. This schedule composes those helpers
# and owns the collective-specific peer mapping, barriers, and counter progression.
from ..send_recv import _copy_atom, _copy_u, _recv_slot, _send_slot  # noqa: E402


@cute.jit
def _launch_a2a(
    in2d,
    out2d,
    buf_ptrs,
    sig_ptrs,
    send_ctr,
    recv_ctr,
    num_blocks: cutlass.Constexpr,
    num_threads: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    dbits: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    cap_elems: cutlass.Constexpr,
    mbp: cutlass.Constexpr,
    num_slots: cutlass.Constexpr,
    tiles_per_slot: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
    cluster: cutlass.Constexpr,
    cluster_y: cutlass.Constexpr,
    stream,
) -> None:
    # in2d / out2d are [world_size, chunk] views; per-peer chunk = row peer.
    # CGA cluster dims (None == off); host guarantees the grid is divisible.
    cl = (
        [cluster, cluster_y, 1]
        if cutlass.const_expr(cluster > 1 or cluster_y > 1)
        else None
    )
    _a2a_kernel(
        in2d,
        out2d,
        buf_ptrs,
        sig_ptrs,
        send_ctr,
        recv_ctr,
        dtype,
        vec,
        dbits,
        num_blocks,
        num_threads,
        world_size,
        local_rank,
        chunk,
        cap_elems,
        mbp,
        num_slots,
        tiles_per_slot,
        unroll,
    ).launch(
        grid=(num_blocks, world_size, 1),
        block=(num_threads, 1, 1),
        cluster=cl,
        stream=stream,
    )


@cute.kernel
def _a2a_kernel(  # noqa: C901
    in2d,
    out2d,
    buf_ptrs,
    sig_ptrs,
    send_ctr,
    recv_ctr,
    dtype: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    dbits: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
    num_threads: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    cap_elems: cutlass.Constexpr,
    mbp: cutlass.Constexpr,
    num_slots: cutlass.Constexpr,
    tiles_per_slot: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
) -> None:
    tidx = cute.arch.thread_idx()[0]
    b = cute.arch.block_idx()[0]
    peer = cute.arch.block_idx()[1]
    u = unroll
    nb = num_blocks

    # Vectorized copy: each thread moves VEC contiguous elements per copy, so the NVLink
    # store is a vectorized st.global (up to 128-bit) instead of a scalar store -- the
    # single biggest copy-bandwidth lever. (num_threads, VEC) are chosen per chunk by the
    # host (`_pick_tile`) so any size tiles exactly (VEC drops to scalar for small/odd
    # chunks, covering the whole 32B-2GB ladder with no tail). Tiler = num_threads*VEC.
    copy_atom = _copy_atom(dtype, vec, dbits)
    tiled_copy = cute.make_tiled_copy_tv(
        copy_atom, cute.make_layout(num_threads), cute.make_layout(vec)
    )
    thr_copy = tiled_copy.get_slice(tidx)

    # This peer's contiguous input/output chunk (row `peer` of the [ws, chunk] views).
    in_chunk = in2d[(peer, None)]
    out_chunk = out2d[(peer, None)]
    tiler = cute.make_layout(num_threads * vec)
    g_in = cute.zipped_divide(in_chunk, tiler)
    g_out = cute.zipped_divide(out_chunk, tiler)
    # Tile count from the actual divided layout (the authoritative value): the host also
    # derives it for _pick_slots, but recomputing here keeps the kernel's tile bound tied
    # to its own zipped_divide and never out of step with a host value.
    num_tiles = cute.size(g_in, mode=[1])

    # CuTe forbids early `return` in a kernel, so the diagonal (local) and the comm path
    # are an if/else (peer is a runtime block index, not constexpr).
    if peer == local_rank:
        # Diagonal: local copy in_chunk -> out_chunk (no comm). Grid-stride `while` stays
        # inline (CuTe captures control flow only here); the unrolled body is `_copy_u`.
        t = b
        while t + (u - 1) * nb < num_tiles:
            _copy_u(thr_copy, copy_atom, g_in, g_out, t, u, num_blocks)
            t += u * nb
        while t < num_tiles:
            _copy_u(thr_copy, copy_atom, g_in, g_out, t, 1, num_blocks)
            t += nb
    else:
        # --- device-side symm-mem addressing for this peer ---
        peer_buf_addr = buf_ptrs[peer]
        my_buf_addr = buf_ptrs[local_rank]
        peer_sig_addr = sig_ptrs[peer]
        my_sig_addr = sig_ptrs[local_rank]

        # Staging regions (elem layout [chunk]); sender `s` occupies the byte slot
        # `s * cap_elems * elem_bytes` in a rank's buffer. SEND -> my sender slot inside
        # peer's buffer; RECV -> peer's sender slot inside my buffer (int64 byte addrs).
        elem_bytes = dbits // 8
        cap_bytes = cap_elems * elem_bytes
        send_addr = peer_buf_addr + local_rank * cap_bytes
        recv_addr = my_buf_addr + peer * cap_bytes
        send_ptr = cute.make_ptr(
            dtype, send_addr, cute.AddressSpace.gmem, assumed_align=16
        )
        recv_ptr = cute.make_ptr(
            dtype, recv_addr, cute.AddressSpace.gmem, assumed_align=16
        )
        send_region = cute.make_tensor(send_ptr, cute.make_layout(chunk))
        recv_region = cute.make_tensor(recv_ptr, cute.make_layout(chunk))
        g_send = cute.zipped_divide(send_region, tiler)
        g_recv = cute.zipped_divide(recv_region, tiler)

        step_idx = peer * mbp + b
        start_send = send_ctr[step_idx]
        start_recv = recv_ctr[step_idx]
        # This (peer, block)'s own signal counters: block b sends ITS tile subset to the
        # peer and signals its own counter; the peer's block b waits the matching counter
        # and drains the SAME subset -- so the pipeline is independent across blocks.
        tail_remote = peer_sig_addr + (local_rank * mbp + b) * 8
        tail_local = my_sig_addr + (peer * mbp + b) * 8
        head_off = world_size * mbp
        head_remote = peer_sig_addr + (head_off + local_rank * mbp + b) * 8
        head_local = my_sig_addr + (head_off + peer * mbp + b) * 8

        if tidx == 0:
            nvl_ops.wait_free(head_local, start_send)
        cute.arch.barrier()

        # Slot pipeline, composed from the shared per-slot primitives: send slot s (NVLink
        # store + TAIL) then drain slot s-1 (local HBM) so the still-in-flight stores of
        # slot s overlap the drain. Disjoint regions (my-staging drain vs peer-staging
        # store) -> no false dependency. num_slots is constexpr so this loop unrolls.
        for s in range(num_slots):
            _send_slot(
                thr_copy,
                copy_atom,
                g_in,
                g_send,
                tail_remote,
                start_send,
                b,
                tidx,
                s,
                tiles_per_slot,
                num_tiles,
                num_blocks,
                unroll,
            )
            if s >= 1:
                _recv_slot(
                    thr_copy,
                    copy_atom,
                    g_recv,
                    g_out,
                    tail_local,
                    start_recv,
                    b,
                    tidx,
                    s - 1,
                    tiles_per_slot,
                    num_tiles,
                    num_blocks,
                    unroll,
                )
        # Drain the final slot (no later send to overlap it).
        _recv_slot(
            thr_copy,
            copy_atom,
            g_recv,
            g_out,
            tail_local,
            start_recv,
            b,
            tidx,
            num_slots - 1,
            tiles_per_slot,
            num_tiles,
            num_blocks,
            unroll,
        )
        cute.arch.barrier()
        if tidx == 0:
            nvl_ops.signal_free(head_remote, start_recv + num_slots)
            send_ctr[step_idx] = start_send + num_slots
            recv_ctr[step_idx] = start_recv + num_slots
