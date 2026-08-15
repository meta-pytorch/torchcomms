# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# pyre-ignore-all-errors[6, 29, 35, 58]: @cute.kernel / @cute.jit constexpr params are
# annotated cutlass.Constexpr; pyre models that as Constexpr[Any] and rejects the
# arithmetic / range() / dataclass-field uses the kernel bodies do on them, and the calls
# to the produce/consume hook constexprs ([29]) -- the values are real compile-time ints /
# callables at trace time. The identical Triton/tuning twins are accepted under the same
# file-level idiom.

"""Device kernels for the fused all_to_all in the CuTe DSL (symmetric-memory NVLink).

The on-device half of the CuTe all_to_all: the free ``@cute.kernel`` schedules, the
``@cute.jit`` launchers (``_launch_a2a`` / ``_launch_a2a_tma``), the shared copy
atoms, the CGA-cluster sizing helper ``_pick_cluster`` (the tile/slot sizing
``_pick_tile`` / ``_pick_slots`` are shared from ``cute/send_recv.py``), the
host-side launch-constant bundle ``_A2ACfg``, and the copy-engine signal kernel
``_CeSignalKernel``. The public host entries that look up a config and
``cute.compile``/launch these kernels live in :mod:`comms.dsl.cute.a2a.host`.

Each ``(peer, block)`` program streams its sub-chunk to the peer's staging over
NVLink, publishes a data-ready signal, then drains the peer's matching staging slot
into the output. Device-side peer addressing uses ``cute.make_ptr`` over the
transport's int64 peer table (the same symmetric-memory base pointers the Triton
kernel casts), so a single fused launch selects the peer on device via ``block_idx``
instead of one launch per peer.

Graph-safe: signalling uses the transport's persistent monotonic per-(peer, block)
step counters + a TAIL (data-ready) / HEAD (slot-free) signal pair, exactly like the
Triton path, so a transport is reusable across calls / CUDA-graph replays. The base
kernel is single-shot per chunk (stage whole sub-chunk, then drain); the slot
pipeline overlaps send/drain on top. The zero-copy ``direct`` path additionally
launches with a CGA thread-block cluster (co-locating every block that targets one
peer on a single GPC) -- on the pure-send path this raises the per-SM NVLink
injection rate and lets direct beat NCCL at the large band (see ``_pick_cluster``).

Scope: equal-split ``all_to_all_single``, identity copy (bf16/fp32), ``numel``
divisible by ``world_size`` and the per-(peer,block) tile by the thread count.
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
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.utils as cutlass_utils

from .. import nvl_ops
from ..tma_smem import make_tma_smem


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
    direct: bool = False
    send_only: bool = False
    cluster: int = 1
    cluster_y: int = 1
    tma: bool = False
    tma_stages: int = 4
    tma_drain_warps: int = 1


# Shared CuTe send/recv substrate (the copy leaves + the per-slot credit-ring
# primitives) lives in ``cute/send_recv.py`` -- the backend substrate home, symmetric
# with the Triton ``triton/send_recv.py`` holding ``send_step``/``recv_step``. a2a and
# the standalone send/recv collective both compose the SAME ``_send_slot``/``_recv_slot``
# (the C2 symmetric-substrate contract).
from ..send_recv import (  # noqa: E402
    _block_tile_u,
    _copy_atom,
    _copy_u,
    _local_u,
    _recv_slot,
    _send_slot,
)


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
    send_only: cutlass.Constexpr,
    direct: cutlass.Constexpr,
    cluster: cutlass.Constexpr,
    cluster_y: cutlass.Constexpr,
    produce: cutlass.Constexpr,
    consume: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    stream,
) -> None:
    # in2d / out2d are [world_size, chunk] views; per-peer chunk = row peer.
    # The `direct` flag is constexpr, so the dispatch test is wrapped in
    # cutlass.const_expr -- this folds the branch at trace time (only the selected
    # schedule is traced). A bare `if flag:` on a constexpr would emit a traced
    # scf.if and walk BOTH arms, instantiating the unselected kernel (e.g. the
    # zero-copy `direct` kernel, whose symm-mem output addressing differs from the
    # staging path -> a mistraced launch).
    # CGA cluster dims (None == off); host guarantees the grid is divisible.
    cl = (
        [cluster, cluster_y, 1]
        if cutlass.const_expr(cluster > 1 or cluster_y > 1)
        else None
    )
    if cutlass.const_expr(direct):
        # zero-copy: writes the peer's symm-mem output buffer; out2d unused.
        _a2a_kernel_direct(
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
            local_rank,
            chunk,
            cap_elems,
            mbp,
            unroll,
        ).launch(
            grid=(num_blocks, world_size, 1),
            block=(num_threads, 1, 1),
            cluster=cl,
            stream=stream,
        )
    else:
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
            send_only,
            produce,
            consume,
            rows,
        ).launch(
            grid=(num_blocks, world_size, 1),
            block=(num_threads, 1, 1),
            cluster=cl,
            stream=stream,
        )


@cute.jit
def _launch_a2a_tma(
    in2d,
    out2d,
    stg2d,
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
    tma_stages: cutlass.Constexpr,
    tma_drain_warps: cutlass.Constexpr,
    cluster: cutlass.Constexpr,
    cluster_y: cutlass.Constexpr,
    stream,
) -> None:
    # Separate @cute.jit entry for the TMA-drain variant. It is selected at the
    # HOST level (a real Python branch in ``all_to_all``) rather than via a
    # ``if tma`` inside ``_launch_a2a`` -- CuTe traces BOTH arms of an in-kernel
    # ``if`` on a host bool, so a dead TMA arm would still touch the TMA-only
    # state and crash the non-TMA compile. stg2d is my local staging buffer
    # as [ws, chunk] (row = sender): the descriptor source for the drain bounce.
    tile_elems = num_threads * vec
    total_bufs = tma_drain_warps * tma_stages
    tma_smem = make_tma_smem(dtype, tile_elems, total_bufs)
    tma_bytes = (dbits // 8) * tile_elems
    cl = (
        [cluster, cluster_y, 1]
        if cutlass.const_expr(cluster > 1 or cluster_y > 1)
        else None
    )
    # Build the TMA descriptors host-side (cutlass cpasync API): one bulk-tensor-
    # tile G2S load over my staging [ws, chunk] and one S2G store over the output
    # [ws, chunk], tiled (1, tile_elems) so a (peer, tile) coord selects each tile.
    smem_layout = cute.make_layout((1, tile_elems))
    cta_tiler = (1, tile_elems)
    load_atom, load_tns = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), stg2d, smem_layout, cta_tiler
    )
    store_atom, store_tns = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), out2d, smem_layout, cta_tiler
    )
    _a2a_kernel_tma(
        in2d,
        out2d,
        buf_ptrs,
        sig_ptrs,
        send_ctr,
        recv_ctr,
        load_atom,
        load_tns,
        store_atom,
        store_tns,
        dtype,
        vec,
        dbits,
        num_blocks,
        num_threads,
        local_rank,
        chunk,
        cap_elems,
        mbp,
        num_slots,
        tiles_per_slot,
        unroll,
        tma_stages,
        tma_drain_warps,
        tma_smem,
        tile_elems,
        tma_bytes,
    ).launch(
        grid=(num_blocks, world_size, 1),
        block=(num_threads + tma_drain_warps * 32, 1, 1),
        cluster=cl,
        smem=tma_smem.size_in_bytes(),
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
    send_only: cutlass.Constexpr,
    produce: cutlass.Constexpr,
    consume: cutlass.Constexpr,
    rows: cutlass.Constexpr,
) -> None:
    tidx = cute.arch.thread_idx()[0]
    b = cute.arch.block_idx()[0]
    peer = cute.arch.block_idx()[1]
    u = unroll
    nb = num_blocks

    # Vectorized copy: each thread moves VEC contiguous elements per copy, so
    # the NVLink store is a vectorized st.global (up to 128-bit) instead of a
    # scalar 16-bit store -- the single biggest copy-bandwidth lever (mirrors
    # the Triton 128-bit-store fix). (num_threads, VEC) are chosen per chunk by
    # the host (`_pick_tile`) so any size tiles exactly -- VEC drops to keep
    # small/odd chunks correct (down to scalar), covering the whole 32B-2GB
    # ladder with no tail. The tiler is num_threads*VEC elems/tile.
    copy_atom = _copy_atom(dtype, vec, dbits)
    tiled_copy = cute.make_tiled_copy_tv(
        copy_atom, cute.make_layout(num_threads), cute.make_layout(vec)
    )
    thr_copy = tiled_copy.get_slice(tidx)

    # This peer's contiguous input/output chunk (row `peer` of the [ws, chunk] views). Layout
    # changes (rows>0 transpose) are NOT done here -- they run on the block-tile `transpose_tile`
    # hook over a SMEM-staged 2D tile (see `host.all_to_all_transpose`); this schedule handles
    # the contiguous copy + the per-element value hooks (produce/consume).
    in_chunk = in2d[(peer, None)]
    raw_chunk = in_chunk  # source handed to a gather hook via Ctx.src
    out_chunk = out2d[(peer, None)]
    tiler = cute.make_layout(num_threads * vec)
    g_in = cute.zipped_divide(in_chunk, tiler)
    g_out = cute.zipped_divide(out_chunk, tiler)
    # Tile count from the actual divided layout (the authoritative value): the host
    # also derives it for _pick_slots, but recomputing here keeps the kernel's tile
    # bound tied to its own zipped_divide and never out of step with a host value.
    num_tiles = cute.size(g_in, mode=[1])

    # CuTe forbids early `return` in a kernel, so the diagonal (local) and the
    # comm path are an if/else (peer is a runtime block index, not constexpr).
    if peer == local_rank:
        # Diagonal: local copy in_chunk -> out_chunk (no comm). Grid-stride
        # `while` stays inline (CuTe captures control flow only here); the
        # unrolled body is the straight-line `_copy_u`.
        t = b
        while t + (u - 1) * nb < num_tiles:
            _local_u(
                thr_copy,
                copy_atom,
                g_in,
                g_out,
                t,
                u,
                num_blocks,
                produce,
                consume,
                src=raw_chunk,
                tiler=tiler,
                rows=rows,
                chunk=chunk,
                peer=peer,
            )
            t += u * nb
        while t < num_tiles:
            _local_u(
                thr_copy,
                copy_atom,
                g_in,
                g_out,
                t,
                1,
                num_blocks,
                produce,
                consume,
                src=raw_chunk,
                tiler=tiler,
                rows=rows,
                chunk=chunk,
                peer=peer,
            )
            t += nb
    else:
        # --- device-side symm-mem addressing for this peer ---
        peer_buf_addr = buf_ptrs[peer]
        my_buf_addr = buf_ptrs[local_rank]
        peer_sig_addr = sig_ptrs[peer]
        my_sig_addr = sig_ptrs[local_rank]

        # Staging regions (elem layout [chunk]); sender `s` occupies the byte
        # slot `s * cap_elems * elem_bytes` in a rank's buffer (Triton-path
        # convention). SEND -> my sender slot inside peer's buffer; RECV ->
        # peer's sender slot inside my buffer (byte addresses, int64).
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
        # Tail (data-ready) + Head (slot-free): head = tail + ws*mbp*8 (transport 2x pad)
        tail_remote = peer_sig_addr + (local_rank * mbp + b) * 8
        tail_local = my_sig_addr + (peer * mbp + b) * 8
        head_local = my_sig_addr + world_size * mbp * 8 + (peer * mbp + b) * 8
        head_remote = peer_sig_addr + world_size * mbp * 8 + (local_rank * mbp + b) * 8

        # Slot pipeline, composed from the shared per-slot primitives: send slot s
        # (NVLink store + TAIL) then drain slot s-1 (local HBM) so the still-in-flight
        # stores of slot s overlap the drain. Disjoint regions (my-staging drain vs
        # peer-staging store) -> no false dependency. num_slots is constexpr so this
        # loop unrolls at trace time. _send_slot / _recv_slot are the CuTe twins of the
        # Triton send_step / recv_step: a2a and the standalone send/recv compose the
        # SAME primitives (the symmetric-substrate contract).
        for s in range(num_slots):
            _send_slot(
                thr_copy,
                copy_atom,
                g_in,
                g_send,
                tail_remote,
                head_local,
                start_send,
                b,
                tidx,
                s,
                tiles_per_slot,
                num_tiles,
                num_blocks,
                num_slots,
                unroll,
                produce,
                src=raw_chunk,
                tiler=tiler,
                rows=rows,
                chunk=chunk,
                peer=peer,
            )
            if not send_only:
                if s >= 1:
                    _recv_slot(
                        thr_copy,
                        copy_atom,
                        g_recv,
                        g_out,
                        tail_local,
                        head_remote,
                        start_recv,
                        b,
                        tidx,
                        s - 1,
                        tiles_per_slot,
                        num_tiles,
                        num_blocks,
                        unroll,
                        consume,
                        peer=peer,
                    )
        # Drain the final slot (no later send to overlap it).
        if not send_only:
            _recv_slot(
                thr_copy,
                copy_atom,
                g_recv,
                g_out,
                tail_local,
                head_remote,
                start_recv,
                b,
                tidx,
                num_slots - 1,
                tiles_per_slot,
                num_tiles,
                num_blocks,
                unroll,
                consume,
                peer=peer,
            )
        if tidx == 0:
            send_ctr[step_idx] = start_send + num_slots
            recv_ctr[step_idx] = start_recv + num_slots


@cute.kernel
def _a2a_kernel_tma(  # noqa: C901
    in2d,
    out2d,
    buf_ptrs,
    sig_ptrs,
    send_ctr,
    recv_ctr,
    load_atom,
    load_tns,
    store_atom,
    store_tns,
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
    tma_stages: cutlass.Constexpr,
    tma_drain_warps: cutlass.Constexpr,
    tma_smem: cutlass.Constexpr,
    tma_tile_elems: cutlass.Constexpr,
    tma_bytes: cutlass.Constexpr,
) -> None:
    # TMA-drain: warps 1..N SEND (LSU NVLink stores, num_threads wide), warp 0
    # DRAINS by bouncing my-staging -> smem -> out through the TMA/bulk-copy
    # engine. The two legs touch disjoint memory and are coupled only by the
    # cross-GPU signal pad, so they need NO intra-CTA barrier: warp 0 drains
    # slot s (after the peer signals it) while the send warps fill slot s+1.
    tidx = cute.arch.thread_idx()[0]
    b = cute.arch.block_idx()[0]
    peer = cute.arch.block_idx()[1]
    u = unroll
    nb = num_blocks
    nt = num_threads
    d_warps = tma_drain_warps
    d_threads = d_warps * 32
    warp = tidx // 32

    copy_atom = _copy_atom(dtype, vec, dbits)
    tiled_copy = cute.make_tiled_copy_tv(
        copy_atom, cute.make_layout(nt), cute.make_layout(vec)
    )
    in_chunk = in2d[(peer, None)]
    out_chunk = out2d[(peer, None)]
    tiler = cute.make_layout(nt * vec)
    g_in = cute.zipped_divide(in_chunk, tiler)
    num_tiles = cute.size(g_in, mode=[1])

    if peer == local_rank:
        # Diagonal: the send warps copy the local chunk straight to out (no comm,
        # no drain); the drain warps are idle.
        if warp >= d_warps:
            g_out = cute.zipped_divide(out_chunk, tiler)
            thr = tiled_copy.get_slice(tidx - d_threads)
            t = b
            while t + (u - 1) * nb < num_tiles:
                _copy_u(thr, copy_atom, g_in, g_out, t, u, num_blocks)
                t += u * nb
            while t < num_tiles:
                _copy_u(thr, copy_atom, g_in, g_out, t, 1, num_blocks)
                t += nb
    else:
        peer_buf_addr = buf_ptrs[peer]
        peer_sig_addr = sig_ptrs[peer]
        my_sig_addr = sig_ptrs[local_rank]
        elem_bytes = dbits // 8
        cap_bytes = cap_elems * elem_bytes
        send_ptr = cute.make_ptr(
            dtype,
            peer_buf_addr + local_rank * cap_bytes,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        g_send = cute.zipped_divide(
            cute.make_tensor(send_ptr, cute.make_layout(chunk)), tiler
        )
        step_idx = peer * mbp + b
        start_send = send_ctr[step_idx]
        start_recv = recv_ctr[step_idx]
        tail_remote = peer_sig_addr + (local_rank * mbp + b) * 8
        tail_local = my_sig_addr + (peer * mbp + b) * 8
        head_local = my_sig_addr + world_size * mbp * 8 + (peer * mbp + b) * 8
        head_remote = peer_sig_addr + world_size * mbp * 8 + (local_rank * mbp + b) * 8

        if warp >= d_warps:
            # SEND warps: wait free-credit, stage each slot to peer, then publish TAIL.
            # Uses credit protocol so faster rank cannot overwrite slot peer hasn't drained.
            thr = tiled_copy.get_slice(tidx - d_threads)
            for s in range(num_slots):
                # Wait free-credit before overwriting staging
                if tidx == d_threads:
                    nvl_ops.wait_free(head_local, start_send + s + 1 - num_slots)
                cute.arch.barrier(barrier_id=1, number_of_threads=nt)
                s_lo = s * tiles_per_slot
                s_hi = min((s + 1) * tiles_per_slot, num_tiles)
                t = s_lo + b
                while t + (u - 1) * nb < s_hi:
                    _copy_u(thr, copy_atom, g_in, g_send, t, u, num_blocks)
                    t += u * nb
                while t < s_hi:
                    _copy_u(thr, copy_atom, g_in, g_send, t, 1, num_blocks)
                    t += nb
                cute.arch.barrier(barrier_id=1, number_of_threads=nt)
                if tidx == d_threads:
                    nvl_ops.signal(tail_remote, start_send + s + 1)
            if tidx == d_threads:
                send_ctr[step_idx] = start_send + num_slots
        else:
            # DRAIN warps (warps 0..D-1): wait the peer's slot, then TMA-bounce its
            # tiles my-staging[peer, t] -> smem -> out[peer, t] on the TMA engine,
            # concurrently with the send warps' LSU stores. Each drain warp d owns a
            # STRIDED tile subset (every D-th tile, offset d) + its own smem stages
            # + its own mbarriers, so D warps issue TMA in parallel -- scaling the
            # drain past the single-warp throughput ceiling. Within a warp the
            # subset is drained in groups of ``tma_stages`` bounces in flight.
            d = warp
            stg = tma_stages
            smem = cutlass_utils.SmemAllocator()
            storage = smem.allocate(tma_smem)
            mbar_ptr = storage.mbar.data_ptr()
            staged = storage.buf.get_tensor(
                cute.make_layout((1, tma_tile_elems, d_warps * stg))
            )
            load_tiled = cute.zipped_divide(load_tns, (1, tma_tile_elems))
            store_tiled = cute.zipped_divide(store_tns, (1, tma_tile_elems))
            wstride = d_warps * nb  # tile stride between this warp's tiles
            for s in range(num_slots):
                if tidx == 0:
                    nvl_ops.wait(tail_local, start_recv + s + 1)
                    for j in range(d_warps * stg):
                        cute.arch.mbarrier_init(mbar_ptr + j, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.barrier(barrier_id=2, number_of_threads=d_threads)
                s_hi = min((s + 1) * tiles_per_slot, num_tiles)
                base = s * tiles_per_slot + b + d * nb
                gphase = 0
                while base < s_hi:
                    for j in range(stg):
                        tj = base + j * wstride
                        if tj < s_hi:
                            buf = d * stg + j
                            smt = cute.group_modes(
                                cute.slice_(staged, (None, None, buf)), 0, 2
                            )
                            g_l = cute.group_modes(
                                load_tiled[(None, None), (peer, tj)], 0, 2
                            )
                            sp, gp = cpasync.tma_partition(
                                load_atom, 0, cute.make_layout(1), smt, g_l
                            )
                            if tidx == d * 32:
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    mbar_ptr + buf, tma_bytes
                                )
                            cute.copy(load_atom, gp, sp, tma_bar_ptr=mbar_ptr + buf)
                    for j in range(stg):
                        if base + j * wstride < s_hi:
                            cute.arch.mbarrier_wait(mbar_ptr + d * stg + j, gphase)
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    for j in range(stg):
                        tj = base + j * wstride
                        if tj < s_hi:
                            buf = d * stg + j
                            smt = cute.group_modes(
                                cute.slice_(staged, (None, None, buf)), 0, 2
                            )
                            g_s = cute.group_modes(
                                store_tiled[(None, None), (peer, tj)], 0, 2
                            )
                            sps, gps = cpasync.tma_partition(
                                store_atom, 0, cute.make_layout(1), smt, g_s
                            )
                            cute.copy(store_atom, sps, gps)
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)
                    gphase = 1 - gphase
                    base += stg * wstride
                # Publish free-credit HEAD so sender may reuse slot (after all TMA loads done)
                cute.arch.barrier(barrier_id=2, number_of_threads=d_threads)
                if tidx == 0:
                    nvl_ops.signal_free(head_remote, start_recv + s + 1)
                # Slot-end sync: every drain warp must finish slot s's bounces
                # and HEAD publish before thread 0 re-arms mbarriers for next slot
                cute.arch.barrier(barrier_id=2, number_of_threads=d_threads)
            if tidx == 0:
                recv_ctr[step_idx] = start_recv + num_slots


@cute.kernel
def _a2a_kernel_direct(  # noqa: C901
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
    unroll: cutlass.Constexpr,
) -> None:
    # Zero-copy DirectWrite: each (peer, block) writes this rank's chunk for
    # `peer` STRAIGHT into peer's symmetric-memory OUTPUT buffer at this rank's
    # slot (no staging, no drain), publishes data-ready, and waits for `peer`'s
    # write into MY output slot. After all blocks, this rank's symm-mem buffer is
    # the complete a2a output (slot s = chunk from sender s) -- the caller reads
    # it via handle.get_buffer (out2d is unused). The theoretical-minimum work:
    # one NVLink store/elem + one fence; its busbw is the per-SM send-rate
    # ceiling (== the SEND-ONLY diagnostic), the upper bound every staging
    # variant is bounded by (see a2a_cute_perf_and_exhaustion.md).
    tidx = cute.arch.thread_idx()[0]
    b = cute.arch.block_idx()[0]
    peer = cute.arch.block_idx()[1]
    u = unroll
    nb = num_blocks
    copy_atom = _copy_atom(dtype, vec, dbits)
    thr_copy = cute.make_tiled_copy_tv(
        copy_atom, cute.make_layout(num_threads), cute.make_layout(vec)
    ).get_slice(tidx)
    tiler = cute.make_layout(num_threads * vec)
    g_in = cute.zipped_divide(in2d[(peer, None)], tiler)
    num_tiles = cute.size(g_in, mode=[1])
    elem_bytes = dbits // 8
    cap_bytes = cap_elems * elem_bytes
    # Destination = peer's output buffer at THIS rank's slot (peer==self -> my
    # own buffer, the local diagonal). Same slot convention as staging.
    dst_ptr = cute.make_ptr(
        dtype,
        buf_ptrs[peer] + local_rank * cap_bytes,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    g_dst = cute.zipped_divide(
        cute.make_tensor(dst_ptr, cute.make_layout(chunk)), tiler
    )
    t = b
    while t + (u - 1) * nb < num_tiles:
        _copy_u(thr_copy, copy_atom, g_in, g_dst, t, u, num_blocks)
        t += u * nb
    while t < num_tiles:
        _copy_u(thr_copy, copy_atom, g_in, g_dst, t, 1, num_blocks)
        t += nb
    if peer != local_rank:
        cute.arch.barrier()
        step_idx = peer * mbp + b
        start_send = send_ctr[step_idx]
        start_recv = recv_ctr[step_idx]
        if tidx == 0:
            tail_remote = sig_ptrs[peer] + (local_rank * mbp + b) * 8
            nvl_ops.signal(tail_remote, start_send + 1)
            tail_local = sig_ptrs[local_rank] + (peer * mbp + b) * 8
            nvl_ops.wait(tail_local, start_recv + 1)
            send_ctr[step_idx] = start_send + 1
            recv_ctr[step_idx] = start_recv + 1


def _make_transpose_smem(dtype, tile: int, depth: int = 1):
    """Per-CTA SMEM for the block-cooperative transpose: ``depth`` stacked ``(tile, tile+1)``
    tiles (the +1 pad makes the transposed read bank-conflict-free). ``depth>1`` is the store-
    unroll: the kernel loads ``depth`` tiles, barriers once, then stores all ``depth`` (so
    ``depth * tile/block_rows`` NVLink stores are in flight before the next barrier)."""

    @cute.struct
    class _TrSmem:
        buf: cute.struct.MemRange[dtype, depth * tile * (tile + 1)]

    return _TrSmem


@cute.kernel
def _a2a_kernel_transpose_direct(  # noqa: C901
    in2d,
    out2d,
    buf_ptrs,
    sig_ptrs,
    send_ctr,
    recv_ctr,
    dtype: cutlass.Constexpr,
    dbits: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    cap_elems: cutlass.Constexpr,
    mbp: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    cols: cutlass.Constexpr,
    tile: cutlass.Constexpr,
    block_rows: cutlass.Constexpr,
    pipeline_depth: cutlass.Constexpr,
    smem_struct: cutlass.Constexpr,
    layout_hook: cutlass.Constexpr,
) -> None:
    # Zero-copy DirectWrite + FUSED block-cooperative SMEM transform (block-tile HOOK). Each
    # (peer, block)
    # transposes this rank's [rows, cols] chunk-for-peer into [cols, rows] and writes it
    # STRAIGHT into peer's symm-mem OUTPUT buffer at this rank's slot -- one kernel, no
    # staging, no scratch. The transpose is done in SMEM (coalesced load -> padded smem ->
    # coalesced transposed store), so BOTH the local HBM read and the NVLink store are
    # coalesced (unlike the per-element gather hook's uncoalesced vec=1). This is the CuTe
    # twin of the reference sender-side tl.trans, but zero-copy instead of staged.
    tidx = cute.arch.thread_idx()[0]
    b = cute.arch.block_idx()[0]
    peer = cute.arch.block_idx()[1]
    tx = (
        tidx % tile
    )  # lane -> contiguous gmem dim on BOTH legs (coalesced load AND store)
    ty = tidx // tile
    elem_bytes = dbits // 8
    cap_bytes = cap_elems * elem_bytes
    # src = my input chunk for peer, as [rows, cols]; dst = peer's output buffer at my slot,
    # as [cols, rows] (the transposed output layout).
    src = cute.make_tensor(
        in2d[(peer, None)].iterator, cute.make_layout((rows, cols), stride=(cols, 1))
    )
    dst_ptr = cute.make_ptr(
        dtype,
        buf_ptrs[peer] + local_rank * cap_bytes,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    dst = cute.make_tensor(dst_ptr, cute.make_layout((cols, rows), stride=(rows, 1)))
    smem = cutlass_utils.SmemAllocator()
    storage = smem.allocate(smem_struct)
    tlayout = cute.make_layout((tile, tile + 1), stride=(tile + 1, 1))
    ntiles_r = rows // tile
    ntiles_c = cols // tile
    total = ntiles_r * ntiles_c
    if cutlass.const_expr(pipeline_depth > 1):
        # Store-unroll via a ROTATING SMEM buffer (depth-``pipeline_depth`` software pipeline):
        # each tile lands in buffer ``(seq % depth)``, so consecutive tiles use different buffers
        # and the post-store WAR barrier is dropped (a buffer is reused ``depth`` tiles later,
        # covered by the intervening load barriers). One barrier per tile (not two) AND each
        # tile's NVLink store overlaps the next tile's coalesced HBM load -> more stores in
        # flight, closing the mid-band stall. The buffer index is a runtime slice (same as the
        # per-peer ``in2d`` slice), so there is no Python-state toggle across the traced loop.
        sAbuf = storage.buf.get_tensor(
            cute.make_layout(
                (pipeline_depth, tile, tile + 1),
                stride=(tile * (tile + 1), tile + 1, 1),
            )
        )
        gt = b
        while gt < total:
            br = gt // ntiles_c
            bc = gt % ntiles_c
            bidx = (gt // num_blocks) % pipeline_depth
            # Shared block-tile leaf; rotating buffer -> war_barrier=False (WAR covered by the
            # intervening RAW barriers of the next `depth` tiles).
            _block_tile_u(
                sAbuf[(bidx, None, None)],
                src,
                dst,
                br,
                bc,
                tile,
                block_rows,
                tx,
                ty,
                layout_hook,
                war_barrier=False,
            )
            gt += num_blocks
        cute.arch.barrier()  # last tiles' stores done reading SMEM before the signal
    else:
        sA = storage.buf.get_tensor(tlayout)
        gt = b
        while gt < total:
            br = gt // ntiles_c
            bc = gt % ntiles_c
            _block_tile_u(sA, src, dst, br, bc, tile, block_rows, tx, ty, layout_hook)
            gt += num_blocks
    if peer != local_rank:
        cute.arch.barrier()
        step_idx = peer * mbp + b
        start_send = send_ctr[step_idx]
        start_recv = recv_ctr[step_idx]
        if tidx == 0:
            tail_remote = sig_ptrs[peer] + (local_rank * mbp + b) * 8
            nvl_ops.signal(tail_remote, start_send + 1)
            tail_local = sig_ptrs[local_rank] + (peer * mbp + b) * 8
            nvl_ops.wait(tail_local, start_recv + 1)
            send_ctr[step_idx] = start_send + 1
            recv_ctr[step_idx] = start_recv + 1


@cute.jit
def _launch_a2a_transpose(
    in2d,
    out2d,
    buf_ptrs,
    sig_ptrs,
    send_ctr,
    recv_ctr,
    dtype: cutlass.Constexpr,
    dbits: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
    local_rank: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    cap_elems: cutlass.Constexpr,
    mbp: cutlass.Constexpr,
    world_size: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    cols: cutlass.Constexpr,
    tile: cutlass.Constexpr,
    block_rows: cutlass.Constexpr,
    pipeline_depth: cutlass.Constexpr,
    layout_hook: cutlass.Constexpr,
    stream,
) -> None:
    smem_struct = _make_transpose_smem(dtype, tile, pipeline_depth)
    _a2a_kernel_transpose_direct(
        in2d,
        out2d,
        buf_ptrs,
        sig_ptrs,
        send_ctr,
        recv_ctr,
        dtype,
        dbits,
        num_blocks,
        local_rank,
        chunk,
        cap_elems,
        mbp,
        rows,
        cols,
        tile,
        block_rows,
        pipeline_depth,
        smem_struct,
        layout_hook,
    ).launch(
        grid=(num_blocks, world_size, 1),
        block=(tile * block_rows, 1, 1),
        stream=stream,
        smem=smem_struct.size_in_bytes(),
    )


# CGA cluster co-locates all CTAs targeting one peer on a single GPC, so their
# NVLink stores inject through the shared GPC->NVLink path with less contention --
# this RAISES the per-SM send-rate ceiling on the pure-send (direct) path. Measured
# 8xH100 (apple-to-apple, num_blocks=4): direct large-band 0.74-0.94x -> 1.01-1.15x
# (beats NCCL at 64MB-1GB). Only at the few-CTA / large-chunk band: with many CTAs
# (mid sizes) the GPU is already saturated and a wide cluster over-constrains
# placement (regresses). Staging gains little (it is drain/sync-bound, not send-
# bound), so this is a direct-path default; an explicit A2A_CUTE_CLUSTER still wins.
_MIN_CLUSTER_CHUNK_BYTES: int = 1024 * 1024  # 1 MiB per-peer chunk
_MAX_PORTABLE_CLUSTER: int = 8  # Hopper/Blackwell portable cluster cap


def _pick_cluster(num_blocks: int, chunk_bytes: int) -> int:
    """Default CGA cluster size along the block axis for the direct path: cluster
    every block targeting one peer (``num_blocks``) when in the few-CTA / large-
    chunk band, else 1 (off). Capped at the portable cluster size; an explicit
    ``A2A_CUTE_CLUSTER`` env overrides (``<=0`` is the "max" = num_blocks sentinel).
    """
    env = os.environ.get("A2A_CUTE_CLUSTER")
    if env is not None:
        v = int(env)
        # Cap the env-derived cluster at the portable cap (matches the staging path), so
        # A2A_CUTE_CLUSTER=0 with num_blocks>cap can't request an invalid cluster size on
        # Hopper (CUDA_ERROR_INVALID_CLUSTER_SIZE).
        c = min(num_blocks if v <= 0 else v, _MAX_PORTABLE_CLUSTER)
        return c if num_blocks % c == 0 else 1
    if num_blocks <= _MAX_PORTABLE_CLUSTER and chunk_bytes >= _MIN_CLUSTER_CHUNK_BYTES:
        return num_blocks
    return 1


class _CeSignalKernel:
    """1-CTA signal/recv companion for the copy-engine (CE) all_to_all.

    The data movement is done host-side by ``cuMemcpyAsync`` on the copy engines
    (zero SM); this tiny kernel does only the cross-rank completion handshake on
    the symmetric-memory signal pad, stream-ordered AFTER the memcpys. Thread
    ``peer`` (one per peer, peer != self) publishes a data-ready signal into peer's
    pad and waits for peer's signal into mine, using the transport's persistent
    monotonic counters (graph-safe; no reset needed)."""

    def __init__(self, *, world_size: int, local_rank: int, mbp: int) -> None:
        self.world_size = world_size
        self.local_rank = local_rank
        self.mbp = mbp

    @cute.jit
    def __call__(self, sig_ptrs, send_ctr, recv_ctr, stream) -> None:
        # One thread per peer (kernel indexes peer = thread_idx). Size the block (warp-
        # rounded) to world_size so peers >=32 (e.g. GB200 NVL72) still get a thread; the
        # `peer < world_size` guard makes the rounding-up extras no-ops.
        block_threads = max(32, (self.world_size + 31) // 32 * 32)
        self.kernel(sig_ptrs, send_ctr, recv_ctr).launch(
            grid=(1, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, sig_ptrs, send_ctr, recv_ctr) -> None:
        peer = cute.arch.thread_idx()[0]
        if peer < self.world_size:
            if peer != self.local_rank:
                step_idx = peer * self.mbp
                start_send = send_ctr[step_idx]
                start_recv = recv_ctr[step_idx]
                # Publish "my chunk for `peer` has landed in peer's buffer" into
                # peer's pad slot for me; wait for peer's matching signal into mine.
                tail_remote = sig_ptrs[peer] + (self.local_rank * self.mbp) * 8
                nvl_ops.signal(tail_remote, start_send + 1)
                tail_local = sig_ptrs[self.local_rank] + (peer * self.mbp) * 8
                nvl_ops.wait(tail_local, start_recv + 1)
                send_ctr[step_idx] = start_send + 1
                recv_ctr[step_idx] = start_recv + 1
