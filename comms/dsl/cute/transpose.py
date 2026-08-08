# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# pyre-ignore-all-errors[6, 29, 35, 58]: @cute.jit/@cute.kernel constexpr params are annotated
# cutlass.Constexpr; pyre models that as Constexpr[Any]. Same idiom as schedules/send_recv.

"""Batched SMEM-staged transpose: each of ``ws`` contiguous ``[rows, cols]`` chunks -> ``[cols,
rows]``, bit-exact, at ~coalesced HBM bandwidth.

The fast realization of the a2a transpose. A naive per-element gather hook reads through a
strided ``(rows, cols):(cols, 1)`` view (uncoalesced vec=1 gather -> ~0.26x of coalesced BW);
this does the classic bank-conflict-free SMEM transpose instead -- coalesced load into a padded
``[TILE, TILE+1]`` smem tile, then coalesced store of the transposed tile -- reaching ~plain
copy bandwidth (H100 microbench: 48MB 1.01x, 128MB 0.97x of a plain contiguous copy). It shares
the block-tile ``transpose_tile`` hook with the fused schedule (see ``hooks.transpose_tile``).

Because the per-(sender,receiver)-chunk transpose COMMUTES with the equal-split all_to_all
(transpose-input-then-a2a == a2a-then-transpose-output), this powers the ORCHESTRATED transpose
variant (``host.all_to_all_transpose_orchestrated``): transpose the local input chunks with this
kernel, then a plain best-variant a2a (incl. the zero-copy direct/ce paths) -- each leg at its
own peak. The DEFAULT (``all_to_all_transpose``) is instead a single fused SMEM-staging kernel
that stores transposed straight into the peer output buffer (one launch, no scratch).
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass.cute.runtime import from_dlpack

# Reuse the backend's one-time CUTE_DSL_ARCH detection + cuda-bindings shim + dtype table.
from .hooks import transpose_tile
from .send_recv import _block_tile_u, _CUTLASS_DTYPE, _ensure_cuda_rt_compat

_TILE: int = 32
_BLOCK_ROWS: int = 8
_COMPILED: dict = {}


def _make_smem_t(dtype, tile: int):
    @cute.struct
    class _SmemT:
        buf: cute.struct.MemRange[dtype, tile * (tile + 1)]

    return _SmemT


@cute.kernel
def _transpose_kernel(
    src,  # [ws, rows, cols] contiguous
    dst,  # [ws, cols, rows] contiguous
    ntiles_r: cutlass.Constexpr,  # rows // TILE
    ntiles_c: cutlass.Constexpr,  # cols // TILE
    ws: cutlass.Constexpr,
    smem_struct: cutlass.Constexpr,
    tile: cutlass.Constexpr,
    block_rows: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
    layout_hook: cutlass.Constexpr,
) -> None:
    tidx = cute.arch.thread_idx()[0]
    bid = cute.arch.block_idx()[0]
    tx = (
        tidx % tile
    )  # lane -> contiguous gmem dim on BOTH legs (coalesced load AND store)
    ty = tidx // tile
    smem = cutlass_utils.SmemAllocator()
    storage = smem.allocate(smem_struct)
    # (tile, tile+1) padded so the transposed smem read hits distinct banks (no conflict).
    sA = storage.buf.get_tensor(
        cute.make_layout((tile, tile + 1), stride=(tile + 1, 1))
    )
    tiles_per_chunk = ntiles_r * ntiles_c
    total = ws * tiles_per_chunk
    gt = bid
    while gt < total:
        peer = gt // tiles_per_chunk
        tb = gt % tiles_per_chunk
        br = tb // ntiles_c  # tile-row into rows
        bc = tb % ntiles_c  # tile-col into cols
        # Shared block-tile leaf: coalesced-load src[peer] tile -> padded SMEM, barrier, then the
        # HOOK does the in-SMEM transform + coalesced store into dst[peer] (transpose_tile default).
        _block_tile_u(
            sA,
            src[(peer, None, None)],
            dst[(peer, None, None)],
            br,
            bc,
            tile,
            block_rows,
            tx,
            ty,
            layout_hook,
        )
        gt += num_blocks


@cute.jit
def _transpose_launch(
    src,
    dst,
    rows: cutlass.Constexpr,
    cols: cutlass.Constexpr,
    ws: cutlass.Constexpr,
    tile: cutlass.Constexpr,
    block_rows: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    num_blocks: cutlass.Constexpr,
    stream,
) -> None:
    src3d = cute.make_tensor(
        src.iterator, cute.make_layout((ws, rows, cols), stride=(rows * cols, cols, 1))
    )
    dst3d = cute.make_tensor(
        dst.iterator, cute.make_layout((ws, cols, rows), stride=(cols * rows, rows, 1))
    )
    smem_struct = _make_smem_t(dtype, tile)
    _transpose_kernel(
        src3d,
        dst3d,
        rows // tile,
        cols // tile,
        ws,
        smem_struct,
        tile,
        block_rows,
        num_blocks,
        transpose_tile,
    ).launch(
        grid=(num_blocks, 1, 1),
        block=(tile * block_rows, 1, 1),
        stream=stream,
        smem=smem_struct.size_in_bytes(),
    )


def _sms(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def transpose_chunks(
    dst: torch.Tensor, src: torch.Tensor, ws: int, rows: int, cols: int
) -> None:
    """Transpose each of ``ws`` contiguous ``[rows, cols]`` chunks of ``src`` into ``[cols, rows]``
    in ``dst`` (both 1-D length ``ws*rows*cols``), bit-exact, at ~coalesced bandwidth.

    ``rows`` and ``cols`` must be multiples of the SMEM tile (32). Graph-capture-safe (no host
    sync / alloc); the caller owns ``dst``."""
    if rows % _TILE or cols % _TILE:
        raise ValueError(
            f"transpose_chunks requires rows/cols % {_TILE} == 0, got {rows}x{cols}"
        )
    if src.dtype not in _CUTLASS_DTYPE:
        raise ValueError(
            f"transpose_chunks supports {list(_CUTLASS_DTYPE)}, got {src.dtype}"
        )
    cdtype, _ = _CUTLASS_DTYPE[src.dtype]
    # High block count: the SMEM+barrier latency needs many waves to reach peak (H100 microbench:
    # 48MB needed 4x SMs). Cap at the actual tile count so tiny chunks don't over-launch.
    total_tiles = ws * (rows // _TILE) * (cols // _TILE)
    num_blocks = min(total_tiles, 8 * _sms(src.device))
    import cuda.bindings.driver as _drv

    _ensure_cuda_rt_compat()
    src_c = from_dlpack(src, assumed_align=16)
    dst_c = from_dlpack(dst, assumed_align=16)
    stream = _drv.CUstream(torch.cuda.current_stream().cuda_stream)
    key = ("transpose", rows, cols, ws, src.dtype, num_blocks)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(
            _transpose_launch,
            src_c,
            dst_c,
            rows,
            cols,
            ws,
            _TILE,
            _BLOCK_ROWS,
            cdtype,
            num_blocks,
            stream,
        )
        _COMPILED[key] = compiled
    compiled(src_c, dst_c, stream)
