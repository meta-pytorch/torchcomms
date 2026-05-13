# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe
"""Triton kernels backing the symmem backend.

The kernels read directly from peers' symmetric-memory buffers using the
device-side pointer table returned by ``_SymmetricMemory.buffer_ptrs_dev``
and apply reductions / element-wise operations in shared/register memory.

The host issues a ``_SymmetricMemory.barrier`` (or a signal-pad-based
sub-group barrier) around the kernel launch so that the kernel only sees
consistent peer data.
"""

from __future__ import annotations

import triton
import triton.language as tl


# Op codes passed as a ``tl.constexpr`` to specialize each kernel.
# Wrapped in tl.constexpr so they can be referenced from @triton.jit
# functions (Triton disallows access to plain Python globals).
_OP_SUM = tl.constexpr(0)
_OP_PROD = tl.constexpr(1)
_OP_MIN = tl.constexpr(2)
_OP_MAX = tl.constexpr(3)
_OP_AVG = tl.constexpr(4)
_OP_PREMUL_SUM = tl.constexpr(5)
_OP_BAND = tl.constexpr(6)
_OP_BOR = tl.constexpr(7)
_OP_BXOR = tl.constexpr(8)


@triton.jit
def _reduce_step(acc, val, OP: tl.constexpr):
    """Combine ``acc`` with ``val`` according to ``OP``."""
    if OP == _OP_SUM:
        return acc + val
    elif OP == _OP_PROD:
        return acc * val
    elif OP == _OP_MIN:
        return tl.minimum(acc, val)
    elif OP == _OP_MAX:
        return tl.maximum(acc, val)
    elif OP == _OP_AVG:
        return acc + val
    elif OP == _OP_PREMUL_SUM:
        return acc + val  # caller multiplies each input by factor before sum
    elif OP == _OP_BAND:
        return acc & val
    elif OP == _OP_BOR:
        return acc | val
    elif OP == _OP_BXOR:
        return acc ^ val
    return acc


@triton.jit
def _all_reduce_kernel(
    # Per-peer pointers laid out as world_size contiguous uint64s, each one
    # the base of that peer's symmetric-memory buffer. Passed as a *int64
    # pointer (the underlying storage is a CUDA array of pointers).
    buffer_ptrs_dev,
    # Output pointer (may alias one of the peer buffers).
    out_ptr,
    # Element offset (in DTYPE units) added to each peer pointer.
    elem_offset,
    n_elements,
    world_size: tl.constexpr,
    OP: tl.constexpr,
    # Factor for PREMUL_SUM (ignored for other ops).
    scale,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    offs = offsets + elem_offset

    # Initialise accumulator with peer 0's contribution; this avoids needing
    # an explicit identity element (e.g. -inf for MAX, +inf for MIN, 1 for
    # PROD) which would also need correct typing per dtype.
    peer0_raw = tl.load(buffer_ptrs_dev + 0)
    peer0_ptr = tl.cast(peer0_raw, tl.pointer_type(DTYPE))
    acc = tl.load(peer0_ptr + offs, mask=mask, other=0)
    if OP == _OP_PREMUL_SUM:
        acc = (acc.to(tl.float32) * scale).to(DTYPE)

    for r in range(1, world_size):
        peer_raw = tl.load(buffer_ptrs_dev + r)
        peer_ptr = tl.cast(peer_raw, tl.pointer_type(DTYPE))
        val = tl.load(peer_ptr + offs, mask=mask, other=0)
        if OP == _OP_PREMUL_SUM:
            val = (val.to(tl.float32) * scale).to(DTYPE)
        acc = _reduce_step(acc, val, OP)

    if OP == _OP_AVG:
        if acc.dtype.is_floating():
            acc = acc / world_size
        else:
            acc = acc // world_size

    tl.store(out_ptr + offsets, acc, mask=mask)


def all_reduce(
    ptr_tensor, out_buf, elem_offset, n_elements, world_size, op_code, scale
):
    """Launch the all-reduce kernel.

    Args:
        ptr_tensor (torch.Tensor): 1-D int64 CUDA tensor of per-peer base
            pointers (length must be >= world_size).
        out_buf (torch.Tensor): 1-D contiguous output tensor.
        elem_offset (int): element index inside each peer buffer where the
            reduction window begins. The peer buffers are interpreted with
            the same dtype as ``out_buf``.
        n_elements (int): number of elements to reduce.
        world_size (int): number of peers.
        op_code (int): one of the ``_OP_*`` constants.
        scale (float): factor for AVG / PREMUL_SUM (ignored otherwise).
    """
    import torch

    assert out_buf.is_contiguous()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _dtype_map = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float64: tl.float64,
        torch.int32: tl.int32,
        torch.int64: tl.int64,
        torch.int8: tl.int8,
        torch.uint8: tl.uint8,
    }
    tl_dtype = _dtype_map.get(out_buf.dtype)
    if tl_dtype is None:
        raise RuntimeError(f"unsupported dtype for triton all-reduce: {out_buf.dtype}")
    _all_reduce_kernel[grid](
        ptr_tensor,
        out_buf,
        elem_offset,
        n_elements,
        world_size=world_size,
        OP=op_code,
        scale=float(scale),
        BLOCK_SIZE=BLOCK_SIZE,
        DTYPE=tl_dtype,
    )


# Op-code mapping used by the host-side dispatcher (plain ints).
OP_CODES = {
    "sum": 0,
    "product": 1,
    "min": 2,
    "max": 3,
    "avg": 4,
    "premul_sum": 5,
    "band": 6,
    "bor": 7,
    "bxor": 8,
}
