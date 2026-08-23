# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Minimal demo + example hooks (Triton), all in the ``Ctx`` form.

Hooks take a single ``Ctx`` (see ``triton/ctx.py``); ``consume`` also receives the
loaded tile payload ``regs``. ``copy_*`` are the trivial identity hooks; the
``scale2``/``addone`` pair are non-trivial examples that prove the seam composes.
Real Ops (transpose, gather, quantize, accumulate, …) supply their own hooks
against this same ``ctx`` form — adding a ``Ctx`` field never changes a signature.
"""

import triton
import triton.language as tl


@triton.jit
def copy_produce(ctx):
    """Load a tile from the input (no transform)."""
    return tl.load(ctx.ptr + ctx.idx, mask=ctx.mask)


@triton.jit
def copy_consume(ctx, regs):
    """Write a received tile to the output (overwrite)."""
    tl.store(ctx.ptr + ctx.idx, regs, mask=ctx.mask)


@triton.jit
def scale2_produce(ctx):
    """Send-side hook: load a tile and multiply by 2 (a real value transform)."""
    return tl.load(ctx.ptr + ctx.idx, mask=ctx.mask) * 2.0


@triton.jit
def addone_consume(ctx, regs):
    """Recv-side hook: add 1 to the received tile before storing."""
    tl.store(ctx.ptr + ctx.idx, regs + 1.0, mask=ctx.mask)
