# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Minimal demo hooks: a plain identity copy through staging.

The trivial ``produce``/``consume`` used by the minimal launcher and smoke test.
Hooks take a single ``Ctx`` (see ``triton/ctx.py``); ``consume`` also receives the
loaded tile payload ``regs``. Real Ops (transpose, gather, quantize, accumulate,
…) supply their own hooks against this same ``ctx`` form.
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
