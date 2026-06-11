# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""CuTe hooks in the unified ``produce(ctx)`` / ``consume(ctx, frag)`` form.

The CuTe twin of ``triton/hooks.py``: the default identity ``copy_*`` hooks plus
example transforms. Each hook owns its leg — ``produce`` loads the input tile
into a register fragment and returns it; ``consume`` writes a fragment to the
output tile. The elementwise transform is register-only CuTe math.
"""

from __future__ import annotations

import cutlass.cute as cute


def copy_produce(ctx):
    """produce: load the input tile into a register fragment (no transform)."""
    frag = cute.make_fragment_like(ctx.part)
    cute.copy(ctx.atom, ctx.part, frag)
    return frag


def copy_consume(ctx, frag):
    """consume: store a received fragment to the output tile (overwrite)."""
    cute.copy(ctx.atom, frag, ctx.part)
