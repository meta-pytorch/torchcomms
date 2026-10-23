# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""CuTe hooks -- two tiers (see ``ctx.py`` for the ``Ctx`` / ``BlockCtx`` contracts).

* **Per-element / value** (``produce(ctx)`` / ``consume(ctx, frag)``): the default identity
  ``copy_*`` hooks and example value-transforms (``scale2`` / ``addone``, register-only math on
  ``ctx.part``). ``produce`` loads the input tile into a register fragment and returns it;
  ``consume`` writes a fragment to the output tile. This tier is per-thread-per-tile.
* **Block-tile / layout** (``layout_hook(bctx)``): a CTA-cooperative 2D SMEM-tile transform for
  coalescing-critical layout changes. ``transpose_tile`` is the flagship (coalesced-store the
  SMEM tile transposed); permute/reshape reuse the same path. The framework does the coalesced
  SMEM load + barriers (the shared substrate leaf ``send_recv._block_tile_u``, the block-tile twin
  of the value leaf ``_send_u``); the hook does the in-SMEM transform. The CuTe twin of genai's
  ``tl.trans``. The a2a transpose schedules compose this leaf in the next layer.
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


def scale2_produce(ctx):
    """Send-side: load the input tile, multiply by 2, return the fragment."""
    frag = cute.make_fragment_like(ctx.part)
    cute.copy(ctx.atom, ctx.part, frag)
    frag.store(frag.load() * 2.0)
    return frag


def addone_consume(ctx, frag):
    """Recv-side: add 1 to the received fragment, store to the output tile."""
    frag.store(frag.load() + 1.0)
    cute.copy(ctx.atom, frag, ctx.part)


def transpose_tile(bctx):
    """Block-tile (layout) hook: coalesced-store the SMEM tile TRANSPOSED into ``dst``.

    The flagship block-tile hook and the CuTe twin of the on-chip ``tl.trans`` implementation. The framework
    has already coalesced-loaded the ``[tile, tile]`` input tile into the padded SMEM ``bctx.sA``
    and barriered; this hook reads ``sA`` with swapped indices (bank-conflict-free thanks to the
    +1 pad) and coalesced-stores it into ``dst`` at the transposed position. Both gmem legs stay
    coalesced (``tx`` = lane is the contiguous dim on load AND store) -- the whole point of the
    block-tile tier vs the per-element gather. A permute/reshape hook reuses the same path with a
    different index mapping. ``all_to_all(..., rows=R)`` selects this hook by default."""
    tile = bctx.tile
    for r in range(0, tile, bctx.block_rows):
        bctx.dst[(bctx.bc * tile + bctx.ty + r, bctx.br * tile + bctx.tx)] = bctx.sA[
            (bctx.tx, bctx.ty + r)
        ]
