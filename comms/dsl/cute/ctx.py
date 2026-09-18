# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""CuTe hook-contract views. Two tiers, matching what the transform needs:

* :class:`Ctx` -- the **per-element / value** tier. ``part`` is this tile's partitioned tensor
  (input for ``produce``, output for ``consume``), ``atom`` the gmem<->rmem copy atom; a value
  hook (scale / quantize / accumulate) loads ``part``, transforms in registers, returns the
  fragment. Plus stable read-only FACTS -- ``coord`` (tile index), ``peer`` (dest/source rank),
  ``rank`` / ``world_size``, and shape ``rows`` / ``cols`` / ``chunk`` -- for position/peer/
  rank-dependent value transforms (masking, positional encoding, per-peer scale) without touching
  any machinery. This tier is per-thread-per-tile: it CANNOT do a block-cooperative transform.

* :class:`BlockCtx` -- the **block-tile / layout** tier. A coalescing-critical layout change
  (transpose, permute, block-reshape) needs the whole CTA to cooperate on a 2D SMEM tile, which
  the per-element tier cannot express. The framework does the coalescing 95% -- coalesced-load a
  ``[tile, tile]`` input tile into padded SMEM (``sA``) + ``barrier`` -- and the block hook does
  the 5%: the in-SMEM transform / transposed store into ``dst``. See ``hooks.transpose_tile``.
  The framework side is the shared substrate leaf ``send_recv._block_tile_u`` (the block-tile twin
  of the value leaf ``_send_u``), which the a2a transpose schedules compose in the next layer.

The CuTe twin of ``triton/ctx.py``. Kept importable GPU-free (no module-scope cutlass); both
classes are plain field holders, constructed by the schedule inside the kernel (GPU present).
"""

from __future__ import annotations

from typing import Any


class Ctx:
    """Per-element (value) hook view: ``part`` + ``atom`` + read-only facts."""

    __slots__ = (
        "part",
        "atom",
        "coord",
        "peer",
        "rank",
        "world_size",
        "rows",
        "cols",
        "chunk",
    )

    def __init__(
        self,
        part: Any,
        atom: Any,
        coord: Any = None,
        peer: Any = None,
        rank: Any = None,
        world_size: Any = None,
        rows: Any = 0,
        chunk: Any = 0,
    ) -> None:
        self.part = part
        self.atom = atom
        # Position / identity / shape facts -- stable read-only scalars a value hook may use.
        self.coord = coord
        self.peer = peer
        self.rank = rank
        self.world_size = world_size
        self.rows = rows
        self.chunk = chunk
        self.cols = (chunk // rows) if rows else chunk


class BlockCtx:
    """Block-tile (layout) hook view: a CTA-cooperative 2D SMEM tile transform.

    ``sA`` is the coalesced-loaded, padded ``[tile, tile+1]`` SMEM input tile (framework-loaded,
    post-barrier). ``dst`` is this peer's destination tensor (``[cols, rows]`` for a transpose),
    into which the hook coalesced-stores the transformed tile. ``(br, bc)`` is the 2D tile coord
    (units of ``tile``); ``(tx, ty)`` this thread's lane/row within the tile; ``block_rows`` the
    row-stride each thread streams. A hook loops ``r in range(0, tile, block_rows)`` and writes
    ``dst`` from ``sA`` under its own mapping (transpose = swapped indices)."""

    __slots__ = ("sA", "dst", "tile", "block_rows", "tx", "ty", "br", "bc")

    def __init__(
        self,
        sA: Any,
        dst: Any,
        tile: Any,
        block_rows: Any,
        tx: Any,
        ty: Any,
        br: Any,
        bc: Any,
    ) -> None:
        self.sA = sA
        self.dst = dst
        self.tile = tile
        self.block_rows = block_rows
        self.tx = tx
        self.ty = ty
        self.br = br
        self.bc = bc
