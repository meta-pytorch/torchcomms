# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Shared-memory layout factory for the TMA-drain a2a variant.

This module intentionally does NOT use ``from __future__ import annotations``:
``@cute.struct`` reads the REAL type objects off a class's annotations to build the
smem layout, and PEP 563 stringized annotations (which ``collectives.py`` enables)
turn ``cute.struct.MemRange[...]`` into a plain string that ``@cute.struct`` rejects
("Struct element only support struct/array/base_dsl scalar"). Keeping the struct
definition here, annotation-eager, sidesteps that.
"""

import cutlass
import cutlass.cute as cute


def make_tma_smem(dtype, tile_elems: int, stages: int):
    """Build the per-CTA shared-memory struct for the TMA-drain bounce: ``stages``
    mbarriers (two int64 slots each -- full + empty) plus a ``tile_elems * stages``
    element bounce buffer, 128B-aligned for TMA."""

    @cute.struct
    class _TmaSmem:
        mbar: cute.struct.MemRange[cutlass.Int64, stages * 2]
        buf: cute.struct.Align[
            cute.struct.MemRange[dtype, tile_elems * stages],
            128,
        ]

    return _TmaSmem
