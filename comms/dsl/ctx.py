# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""DSL-agnostic hook contract for the composable send/recv framework.

``Ctx`` describes the *fields* a per-tile compute hook may read. It is the stable
contract shared by every backend, realized on-device per DSL: the Triton backend
as a ``@_aggregate`` struct (``triton/ctx.py``) and the CuTe backend as a
``@dataclass`` (``cute/send_recv.py``). The field set stays identical so the same
hook is portable across backends and across schedules (a user-written schedule
today, a library schedule skeleton later).

This dataclass is the DSL-agnostic *spec* of that field set (and is used by the
GPU-free tests). The device hooks read the per-backend realization; adding a
field here (and to each realization) never changes a hook signature.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Ctx:
    """The full-leg hook contract (field set a ``produce``/``consume`` may read)."""

    # --- pointers ---
    in_ptr: int = 0
    out_ptr: int = 0
    # Base pointer for the current tile's transport-addressable data region.
    # On send paths derived from PeerEndpoint.send_dst; on recv from .recv_src.
    region_ptr: int = 0
    slot_base: int = 0
    # --- pipeline position ---
    tile_idx: int = 0
    slot: int = 0
    step: int = 0
    tile_rows: int = 0
    tile_cols: int = 0
    # --- intra-block work split ---
    block_id: int = 0
    flat_tid: int = 0
    num_blocks: int = 0
    # --- schedule state (lets one hook serve multiple schedules) ---
    peer: int = 0
    recv_peer: int = 0
    shard_idx: int = 0
    # --- layout ---
    in_strides: tuple[int, ...] = ()
    out_strides: tuple[int, ...] = ()
    # --- op-specific extension point (e.g. expert_counts, scales) ---
    extra: dict[str, object] = field(default_factory=dict)


# The two hooks. ``produce(ctx) -> regs``: HBM -> staging (reads the input leg).
# ``consume(ctx, regs)``: staging -> HBM (writes the output leg with the loaded
# tile payload ``regs``). These aliases document the contract; concrete device
# hooks are DSL kernels reading the per-backend ``Ctx`` realization.
ProduceFn = Callable[[Ctx], object]
ConsumeFn = Callable[[Ctx, object], None]
