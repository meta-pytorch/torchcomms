# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# pyre-ignore-all-errors[35]: pyre mis-flags this frozen-dataclass-subclass's fields as
# illegal annotation targets; the dataclass is correct and runtime-validated.

"""Tunable Config for the CuTe all_to_all copy schedule.

A frozen ``CuteA2AConfig`` of launch tunables + a safe ``DEFAULT_A2A_CONFIG``. A field
left at its ``0`` sentinel means "use the analytic adaptive pick" (``_pick_tile`` /
``_pick_slots`` in ``a2a.schedules`` / ``cute.send_recv``), so the default config
reproduces the size-aware analytic defaults exactly. The public host entry accepts an
explicit config or falls back to ``DEFAULT_A2A_CONFIG``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...tuning_base import BaseTunableConfig as _BaseConfig


@dataclass(frozen=True)
class CuteA2AConfig(_BaseConfig):
    """Launch tunables for the copy schedule (perf axis only).

    ``num_blocks`` is the block count per peer (device grid = ``world_size *
    num_blocks``; one CTA per ``(peer, block)`` streams its sub-chunk). The remaining
    knobs default to ``0`` = "use the analytic adaptive pick", so the default config
    reproduces the size-aware analytic defaults exactly:

    * ``num_threads`` -- threads/CTA (``0`` -> ``_pick_tile``); the per-thread vector
      width is always the widest the chunk allows and is not independently tuned.
    * ``num_slots`` -- send/drain pipeline slots (``0`` -> ``_pick_slots``);
      ``tiles_per_slot`` is derived from it.
    * ``unroll`` -- register-blocking unroll of the NVLink store loop (``0`` ->
      size-aware default, 8 once the per-peer chunk is large enough else 1).
    * ``cluster`` -- CGA thread-block cluster size along the block axis (``0`` ->
      size-aware default; ``-1`` = max = ``num_blocks``; ``>0`` = explicit).
    * ``cluster_y`` -- CGA cluster size along the peer (grid-y) axis; ``1`` = off
      (the default). Collapsed to 1 unless it divides ``world_size``.

    ``primitive`` selects the transfer schedule; this revision supports only ``"copy"``
    (the slot-pipelined per-thread staging copy).
    """

    num_blocks: int = 8
    num_threads: int = 0
    num_slots: int = 0
    unroll: int = 0
    primitive: str = "copy"
    cluster: int = 0
    cluster_y: int = 1


# Fields whose value changes the physical staging geometry (grid partition / output
# semantics) rather than launch-only packing; switching one on a reused transport is the
# documented hazard the runtime geometry guard catches. The tile/slot/unroll/cluster knobs
# are launch-only and free to sweep.
CUTE_A2A_GEOMETRY_FIELDS: frozenset[str] = frozenset({"num_blocks", "primitive"})

# Safe analytic default (every adaptive knob at its 0 sentinel).
DEFAULT_A2A_CONFIG: CuteA2AConfig = CuteA2AConfig()

# Max thread-blocks that can be co-resident per SM (Volta+ hardware cap); the true limit
# is the min of this and the thread / register / smem occupancy.
_MAX_BLOCKS_PER_SM: int = 32


def _max_coresident_ctas(sm_count: int, threads_per_sm: int, num_threads: int) -> int:
    """Optimistic upper bound on CTAs that can be simultaneously resident on the device.

    The fused a2a kernel spins on cross-CTA signals, so EVERY launched CTA must be
    co-resident or a resident CTA can wait forever on a signal from a never-scheduled one
    -- a systematic deadlock at high world_size (e.g. NVL72). This is the thread-limited
    occupancy bound (blocks/SM = ``threads_per_sm // num_threads``, capped at the hardware
    ``_MAX_BLOCKS_PER_SM``) times the SM count. It ignores register / smem pressure, so it
    is a ceiling: a grid above it definitely cannot co-reside (the host guard rejects it),
    while a grid below it may still be occupancy-limited in practice.
    """
    blocks_per_sm = min(_MAX_BLOCKS_PER_SM, threads_per_sm // max(1, num_threads))
    return sm_count * max(1, blocks_per_sm)
