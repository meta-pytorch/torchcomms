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

    ``num_blocks`` sets the device grid to ``world_size * num_blocks`` CTAs. The classic
    schedule interprets it as the block count per peer; channel schedules reinterpret the
    same strictly measured grid as logical channels. The remaining knobs default to ``0``
    = "use the analytic adaptive pick", so the default config reproduces the size-aware
    analytic defaults exactly:

    * ``num_threads`` -- threads/CTA (``0`` -> ``_pick_tile``); the per-thread vector
      width is always the widest the chunk allows and is not independently tuned.
    * ``num_slots`` -- classic send/drain pipeline slots (``0`` -> ``_pick_slots``), or
      bounded-FIFO slots for ``copy_channel_ring``. The full-staging channel schedules
      reject a nonzero value.
    * ``unroll`` -- register-blocking unroll of the NVLink store loop (``0`` ->
      size-aware default, 8 once the per-peer chunk is large enough else 1).
    * ``cluster`` -- CGA thread-block cluster size along the block axis (``0`` ->
      size-aware default; ``-1`` = max = ``num_blocks``; ``>0`` = explicit) for the
      classic schedule. Channel schedules resolve ``0`` and ``1`` to a non-clustered
      launch and reject other values.
    * ``cluster_y`` -- CGA cluster size along the peer (grid-y) axis; ``1`` = off
      (the default). Collapsed to 1 unless it divides ``world_size``.
    * ``send_threads`` -- threads in the send warp group for warp-specialized schedules
      (``0`` -> half of ``num_threads``); both groups must contain whole warps. The
      full-staging schedule requires an even 512/512 direction split when
      ``peer_fanout`` is greater than 1.
    * ``peer_fanout`` -- concurrent remote-peer groups per direction for
      ``copy_channel_full`` (``0`` -> 1); supported values are 1, 2, and 4. Values above
      1 use the fixed 8-rank, 32-CTA, 1024-thread launch. Other primitives reject it.

    ``primitive`` selects the transfer schedule. ``"copy"`` is the slot-pipelined
    per-peer staging copy; ``"copy_channel_full"`` is the full-staging peer-packed
    warp-specialized schedule and ``"copy_channel_ring"`` is its bounded-FIFO counterpart.
    No other primitives are supported by this config.
    """

    num_blocks: int = 8
    num_threads: int = 0
    num_slots: int = 0
    unroll: int = 0
    primitive: str = "copy"
    cluster: int = 0
    cluster_y: int = 1
    send_threads: int = 0
    peer_fanout: int = 0


# Config fields that always change physical staging ownership. The host augments these with
# schedule-specific resolved fields (shape, dtype/vector width, and slot layout) before
# checking a reused transport.
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
