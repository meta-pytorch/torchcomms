# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# pyre-ignore-all-errors[35]: pyre mis-flags this frozen-dataclass-subclass's fields as
# illegal annotation targets (the identical Triton A2AConfig/A2AKey twin is accepted); the
# dataclasses are correct and runtime-validated.

"""Tunable Config + lookup Key for the CuTe all_to_all schedule.

The CuTe twin of ``triton/a2a/tuning.py``: same autotuner-shaped surface
(see ``CuteCommTuner`` / ``comm_tuning``) -- a frozen ``CuteA2AConfig`` of launch
tunables, a ``CuteA2AKey`` rebuilt identically offline (tuning) and at runtime
(lookup), a safe ``DEFAULT_A2A_CONFIG``, and ``get_a2a_config(key)`` that
reads the tuner-emitted map and falls back to the default. The kernel takes
``config=None`` and looks it up; a tuning adapter passes an explicit config.

Tier note: only LAUNCH tunables live here. A field left at its sentinel (``0``)
means "use the analytic adaptive pick" (``_pick_tile`` / ``_pick_slots`` /
``_pick_cluster`` in ``a2a.schedules``), so the default config reproduces the
analytic defaults exactly; a tuned table overrides per (size, ws, device).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from ...tuning_base import (
    _device_tag,
    BaseTunableConfig as _BaseConfig,
    BaseTuningKey as _BaseKey,
    geometry_signature as _geometry_signature,
    nondefault_inapplicable_fields as _nondefault_inapplicable_fields,
    resolve_variant as _resolve_variant_impl,
)
from ..hooks import copy_consume, copy_produce


@dataclass(frozen=True)
class CuteA2AConfig(_BaseConfig):
    """Launch tunables swept by the autotuner (perf axis only).

    ``num_blocks`` is the block count per peer (device grid = ``world_size *
    num_blocks``; one CTA per ``(peer, block)`` streams its sub-chunk). The remaining
    knobs default to ``0`` = "use the analytic adaptive pick", so the default config
    reproduces the size-aware analytic defaults exactly:

    * ``num_threads`` -- threads/CTA (``0`` -> ``_pick_tile``); the per-thread vector
      width ``vec`` is always the widest the chunk allows and is not independently
      tuned.
    * ``num_slots`` -- send/drain pipeline slots (``0`` -> ``_pick_slots``);
      ``tiles_per_slot`` is derived from it.
    * ``unroll`` -- register-blocking unroll of the NVLink store loop (``0`` ->
      size-aware default, 8 once the per-peer chunk is large enough else 1).
    * ``cluster`` -- CGA thread-block cluster size along the block axis (``0`` ->
      ``_pick_cluster``; ``-1`` = max = ``num_blocks``; ``>0`` = explicit).

    ``primitive`` selects the transfer schedule (one enum, the CuTe twin of the
    Triton ``primitive``):

    * ``"copy"`` (default) -- slot-pipelined per-thread staging copy.
    * ``"tma"`` -- TMA-drain bounce variant (the TMA engine drains staging while warps
      issue the NVLink send); launched via :func:`all_to_all` with ``primitive="tma"``.
    * ``"direct"`` -- drain-free DirectWrite into the peer's symm-mem output;
      launched via :func:`all_to_all_zc`, not :func:`all_to_all`.
    * ``"ce"`` -- copy-engine zero-copy (``cuMemcpyAsync`` + a 1-CTA signal kernel);
      launched via :func:`all_to_all_zc` with ``primitive="ce"``.
    """

    num_blocks: int = 8
    num_threads: int = 0
    num_slots: int = 0
    unroll: int = 0
    primitive: str = "copy"
    cluster: int = 0
    cluster_y: int = 1
    tma_drain_warps: int = 1


# Fields whose value changes the physical staging geometry (grid partition / output
# semantics) rather than launch-only packing; switching one on a reused transport is
# the documented hazard the runtime geometry guard catches. The tile/slot/unroll/
# cluster knobs are launch-only and free to sweep. Mirrors A2A_GEOMETRY_FIELDS.
CUTE_A2A_GEOMETRY_FIELDS: frozenset[str] = frozenset({"num_blocks", "primitive"})

# Core tunable field names, for the adapter's candidate-grid validation. Derived from the
# dataclass so it can never drift from CuteA2AConfig; mirrors A2A_CORE_FIELDS.
CUTE_A2A_CORE_FIELDS: frozenset[str] = frozenset(f.name for f in fields(CuteA2AConfig))

# Launch knobs each primitive actually consumes at launch (the apply-path audit, made
# structural so the candidate grid can never emit a field that does nothing for its
# primitive -- the WS5 honesty invariant). A field NOT listed for a primitive must be left
# at its dataclass default when that primitive is swept, so a tuned entry never carries a
# silently-ignored value. ``num_blocks``/``primitive`` are always meaningful and excluded
# from the non-default check (``num_blocks`` shapes the grid for every path; ``primitive`` is
# the selector). ``copy`` is the staging schedule (all knobs apply); ``direct`` is the
# drain-free single-shot DirectWrite (no slot pipeline, so ``num_slots`` does NOT apply);
# ``ce`` is the host copy-engine path (no device grid -> no launch knobs).
CUTE_PRIMITIVE_APPLIES: dict[str, frozenset[str]] = {
    "copy": frozenset(
        {
            "num_threads",
            "num_slots",
            "unroll",
            "cluster",
            "cluster_y",
            "tma_drain_warps",
        }
    ),
    "tma": frozenset(
        {
            "num_threads",
            "num_slots",
            "unroll",
            "cluster",
            "cluster_y",
            "tma_drain_warps",
        }
    ),
    "direct": frozenset({"num_threads", "unroll", "cluster", "cluster_y"}),
    "ce": frozenset(),
}


def geometry_signature(config: "CuteA2AConfig") -> tuple:
    """Geometry-defining field values for the CuTe a2a config (the
    ``CUTE_A2A_GEOMETRY_FIELDS`` subset; see ``tuning_base.geometry_signature``)."""
    return _geometry_signature(config, CUTE_A2A_GEOMETRY_FIELDS)


def _resolve_variant(produce, consume, variant: str = "") -> str:
    """CuTe a2a hook discriminator: the default identity copy hooks map to ``""``; a
    non-default hook needs an explicit ``variant`` (see ``tuning_base.resolve_variant``).
    """
    return _resolve_variant_impl(
        produce,
        consume,
        variant,
        default_produce=copy_produce,
        default_consume=copy_consume,
    )


@dataclass(frozen=True)
class CuteA2AKey(_BaseKey):
    """Runtime lookup key; must be rebuilt identically offline and at runtime."""

    # Inherits world_size, dtype, numel, rows, transport_kind, device, backend, variant
    # from BaseTuningKey. The agnostic base stays backend-neutral; the CuTe default
    # lives here on the subclass (the twin of A2AKey.backend = "triton").
    backend: str = "cute"


DEFAULT_A2A_CONFIG = CuteA2AConfig()


def nondefault_inapplicable_fields(config: "CuteA2AConfig") -> set[str]:
    """Names of fields this config sets to a NON-default value that its ``primitive`` does
    not consume at launch (empty == honest). The candidate grid's feasibility filter uses
    this so a swept config can never carry a silently-ignored knob for its primitive."""
    return _nondefault_inapplicable_fields(
        config,
        core_fields=CUTE_A2A_CORE_FIELDS,
        primitive_applies=CUTE_PRIMITIVE_APPLIES,
        default=DEFAULT_A2A_CONFIG,
    )


# Emitted by the tuner; absent until a tuning run has produced it.
try:
    # pyre-ignore[21]: generated by the tuner; absent until a tuning run exists.
    from comms.dsl.cute.a2a.generated.a2a_tuned_configs import (  # noqa: F401
        TUNED_A2A_CONFIGS,
    )
except ImportError:
    TUNED_A2A_CONFIGS = {}


def make_a2a_key(
    input,
    transport,
    *,
    rows: int = 0,
    produce=copy_produce,
    consume=copy_consume,
    variant: str = "",
    backend: str = "cute",
) -> CuteA2AKey:
    """Build the lookup key from runtime args (same fields the tuner keys on).

    The hooks + ``variant`` resolve the tuned-table hook discriminator through the single
    ``_resolve_variant`` chokepoint, so the runtime lookup and the offline tuner derive an
    identical key for the same logical inputs + hook (no dual-key drift)."""
    return CuteA2AKey(
        world_size=transport.world_size,
        dtype=str(input.dtype).removeprefix("torch."),
        numel=int(input.numel()),
        rows=rows,
        transport_kind=type(transport).__name__,
        device=_device_tag(input.device),
        backend=backend,
        variant=_resolve_variant(produce, consume, variant),
    )


def get_a2a_config(key: CuteA2AKey) -> CuteA2AConfig:
    """Tuned config for this key, falling back to the safe default."""
    return TUNED_A2A_CONFIGS.get(key, DEFAULT_A2A_CONFIG)
