# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# pyre-ignore-all-errors[6, 7, 14, 29, 35]: the adapter intentionally narrows BaseA2AAdapter's
# generic overrides to its declared config_cls/key_cls/spec_cls (the comm_tuning engine always
# passes the concrete subtype), and the declarative candidate source is `object`.

"""CuteCommTuner adapter for the comms/dsl CuTe all_to_all schedule.

The CuTe twin of ``triton/a2a/adapter.py``: binds the CuTe all_to_all into the comms-owned
tuner (``comm_tuning``) by supplying the backend-specific pieces to
:class:`~comms.dsl.a2a_base.BaseA2AAdapter` -- the ``CuteA2AConfig`` / ``CuteA2AKey`` /
``CuteA2AInputSpec`` classes and identity-copy hooks, the expert size-banded candidate grid,
the static feasibility filter, the generated-table render template, and the collective
launch. The shared adapter lifecycle (CLI, shape/key/input enumeration, NCCL baseline +
correctness, serialization) lives in the base.

Key sharing: the base ``make_key`` matches runtime ``make_a2a_key`` (a2a.tuning)
so the tuned map looks up. This adapter pins the default (copy) hook; a transformed variant
is a separate adapter instance with its own tuned map (keyed by ``variant``).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import torch
from comms.dsl.a2a_base import BaseA2AAdapter
from comms.dsl.tuning_base import BaseInputSpec

from ..hooks import copy_consume, copy_produce
from .host import all_to_all
from .tuning import (
    CUTE_A2A_CORE_FIELDS,
    CuteA2AConfig,
    CuteA2AKey,
    nondefault_inapplicable_fields,
)


@dataclass(frozen=True)
class CuteA2AInputSpec(BaseInputSpec):
    """Serializable recipe for one input case to tune (numel/dtype/rows)."""


def _default_candidate_configs(
    spec: CuteA2AInputSpec, key: CuteA2AKey
) -> list[CuteA2AConfig]:
    """The shipped expert, size-banded candidate grid (the zero-effort default).

    Size-aware launch grid over the CuTe launch knobs. ``num_blocks`` is bounded by the
    SM-budget guard, so oversized candidates fail run_candidate and are skipped. The
    analytic defaults (``num_threads``/``num_slots``/``unroll``/``cluster`` = 0) already
    approximate the optimum; the sweep confirms/locks them per size. A user can override
    with a flat ``candidates`` grid or a callable; this banded logic is the default that
    preserves the expert pruning."""
    per_rank = spec.numel * torch.tensor([], dtype=spec.dtype).element_size()
    if per_rank <= 256 * 1024:
        # Latency-bound: few CTAs, narrow tile, single shot.
        threads, slots, unrolls = [256, 512], [1], [1]
        num_blocks_grid, clusters, primitives = [1, 2], [0], ["copy"]
    elif per_rank <= 16 * 1024 * 1024:
        # SM-scaling band: scale CTAs to the SM budget; single-shot still wins.
        threads, slots, unrolls = [512, 1024], [1, 8], [1, 8]
        num_blocks_grid, clusters, primitives = [1, 2, 4, 8], [0], ["copy"]
    else:
        # Large band: deep run-ahead pipeline + register-blocked stores + the CGA cluster
        # all maximize the NVLink send/drain overlap.
        threads, slots, unrolls = [512, 1024], [8], [8]
        num_blocks_grid, clusters, primitives = [4, 8, 16], [0, -1], ["copy"]

    # itertools.product advances the rightmost (primitives) fastest, so the emitted order
    # is identical to the equivalent nested num_blocks/threads/slots/unrolls/clusters/
    # primitives loops; flattened per python.md's max-3-indent-levels.
    return [
        CuteA2AConfig(
            num_blocks=nb,
            num_threads=nt,
            num_slots=s,
            unroll=u,
            cluster=cl,
            primitive=p,
        )
        for nb, nt, s, u, cl, p in itertools.product(
            num_blocks_grid, threads, slots, unrolls, clusters, primitives
        )
    ]


def _default_feasible(config: CuteA2AConfig, spec: CuteA2AInputSpec) -> bool:
    """Cheap, shape-independent static feasibility pre-filter for user grids.
    world_size-dependent / SM-budget invalids still drop at run time. Override via
    ``feasible=``.

    Also enforces the apply-path honesty invariant: a candidate must not set a knob its
    ``primitive`` does not consume at launch (e.g. ``num_slots`` on the single-shot
    ``direct`` path), so the emitted tuned table never carries a silently-ignored field."""
    if config.num_blocks < 1:
        return False
    if nondefault_inapplicable_fields(config):
        return False
    return True


class CuteA2ATuningAdapter(BaseA2AAdapter):
    """Tune the comms/dsl CuTe all_to_all (default copy hook) against dist.all_to_all_single."""

    name = "comms_dsl_a2a_cute"

    # BaseA2AAdapter bindings: the CuTe config/key/spec classes + identity-copy hooks +
    # generated-artifact names. The shared lifecycle lives in the base.
    config_cls = CuteA2AConfig
    key_cls = CuteA2AKey
    spec_cls = CuteA2AInputSpec
    core_config_fields = CUTE_A2A_CORE_FIELDS
    backend_name = "cute"
    # staticmethod so instance access yields the bare hook function: a plain-function hook
    # would otherwise bind to the adapter instance and fail the identity check that maps the
    # default hooks to the empty variant.
    default_produce = staticmethod(copy_produce)
    default_consume = staticmethod(copy_consume)
    spec_tag = "cute_a2a_input_spec"
    generated_table = "TUNED_A2A_CONFIGS"
    generated_file = "a2a_tuned_configs.py"
    generated_import = "from comms.dsl.cute.a2a.tuning import CuteA2AConfig, CuteA2AKey"
    host_all_to_all = staticmethod(all_to_all)

    def default_candidate_configs(
        self, spec: CuteA2AInputSpec, key: CuteA2AKey
    ) -> list[CuteA2AConfig]:
        return _default_candidate_configs(spec, key)

    def default_feasible(self, config: CuteA2AConfig, spec: CuteA2AInputSpec) -> bool:
        return _default_feasible(config, spec)

    def render_config(self, config_type: str, config: dict[str, Any]) -> str:
        base = (
            "CuteA2AConfig(num_blocks={num_blocks}, num_threads={num_threads}, "
            "num_slots={num_slots}, unroll={unroll}, primitive={primitive!r}, "
            "cluster={cluster}, cluster_y={cluster_y}, "
            "tma_drain_warps={tma_drain_warps})"
        )
        return base.format(**config)
