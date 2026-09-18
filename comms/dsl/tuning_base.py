# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""DSL-agnostic tuning base: shared Config / Key / Spec / Adapter mixin for backends.

This layer owns everything the backends share: the tunable Config / lookup Key /
input Spec dataclasses, their JSON roundtrip, rendering, and the dict-grid candidate
expansion. Backends inherit and add their core fields. A generated-table entry that
does not name a backend is attributed to the loading adapter's backend.
"""

from __future__ import annotations

import importlib
import itertools
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import torch

try:  # optional dep: the comms-owned tuner engine; absent until comm_tuning lands.
    # pyre-ignore[21]: optional dependency, resolved at runtime when the tuner is present.
    from comm_tuning.adapter import CommKernelTuningAdapter as _CommTunerBase
except ImportError:  # the adapter base is still importable/testable without the engine
    _CommTunerBase = object

logger: logging.Logger = logging.getLogger(__name__)


def _device_tag(device: torch.device | int | str | None = None) -> str:
    """Short, stable hardware tag for the tuning key (e.g. ``H100``, ``GB300``,
    ``B200``, ``A100``). Tuned configs are hardware-specific -- the best launch config
    differs by GPU/NVLink generation -- so the key includes the device so an H100-tuned
    table and a GB300-tuned table coexist in one map instead of colliding on identical
    (world_size, dtype, numel, rows, transport) keys. Backend-agnostic, so both DSLs build
    an identical tag; MUST be computed identically offline (tuner) and at runtime (lookup).
    """
    try:
        name = torch.cuda.get_device_name(device)  # e.g. "NVIDIA H100", "NVIDIA GB300"
    except (RuntimeError, ValueError, AssertionError):
        # torch raises RuntimeError when CUDA is unavailable / the index is out of range,
        # ValueError for a non-CUDA device (e.g. a CPU tensor's device), and AssertionError
        # on a bad device arg; "unknown" is the intended fallback only for those. Anything
        # else is a real bug and must not be folded into a tuning key.
        return "unknown"
    # The (?!\d) lookahead stops a 4-digit model (e.g. "RTX A6000") from partial-matching
    # to "A600"; such names fall through to the readable cleaned-name path instead.
    m = re.search(r"(GB\d{3}|GH\d{3}|B\d{3}|H\d{3}|A\d{3})(?!\d)", name.upper())
    return m.group(1) if m else name.replace(" ", "_")


# Default per-rank message-size ladder for the autotuner: the 32 B .. 2 GB 2x ladder (27 sizes)
# plus 48 MB / 96 MB for mid-band resolution where the launch-config dip lives = 29 sizes. The
# best launch config shifts across sizes, so each is tuned independently; mirrors the benchmark
# size sweep. Used by the adapter's enumerate_input_specs when no explicit shapes are given.
SIZE_LADDER: tuple[int, ...] = (
    32,
    64,
    128,
    256,
    512,
    1024,
    2 * 1024,
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1 * 1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
    48 * 1024 * 1024,
    64 * 1024 * 1024,
    96 * 1024 * 1024,
    128 * 1024 * 1024,
    256 * 1024 * 1024,
    512 * 1024 * 1024,
    1024 * 1024 * 1024,
    2 * 1024 * 1024 * 1024,
)


def import_obj(path: str):
    """Resolve a ``module:attr`` import path to the object, for CLI ``--produce`` /
    ``--reference`` so a user can tune a custom hook with no adapter file."""
    mod, _, name = path.partition(":")
    if not mod or not name:
        raise ValueError(f"expected 'module:attr', got {path!r}")
    return getattr(importlib.import_module(mod), name)


def resolve_variant(
    produce, consume, variant: str, *, default_produce, default_consume
) -> str:
    """Single chokepoint for the tuned-table hook discriminator. The default identity copy
    hooks map to ``""`` (back-compat). Any *non-default* hook MUST carry an explicit
    ``variant`` -- else its tuned configs would silently collide with the default copy
    hook's, for a kernel with a different access pattern."""
    if produce is default_produce and consume is default_consume:
        return variant
    if not variant:
        raise ValueError(
            "a non-default produce/consume hook needs an explicit variant tag so its tuned "
            "configs don't collide with the default copy hook; pass variant=... "
            "(e.g. A2A(produce=my_op, variant='myq'))."
        )
    return variant


def geometry_signature(config: Any, fields: frozenset[str]) -> tuple:
    """Geometry-defining field values: two configs with the same signature can reuse one
    transport back-to-back; a different signature on a reused transport is the documented
    hazard the runtime geometry guard catches. ``fields`` is the per-collective set of
    knobs that change the physical staging geometry."""
    return tuple(getattr(config, f) for f in sorted(fields))


def nondefault_inapplicable_fields(
    config: Any,
    *,
    core_fields: frozenset[str],
    primitive_applies: dict[str, frozenset[str]],
    default: Any,
) -> set[str]:
    """Names of fields ``config`` sets to a NON-default value that its ``primitive`` does not
    consume at launch (empty == honest). Shared skeleton for the per-backend feasibility
    filter; the backend passes its ``core_fields`` / ``primitive_applies`` map / ``default``
    config. ``num_blocks`` and ``primitive`` are always excluded (grid axes, not knobs)."""
    applies = primitive_applies.get(config.primitive, frozenset())
    checked = core_fields - {"num_blocks", "primitive"}
    return {
        name
        for name in checked
        if name not in applies and getattr(config, name) != getattr(default, name)
    }


def check_geometry(
    transport: Any, config: Any, fields: frozenset[str], *, collective: str
) -> None:
    """Guard against an unsafe staging-geometry switch on a reused transport (both backends).

    The kernel's persistent step counters + shared staging buffer make back-to-back launches
    safe only when the geometry-defining ``fields`` are unchanged (see ``geometry_signature``).
    Switching geometry on the same transport without a drain reinterprets in-flight bytes; the
    framework has no runtime drain, so this guard surfaces the hazard.

    A geometry switch on a reused transport is a hard error by default: there is no runtime
    drain, so reinterpreting in-flight bytes silently corrupts staging data and a warn-and-
    proceed default would let that corruption through unnoticed. ``COMMS_DSL_ALLOW_GEOMETRY_SWITCH=1``
    downgrades it to a silent advance for callers whose successive launches are device-sync-
    separated (a benchmark / tuner sweeping configs at one size) -- they know no bytes are in
    flight across the switch. The transport is a non-frozen dataclass, so we stash the last
    accepted geometry on it directly (private attr).

    ``collective`` names the calling collective (e.g. ``"all_to_all"``) purely to attribute the
    hazard in the error message; this module is collective-agnostic, so it is required rather
    than defaulted -- a baked-in default would misattribute a different collective's hazard.
    """
    sig = geometry_signature(config, fields)
    prev = getattr(transport, "_last_geometry_signature", None)
    # The cache advances only on an accepted switch: reseeding the baseline before the raise
    # would let the next (still hazardous) switch slip through unflagged.
    if prev is None or prev == sig:
        transport._last_geometry_signature = (
            sig  # pyre-ignore[16]: runtime cache on the transport
        )
        return
    if os.environ.get("COMMS_DSL_ALLOW_GEOMETRY_SWITCH") == "1":
        transport._last_geometry_signature = sig  # pyre-ignore[16]: runtime cache
        return
    names = sorted(fields)
    raise ValueError(
        f"{collective}: staging geometry changed on a reused transport "
        f"({dict(zip(names, prev))} -> {dict(zip(names, sig))}). There is no runtime drain yet, "
        "so back-to-back calls of differing geometry on one transport (without an intervening "
        "device sync) would corrupt in-flight staging data. Use a fresh transport per "
        "geometry/shape, or set COMMS_DSL_ALLOW_GEOMETRY_SWITCH=1 if your calls are sync-separated."
    )


@dataclass(frozen=True)
class BaseTunableConfig:
    """Base tunable config shared across backends.

    Subclasses add backend-specific core fields.
    """

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseTunableConfig":
        # Subclasses override to construct their core fields.
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class BaseTuningKey:
    """Base lookup key shared across backends.

    The base stays backend-neutral: subclasses set the ``backend`` field (e.g. the Triton
    ``A2AKey`` defaults it to ``"triton"``). A table entry that does not name a backend is
    attributed to the loading adapter's backend on the deserialization path (see
    ``BaseAdapterMixin.key_from_json``), not here -- a default baked into the agnostic base
    would silently mislabel a different backend's key.
    """

    world_size: int
    dtype: str
    numel: int
    rows: int
    transport_kind: str
    device: str = ""
    backend: str = ""
    variant: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseTuningKey":
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class BaseInputSpec:
    """Serializable input spec shared across backends."""

    numel: int
    dtype: torch.dtype
    rows: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # torch.dtype -> string for JSON
        d["dtype"] = str(self.dtype).removeprefix("torch.")
        return d

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "BaseInputSpec":
        dtype_str = data["dtype"]
        dtype = getattr(torch, dtype_str) if isinstance(dtype_str, str) else dtype_str
        return cls(
            numel=int(data["numel"]),
            dtype=dtype,
            rows=int(data.get("rows", 0)),
        )


def expand_config_grid(
    grid: dict[str, list[Any]],
    config_cls: type,
    core_fields: frozenset[str],
) -> list[Any]:
    """Expand a flat ``{field: [values]}`` grid into the cartesian product of config objects.

    Every grid key must name a declared core config field; an unknown name is rejected.
    The single grid-expansion implementation, shared by
    ``BaseAdapterMixin.expand_candidate_grid`` and the per-backend grid helpers.
    """
    unknown = [k for k in grid if k not in core_fields]
    if unknown:
        raise ValueError(
            f"unknown grid field(s) {unknown}; allowed: {sorted(core_fields)}"
        )
    keys = list(grid)
    values = [grid[k] for k in keys]
    return [
        config_cls(**dict(zip(keys, combo))) for combo in itertools.product(*values)
    ]


class BaseAdapterMixin:
    """Shared adapter helpers for tuner adapters across backends.

    Subclasses must define:
      - core_config_fields: frozenset of core tunable field names
      - config_cls
      - key_cls
      - spec_cls
      - backend_name: str e.g. "triton" or "cute"
    and implement backend-specific abstract methods: enumerate_candidate_configs core logic,
    run_candidate, make_inputs, etc. This mixin provides shared JSON, render, expand, and key building.
    """

    # To be overridden by subclasses
    core_config_fields: frozenset[str] = frozenset()
    config_cls: type = BaseTunableConfig
    key_cls: type = BaseTuningKey
    spec_cls: type = BaseInputSpec
    backend_name: str = ""

    # Shared helpers below – subclasses may override for customization

    def expand_candidate_grid(self, grid: dict[str, list[Any]]) -> list[Any]:
        """Expand declarative grid into list of config objects."""
        return expand_config_grid(grid, self.config_cls, self.core_config_fields)

    def make_key_from_spec(
        self,
        spec: BaseInputSpec,
        world_size: int,
        transport_kind: str,
        device: str,
        variant: str,
    ) -> BaseTuningKey:
        # dtype string normalization
        dtype_str = (
            spec.dtype
            if isinstance(spec.dtype, str)
            else str(spec.dtype).removeprefix("torch.")
        )
        return self.key_cls(
            world_size=world_size,
            dtype=dtype_str,
            numel=spec.numel,
            rows=spec.rows,
            transport_kind=transport_kind,
            device=device,
            backend=self.backend_name,
            variant=variant,
        )

    def key_to_json(self, key: BaseTuningKey) -> tuple[str, dict[str, Any]]:
        return (type(key).__name__, asdict(key))

    def key_from_json(self, key_type: str, data: dict[str, Any]) -> BaseTuningKey:
        # key_type ignored, we trust subclass key_cls.
        data = dict(data)
        # A table entry that does not name a backend is attributed to this adapter's backend
        # (set by the subclass) and flagged, rather than assuming a backend in the base.
        if not data.get("backend") and self.backend_name:
            logger.warning(
                "tuned-table key missing 'backend'; backfilling %r", self.backend_name
            )
            data["backend"] = self.backend_name
        return self.key_cls(**data)

    def config_to_json(self, config: BaseTunableConfig) -> tuple[str, dict[str, Any]]:
        return (type(config).__name__, asdict(config))

    def config_from_json(
        self, config_type: str, data: dict[str, Any]
    ) -> BaseTunableConfig:
        return self.config_cls(**data)

    def spec_to_json(self, spec: BaseInputSpec) -> tuple[str, dict[str, Any]]:
        return ("BaseInputSpec", spec.to_json_dict())

    def spec_from_json(self, spec_type: str, data: dict[str, Any]) -> BaseInputSpec:
        # Subclass may override to return its specific spec subclass; base returns BaseInputSpec
        return self.spec_cls.from_json_dict(data)

    def render_key(self, key_type: str, key_dict: dict[str, Any]) -> str:
        # Build dynamic string template preserving core order for readability.
        base_order = [
            "world_size",
            "dtype",
            "numel",
            "rows",
            "transport_kind",
            "device",
            "backend",
            "variant",
        ]
        parts = []
        for f in base_order:
            if f in key_dict:
                v = key_dict[f]
                parts.append(f"{f}={v!r}" if isinstance(v, str) else f"{f}={v}")
        return f"{key_type}({', '.join(parts)})"

    def render_config(self, config_type: str, config_dict: dict[str, Any]) -> str:
        # Subclass should override core order; base does generic sorted keys.
        items = sorted(
            config_dict.items()
        )  # subclasses likely override for pretty order
        parts = [f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}" for k, v in items]
        return f"{config_type}({', '.join(parts)})"
