# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""DSL-agnostic tuning base: the shared tunable Config + reused-transport geometry guard.

Owns the runtime pieces shared by tunable collectives: the base config dataclass,
geometry signatures, and the reused-transport geometry guard.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass
from typing import Any


def geometry_signature(
    config: Any,
    fields: frozenset[str],
    resolved: tuple[tuple[str, Any], ...] = (),
) -> tuple:
    """Geometry-defining field values: two configs with the same signature can reuse one
    transport back-to-back; a different signature on a reused transport is the documented
    hazard the runtime geometry guard catches. ``fields`` is the per-collective set of
    knobs that change the physical staging geometry. ``resolved`` carries runtime-derived
    geometry such as the effective vector width or shape."""
    return tuple(getattr(config, f) for f in sorted(fields)) + tuple(
        value for _, value in resolved
    )


def check_geometry(
    transport: Any,
    config: Any,
    fields: frozenset[str],
    label: str = "all_to_all",
    *,
    resolved: tuple[tuple[str, Any], ...] = (),
) -> None:
    """Guard against an unsafe staging-geometry switch on a reused transport.

    ``label`` names the collective in the error message (this helper is DSL/collective-
    agnostic; later collectives pass their own name).

    The kernel's persistent step counters + shared staging buffer make back-to-back
    launches safe only when the geometry-defining ``fields`` are unchanged (see
    ``geometry_signature``). Switching geometry on the same transport without a drain
    reinterprets in-flight bytes; the framework has no runtime drain, so this guard
    surfaces the hazard.

    A geometry switch on a reused transport is a hard error by default: there is no
    runtime drain, so reinterpreting in-flight bytes silently corrupts staging data and a
    warn-and-proceed default would let that corruption through unnoticed.
    ``COMMS_DSL_ALLOW_GEOMETRY_SWITCH=1`` downgrades it to a silent advance for callers
    whose successive launches are device-sync-separated (a benchmark / tuner sweeping
    configs at one size) -- they know no bytes are in flight across the switch. The
    transport is a non-frozen dataclass, so we stash the last accepted geometry on it
    directly (private attr).
    """
    sig = geometry_signature(config, fields, resolved)
    prev = getattr(transport, "_last_geometry_signature", None)
    # The cache advances only on an accepted switch: reseeding the baseline before the
    # raise would let the next (still hazardous) switch slip through unflagged.
    if prev is None or prev == sig:
        transport._last_geometry_signature = sig
        return
    # Forgiving parse of the documented escape hatch: accept 1/true/yes (case-insensitive,
    # whitespace-trimmed); warn (not silently ignore) on a present-but-unrecognized value.
    _allow = os.environ.get("COMMS_DSL_ALLOW_GEOMETRY_SWITCH", "").strip().lower()
    if _allow in ("1", "true", "yes"):
        transport._last_geometry_signature = sig
        return
    if _allow not in ("", "0", "false", "no"):
        warnings.warn(
            f"COMMS_DSL_ALLOW_GEOMETRY_SWITCH={_allow!r} unrecognized; treating as unset "
            "(use 1/true/yes to enable).",
            stacklevel=2,
        )
    names = sorted(fields) + [name for name, _ in resolved]
    raise ValueError(
        f"{label}: staging geometry changed on a reused transport "
        f"({dict(zip(names, prev))} -> {dict(zip(names, sig))}). There is no runtime drain "
        "yet, so back-to-back calls of differing geometry on one transport (without an "
        "intervening device sync) would corrupt in-flight staging data. Use a fresh "
        "transport per geometry/shape, or set COMMS_DSL_ALLOW_GEOMETRY_SWITCH=1 if your "
        "calls are sync-separated."
    )


@dataclass(frozen=True)
class BaseTunableConfig:
    """Base tunable config shared across backends. Subclasses add core fields."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseTunableConfig":
        return cls(**data)  # type: ignore[arg-type]
