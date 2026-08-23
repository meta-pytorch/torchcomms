# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""CuTe collective backend over the shared NVLink transport.

Symbols are exposed lazily so importing the package does not eagerly load the CuTe DSL.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "nvl_ops",
]

# Exported name -> (submodule, attribute). ``attr is None`` exports the submodule
# itself (e.g. ``nvl_ops``).
_LAZY: dict[str, tuple[str, str | None]] = {
    "nvl_ops": ("nvl_ops", None),
}


def __getattr__(name: str) -> Any:
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = entry
    mod = import_module(f".{mod_name}", __name__)
    return mod if attr is None else getattr(mod, attr)


def __dir__() -> list[str]:
    return sorted(__all__)
