# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""CuTe backend: send/recv tile primitives + hooks over the shared transport.

Symbols are exposed lazily (PEP 562 ``__getattr__``) so that merely importing
this package does not eagerly pull the cutlass DSL. The cutlass-backed
primitives (``send_tiles``/``recv_tiles``/``nvl_ops``/``copy_*``) load only on
first access, while the pure-Python submodules (``ctx``, ``ib_ops``) stay
importable in a GPU-less sandbox. The fused ``all_to_all`` collective is layered
on top of these primitives and registers its own exports in a later diff.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "send_tiles",
    "recv_tiles",
    "nvl_ops",
    "ib_ops",
    "copy_produce",
    "copy_consume",
]

# Exported name -> (submodule, attribute). ``attr is None`` exports the submodule
# itself (e.g. ``nvl_ops`` / ``ib_ops``).
_LAZY: dict[str, tuple[str, str | None]] = {
    "send_tiles": ("send_recv", "send_tiles"),
    "recv_tiles": ("send_recv", "recv_tiles"),
    "copy_produce": ("hooks", "copy_produce"),
    "copy_consume": ("hooks", "copy_consume"),
    "nvl_ops": ("nvl_ops", None),
    "ib_ops": ("ib_ops", None),
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
