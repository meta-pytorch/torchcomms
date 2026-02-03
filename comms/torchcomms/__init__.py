# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict
import ctypes
import os
import sys
from datetime import timedelta
from importlib.metadata import entry_points

# We need to load this upfront since libtorchcomms depend on libtorch
import torch  # noqa: F401
from torchcomms.functional import is_torch_compile_supported_and_enabled

torch_compile_supported_and_enabled: bool = is_torch_compile_supported_and_enabled()

if torch_compile_supported_and_enabled:
    from torch._opaque_base import OpaqueBaseMeta

    # make the metaclass available to the pybind module
    sys.modules["torchcomms._opaque_meta"] = type(
        "module", (), {"OpaqueBaseMeta": OpaqueBaseMeta}
    )()

    # to support opaque registration for time delta.
    class Timeout(timedelta, metaclass=OpaqueBaseMeta):
        pass
else:
    # When compile support is disabled, define Timeout without the metaclass
    class Timeout(timedelta):
        pass


def _load_libtorchcomms() -> None:
    libtorchcomms_path = os.path.join(os.path.dirname(__file__), "libtorchcomms.so")
    # OSS build, buck native linking links everything together so this is not needed
    if os.path.exists(libtorchcomms_path):
        # load this using RTLD_LOCAL so that we don't pollute the global namespace
        # We need to load this upfront since _comms and _comms_* depend on it
        # and won't be able to find it themselves.
        ctypes.CDLL(libtorchcomms_path, mode=ctypes.RTLD_LOCAL)


_load_libtorchcomms()
from torchcomms._comms import *  # noqa: E402, F401, F403
import torchcomms.objcol as objcol  # noqa: E402, F401, F403

if torch_compile_supported_and_enabled:
    # Import collectives first to ensure all operations are registered
    # This must happen before patch_torchcomm() so that window operations
    # and other collectives are registered and can be patched
    from torchcomms.functional import collectives  # noqa: F401


# The documentation uses __all__ to determine what is documented and in what
# order.
__all__ = [  # noqa: F405
    "new_comm",
    "TorchComm",
    "ReduceOp",
    "TorchWork",
    "Timeout",
    "BatchP2POptions",
    "BatchSendRecv",
    "P2POp",
    "CommOptions",
    "TorchCommWindow",
]

for name in __all__:
    cls = globals()[name]
    cls.__module__ = "torchcomms"


def _load_backend(backend: str) -> None:
    """Used to load backends lazily from C++

    If a backend is already loaded, this function is a no-op.
    """
    found = entry_points(group="torchcomms.backends", name=backend)
    if not found:
        raise ModuleNotFoundError(
            f"failed to find backend {backend}, is it registered via entry_points.txt?"
        )
    wheel = next(iter(found))
    wheel.load()
