# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict
import ctypes
import logging
import os
import sys
from contextlib import contextmanager
from datetime import timedelta
from importlib.metadata import entry_points
from typing import Generator, Optional

# We need to load this upfront since libtorchcomms depend on libtorch
import torch  # noqa: F401


# to support opaque registration for time delta.
class Timeout(timedelta, metaclass=OpaqueBaseMeta):
    pass


torchcomms_compile_support_enabled: bool = os.environ.get(
    "TORCHCOMMS_PATCH_FOR_COMPILE", ""
).lower() in (
    "1",
    "true",
)

if torchcomms_compile_support_enabled:
    from torch._opaque_base import OpaqueBaseMeta

    # make the metaclass available to the pybind module
    sys.modules["torchcomms._opaque_meta"] = type(
        "module", (), {"OpaqueBaseMeta": OpaqueBaseMeta}
    )()

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _load_libtorchcomms() -> None:
    libtorchcomms_path = os.path.join(os.path.dirname(__file__), "libtorchcomms.so")
    # OSS build, buck native linking links everything together so this is not needed
    if os.path.exists(libtorchcomms_path):
        # load this using RTLD_LOCAL so that we don't pollute the global namespace
        # We need to load this upfront since _comms and _comms_* depend on it
        # and won't be able to find it themselves.
        ctypes.CDLL(libtorchcomms_path, mode=ctypes.RTLD_LOCAL)


_load_libtorchcomms()
from torchcomms._comms import *  # noqa: F401, F403
import torchcomms.objcol as objcol  # noqa: F401, F403

if os.environ.get("TORCHCOMMS_PATCH_FOR_COMPILE", "").lower() in ("1", "true"):
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
    "coalescing",
    "CoalescingManager",
]

for name in __all__:
    try:
        globals()[name].__module__ = "torchcomms"
    except KeyError:  # ignore non-c++ bindings
        pass


def _load_backend(backend: str) -> None:
    """Used to load backends lazily from C++

    If a backend is already loaded, this function is a no-op.
    """
    found = entry_points(group="torchcomms.backends", name=backend)
    if not found:
        raise ModuleNotFoundError(
            f"failed to find backend {backend}, is it registered via entry_points.txt?"
        )
    (wheel,) = found
    wheel.load()


class CoalescingManager:
    """Manager returned by coalescing() context manager.

    Provides access to the work handle and coalesced tensors after
    the coalescing block completes.
    """

    def __init__(self) -> None:
        self.work: Optional["TorchWork"] = None  # noqa: F405
        self._waited: bool = False

    def wait(self) -> None:
        """Wait for all coalesced operations to complete.

        In eager mode, it waits on the work handle directly.

        Raises:
            RuntimeError: If wait() has already been called on this manager.
        """
        if self._waited:
            raise RuntimeError(
                "CoalescingManager.wait() has already been called. "
                "Each coalescing block can only be waited on once."
            )

        self._waited = True

        if self.work is not None:
            self.work.wait()


@contextmanager
def coalesce(
    comm: "TorchComm",  # noqa: F405
) -> Generator[CoalescingManager, None, None]:
    """Context manager for coalescing collective operations.

    Within this context, collective operations are batched together and
    executed as a single fused operation when the context exits.

    For torch.compile compatibility, the returned manager provides access
    to the work handle and tensors for proper data dependency tracking.

    Args:
        comm: The TorchComm communicator to use.

    Yields:
        CoalescingManager with work handle and tensors available after exit.

    Example:
        with torchcomms.coalesce(comm) as cm:
            comm.all_reduce(tensor1, op, async_op=True)
            comm.all_reduce(tensor2, op, async_op=True)
        # After context exit, cm.work is the coalesced work handle
        cm.wait()  # Wait for all operations
    """
    manager = CoalescingManager()

    logger.info(
        "[torchcomms] Starting coalescing block on comm=%s",
        comm.get_name(),
    )

    comm.start_coalescing()
    try:
        yield manager
    finally:
        manager.work = comm.end_coalescing()
        logger.info(
            "[torchcomms] Coalescing block ended on comm=%s",
            comm.get_name(),
        )
