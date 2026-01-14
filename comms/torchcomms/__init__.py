# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe
import ctypes
import logging
import os
from contextlib import contextmanager
from importlib.metadata import entry_points
from typing import Generator, List, Optional

# We need to load this upfront since libtorchcomms depend on libtorch
import torch  # noqa: F401

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

# Import pickle support for object identity preservation
import torchcomms._pickle_support  # noqa: F401
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
    "BatchP2POptions",
    "BatchSendRecv",
    "P2POp",
    "CommOptions",
    "TorchCommWindow",
    "coalescing",
    "CoalescingManager",
]

# Set __module__ for C++ bindings (Python-defined ones are set automatically)
_cpp_exports = [
    "new_comm",
    "TorchComm",
    "ReduceOp",
    "TorchWork",
    "BatchP2POptions",
    "BatchSendRecv",
    "P2POp",
    "CommOptions",
    "TorchCommWindow",
]
for name in _cpp_exports:
    type = globals()[name]
    type.__module__ = "torchcomms"


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


def _are_we_tracing() -> bool:
    """Check if we're in a tracing/compiling context."""
    from torch.compiler import is_compiling as is_torchdynamo_compiling

    if is_torchdynamo_compiling():
        return True
    # If fake mode is turned on, we are almost definitely compiling/tracing
    if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is not None:
        return True
    return False


class CoalescingManager:
    """Manager returned by coalescing() context manager.

    Provides access to the work handle and coalesced tensors after
    the coalescing block completes.
    """

    def __init__(self) -> None:
        self.work: Optional["TorchWork"] = None  # noqa: F405
        self.tensors: List[torch.Tensor] = []
        self._waited: bool = False

    def wait(self) -> None:
        """Wait for all coalesced operations to complete.

        In tracing/compile mode, this generates a torchcomm_wait_tensors_ op
        that establishes proper data dependencies for all coalesced tensors.
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

        if _are_we_tracing() and self.tensors:
            # Use the wait_tensors op to establish data dependencies in traced graph
            # This is critical for torch.compile - it creates a dependency from
            # all output tensors to this wait point
            torch.ops.torchcomms.torchcomm_wait_tensors_(self.tensors)
        elif self.work is not None:
            # Eager mode - wait on the work handle
            self.work.wait()


def _log_coalesced_ops(
    manager: CoalescingManager,
    comm: "TorchComm",  # noqa: F405
) -> None:
    """Log information about coalesced operations.

    Logs tensor shapes/dtypes and, if in compile mode, attempts to log
    FX graph information.
    """
    num_tensors = len(manager.tensors)
    logger.info(
        "[torchcomms] Coalescing block ended on comm=%s with %d tensor(s)",
        comm.get_name(),
        num_tensors,
    )

    if num_tensors > 0:
        # Log tensor details
        tensor_info = []
        for i, t in enumerate(manager.tensors):
            info = f"  [{i}] shape={list(t.shape)}, dtype={t.dtype}, device={t.device}"
            tensor_info.append(info)
        logger.info("[torchcomms] Coalesced tensors:\n%s", "\n".join(tensor_info))

    # If we're in compile/tracing mode, try to get FX graph info
    if _are_we_tracing():
        try:
            # Try to get the current FX tracer context
            from torch._dynamo import current_trace
            from torch._dynamo.output_graph import OutputGraph

            tracer = current_trace()
            if tracer is not None and isinstance(tracer, OutputGraph):
                # Log the graph nodes related to our coalesced tensors
                logger.info(
                    "[torchcomms] FX graph capture active, graph has %d nodes",
                    len(tracer.graph.nodes),
                )
                # Log the most recent nodes (likely our collectives)
                nodes = list(tracer.graph.nodes)
                recent_nodes = nodes[-min(10, len(nodes)) :]
                node_strs = []
                for node in recent_nodes:
                    node_strs.append(f"  {node.op}: {node.name} = {node.target}")
                if node_strs:
                    logger.info(
                        "[torchcomms] Recent FX graph nodes:\n%s",
                        "\n".join(node_strs),
                    )
        except Exception as e:
            # Don't fail if we can't get the graph - this is just for debugging
            logger.debug("[torchcomms] Could not capture FX graph info: %s", str(e))


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
        "[torchcomms] Starting coalescing block on comm=%s (tracing=%s)",
        comm.get_name(),
        _are_we_tracing(),
    )

    comm.start_coalescing()
    try:
        yield manager
    finally:
        manager.work = comm.end_coalescing()
        _log_coalesced_ops(manager, comm)
