# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

import logging
from contextlib import contextmanager
from typing import Generator, List, Optional

import torch
from torchcomms._comms import TorchComm, TorchWork

logger: logging.Logger = logging.getLogger(__name__)
logger_is_enabled_for_debug: bool = logger.isEnabledFor(logging.DEBUG)


def _log(msg: str, *args: object) -> None:
    """Log a message, using HOP print during compilation if DEBUG is enabled."""
    if torch.compiler.is_compiling():
        if logger_is_enabled_for_debug:
            # Use HOP print to avoid graph breaks during compilation
            # and gate behind logger_is_enabled_for_debug so we don't
            # pollute the graph unless absolutely needed.
            formatted = msg % args if args else msg
            torch._higher_order_ops.print("{}", formatted)
    else:
        logger.debug(msg, *args)


def _are_we_tracing() -> bool:
    """Check if we're in a tracing/compiling context."""
    from torch.compiler import is_compiling as is_torchdynamo_compiling

    if is_torchdynamo_compiling():
        return True
    # If fake mode is turned on, we are almost definitely compiling/tracing
    if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is not None:
        return True
    return False


def _log_coalesced_ops(
    manager: "CoalescingManager",
    comm: "TorchComm",  # noqa: F405
) -> None:
    """Log information about coalesced operations."""
    num_tensors = len(manager.tensors)
    _log(
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
        _log("[torchcomms] Coalesced tensors:\n%s", "\n".join(tensor_info))


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

        if _are_we_tracing():
            # Call the traced wait function - this is intercepted by dynamo
            # to generate wait_tensors with all tracked coalesced tensors
            _coalescing_wait_traced()
        elif self.work is not None:
            # Eager mode - wait on the work handle
            self.work.wait()


def _coalescing_wait_traced() -> None:
    """Placeholder function for coalescing wait during tracing.

    This function is intercepted by dynamo via a registered handler.
    During tracing, it generates wait_tensors for all tracked coalesced tensors.
    In eager mode, this should never be called (CoalescingManager.wait uses work.wait).
    """
    # This should only be called during tracing, where it's intercepted by dynamo.
    # If we reach here in eager mode, it's a bug.
    pass


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
        with torchcomms.coalescing.coalesce(comm) as cm:
            comm.all_reduce(tensor1, op, async_op=True)
            comm.all_reduce(tensor2, op, async_op=True)
        # After context exit, cm.work is the coalesced work handle
        cm.wait()  # Wait for all operations
    """
    # Import coalescing context tracking functions
    from torchcomms.functional.collectives import (
        _end_coalescing_context,
        _start_coalescing_context,
    )

    manager = CoalescingManager()

    _log(
        "[torchcomms] Starting coalescing block on comm=%s (tracing=%s)",
        comm.get_name(),
        _are_we_tracing(),
    )

    # Start Python-level tracking for torch.compile fullgraph support
    # Pass the manager so tensors are registered directly on it
    _start_coalescing_context(manager)

    comm.start_coalescing()
    try:
        yield manager
    finally:
        manager.work = comm.end_coalescing()
        _end_coalescing_context()
        _log_coalesced_ops(manager, comm)
