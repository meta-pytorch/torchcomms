# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""
NanCheckHook for detecting NaN values in tensors before collective operations.

Catches numerical instability early before it propagates across ranks.

Example:
    >>> from torchcomms.hooks import NanCheckHook
    >>> import torchcomms
    >>> comm = torchcomms.new_comm("nccl", device, "world")
    >>> nan_check = NanCheckHook()
    >>> nan_check.register_with_comm(comm)
    >>> # NaN in tensors will now raise RuntimeError before collective runs
"""

from __future__ import annotations

from typing import Any

import torch
from torchcomms._comms import PreHookArgs, RemovableHandle

# Lazily resolved reference to torch.ops.c10d.check_for_nan.
# The op is registered when torch.distributed is imported (which happens
# before any communicator is created), so we resolve on first use to
# avoid a circular import at module load time.
# pyre-ignore[5]: Global has no type annotation.
_check_for_nan = None


def _get_check_for_nan() -> Any:
    global _check_for_nan
    if _check_for_nan is None:
        _check_for_nan = torch.ops.c10d.check_for_nan
    return _check_for_nan


def _check_tensor(tensor: Any, label: str, op_name: str, comm_name: str) -> None:
    """Check a single tensor for NaN values using c10d::check_for_nan."""
    try:
        _get_check_for_nan()(tensor)
    except RuntimeError as e:
        raise RuntimeError(
            f"NaN detected in {label} tensor for '{op_name}' on comm '{comm_name}': {e}"
        ) from None


class NanCheckHook:
    """Hook that checks for NaN values in tensors before collective operations.

    Registers a pre-hook on communicators that inspects input and/or output
    tensors for NaN values using the dispatched ``c10d::check_for_nan``
    op (works on both CPU and CUDA). If detected, raises a ``RuntimeError``
    with context about which operation and communicator triggered the check.

    Args:
        check_inputs: Whether to check input tensors. Default: True.
        check_outputs: Whether to check output tensors. Default: False.
    """

    def __init__(
        self,
        check_inputs: bool = True,
        check_outputs: bool = False,
    ) -> None:
        self._check_inputs = check_inputs
        self._check_outputs = check_outputs
        self._handles: list[tuple[Any, RemovableHandle]] = []

    def register_with_comm(self, comm: Any) -> None:
        """Register the NaN check hook with a communicator.

        Args:
            comm: A TorchComm communicator instance.
        """
        comm_name: str = comm.get_name()

        def _pre_hook(args: PreHookArgs) -> None:
            op_name = str(args.name).rsplit(".", 1)[-1]

            if self._check_inputs:
                t = args.input_tensor
                if t is not None and t.is_floating_point():
                    _check_tensor(t, "input", op_name, comm_name)
                ts = args.input_tensors
                if ts is not None:
                    for tensor in ts:
                        if tensor.is_floating_point():
                            _check_tensor(tensor, "input", op_name, comm_name)

            if self._check_outputs:
                t = args.output_tensor
                if t is not None and t.is_floating_point():
                    _check_tensor(t, "output", op_name, comm_name)
                ts = args.output_tensors
                if ts is not None:
                    for tensor in ts:
                        if tensor.is_floating_point():
                            _check_tensor(tensor, "output", op_name, comm_name)

        handle = comm.register_pre_hook(_pre_hook)
        self._handles.append((comm, handle))

    def unregister(self) -> None:
        """Remove all registered hooks."""
        for _comm, handle in self._handles:
            handle.remove()
        self._handles.clear()

    def is_enabled(self) -> bool:
        """Return whether any communicators are registered."""
        return len(self._handles) > 0
