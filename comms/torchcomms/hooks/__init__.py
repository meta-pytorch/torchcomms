# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""
TorchComm hooks module for tracking and validating collective operations.

This module provides hook classes for TorchComm communicators:

- ``FlightRecorderHook``: Tracks all collective operations in flight. Output
  format matches the OSS FlightRecorder from PyTorch's distributed module.
- ``NanCheckHook``: Checks for NaN/Inf in tensors before collectives run,
  catching numerical instability before it propagates across ranks.

Example:
    >>> from torchcomms import hooks
    >>> import torchcomms
    >>> comm = torchcomms.new_comm("nccl", device, "world")
    >>> recorder = hooks.FlightRecorderHook(max_entries=1024)
    >>> recorder.register_with_comm(comm)
    >>> nan_check = hooks.NanCheckHook()
    >>> nan_check.register_with_comm(comm)
"""

from typing import TYPE_CHECKING

from torchcomms.hooks.nan_check import NanCheckHook

if TYPE_CHECKING:
    from torchcomms.hooks._fr import FlightRecorderHook
else:
    from torchcomms._comms.hooks import FlightRecorderHook

__all__ = [
    "FlightRecorderHook",
    "NanCheckHook",
]

for name in __all__:
    cls = globals()[name]
    cls.__module__ = "torchcomms.hooks"
