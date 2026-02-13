# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""
TorchComm hooks module for tracking collective operations.

This module provides the FlightRecorderHook class for tracking all collective
operations in flight for TorchComm communicators. The output format matches
the OSS FlightRecorder format from PyTorch's distributed module, so traces
can be analyzed using the same fr_trace analysis tools.

Example:
    >>> from torchcomms import hooks
    >>> import torchcomms
    >>> comm = torchcomms.new_comm("nccl", device, "world")
    >>> recorder = hooks.FlightRecorderHook(max_entries=1024)
    >>> recorder.register_with_comm(comm)
    >>> # ... run some collectives ...
    >>> json_trace = recorder.dump_json()
"""

from torchcomms._comms.fr import FlightRecorderHook

__all__ = [
    "FlightRecorderHook",
]

for name in __all__:
    cls = globals()[name]
    cls.__module__ = "torchcomms.hooks"
