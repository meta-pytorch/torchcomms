# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""
Flight Recorder module for TorchComm hooks.

This module provides the FlightRecorderHook class for tracking all collective
operations in flight for TorchComm communicators. The output format matches
the OSS FlightRecorder format from PyTorch's distributed module, so traces
can be analyzed using the same fr_trace analysis tools.

Example:
    >>> from torchcomms.hooks import fr
    >>> import torchcomms
    >>> comm = torchcomms.new_comm("nccl", device, "world")
    >>> recorder = fr.FlightRecorderHook(max_entries=1024)
    >>> recorder.register_with_comm(comm)
    >>> # ... run some collectives ...
    >>> json_trace = recorder.dump_json()
"""

from torchcomms.hooks.fr._fr import FlightRecorderHook

__all__ = [
    "FlightRecorderHook",
]
