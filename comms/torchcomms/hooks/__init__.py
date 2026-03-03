# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""
TorchComm hooks module.

This module serves as a namespace for TorchComm hook types.
"""

from torchcomms.hooks.fr import FlightRecorderHook

__all__ = [
    "FlightRecorderHook",
]

for name in __all__:
    cls = globals()[name]
    cls.__module__ = "torchcomms.hooks"
