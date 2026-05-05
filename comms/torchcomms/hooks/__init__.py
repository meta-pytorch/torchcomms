# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict
# patternlint-disable fbcode-nonempty-init-py

"""
TorchComm hooks module.

This module serves as a namespace for TorchComm hook types.
"""

from torchcomms.hooks.chash import chash
from torchcomms.hooks.clog import clog
from torchcomms.hooks.fr import FlightRecorderHook
from torchcomms.hooks.nan_check import NanCheckHook

__all__ = [
    "chash",
    "clog",
    "FlightRecorderHook",
    "NanCheckHook",
]

for name in __all__:
    cls = globals()[name]
    cls.__module__ = "torchcomms.hooks"
