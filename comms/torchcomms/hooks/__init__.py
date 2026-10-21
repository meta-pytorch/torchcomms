# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict
# patternlint-disable fbcode-nonempty-init-py

"""
TorchComm hooks module.

This module serves as a namespace for TorchComm hook types.
"""

from torchcomms.hooks.clog import clog

__all__ = [
    "clog",
]

for name in __all__:
    cls = globals()[name]
    cls.__module__ = "torchcomms.hooks"
