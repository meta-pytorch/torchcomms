# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict
# patternlint-disable fbcode-nonempty-init-py

"""
NanCheck module for TorchComm hooks.

This module provides the NanCheckHook class for detecting NaN values
in tensors before collective operations.

Example:
    >>> from torchcomms.hooks import NanCheckHook
    >>> import torchcomms
    >>> comm = torchcomms.new_comm("nccl", device, "world")
    >>> nan_check = NanCheckHook()
    >>> nan_check.register_with_comm(comm)
    >>> # NaN in tensors will now raise RuntimeError before collective runs
"""

from torchcomms.hooks.nan_check.nan_check import NanCheckHook

__all__ = [
    "NanCheckHook",
]
