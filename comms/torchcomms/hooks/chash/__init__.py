# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict
# patternlint-disable fbcode-nonempty-init-py

"""
chash module for TorchComm hooks.

Computes and logs hashes of communication buffers before and after
each collective operation for silent data corruption detection.

Example:
    >>> from torchcomms.hooks import chash
    >>> hasher = chash(output="/tmp/chash.log")
    >>> hasher.register_with_comm(comm)
    >>> # ... run collectives ...
    >>> comm.finalize()
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchcomms.hooks.chash._chash import chash
else:
    import torchcomms._comms as _comms_mod

    chash = _comms_mod.hooks.chash.chash

__all__ = [
    "chash",
]
