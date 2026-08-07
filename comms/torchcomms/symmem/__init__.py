# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict
# patternlint-disable fbcode-nonempty-init-py

import torchcomms
from torchcomms._comms import _is_backend_registered, register_backend
from torchcomms.symmem.backend import TorchCommSymmem

__all__ = ["TorchCommSymmem"]


if not _is_backend_registered("symmem"):
    register_backend("symmem", TorchCommSymmem)
