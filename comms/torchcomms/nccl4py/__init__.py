# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe

from torchcomms._comms import _is_backend_registered, register_backend
from torchcomms.nccl4py.backend import TorchCommNCCL4Py

if not _is_backend_registered("nccl4py"):
    register_backend("nccl4py", TorchCommNCCL4Py)


__all__ = ["TorchCommNCCL4Py"]
