# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
DO NOT DELETE

This file is used to test the build of the torchcomms package in CI.
"""

import importlib.util
import os

import torch
import torchcomms  # noqa: F401

# `import torchcomms` does not reach the backend extensions: they are resolved
# lazily through the torchcomms.backends entry points, and only when a comm is
# created. So an extension that links but cannot load is invisible to every
# other check in this file, and the wheel ships green.
if importlib.util.find_spec("torchcomms._comms_ncclx") is not None:
    import torchcomms._comms_ncclx  # noqa: F401

if importlib.util.find_spec("torchcomms._transport") is not None:
    from torchcomms._transport import RdmaMemory, RdmaTransport

    if torch.cuda.is_available() and RdmaTransport.supported():
        # DMA-BUF exports require page-aligned ranges.
        page_size = os.sysconf("SC_PAGESIZE")
        storage = torch.empty(2 * page_size, dtype=torch.uint8, device="cuda:0")
        offset = -storage.data_ptr() % page_size
        tensor = storage[offset : offset + page_size]
        memory = RdmaMemory(tensor)
        assert memory.to_view().size() == tensor.nbytes
