# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
"""
Utility helpers for the AlltoallvOp high-level wrapper.

Provides convenience functions for transport-compatible buffer allocation,
eliminating boilerplate that users would otherwise need to write manually.
"""

from typing import Sequence, Union

import torch
import torchcomms


__all__ = [
    "alloc_comms_buffer",
]


def alloc_comms_buffer(
    shape: Union[int, Sequence[int]],
    dtype: torch.dtype,
    device: Union[str, torch.device],
    backend: str = "ncclx",
) -> tuple[torch.Tensor, "torch.cuda.MemPool"]:
    """
    Allocate a CUDA tensor using the TorchComms memory allocator,
    sized exactly to the requested shape.

    This allocator returns buffers that are compatible with one-sided put
    operations over any transport (NVLink, RDMA, etc.).

    Args:
        shape: Tensor shape (int for 1-D, or a sequence of ints).
        dtype: Tensor dtype (e.g., torch.float32).
        device: CUDA device (e.g., "cuda:0" or torch.device("cuda", 0)).
        backend: TorchComms backend name (default: "ncclx").  Must match
                 the backend used by the TorchComm communicator.  Obtain
                 via ``comm.get_backend()``.

    Returns:
        (tensor, pool) — the allocated tensor and its memory pool.
        ``AlltoallvOp`` manages pools internally; this helper is primarily
        used by ``AlltoallvOp.__init__`` and advanced callers who need
        direct transport-compatible buffer allocation.

    Note:
        The returned ``pool`` object MUST remain alive for as long as the
        tensor is registered for one-sided operations. If the pool is garbage
        collected while the tensor is still registered, the underlying memory
        deregistration will race with active DMA and cause GPU faults.
    """
    allocator = torchcomms.get_mem_allocator(backend)
    pool = torch.cuda.MemPool(allocator)
    # Normalize shape to a tuple for torch.zeros
    size: Sequence[int] = (shape,) if isinstance(shape, int) else tuple(shape)
    with torch.cuda.use_mem_pool(pool):
        tensor = torch.zeros(size, dtype=dtype, device=device)
    return tensor, pool
