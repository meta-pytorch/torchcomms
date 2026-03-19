# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
"""
Pipes Triton Collectives Module.

Device-initiated collective operations using TorchComms Triton APIs.
Collectives run entirely on GPU without CPU involvement in the data path.

Available Collectives:
    device_alltoallv_dynamic: AlltoAllv with GPU-resident counts

High-level API (recommended):
    AlltoallvOp: Token-level alltoallv with zero-copy buffer ownership.
        User provides both input and output tensors (MANDATORY).
        100% memory reduction - no internal allocation.
        User updates input tensor in-place between calls.

Buffer Allocation:
    AlltoallvOp.alloc_buffer(): Instance method to allocate GIN-compatible
        tensors from the op's internal memory pool. Must be called AFTER
        setup() (i.e., inside the ``with op:`` block).

Zero-Copy Architecture
----------------------
AlltoallvOp takes ownership of user-provided tensors, registering them
as GIN send/recv buffers. The user's input_tensor IS the send buffer,
and the user's output_tensor IS the recv buffer. No internal allocation,
no copies.

**IMPORTANT**: Tensors must be allocated using alloc_buffer() which
uses ncclMemAlloc (cuMem APIs). Regular torch.empty() uses cudaMalloc
which is NOT compatible with GIN registration.

Usage Example::

    op = AlltoallvOp(comm, max_tokens, D, dtype, device, max_recv_per_peer)

    with op:
        # Allocate GIN-compatible tensors (pool managed internally)
        input_tensor = op.alloc_buffer((max_tokens, D))
        output_tensor = op.alloc_buffer((world_size * max_recv_per_peer, D))

        # Use the tensors
        input_tensor[:] = data
        result = op.alltoallv(input_tensor, output_tensor,
                              output_splits, input_splits)

Key Feature: GPU-Resident Counts
--------------------------------
Unlike traditional CPU-initiated collectives where counts must be known on
the host before launch, these implementations read counts directly from GPU
memory. This enables fused compute + communication pipelines without CPU
roundtrips.
"""

from comms.pipes.collectives.triton.alltoallv_op import AlltoallvOp
from comms.pipes.collectives.triton.device_alltoallv_dynamic import (
    auto_tune_alltoallv_params,
    compute_offsets_from_sizes,
    device_alltoallv_dynamic,
    exchange_offsets,
    prewarm_completion_counters,
)


__all__ = [
    # High-level API (recommended — MSL-compatible signature)
    "AlltoallvOp",
    # Raw collective APIs
    "device_alltoallv_dynamic",
    "auto_tune_alltoallv_params",
    "compute_offsets_from_sizes",
    "exchange_offsets",
    "prewarm_completion_counters",
]
