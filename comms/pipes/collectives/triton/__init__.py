# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
"""
Pipes Triton Collectives Module.

This module provides device-initiated collective operations implemented
using TorchComms Triton APIs. These collectives run entirely on GPU
without CPU involvement in the data path.

Available Collectives:
    device_alltoallv_dynamic: AlltoAllv with GPU-resident counts

High-level API (recommended — MSL-compatible signature):
    AlltoallvOp: Token-level alltoallv with internal buffer management

Helpers:
    alloc_comms_buffer: Transport-compatible buffer allocation helper

Key Feature: GPU-Resident Counts
--------------------------------
Unlike traditional CPU-initiated collectives where counts must be known on
the host before launch, these implementations read counts directly from GPU
memory. This enables fused compute + communication pipelines without CPU
roundtrips.
"""

from comms.pipes.collectives.triton.utils import alloc_comms_buffer

try:
    from comms.pipes.collectives.triton.alltoallv_op import AlltoallvOp
    from comms.pipes.collectives.triton.device_alltoallv_dynamic import (
        auto_tune_alltoallv_params,
        compute_offsets_from_sizes,
        device_alltoallv_dynamic,
        exchange_offsets,
        prewarm_completion_counters,
    )
except ImportError:
    # Triton or torchcomms.triton.fb not available (e.g. no GPU in CI)
    pass


__all__ = [
    # High-level API (recommended — MSL-compatible signature)
    "AlltoallvOp",
    # Helpers
    "alloc_comms_buffer",
    # Raw collective APIs
    "device_alltoallv_dynamic",
    "auto_tune_alltoallv_params",
    "compute_offsets_from_sizes",
    "exchange_offsets",
    "prewarm_completion_counters",
]
