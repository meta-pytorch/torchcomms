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

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from comms.ctran.prims.collectives.triton.alltoallv_op import (
        AlltoallvOp as AlltoallvOp,
    )
    from comms.ctran.prims.collectives.triton.device_alltoallv_dynamic import (
        auto_tune_alltoallv_params as auto_tune_alltoallv_params,
        compute_offsets_from_sizes as compute_offsets_from_sizes,
        device_alltoallv_dynamic as device_alltoallv_dynamic,
        exchange_offsets as exchange_offsets,
        prewarm_completion_counters as prewarm_completion_counters,
    )
    from comms.ctran.prims.collectives.triton.utils import (
        alloc_comms_buffer as alloc_comms_buffer,
    )

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


def __getattr__(name: str) -> Any:
    if name == "AlltoallvOp":
        from comms.ctran.prims.collectives.triton.alltoallv_op import AlltoallvOp

        return AlltoallvOp
    if name == "alloc_comms_buffer":
        from comms.ctran.prims.collectives.triton.utils import alloc_comms_buffer

        return alloc_comms_buffer
    if name in {
        "auto_tune_alltoallv_params",
        "compute_offsets_from_sizes",
        "device_alltoallv_dynamic",
        "exchange_offsets",
        "prewarm_completion_counters",
    }:
        from importlib import import_module

        module = import_module(
            "comms.ctran.prims.collectives.triton.device_alltoallv_dynamic"
        )
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
