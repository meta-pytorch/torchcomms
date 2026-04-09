# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

from typing import Any, List

import torch
from torch.distributed import ProcessGroup, ReduceOp
from torchcomms.distwrap.collectives import _convert_reduce_op
from torchcomms.distwrap.pginfo import pg_info_get_data
from torchcomms.distwrap.utils import (
    get_group,
    get_torchcomms_instance,
    parse_backend_string,
    torchcomms_is_enabled,
)


# =============================================================================
# Window Operations (torchcomms-only)
# =============================================================================


def new_window(
    group: ProcessGroup | None = None,
    backend: str | None = None,
) -> Any:
    if not torchcomms_is_enabled():
        raise AssertionError("new_window requires torchcomms to be enabled")

    pg = get_group(group)

    if backend is not None:
        # Determine device type from backend parameter
        device_backends = parse_backend_string(backend)
        # Use the first (and typically only) device type from the parsed backend
        device_type = next(iter(device_backends.keys()))
        tc = get_torchcomms_instance(pg, device_type=device_type)
    else:
        # Pick the first available torchcomms instance from the group
        torchcomms_instances = pg_info_get_data(pg, "torchcomms")
        if not torchcomms_instances:
            raise AssertionError(
                f"No torchcomms instances found for process group {pg}"
            )
        tc = next(iter(torchcomms_instances.values()))

    return tc.new_window()


# =============================================================================
# AlltoAllv-Dedup Operations (torchcomms/ncclx-only)
# =============================================================================


def alltoallv_dedup_init(
    num_tokens: int,
    token_size: int,
    topk_k: int,
    experts_per_rank: int,
    dtype: torch.dtype,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> Any:
    if not torchcomms_is_enabled():
        raise AssertionError("alltoallv_dedup_init requires torchcomms to be enabled")

    pg = get_group(group)
    tc = get_torchcomms_instance(pg, device_type="cuda")

    # Verify backend is ncclx
    backend_name = tc.get_backend()
    if backend_name != "ncclx":
        raise AssertionError(
            f"alltoallv_dedup_init requires ncclx backend, got {backend_name}"
        )

    ncclx_backend = tc.get_backend_impl()
    return ncclx_backend.alltoallv_dedup_init(
        num_tokens, token_size, topk_k, experts_per_rank, dtype, async_op
    )


def alltoallv_dedup_exec(
    output_tensor: torch.Tensor,
    recv_block_ids: torch.Tensor,
    input_tensor: torch.Tensor,
    send_indices: torch.Tensor,
    forward_indices: torch.Tensor,
    recv_indices: torch.Tensor,
    persist_req: Any,
    group: ProcessGroup | None = None,
) -> Any:
    if not torchcomms_is_enabled():
        raise AssertionError("alltoallv_dedup_exec requires torchcomms to be enabled")

    pg = get_group(group)
    tc = get_torchcomms_instance(pg, device_type="cuda")

    # Verify backend is ncclx
    backend_name = tc.get_backend()
    if backend_name != "ncclx":
        raise AssertionError(
            f"alltoallv_dedup_exec requires ncclx backend, got {backend_name}"
        )

    ncclx_backend = tc.get_backend_impl()
    return ncclx_backend.alltoallv_dedup_exec(
        output_tensor,
        recv_block_ids,
        input_tensor,
        send_indices,
        forward_indices,
        recv_indices,
        persist_req,
    )


# =============================================================================
# AlltoAllv-Dynamic Operations (torchcomms/ncclx-only)
# =============================================================================


def alltoallv_dynamic_dispatch(
    output_tensors: List[torch.Tensor],
    output_split_sizes: torch.Tensor,
    input_tensor: torch.Tensor,
    input_split_sizes: torch.Tensor,
    input_split_indices: torch.Tensor,
    input_split_indices_per_rank: torch.Tensor,
    D: int,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> Any:
    if not torchcomms_is_enabled():
        raise AssertionError(
            "alltoallv_dynamic_dispatch requires torchcomms to be enabled"
        )

    pg = get_group(group)
    tc = get_torchcomms_instance(pg, device_type="cuda")

    # Verify backend is ncclx
    backend_name = tc.get_backend()
    if backend_name != "ncclx":
        raise AssertionError(
            f"alltoallv_dynamic_dispatch requires ncclx backend, got {backend_name}"
        )

    ncclx_backend = tc.get_backend_impl()
    return ncclx_backend.alltoallv_dynamic_dispatch(
        output_tensors,
        output_split_sizes,
        input_tensor,
        input_split_sizes,
        input_split_indices,
        input_split_indices_per_rank,
        D,
        async_op,
    )


def alltoallv_dynamic_combine(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    input_split_sizes: torch.Tensor,
    input_split_indices: torch.Tensor,
    input_split_indices_per_rank: torch.Tensor,
    D: int,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> None:
    if not torchcomms_is_enabled():
        raise AssertionError(
            "alltoallv_dynamic_combine requires torchcomms to be enabled"
        )

    pg = get_group(group)
    tc = get_torchcomms_instance(pg, device_type="cuda")

    # Verify backend is ncclx
    backend_name = tc.get_backend()
    if backend_name != "ncclx":
        raise AssertionError(
            f"alltoallv_dynamic_combine requires ncclx backend, got {backend_name}"
        )

    ncclx_backend = tc.get_backend_impl()
    ncclx_backend.alltoallv_dynamic_combine(
        output_tensor,
        input_tensor,
        input_split_sizes,
        input_split_indices,
        input_split_indices_per_rank,
        D,
        async_op,
    )


# =============================================================================
# ReduceScatter-Quantized Operations (torchcomms/ncclx-only)
# =============================================================================


def reduce_scatter_quantized(
    output: torch.Tensor,
    input: torch.Tensor,
    op: ReduceOp,
    seed: torch.Tensor,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> Any:
    """Reduce-scatter with stochastic quantization (FP32 input/output, BF16 transport).

    Performs a reduce-scatter where data is stochastically rounded from FP32 to BF16
    for transport between peers, then reduced in FP32 at the destination. Uses the
    PAT algorithm with Philox-based stochastic rounding.

    Args:
        output: Output tensor (FP32). Will contain the reduced scatter result.
        input: Input tensor (FP32). Size must be output.numel() * world_size.
        op: Reduction operation (ReduceOp.SUM or ReduceOp.AVG).
        seed: A single-element int64 CUDA tensor containing the Philox RNG seed.
        group: Process group to use. Defaults to the default group.
        async_op: Whether to perform the operation asynchronously.

    Returns:
        TorchWork object for tracking operation completion.
    """
    if not torchcomms_is_enabled():
        raise AssertionError(
            "reduce_scatter_quantized requires torchcomms to be enabled"
        )

    pg = get_group(group)
    tc = get_torchcomms_instance(pg, device_type="cuda")

    # Verify backend is ncclx
    backend_name = tc.get_backend()
    if backend_name != "ncclx":
        raise AssertionError(
            f"reduce_scatter_quantized requires ncclx backend, got {backend_name}"
        )

    ncclx_backend = tc.unsafe_get_backend()

    # Capability check: verify method exists
    if not hasattr(ncclx_backend, "reduce_scatter_quantized"):
        raise AssertionError(
            "ncclx_backend does not implement reduce_scatter_quantized"
        )

    return ncclx_backend.reduce_scatter_quantized(
        output,
        input,
        _convert_reduce_op(op),
        seed,
        async_op,
    )
