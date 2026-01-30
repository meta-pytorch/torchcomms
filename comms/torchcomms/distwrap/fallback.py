# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""
Fallback implementations for collectives that are not supported by certain backends.

These functions provide software implementations of collectives using simpler
primitives when the native backend implementation is not available.
"""

from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


def fallback_split_group_new_group(
    parent_pg: ProcessGroup,
    my_group_ranks: list[int],
    timeout: timedelta | None,
    pg_options: Any | None,
    group_desc: str | None,
) -> ProcessGroup:
    my_rank = dist.get_rank(parent_pg)

    # Only the smallest rank in each group provides the list
    if my_rank == min(my_group_ranks):
        my_contribution: list[int] | None = my_group_ranks
    else:
        my_contribution = None

    # All-gather to get all unique split lists
    gathered_lists: list[list[int] | None] = [None] * dist.get_world_size(parent_pg)
    dist.all_gather_object(gathered_lists, my_contribution, group=parent_pg)

    # Filter out Nones to get unique split lists
    all_split_ranks = [r for r in gathered_lists if r is not None]

    # Create process groups for each split
    new_pg = None
    for rank_list in all_split_ranks:
        if my_rank in rank_list:
            new_pg = dist.new_group(
                ranks=rank_list,
                timeout=timeout,
                pg_options=pg_options,
                group_desc=group_desc,
            )
        else:
            # Other ranks still need to participate in the collective
            dist.new_group(
                ranks=rank_list,
                timeout=timeout,
                pg_options=pg_options,
                group_desc=group_desc,
            )

    if new_pg is None:
        raise AssertionError("Failed to create process group for this rank")

    return new_pg


def fallback_all_gather_gloo(
    tensor_list: list[torch.Tensor],
    tensor: torch.Tensor,
    pg: ProcessGroup,
    async_op: bool,
) -> dist.Work | None:
    """Implement all_gather using all_to_all for gloo backend."""
    # For all_gather, each rank sends its tensor to all ranks
    # Create input list where every element is the same tensor
    input_tensor_list = [tensor] * len(tensor_list)
    return fallback_all_to_all_gloo(tensor_list, input_tensor_list, pg, async_op)


def fallback_all_to_all_gloo(
    output_tensor_list: list[torch.Tensor],
    input_tensor_list: list[torch.Tensor],
    pg: ProcessGroup,
    async_op: bool,
) -> dist.Work | None:
    """Implement all_to_all using batched send/recv for gloo backend."""
    my_rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)

    # Local copy: my own data stays local
    output_tensor_list[my_rank].copy_(input_tensor_list[my_rank])

    # Build list of P2P operations for all remote exchanges
    p2p_ops = []
    for i in range(world_size):
        if i == my_rank:
            continue
        # Receive from rank i into output_tensor_list[i]
        p2p_ops.append(dist.P2POp(dist.irecv, output_tensor_list[i], i, pg))
        # Send to rank i from input_tensor_list[i]
        p2p_ops.append(dist.P2POp(dist.isend, input_tensor_list[i], i, pg))

    if p2p_ops:
        reqs = dist.batch_isend_irecv(p2p_ops)
        for req in reqs[:-1]:
            req.wait()
        if async_op:
            return reqs[-1]
        else:
            reqs[-1].wait()

    return None


def fallback_all_to_all_single_gloo(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    pg: ProcessGroup,
    async_op: bool,
) -> dist.Work | None:
    """Implement all_to_all_single using batched send/recv for gloo backend."""
    my_rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)

    if input_split_sizes is None:
        in_split = input.numel() // world_size
        input_split_sizes = [in_split] * world_size
    if output_split_sizes is None:
        out_split = output.numel() // world_size
        output_split_sizes = [out_split] * world_size

    # Compute offsets for input and output tensors
    input_offsets = [0]
    for size in input_split_sizes[:-1]:
        input_offsets.append(input_offsets[-1] + size)
    output_offsets = [0]
    for size in output_split_sizes[:-1]:
        output_offsets.append(output_offsets[-1] + size)

    # Local copy: my own data stays local
    local_in_start = input_offsets[my_rank]
    local_in_end = local_in_start + input_split_sizes[my_rank]
    local_out_start = output_offsets[my_rank]
    local_out_end = local_out_start + output_split_sizes[my_rank]
    output[local_out_start:local_out_end].copy_(input[local_in_start:local_in_end])

    # Build list of P2P operations for all remote exchanges
    p2p_ops = []
    for i in range(world_size):
        if i == my_rank:
            continue
        # Receive from rank i into output slice for rank i
        out_start = output_offsets[i]
        out_end = out_start + output_split_sizes[i]
        recv_tensor = output[out_start:out_end]
        p2p_ops.append(dist.P2POp(dist.irecv, recv_tensor, i, pg))

        # Send to rank i from input slice for rank i
        in_start = input_offsets[i]
        in_end = in_start + input_split_sizes[i]
        send_tensor = input[in_start:in_end].contiguous()
        p2p_ops.append(dist.P2POp(dist.isend, send_tensor, i, pg))

    if p2p_ops:
        reqs = dist.batch_isend_irecv(p2p_ops)
        for req in reqs[:-1]:
            req.wait()
        if async_op:
            return reqs[-1]
        else:
            reqs[-1].wait()

    return None
