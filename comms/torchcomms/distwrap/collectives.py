# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp
from torchcomms import objcol, ReduceOp as TcReduceOp, TorchComm
from torchcomms.distwrap.fallback import (
    fallback_all_gather_gloo,
    fallback_all_to_all_gloo,
    fallback_all_to_all_single_gloo,
)
from torchcomms.distwrap.pginfo import pg_info_assert_registered, pg_info_get_data
from torchcomms.distwrap.utils import (
    get_backend_for_device,
    get_group,
    get_group_rank,
    get_torchcomms_instance,
    torchcomms_is_enabled,
)


# =============================================================================
# Module-level State
# =============================================================================

# Dictionary mapping dist PREMUL_SUM ops to their torchcomms equivalents
_PREMUL_SUM_OPS: dict[ReduceOp, TcReduceOp] = {}


# =============================================================================
# Private Helper Functions
# =============================================================================


def _convert_reduce_op(op: ReduceOp) -> TcReduceOp:
    """Convert torch.distributed.ReduceOp to torchcomms.ReduceOp."""
    # Check for PREMUL_SUM ops first
    if op in _PREMUL_SUM_OPS:
        return _PREMUL_SUM_OPS[op]

    # Map from torch.distributed ReduceOp to torchcomms ReduceOp
    op_map = {
        ReduceOp.SUM: TcReduceOp.SUM,
        ReduceOp.PRODUCT: TcReduceOp.PRODUCT,
        ReduceOp.MIN: TcReduceOp.MIN,
        ReduceOp.MAX: TcReduceOp.MAX,
        ReduceOp.BAND: TcReduceOp.BAND,
        ReduceOp.BOR: TcReduceOp.BOR,
        ReduceOp.BXOR: TcReduceOp.BXOR,
        ReduceOp.AVG: TcReduceOp.AVG,
    }
    if op in op_map:
        return op_map[op]
    raise ValueError(f"Unsupported ReduceOp: {op}")


def _get_default_torchcomms_instance(pg: ProcessGroup) -> TorchComm:
    """
    Get a default torchcomms instance for the given process group.

    This is used for operations that don't have a tensor to infer the device
    from (e.g., object collectives, barrier).

    Args:
        pg: The process group.

    Returns:
        A torchcomms instance.

    Raises:
        AssertionError: If no torchcomms instances found for the process group.
    """
    torchcomms_instances = pg_info_get_data(pg, "torchcomms")
    if not torchcomms_instances:
        raise AssertionError(f"No torchcomms instances found for process group {pg}")
    # Use the first available torchcomms instance
    return next(iter(torchcomms_instances.values()))


# =============================================================================
# Public API Functions
# =============================================================================


def _make_nccl_premul_sum(mul_factor: float | torch.Tensor) -> ReduceOp:
    """
    Create a NCCL PREMUL_SUM reduce operation with the given multiplication factor.

    This creates a reduce operation that multiplies all inputs by mul_factor before
    summing them. When torchcomms is enabled, it also registers the corresponding
    torchcomms reduce operation.

    Args:
        mul_factor: The multiplication factor (scalar or tensor).

    Returns:
        A ReduceOp that can be used with all_reduce, reduce_scatter, etc.
    """
    op = dist._make_nccl_premul_sum(mul_factor)  # type: ignore[attr-defined]
    if torchcomms_is_enabled():
        tc_op = TcReduceOp.PREMUL_SUM(mul_factor)  # type: ignore[attr-defined]
        _PREMUL_SUM_OPS[op] = tc_op
    return op


def all_reduce(
    tensor: torch.Tensor,
    op: ReduceOp = ReduceOp.SUM,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> dist.Work | None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        tc = get_torchcomms_instance(pg, tensor=tensor)
        tc_op = _convert_reduce_op(op)
        work = tc.all_reduce(tensor, tc_op, async_op)
        return work if async_op else None
    else:
        return dist.all_reduce(tensor, op, pg, async_op)


def broadcast(
    tensor: torch.Tensor,
    src: int | None = None,
    group: ProcessGroup | None = None,
    async_op: bool = False,
    group_src: int | None = None,
) -> dist.Work | None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        # Resolve group_src for torchcomms
        if group_src is None:
            if src is None:
                raise ValueError("Either src or group_src must be specified")
            group_src = get_group_rank(pg, src)
        elif src is not None:
            raise ValueError("Cannot specify both src and group_src")

        tc = get_torchcomms_instance(pg, tensor=tensor)
        work = tc.broadcast(tensor, group_src, async_op)
        return work if async_op else None
    else:
        return dist.broadcast(
            tensor, src=src, group=pg, async_op=async_op, group_src=group_src
        )


def reduce(
    tensor: torch.Tensor,
    dst: int | None = None,
    op: ReduceOp = ReduceOp.SUM,
    group: ProcessGroup | None = None,
    async_op: bool = False,
    group_dst: int | None = None,
) -> dist.Work | None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        # Resolve group_dst for torchcomms
        if group_dst is None:
            if dst is None:
                raise ValueError("Either dst or group_dst must be specified")
            group_dst = get_group_rank(pg, dst)
        elif dst is not None:
            raise ValueError("Cannot specify both dst and group_dst")

        tc = get_torchcomms_instance(pg, tensor=tensor)
        tc_op = _convert_reduce_op(op)
        work = tc.reduce(tensor, group_dst, tc_op, async_op)
        return work if async_op else None
    else:
        return dist.reduce(
            tensor, dst=dst, op=op, group=pg, async_op=async_op, group_dst=group_dst
        )


def all_gather(
    tensor_list: list[torch.Tensor],
    tensor: torch.Tensor,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> dist.Work | None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        tc = get_torchcomms_instance(pg, tensor=tensor)
        # Check if tensors have variable sizes (need all_gather_v)
        sizes = [t.numel() for t in tensor_list]
        if len(set(sizes)) > 1:
            # Variable-size all_gather
            work = tc.all_gather_v(tensor_list, tensor, async_op)
        else:
            work = tc.all_gather(tensor_list, tensor, async_op)
        return work if async_op else None
    else:
        # Check if tensors have variable sizes - Gloo doesn't support this
        sizes = [t.numel() for t in tensor_list]
        if (
            len(set(sizes)) > 1
            and get_backend_for_device(pg, tensor.device.type) == "gloo"
        ):
            return fallback_all_gather_gloo(tensor_list, tensor, pg, async_op)
        else:
            return dist.all_gather(tensor_list, tensor, pg, async_op)


def all_gather_into_tensor(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> dist.Work | None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        tc = get_torchcomms_instance(pg, tensor=input_tensor)
        work = tc.all_gather_single(output_tensor, input_tensor, async_op)
        return work if async_op else None
    else:
        return dist.all_gather_into_tensor(output_tensor, input_tensor, pg, async_op)


def reduce_scatter(
    output: torch.Tensor,
    input_list: list[torch.Tensor],
    op: ReduceOp = ReduceOp.SUM,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> dist.Work | None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        tc = get_torchcomms_instance(pg, tensor=output)
        tc_op = _convert_reduce_op(op)
        # Check if tensors have variable sizes (need reduce_scatter_v)
        sizes = [t.numel() for t in input_list]
        if len(set(sizes)) > 1:
            # Variable-size reduce_scatter
            work = tc.reduce_scatter_v(output, input_list, tc_op, async_op)
        else:
            work = tc.reduce_scatter(output, input_list, tc_op, async_op)
        return work if async_op else None
    else:
        return dist.reduce_scatter(output, input_list, op, pg, async_op)


def reduce_scatter_tensor(
    output: torch.Tensor,
    input: torch.Tensor,
    op: ReduceOp = ReduceOp.SUM,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> dist.Work | None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        tc = get_torchcomms_instance(pg, tensor=output)
        tc_op = _convert_reduce_op(op)
        work = tc.reduce_scatter_single(output, input, tc_op, async_op)
        return work if async_op else None
    else:
        return dist.reduce_scatter_tensor(output, input, op, pg, async_op)


def all_to_all(
    output_tensor_list: list[torch.Tensor],
    input_tensor_list: list[torch.Tensor],
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> dist.Work | None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        tc = get_torchcomms_instance(pg, tensor=input_tensor_list[0])
        work = tc.all_to_all(output_tensor_list, input_tensor_list, async_op)
        return work if async_op else None
    else:
        # Gloo doesn't support all_to_all
        if get_backend_for_device(pg, input_tensor_list[0].device.type) == "gloo":
            return fallback_all_to_all_gloo(
                output_tensor_list, input_tensor_list, pg, async_op
            )
        else:
            return dist.all_to_all(output_tensor_list, input_tensor_list, pg, async_op)


def all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: list[int] | None = None,
    input_split_sizes: list[int] | None = None,
    group: ProcessGroup | None = None,
    async_op: bool = False,
) -> dist.Work | None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        tc = get_torchcomms_instance(pg, tensor=input)
        if output_split_sizes is None and input_split_sizes is None:
            work = tc.all_to_all_single(output, input, async_op)
        else:
            work = tc.all_to_all_v_single(
                output,
                input,
                output_split_sizes or [],
                input_split_sizes or [],
                async_op,
            )
        return work if async_op else None
    else:
        # Gloo doesn't support all_to_all_single
        if get_backend_for_device(pg, input.device.type) == "gloo":
            return fallback_all_to_all_single_gloo(
                output, input, output_split_sizes, input_split_sizes, pg, async_op
            )
        else:
            return dist.all_to_all_single(
                output, input, output_split_sizes, input_split_sizes, pg, async_op
            )


def scatter(
    tensor: torch.Tensor,
    scatter_list: list[torch.Tensor] | None = None,
    src: int | None = None,
    group: ProcessGroup | None = None,
    async_op: bool = False,
    group_src: int | None = None,
) -> dist.Work | None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        # Resolve group_src for torchcomms
        if group_src is None:
            if src is None:
                raise ValueError("Either src or group_src must be specified")
            group_src = get_group_rank(pg, src)
        elif src is not None:
            raise ValueError("Cannot specify both src and group_src")

        tc = get_torchcomms_instance(pg, tensor=tensor)
        work = tc.scatter(tensor, scatter_list or [], group_src, async_op)
        return work if async_op else None
    else:
        return dist.scatter(
            tensor,
            scatter_list,
            src=src,
            group=pg,
            async_op=async_op,
            group_src=group_src,
        )


def gather(
    tensor: torch.Tensor,
    gather_list: list[torch.Tensor] | None = None,
    dst: int | None = None,
    group: ProcessGroup | None = None,
    async_op: bool = False,
    group_dst: int | None = None,
) -> dist.Work | None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        # Resolve group_dst for torchcomms
        if group_dst is None:
            if dst is None:
                raise ValueError("Either dst or group_dst must be specified")
            group_dst = get_group_rank(pg, dst)
        elif dst is not None:
            raise ValueError("Cannot specify both dst and group_dst")

        tc = get_torchcomms_instance(pg, tensor=tensor)
        work = tc.gather(gather_list or [], tensor, group_dst, async_op)
        return work if async_op else None
    else:
        return dist.gather(
            tensor,
            gather_list,
            dst=dst,
            group=pg,
            async_op=async_op,
            group_dst=group_dst,
        )


def barrier(
    group: ProcessGroup | None = None,
    async_op: bool = False,
    device_ids: list[int] | None = None,
) -> dist.Work | None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        tc = _get_default_torchcomms_instance(pg)
        work = tc.barrier(async_op)
        # pyre-ignore[7]: TorchWork is compatible with Work
        return work if async_op else None
    else:
        return dist.barrier(pg, async_op, device_ids)


def send(
    tensor: torch.Tensor,
    dst: int | None = None,
    group: ProcessGroup | None = None,
    tag: int = 0,
    group_dst: int | None = None,
) -> None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        # Resolve group_dst for torchcomms
        if group_dst is None:
            if dst is None:
                raise ValueError("Either dst or group_dst must be specified")
            group_dst = get_group_rank(pg, dst)
        elif dst is not None:
            raise ValueError("Cannot specify both dst and group_dst")

        tc = get_torchcomms_instance(pg, tensor=tensor)
        work = tc.send(tensor, group_dst, False)
        work.wait()
    else:
        dist.send(tensor, dst=dst, group=pg, tag=tag, group_dst=group_dst)


def recv(
    tensor: torch.Tensor,
    src: int | None = None,
    group: ProcessGroup | None = None,
    tag: int = 0,
    group_src: int | None = None,
) -> int:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        # Resolve group_src for torchcomms
        if group_src is None:
            if src is not None:
                group_src = get_group_rank(pg, src)
            else:
                raise ValueError("torchcomms does not support wildcard recv (src=None)")
        elif src is not None:
            raise ValueError("Cannot specify both src and group_src")

        tc = get_torchcomms_instance(pg, tensor=tensor)
        work = tc.recv(tensor, group_src, False)
        work.wait()
        return src if src is not None else dist.get_process_group_ranks(pg)[group_src]
    else:
        return dist.recv(tensor, src=src, group=pg, tag=tag, group_src=group_src)


def isend(
    tensor: torch.Tensor,
    dst: int | None = None,
    group: ProcessGroup | None = None,
    tag: int = 0,
    group_dst: int | None = None,
) -> dist.Work:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        # Resolve group_dst for torchcomms
        if group_dst is None:
            if dst is None:
                raise ValueError("Either dst or group_dst must be specified")
            group_dst = get_group_rank(pg, dst)
        elif dst is not None:
            raise ValueError("Cannot specify both dst and group_dst")

        tc = get_torchcomms_instance(pg, tensor=tensor)
        return tc.send(tensor, group_dst, True)
    else:
        work = dist.isend(tensor, dst=dst, group=pg, tag=tag, group_dst=group_dst)
        if work is None:
            raise AssertionError("dist.isend returned None")
        return work


def irecv(
    tensor: torch.Tensor,
    src: int | None = None,
    group: ProcessGroup | None = None,
    tag: int = 0,
    group_src: int | None = None,
) -> dist.Work:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        # Resolve group_src for torchcomms
        if group_src is None:
            if src is not None:
                group_src = get_group_rank(pg, src)
            else:
                raise ValueError(
                    "torchcomms does not support wildcard irecv (src=None)"
                )
        elif src is not None:
            raise ValueError("Cannot specify both src and group_src")

        tc = get_torchcomms_instance(pg, tensor=tensor)
        return tc.recv(tensor, group_src, True)
    else:
        work = dist.irecv(tensor, src=src, group=pg, tag=tag, group_src=group_src)
        if work is None:
            raise AssertionError("dist.irecv returned None")
        return work


def batch_isend_irecv(p2p_op_list: list[Any]) -> list[dist.Work]:
    if torchcomms_is_enabled():
        works: list[dist.Work] = []
        for p2p_op in p2p_op_list:
            # Handle both distwrap P2POp and torch.distributed.P2POp
            actual_op = getattr(p2p_op, "_p2p_op", p2p_op)
            pg = get_group(actual_op.group)
            pg_info_assert_registered(pg)
            tc = get_torchcomms_instance(pg, tensor=actual_op.tensor)
            op_name = actual_op.op.__name__

            # Get group_peer, converting from global peer if needed
            group_peer = actual_op.group_peer
            if group_peer is None:
                if actual_op.peer is not None:
                    group_peer = get_group_rank(pg, actual_op.peer)
                else:
                    raise ValueError(
                        "torchcomms does not support wildcard P2P operations "
                        "(peer=None)"
                    )

            if "send" in op_name:
                work = tc.send(actual_op.tensor, group_peer, True)
            elif "recv" in op_name:
                work = tc.recv(actual_op.tensor, group_peer, True)
            else:
                raise ValueError(f"Unknown P2P operation: {op_name}")
            works.append(work)
        return works
    # Unwrap distwrap P2POp objects to torch.distributed.P2POp
    unwrapped_ops = [getattr(op, "_p2p_op", op) for op in p2p_op_list]
    works = dist.batch_isend_irecv(unwrapped_ops)
    if works is None:
        raise AssertionError("dist.batch_isend_irecv returned None")
    return works


def all_gather_object(
    object_list: list[Any],
    obj: Any,
    group: ProcessGroup | None = None,
) -> None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        tc = _get_default_torchcomms_instance(pg)
        objcol.all_gather_object(tc, object_list, obj)
    else:
        dist.all_gather_object(object_list, obj, pg)


def gather_object(
    obj: Any,
    object_gather_list: list[Any] | None = None,
    dst: int | None = None,
    group: ProcessGroup | None = None,
    group_dst: int | None = None,
) -> None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        # Resolve group_dst for torchcomms
        if group_dst is None:
            if dst is None:
                raise ValueError("Either dst or group_dst must be specified")
            group_dst = get_group_rank(pg, dst)
        elif dst is not None:
            raise ValueError("Cannot specify both dst and group_dst")

        tc = _get_default_torchcomms_instance(pg)
        objcol.gather_object(
            tc, obj, root=group_dst, object_gather_list=object_gather_list
        )
    else:
        dist.gather_object(
            obj, object_gather_list, dst=dst, group=pg, group_dst=group_dst
        )


def scatter_object_list(
    scatter_object_output_list: list[Any],
    scatter_object_input_list: list[Any] | None = None,
    src: int | None = None,
    group: ProcessGroup | None = None,
    group_src: int | None = None,
) -> None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        # Resolve group_src for torchcomms
        if group_src is None:
            if src is None:
                raise ValueError("Either src or group_src must be specified")
            group_src = get_group_rank(pg, src)
        elif src is not None:
            raise ValueError("Cannot specify both src and group_src")

        tc = _get_default_torchcomms_instance(pg)
        objcol.scatter_object_list(
            tc,
            root=group_src,
            scatter_object_output_list=scatter_object_output_list,
            scatter_object_input_list=scatter_object_input_list,
        )
    else:
        dist.scatter_object_list(
            scatter_object_output_list,
            scatter_object_input_list,
            src=src,
            group=pg,
            group_src=group_src,
        )


def broadcast_object_list(
    object_list: list[Any],
    src: int | None = None,
    group: ProcessGroup | None = None,
    device: torch.device | None = None,
    group_src: int | None = None,
) -> None:
    pg = get_group(group)
    pg_info_assert_registered(pg)

    if torchcomms_is_enabled():
        # Resolve group_src for torchcomms
        if group_src is None:
            if src is None:
                raise ValueError("Either src or group_src must be specified")
            group_src = get_group_rank(pg, src)
        elif src is not None:
            raise ValueError("Cannot specify both src and group_src")

        tc = _get_default_torchcomms_instance(pg)
        objcol.broadcast_object_list(tc, object_list, root=group_src)
    else:
        dist.broadcast_object_list(
            object_list, src=src, group=pg, device=device, group_src=group_src
        )


# =============================================================================
# Memory Pool Operations
# =============================================================================


def get_mem_allocator(
    group: ProcessGroup | None,
    device: torch.device,
) -> Any:
    """
    Get the memory allocator for the given process group and device.

    Args:
        group: The process group (None for WORLD).
        device: The device to get the allocator for.

    Returns:
        The memory allocator for the specified device.
    """
    pg = get_group(group)

    if torchcomms_is_enabled():
        tc = get_torchcomms_instance(pg, device_type=device.type)
        return tc.mem_allocator
    else:
        return pg._get_backend(device).mem_allocator  # type: ignore[union-attr]


def register_mem_pool(
    group: ProcessGroup | None,
    device: torch.device,
    pool: Any,
) -> None:
    """
    Register a memory pool with the process group backend.

    Note: This is a no-op when torchcomms is enabled, as torchcomms manages
    its own memory pools.

    Args:
        group: The process group (None for WORLD).
        device: The device to register the pool for.
        pool: The memory pool to register.
    """
    # We don't need to register the mem pool for torchcomms
    if not torchcomms_is_enabled():
        pg = get_group(group)
        pg._get_backend(device).register_mem_pool(pool)  # type: ignore[union-attr]
