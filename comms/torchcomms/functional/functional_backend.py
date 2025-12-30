# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Register torchcomms as a backend for torch.distributed._functional_collectives.

This module patches the functional collectives in PyTorch to dispatch to torchcomms
when the DeviceMesh has a torchcomms comm object registered.

Usage:
    from torchcomms.functional.functional_backend import register_torchcomms_funcols_impl
    register_torchcomms_funcols_impl()  # Call once at startup

After registration, calling functional collectives with a DeviceMesh that has
a torchcomms comm object will automatically dispatch to torchcomms:

    mesh = DeviceMesh("cuda", [[0, 1], [2, 3]])
    mesh.set_comm_object("torchcomms", comm, mesh_dim=0)

    # This will now use torchcomms instead of c10d
    result = torch.distributed._functional_collectives.all_reduce(tensor, "sum", (mesh, 0))
"""

import logging
from typing import Callable

import torch
from torch.distributed.device_mesh import DeviceMesh
from torchcomms._comms import ReduceOp, TorchComm
from torchcomms.coalescing import coalesce
from torchcomms.functional.async_tensor import (
    _are_we_tracing,
    _maybe_wrap_tensor,
    _OnceWaitWork,
)

# Mapping from string reduce op to ReduceOp enum
_REDUCE_OP_MAP = {
    "sum": ReduceOp.SUM,
    "avg": ReduceOp.AVG,
    "product": ReduceOp.PRODUCT,
    "min": ReduceOp.MIN,
    "max": ReduceOp.MAX,
}


def _get_reduce_op(reduce_op: str):
    """Convert string reduce op to ReduceOp enum."""
    op = _REDUCE_OP_MAP.get(reduce_op.lower())
    if op is None:
        raise ValueError(f"Unknown reduce operation: {reduce_op}")
    return op


logger = logging.getLogger(__name__)


_registered = False


def _get_comm(mesh: DeviceMesh, mesh_dim: int) -> TorchComm:
    """Get torchcomms comm object from mesh."""
    comm: TorchComm | None = mesh.get_comm_object("torchcomms", mesh_dim)  # type: ignore
    if comm is None:
        raise RuntimeError(f"No torchcomms comm object found for mesh_dim={mesh_dim}")
    return comm


# Registry of functional collective name -> implementation
# Each implementation takes (tensor, *args, mesh, mesh_dim) and returns tensor
_FUNCTIONAL_COLLECTIVE_IMPLS: dict[str, Callable[..., torch.Tensor]] = {}


def register_functional_collective(name: str) -> Callable:
    """Decorator to register a functional collective implementation."""

    def decorator(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        _FUNCTIONAL_COLLECTIVE_IMPLS[name] = fn
        return fn

    return decorator


@register_functional_collective("broadcast")
def _broadcast(
    tensor: torch.Tensor,
    src: int,
    mesh: DeviceMesh,
    mesh_dim: int,
) -> torch.Tensor:
    """Torchcomms implementation of broadcast for functional collectives."""
    comm = _get_comm(mesh, mesh_dim)
    work = comm.broadcast(tensor, src, async_op=True)
    return _maybe_wrap_tensor(tensor, work)


@register_functional_collective("scatter")
def _scatter(
    output: torch.Tensor,
    scatter_list: list[torch.Tensor],
    src: int,
    mesh: DeviceMesh,
    mesh_dim: int,
) -> torch.Tensor:
    """Torchcomms implementation of scatter for functional collectives."""
    comm = _get_comm(mesh, mesh_dim)
    work = (
        comm.scatter(output, scatter_list, root=src, async_op=True)
        if comm.get_rank() == src
        else comm.scatter(output, [], root=src, async_op=True)
    )
    return _maybe_wrap_tensor(output, work)


@register_functional_collective("gather")
def _gather(
    input: torch.Tensor,
    gather_list: list[torch.Tensor],
    dst: int,
    mesh: DeviceMesh,
    mesh_dim: int,
) -> list[torch.Tensor]:
    """Torchcomms implementation of gather for functional collectives."""
    comm = _get_comm(mesh, mesh_dim)
    rank = comm.get_rank()
    group_size = comm.get_size()
    use_dummy_out = False

    # On dst rank: need gather_list with group_size tensors to receive into
    # On non-dst ranks: gather_list should be empty
    if rank == dst:
        if not gather_list:
            gather_list = [torch.empty_like(input) for _ in range(group_size)]
    elif _are_we_tracing():
        # if we are tracing, we can create a dummy to connect
        # the collective and the wait operation
        gather_list = [torch.empty(0, device=input.device, dtype=input.dtype)]
        use_dummy_out = True
    elif not input.requires_grad:
        # we can't always get rid of gather_list since we need
        # the relationship between input/output when
        # creating the autograd graph so that the
        # bw scatter doesn't dedlock
        gather_list = []

    work = comm.gather(
        gather_list,
        input,
        root=dst,
        async_op=True,
    )

    if work is not None:
        # if we needed a dummy to connect the gather and the wait
        # during tracing, then just wait and return []...
        # passes should drop the wait as much as possible.
        if use_dummy_out:
            work.wait()
            return []

        # Autograd wrapper returned the tensor with grad_fn - use it
        # note that it's already been wrapped internally using
        # _wrap_result_with_registered_work
        if isinstance(work, tuple):
            return list(work)
        elif isinstance(work, list):
            return work

        work = _OnceWaitWork(work)

    return [_maybe_wrap_tensor(t, work) for t in gather_list]


@register_functional_collective("all_reduce_coalesced")
def _all_reduce_coalesced(
    tensors: list[torch.Tensor],
    reduce_op: str,
    mesh: DeviceMesh,
    mesh_dim: int,
) -> list[torch.Tensor]:
    """Torchcomms implementation of all_reduce_coalesced for functional collectives."""

    comm = _get_comm(mesh, mesh_dim)
    op = _get_reduce_op(reduce_op)

    if not tensors:
        return []

    with coalesce(comm) as cm:
        for tensor in tensors:
            comm.all_reduce(tensor, op, async_op=True)

    if _are_we_tracing():
        cm.wait()
        return tensors

    work = None
    if cm.work is not None:
        work = _OnceWaitWork(cm.work)

    return [_maybe_wrap_tensor(tensor, work) for tensor in tensors]


@register_functional_collective("all_gather_into_tensor_coalesced")
def _all_gather_into_tensor_coalesced(
    tensors: list[torch.Tensor],
    mesh: DeviceMesh,
    mesh_dim: int,
) -> list[torch.Tensor]:
    """Torchcomms implementation of all_gather_into_tensor_coalesced for functional collectives."""

    comm = _get_comm(mesh, mesh_dim)
    group_size = comm.get_size()

    if not tensors:
        return []

    # Allocate output tensors (group_size times larger in first dim)
    outputs = []
    for tensor in tensors:
        out_shape = list(tensor.shape)
        out_shape[0] *= group_size
        outputs.append(tensor.new_empty(out_shape))

    with coalesce(comm) as cm:
        for output, tensor in zip(outputs, tensors):
            comm.all_gather_single(output, tensor, async_op=True)

    if _are_we_tracing():
        cm.wait()
        return outputs

    work = None
    if cm.work is not None:
        work = _OnceWaitWork(cm.work)

    return [_maybe_wrap_tensor(tensor, work) for tensor in outputs]


@register_functional_collective("reduce_scatter_tensor_coalesced")
def _reduce_scatter_tensor_coalesced(
    tensors: list[torch.Tensor],
    reduce_op: str,
    scatter_dims: list[int],
    mesh: DeviceMesh,
    mesh_dim: int,
) -> list[torch.Tensor]:
    """Torchcomms implementation of reduce_scatter_tensor_coalesced for functional collectives."""

    comm = _get_comm(mesh, mesh_dim)
    group_size = comm.get_size()
    op = _get_reduce_op(reduce_op)

    if not tensors:
        return []

    # Prepare input tensors (handle non-zero scatter_dims) and allocate outputs
    inputs = []
    outputs = []
    for tensor, scatter_dim in zip(tensors, scatter_dims):
        if scatter_dim != 0:
            # Rechunk along scatter_dim and concatenate along dim 0
            tensor_list = torch.chunk(tensor, group_size, dim=scatter_dim)
            tensor = torch.cat(tensor_list)
        inputs.append(tensor)
        # Output is 1/group_size of input along dim 0
        out_shape = list(tensor.shape)
        out_shape[0] //= group_size
        outputs.append(tensor.new_empty(out_shape))

    with coalesce(comm) as cm:
        for output, input_tensor in zip(outputs, inputs):
            comm.reduce_scatter_single(output, input_tensor, op, async_op=True)

    if _are_we_tracing():
        cm.wait()
        return outputs

    work = None
    if cm.work is not None:
        work = _OnceWaitWork(cm.work)

    return [_maybe_wrap_tensor(tensor, work) for tensor in outputs]


# =============================================================================
# Registration
# =============================================================================


@register_functional_collective("all_reduce")
def _all_reduce_funcol(
    tensor: torch.Tensor, reduce_op: str, mesh: "DeviceMesh", mesh_dim: int
) -> torch.Tensor:
    """all_reduce: funcol(tensor, reduce_op, mesh, mesh_dim)"""
    comm = _get_comm(mesh, mesh_dim)
    op = _get_reduce_op(reduce_op)

    # The patched comm.all_reduce will dispatch to functional op when requires_grad=True or FakeTensor
    work = comm.all_reduce(tensor, op, async_op=True)
    return _maybe_wrap_tensor(tensor, work)


@register_functional_collective("all_gather_tensor")
def _all_gather_tensor_funcol(
    tensor: torch.Tensor, gather_dim: int, mesh: "DeviceMesh", mesh_dim: int
) -> torch.Tensor:
    """all_gather_tensor: funcol(tensor, gather_dim, mesh, mesh_dim)"""
    comm = _get_comm(mesh, mesh_dim)
    group_size = comm.get_size()

    # Create output tensor with correct shape for all_gather_single (always gathers on dim 0)
    out_size = list(tensor.size())
    out_size[0] *= group_size
    output = tensor.new_empty(out_size)

    # The patched comm.all_gather_single will dispatch to functional op when requires_grad=True
    # or when FakeTensor/FunctionalTensor is detected (compilation context)
    work = comm.all_gather_single(output, tensor, async_op=True)

    # Rearrange if gather_dim != 0
    if gather_dim != 0:
        work.wait()
        output = torch.cat(torch.chunk(output, group_size, dim=0), dim=gather_dim)
        return output

    return _maybe_wrap_tensor(output, work)


@register_functional_collective("reduce_scatter_tensor")
def _reduce_scatter_tensor_funcol(
    tensor: torch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    mesh: "DeviceMesh",
    mesh_dim: int,
) -> torch.Tensor:
    """reduce_scatter_tensor: funcol(tensor, reduce_op, scatter_dim, mesh, mesh_dim)"""
    comm = _get_comm(mesh, mesh_dim)
    group_size = comm.get_size()
    op = _get_reduce_op(reduce_op)

    # Handle non-zero scatter_dim by rearranging first
    input_tensor = tensor
    if scatter_dim != 0:
        tensor_list = torch.chunk(tensor, group_size, dim=scatter_dim)
        input_tensor = torch.cat(tensor_list, dim=0)

    # Create output tensor with correct shape
    out_size = list(input_tensor.size())
    out_size[0] //= group_size
    output = input_tensor.new_empty(out_size)

    # The patched comm.reduce_scatter_single will dispatch to functional op when requires_grad=True
    # or when FakeTensor/FunctionalTensor is detected (compilation context)
    work = comm.reduce_scatter_single(output, input_tensor, op, async_op=True)
    return _maybe_wrap_tensor(output, work)


@register_functional_collective("all_to_all_single")
def _all_to_all_single_funcol(
    tensor: torch.Tensor,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    mesh: "DeviceMesh",
    mesh_dim: int,
) -> torch.Tensor:
    """all_to_all_single: funcol(tensor, output_split_sizes, input_split_sizes, mesh, mesh_dim)"""
    comm = _get_comm(mesh, mesh_dim)
    group_size = comm.get_size()

    # For uniform splits (None, None), use the simple op
    if output_split_sizes is None and input_split_sizes is None:
        output = tensor.new_empty(tensor.size())

        # The patched comm.all_to_all_single will dispatch to functional op when requires_grad=True
        # or when FakeTensor/FunctionalTensor is detected (compilation context)
        work = comm.all_to_all_single(output, tensor, async_op=True)
        return _maybe_wrap_tensor(output, work)

    # For variable splits, use the _v variant
    # Handle None split sizes (equal splits)
    if output_split_sizes is None:
        output_split_sizes = [tensor.shape[0] // group_size] * group_size
    if input_split_sizes is None:
        input_split_sizes = [tensor.shape[0] // group_size] * group_size

    # Create output tensor
    out_size = list(tensor.size())
    out_size[0] = sum(output_split_sizes)
    output = tensor.new_empty(out_size)

    # The patched comm.all_to_all_v_single will dispatch to functional op when requires_grad=True
    # or when FakeTensor/FunctionalTensor is detected (compilation context)
    work = comm.all_to_all_v_single(
        output, tensor, output_split_sizes, input_split_sizes, async_op=True
    )

    return _maybe_wrap_tensor(output, work)


def register_torchcomms_funcols_impl() -> None:
    """
    Register torchcomms as a backend for torch.distributed._functional_collectives.

    After calling this function, functional collectives will dispatch to torchcomms
    when the group is a (DeviceMesh, int) tuple and the mesh has a torchcomms
    comm object registered for that dimension.

    This function is idempotent and can be called multiple times safely.
    """
    global _registered
    if _registered:
        return

    try:
        from torch.distributed._functional_collectives import (
            register_backend_collectives,
        )
    except ImportError:
        logger.warning(
            "torch.distributed._functional_collectives.register_backend_collectives "
            "not available, skipping torchcomms backend registration"
        )
        return

    # Register with PyTorch's functional collectives backend system
    # Implementations are registered at module level via @register_functional_collective
    register_backend_collectives("torchcomms", _FUNCTIONAL_COLLECTIVE_IMPLS)

    logger.info(
        f"Registered torchcomms backend with {len(_FUNCTIONAL_COLLECTIVE_IMPLS)} collectives: "
        f"{list(_FUNCTIONAL_COLLECTIVE_IMPLS.keys())}"
    )

    _registered = True
