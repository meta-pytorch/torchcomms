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

    # Record shapes for splitting later
    shapes = [t.shape for t in tensors]
    sizes = [t.numel() for t in tensors]

    # Flatten and concatenate all input tensors
    flat_tensors = [t.flatten() for t in tensors]
    concat = torch.cat(flat_tensors)

    work = comm.all_reduce(concat, op, async_op=True)
    if work is not None:
        # Autograd wrapper returned the tensor with grad_fn - use it
        # note that it's already been wrapped internally using
        # _wrap_result_with_registered_work
        if isinstance(work, torch.Tensor):
            concat = work
            work = None
        else:
            work = _OnceWaitWork(work)

    # Split and reshape outputs - wrap each with work handle so first usage triggers wait
    # Note: concat inherits requires_grad from inputs via torch.cat
    outputs = []
    offset = 0
    for shape, size in zip(shapes, sizes):
        # Slicing and view create views (no data access), so wrapping works
        out_flat = concat[offset : offset + size].view(shape)
        outputs.append(_maybe_wrap_tensor(out_flat, work))
        offset += size

    return outputs


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

    # Record sizes for splitting later
    sizes = [t.numel() for t in tensors]

    # Flatten and concatenate all input tensors
    flat_tensors = [t.flatten() for t in tensors]
    input_concat = torch.cat(flat_tensors)

    # Create output tensor (group_size times larger)
    output_concat = input_concat.new_empty(input_concat.numel() * group_size)

    work = comm.all_gather_single(output_concat, input_concat, async_op=True)

    # Wrap output_concat so that subsequent operations trigger the wait
    output_concat = _maybe_wrap_tensor(output_concat, work)

    # Split and reshape outputs
    outputs = []
    gathered_chunks = torch.chunk(output_concat, group_size, dim=0)

    for i, size in enumerate(sizes):
        # Gather the i-th tensor's data from each rank
        offset = sum(sizes[:i])
        tensor_parts = [chunk[offset : offset + size] for chunk in gathered_chunks]
        gathered = torch.cat(tensor_parts)
        # Reshape to match expected output shape
        out_shape = list(tensors[i].shape)
        out_shape[0] *= group_size
        outputs.append(gathered.view(out_shape))

    return outputs


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

    # Handle non-zero scatter_dims and record output sizes
    processed_tensors = []
    output_sizes = []
    for tensor, scatter_dim in zip(tensors, scatter_dims):
        if scatter_dim != 0:
            tensor_list = torch.chunk(tensor, group_size, dim=scatter_dim)
            tensor = torch.cat(tensor_list)
        processed_tensors.append(tensor)
        output_sizes.append(tensor.numel() // group_size)

    # Chunk each tensor by group_size, then interleave so that after reduce_scatter
    # each rank gets [t1_chunk, t2_chunk, ...] for its portion
    chunks_per_tensor = [
        torch.chunk(t.flatten(), group_size, dim=0) for t in processed_tensors
    ]

    # Interleave: [t1_chunk0, t2_chunk0, t1_chunk1, t2_chunk1, ...]
    interleaved = []
    for rank_idx in range(group_size):
        for tensor_idx in range(len(processed_tensors)):
            interleaved.append(chunks_per_tensor[tensor_idx][rank_idx])

    input_concat = torch.cat(interleaved)

    # Create output tensor - each rank gets sum(output_sizes) elements
    # Inherit requires_grad from inputs
    output_size = sum(output_sizes)
    output_concat = input_concat.new_empty(output_size)

    work = comm.reduce_scatter_single(output_concat, input_concat, op, async_op=True)
    if work is not None:
        if isinstance(work, torch.Tensor):
            # Autograd wrapper returned the tensor with grad_fn - use it
            # note that it's already been wrapped internally using
            # _wrap_result_with_registered_work
            output_concat = work
            work = None
        else:
            work = _OnceWaitWork(work)

    # Split outputs - now output_concat is [t1_chunk, t2_chunk, ...] for this rank
    # Wrap each with work handle so first usage triggers wait
    outputs = []
    offset = 0
    for i, size in enumerate(output_sizes):
        # Slicing and view create views (no data access), so wrapping works
        out_flat = output_concat[offset : offset + size]
        out_shape = list(processed_tensors[i].shape)
        out_shape[0] //= group_size
        outputs.append(_maybe_wrap_tensor(out_flat.view(out_shape), work))
        offset += size

    return outputs


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
