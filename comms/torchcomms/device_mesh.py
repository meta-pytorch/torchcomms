# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe

import math
from typing import Any, cast, Optional

import torch

import torch.distributed as dist
from torch.distributed.device_mesh import _mesh_resources

from torch.distributed.distributed_c10d import GroupName
from torchcomms._comms import _BackendWrapper, _get_store, new_comm, TorchComm

try:
    from torch.distributed.distributed_c10d import GroupName
except ImportError:
    print("GroupName is not available.")
    # Fallback: GroupName is effectively just str when not available from torch
    # We use cast to satisfy type checkers while keeping runtime behavior simple
    GroupName = str  # type: ignore[misc]


def _create_torchcomm_process_group(
    comm: TorchComm,
    group_name: str,
    backend_str: str = "torchcomm",
    prefix_store: Optional[object] = None,
    global_ranks_mapping: Optional[dict[int, int]] = None,
) -> dist.ProcessGroup:
    """
    Helper function to create a ProcessGroup backed by TorchComm and register it
    with the distributed runtime.

    Args:
        comm: TorchComm instance to wrap
        group_name: Name for the process group
        backend_str: Backend string identifier
        prefix_store: Store for the process group (can be None)
        global_ranks_mapping: Mapping from global rank to group rank

    Returns:
        The created and registered ProcessGroup instance
    """
    # Make the linter happy. GroupName is just an alias for str. The cost of
    # this conversion is negligible.
    group_name = GroupName(group_name)

    wrapper = _BackendWrapper(comm)  # noqa: F405
    backend_type = dist.ProcessGroup.BackendType.CUSTOM  # noqa: F841
    backend_config = dist.BackendConfig(dist.Backend(backend_str))

    # Create process group
    # pyre-fixme[6]: support store=None
    pg = dist.ProcessGroup(prefix_store, comm.get_rank(), comm.get_size())

    # Register backend
    # pyre-fixme[6]: BackendWrapper implements dist.Backend but types isn't aware
    pg._register_backend(comm.get_device(), backend_type, wrapper)
    pg._set_group_name(group_name)

    # Update global state
    # pyre-fixme[6]: support store=None
    dist.distributed_c10d._world.pg_map[pg] = (backend_str, prefix_store)
    dist.distributed_c10d._world.pg_names[pg] = group_name
    dist.distributed_c10d._world.pg_backend_config[pg] = str(backend_config)
    dist.distributed_c10d._register_process_group(group_name, pg)

    # Set up rank mapping
    if global_ranks_mapping is not None:
        dist.distributed_c10d._world.pg_group_ranks[pg] = global_ranks_mapping
    else:
        # Default mapping for global process groups
        dist.distributed_c10d._world.pg_group_ranks[pg] = {
            i: i for i in range(comm.get_size())
        }

    # Set up process group tag
    pg_tag = f"ptd:{group_name}"
    dist.distributed_c10d._world.tags_to_pg.setdefault(pg_tag, []).append(pg)
    dist.distributed_c10d._world.pg_to_tag[pg] = pg_tag

    return pg


def _get_store_for_pg() -> dist.Store:
    if not hasattr(_get_store_for_pg, "_store"):
        _get_store_for_pg._store = _get_store(  # pyre-ignore[16]
            "torchcomm", "store_dist"
        )
    return _get_store_for_pg._store


def init_device_mesh(
    mesh_dim_comms: tuple[TorchComm, ...],  # noqa: F405
    mesh_dim_names: tuple[str, ...],
    _global_comm: Optional[TorchComm] = None,  # noqa: F405
) -> dist.DeviceMesh:
    """
    Initializes a `DeviceMesh` from the list of provided `TorchComm` instances.

    See `DeviceMesh` for more details.
    """

    device = mesh_dim_comms[0].get_device()
    mesh_shape = tuple(comm.get_size() for comm in mesh_dim_comms)
    world_size = math.prod(mesh_shape)

    mesh = torch.arange(world_size, dtype=torch.int, device="cpu").view(mesh_shape)

    local_ranks = [comm.get_rank() for comm in mesh_dim_comms]
    global_rank = cast(int, mesh[tuple(local_ranks)].item())
    prefix_store = _get_store_for_pg()
    backend_str = "torchcomm"
    # Register the backend
    dist.Backend.register_backend(backend_str, new_comm)

    global_pg = None
    if _global_comm is not None:
        global_pg = _create_torchcomm_process_group(
            comm=_global_comm,
            group_name=_global_comm.get_name(),
            prefix_store=dist.PrefixStore("default", prefix_store),
            global_ranks_mapping=None,  # Will use default mapping
        )
    elif len(mesh_dim_comms) != 1:
        raise RuntimeError(
            "More than one torch comm objects are passed but no global comm(_global_comm) is provided. "
            "Please provide a global comm object via _global_comm."
        )

    group_names = []
    idx = 0
    for comm, name in zip(mesh_dim_comms, mesh_dim_names):
        group_name = name

        # Calculate global ranks mapping for this mesh dimension
        global_ranks = mesh.transpose(idx, -1).reshape(-1, mesh.size(idx))
        # Find the row containing the global rank
        row_idx = int(torch.where(global_ranks == global_rank)[0].item())
        list_rank = global_ranks[row_idx].tolist()
        global_ranks_mapping = {x: j for j, x in enumerate(list_rank)}

        # Use helper function to create the process group
        pg = _create_torchcomm_process_group(
            comm=comm,
            group_name=group_name,
            backend_str=backend_str,
            prefix_store=dist.PrefixStore(name, prefix_store),
            global_ranks_mapping=global_ranks_mapping,
        )
        if _global_comm is None and idx == 0:
            global_pg = pg

        group_names.append(group_name)
        idx += 1

    # Set as the default world process group
    dist.distributed_c10d.GroupMember.WORLD = global_pg

    device_mesh = dist.DeviceMesh(
        device_type=device.type,
        mesh=mesh,
        mesh_dim_names=mesh_dim_names,
        _init_backend=False,
        _rank=global_rank,
    )
    device_mesh._dim_group_names = group_names

    return device_mesh


def init_native_device_mesh(
    mesh_dim_comms: tuple[TorchComm, ...],
    mesh_dim_names: tuple[str, ...],
    backend: str = "torchcomms",
) -> dist.DeviceMesh:
    """
    Initializes a `DeviceMesh` directly from TorchComm instances, bypassing ProcessGroup.

    This creates a DeviceMesh that stores TorchComm objects under a backend identifier,
    without wrapping them in ProcessGroups or registering with the distributed runtime.
    Use `mesh.get_comm(backend, mesh_dim)` to retrieve the comm for a mesh dimension.

    The torchcomms backend is automatically registered with funcol, so standard
    functional collective calls like `funcol.all_reduce(tensor, "sum", (mesh, dim))`
    will automatically dispatch to torchcomms.

    Args:
        mesh_dim_comms: Tuple of TorchComm instances, one per mesh dimension.
        mesh_dim_names: Tuple of names for each mesh dimension.
        backend: Backend identifier string (default: "torchcomms").

    Returns:
        A DeviceMesh with comm objects stored under the backend identifier.

    Example:
        >>> mesh = init_native_device_mesh(
        ...     mesh_dim_comms=(dp_comm, tp_comm),
        ...     mesh_dim_names=("dp", "tp"),
        ... )
        >>> # Direct access
        >>> comm = mesh.get_comm("torchcomms", "tp")
        >>> # Or use funcol (auto-dispatches to torchcomms)
        >>> funcol.all_reduce(tensor, "sum", (mesh, 1))
    """
    if len(mesh_dim_comms) != len(mesh_dim_names):
        raise ValueError(
            f"mesh_dim_comms length ({len(mesh_dim_comms)}) must match "
            f"mesh_dim_names length ({len(mesh_dim_names)})"
        )

    device = mesh_dim_comms[0].get_device()
    mesh_shape = tuple(comm.get_size() for comm in mesh_dim_comms)
    world_size = math.prod(mesh_shape)

    mesh_tensor = torch.arange(world_size, dtype=torch.int, device="cpu").view(
        mesh_shape
    )

    local_ranks = [comm.get_rank() for comm in mesh_dim_comms]
    global_rank = cast(int, mesh_tensor[tuple(local_ranks)].item())

    device_mesh = dist.DeviceMesh(
        device_type=device.type,
        mesh=mesh_tensor,
        mesh_dim_names=mesh_dim_names,
        _init_backend=False,
        _rank=global_rank,
    )

    # Register comm objects under backend identifier using the native method
    dim_comms = {name: comm for name, comm in zip(mesh_dim_names, mesh_dim_comms)}
    device_mesh.register_comm_backend(backend, dim_comms)  # pyre-ignore[6]
    device_mesh._dim_group_names = list(mesh_dim_names)

    # Register torchcomms with the functional collectives backend (if not already registered)
    from torchcomms.functional.functional_backend import (
        register_torchcomms_funcols_impl,
    )

    register_torchcomms_funcols_impl()

    for comm in dim_comms.values():
        comm.barrier(async_op=False)

    return device_mesh


def _flatten_with_comm(
    mesh: dist.DeviceMesh,
    mesh_dim_name: str,
    comm: TorchComm,  # noqa: F405
    global_ranks: list[int],
    layout: Any,  # noqa: F405
) -> dist.DeviceMesh:
    backend_str = "torchcomm"
    prefix_store = _get_store_for_pg()
    prefix_store = dist.PrefixStore(mesh_dim_name, prefix_store)
    global_ranks_mapping = {global_ranks[i]: i for i in range(comm.get_size())}
    # We still need to register the process group for the flattened mesh
    _create_torchcomm_process_group(
        comm=comm,
        group_name=mesh_dim_name,
        backend_str=backend_str,
        prefix_store=prefix_store,
        global_ranks_mapping=global_ranks_mapping,
    )

    # We had a refactor recently that changed the way we create a DeviceMesh
    # We need to create a new DeviceMesh with the new API.
    # TODO: Clean up this code once torchcomm releases.
    if hasattr(mesh, "_rank_map"):
        flattened_device_mesh = dist.DeviceMesh(
            device_type=comm.get_device(),
            mesh_dim_names=(mesh_dim_name,),
            _init_backend=False,
            _rank=comm.get_rank(),
            _layout=layout.coalesce(),
            _rank_map=mesh._rank_map,
            _root_mesh=mesh,
        )
    else:
        flattened_device_mesh = dist.DeviceMesh(
            device_type=comm.get_device(),
            mesh=torch.tensor(global_ranks, device="cpu"),
            mesh_dim_names=(mesh_dim_name,),
            _init_backend=False,
            _rank=comm.get_rank(),
            _layout=layout.coalesce(),
        )
    flattened_device_mesh._dim_group_names = [mesh_dim_name]

    try:
        flattened_device_mesh._root_mesh = mesh._get_root_mesh()
        flattened_device_mesh._root_mesh._flatten_mapping[mesh_dim_name] = (
            flattened_device_mesh
        )
    except Exception:
        if hasattr(_mesh_resources, "flatten_name_to_root_dims"):
            raise NotImplementedError(
                "Flattening with torchcomm is not supported for device mesh without mesh layout."
            )
        root_mesh = _mesh_resources.get_root_mesh(mesh)
        _mesh_resources.child_to_root_mapping[  # pyre-ignore[16]
            flattened_device_mesh
        ] = root_mesh
        _mesh_resources.root_to_flatten_mapping.setdefault(  # pyre-ignore[16]
            root_mesh, {}
        )[mesh_dim_name] = flattened_device_mesh

    return flattened_device_mesh
