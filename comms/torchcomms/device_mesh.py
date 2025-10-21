# Copyright (c) Meta Platforms, Inc. and affiliates.
import math
from typing import Any, cast, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import _mesh_resources

from torchcomms._comms import _BackendWrapper, new_comm, TorchComm


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
    wrapper = _BackendWrapper(comm)  # noqa: F405
    backend_type = dist.ProcessGroup.BackendType.CUSTOM  # noqa: F841
    backend_config = dist.BackendConfig(dist.Backend(backend_str))

    # Create process group
    # pyre-fixme[6]: support store=None
    pg = dist.ProcessGroup(None, comm.get_rank(), comm.get_size())

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

    return pg


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
    prefix_store = None
    backend_str = "torchcomm"
    # Register the backend
    dist.Backend.register_backend(backend_str, new_comm)

    global_pg = None
    if _global_comm is not None:
        global_pg = _create_torchcomm_process_group(
            comm=_global_comm,
            group_name=_global_comm.get_name(),
            prefix_store=prefix_store,
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
        global_ranks = mesh.transpose(idx, -1).reshape(-1, mesh.size(idx)).tolist()
        global_ranks_mapping = {x: j for sub in global_ranks for j, x in enumerate(sub)}

        # Use helper function to create the process group
        pg = _create_torchcomm_process_group(
            comm=comm,
            group_name=group_name,
            backend_str=backend_str,
            prefix_store=prefix_store,
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


def _flatten_with_comm(
    mesh: dist.DeviceMesh,
    mesh_dim_name: str,
    comm: TorchComm,  # noqa: F405
    global_ranks: list[int],
    layout: Any,  # noqa: F405
) -> dist.DeviceMesh:
    backend_str = "torchcomm"
    prefix_store = None
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
        flattened_device_mesh = dist.DeviceMesh(  # pyre-ignore[28]
            device_type=comm.get_device(),
            mesh_dim_names=(mesh_dim_name,),
            _init_backend=False,
            _rank=comm.get_rank(),
            _layout=layout.coalesce(),
            _rank_map=mesh._rank_map,  # pyre-ignore[16]
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
