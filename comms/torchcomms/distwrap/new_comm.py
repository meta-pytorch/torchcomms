# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, TCPStore
from torchcomms._comms import new_comm, TorchComm
from torchcomms.distwrap.fallback import fallback_split_group_new_group
from torchcomms.distwrap.pginfo import (
    pg_info_assert_registered,
    pg_info_create,
    pg_info_destroy,
    pg_info_get_data,
    pg_info_set_data,
)
from torchcomms.distwrap.utils import (
    _BACKEND_RENAME_FOR_DIST,
    _block_torch_distributed_collectives,
    eager_init_is_enabled,
    format_backend_string,
    get_rank_and_world_size,
    parse_backend_string,
    set_eager_init_enabled,
    set_torchcomms_enabled,
    torchcomms_create_split_group,
    torchcomms_is_enabled,
)


def init_process_group(
    backend: str | dist.Backend | None = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: dist.Store | None = None,
    group_name: str = "",
    pg_options: Any | None = None,
    device_id: torch.device | int | None = None,
    # Extension API
    use_torchcomms: bool = False,
) -> None:
    """
    Initialize the process group for distributed communication.

    Note: If use_torchcomms is True, device_id is ignored and set to None
    internally to disable eager mode in torch.distributed.
    """
    if backend is None:
        raise ValueError("backend must be specified")

    # Store the use_torchcomms setting globally
    set_torchcomms_enabled(use_torchcomms)

    # Get rank and world_size from parameters or environment variables
    rank, world_size = get_rank_and_world_size(rank, world_size)

    # If torchcomms is active, turn off eager mode for torch.distributed
    # and create a shared store if one is not provided
    tc_store: TCPStore | None = None
    if use_torchcomms:
        device_id = None
        # Create a TCPStore to share between torch.distributed and torchcomms
        # This avoids port conflicts since both will use the same store
        if store is None:
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = int(os.environ.get("MASTER_PORT", "29500"))
            tc_store = TCPStore(
                host_name=master_addr,
                port=master_port,
                world_size=world_size,
                is_master=(rank == 0),
                wait_for_workers=True,
            )
            store = tc_store
        else:
            # Use provided store for torchcomms as well
            tc_store = store  # type: ignore[assignment]

    # Store the eager_init setting globally
    set_eager_init_enabled(device_id is not None)

    # Parse backend string and get dist-compatible backend string
    device_backends = parse_backend_string(backend)

    # torch.distributed does not understand ncclx/rcclx backends,
    # so rename them to nccl/rccl respectively
    dist_device_backends = {
        device: _BACKEND_RENAME_FOR_DIST.get(be, be)
        for device, be in device_backends.items()
    }
    dist_backend = format_backend_string(dist_device_backends)

    # Call dist.init_process_group
    dist.init_process_group(
        backend=dist_backend,
        init_method=init_method,
        timeout=timeout,
        world_size=world_size,
        rank=rank,
        store=store,
        group_name=group_name,
        pg_options=pg_options,
        device_id=device_id,
    )

    # Register the world process group
    world_ranks = list(range(world_size))
    world_pg = dist.group.WORLD
    if world_pg is None:
        raise AssertionError("World process group is not initialized")
    pg_info_create(world_pg, world_ranks, group_name)

    # Store the device_backends mapping
    pg_info_set_data(world_pg, "device_backends", device_backends)

    # If use_torchcomms is set, create torchcomms communicators for each backend
    if use_torchcomms:
        # Block direct calls to torch.distributed collectives
        _block_torch_distributed_collectives()

        # Store the shared store reference for cleanup later
        pg_info_set_data(world_pg, "tc_store", tc_store)

        torchcomms_instances: dict[str, TorchComm] = {}
        for device_type, backend_name in device_backends.items():
            name = group_name or f"default_torchcomm_{backend_name}"

            # Create torchcomms instance with the shared store
            torchcomms_instance = new_comm(
                backend=backend_name,
                device=torch.device(device_type),
                name=name,
                store=tc_store,
            )

            torchcomms_instances[device_type] = torchcomms_instance

        # Store torchcomms instances in pg_info data field
        pg_info_set_data(world_pg, "torchcomms", torchcomms_instances)


def new_group(
    ranks: list[int] | None = None,
    timeout: timedelta | None = None,
    backend: str | None = None,
    pg_options: Any | None = None,
    use_local_synchronization: bool = False,
    group_desc: str | None = None,
    device_id: torch.device | None = None,
) -> ProcessGroup:
    if not dist.is_initialized():
        raise AssertionError("new_group called before init_process_group")
    world_pg = dist.group.WORLD
    assert world_pg is not None
    pg_info_assert_registered(world_pg)

    if torchcomms_is_enabled():
        raise AssertionError(
            "new_group is not supported with torchcomms. Use split_group instead."
        )

    if backend is None:
        raise ValueError("backend must be specified")

    if ranks is None:
        ranks = list(range(dist.get_world_size()))

    # Get device_backends from backend parameter
    device_backends = parse_backend_string(backend)

    # torch.distributed does not understand ncclx/rcclx backends,
    # so rename them to nccl/rccl respectively
    dist_device_backends = {
        device: _BACKEND_RENAME_FOR_DIST.get(be, be)
        for device, be in device_backends.items()
    }
    dist_backend = format_backend_string(dist_device_backends)

    # Create the process group
    new_pg = dist.new_group(
        ranks=ranks,
        timeout=timeout,
        backend=dist_backend,
        pg_options=pg_options,
        use_local_synchronization=use_local_synchronization,
        group_desc=group_desc,
        device_id=device_id,
    )

    # Register the new process group
    pg_info_create(new_pg, ranks, group_desc)

    # Store device_backends
    pg_info_set_data(new_pg, "device_backends", device_backends)

    return new_pg


def destroy_process_group(group: ProcessGroup | None = None) -> None:
    """Destroy the process group and clean up resources."""
    if group is None:
        group = dist.group.WORLD

    if group is None:
        raise AssertionError("Process group is not initialized")

    # Finalize torchcomms instances if they exist
    if torchcomms_is_enabled():
        torchcomms_instances = pg_info_get_data(group, "torchcomms")
        for _device_type, tc_instance in torchcomms_instances.items():
            tc_instance.finalize()

        # Clear the store reference - it will be garbage collected
        # and its destructor will handle cleanup (shutdown server if master)
        pg_info_set_data(group, "tc_store", None)

    # Clean up pg_info registry
    pg_info_destroy(group)

    # Call dist.destroy_process_group
    dist.destroy_process_group(group)


def split_group(  # noqa: C901
    group: ProcessGroup | None = None,
    split_ranks: list[list[int]] | None = None,
    timeout: timedelta | None = None,
    pg_options: Any | None = None,
    group_desc: str | None = None,
    # Extension API
    backend: str | None = None,
) -> ProcessGroup:
    if not dist.is_initialized():
        raise AssertionError("split_group called before init_process_group")

    # Default to WORLD if no group specified
    if group is None:
        parent_pg = dist.group.WORLD
        if parent_pg is None:
            raise AssertionError("World process group is not initialized")
    else:
        parent_pg = group
    pg_info_assert_registered(parent_pg)

    if split_ranks is None:
        split_ranks = [list(range(dist.get_world_size(parent_pg)))]

    # Find which group this rank belongs to
    my_rank = dist.get_rank(parent_pg)
    my_group_ranks = None
    for rank_list in split_ranks:
        if my_rank in rank_list:
            my_group_ranks = rank_list
            break

    if my_group_ranks is None:
        raise ValueError(f"Rank {my_rank} not found in any split_ranks group")

    # Get device_backends from backend parameter or parent group
    if backend is not None:
        device_backends = parse_backend_string(backend)
    else:
        device_backends = pg_info_get_data(parent_pg, "device_backends")

    if torchcomms_is_enabled():
        # Create a pg with new_group
        new_pg = dist.new_group(
            ranks=my_group_ranks,
            pg_options=pg_options,
            group_desc=group_desc,
        )

        # Register the new process group
        pg_info_create(new_pg, my_group_ranks, group_desc)

        # Store device_backends
        pg_info_set_data(new_pg, "device_backends", device_backends)

        # Create torchcomms instances by splitting from the parent pg
        torchcomms_create_split_group(
            new_pg,
            parent_pg,
            split_ranks,
            group_desc or "",
            pg_options,
            device_backends,
        )
    else:
        # Check if split_group is supported
        split_group_supported = True
        for be in device_backends.values():
            if be == "gloo":
                split_group_supported = False
                break
        if not eager_init_is_enabled():
            split_group_supported = False

        if split_group_supported:
            # Use dist.split_group
            new_pg = dist.split_group(
                parent_pg,
                split_ranks=split_ranks,
                timeout=timeout,
                pg_options=pg_options,
                group_desc=group_desc,
            )
        else:
            # Slow path: implement split_group using new_group
            new_pg = fallback_split_group_new_group(
                parent_pg, my_group_ranks, timeout, pg_options, group_desc
            )

        assert new_pg is not None
        # Register the new process group
        pg_info_create(new_pg, my_group_ranks, group_desc)

        # Store device_backends
        pg_info_set_data(new_pg, "device_backends", device_backends)

    return new_pg
