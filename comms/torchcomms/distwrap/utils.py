# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

import os
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torchcomms._comms import TorchComm
from torchcomms.distwrap.pginfo import pg_info_get_data, pg_info_set_data


# =============================================================================
# Public API Functions
# =============================================================================


def set_torchcomms_enabled(enabled: bool) -> None:
    """Set whether torchcomms is enabled."""
    global _use_torchcomms
    _use_torchcomms = enabled


def torchcomms_is_enabled() -> bool:
    """Check if torchcomms is enabled."""
    return _use_torchcomms


def set_eager_init_enabled(enabled: bool) -> None:
    """Set whether eager init is enabled (device_id was passed to init_process_group)."""
    global _eager_init_enabled
    _eager_init_enabled = enabled


def eager_init_is_enabled() -> bool:
    """Check if eager init is enabled."""
    return _eager_init_enabled


def parse_backend_string(backend: str) -> dict[str, str]:
    """
    Parse a backend string into its components.

    The backend string can be:
    - Simple form: "nccl", "gloo", "ncclx", "rccl", "rcclx", "hccl"
    - Merged form: "cpu:gloo,gpu:nccl"

    Known backends and their default device types:
    - nccl -> cuda
    - ncclx -> cuda
    - gloo -> cpu
    - hccl -> mtia

    Args:
        backend: Backend string

    Returns:
        device_backends: Dict mapping device type ("cpu"/"gpu"/"mtia") to backend
    """
    device_backends = {}

    # Check if it's a merged form like "cpu:gloo,gpu:nccl"
    if ":" in backend:
        # Parse the merged form
        parts = backend.split(",")
        for part in parts:
            split_result = part.split(":")
            if len(split_result) != 2:
                raise ValueError(
                    f"Invalid backend format: '{part}'. Each part in merged form "
                    f"must have exactly one colon (e.g., 'cpu:gloo,gpu:nccl'), "
                    f"got {len(split_result) - 1} colons."
                )
            device_type, be = split_result
            device_type = device_type.strip()
            be = be.strip()
            device_backends[device_type] = be
    else:
        # Simple form: just the backend name
        # Use known default device type, or raise error for unknown backends
        if backend not in _BACKEND_DEFAULT_DEVICES:
            raise ValueError(
                f"Unknown backend: '{backend}'. Known backends are: "
                f"{', '.join(sorted(_BACKEND_DEFAULT_DEVICES.keys()))}"
            )
        default_device = _BACKEND_DEFAULT_DEVICES[backend]
        device_backends[default_device] = backend

    return device_backends


def format_backend_string(device_backends: dict[str, str]) -> str:
    """
    Convert a device_backends dictionary to a backend string.

    This is the inverse of parse_backend_string.

    Args:
        device_backends: Dict mapping device type to backend name
            (e.g., {"cpu": "gloo"} or {"cpu": "gloo", "cuda": "nccl"})

    Returns:
        Backend string (e.g., "cpu:gloo" or "cpu:gloo,cuda:nccl")
    """
    return ",".join(f"{device}:{be}" for device, be in device_backends.items())


def get_rank_and_world_size(rank: int, world_size: int) -> tuple[int, int]:
    """
    Resolve rank and world_size from parameters or environment variables.

    If rank or world_size is -1, attempt to derive from environment variables.
    Checks multiple environment variable pairs in order:
    - OMPI_COMM_WORLD_RANK / OMPI_COMM_WORLD_SIZE (MPI)
    - RANK / WORLD_SIZE (torchrun)
    - SLURM_PROCID / SLURM_NTASKS (SLURM)
    - PMI_RANK / PMI_SIZE (PMI)

    Args:
        rank: Rank parameter (-1 means derive from environment)
        world_size: World size parameter (-1 means derive from environment)

    Returns:
        Tuple of (rank, world_size)

    Raises:
        AssertionError: If rank or world_size cannot be determined
    """
    env_pairs = [
        ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
        ("RANK", "WORLD_SIZE"),
        ("SLURM_PROCID", "SLURM_NTASKS"),
        ("PMI_RANK", "PMI_SIZE"),
    ]

    # If rank is not set, derive it from environment
    if rank == -1:
        for rank_var, _ in env_pairs:
            if rank_var in os.environ:
                rank = int(os.environ[rank_var])
                break

    # If world_size is not set, derive it from environment
    if world_size == -1:
        for _, size_var in env_pairs:
            if size_var in os.environ:
                world_size = int(os.environ[size_var])
                break

    # Ensure rank and world_size are set
    if rank == -1:
        raise AssertionError(
            "rank must be specified either as a parameter or via environment "
            "variables (RANK, OMPI_COMM_WORLD_RANK, SLURM_PROCID, or PMI_RANK)"
        )
    if world_size == -1:
        raise AssertionError(
            "world_size must be specified either as a parameter or via environment "
            "variables (WORLD_SIZE, OMPI_COMM_WORLD_SIZE, SLURM_NTASKS, or PMI_SIZE)"
        )

    return rank, world_size


def torchcomms_create_split_group(
    pg: ProcessGroup,
    parent_pg: ProcessGroup,
    split_ranks: list[list[int]],
    group_desc: str,
    pg_options: dict[str, Any] | None,
    device_backends: dict[str, str],
) -> None:
    """
    Create torchcomms instances for a split process group.

    Splits the torchcomms instances from the parent process group to create
    new instances for the given process group.

    Args:
        pg: The new process group to create torchcomms instances for.
        parent_pg: The parent process group to split from.
        split_ranks: List of lists of ranks defining the split.
        group_desc: Description for the new process group.
        pg_options: Dictionary of options to pass to torchcomms.
        device_backends: Dict mapping device type to backend name.
    """
    # Get the torchcomms instances from the parent process group
    parent_torchcomms = pg_info_get_data(parent_pg, "torchcomms")
    if not parent_torchcomms:
        raise AssertionError(
            f"No torchcomms instances found for parent process group {parent_pg}"
        )

    # Find which ranks this process belongs to
    my_rank = dist.get_rank(parent_pg)
    ranks = []
    for rank_list in split_ranks:
        if my_rank in rank_list:
            ranks = rank_list
            break

    # Convert pg_options to hints for torchcomms
    hints = _pg_options_to_hints(pg_options)

    # Create torchcomms instances for the new process group using split
    device_types_to_create = set(device_backends.keys())
    new_torchcomms_instances: dict[str, TorchComm] = {}
    for device_type, parent_tc in parent_torchcomms.items():
        if device_type not in device_types_to_create:
            continue
        split_torchcomms = parent_tc.split(ranks, name=group_desc, hints=hints)

        new_torchcomms_instances[device_type] = split_torchcomms

    # Store the new torchcomms instances
    pg_info_set_data(pg, "torchcomms", new_torchcomms_instances)


# =============================================================================
# Private Data and Helper Functions
# =============================================================================


# Global state for torchcomms usage
_use_torchcomms: bool = False
_eager_init_enabled: bool = False

# Known backends and their default device types
_BACKEND_DEFAULT_DEVICES: dict[str, str] = {
    "nccl": "cuda",
    "ncclx": "cuda",
    "rccl": "cuda",
    "rcclx": "cuda",
    "gloo": "cpu",
    "hccl": "mtia",
}

# Backend rename mapping for torch.distributed compatibility
_BACKEND_RENAME_FOR_DIST: dict[str, str] = {
    "ncclx": "nccl",
    "rcclx": "rccl",
}

# List of torch.distributed collectives to block when torchcomms is enabled
_BLOCKED_COLLECTIVES: list[str] = [
    "send",
    "recv",
    "isend",
    "irecv",
    "broadcast",
    "all_reduce",
    "reduce",
    "all_gather",
    "all_gather_into_tensor",
    "gather",
    "scatter",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "all_to_all_single",
    "all_to_all",
    "barrier",
    "batch_isend_irecv",
    "all_gather_object",
    "gather_object",
    "scatter_object_list",
    "broadcast_object_list",
]


def get_group(group: ProcessGroup | None) -> ProcessGroup:
    """Get the process group, defaulting to WORLD if None."""
    if group is None:
        pg = dist.group.WORLD
        if pg is None:
            raise AssertionError("dist.group.WORLD is not initialized")
        return pg
    return group


def get_torchcomms_instance(
    group: ProcessGroup,
    device_type: str | None = None,
    tensor: torch.Tensor | None = None,
) -> Any:
    """
    Get the torchcomms instance for the given group.

    Args:
        group: The process group.
        device_type: The device type string (e.g., "cuda", "cpu"). If provided, uses this directly.
        tensor: A tensor to infer device type from. Used if device_type is not provided.

    Returns:
        The torchcomms instance for the specified device type.

    Raises:
        ValueError: If neither device_type nor tensor is provided, or if device type not found.
    """
    if device_type is None and tensor is None:
        raise ValueError("Either device_type or tensor must be provided")

    if device_type is None:
        if tensor is None:
            raise AssertionError("tensor should not be None at this point")
        device_type = tensor.device.type

    torchcomms_instances = pg_info_get_data(group, "torchcomms")
    if not torchcomms_instances:
        raise AssertionError(f"No torchcomms instances found for process group {group}")

    if device_type not in torchcomms_instances:
        raise ValueError(
            f"No torchcomms instance for device type '{device_type}'. "
            f"Available: {', '.join(sorted(torchcomms_instances.keys()))}"
        )

    return torchcomms_instances[device_type]


def get_group_rank(group: ProcessGroup, global_rank: int) -> int:
    """
    Convert a global rank to a group-local rank.

    Args:
        group: The process group.
        global_rank: The global rank to convert.

    Returns:
        The group-local rank (index within the group).

    Raises:
        ValueError: If the global rank is not in the group.
    """
    from torchcomms.distwrap.pginfo import pg_info_get_global_ranks

    global_ranks = pg_info_get_global_ranks(group)
    try:
        return global_ranks.index(global_rank)
    except ValueError:
        raise ValueError(
            f"Global rank {global_rank} is not in process group. "
            f"Group ranks: {global_ranks}"
        )


def get_backend_for_device(group: ProcessGroup, device_type: str) -> str:
    """
    Get the backend name for a given device type in a process group.

    Args:
        group: The process group.
        device_type: The device type (e.g., "cpu", "cuda").

    Returns:
        The backend name for the device type.
    """
    device_backends = pg_info_get_data(group, "device_backends")
    assert device_backends is not None, (
        f"No device_backends found for process group {group}"
    )
    assert device_type in device_backends, (
        f"Device type '{device_type}' not found in device_backends. "
        f"Available: {', '.join(sorted(device_backends.keys()))}"
    )
    return device_backends[device_type]


def _block_torch_distributed_collectives() -> None:
    """
    Block torch.distributed collective calls when torchcomms is enabled.

    This patches torch.distributed collective functions to raise
    NotImplementedError, ensuring users don't accidentally call dist.*
    functions directly when they should be using the distwrap wrappers.
    """
    from unittest.mock import patch

    for coll in _BLOCKED_COLLECTIVES:
        patch(
            f"torch.distributed.{coll}",
            side_effect=NotImplementedError(
                f"{coll}: torch.distributed APIs have been disabled. "
                "Use torchcomms.distwrap collectives instead."
            ),
        ).__enter__()


def _pg_options_to_hints(pg_options: Any | None) -> dict[str, str] | None:  # noqa: C901
    """
    Convert ProcessGroup options to hints dictionary for torchcomms.

    Args:
        pg_options: ProcessGroupNCCL.Options or similar options object

    Returns:
        Dictionary of string hints, or None if no options provided
    """
    if pg_options is None:
        return None

    hints = {}

    # Check if it's a ProcessGroupNCCL.Options object
    if hasattr(pg_options, "config"):
        # Extract config attributes
        config = pg_options.config
        # pyre-ignore[16]: ProcessGroupNCCL may not be available
        process_group_nccl = getattr(dist, "ProcessGroupNCCL", None)
        if process_group_nccl is not None:
            nccl_config_class = getattr(process_group_nccl, "NCCLConfig", None)
            if nccl_config_class:
                for attr in dir(config):
                    if not attr.startswith("_"):
                        try:
                            value = getattr(config, attr)
                            if not callable(value):
                                hints[attr] = str(value)
                        except Exception:
                            pass

    # Also extract direct attributes from pg_options
    for attr in dir(pg_options):
        if not attr.startswith("_") and attr != "config":
            try:
                value = getattr(pg_options, attr)
                if not callable(value):
                    hints[attr] = str(value)
            except Exception:
                pass

    return hints if hints else None
