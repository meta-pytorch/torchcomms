#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from typing import Tuple

import torch
from torchcomms import distwrap as dist


def get_dtype_name(dtype: torch.dtype) -> str:
    """Helper function to get a string representation of the datatype."""
    dtype_names = {
        torch.half: "Half",
        torch.float: "Float",
        torch.bfloat16: "BFloat16",
        torch.double: "Double",
        torch.int: "Int",
        torch.int8: "SignedChar",
    }
    return dtype_names.get(dtype, "Unknown")


def get_op_name(op: dist.ReduceOp) -> str:
    """Helper function to get a string representation of the reduction operation."""
    op_names = {
        dist.ReduceOp.SUM: "Sum",
        dist.ReduceOp.PRODUCT: "Product",
        dist.ReduceOp.MIN: "Min",
        dist.ReduceOp.MAX: "Max",
        dist.ReduceOp.BAND: "BAnd",
        dist.ReduceOp.BOR: "BOr",
        dist.ReduceOp.BXOR: "BXor",
        dist.ReduceOp.AVG: "Avg",
    }
    return op_names.get(op, f"Unknown: {op}")


def get_rank_and_size() -> Tuple[int, int]:
    """
    Get rank and world size from environment variables.
    Returns (rank, size) tuple.
    """
    env_pairs = [
        ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
        ("RANK", "WORLD_SIZE"),
        ("SLURM_PROCID", "SLURM_NTASKS"),
        ("PMI_RANK", "PMI_SIZE"),
    ]

    for rank_var, size_var in env_pairs:
        rank = os.environ.get(rank_var)
        size = os.environ.get(size_var)
        if rank is not None and size is not None:
            return int(rank), int(size)

    raise RuntimeError(
        "Could not determine rank or world size from environment variables."
    )


def get_device(rank: int) -> torch.device:
    """Get the device to use for the test."""
    if device_str := os.environ.get("TEST_DEVICE"):
        return torch.device(device_str)

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 0:
            device_id = rank % device_count
            return torch.device(f"cuda:{device_id}")

    return torch.device("cpu")


def get_backend() -> str:
    """Get the backend from environment variable."""
    backend = os.getenv("TEST_BACKEND")
    if backend is None:
        raise RuntimeError("TEST_BACKEND environment variable is not set")
    return backend


def use_torchcomms() -> bool:
    """Check if torchcomms should be used from environment variable."""
    value = os.getenv("USE_TORCHCOMMS")
    if value is None:
        raise RuntimeError("USE_TORCHCOMMS environment variable is not set")
    return value == "1"


def get_pg_options(backend: str) -> object | None:
    """
    Get the appropriate ProcessGroup options for the given backend.

    Returns the backend-specific options object, or None if the backend
    doesn't have a supported options class.
    """
    if backend in ["nccl", "ncclx"]:
        # pyre-ignore[16]: ProcessGroupNCCL not in Pyre stubs
        return dist.ProcessGroupNCCL.Options()
    elif backend in ["gloo"]:
        # pyre-ignore[16]: ProcessGroupGloo not in Pyre stubs
        return dist.ProcessGroupGloo._Options()
    else:
        return None
