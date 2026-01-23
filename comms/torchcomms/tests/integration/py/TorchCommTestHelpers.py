#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from typing import Tuple, Union

import torch
from torchcomms import new_comm, RedOpType
from torchcomms._comms import _get_store


def get_dtype_name(dtype):
    """Helper function to get a string representation of the datatype."""
    if dtype == torch.half:
        return "Half"
    elif dtype == torch.float:
        return "Float"
    elif dtype == torch.bfloat16:
        return "BFloat16"
    elif dtype == torch.double:
        return "Double"
    elif dtype == torch.int:
        return "Int"
    elif dtype == torch.int8:
        return "SignedChar"
    else:
        return "Unknown"


def get_op_name(op):
    """Helper function to get a string representation of the reduction operation."""
    if op.type == RedOpType.SUM:
        return "Sum"
    elif op.type == RedOpType.PRODUCT:
        return "Product"
    elif op.type == RedOpType.MIN:
        return "Min"
    elif op.type == RedOpType.MAX:
        return "Max"
    elif op.type == RedOpType.BAND:
        return "BAnd"
    elif op.type == RedOpType.BOR:
        return "BOr"
    elif op.type == RedOpType.BXOR:
        return "BXor"
    elif op.type == RedOpType.PREMUL_SUM:
        return "PremulSum"
    else:
        return "Unknown: " + str(op.type)


def get_rank_and_size() -> Tuple[int, int]:
    """
    Get rank and world size from environment variables.
    Returns (rank, size) tuple.
    """
    # Try OpenMPI environment variables first
    ompi_rank = os.environ.get("OMPI_COMM_WORLD_RANK")
    ompi_size = os.environ.get("OMPI_COMM_WORLD_SIZE")

    if ompi_rank is not None and ompi_size is not None:
        return int(ompi_rank), int(ompi_size)

    # Try TorchComms environment variables
    rank = os.environ.get("TORCHCOMM_RANK")
    size = os.environ.get("TORCHCOMM_SIZE")

    if rank is not None and size is not None:
        return int(rank), int(size)

    # Try SLURM environment variables
    slurm_rank = os.environ.get("SLURM_PROCID")
    slurm_size = os.environ.get("SLURM_NTASKS")

    if slurm_rank is not None and slurm_size is not None:
        return int(slurm_rank), int(slurm_size)

    # Try PMI environment variables
    pmi_rank = os.environ.get("PMI_RANK")
    pmi_size = os.environ.get("PMI_SIZE")

    if pmi_rank is not None and pmi_size is not None:
        return int(pmi_rank), int(pmi_size)

    # Try torchrun environment variables
    torchrun_rank = os.environ.get("RANK")
    torchrun_size = os.environ.get("WORLD_SIZE")

    if torchrun_rank is not None and torchrun_size is not None:
        return int(torchrun_rank), int(torchrun_size)

    raise RuntimeError(
        "Could not determine rank or world size from environment variables."
    )


def maybe_set_rank_envs():
    ranksize_query_method = os.getenv("TORCHCOMM_BOOTSTRAP_RANKSIZE_QUERY_METHOD")
    if ranksize_query_method == "manual":
        # Get rank and size from environment variables
        rank, size = get_rank_and_size()
        os.environ["TORCHCOMM_RANK"] = str(rank)
        os.environ["TORCHCOMM_SIZE"] = str(size)
        del os.environ["TORCHCOMM_BOOTSTRAP_RANKSIZE_QUERY_METHOD"]


NEXT_STORE_ID = 0


def create_store():
    """Create a TCPStore object for coordination."""
    maybe_set_rank_envs()

    global NEXT_STORE_ID
    NEXT_STORE_ID += 1
    return _get_store("my_backend", f"test_comm_{NEXT_STORE_ID}")


def verify_tensor_equality(
    output: torch.Tensor,
    expected: Union[torch.Tensor, int, float],
    description: str = "",
):
    """
    Verify tensor equality with appropriate comparison for different dtypes.

    Args:
        output: The output tensor to verify
        expected: Either an expected tensor or a scalar value
        description: Optional description for error messages
    """
    # Skip verification if tensor is empty
    if output.numel() == 0:
        return

    # Create expected tensor if a scalar value was provided
    if isinstance(expected, (int, float)):
        expected_tensor = torch.full_like(output.cpu(), float(expected))
    else:
        expected_tensor = expected.cpu()

    # Ensure output is on CPU
    output_cpu = output.cpu()

    # Check that tensors have the same shape
    assert (
        output_cpu.size() == expected_tensor.size()
    ), f"Tensor shapes don't match for {description}"

    # Check that tensors have the same dtype
    assert (
        output_cpu.dtype == expected_tensor.dtype
    ), f"Tensor dtypes don't match for {description}"

    # Different verification based on dtype
    if output_cpu.dtype == torch.float:
        # For float tensors, check if they are close enough
        diff = torch.abs(output_cpu - expected_tensor)
        all_close = diff.max().item() < 1e-5
        assert all_close, f"Tensors are not close enough for {description}"

        # If not all close, print individual differences for debugging
        if not all_close:
            # Find indices where difference is significant
            significant_diff = diff > 1e-5
            indices = significant_diff.nonzero()

            # Print up to 10 differences
            num_diffs = min(10, indices.size(0))
            for i in range(num_diffs):
                idx = indices[i]
                flat_idx = int(idx.item())
                print(
                    f"Difference at index {flat_idx}: "
                    f"output={output_cpu.flatten()[flat_idx].item()}, "
                    f"expected={expected_tensor.flatten()[flat_idx].item()}"
                )
    else:
        # For integer types, check exact equality
        equal = torch.all(output_cpu.eq(expected_tensor)).item()
        assert equal, f"Tensors are not equal for {description}"

        # If not equal, print individual differences for debugging
        if not equal:
            # Find indices where values differ
            diff_indices = (output_cpu != expected_tensor).nonzero()

            # Print up to 10 differences
            num_diffs = min(10, diff_indices.size(0))
            for i in range(num_diffs):
                idx = diff_indices[i]
                flat_idx = int(idx.item())
                print(
                    f"Difference at index {flat_idx}: "
                    f"output={output_cpu.flatten()[flat_idx].item()}, "
                    f"expected={expected_tensor.flatten()[flat_idx].item()}"
                )


class TorchCommTestWrapper:
    """Wrapper class for TorchComm tests, similar to the C++ TorchCommTestWrapper."""

    NEXT_COMM_ID = 0

    def get_device(self, backend, rank) -> torch.device:
        if device_str := os.environ.get("TEST_DEVICE"):
            return torch.device(device_str)

        if torch.accelerator.is_available():
            device_count = torch.accelerator.device_count()
            if device_count > 0:
                device_id = rank % device_count
                accelerator = torch.accelerator.current_accelerator()
                assert accelerator is not None
                device_type = accelerator.type
                return torch.device(f"{device_type}:{device_id}")
        # Fallback to CPU if an accelerator is not found or device_count is 0
        return torch.device("cpu")

    def get_hints_from_env(self):
        hints = {}
        if fast_init_mode := os.environ.get("TEST_FAST_INIT_MODE"):
            hints.update({"fastInitMode": fast_init_mode})
        return hints

    def __init__(self, store=None):
        maybe_set_rank_envs()

        # Get backend from TEST_BACKEND environment variable, throw if not set
        backend = os.getenv("TEST_BACKEND")
        if backend is None:
            raise RuntimeError("TEST_BACKEND environment variable is not set")

        rank, size = get_rank_and_size()
        device = self.get_device(os.environ["TEST_BACKEND"], rank)
        hints = self.get_hints_from_env()
        # Create and initialize TorchComm instance
        TorchCommTestWrapper.NEXT_COMM_ID += 1
        self.torchcomm = new_comm(
            backend,
            device,
            store=store,
            name=f"comms_test_{TorchCommTestWrapper.NEXT_COMM_ID}",
            hints=hints,
        )

        print(
            f"TorchComm created with params: backend {self.torchcomm.get_backend()}, device {self.torchcomm.get_device()}, options {self.torchcomm.get_options()}"
        )

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "torchcomm") and self.torchcomm:
            self.torchcomm.finalize()
            self.torchcomm = None

    def get_torchcomm(self):
        """Get the TorchComm instance."""
        return self.torchcomm
