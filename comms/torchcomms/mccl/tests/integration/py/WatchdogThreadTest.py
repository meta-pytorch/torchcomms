#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import time
import unittest

import torch
from torchcomms import ReduceOp
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    TorchCommTestWrapper,
)


def get_cuda_memory(device: torch.device, label: str) -> int:
    """Print CUDA memory allocation and return the value in bytes."""
    mem = torch.cuda.memory_allocated(device)
    print(f"{label}: {mem / 1024**2:.2f} MB")
    return mem


class WatchDogThreadTest(unittest.TestCase):
    wrapper: TorchCommTestWrapper | None

    def get_wrapper(self) -> TorchCommTestWrapper:
        return TorchCommTestWrapper()

    def setUp(self) -> None:
        """Set up test environment before each test."""
        os.environ["NCCL_COMM_STATE_DEBUG_TOPO"] = "nolocal"
        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

    def tearDown(self) -> None:
        """Clean up after each test."""
        # Explicitly reset the TorchComm object to ensure proper cleanup
        self.torchcomm = None
        self.wrapper = None

    def test_mem_release(self) -> None:
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        initial_mem = get_cuda_memory(self.device, "Initial memory")

        # Create a large 4GB tensor with float
        tensor_size = 1024 * 1024 * 1024
        input_tensor = torch.ones(tensor_size, dtype=torch.float, device=self.device)
        op = ReduceOp.SUM
        torch.cuda.synchronize(self.device)
        get_cuda_memory(self.device, "Peak memory (after tensor allocation)")

        self.torchcomm.all_reduce(
            input_tensor,
            op,
            True,
        )
        del input_tensor

        torch.cuda.synchronize(self.device)

        # watchdog thread poll every 100ms
        # TODO: We should add more robust sleep time.
        time.sleep(1)
        torch.cuda.empty_cache()
        final_mem = get_cuda_memory(self.device, "Final memory (after cleanup)")

        leaked_mem = final_mem - initial_mem
        print(f"Memory difference: {leaked_mem / 1024**2:.2f} MB")

        self.assertLessEqual(
            leaked_mem,
            1024 * 1024,
            f"Memory leak detected: {leaked_mem / 1024**2:.2f} MB leaked",
        )


if __name__ == "__main__":
    unittest.main()
