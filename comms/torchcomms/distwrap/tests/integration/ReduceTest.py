#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Integration tests for reduce collective."""

import unittest

import torch
from torchcomms import distwrap as dist
from torchcomms.distwrap.tests.integration.test_helpers import (
    get_backend,
    get_device,
    get_rank_and_size,
    use_torchcomms,
)


class ReduceTest(unittest.TestCase):
    """Test class for reduce operations using distwrap."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize distwrap once for all tests."""
        rank, _ = get_rank_and_size()
        device = get_device(rank)
        backend = get_backend()

        dist.init_process_group(
            backend=backend,
            use_torchcomms=use_torchcomms(),
        )

        if device.type == "cuda":
            torch.cuda.set_device(device)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up distwrap after all tests."""
        dist.destroy_process_group()

    def test_sync_reduce(self) -> None:
        """Test synchronous reduce to rank 0 with SUM."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        input_tensor = torch.ones(1024, dtype=torch.float, device=device) * (rank + 1)

        dist.reduce(input_tensor, dst=0, op=dist.ReduceOp.SUM, async_op=False)

        if rank == 0:
            expected = num_ranks * (num_ranks + 1) // 2
            expected_tensor = torch.full_like(input_tensor.cpu(), expected)
            torch.testing.assert_close(input_tensor.cpu(), expected_tensor)

    def test_async_reduce(self) -> None:
        """Test asynchronous reduce to rank 0 with SUM."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        input_tensor = torch.ones(1024, dtype=torch.float, device=device) * (rank + 1)

        work = dist.reduce(input_tensor, dst=0, op=dist.ReduceOp.SUM, async_op=True)
        if work is None:
            raise AssertionError("work is None")
        work.wait()

        if rank == 0:
            expected = num_ranks * (num_ranks + 1) // 2
            expected_tensor = torch.full_like(input_tensor.cpu(), expected)
            torch.testing.assert_close(input_tensor.cpu(), expected_tensor)


if __name__ == "__main__":
    unittest.main()
