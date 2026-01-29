#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Integration tests for scatter and gather collectives."""

import unittest

import torch
from torchcomms import distwrap as dist
from torchcomms.distwrap.tests.integration.test_helpers import (
    get_backend,
    get_device,
    get_rank_and_size,
    use_torchcomms,
)


class ScatterGatherTest(unittest.TestCase):
    """Test class for scatter and gather operations using distwrap."""

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

    def test_sync_scatter(self) -> None:
        """Test synchronous scatter from rank 0."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        output_tensor = torch.zeros(1024, dtype=torch.float, device=device)

        if rank == 0:
            scatter_list = [
                torch.ones(1024, dtype=torch.float, device=device) * (i + 1)
                for i in range(num_ranks)
            ]
        else:
            scatter_list = None

        dist.scatter(output_tensor, scatter_list, src=0, async_op=False)

        expected = torch.full_like(output_tensor.cpu(), rank + 1)
        torch.testing.assert_close(output_tensor.cpu(), expected)

    def test_async_scatter(self) -> None:
        """Test asynchronous scatter from rank 0."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        output_tensor = torch.zeros(1024, dtype=torch.float, device=device)

        if rank == 0:
            scatter_list = [
                torch.ones(1024, dtype=torch.float, device=device) * (i + 1)
                for i in range(num_ranks)
            ]
        else:
            scatter_list = None

        work = dist.scatter(output_tensor, scatter_list, src=0, async_op=True)
        if work is None:
            raise AssertionError("work is None")
        work.wait()

        expected = torch.full_like(output_tensor.cpu(), rank + 1)
        torch.testing.assert_close(output_tensor.cpu(), expected)

    def test_sync_gather(self) -> None:
        """Test synchronous gather to rank 0."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        input_tensor = torch.ones(1024, dtype=torch.float, device=device) * (rank + 1)

        if rank == 0:
            gather_list = [
                torch.zeros(1024, dtype=torch.float, device=device)
                for _ in range(num_ranks)
            ]
        else:
            gather_list = None

        dist.gather(input_tensor, gather_list, dst=0, async_op=False)

        if rank == 0:
            if gather_list is None:
                raise AssertionError("gather_list is None")
            for i, gathered in enumerate(gather_list):
                expected = torch.full_like(gathered.cpu(), i + 1)
                torch.testing.assert_close(gathered.cpu(), expected)

    def test_async_gather(self) -> None:
        """Test asynchronous gather to rank 0."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        input_tensor = torch.ones(1024, dtype=torch.float, device=device) * (rank + 1)

        if rank == 0:
            gather_list = [
                torch.zeros(1024, dtype=torch.float, device=device)
                for _ in range(num_ranks)
            ]
        else:
            gather_list = None

        work = dist.gather(input_tensor, gather_list, dst=0, async_op=True)
        if work is None:
            raise AssertionError("work is None")
        work.wait()

        if rank == 0:
            if gather_list is None:
                raise AssertionError("gather_list is None")
            for i, gathered in enumerate(gather_list):
                expected = torch.full_like(gathered.cpu(), i + 1)
                torch.testing.assert_close(gathered.cpu(), expected)


if __name__ == "__main__":
    unittest.main()
