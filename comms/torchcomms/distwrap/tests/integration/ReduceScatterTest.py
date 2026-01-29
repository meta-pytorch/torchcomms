#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Integration tests for reduce_scatter and reduce_scatter_tensor collectives."""

import unittest

import torch
from torchcomms import distwrap as dist
from torchcomms.distwrap.tests.integration.test_helpers import (
    get_backend,
    get_device,
    get_rank_and_size,
    use_torchcomms,
)


class ReduceScatterTest(unittest.TestCase):
    """Test class for reduce_scatter operations using distwrap."""

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

    def test_sync_reduce_scatter(self) -> None:
        """Test synchronous reduce_scatter with SUM."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        input_list = [
            torch.ones(1024, dtype=torch.float, device=device) * (rank + 1)
            for _ in range(num_ranks)
        ]
        output_tensor = torch.zeros(1024, dtype=torch.float, device=device)

        dist.reduce_scatter(
            output_tensor, input_list, op=dist.ReduceOp.SUM, async_op=False
        )

        # Each rank gets the sum of all ranks' contribution for its chunk
        expected = num_ranks * (num_ranks + 1) // 2
        expected_tensor = torch.full_like(output_tensor.cpu(), expected)
        torch.testing.assert_close(output_tensor.cpu(), expected_tensor)

    def test_async_reduce_scatter(self) -> None:
        """Test asynchronous reduce_scatter with SUM."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        input_list = [
            torch.ones(1024, dtype=torch.float, device=device) * (rank + 1)
            for _ in range(num_ranks)
        ]
        output_tensor = torch.zeros(1024, dtype=torch.float, device=device)

        work = dist.reduce_scatter(
            output_tensor, input_list, op=dist.ReduceOp.SUM, async_op=True
        )
        if work is None:
            raise AssertionError("work is None")
        work.wait()

        expected = num_ranks * (num_ranks + 1) // 2
        expected_tensor = torch.full_like(output_tensor.cpu(), expected)
        torch.testing.assert_close(output_tensor.cpu(), expected_tensor)

    def test_sync_reduce_scatter_tensor(self) -> None:
        """Test synchronous reduce_scatter_tensor with SUM."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        input_tensor = torch.ones(
            1024 * num_ranks, dtype=torch.float, device=device
        ) * (rank + 1)
        output_tensor = torch.zeros(1024, dtype=torch.float, device=device)

        dist.reduce_scatter_tensor(
            output_tensor, input_tensor, op=dist.ReduceOp.SUM, async_op=False
        )

        expected = num_ranks * (num_ranks + 1) // 2
        expected_tensor = torch.full_like(output_tensor.cpu(), expected)
        torch.testing.assert_close(output_tensor.cpu(), expected_tensor)

    def test_async_reduce_scatter_tensor(self) -> None:
        """Test asynchronous reduce_scatter_tensor with SUM."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        input_tensor = torch.ones(
            1024 * num_ranks, dtype=torch.float, device=device
        ) * (rank + 1)
        output_tensor = torch.zeros(1024, dtype=torch.float, device=device)

        work = dist.reduce_scatter_tensor(
            output_tensor, input_tensor, op=dist.ReduceOp.SUM, async_op=True
        )
        if work is None:
            raise AssertionError("work is None")
        work.wait()

        expected = num_ranks * (num_ranks + 1) // 2
        expected_tensor = torch.full_like(output_tensor.cpu(), expected)
        torch.testing.assert_close(output_tensor.cpu(), expected_tensor)

    def test_reduce_scatter_variable_sizes(self) -> None:
        """Test reduce_scatter with variable-sized input tensors."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        # Each input tensor has size (i + 1) * 256 for rank i's chunk
        input_list = [
            torch.ones((i + 1) * 256, dtype=torch.float, device=device) * (rank + 1)
            for i in range(num_ranks)
        ]

        # Output tensor receives the chunk for this rank
        output_size = (rank + 1) * 256
        output_tensor = torch.zeros(output_size, dtype=torch.float, device=device)

        dist.reduce_scatter(
            output_tensor, input_list, op=dist.ReduceOp.SUM, async_op=False
        )

        # Each rank gets the sum of all ranks' contributions
        expected = num_ranks * (num_ranks + 1) // 2
        expected_tensor = torch.full_like(output_tensor.cpu(), expected)
        torch.testing.assert_close(output_tensor.cpu(), expected_tensor)


if __name__ == "__main__":
    unittest.main()
