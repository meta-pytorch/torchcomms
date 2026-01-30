#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Integration tests for all_to_all and all_to_all_single collectives."""

import unittest

import torch
from torchcomms import distwrap as dist
from torchcomms.distwrap.tests.integration.test_helpers import (
    get_backend,
    get_device,
    get_rank_and_size,
    use_torchcomms,
)


class AllToAllTest(unittest.TestCase):
    """Test class for all_to_all operations using distwrap."""

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

    def test_sync_all_to_all(self) -> None:
        """Test synchronous all_to_all."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        # Each rank sends its rank value to all other ranks
        input_list = [
            torch.ones(1024, dtype=torch.float, device=device) * (rank + 1)
            for _ in range(num_ranks)
        ]
        output_list = [
            torch.zeros(1024, dtype=torch.float, device=device)
            for _ in range(num_ranks)
        ]

        dist.all_to_all(output_list, input_list, async_op=False)

        # After all_to_all, output_list[i] should contain data from rank i
        for i, output in enumerate(output_list):
            expected = torch.full_like(output.cpu(), i + 1)
            torch.testing.assert_close(output.cpu(), expected)

    def test_async_all_to_all(self) -> None:
        """Test asynchronous all_to_all."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        input_list = [
            torch.ones(1024, dtype=torch.float, device=device) * (rank + 1)
            for _ in range(num_ranks)
        ]
        output_list = [
            torch.zeros(1024, dtype=torch.float, device=device)
            for _ in range(num_ranks)
        ]

        work = dist.all_to_all(output_list, input_list, async_op=True)
        if work is None:
            raise AssertionError("work is None")
        work.wait()

        for i, output in enumerate(output_list):
            expected = torch.full_like(output.cpu(), i + 1)
            torch.testing.assert_close(output.cpu(), expected)

    def test_sync_all_to_all_single(self) -> None:
        """Test synchronous all_to_all_single."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        # Create input tensor with rank-specific values for each destination
        input_tensor = torch.zeros(1024 * num_ranks, dtype=torch.float, device=device)
        for i in range(num_ranks):
            input_tensor[i * 1024 : (i + 1) * 1024] = rank + 1

        output_tensor = torch.zeros(1024 * num_ranks, dtype=torch.float, device=device)

        dist.all_to_all_single(output_tensor, input_tensor, async_op=False)

        # After all_to_all_single, chunk i should contain data from rank i
        for i in range(num_ranks):
            chunk = output_tensor[i * 1024 : (i + 1) * 1024].cpu()
            expected = torch.full_like(chunk, i + 1)
            torch.testing.assert_close(chunk, expected)

    def test_async_all_to_all_single(self) -> None:
        """Test asynchronous all_to_all_single."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        input_tensor = torch.zeros(1024 * num_ranks, dtype=torch.float, device=device)
        for i in range(num_ranks):
            input_tensor[i * 1024 : (i + 1) * 1024] = rank + 1

        output_tensor = torch.zeros(1024 * num_ranks, dtype=torch.float, device=device)

        work = dist.all_to_all_single(output_tensor, input_tensor, async_op=True)
        if work is None:
            raise AssertionError("work is None")
        work.wait()

        for i in range(num_ranks):
            chunk = output_tensor[i * 1024 : (i + 1) * 1024].cpu()
            expected = torch.full_like(chunk, i + 1)
            torch.testing.assert_close(chunk, expected)

    def test_all_to_all_single_with_split_sizes(self) -> None:
        """Test all_to_all_single with variable split sizes."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        # Each rank sends (i + 1) * 128 elements to rank i
        input_split_sizes = [(i + 1) * 128 for i in range(num_ranks)]
        output_split_sizes = [(rank + 1) * 128 for _ in range(num_ranks)]

        total_input = sum(input_split_sizes)
        total_output = sum(output_split_sizes)

        # Fill input tensor with rank value
        input_tensor = torch.ones(total_input, dtype=torch.float, device=device) * (
            rank + 1
        )
        output_tensor = torch.zeros(total_output, dtype=torch.float, device=device)

        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            async_op=False,
        )

        # Verify output: each chunk should contain data from the corresponding rank
        offset = 0
        for i in range(num_ranks):
            chunk_size = output_split_sizes[i]
            chunk = output_tensor[offset : offset + chunk_size].cpu()
            expected = torch.full_like(chunk, i + 1)
            torch.testing.assert_close(chunk, expected)
            offset += chunk_size


if __name__ == "__main__":
    unittest.main()
