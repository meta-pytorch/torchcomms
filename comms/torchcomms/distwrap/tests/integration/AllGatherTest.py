#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Integration tests for all_gather and all_gather_into_tensor collectives."""

import unittest

import torch
from torchcomms import distwrap as dist
from torchcomms.distwrap.tests.integration.test_helpers import (
    get_backend,
    get_device,
    get_rank_and_size,
    use_torchcomms,
)


class AllGatherTest(unittest.TestCase):
    """Test class for all_gather operations using distwrap."""

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

    def test_sync_all_gather(self) -> None:
        """Test synchronous all_gather."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        input_tensor = torch.ones(1024, dtype=torch.float, device=device) * (rank + 1)
        output_list = [
            torch.zeros(1024, dtype=torch.float, device=device)
            for _ in range(num_ranks)
        ]

        dist.all_gather(output_list, input_tensor, async_op=False)

        for i, output in enumerate(output_list):
            expected = torch.full_like(output.cpu(), i + 1)
            torch.testing.assert_close(output.cpu(), expected)

    def test_async_all_gather(self) -> None:
        """Test asynchronous all_gather."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        input_tensor = torch.ones(1024, dtype=torch.float, device=device) * (rank + 1)
        output_list = [
            torch.zeros(1024, dtype=torch.float, device=device)
            for _ in range(num_ranks)
        ]

        work = dist.all_gather(output_list, input_tensor, async_op=True)
        if work is None:
            raise AssertionError("work is None")
        work.wait()

        for i, output in enumerate(output_list):
            expected = torch.full_like(output.cpu(), i + 1)
            torch.testing.assert_close(output.cpu(), expected)

    def test_sync_all_gather_into_tensor(self) -> None:
        """Test synchronous all_gather_into_tensor."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        input_tensor = torch.ones(1024, dtype=torch.float, device=device) * (rank + 1)
        output_tensor = torch.zeros(1024 * num_ranks, dtype=torch.float, device=device)

        dist.all_gather_into_tensor(output_tensor, input_tensor, async_op=False)

        for i in range(num_ranks):
            chunk = output_tensor[i * 1024 : (i + 1) * 1024].cpu()
            expected = torch.full_like(chunk, i + 1)
            torch.testing.assert_close(chunk, expected)

    def test_async_all_gather_into_tensor(self) -> None:
        """Test asynchronous all_gather_into_tensor."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        input_tensor = torch.ones(1024, dtype=torch.float, device=device) * (rank + 1)
        output_tensor = torch.zeros(1024 * num_ranks, dtype=torch.float, device=device)

        work = dist.all_gather_into_tensor(output_tensor, input_tensor, async_op=True)
        if work is None:
            raise AssertionError("work is None")
        work.wait()

        for i in range(num_ranks):
            chunk = output_tensor[i * 1024 : (i + 1) * 1024].cpu()
            expected = torch.full_like(chunk, i + 1)
            torch.testing.assert_close(chunk, expected)

    def test_all_gather_variable_sizes(self) -> None:
        """Test all_gather with variable-sized output tensors."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        # Each rank contributes a tensor of size (rank + 1) * 256
        input_size = (rank + 1) * 256
        input_tensor = torch.ones(input_size, dtype=torch.float, device=device) * (
            rank + 1
        )

        # Output list has variable-sized tensors matching each rank's contribution
        output_list = [
            torch.zeros((i + 1) * 256, dtype=torch.float, device=device)
            for i in range(num_ranks)
        ]

        dist.all_gather(output_list, input_tensor, async_op=False)

        for i, output in enumerate(output_list):
            expected_size = (i + 1) * 256
            self.assertEqual(output.numel(), expected_size)
            expected = torch.full_like(output.cpu(), i + 1)
            torch.testing.assert_close(output.cpu(), expected)


if __name__ == "__main__":
    unittest.main()
