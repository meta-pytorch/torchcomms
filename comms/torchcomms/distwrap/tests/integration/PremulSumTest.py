#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Integration tests for _make_nccl_premul_sum collective operations."""

import os
import unittest

import torch
from torchcomms import distwrap as dist
from torchcomms.distwrap.tests.integration.test_helpers import (
    get_backend,
    get_device,
    get_rank_and_size,
    use_torchcomms,
)


class PremulSumTest(unittest.TestCase):
    """Test class for PREMUL_SUM operations using distwrap."""

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

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") == "gloo" and os.getenv("USE_TORCHCOMMS") != "1",
        "PREMUL_SUM not supported on Gloo without torchcomms",
    )
    def test_all_reduce_premul_sum_float_factor(self) -> None:
        """Test all_reduce with PREMUL_SUM using a float multiplication factor."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        # Each rank contributes ones, multiplied by 0.5 before sum
        input_tensor = torch.ones(1024, dtype=torch.float, device=device)
        mul_factor = 0.5

        premul_sum_op = dist._make_nccl_premul_sum(mul_factor)
        dist.all_reduce(input_tensor, premul_sum_op, async_op=False)

        # Expected: sum of (1.0 * 0.5) for each rank = 0.5 * num_ranks
        expected = 0.5 * num_ranks
        expected_tensor = torch.full_like(input_tensor.cpu(), expected)
        torch.testing.assert_close(input_tensor.cpu(), expected_tensor)

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") == "gloo" and os.getenv("USE_TORCHCOMMS") != "1",
        "PREMUL_SUM with tensor factor not supported on Gloo without torchcomms",
    )
    def test_all_reduce_premul_sum_tensor_factor(self) -> None:
        """Test all_reduce with PREMUL_SUM using a tensor multiplication factor."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        # Each rank contributes ones, multiplied by factor tensor before sum
        input_tensor = torch.ones(1024, dtype=torch.float, device=device)
        mul_factor = torch.tensor(0.25, dtype=torch.float, device=device)

        premul_sum_op = dist._make_nccl_premul_sum(mul_factor)
        dist.all_reduce(input_tensor, premul_sum_op, async_op=False)

        # Expected: sum of (1.0 * 0.25) for each rank = 0.25 * num_ranks
        expected = 0.25 * num_ranks
        expected_tensor = torch.full_like(input_tensor.cpu(), expected)
        torch.testing.assert_close(input_tensor.cpu(), expected_tensor)

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") == "gloo" and os.getenv("USE_TORCHCOMMS") != "1",
        "PREMUL_SUM not supported on Gloo without torchcomms",
    )
    def test_async_all_reduce_premul_sum(self) -> None:
        """Test asynchronous all_reduce with PREMUL_SUM."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        input_tensor = torch.ones(1024, dtype=torch.float, device=device) * (rank + 1)
        mul_factor = 2.0

        premul_sum_op = dist._make_nccl_premul_sum(mul_factor)
        work = dist.all_reduce(input_tensor, premul_sum_op, async_op=True)
        if work is None:
            raise AssertionError("work is None")
        work.wait()

        # Expected: sum of ((rank + 1) * 2.0) for each rank
        # = 2.0 * (1 + 2 + ... + num_ranks) = 2.0 * num_ranks * (num_ranks + 1) / 2
        expected = 2.0 * num_ranks * (num_ranks + 1) / 2
        expected_tensor = torch.full_like(input_tensor.cpu(), expected)
        torch.testing.assert_close(input_tensor.cpu(), expected_tensor)

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") == "gloo" and os.getenv("USE_TORCHCOMMS") != "1",
        "PREMUL_SUM not supported on Gloo without torchcomms",
    )
    def test_reduce_scatter_tensor_premul_sum(self) -> None:
        """Test reduce_scatter_tensor with PREMUL_SUM."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        input_tensor = torch.ones(
            1024 * num_ranks, dtype=torch.float, device=device
        ) * (rank + 1)
        output_tensor = torch.zeros(1024, dtype=torch.float, device=device)
        mul_factor = 0.5

        premul_sum_op = dist._make_nccl_premul_sum(mul_factor)
        dist.reduce_scatter_tensor(
            output_tensor, input_tensor, op=premul_sum_op, async_op=False
        )

        # Expected: 0.5 * sum of (rank + 1) for all ranks
        # = 0.5 * num_ranks * (num_ranks + 1) / 2
        expected = 0.5 * num_ranks * (num_ranks + 1) / 2
        expected_tensor = torch.full_like(output_tensor.cpu(), expected)
        torch.testing.assert_close(output_tensor.cpu(), expected_tensor)


if __name__ == "__main__":
    unittest.main()
