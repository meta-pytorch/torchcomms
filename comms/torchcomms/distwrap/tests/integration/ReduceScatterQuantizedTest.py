#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Integration tests for reduce_scatter_quantized via distwrap."""

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


class ReduceScatterQuantizedTest(unittest.TestCase):
    """Test class for reduce_scatter_quantized operations using distwrap."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["NCCL_PAT_ENABLE"] = "1"

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

    def test_reduce_scatter_quantized_sum(self) -> None:
        """Test reduce_scatter_quantized with SUM reduction.

        Each rank contributes a tensor of ones. After SUM reduce-scatter,
        each element should be approximately num_ranks.
        """
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        chunk_size = 1024
        input_tensor = torch.ones(
            num_ranks * chunk_size, dtype=torch.float32, device=device
        )
        output_tensor = torch.zeros(chunk_size, dtype=torch.float32, device=device)
        seed = torch.tensor([42 + rank], dtype=torch.int64, device=device)

        work = dist.reduce_scatter_quantized(
            output_tensor,
            input_tensor,
            dist.ReduceOp.SUM,
            seed,
            async_op=False,
        )
        work.wait()

        expected = torch.full(
            (chunk_size,),
            float(num_ranks),
            dtype=torch.float32,
            device=device,
        )
        torch.testing.assert_close(
            output_tensor,
            expected,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_reduce_scatter_quantized_async(self) -> None:
        """Test that async_op=True returns a valid work handle."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        chunk_size = 512
        input_tensor = torch.ones(
            num_ranks * chunk_size, dtype=torch.float32, device=device
        )
        output_tensor = torch.zeros(chunk_size, dtype=torch.float32, device=device)
        seed = torch.tensor([7], dtype=torch.int64, device=device)

        work = dist.reduce_scatter_quantized(
            output_tensor,
            input_tensor,
            dist.ReduceOp.SUM,
            seed,
            async_op=True,
        )
        self.assertIsNotNone(work)
        work.wait()

        expected = torch.full(
            (chunk_size,),
            float(num_ranks),
            dtype=torch.float32,
            device=device,
        )
        torch.testing.assert_close(
            output_tensor,
            expected,
            atol=1e-2,
            rtol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
