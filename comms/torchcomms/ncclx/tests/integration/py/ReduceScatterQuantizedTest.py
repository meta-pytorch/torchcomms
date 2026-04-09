#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

import torch
import torchcomms
from torchcomms.tests.integration.py.TorchCommTestHelpers import TorchCommTestWrapper


class ReduceScatterQuantizedTest(unittest.TestCase):
    """Integration tests for the ncclReduceScatterQuantize collective."""

    def setUp(self):
        """Set up test environment before each test."""
        os.environ["NCCL_PAT_ENABLE"] = "1"

        self.wrapper = TorchCommTestWrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

        # Get the NCCLX backend for NCCLX-specific APIs
        self.ncclx_backend = self.torchcomm.unsafe_get_backend()

    def tearDown(self):
        """Clean up after each test."""
        self.torchcomm = None
        self.wrapper = None

    def test_reduce_scatter_quantized_sum(self):
        """Test reduce-scatter quantized with SUM reduction.

        Each rank contributes a tensor of ones, so the expected result after
        SUM reduce-scatter is a tensor of num_ranks (approximately, due to
        stochastic rounding in BF16).
        """
        chunk_size = 1024
        input_tensor = torch.ones(
            self.num_ranks * chunk_size, dtype=torch.float32, device=self.device
        )
        output_tensor = torch.zeros(chunk_size, dtype=torch.float32, device=self.device)
        seed = torch.tensor([42 + self.rank], dtype=torch.int64, device=self.device)

        work = self.ncclx_backend.reduce_scatter_quantized(
            output_tensor,
            input_tensor,
            torchcomms.ReduceOp.SUM,
            seed,
            async_op=False,
        )
        work.wait()

        # Expected: each element should be approximately num_ranks
        # (exact for SUM of ones since 1.0 is exactly representable in BF16)
        expected = torch.full(
            (chunk_size,),
            float(self.num_ranks),
            dtype=torch.float32,
            device=self.device,
        )
        torch.testing.assert_close(
            output_tensor,
            expected,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Rank {self.rank}: reduce_scatter_quantized SUM result mismatch",
        )

    # def test_reduce_scatter_quantized_avg(self):
    #     """Test reduce-scatter quantized with AVG reduction.

    #     Each rank contributes rank-specific values. After AVG, each element
    #     should be approximately the average across all ranks.
    #     """
    #     chunk_size = 1024
    #     # Each rank fills its input with (rank + 1.0)
    #     fill_value = float(self.rank + 1)
    #     input_tensor = torch.full(
    #         (self.num_ranks * chunk_size,),
    #         fill_value,
    #         dtype=torch.float32,
    #         device=self.device,
    #     )
    #     output_tensor = torch.zeros(chunk_size, dtype=torch.float32, device=self.device)
    #     seed = torch.tensor([123 + self.rank], dtype=torch.int64, device=self.device)

    #     work = self.ncclx_backend.reduce_scatter_quantized(
    #         output_tensor,
    #         input_tensor,
    #         torchcomms.ReduceOp.AVG,
    #         seed,
    #         async_op=False,
    #     )
    #     work.wait()

    #     # Expected: average of (1, 2, ..., num_ranks) = (num_ranks + 1) / 2
    #     expected_value = (self.num_ranks + 1.0) / 2.0
    #     expected = torch.full(
    #         (chunk_size,), expected_value, dtype=torch.float32, device=self.device
    #     )
    #     torch.testing.assert_close(
    #         output_tensor,
    #         expected,
    #         atol=0.5,
    #         rtol=0.1,
    #         msg=f"Rank {self.rank}: reduce_scatter_quantized AVG result mismatch",
    #     )

    def test_reduce_scatter_quantized_async(self):
        """Test that async_op=True returns a valid work handle that can be waited on."""
        chunk_size = 512
        input_tensor = torch.ones(
            self.num_ranks * chunk_size, dtype=torch.float32, device=self.device
        )
        output_tensor = torch.zeros(chunk_size, dtype=torch.float32, device=self.device)
        seed = torch.tensor([7], dtype=torch.int64, device=self.device)

        work = self.ncclx_backend.reduce_scatter_quantized(
            output_tensor,
            input_tensor,
            torchcomms.ReduceOp.SUM,
            seed,
            async_op=True,
        )
        self.assertIsNotNone(work)
        work.wait()

        expected = torch.full(
            (chunk_size,),
            float(self.num_ranks),
            dtype=torch.float32,
            device=self.device,
        )
        torch.testing.assert_close(
            output_tensor,
            expected,
            atol=1e-2,
            rtol=1e-2,
            msg=f"Rank {self.rank}: async reduce_scatter_quantized result mismatch",
        )

    def test_reduce_scatter_quantized_correctness(self):
        """Compare quantized reduce-scatter against standard reduce-scatter.

        The quantized version should produce results close to the non-quantized
        version, with differences bounded by BF16 stochastic rounding error.
        """
        chunk_size = 2048
        # Use random data to exercise stochastic rounding
        torch.manual_seed(self.rank * 1000 + 42)
        input_tensor = torch.randn(
            self.num_ranks * chunk_size, dtype=torch.float32, device=self.device
        )
        output_quantized = torch.zeros(
            chunk_size, dtype=torch.float32, device=self.device
        )
        output_reference = torch.zeros(
            chunk_size, dtype=torch.float32, device=self.device
        )
        seed = torch.tensor([999 + self.rank], dtype=torch.int64, device=self.device)

        # Run quantized reduce-scatter
        work_q = self.ncclx_backend.reduce_scatter_quantized(
            output_quantized,
            input_tensor,
            torchcomms.ReduceOp.SUM,
            seed,
            async_op=False,
        )
        work_q.wait()

        # Run standard reduce-scatter for reference
        self.torchcomm.reduce_scatter_single(
            output_reference,
            input_tensor,
            torchcomms.ReduceOp.SUM,
            async_op=False,
        )

        # They should be close but not identical due to stochastic rounding.
        # BF16 has ~7 bits of mantissa, so relative error can be up to ~1%.
        # With multiple PAT rounds of rounding, tolerance needs to be a bit
        # wider.
        torch.testing.assert_close(
            output_quantized,
            output_reference,
            atol=0.5,
            rtol=0.05,
            msg=f"Rank {self.rank}: quantized vs reference reduce_scatter mismatch",
        )


if __name__ == "__main__":
    unittest.main()
