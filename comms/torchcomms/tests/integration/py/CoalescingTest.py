#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for coalesced collective operations in TorchComm."""

import logging
import unittest

import torch
import torchcomms
from torchcomms import ReduceOp
from torchcomms.tests.integration.py.TorchCommTestHelpers import TorchCommTestWrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CoalescingTest(unittest.TestCase):
    """Test class for coalesced collective operations."""

    def get_wrapper(self):
        return TorchCommTestWrapper()


    def setUp(self):
        """Set up test environment before each test."""

        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

    def tearDown(self):
        """Clean up after each test."""

        self.torchcomm = None
        self.wrapper = None

    def test_coalesced_all_reduce_basic(self):
        """Test basic coalesced all_reduce with multiple tensors."""
        # Create input tensors with rank-specific values
        tensor1 = torch.full((10,), float(self.rank + 1), device=self.device)
        tensor2 = torch.full((20,), float(self.rank + 1), device=self.device)

        # Coalesce two all_reduce operations
        self.torchcomm.start_coalescing()
        self.torchcomm.all_reduce(tensor1, ReduceOp.SUM, async_op=True)
        self.torchcomm.all_reduce(tensor2, ReduceOp.SUM, async_op=True)
        work = self.torchcomm.end_coalescing()

        # Wait for completion
        if work is not None:
            work.wait()

        # Verify results - sum of ranks 0..n-1 is n*(n+1)/2 for values rank+1
        expected_sum = sum(range(1, self.num_ranks + 1))
        self.assertTrue(
            torch.allclose(tensor1, torch.full_like(tensor1, expected_sum)),
            f"tensor1 mismatch: expected {expected_sum}, got {tensor1[0].item()}",
        )
        self.assertTrue(
            torch.allclose(tensor2, torch.full_like(tensor2, expected_sum)),
            f"tensor2 mismatch: expected {expected_sum}, got {tensor2[0].item()}",
        )

    def test_coalesced_all_reduce_context_manager(self):
        """Test coalesced all_reduce using context manager."""
        tensor1 = torch.full((10,), float(self.rank + 1), device=self.device)
        tensor2 = torch.full((20,), float(self.rank + 1), device=self.device)
        tensor3 = torch.full((15,), float(self.rank + 1), device=self.device)

        with torchcomms.coalesce(self.torchcomm) as cm:
            self.torchcomm.all_reduce(tensor1, ReduceOp.SUM, async_op=True)
            self.torchcomm.all_reduce(tensor2, ReduceOp.SUM, async_op=True)
            self.torchcomm.all_reduce(tensor3, ReduceOp.SUM, async_op=True)

        # Context manager should have populated work and tensors
        self.assertIsNotNone(cm.work)

        # Wait using the manager
        cm.wait()

        # Verify results
        expected_sum = sum(range(1, self.num_ranks + 1))
        for t in [tensor1, tensor2, tensor3]:
            self.assertTrue(
                torch.allclose(t, torch.full_like(t, expected_sum)),
                f"tensor mismatch: expected {expected_sum}, got {t[0].item()}",
            )

    def test_coalesced_mixed_collectives(self):
        """Test coalescing with different collective types."""
        # Input tensors
        reduce_tensor = torch.full((10,), float(self.rank + 1), device=self.device)
        gather_input = torch.full((5,), float(self.rank + 1), device=self.device)
        gather_output = torch.empty(5 * self.num_ranks, device=self.device)
        scatter_input = torch.full((5,), float(self.rank + 1), device=self.device)
        scatter_output = torch.empty(5 * self.num_ranks, device=self.device)

        with torchcomms.coalesce(self.torchcomm) as cm:
            self.torchcomm.all_reduce(reduce_tensor, ReduceOp.SUM, async_op=True)
            self.torchcomm.all_gather_single(
                scatter_output, scatter_input, async_op=True
            )

        cm.wait()

        # Verify all_reduce result
        expected_sum = sum(range(1, self.num_ranks + 1))
        self.assertTrue(
            torch.allclose(reduce_tensor, torch.full_like(reduce_tensor, expected_sum)),
            f"reduce_tensor mismatch: expected {expected_sum}",
        )

        # Verify all_gather result
        for i in range(self.num_ranks):
            expected = torch.full((5,), float(i + 1), device=self.device)
            actual = scatter_output[i * 5 : (i + 1) * 5]
            self.assertTrue(
                torch.allclose(actual, expected),
                f"all_gather mismatch at rank {i}: expected {i + 1}, got {actual[0].item()}",
            )

    def test_coalesced_empty_block(self):
        """Test coalescing with no operations inside."""
        self.torchcomm.start_coalescing()
        work = self.torchcomm.end_coalescing()

        # Should return None when no operations were coalesced
        self.assertIsNone(work)

    def test_coalesced_single_operation(self):
        """Test coalescing with just one operation."""
        tensor = torch.full((10,), float(self.rank + 1), device=self.device)

        with torchcomms.coalesce(self.torchcomm) as cm:
            self.torchcomm.all_reduce(tensor, ReduceOp.SUM, async_op=True)

        cm.wait()

        expected_sum = sum(range(1, self.num_ranks + 1))
        self.assertTrue(
            torch.allclose(tensor, torch.full_like(tensor, expected_sum)),
            f"tensor mismatch: expected {expected_sum}, got {tensor[0].item()}",
        )

    def test_coalesced_large_batch(self):
        """Test coalescing with many operations."""
        num_tensors = 10
        tensors = [
            torch.full((100,), float(self.rank + 1), device=self.device)
            for _ in range(num_tensors)
        ]

        with torchcomms.coalesce(self.torchcomm) as cm:
            for t in tensors:
                self.torchcomm.all_reduce(t, ReduceOp.SUM, async_op=True)

        cm.wait()

        expected_sum = sum(range(1, self.num_ranks + 1))
        for i, t in enumerate(tensors):
            self.assertTrue(
                torch.allclose(t, torch.full_like(t, expected_sum)),
                f"tensor {i} mismatch: expected {expected_sum}, got {t[0].item()}",
            )

    def test_coalesced_sync_ops(self):
        """Test that sync ops inside coalescing block complete at block exit."""
        tensor1 = torch.full((10,), float(self.rank + 1), device=self.device)
        tensor2 = torch.full((20,), float(self.rank + 1), device=self.device)

        with torchcomms.coalesce(self.torchcomm):
            # Using async_op=False - ops should complete at block exit
            self.torchcomm.all_reduce(tensor1, ReduceOp.SUM, async_op=False)
            self.torchcomm.all_reduce(tensor2, ReduceOp.SUM, async_op=False)

        # No cm.wait() needed - sync ops complete at block exit
        # Verify results are already available
        expected_sum = sum(range(1, self.num_ranks + 1))
        self.assertTrue(
            torch.allclose(tensor1, torch.full_like(tensor1, expected_sum)),
            f"tensor1 mismatch: expected {expected_sum}, got {tensor1[0].item()}",
        )
        self.assertTrue(
            torch.allclose(tensor2, torch.full_like(tensor2, expected_sum)),
            f"tensor2 mismatch: expected {expected_sum}, got {tensor2[0].item()}",
        )
        logging.info("[test] test_coalesced_sync_ops passed")

    def test_coalesced_reduce_scatter(self):
        """Test coalesced reduce_scatter operations."""
        # Each rank contributes data for all ranks
        input_tensor = torch.full(
            (10 * self.num_ranks,), float(self.rank + 1), device=self.device
        )
        output_tensor = torch.empty(10, device=self.device)

        with torchcomms.coalesce(self.torchcomm) as cm:
            self.torchcomm.reduce_scatter_single(
                output_tensor, input_tensor, ReduceOp.SUM, async_op=True
            )

        cm.wait()

        # Each rank receives sum of all ranks' contributions
        expected_sum = sum(range(1, self.num_ranks + 1))
        self.assertTrue(
            torch.allclose(output_tensor, torch.full_like(output_tensor, expected_sum)),
            f"reduce_scatter mismatch: expected {expected_sum}, got {output_tensor[0].item()}",
        )

    def test_error_end_without_start(self):
        """Test that end_coalescing without start_coalescing raises error."""
        with self.assertRaises(RuntimeError):
            self.torchcomm.end_coalescing()

    def test_error_nested_coalescing(self):
        """Test that nested coalescing raises error."""
        self.torchcomm.start_coalescing()
        try:
            with self.assertRaises(RuntimeError):
                self.torchcomm.start_coalescing()
        finally:
            # Clean up
            self.torchcomm.end_coalescing()


    def test_coalescing_manager_double_wait_raises(self):
        """Test that calling wait() twice on a CoalescingManager raises an error."""
        tensor = torch.full((10,), float(self.rank + 1), device=self.device)

        with torchcomms.coalesce(self.torchcomm) as cm:
            self.torchcomm.all_reduce(tensor, ReduceOp.SUM, async_op=True)

        # First wait should succeed
        cm.wait()

        # Second wait should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            cm.wait()

        self.assertIn("already been called", str(context.exception))
        logging.info("[test] test_coalescing_manager_double_wait_raises passed")

    def test_unsupported_op_in_coalescing_block_raises(self):
        """Test that calling an unsupported operation inside a coalescing block raises an error."""
        # barrier does not support coalescing because it uses internal groupStart/groupEnd
        with self.assertRaises(RuntimeError) as context:
            with torchcomms.coalesce(self.torchcomm):
                self.torchcomm.barrier(async_op=True)

        self.assertIn("does not support coalescing", str(context.exception))
        logging.info("[test] test_unsupported_op_in_coalescing_block_raises passed")


if __name__ == "__main__":
    unittest.main()
