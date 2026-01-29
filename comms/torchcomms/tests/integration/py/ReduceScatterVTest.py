#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import unittest

import torch
from torchcomms import ReduceOp
from torchcomms.tests.integration.py.TorchCommTestHelpers import (
    get_dtype_name,
    get_op_name,
    TorchCommTestWrapper,
)


class ReduceScatterVTest(unittest.TestCase):
    """Test class for reduce_scatter_v operations in TorchComm."""

    # Class variables for test parameters
    counts = [0, 4, 1024, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8]
    ops = [ReduceOp.SUM, ReduceOp.MAX]
    num_replays = 4

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
        # Explicitly reset the TorchComm object to ensure proper cleanup
        self.torchcomm = None
        self.wrapper = None

    def _sync_reduce_scatter_v(self, count, dtype, op):
        """Test synchronous reduce_scatter_v with work object."""
        print(
            f"Testing sync reduce_scatter_v with count={count}, dtype={get_dtype_name(dtype)}, and op={get_op_name(op)}"
        )

        # Create input and output tensors
        counts = [count] * self.num_ranks
        for i in range(self.num_ranks):
            counts[i] = counts[i] + i
        input_tensors = self._create_input_tensors(counts, dtype)
        output_tensor = self._create_output_tensor(counts[self.rank], dtype)

        # Call reduce_scatter_v
        work = self.torchcomm.reduce_scatter_v(output_tensor, input_tensors, op, False)
        work.wait()

        # Verify the results
        self._verify_results(output_tensor, op)

    def _create_input_tensors(self, count, dtype):
        """Create input tensors with rank-specific values."""
        input_tensors = []
        options = {"dtype": dtype, "device": self.device}

        for r in range(self.num_ranks):
            # Each tensor has rank-specific values
            element_count = count[r]

            if dtype == torch.float or dtype == torch.bfloat16:
                tensor = torch.ones(element_count, **options) * float(r + 1)
            elif dtype == torch.int:
                tensor = torch.ones(element_count, **options) * int(r + 1)
            elif dtype == torch.int8:
                tensor = torch.ones(element_count, **options) * int(r + 1)
            input_tensors.append(tensor)

        return input_tensors

    def _create_output_tensor(self, count, dtype):
        """Create output tensor to store results."""
        options = {"dtype": dtype, "device": self.device}
        return torch.zeros(count, **options)

    def _calculate_expected_result(self, op):
        """Calculate expected result based on operation."""
        if op == ReduceOp.SUM:
            return self.num_ranks * (self.rank + 1)
        elif op == ReduceOp.MAX:
            return self.rank + 1
        else:
            raise RuntimeError("Unsupported reduce operation")

    def _verify_results(self, output_tensor, op):
        """Verify the results of the reduce_scatter_v operation."""
        # Calculate expected result
        expected = self._calculate_expected_result(op)

        # Compare output with expected tensor
        description = f"reduce_scatter_v with op {get_op_name(op)}"

        # Create expected tensor with the same size and dtype as output
        if output_tensor.dtype == torch.float:
            expected_tensor = torch.full_like(output_tensor.cpu(), float(expected))
            self.assertTrue(
                torch.allclose(output_tensor.cpu(), expected_tensor),
                f"Tensors not close enough for {description}",
            )
        else:
            expected_tensor = torch.full_like(output_tensor.cpu(), expected)
            self.assertTrue(
                torch.equal(output_tensor.cpu(), expected_tensor),
                f"Tensors not equal for {description}",
            )

    def test_sync_reduce_scatter_v(self):
        """Test synchronous reduce_scatter_v with work object."""
        for count, dtype, op in itertools.product(self.counts, self.dtypes, self.ops):
            with self.subTest(count=count, dtype=dtype, op=op):
                self._sync_reduce_scatter_v(count, dtype, op)


if __name__ == "__main__":
    unittest.main()
