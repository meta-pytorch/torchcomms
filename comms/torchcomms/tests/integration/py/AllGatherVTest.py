#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import os
import unittest

import torch
from torchcomms.tests.integration.py.TorchCommTestHelpers import (
    get_dtype_name,
    TorchCommTestWrapper,
)


class AllGatherVTest(unittest.TestCase):
    """Test class for all_gather_v operations in TorchComm."""

    # Class variables for test parameters
    counts = [0, 4, 1024, 1024 * 1024]
    dtypes = [torch.float, torch.int, torch.int8]
    if os.environ.get("TEST_BACKEND") == "xccl":
        counts = [4, 1024, 1024 * 1024]
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

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") != "ncclx",
        "Skipping all_gather_v test for non-NCCLX backend",
    )
    def _sync_all_gather_v(self, count, dtype):
        """Test synchronous all_gather_v with work object."""
        print(
            f"Testing sync all_gather_v with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Create input and output tensors
        counts = [count] * self.num_ranks
        for i in range(self.num_ranks):
            counts[i] = counts[i] + i
        input_tensor = self._create_input_tensor(counts[self.rank], dtype)
        output_tensors = self._create_output_tensors(counts, dtype)

        # Call all_gather
        work = self.torchcomm.all_gather_v(output_tensors, input_tensor, False)
        work.wait()

        # Verify the results
        self._verify_results(output_tensors)

    def _create_input_tensor(self, count, dtype):
        """Create input tensor with rank-specific values."""
        options = {"dtype": dtype, "device": self.device}
        if dtype == torch.float or dtype == torch.bfloat16:
            return torch.ones(count, **options) * float(self.rank + 1)
        elif dtype == torch.int:
            return torch.ones(count, **options) * int(self.rank + 1)
        elif dtype == torch.int8:
            return torch.ones(count, **options) * int(self.rank + 1)
        return None

    def _create_output_tensors(self, count, dtype):
        """Create output tensors to store results."""
        options = {"dtype": dtype, "device": self.device}
        output_tensors = []
        for i in range(self.num_ranks):
            output_tensors.append(torch.zeros(count[i], **options))
        return output_tensors

    def _verify_results(self, output_tensors):
        """Verify the results of the all_gather operation."""
        for i in range(self.num_ranks):
            # Extract dtype from the tensor
            dtype = output_tensors[i].dtype
            count = output_tensors[i].numel()

            # Expected value for this tensor
            if dtype == torch.float:
                expected = torch.ones(count, dtype=dtype) * float(i + 1)
                self.assertTrue(
                    torch.allclose(output_tensors[i].cpu(), expected),
                    f"Tensors not close enough for rank {i} tensor",
                )
            else:
                expected = torch.ones(count, dtype=dtype) * int(i + 1)
                self.assertTrue(
                    torch.equal(output_tensors[i].cpu(), expected),
                    f"Tensors not equal for rank {i} tensor",
                )

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") != "ncclx",
        "Skipping all_gather_v test for non-NCCLX backends",
    )
    def test_sync_all_gather_v(self):
        """Test synchronous all_gather with work object."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._sync_all_gather_v(count, dtype)


if __name__ == "__main__":
    unittest.main()
