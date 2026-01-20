#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Check that large allocations that span multiple physical caching allocator segment
chunks are correctly registered. This is the behavior when expandable segments is set.
"""

import os
import unittest

os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_DEBUG_SUBSYS", "ALLOC")
os.environ.setdefault("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal")

import torch
from torchcomms import ReduceOp
from torchcomms.tests.integration.py.TorchCommTestHelpers import TorchCommTestWrapper


class ExpandableSegmentsTest(unittest.TestCase):
    """Test class for expandable segments multi-segment registration."""

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

    def tearDown(self):
        self.torchcomm = None
        self.wrapper = None

    def _run_large_allocation_test(self):
        """
        Test that large allocations spanning multiple physical memory allocations work correctly.

        With expandable segments enabled, PyTorch allocates memory in 20MB segment chunks.
        A 100MB allocation should span 5 physical 20 MB segment chunks. This test verifies that:
        1. The CCA hook receives the SEGMENT_MAP event for the full range
        2. ctran's pinRange() discovers all underlying physical segments and caches them
        3. At collective time, the elements of the segment cache backing the input tensor are
           correctly registered with NCCL
        """
        count = 100 * 1024 * 1024
        dtype = torch.uint8

        # Create input tensor - this triggers SEGMENT_MAP for a ~100MB allocation
        input_tensor = torch.ones(count, dtype=dtype, device=self.device) * float(
            self.rank + 1
        )

        # Perform all_reduce - this exercises the full ctran registration path
        work = self.torchcomm.all_reduce(input_tensor, ReduceOp.SUM, False)
        work.wait()

        # Check collective result
        expected = self.num_ranks * (self.num_ranks + 1) // 2
        expected_tensor = torch.full_like(input_tensor.cpu(), float(expected))
        torch.testing.assert_close(input_tensor.cpu(), expected_tensor)

    def test_async_registration(self):
        os.environ["NCCL_CTRAN_REGISTER"] = "async"
        self._run_large_allocation_test()

    def test_eager_registration(self):
        os.environ["NCCL_CTRAN_REGISTER"] = "eager"
        self._run_large_allocation_test()


if __name__ == "__main__":
    unittest.main()
