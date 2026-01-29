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
        A 110MB allocation should span 6 physical 20 MB segment chunks. This test verifies that:
        1. The CCA hook receives the SEGMENT_MAP event for the full range
        2. ctran's pinRange() discovers all underlying physical segments and caches them
        3. At collective time, the elements of the segment cache backing the input tensor are
           correctly registered with NCCL
        """
        count = 110 * 1024 * 1024
        dtype = torch.uint8

        # Create input tensor - this triggers SEGMENT_MAP for a ~110MB allocation
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

    def _run_mixed_allocation_test(self):
        """
        Test that both expandable segments (multi-segment) and regular (single-segment)
        allocations work correctly together.

        This test:
        1. Allocates tensors with expandable segments ON (multi-segment buffers)
        2. Turns off expandable segments via set_allocator_settings
        3. Allocates tensors with expandable segments OFF (single-segment buffers)
        4. Verifies collectives work correctly with both buffer types coexisting
        """
        # Phase 1: Allocate with expandable segments ON (multi-segment)
        # Large allocation spans multiple 20MB segment chunks
        large_tensor_expandable = torch.ones(
            110 * 1024 * 1024, dtype=torch.uint8, device=self.device
        ) * float(self.rank + 1)

        # Small allocation that may fit in single segment chunk
        small_tensor_expandable = torch.ones(
            5 * 1024 * 1024, dtype=torch.uint8, device=self.device
        ) * float(self.rank + 1)

        # Phase 2: Turn off expandable segments
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")

        # Phase 3: Allocate with expandable segments OFF (single-segment)
        large_tensor_regular = torch.ones(
            110 * 1024 * 1024, dtype=torch.uint8, device=self.device
        ) * float(self.rank + 1)

        small_tensor_regular = torch.ones(
            5 * 1024 * 1024, dtype=torch.uint8, device=self.device
        ) * float(self.rank + 1)

        # Phase 4: Run collectives on all tensors (both multi-segment and single-segment)
        work1 = self.torchcomm.all_reduce(large_tensor_expandable, ReduceOp.SUM, False)
        work2 = self.torchcomm.all_reduce(small_tensor_expandable, ReduceOp.SUM, False)
        work3 = self.torchcomm.all_reduce(large_tensor_regular, ReduceOp.SUM, False)
        work4 = self.torchcomm.all_reduce(small_tensor_regular, ReduceOp.SUM, False)
        work1.wait()
        work2.wait()
        work3.wait()
        work4.wait()

        # Verify all results
        expected = self.num_ranks * (self.num_ranks + 1) // 2
        torch.testing.assert_close(
            large_tensor_expandable.cpu(),
            torch.full_like(large_tensor_expandable.cpu(), float(expected)),
        )
        torch.testing.assert_close(
            small_tensor_expandable.cpu(),
            torch.full_like(small_tensor_expandable.cpu(), float(expected)),
        )
        torch.testing.assert_close(
            large_tensor_regular.cpu(),
            torch.full_like(large_tensor_regular.cpu(), float(expected)),
        )
        torch.testing.assert_close(
            small_tensor_regular.cpu(),
            torch.full_like(small_tensor_regular.cpu(), float(expected)),
        )

    @unittest.skipUnless(
        "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "Only runs with expandable segments enabled",
    )
    def test_mixed_allocation_async(self):
        """Test mixed expandable/regular allocations with async registration."""
        os.environ["NCCL_CTRAN_REGISTER"] = "async"
        self._run_mixed_allocation_test()

    @unittest.skipUnless(
        "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "Only runs with expandable segments enabled",
    )
    def test_mixed_allocation_eager(self):
        """Test mixed expandable/regular allocations with eager registration."""
        os.environ["NCCL_CTRAN_REGISTER"] = "eager"
        self._run_mixed_allocation_test()


if __name__ == "__main__":
    unittest.main()
