#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

import torch
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    TorchCommTestWrapper,
)


class AllGatherCPUTest(unittest.TestCase):
    """Test class for CPU allGather operations in TorchCommMCCL."""

    dtypes = [torch.float, torch.int, torch.int8]

    def setUp(self):
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
        self.wrapper = TorchCommTestWrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()

    def tearDown(self):
        self.torchcomm = None
        self.wrapper = None

    def _all_gather_cpu_sync(self, dtype):
        """Test sync CPU allGather: each rank contributes its rank value."""
        print(f"Testing sync CPU allGather with dtype={dtype}")

        input_tensor = torch.tensor([self.rank], dtype=dtype, device="cpu")
        output_tensor = torch.zeros(self.num_ranks, dtype=dtype, device="cpu")

        work = self.torchcomm.all_gather_single(output_tensor, input_tensor, False)
        self.assertTrue(work.is_completed())

        expected = torch.arange(self.num_ranks, dtype=dtype, device="cpu")
        self.assertTrue(
            torch.equal(output_tensor, expected),
            f"Rank {self.rank}: expected {expected} but got {output_tensor}",
        )

    def _all_gather_cpu_async(self, dtype):
        """Test async CPU allGather: each rank contributes its rank value."""
        print(f"Testing async CPU allGather with dtype={dtype}")

        input_tensor = torch.tensor([self.rank], dtype=dtype, device="cpu")
        output_tensor = torch.zeros(self.num_ranks, dtype=dtype, device="cpu")

        work = self.torchcomm.all_gather_single(output_tensor, input_tensor, True)
        work.wait()
        self.assertTrue(work.is_completed())

        expected = torch.arange(self.num_ranks, dtype=dtype, device="cpu")
        self.assertTrue(
            torch.equal(output_tensor, expected),
            f"Rank {self.rank}: expected {expected} but got {output_tensor}",
        )

    def test_all_gather_cpu_sync(self):
        """Test sync CPU allGather with multiple data types."""
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self._all_gather_cpu_sync(dtype)

    def test_all_gather_cpu_async(self):
        """Test async CPU allGather with multiple data types."""
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self._all_gather_cpu_async(dtype)


if __name__ == "__main__":
    unittest.main()
