#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

import torch
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    TorchCommTestWrapper,
)


class GatherSingleCPUTest(unittest.TestCase):
    """Test class for CPU gather_single operations in TorchCommMCCL."""

    def setUp(self) -> None:
        self.wrapper = TorchCommTestWrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()

    def tearDown(self):
        self.torchcomm = None
        self.wrapper = None

    def test_gather_single_cpu_sync(self) -> None:
        """Test sync CPU gather_single: each rank contributes its rank value to root 0."""
        root = 0

        input_tensor = torch.tensor([self.rank], dtype=torch.int32, device="cpu")
        output_tensor = torch.zeros(self.num_ranks, dtype=torch.int32, device="cpu")

        work = self.torchcomm.gather_single(output_tensor, input_tensor, root, False)
        self.assertTrue(work.is_completed())

        if self.rank == root:
            expected = torch.arange(self.num_ranks, dtype=torch.int32, device="cpu")
            self.assertTrue(
                torch.equal(output_tensor, expected),
                f"Root rank {self.rank}: expected {expected} but got {output_tensor}",
            )

    def test_gather_single_cpu_async(self) -> None:
        """Test async CPU gather_single: each rank contributes its rank value to root 0."""
        root = 0

        input_tensor = torch.tensor([self.rank], dtype=torch.int32, device="cpu")
        output_tensor = torch.zeros(self.num_ranks, dtype=torch.int32, device="cpu")

        work = self.torchcomm.gather_single(output_tensor, input_tensor, root, True)
        work.wait()
        self.assertTrue(work.is_completed())

        if self.rank == root:
            expected = torch.arange(self.num_ranks, dtype=torch.int32, device="cpu")
            self.assertTrue(
                torch.equal(output_tensor, expected),
                f"Root rank {self.rank}: expected {expected} but got {output_tensor}",
            )

    def test_gather_single_cpu_non_zero_root(self) -> None:
        """Test CPU gather_single with non-zero root."""
        root = min(2, self.num_ranks - 1)

        input_tensor = torch.tensor(
            [self.rank * 10 + 5], dtype=torch.int32, device="cpu"
        )
        output_tensor = torch.zeros(self.num_ranks, dtype=torch.int32, device="cpu")

        work = self.torchcomm.gather_single(output_tensor, input_tensor, root, False)
        self.assertTrue(work.is_completed())

        if self.rank == root:
            expected = torch.tensor(
                [i * 10 + 5 for i in range(self.num_ranks)],
                dtype=torch.int32,
                device="cpu",
            )
            self.assertTrue(
                torch.equal(output_tensor, expected),
                f"Root rank {self.rank}: expected {expected} but got {output_tensor}",
            )


if __name__ == "__main__":
    unittest.main()
