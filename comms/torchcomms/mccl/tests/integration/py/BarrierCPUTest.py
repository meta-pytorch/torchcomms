#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    TorchCommTestWrapper,
)


class BarrierCPUTest(unittest.TestCase):
    """Test class for CPU barrier operations in TorchCommMCCL."""

    def setUp(self):
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
        self.wrapper = TorchCommTestWrapper(hints={"use_cpu_barrier": "true"})
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()

    def tearDown(self):
        self.torchcomm = None
        self.wrapper = None

    def test_barrier_cpu_sync(self):
        """Test a single CPU barrier across all ranks."""
        work = self.torchcomm.barrier(False)
        self.assertTrue(work.is_completed())

    def test_barrier_cpu_async(self):
        """Test a single async CPU barrier across all ranks."""
        work = self.torchcomm.barrier(True)
        work.wait()
        self.assertTrue(work.is_completed())

    def test_barrier_cpu_multiple(self):
        """Test multiple CPU barriers in sequence."""
        for i in range(5):
            work = self.torchcomm.barrier(True)
            work.wait()
            self.assertTrue(
                work.is_completed(),
                f"CPU barrier iteration {i} did not complete",
            )


if __name__ == "__main__":
    unittest.main()
