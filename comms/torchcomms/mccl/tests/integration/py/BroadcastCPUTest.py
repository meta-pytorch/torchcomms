#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import os
import unittest

import torch
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    TorchCommTestWrapper,
)


class BroadcastCPUTest(unittest.TestCase):
    """Test class for CPU broadcast operations in TorchCommMCCL."""

    counts = [4, 1024, 1024 * 1024]
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

    def _verify_broadcast_results(self, tensor, expected_value, count, dtype):
        expected = torch.ones(count, dtype=dtype, device="cpu") * expected_value
        if dtype == torch.float:
            self.assertTrue(
                torch.allclose(tensor, expected),
                f"CPU broadcast tensors not close enough for count={count}",
            )
        else:
            self.assertTrue(
                torch.equal(tensor, expected),
                f"CPU broadcast tensors not equal for count={count}",
            )

    def _cpu_broadcast_no_register(self, count, dtype):
        """Test CPU broadcast without registering tensor memory."""
        print(
            f"Testing CPU broadcast without register with count={count} and dtype={dtype}"
        )

        root_rank = 0
        root_value = 99

        if self.rank == root_rank:
            tensor = torch.ones(count, dtype=dtype, device="cpu") * root_value
        else:
            tensor = torch.zeros(count, dtype=dtype, device="cpu")

        work = self.torchcomm.broadcast(tensor, root_rank, False)
        self.assertTrue(work.is_completed())
        self._verify_broadcast_results(tensor, root_value, count, dtype)

    def _cpu_broadcast_no_register_async(self, count, dtype):
        """Test async CPU broadcast without registering tensor memory."""
        print(
            f"Testing async CPU broadcast without register with count={count} and dtype={dtype}"
        )

        root_rank = 0
        root_value = 99

        if self.rank == root_rank:
            tensor = torch.ones(count, dtype=dtype, device="cpu") * root_value
        else:
            tensor = torch.zeros(count, dtype=dtype, device="cpu")

        work = self.torchcomm.broadcast(tensor, root_rank, True)
        work.wait()
        self.assertTrue(work.is_completed())
        self._verify_broadcast_results(tensor, root_value, count, dtype)

    def _cpu_broadcast_with_register(self, count, dtype):
        """Test CPU broadcast with tensor memory registration."""
        print(
            f"Testing CPU broadcast with register with count={count} and dtype={dtype}"
        )

        root_rank = 0
        root_value = 99

        if self.rank == root_rank:
            tensor = torch.ones(count, dtype=dtype, device="cpu") * root_value
        else:
            tensor = torch.zeros(count, dtype=dtype, device="cpu")

        # Register tensor memory before broadcast using public TorchComm API
        self.torchcomm.tensor_register(tensor)

        try:
            work = self.torchcomm.broadcast(tensor, root_rank, False)
            self.assertTrue(work.is_completed())
            self._verify_broadcast_results(tensor, root_value, count, dtype)
        finally:
            # Deregister tensor memory after broadcast
            self.torchcomm.tensor_deregister(tensor)

    def test_cpu_broadcast_no_register(self):
        """Test CPU broadcast without registering tensor memory."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._cpu_broadcast_no_register(count, dtype)

    def test_cpu_broadcast_no_register_async(self):
        """Test async CPU broadcast without registering tensor memory."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._cpu_broadcast_no_register_async(count, dtype)

    def test_cpu_broadcast_with_register(self):
        """Test CPU broadcast with tensor memory registration."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._cpu_broadcast_with_register(count, dtype)


if __name__ == "__main__":
    unittest.main()
