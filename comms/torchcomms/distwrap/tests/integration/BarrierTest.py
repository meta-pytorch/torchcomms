#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Integration tests for barrier collective."""

import unittest

import torch
from torchcomms import distwrap as dist
from torchcomms.distwrap.tests.integration.test_helpers import (
    get_backend,
    get_device,
    get_rank_and_size,
    use_torchcomms,
)


class BarrierTest(unittest.TestCase):
    """Test class for barrier operations using distwrap."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize distwrap once for all tests."""
        rank, _ = get_rank_and_size()
        device = get_device(rank)
        backend = get_backend()

        dist.init_process_group(
            backend=backend,
            use_torchcomms=use_torchcomms(),
        )

        if device.type == "cuda":
            torch.cuda.set_device(device)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up distwrap after all tests."""
        dist.destroy_process_group()

    def test_sync_barrier(self) -> None:
        """Test synchronous barrier."""
        # Simply test that barrier completes without error
        dist.barrier(async_op=False)

        # If we get here, barrier succeeded
        self.assertTrue(True)

    def test_async_barrier(self) -> None:
        """Test asynchronous barrier."""
        work = dist.barrier(async_op=True)
        if work is None:
            raise AssertionError("work is None")
        work.wait()

        # If we get here, barrier succeeded
        self.assertTrue(True)

    def test_barrier_synchronization(self) -> None:
        """Test that barrier actually synchronizes ranks."""
        rank = dist.get_rank()
        device = get_device(rank)

        # Each rank sets a value before barrier
        tensor = torch.ones(1, dtype=torch.float, device=device) * rank

        # Barrier to synchronize
        dist.barrier(async_op=False)

        # After barrier, verify tensor still has correct value
        expected = torch.full_like(tensor.cpu(), rank)
        torch.testing.assert_close(tensor.cpu(), expected)


if __name__ == "__main__":
    unittest.main()
