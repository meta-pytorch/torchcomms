#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Integration tests for memory pool operations (get_mem_allocator, register_mem_pool)."""

import os
import unittest

import torch
from torchcomms import distwrap as dist
from torchcomms.distwrap.tests.integration.test_helpers import (
    get_backend,
    get_device,
    get_rank_and_size,
    use_torchcomms,
)


class MemPoolTest(unittest.TestCase):
    """Test class for memory pool operations using distwrap."""

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

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") == "gloo",
        "Gloo backend does not support getMemAllocator",
    )
    def test_get_mem_allocator_returns_allocator(self) -> None:
        """Test that get_mem_allocator returns a valid memory allocator."""
        rank = dist.get_rank()
        device = get_device(rank)

        # Get the memory allocator for the default group
        allocator = dist.get_mem_allocator(None, device)

        # The allocator should not be None
        self.assertIsNotNone(allocator)

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") == "gloo",
        "Gloo backend does not support getMemAllocator",
    )
    def test_register_mem_pool_no_error(self) -> None:
        """Test that register_mem_pool completes without error."""
        rank = dist.get_rank()
        device = get_device(rank)

        # Get the memory allocator first
        allocator = dist.get_mem_allocator(None, device)

        # Create a simple memory pool (using the allocator's pool if available)
        # For this test, we just verify that register_mem_pool doesn't raise
        # when called with the allocator's existing pool
        if hasattr(allocator, "pool"):
            pool = allocator.pool
            # This should not raise an error
            dist.register_mem_pool(None, device, pool)


if __name__ == "__main__":
    unittest.main()
