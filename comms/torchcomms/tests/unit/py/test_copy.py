#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import os
import unittest

import torch
from torchcomms.tests.integration.py.TorchCommTestHelpers import TorchCommTestWrapper


@unittest.skipIf(
    os.getenv("TEST_BACKEND") != "ncclx", "Skipping NCCLX-only window tests"
)
class TestWindowCopy(unittest.TestCase):
    """Tests for TorchCommWindow copy/deepcopy.

    These tests require a backend that supports windows (e.g., ncclx).
    """

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
        self.torchcomm = None
        self.wrapper = None

    def test_window_copy(self):
        """Test that TorchCommWindow copy returns the same object."""
        window = self.torchcomm.new_window()
        window_copy = copy.copy(window)

        self.assertIs(window, window_copy)

    def test_window_deepcopy(self):
        """Test that TorchCommWindow deepcopy creates a new window."""
        window = self.torchcomm.new_window()
        memo = {}
        window_copy = copy.deepcopy(window, memo)

        self.assertIsNot(window, window_copy)
        self.assertIn(id(window), memo)

    def test_window_deepcopy_with_tensor(self):
        """Test that TorchCommWindow deepcopy clones the registered tensor."""
        window = self.torchcomm.new_window()

        device = self.torchcomm.get_device()
        tensor = torch.zeros(10, dtype=torch.float32, device=device)
        window.tensor_register(tensor)

        memo = {}
        window_copy = copy.deepcopy(window, memo)

        self.assertIsNot(window, window_copy)
        self.assertIn(id(window), memo)

        cloned_tensor = window_copy.get_tensor()
        self.assertIsNotNone(cloned_tensor)

        self.assertIsNot(tensor, cloned_tensor)
        self.assertTrue(torch.equal(tensor, cloned_tensor))

        tensor[0] = 42.0
        self.assertNotEqual(tensor[0].item(), cloned_tensor[0].item())

        # Cleanup
        window.tensor_deregister()
        window_copy.tensor_deregister()


if __name__ == "__main__":
    unittest.main()
