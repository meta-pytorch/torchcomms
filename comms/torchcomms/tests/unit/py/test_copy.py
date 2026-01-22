#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import os
import unittest

import torch
from torchcomms._comms import ReduceOp
from torchcomms.tests.integration.py.TorchCommTestHelpers import TorchCommTestWrapper


class TestReduceOpCopy(unittest.TestCase):
    """Tests for ReduceOp copy/deepcopy."""

    def test_reduceop_copy(self):
        """Test that ReduceOp copy creates a new object with same value."""
        op = ReduceOp.SUM
        op_copy = copy.copy(op)

        self.assertEqual(op.type, op_copy.type)

    def test_reduceop_deepcopy(self):
        """Test that ReduceOp deepcopy creates a new object and updates memo."""
        op = ReduceOp.SUM
        memo = {}
        op_copy = copy.deepcopy(op, memo)

        self.assertEqual(op.type, op_copy.type)
        self.assertIn(id(op), memo)

    def test_reduceop_deepcopy_memo_first(self):
        """Test that ReduceOp deepcopy returns memoized object if present."""
        op = ReduceOp.SUM

        sentinel = object()
        memo = {id(op): sentinel}

        result = copy.deepcopy(op, memo)
        self.assertIs(result, sentinel)


class TestCommCopy(unittest.TestCase):
    """Tests for TorchComm copy/deepcopy."""

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

    def test_comm_copy(self):
        """Test that TorchComm copy returns the same object."""
        comm_copy = copy.copy(self.torchcomm)
        self.assertIs(self.torchcomm, comm_copy)

    def test_comm_deepcopy(self):
        """Test that TorchComm deepcopy returns the same object and updates memo."""
        memo = {}
        comm_copy = copy.deepcopy(self.torchcomm, memo)

        self.assertIs(self.torchcomm, comm_copy)
        self.assertIn(id(self.torchcomm), memo)

    def test_comm_deepcopy_memo_first(self):
        """Test that TorchComm deepcopy returns memoized object if present."""
        sentinel = object()
        memo = {id(self.torchcomm): sentinel}

        result = copy.deepcopy(self.torchcomm, memo)
        self.assertIs(result, sentinel)


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

        window.tensor_deregister()
        window_copy.tensor_deregister()

    def test_window_deepcopy_memo_first(self):
        """Test that TorchCommWindow deepcopy returns memoized object if present."""
        window = self.torchcomm.new_window()

        sentinel = object()
        memo = {id(window): sentinel}

        result = copy.deepcopy(window, memo)
        self.assertIs(result, sentinel)


if __name__ == "__main__":
    unittest.main()
