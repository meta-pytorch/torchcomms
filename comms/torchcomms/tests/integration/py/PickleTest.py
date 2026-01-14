#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Tests for pickle support of TorchComm objects.

These tests verify that TorchComm, TorchCommWindow, CommOptions, and ReduceOp
can be pickled and unpickled correctly, which is required for torch.compile support.
"""

import pickle
import unittest
from datetime import timedelta

import torch
from torchcomms import CommOptions, ReduceOp
from torchcomms.tests.integration.py.TorchCommTestHelpers import TorchCommTestWrapper


class PickleTest(unittest.TestCase):
    """Test class for pickle support of TorchComm objects."""

    def setUp(self):
        """Set up test environment before each test."""
        self.wrapper = TorchCommTestWrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

    def tearDown(self):
        """Clean up after each test."""
        self.torchcomm = None
        self.wrapper = None

    def test_reduce_op_pickle(self):
        """Test that ReduceOp can be pickled and unpickled."""
        print("Testing ReduceOp pickle support")

        # Test basic reduce ops
        for op in [ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN, ReduceOp.PRODUCT]:
            pickled = pickle.dumps(op)
            unpickled = pickle.loads(pickled)
            self.assertEqual(op.type, unpickled.type)

        # Test PREMUL_SUM with float factor
        premul_op = ReduceOp.PREMUL_SUM(2.0)
        pickled = pickle.dumps(premul_op)
        unpickled = pickle.loads(pickled)
        self.assertEqual(premul_op.type, unpickled.type)

        # Test PREMUL_SUM with tensor factor (use CPU tensor to avoid GPU pickle issues)
        tensor_factor = torch.tensor(2.0, device="cpu")
        premul_tensor_op = ReduceOp.PREMUL_SUM(tensor_factor)
        pickled = pickle.dumps(premul_tensor_op)
        unpickled = pickle.loads(pickled)
        self.assertEqual(premul_tensor_op.type, unpickled.type)

    def test_torchcomm_pickle(self):
        """Test that TorchComm can be pickled and unpickled."""
        print("Testing TorchComm pickle support")

        # Get original properties
        orig_backend = self.torchcomm.get_backend()
        orig_device = self.torchcomm.get_device()
        orig_name = self.torchcomm.get_name()
        orig_rank = self.torchcomm.get_rank()
        orig_size = self.torchcomm.get_size()

        # Pickle and unpickle
        pickled = pickle.dumps(self.torchcomm)
        unpickled = pickle.loads(pickled)

        # Verify properties match
        self.assertEqual(orig_backend, unpickled.get_backend())
        self.assertEqual(orig_device, unpickled.get_device())
        self.assertEqual(orig_name, unpickled.get_name())
        self.assertEqual(orig_rank, unpickled.get_rank())
        self.assertEqual(orig_size, unpickled.get_size())

        # Verify the unpickled comm is the same Python object
        self.assertIs(self.torchcomm, unpickled)

        # Delete the original object and unpickle again
        del self.torchcomm
        new_unpickled = pickle.loads(pickled)

        # Verify a new object was created with same properties
        self.assertEqual(orig_backend, new_unpickled.get_backend())
        self.assertEqual(orig_device, new_unpickled.get_device())
        self.assertEqual(orig_name, new_unpickled.get_name())
        self.assertEqual(orig_rank, new_unpickled.get_rank())
        self.assertEqual(orig_size, new_unpickled.get_size())

        # Unpickle again and verify it's the same as the first new object
        another_unpickled = pickle.loads(pickled)
        self.assertIs(new_unpickled, another_unpickled)

    def test_torchcomm_window_pickle(self):
        """Test that TorchCommWindow can be pickled and unpickled."""
        print("Testing TorchCommWindow pickle support")

        window_size = 1024

        # Create a window and register a tensor buffer
        window = self.torchcomm.new_window()
        buf_tensor = torch.empty(window_size, dtype=torch.uint8, device=self.device)
        window.tensor_register(buf_tensor)

        # Sync to ensure window is allocated on all ranks
        self.torchcomm.barrier(False)

        # Get original properties
        orig_size = window.get_size()
        orig_comm_backend = window.get_comm_backend()
        orig_tensor = window.map_remote_tensor(self.rank)

        # Pickle and unpickle
        pickled = pickle.dumps(window)
        unpickled = pickle.loads(pickled)

        # Verify properties match
        self.assertEqual(orig_size, unpickled.get_size())

        # Verify the parent comm backend is the same object
        self.assertIs(orig_comm_backend, unpickled.get_comm_backend())

        # Verify the tensor storage is the same (same window object)
        unpickled_tensor = unpickled.map_remote_tensor(self.rank)
        self.assertTrue(orig_tensor.data_ptr() == unpickled_tensor.data_ptr())

        # Verify the unpickled window is the same Python object
        self.assertIs(window, unpickled)

        # Sync before deleting
        self.torchcomm.barrier(False)

        # Delete the original object and unpickle again
        del window
        del unpickled
        new_unpickled = pickle.loads(pickled)

        # Verify the recreated window has same size (buffer info is preserved)
        self.assertEqual(orig_size, new_unpickled.get_size())

        # Verify the parent comm backend is still the same object
        self.assertIs(orig_comm_backend, new_unpickled.get_comm_backend())

        # Verify the recreated tensor has same shape, dtype, and device
        new_tensor = new_unpickled.map_remote_tensor(self.rank)
        self.assertEqual(orig_tensor.shape, new_tensor.shape)
        self.assertEqual(orig_tensor.dtype, new_tensor.dtype)
        self.assertEqual(orig_tensor.device, new_tensor.device)

        # Unpickle again and verify it's the same as the first new object
        another_unpickled = pickle.loads(pickled)
        self.assertIs(new_unpickled, another_unpickled)

        # Verify tensor storage is the same for repeated unpickle of same recreated window
        another_tensor = another_unpickled.map_remote_tensor(self.rank)
        self.assertTrue(new_tensor.data_ptr() == another_tensor.data_ptr())

        # Cleanup
        del new_unpickled
        del another_unpickled
        self.torchcomm.barrier(False)

    def test_comm_options_pickle(self):
        """Test that CommOptions can be pickled and unpickled."""
        print("Testing CommOptions pickle support")

        # Create options with custom values
        opts = CommOptions()
        opts.abort_process_on_timeout_or_error = False
        opts.timeout = timedelta(seconds=30)
        opts.hints = {"key1": "value1", "key2": "value2"}

        # Pickle and unpickle
        pickled = pickle.dumps(opts)
        unpickled = pickle.loads(pickled)

        # Verify properties match
        self.assertEqual(
            opts.abort_process_on_timeout_or_error,
            unpickled.abort_process_on_timeout_or_error,
        )
        self.assertEqual(opts.timeout, unpickled.timeout)
        self.assertEqual(opts.hints, unpickled.hints)

    def test_batch_sendrecv_pickle(self):
        """Test that BatchSendRecv can be pickled and unpickled."""
        print("Testing BatchSendRecv pickle support")

        # Create a BatchSendRecv object
        batch = self.torchcomm.batch_op_create()

        # Get the original comm backend for comparison
        orig_comm_backend = batch.get_comm_backend()

        # Pickle and unpickle
        pickled = pickle.dumps(batch)
        unpickled = pickle.loads(pickled)

        # Verify ops list is empty
        self.assertEqual(len(unpickled.ops), 0)

        # Verify the parent comm backend is the same object
        self.assertIs(orig_comm_backend, unpickled.get_comm_backend())

        # Verify the unpickled batch is the same Python object
        self.assertIs(batch, unpickled)

        # Delete the original object and unpickle again
        del batch
        del unpickled
        new_unpickled = pickle.loads(pickled)

        # Verify a new object was created with same properties
        self.assertEqual(len(new_unpickled.ops), 0)

        # Verify the parent comm backend is still the same object
        self.assertIs(orig_comm_backend, new_unpickled.get_comm_backend())

        # Unpickle again and verify it's the same as the first new object
        another_unpickled = pickle.loads(pickled)
        self.assertIs(new_unpickled, another_unpickled)


if __name__ == "__main__":
    unittest.main()
