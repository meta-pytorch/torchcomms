#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

import torch
import torchcomms


def _skip_if_no_ucc(fn):
    """Skip test if UCC backend is not available."""

    def wrapper(self):
        try:
            from torchcomms._comms_ucc import TorchCommUCC  # noqa: F401
        except ImportError:
            self.skipTest("UCC backend not available")
        return fn(self)

    return wrapper


class TestUCCFactory(unittest.TestCase):
    def setUp(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"

    @_skip_if_no_ucc
    def test_ucc_create(self):
        """Test that UCC backend can be created and initialized."""
        comm = torchcomms.new_comm("ucc", torch.device("cpu"), "ucc_test")
        self.assertEqual(comm.get_rank(), 0)
        self.assertEqual(comm.get_size(), 1)
        self.assertEqual(comm.get_backend(), "ucc")

        from torchcomms._comms_ucc import TorchCommUCC

        backend = comm.get_backend_impl()
        self.assertIsInstance(backend, TorchCommUCC)
        comm.finalize()

    @_skip_if_no_ucc
    def test_ucc_barrier(self):
        """Test barrier with single rank."""
        comm = torchcomms.new_comm("ucc", torch.device("cpu"), "ucc_barrier")
        work = comm.barrier(False)
        work.wait()
        comm.finalize()

    @_skip_if_no_ucc
    def test_ucc_broadcast(self):
        """Test broadcast with single rank (root=0)."""
        comm = torchcomms.new_comm("ucc", torch.device("cpu"), "ucc_bcast")
        tensor = torch.tensor([1.0, 2.0, 3.0])
        expected = tensor.clone()
        work = comm.broadcast(tensor, 0, False)
        work.wait()
        torch.testing.assert_close(tensor, expected)
        comm.finalize()

    @_skip_if_no_ucc
    def test_ucc_allreduce_sum(self):
        """Test allreduce with SUM op and single rank."""
        comm = torchcomms.new_comm("ucc", torch.device("cpu"), "ucc_ar_sum")
        tensor = torch.tensor([1.0, 2.0, 3.0])
        expected = tensor.clone()
        work = comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, False)
        work.wait()
        torch.testing.assert_close(tensor, expected)
        comm.finalize()

    @_skip_if_no_ucc
    def test_ucc_allreduce_avg(self):
        """Test allreduce with AVG op and single rank."""
        comm = torchcomms.new_comm("ucc", torch.device("cpu"), "ucc_ar_avg")
        tensor = torch.tensor([4.0, 8.0, 12.0])
        expected = tensor.clone()
        work = comm.all_reduce(tensor, torchcomms.ReduceOp.AVG, False)
        work.wait()
        torch.testing.assert_close(tensor, expected)
        comm.finalize()

    @_skip_if_no_ucc
    def test_ucc_allreduce_dtypes(self):
        """Test allreduce with various data types."""
        comm = torchcomms.new_comm("ucc", torch.device("cpu"), "ucc_ar_dt")
        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            tensor = torch.tensor([1, 2, 3], dtype=dtype)
            expected = tensor.clone()
            work = comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, False)
            work.wait()
            torch.testing.assert_close(tensor, expected)
        comm.finalize()

    @_skip_if_no_ucc
    def test_ucc_allgather(self):
        """Test all_gather with single rank."""
        comm = torchcomms.new_comm("ucc", torch.device("cpu"), "ucc_ag")
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        output_tensors = [torch.zeros(3)]
        work = comm.all_gather(output_tensors, input_tensor, False)
        work.wait()
        torch.testing.assert_close(output_tensors[0], input_tensor)
        comm.finalize()

    @_skip_if_no_ucc
    def test_ucc_allgather_single(self):
        """Test all_gather_single with single rank."""
        comm = torchcomms.new_comm("ucc", torch.device("cpu"), "ucc_ags")
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        output_tensor = torch.zeros(3)
        work = comm.all_gather_single(output_tensor, input_tensor, False)
        work.wait()
        torch.testing.assert_close(output_tensor, input_tensor)
        comm.finalize()

    @_skip_if_no_ucc
    def test_ucc_reduce_scatter_single(self):
        """Test reduce_scatter_single with single rank."""
        comm = torchcomms.new_comm("ucc", torch.device("cpu"), "ucc_rss")
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        output_tensor = torch.zeros(3)
        work = comm.reduce_scatter_single(
            output_tensor, input_tensor, torchcomms.ReduceOp.SUM, False
        )
        work.wait()
        torch.testing.assert_close(output_tensor, input_tensor)
        comm.finalize()

    @_skip_if_no_ucc
    def test_ucc_alltoall_single(self):
        """Test all_to_all_single with single rank."""
        comm = torchcomms.new_comm("ucc", torch.device("cpu"), "ucc_a2a")
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        output_tensor = torch.zeros(3)
        work = comm.all_to_all_single(output_tensor, input_tensor, False)
        work.wait()
        torch.testing.assert_close(output_tensor, input_tensor)
        comm.finalize()

    @_skip_if_no_ucc
    def test_ucc_async_op(self):
        """Test async operation with UCC."""
        comm = torchcomms.new_comm("ucc", torch.device("cpu"), "ucc_async")
        tensor = torch.tensor([1.0, 2.0, 3.0])
        expected = tensor.clone()
        work = comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, True)
        work.wait()
        torch.testing.assert_close(tensor, expected)
        comm.finalize()

    @_skip_if_no_ucc
    def test_ucc_reduce(self):
        """Test reduce with single rank."""
        comm = torchcomms.new_comm("ucc", torch.device("cpu"), "ucc_reduce")
        tensor = torch.tensor([1.0, 2.0, 3.0])
        expected = tensor.clone()
        work = comm.reduce(tensor, 0, torchcomms.ReduceOp.SUM, False)
        work.wait()
        torch.testing.assert_close(tensor, expected)
        comm.finalize()


if __name__ == "__main__":
    unittest.main()
