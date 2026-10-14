#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import gc
import os
import unittest

import torch
from torchcomms import ReduceOp
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    TorchCommTestWrapper,
)


class AllocatedWindowTest(unittest.TestCase):
    def test_slices_collective_and_storage_lifetime(self):
        if os.environ["TEST_BACKEND"] != "mccl":
            self.skipTest("Allocated windows require the MCCL backend")
        wrapper = TorchCommTestWrapper()
        comm = wrapper.get_torchcomm()
        rank = comm.get_rank()
        size = comm.get_size()
        count = 1024
        window = comm.create_window([count * size + 128], torch.float32)
        storage = window.as_tensor()
        input_tensor = storage[128:]
        self.assertEqual(input_tensor.data_ptr(), storage.data_ptr() + 128 * 4)
        input_tensor.fill_(rank + 1)
        output = torch.empty(count, device=comm.get_device())
        comm.reduce_scatter_single(output, input_tensor, ReduceOp.SUM, False).wait()
        torch.testing.assert_close(
            output, torch.full_like(output, size * (size + 1) / 2)
        )
        torch.cuda.synchronize()
        window.close()
        window.close()
        with self.assertRaises(RuntimeError):
            window.as_tensor()
        del storage, window
        gc.collect()
        input_tensor.fill_(7)
        torch.testing.assert_close(input_tensor, torch.full_like(input_tensor, 7))
        replacement = comm.create_window([count], torch.float32)
        comm.destroy_window(replacement)
        with self.assertRaises(RuntimeError):
            replacement.as_tensor()
        del input_tensor, replacement, comm, wrapper


if __name__ == "__main__":
    unittest.main()
