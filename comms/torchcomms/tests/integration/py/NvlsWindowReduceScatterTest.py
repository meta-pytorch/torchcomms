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


class NvlsWindowReduceScatterTest(unittest.TestCase):
    def test_window_slices_reuse_and_close(self):
        if os.environ["TEST_BACKEND"] != "mccl":
            self.skipTest("Owned windows require the MCCL backend")
        wrapper = TorchCommTestWrapper()
        comm = wrapper.get_torchcomm()
        rank, size = comm.get_rank(), comm.get_size()
        count = 1025
        window = comm.create_window([128 + 2 * size * count], torch.float32)
        storage = window.as_tensor()
        gradients = storage[128:].split(size * count)
        output = torch.empty(count, device=comm.get_device())
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            for repeat in range(3):
                for index, gradient in enumerate(gradients):
                    gradient.fill_(rank + 1 + repeat + index)
                    comm.reduce_scatter_single(
                        output, gradient, ReduceOp.SUM, True
                    ).wait()
                    torch.testing.assert_close(
                        output,
                        torch.full_like(
                            output, size * (size + 1) / 2 + size * (repeat + index)
                        ),
                    )
            torch.cuda.synchronize()
        kernels = [
            event.name
            for event in profile.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        nvls_kernels = [
            name for name in kernels if "reduceScatterDirectNvlsKernel" in name
        ]
        self.assertEqual(len(nvls_kernels), 6, kernels)
        self.assertFalse(
            any("reduceScatterDirectKernel<" in name for name in kernels), kernels
        )
        print(f"rank={rank}: six NVLS CUDA kernels executed; output verified")
        torch.cuda.synchronize()
        window.close()
        with self.assertRaises(RuntimeError):
            window.as_tensor()
        del window, storage
        gc.collect()
        gradients[0].fill_(7)
        torch.testing.assert_close(gradients[0], torch.full_like(gradients[0], 7))
        replacement = comm.create_window([count], torch.float32)
        comm.destroy_window(replacement)


if __name__ == "__main__":
    unittest.main()
