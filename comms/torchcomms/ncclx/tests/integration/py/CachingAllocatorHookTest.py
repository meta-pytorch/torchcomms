#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

import torch
from torchcomms._comms_ncclx import init_caching_allocator_hook


class CachingAllocatorHookTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device_ = torch.device("cuda")
        self.tensor_size_ = 1024 * 1024

        # Initialize CCA hook WITHOUT creating a communicator
        init_caching_allocator_hook()

    @unittest.skipIf(
        torch.cuda.get_device_capability() < (9, 0),
        "Skipping CCA hook tests on GPU capability < 9.0",
    )
    def test_default_allocator_registers_tensor(self) -> None:
        """Verify that a tensor allocated with the default CUDACachingAllocator
        is automatically registered via the CCA hook, by constructing
        RdmaMemory with cache_reg=True (which throws if not registered)."""
        tensor = torch.ones(self.tensor_size_, device=self.device_)

        from torchcomms._transport import RdmaMemory

        rdma_mem = RdmaMemory(tensor, cache_reg=True)
        self.assertIsNotNone(rdma_mem)

    @unittest.skipIf(
        torch.cuda.get_device_capability() < (9, 0),
        "Skipping CCA hook tests on GPU capability < 9.0",
    )
    def test_mem_pool_registers_tensor(self) -> None:
        """Verify that a tensor allocated from cuda.MemPool is automatically
        registered with globalRegisterWithPtr via the CCA hook, by constructing
        RdmaMemory with cache_reg=True (which throws if not registered)."""
        pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool):
            tensor = torch.ones(self.tensor_size_, device=self.device_)

        from torchcomms._transport import RdmaMemory

        rdma_mem = RdmaMemory(tensor, cache_reg=True)
        self.assertIsNotNone(rdma_mem)


if __name__ == "__main__":
    unittest.main()
