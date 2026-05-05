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

    # TODO: These tests verified the CCA hook by constructing
    # RdmaMemory(tensor, cache_reg=True), which threw if the tensor was not
    # pre-registered. The torchcomms._transport module was rewritten in
    # D101014446 from the RDMA-only API to uniflow's MultiTransport, which
    # has no direct equivalent for "check if a tensor is pre-registered"
    # (register_segment always registers). Re-enable once a uniflow-based
    # check is available, or replace with a different verification mechanism.
    @unittest.skip("disabled pending uniflow API equivalent (D101014446)")
    def test_default_allocator_registers_tensor(self) -> None:
        """Verify that a tensor allocated with the default CUDACachingAllocator
        is automatically registered via the CCA hook."""

    @unittest.skip("disabled pending uniflow API equivalent (D101014446)")
    def test_mem_pool_registers_tensor(self) -> None:
        """Verify that a tensor allocated from cuda.MemPool is automatically
        registered with globalRegisterWithPtr via the CCA hook."""


if __name__ == "__main__":
    unittest.main()
