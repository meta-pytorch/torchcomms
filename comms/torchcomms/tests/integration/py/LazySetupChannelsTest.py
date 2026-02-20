#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

import torch
from torchcomms import ReduceOp
from torchcomms.tests.integration.py.TorchCommTestHelpers import TorchCommTestWrapper


@unittest.skipIf(
    os.getenv("TEST_BACKEND") != "ncclx",
    "Skipping LazySetupChannels tests (ncclx only)",
)
class LazySetupChannelsTest(unittest.TestCase):
    """Test lazy setup channels hint with split comms and collectives."""

    def setUp(self):
        self.wrapper_lazy = TorchCommTestWrapper(
            hints={"lazySetupChannels": "1"},
        )
        self.torchcomm_lazy = self.wrapper_lazy.get_torchcomm()
        self.rank = self.torchcomm_lazy.get_rank()
        self.num_ranks = self.torchcomm_lazy.get_size()
        self.device = self.torchcomm_lazy.get_device()

    def tearDown(self):
        self.torchcomm_lazy = None
        self.wrapper_lazy = None

    def test_lazy_setup_channels_split_allreduce(self):
        """Create a comm with lazy connect, split into smaller subgroups,
        and do a collective on the split comm."""
        # Split into two halves
        split_size = self.num_ranks // 2
        if split_size == 0:
            split_size = 1

        rank_in_group = self.rank < split_size
        ranks = list(range(split_size)) if rank_in_group else []

        split_comm = self.torchcomm_lazy.split(ranks, name="split_from_lazy")

        if rank_in_group:
            self.assertIsNotNone(split_comm)
            self.assertEqual(split_comm.get_rank(), self.rank)
            self.assertEqual(split_comm.get_size(), split_size)

            # Run all_reduce on the split comm
            input_tensor = torch.ones(
                1024, dtype=torch.float, device=self.device
            ) * float(self.rank + 1)
            split_comm.all_reduce(input_tensor, ReduceOp.SUM, False)

            expected_sum = split_size * (split_size + 1) / 2
            expected_tensor = torch.full_like(input_tensor.cpu(), expected_sum)
            torch.testing.assert_close(input_tensor.cpu(), expected_tensor)

            split_comm.finalize()
        else:
            self.assertIsNone(split_comm)


if __name__ == "__main__":
    unittest.main(failfast=True)
