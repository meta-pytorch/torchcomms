# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe
import os
import unittest

import torch
from torchcomms import ReduceOp
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    TorchCommTestWrapper,
)


class InitModeTest(unittest.TestCase):
    os.environ["NCCL_COMM_STATE_DEBUG_TOPO"] = "nolocal"

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

    def _test_static_regime_init_mode(self, mode):
        """Helper: test static regime with a given initMode hint."""
        wrapper = TorchCommTestWrapper(hints={"initMode": mode})
        comm = wrapper.get_torchcomm()
        rank = comm.get_rank()
        num_ranks = comm.get_size()

        # Each rank contributes (rank + 1); expected sum = num_ranks * (num_ranks + 1) / 2
        tensor = torch.tensor([rank + 1], dtype=torch.float, device=comm.get_device())
        work = comm.all_reduce(tensor, ReduceOp.SUM, async_op=True)
        work.wait()
        expected = num_ranks * (num_ranks + 1) / 2
        self.assertEqual(tensor.item(), expected)

        comm.finalize()

    def test_static_ring_init(self):
        """Test static regime initialization with ring init mode."""
        self._test_static_regime_init_mode("ring")

    def test_static_full_mesh_init(self):
        """Test static regime with explicit full_mesh mode (default behavior)."""
        self._test_static_regime_init_mode("full_mesh")


if __name__ == "__main__":
    unittest.main()
