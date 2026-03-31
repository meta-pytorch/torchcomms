#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for allgatherp (CE-based persistent allgather) via the torchcomms NCCLX backend.

Validates that allgatherp init/exec/free work correctly through the
TorchCommNCCLX Python API, and that the output buffer can be reused for
SM compute after the CE-based collective completes.
"""

import os
import unittest

import torch
import torchcomms
from torchcomms.tests.integration.py.TorchCommTestHelpers import TorchCommTestWrapper


class AllGatherPTest(unittest.TestCase):
    """Tests for allgatherp eager execution via the torchcomms NCCLX backend."""

    NUM_REPLAYS = 3
    ELEM_COUNT = 1024

    def setUp(self) -> None:
        self.wrapper = TorchCommTestWrapper()
        self.comm = self.wrapper.get_torchcomm()
        self.rank = self.comm.get_rank()
        self.size = self.comm.get_size()
        self.device = self.comm.get_device()

    def tearDown(self) -> None:
        del self.comm
        del self.wrapper

    @unittest.skipIf(
        os.getenv("NCCL_CTRAN_ENABLE") != "true",
        "Requires ctran transport (NCCL_CTRAN_ENABLE=true)",
    )
    def test_allgatherp(self) -> None:
        """Eager allgatherp followed by compute on the output buffer."""
        count = self.ELEM_COUNT

        input_tensor = (
            torch.ones(count, dtype=torch.float32, device=self.device) * self.rank
        )

        allocator = torchcomms.get_mem_allocator(self.comm.get_backend())
        pool = torch.cuda.MemPool(allocator)
        with torch.cuda.use_mem_pool(pool):
            output_tensor = torch.zeros(
                count * self.size, dtype=torch.float32, device=self.device
            )

        handle = self.comm.all_gather_p_init(output_tensor)
        self.comm.barrier(False)

        for _ in range(self.NUM_REPLAYS):
            work = self.comm.all_gather_p_exec(handle, input_tensor, async_op=True)
            work.wait()
            torch.cuda.synchronize()

            # Verify allgather correctness
            for r in range(self.size):
                chunk = output_tensor[r * count : (r + 1) * count]
                expected = (
                    torch.ones(count, dtype=torch.float32, device=self.device) * r
                )
                torch.testing.assert_close(
                    chunk,
                    expected,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"Rank {self.rank}: chunk for rank {r} mismatch",
                )

            # Reuse the output buffer for compute
            compute_result = output_tensor.sum()
            expected_sum = sum(r * count for r in range(self.size))
            self.assertAlmostEqual(
                compute_result.item(),
                float(expected_sum),
                places=0,
                msg=f"Rank {self.rank}: compute on allgatherp output mismatch",
            )

        self.comm.all_gather_p_free(handle)


if __name__ == "__main__":
    unittest.main()
