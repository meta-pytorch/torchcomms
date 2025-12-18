#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import os
import unittest

import psutil

import torch
from torchcomms.tests.integration.py.TorchCommTestHelpers import (
    get_dtype_name,
    TorchCommTestWrapper,
)


class WindowRmaTest(unittest.TestCase):
    """Test class for Window RMA operations in TorchComm."""

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

    def tearDown(self):
        self.torchcomm = None
        self.wrapper = None

    def _window_put_test(self, count, dtype, async_op, signal, async_signal):
        """Test window put operation."""
        input_tensor = (
            torch.ones(
                [count],
                dtype=dtype,
                device=self.device,
            )
            * self.rank
        )
        pool = torch.cuda.MemPool(self.torchcomm.mem_allocator)
        with torch.cuda.use_mem_pool(pool):
            win_buf = torch.ones(
                [count * self.num_ranks], dtype=dtype, device=self.device
            )

        self.torchcomm.barrier(False)

        win = self.torchcomm.new_window()
        win.tensor_register(win_buf)
        self.torchcomm.barrier(False)

        dst_rank = (self.rank + 1) % self.num_ranks
        src_rank = (self.rank - 1 + self.num_ranks) % self.num_ranks
        # Put the tensor to the Window of the next rank using the current stream if async_op is False, otherwise use the internal op_stream
        work = win.put(input_tensor, dst_rank, dst_rank * count, async_op)
        if async_op:
            work.wait()

        if signal:
            signal_work = win.signal(dst_rank, async_signal)
            wait_signal_work = win.wait_signal(src_rank, async_signal)
            if async_signal:
                signal_work.wait()
                wait_signal_work.wait()
            torch.cuda.current_stream().synchronize()
        else:
            self.torchcomm.barrier(False)
            torch.cuda.current_stream().synchronize()

        output_tensor = win.get_tensor(
            self.rank, list(input_tensor.shape), input_tensor.dtype, self.rank * count
        )

        target_tensor = (
            torch.ones(
                [count],
                dtype=dtype,
                device=self.device,
            )
            * src_rank
        )

        self.torchcomm.barrier(False)

        torch.testing.assert_close(
            output_tensor,
            target_tensor,
            rtol=1e-5,
            atol=1e-5,
            msg=f"Rank {self.rank} Expected {target_tensor} but got {output_tensor}",
        )

        torch.cuda.synchronize()
        self.torchcomm.barrier(False)
        win.tensor_deregister()
        del win

    @unittest.skipIf(
        os.getenv("NCCL_CTRAN_ENABLE", "").lower() not in ("1", "true"),
        "Skipping NCCLX Ctran Window tests",
    )
    @unittest.skipIf(
        os.getenv("TEST_BACKEND") != "ncclx", "Skipping NCCLX-only window tests"
    )
    @unittest.skipIf("beth0" not in psutil.net_if_addrs(), "RDMA nic required")
    def test_all_tests(self):
        """Run all tests with all parameter combinations."""
        counts = [4, 1024, 1024 * 1024]
        dtypes = [torch.float, torch.int, torch.int8]
        async_ops = [True, False]
        signals = [True, False]
        async_signals = [True, False]

        # Nested loops for all parameter combinations
        for (
            count,
            dtype,
            async_op,
            signal,
            async_signal,
        ) in itertools.product(
            counts,
            dtypes,
            async_ops,
            signals,
            async_signals,
        ):
            # Create a descriptive test name for better test output
            test_name = f"Count_{count}_{get_dtype_name(dtype)}_AsyncOp_{async_op}_Signal_{signal}_AsyncSignal_{async_signal}"
            print(f"Running tests with parameters: {test_name}")

            self._window_put_test(count, dtype, async_op, signal, async_signal)


if __name__ == "__main__":
    unittest.main()
