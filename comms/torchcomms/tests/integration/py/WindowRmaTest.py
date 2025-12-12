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
    """Test class for all_to_all operations in TorchComm."""

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        """Set up test environment before each test."""
        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

    def tearDown(self):
        """Clean up after each test."""
        # Explicitly reset the TorchComm object to ensure proper cleanup
        self.torchcomm = None
        self.wrapper = None

    def _window_put_basic_test(self, count, dtype, async_op, signal, async_signal):
        """Test window allocation and free."""
        print(
            f"Testing window allocation and put with count={count} and dtype={get_dtype_name(dtype)}"
        )

        input_tensor = torch.ones([count], dtype=dtype, device=self.device) * self.rank
        self.torchcomm.barrier(False)

        win = self.torchcomm.window_allocate(
            window_size=self.num_ranks
            * input_tensor.element_size()
            * input_tensor.numel(),
            cpu_buf=False,
        )
        # sync before put to ensure that the window is allocated on all ranks
        self.torchcomm.barrier(False)

        dst_rank = (self.rank + 1) % self.num_ranks
        src_rank = (self.rank - 1 + self.num_ranks) % self.num_ranks
        # Put the tensor to the Window of the next rank using the current stream if async_op is False, otherwise use the internal op_stream
        work = win.put(input_tensor, dst_rank, dst_rank * count, async_op)
        if async_op:
            # register async op to current stream if async_op is True
            work.wait()

        # sync to notify remote rank that the put is complete
        if signal:
            # call signal on current stream to notify remote rank that the put is complete
            signal_work = win.signal(dst_rank, async_signal)
            wait_signal_work = win.wait_signal(src_rank, async_signal)
            if async_signal:
                # register async signal/waitSignal op to current stream if async_op is True since they are launched on internal op_stream/wait_stream
                signal_work.wait()
                wait_signal_work.wait()
            # sync to ensure that the wait_synal is complete
            torch.cuda.current_stream().synchronize()
        else:
            # call barrier on current stream to ensure that the put is complete
            self.torchcomm.barrier(False)

        # Check that the tensor was updated correctly
        output_tensor = win.get_tensor(
            self.rank, list(input_tensor.shape), input_tensor.dtype, self.rank * count
        )
        target_tensor = torch.ones([count], dtype=dtype, device=self.device) * src_rank
        torch.testing.assert_close(
            output_tensor,
            target_tensor,
            rtol=1e-5,
            atol=1e-5,
            msg=f"Rank {self.rank} Expected {target_tensor} but got {output_tensor}",
        )

        self.torchcomm.barrier(False)
        # Free the window
        del win

    def _window_put_cpu_test(self, count, dtype, async_op, signal, async_signal):
        """Test window allocation and free."""
        print(
            f"Testing window allocation and put with count={count} and dtype={get_dtype_name(dtype)}"
        )

        # Call all_to_all without keeping the work object
        tensor_size = torch.Size([count])
        dst_rank = (self.rank + 1) % self.num_ranks
        src_rank = (self.rank - 1 + self.num_ranks) % self.num_ranks

        input_tensor = torch.full(
            tensor_size, self.rank + 100, device=self.device, dtype=torch.bfloat16
        )
        target_tensor = torch.full(
            tensor_size, src_rank + 100, device="cpu", dtype=torch.bfloat16
        )
        chunk_size = input_tensor.numel()
        win = self.torchcomm.window_allocate(
            window_size=self.num_ranks
            * input_tensor.numel()
            * input_tensor.element_size(),
            cpu_buf=True,  # use cpu buffer to create window
        )

        # sync before put to ensure that the window is allocated on all ranks
        self.torchcomm.barrier(False)

        work = win.put(input_tensor, dst_rank, dst_rank * chunk_size, async_op)
        if async_op:
            work.wait()

        if signal:
            signal_work = win.signal(dst_rank, async_signal)
            wait_signal_work = win.wait_signal(src_rank, async_signal)
            if async_signal:
                # register async signal/waitSignal op to current stream if async_op is True since they are launched on internal op_stream/wait_stream
                signal_work.wait()
                wait_signal_work.wait()
            # sync to ensure that the wait_synal is complete
            torch.cuda.current_stream().synchronize()
        else:
            self.torchcomm.barrier(False)
            # [TODO]: wait for barrier to accomplish since we are checking on CPU buffer,
            # we should add some feature in TorchCommWindow to help aovid this if CPU
            # buffer is used.
            torch.cuda.current_stream().synchronize()

        # Check that the tensor was updated correctly
        output_tensor = win.get_tensor(
            self.rank,
            list(input_tensor.shape),
            input_tensor.dtype,
            self.rank * chunk_size,
        )

        # Check that the tensor is on CPU
        assert output_tensor.device.type == "cpu"

        self.torchcomm.barrier(False)

        torch.testing.assert_close(
            output_tensor,
            target_tensor,
            rtol=1e-5,
            atol=1e-5,
            msg=f"Rank {self.rank} Expected {target_tensor} but got {input_tensor}",
        )

        self.torchcomm.barrier(False)
        # Free the window
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
        # Define parameter values
        counts = [4, 1024, 1024 * 1024]
        dtypes = [torch.float, torch.int, torch.int8]
        async_ops = [True, False]
        signals = [True, False]
        cpu_bufs = [True, False]
        async_signals = [True, False]

        # Nested loops for all parameter combinations
        for count, dtype, async_op, signal, cpu_buf, async_signal in itertools.product(
            counts, dtypes, async_ops, signals, cpu_bufs, async_signals
        ):
            # Create a descriptive test name for better test output
            test_name = f"Count_{count}_{get_dtype_name(dtype)}_AsyncOp_{async_op}_Signal_{signal}_CpuBuf_{cpu_buf}"
            print(f"Running tests with parameters: {test_name}")

            # Run all test functions with clear tracing
            print("Running _window_allocate_test")
            if cpu_buf:
                self._window_put_cpu_test(count, dtype, async_op, signal, async_signal)
            else:
                self._window_put_basic_test(
                    count, dtype, async_op, signal, async_signal
                )


if __name__ == "__main__":
    unittest.main()
