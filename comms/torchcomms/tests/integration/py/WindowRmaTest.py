#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import os
import unittest

import torch
import torchcomms
from torchcomms import TorchCommWinAccessType
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

        # Get allocator using global function - obtained once and reused
        self.allocator = torchcomms.get_mem_allocator(self.torchcomm.get_backend())

    def tearDown(self):
        self.allocator = None
        self.torchcomm = None
        self.wrapper = None

    def _window_put_test(self, count, dtype, async_op, async_signal):
        """Test window put operation."""
        put_stream = torch.cuda.Stream()
        wait_stream = torch.cuda.Stream()

        # Create input tensor (regular allocation)
        input_tensor = (
            torch.ones(
                [count],
                dtype=dtype,
                device=self.device,
            )
            * self.rank
        )
        # Use the global allocator obtained in setUp
        pool = torch.cuda.MemPool(self.allocator)
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

        # Perform multiple put operations to test repeated usage
        num_iterations = 10
        for iteration in range(num_iterations):
            # Put the tensor to the Window of the next rank
            with torch.cuda.stream(put_stream):
                work = win.put(input_tensor, dst_rank, dst_rank * count, async_op)
                if async_op:
                    work.wait()
            # signal the next rank to proceed
            with torch.cuda.stream(put_stream):
                signal_work = win.signal(dst_rank, async_signal)
                if async_signal:
                    signal_work.wait()

            # wait signal from the previous rank
            with torch.cuda.stream(wait_stream):
                wait_signal_work = win.wait_signal(src_rank, async_signal)
                if async_signal:
                    wait_signal_work.wait()

                local_tensor = win.map_remote_tensor(self.rank)

            # wait for the data from the previous rank to be ready
            wait_stream.synchronize()
            output_tensor = local_tensor[self.rank * count : (self.rank + 1) * count]

            target_tensor = (
                torch.ones(
                    [count],
                    dtype=dtype,
                    device=self.device,
                )
                * src_rank
            )

            torch.testing.assert_close(
                output_tensor,
                target_tensor,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Rank {self.rank} Iteration {iteration} Expected {target_tensor} but got {output_tensor}",
            )

        # Cleanup
        wait_stream.synchronize()
        put_stream.synchronize()
        win.tensor_deregister()
        del win
        del pool
        torch.cuda.synchronize()

    def _window_attributes_test(self, count, dtype):
        """Test window attributes for unified vs separate access types."""
        # Get global allocator for the backend and create memory pool
        allocator = torchcomms.get_mem_allocator(self.torchcomm.get_backend())
        pool = torch.cuda.MemPool(allocator)
        with torch.cuda.use_mem_pool(pool):
            win_buf = torch.ones(
                [count * self.num_ranks], dtype=dtype, device=self.device
            )

        win = self.torchcomm.new_window()
        win.tensor_register(win_buf)
        self.torchcomm.barrier(False)

        # Test window attributes
        if self.num_ranks <= 8:
            next_rank = (self.rank + 1) % self.num_ranks
            win_attr = win.get_attr(next_rank)
            assert (
                win_attr.access_type == TorchCommWinAccessType.WIN_ACCESS_TYPE_UNIFIED
            )
        else:
            next_rank = (self.rank + 8) % self.num_ranks
            win_attr = win.get_attr(next_rank)
            assert (
                win_attr.access_type == TorchCommWinAccessType.WIN_ACCESS_TYPE_SEPARATE
            )

        # Cleanup
        win.tensor_deregister()
        del win
        del pool

    def _map_remote_tensor_device_agnostic_test(self, count, dtype):
        """Helper function to test map_remote_tensor with device-agnostic access."""
        print(
            f"Testing map_remote_tensor_device_agnostic with count={count}, dtype={get_dtype_name(dtype)}"
        )

        # Create memory pool and allocate window buffer within it
        pool = torch.cuda.MemPool(self.torchcomm.mem_allocator)
        with torch.cuda.use_mem_pool(pool):
            win_buf = torch.arange(
                count * self.num_ranks, dtype=dtype, device=self.device
            )

        win = self.torchcomm.new_window()
        win.tensor_register(win_buf)
        self.torchcomm.barrier(False)

        # Test local access
        local_tensor = win.map_remote_tensor(self.rank)
        self.assertEqual(local_tensor.dtype, win_buf.dtype)
        self.assertEqual(local_tensor.shape, win_buf.shape)
        torch.testing.assert_close(local_tensor, win_buf, rtol=0, atol=0)

        # Test remote access (only for unified memory)
        remote_rank = (self.rank + 1) % self.num_ranks
        win_attr = win.get_attr(remote_rank)
        if win_attr.access_type == TorchCommWinAccessType.WIN_ACCESS_TYPE_UNIFIED:
            remote_tensor = win.map_remote_tensor(remote_rank)
            self.assertEqual(remote_tensor.dtype, dtype)
            self.assertEqual(remote_tensor.shape, win_buf.shape)

            expected_data = torch.arange(
                count * self.num_ranks, dtype=dtype, device=self.device
            )
            torch.testing.assert_close(
                remote_tensor, expected_data, rtol=1e-5, atol=1e-5
            )

        # Cleanup
        win.tensor_deregister()
        del win
        del pool

    @unittest.skipIf(
        os.getenv("RUN_RMA_TEST", "").lower() not in ("1", "true"),
        "RMA tests require NCCLX backend with CTran enabled (RUN_RMA_TEST=true)",
    )
    def test_all_tests(self):
        """Run all tests with all parameter combinations."""
        counts = [4, 1024, 1024 * 1024]
        dtypes = [torch.float, torch.int, torch.int8]
        async_ops = [True, False]
        async_signals = [True, False]

        # Nested loops for all parameter combinations
        for (
            count,
            dtype,
            async_op,
            async_signal,
        ) in itertools.product(
            counts,
            dtypes,
            async_ops,
            async_signals,
        ):
            # Create a descriptive test name for better test output
            test_name = f"Count_{count}_{get_dtype_name(dtype)}_AsyncOp_{async_op}_AsyncSignal_{async_signal}"
            print(f"Running _window_put_test with parameters: {test_name}")

            self._window_put_test(count, dtype, async_op, async_signal)

        # Test window attributes with specific parameters
        count = 1024
        dtypes_to_test = [torch.float, torch.int, torch.int8]
        for dtype in dtypes_to_test:
            test_name = f"Count_{count}_{get_dtype_name(dtype)}"
            print(f"Running _window_attributes_test with parameters: {test_name}")
            self._window_attributes_test(count, dtype)

        # Test map_remote_tensor_device_agnostic with specific dtypes
        dtypes_to_test = [torch.float32, torch.int32, torch.bfloat16]
        count = 1024
        for dtype in dtypes_to_test:
            print("Running _map_remote_tensor_device_agnostic_test")
            self._map_remote_tensor_device_agnostic_test(count, dtype)


if __name__ == "__main__":
    unittest.main()
