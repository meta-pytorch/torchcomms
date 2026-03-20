#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

import torch
from torchcomms.tests.integration.py.TorchCommTestHelpers import (
    get_dtype_name,
    TorchCommTestWrapper,
)


class DeviceAllToAllvSingleTest(unittest.TestCase):
    """Test class for device_alltoallv_single operation.

    This tests the device_alltoallv_single API where split sizes are
    CUDA tensors (on GPU), unlike all_to_all_v_single where they are
    host vectors. Displacements are computed internally by the kernel
    as exclusive prefix sums of the counts.

    Test pattern:
    - Each rank creates an input tensor filled with its rank value
    - Split sizes are equal across all ranks (uniform split)
    - After alltoallv, each rank should receive data from all other ranks
    - Rank i's output segment j should contain value j (sent from rank j)
    """

    # CTRAN requires minimum 1024 elements
    chunk_sizes = [1024, 4096]
    dtypes = [torch.float32]

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        """Set up test environment before each test."""
        os.environ["NCCL_CTRAN_ENABLE"] = "1"
        os.environ["NCCL_CTRAN_USE_PIPES"] = "1"
        os.environ["NCCL_CTRAN_PIPES_DISABLE_IB"] = "1"
        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

        # Get the NCCLX backend for NCCLX-specific APIs
        self.ncclx_backend = self.torchcomm.get_backend_impl()

    def tearDown(self):
        """Clean up after each test."""
        self.torchcomm = None
        self.wrapper = None

    def _test_uniform_split(self, chunk_size, dtype):
        """Test with uniform (equal) split sizes across all ranks."""
        print(
            f"Testing device_alltoallv_single uniform split with "
            f"chunk_size={chunk_size} dtype={get_dtype_name(dtype)}"
        )

        total_size = chunk_size * self.num_ranks

        # Create input tensor filled with rank value
        input_tensor = torch.full(
            (total_size,), self.rank, dtype=dtype, device=self.device
        )
        output_tensor = torch.zeros(total_size, dtype=dtype, device=self.device)

        # Create device tensors for split sizes
        # Uniform split: each rank sends/receives chunk_size elements
        send_counts = torch.full(
            (self.num_ranks,), chunk_size, dtype=torch.int64, device=self.device
        )
        recv_counts = torch.full(
            (self.num_ranks,), chunk_size, dtype=torch.int64, device=self.device
        )

        print(f"[Rank {self.rank}] input shape={input_tensor.shape}, value={self.rank}")

        # Call device_alltoallv_single
        self.ncclx_backend.device_alltoallv_single(
            output_tensor,
            input_tensor,
            recv_counts,
            send_counts,
            False,  # async_op
        )

        # Verify output: segment j should contain value j
        for j in range(self.num_ranks):
            start = j * chunk_size
            end = start + chunk_size
            segment = output_tensor[start:end]
            expected_val = j
            if not torch.all(segment == expected_val):
                actual_vals = segment.unique().cpu().tolist()
                self.fail(
                    f"[Rank {self.rank}] Segment {j} expected all {expected_val}, "
                    f"got unique values: {actual_vals}"
                )

        print(f"[Rank {self.rank}] Uniform split test PASSED")

    def _test_variable_split(self, base_chunk_size, dtype):
        """Test with variable (non-uniform) split sizes across ranks."""
        print(
            f"Testing device_alltoallv_single variable split with "
            f"base_chunk_size={base_chunk_size} dtype={get_dtype_name(dtype)}"
        )

        # Each rank sends a different amount to each peer
        # Rank i sends (base_chunk_size + i * 64) elements to rank j
        send_sizes = []
        for _j in range(self.num_ranks):
            send_sizes.append(base_chunk_size + self.rank * 64)

        # Rank i receives (base_chunk_size + j * 64) elements from rank j
        recv_sizes = []
        for j in range(self.num_ranks):
            recv_sizes.append(base_chunk_size + j * 64)

        total_send = sum(send_sizes)
        total_recv = sum(recv_sizes)

        # Create input tensor filled with rank value
        input_tensor = torch.full(
            (total_send,), self.rank, dtype=dtype, device=self.device
        )
        output_tensor = torch.zeros(total_recv, dtype=dtype, device=self.device)

        # Compute expected offsets for verification (prefix sum)
        recv_offsets_list = [0]
        for r in recv_sizes[:-1]:
            recv_offsets_list.append(recv_offsets_list[-1] + r)

        # Create device tensors
        send_counts = torch.tensor(send_sizes, dtype=torch.int64, device=self.device)
        recv_counts = torch.tensor(recv_sizes, dtype=torch.int64, device=self.device)

        print(f"[Rank {self.rank}] send_sizes={send_sizes}, recv_sizes={recv_sizes}")

        # Call device_alltoallv_single
        self.ncclx_backend.device_alltoallv_single(
            output_tensor,
            input_tensor,
            recv_counts,
            send_counts,
            False,  # async_op
        )

        # Verify output: segment from rank j should contain value j
        for j in range(self.num_ranks):
            start = recv_offsets_list[j]
            end = start + recv_sizes[j]
            segment = output_tensor[start:end]
            expected_val = j
            if not torch.all(segment == expected_val):
                actual_vals = segment.unique().cpu().tolist()
                self.fail(
                    f"[Rank {self.rank}] Segment from rank {j} expected all "
                    f"{expected_val}, got unique values: {actual_vals}"
                )

        print(f"[Rank {self.rank}] Variable split test PASSED")

    def test_uniform_split(self):
        """Test device_alltoallv_single with uniform split sizes."""
        for chunk_size in self.chunk_sizes:
            for dtype in self.dtypes:
                with self.subTest(chunk_size=chunk_size, dtype=dtype):
                    self._test_uniform_split(chunk_size, dtype)

    def test_variable_split(self):
        """Test device_alltoallv_single with variable split sizes."""
        for chunk_size in self.chunk_sizes:
            for dtype in self.dtypes:
                with self.subTest(chunk_size=chunk_size, dtype=dtype):
                    self._test_variable_split(chunk_size, dtype)

    def _test_multidim_uniform_split(self, chunk_size, num_cols, dtype):
        """Test with uniform split sizes on 2D tensors.

        Split sizes are in units of rows (dim-0 slices), not element counts.
        Each row has num_cols elements, so the implementation must multiply
        split sizes by elements_per_slice internally.
        """
        print(
            f"Testing device_alltoallv_single multidim uniform split with "
            f"chunk_size={chunk_size} num_cols={num_cols} dtype={get_dtype_name(dtype)}"
        )

        total_rows = chunk_size * self.num_ranks

        # Create 2D input tensor of shape [total_rows, num_cols]
        input_tensor = torch.full(
            (total_rows, num_cols), self.rank, dtype=dtype, device=self.device
        )
        output_tensor = torch.zeros(
            (total_rows, num_cols), dtype=dtype, device=self.device
        )

        # Split sizes are ROW counts (not element counts)
        send_counts = torch.full(
            (self.num_ranks,), chunk_size, dtype=torch.int64, device=self.device
        )
        recv_counts = torch.full(
            (self.num_ranks,), chunk_size, dtype=torch.int64, device=self.device
        )

        print(
            f"[Rank {self.rank}] input shape={input_tensor.shape}, "
            f"value={self.rank}, elements_per_slice={num_cols}"
        )

        self.ncclx_backend.device_alltoallv_single(
            output_tensor,
            input_tensor,
            recv_counts,
            send_counts,
            False,
        )

        # Verify: segment j (chunk_size rows from rank j) should all be j
        for j in range(self.num_ranks):
            start_row = j * chunk_size
            end_row = start_row + chunk_size
            segment = output_tensor[start_row:end_row]
            expected_val = j
            if not torch.all(segment == expected_val):
                actual_vals = segment.unique().cpu().tolist()
                self.fail(
                    f"[Rank {self.rank}] Segment {j} (rows {start_row}:{end_row}) "
                    f"expected all {expected_val}, got unique values: {actual_vals}"
                )

        print(f"[Rank {self.rank}] Multidim uniform split test PASSED")

    def test_multidim_uniform_split(self):
        """Test device_alltoallv_single with 2D tensors and uniform row splits."""
        num_cols_list = [4, 16]
        for chunk_size in self.chunk_sizes:
            for num_cols in num_cols_list:
                for dtype in self.dtypes:
                    with self.subTest(
                        chunk_size=chunk_size, num_cols=num_cols, dtype=dtype
                    ):
                        self._test_multidim_uniform_split(chunk_size, num_cols, dtype)


if __name__ == "__main__":
    unittest.main()
